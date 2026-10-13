"""Append compact-document cosine to immutable V8 training/Tune candidate rows.

No retrieval, new item IDs, label-based sampling or prior fitting occurs here.
Only query vectors are encoded; document vectors must come from a completed,
matching compact HNSW index. The output is a standalone experimental schema9,
explicitly unavailable to the production Reranker until a later integration.
Run only after the primary V8 pipeline has finished, with a new output path.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from avito_candidates.core import SEARCH_FIELDS, normalize, query_key
from avito_candidates.dense import DenseRetriever, _fingerprint, query_text, encoding_input
from avito_candidates.reranker import FEATURE_SCHEMAS
from cache_research_reranker import (validate_research_artifact, check_training_prior_file,
                                    research_prior_policy)
from evaluate_research_reranker import check_research_cache
from produce_document_hypotheses import (load_catalog_documents, completed_index_identity,
                                         check_encoder_base, require_offline_environment)
import evaluate_fresh_validation as EVAL

ADDED_FEATURES = ("compact_dense_score", "compact_dense_known")


def checked_key(key, identity=None):
    if (not isinstance(key, list) or len(key) != len(SEARCH_FIELDS)
            or any(not isinstance(value, str) or value != normalize(value) for value in key)):
        raise ValueError("Secondary dense input requires the exact normalized five-field query context")
    raw = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if identity is not None and digest != identity:
        raise ValueError("Secondary dense query context differs from its canonical group identity")
    return dict(zip(SEARCH_FIELDS, key))


def recover_training_representatives(source, needed_keys, fit_texts, corpus_ids, *, batch_size=65536):
    """Reproduce read_fit_labels' LAST eligible raw representative per key.

The source/batch size must be the same as primary preparation. Raw case and
spacing are retained until the immutable query_text renderer handles them.
No new training labels are sampled or inserted into the candidate rows.
"""
    representatives = {}
    for batch in pq.ParquetFile(source).iter_batches(batch_size=batch_size,
                                                    columns=[*SEARCH_FIELDS, "item_id"]):
        for row in batch.to_pandas().fillna("").astype(str).to_dict("records"):
            if normalize(row["search_query"]) not in fit_texts or row["item_id"] not in corpus_ids:
                continue
            key = list(query_key(row))
            raw = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
            if raw in needed_keys:
                canonical = checked_key(key)
                if query_text(row) != query_text(canonical):
                    raise ValueError("Raw representative and stored key render different encoder input")
                representatives[raw] = {field: row[field] for field in SEARCH_FIELDS}
    if set(representatives) != set(needed_keys):
        raise ValueError("Exact primary raw training representatives could not be recovered")
    return representatives


def current_index(dense, items, directory):
    """Load the completed primary index without loading a second transformer."""
    import hnswlib
    fingerprint = _fingerprint(items, dense.model_name, dense.max_seq_length,
        dense.params_words, dense.description_words, dense.model_signature, document_format="current")
    root = Path(directory).resolve()
    binary, sidecar = root / (fingerprint + ".bin"), root / (fingerprint + ".json")
    expected = dict(dense._metadata())
    expected.pop("document_format", None)
    expected.pop("document_format_version", None)
    if (not binary.is_file() or not sidecar.is_file()
            or json.loads(sidecar.read_text(encoding="utf-8")) != expected):
        raise ValueError("A completed matching current-document index is required for the shared-vector control")
    index = hnswlib.Index(space="cosine", dim=dense.dimension)
    index.load_index(str(binary), max_elements=len(items))
    return index, {"index": {"path": str(binary), "sha256": EVAL.sha256_file(binary)},
        "metadata": {"path": str(sidecar), "sha256": EVAL.sha256_file(sidecar)}, "fingerprint": fingerprint}


def score_both_document_indexes(dense, primary_index, queries, pools):
    """One query encoding, one normalized vector shared by both document views."""
    if len(queries) != len(pools):
        raise ValueError("Shared-vector queries and candidate pools must match")
    if not queries:
        return []
    texts, prompt_kwargs = encoding_input([query_text(query) for query in queries],
                                          dense.model_name, "query", dense.model)
    vectors = np.asarray(dense.model.encode(texts, batch_size=dense.batch_size,
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        **prompt_kwargs), dtype=np.float32)
    if vectors.shape != (len(queries), dense.dimension) or not np.isfinite(vectors).all():
        raise ValueError("Invalid shared query vectors")
    if getattr(dense, "_id_to_label", None) is None:
        dense._id_to_label = {item: index for index, item in enumerate(dense.ids)}
    labels = dense._id_to_label
    output = []
    for vector, pool in zip(vectors, pools):
        if len(pool) != len(set(pool)) or len(pool) > 998 or any(item not in labels for item in pool):
            raise ValueError("Shared-vector rescoring requires exactly the existing valid candidate IDs")
        if not pool:
            output.append(({}, {}))
            continue
        norm = float(np.linalg.norm(vector))
        if not norm or not math.isfinite(norm):
            raise ValueError("Shared query vector has invalid norm")
        normalized = vector / norm
        positions = [labels[item] for item in pool]
        maps = []
        for index in (primary_index, dense.index):
            stored = np.asarray(index.get_items(positions), dtype=np.float32)
            if stored.shape != (len(pool), dense.dimension) or not np.isfinite(stored).all():
                raise ValueError("Invalid stored document vectors")
            scores = stored @ normalized
            if not np.isfinite(scores).all() or np.max(np.abs(scores), initial=0) > 1.0001:
                raise ValueError("Invalid shared-vector cosine scores")
            maps.append({item: float(score) for item, score in zip(pool, scores)})
        output.append(tuple(maps))
    return output


def iter_complete_groups(path, metadata, mode, batch_size=8192):
    """Carry a group over row-batch boundaries; never reorder or resample rows."""
    pending, current, seen = [], None, set()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        ids = batch.column(batch.schema.get_field_index("group_id")).to_pylist()
        start = 0
        while start < len(ids):
            identity = ids[start]
            stop = start + 1
            while stop < len(ids) and ids[stop] == identity:
                stop += 1
            if current is not None and current != identity:
                yield _group_payload(current, pa.concat_tables(pending), metadata, mode)
                seen.add(current)
                pending = []
            if identity != current and identity in seen:
                raise ValueError("Secondary dense source repeats a noncontiguous query group")
            current = identity
            pending.append(pa.Table.from_batches([batch.slice(start, stop - start)]))
            start = stop
    if pending:
        yield _group_payload(current, pa.concat_tables(pending), metadata, mode)


def _group_payload(identity, table, metadata, mode):
    items = table["item_id"].to_pylist()
    if len(items) != len(set(items)) or len(items) > 998:
        raise ValueError("Secondary dense source has duplicate IDs or exceeds the existing pool budget")
    if mode == "train":
        keys = set(table["query_key_json"].to_pylist())
        texts = set(table["query_text"].to_pylist())
        if len(keys) != 1 or len(texts) != 1:
            raise ValueError("Training source mixes query contexts inside one group")
        raw = next(iter(keys))
        key = json.loads(raw)
        query = checked_key(key, identity)
        if raw != json.dumps(key, ensure_ascii=False, separators=(",", ":")) or texts != {key[0]}:
            raise ValueError("Training source context normalization differs")
    else:
        groups = metadata["groups"]
        if isinstance(identity, bool) or not isinstance(identity, int) or not 0 <= identity < len(groups):
            raise ValueError("Tune source group ID is outside its sealed metadata")
        group = groups[identity]
        query = checked_key(group["query_key"], group["group_identity"])
        if group["group_id"] != identity or group["pool_size"] != len(items):
            raise ValueError("Tune rows differ from the already completed candidate pool")
    return query, items, table


def append_cosine(table, items, scores, primary_control=None, tolerance=.005):
    if set(scores) != set(items):
        raise ValueError("Compact cosine must cover exactly the supplied existing candidate IDs")
    values = [float(scores[item]) for item in items]
    if any(not math.isfinite(value) or abs(value) > 1.0001 for value in values):
        raise ValueError("Compact cosine must be finite and within normalized similarity bounds")
    if any(name in table.column_names for name in ADDED_FEATURES):
        raise ValueError("Refusing to append the same secondary features twice")
    if primary_control is not None:
        if set(primary_control) != set(items) or "full_dense_score" not in table.column_names:
            raise ValueError("Primary shared-vector control must cover the existing V8 rows")
        old = np.asarray(table["full_dense_score"].to_numpy(), dtype=np.float32)
        if "full_dense_known" not in table.column_names or not np.all(table["full_dense_known"].to_numpy() == 1):
            raise ValueError("V8 primary cosine must be measured for every existing candidate")
        measured = np.asarray([primary_control[item] for item in items], dtype=np.float32)
        if (not np.isfinite(measured).all() or not np.isfinite(old).all()
                or np.max(np.abs(old - measured), initial=0) > tolerance):
            raise ValueError("Recovered query encoding disagrees with the primary cached cosine")
    augmented = table.append_column(ADDED_FEATURES[0], pa.array(values, type=pa.float32()))
    augmented = augmented.append_column(ADDED_FEATURES[1], pa.array(np.ones(len(items)), type=pa.float32()))
    for name in table.column_names:
        if not table[name].equals(augmented[name]):
            raise ValueError("Secondary dense derivation changed an original feature/label/weight/ID column")
    return augmented


def checked_source(args, artifact, manifest, protocol_hash):
    metadata_path = (args.source.with_suffix(args.source.suffix + ".json") if args.mode == "train"
                     else args.source.with_suffix(".json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = list(FEATURE_SCHEMAS[8])
    if metadata.get("feature_names") != features or artifact["metadata"].get("feature_version") != 8:
        raise ValueError("Secondary experiment requires a completed exact V8 feature schema")
    source_hash = EVAL.sha256_file(args.source)
    if args.mode == "train":
        EVAL.check_training_data(args.source, artifact, args.protocol_dir, manifest)
        check_training_prior_file(args.source, artifact)
        if metadata.get("feature_version") != 8:
            raise ValueError("Training source is not the V8 reference cache")
    else:
        if args.reference_train_data is None:
            raise ValueError("Tune derivation requires the V8 model's proven fit cache")
        train_hash = EVAL.check_training_data(args.reference_train_data, artifact, args.protocol_dir, manifest)
        check_training_prior_file(args.reference_train_data, artifact)
        EVAL.ensure_compatible(artifact["metadata"], metadata, "secondary-dense V8 source")
        EVAL.ensure_prior_policy(artifact, metadata, "secondary-dense V8 source")
        check_research_cache(metadata, artifact, train_hash)
        EVAL.check_phase_cache_inputs(args.source, metadata, "tune", protocol_hash,
                                     EVAL.sha256_file(args.reference_model))
    parquet = pq.ParquetFile(args.source)
    if any(name not in parquet.schema_arrow.names or str(parquet.schema_arrow.field(name).type) != "float"
           for name in features):
        raise ValueError("V8 source feature columns must remain float32")
    return metadata_path, metadata, source_hash, EVAL.sha256_file(metadata_path)


def derive(args):
    require_offline_environment()
    output_metadata = (args.output.with_suffix(args.output.suffix + ".json") if args.mode == "train"
                       else args.output.with_suffix(".json"))
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    marker = args.output.with_suffix(args.output.suffix + ".INCOMPLETE")
    priors_output = args.output.with_suffix(args.output.suffix + ".priors.joblib")
    if any(path.exists() for path in (args.output, output_metadata, temporary, marker, priors_output)):
        raise FileExistsError("Secondary derivation requires new unused output/sidecar paths")
    manifest, protocol_hash = EVAL.checked_protocol(args.protocol_dir)
    import joblib
    reference_hash = EVAL.sha256_file(args.reference_model)
    artifact = joblib.load(args.reference_model)
    fit_path = EVAL.protocol_file(args.protocol_dir, manifest, "fit_texts")
    fit_texts = set(json.loads(fit_path.read_text(encoding="utf-8"))["texts"])
    validate_research_artifact(artifact, fit_texts, manifest)
    EVAL.check_encoder_identity(artifact, manifest)
    source_metadata_path, metadata, source_hash, metadata_hash = checked_source(args, artifact, manifest, protocol_hash)
    train_data = args.source if args.mode == "train" else args.reference_train_data
    provenance = artifact["provenance"]
    training_inputs = {name: {"path": str(path.resolve()), "sha256": provenance[field]}
        for name, path, field in (
            ("data", train_data, "training_data_sha256"),
            ("metadata", train_data.with_suffix(train_data.suffix + ".json"), "training_metadata_sha256"),
            ("priors", train_data.with_suffix(train_data.suffix + ".priors.joblib"), "training_priors_sha256"))}
    root = Path(__file__).resolve().parents[1]
    source_closure = {str(path.resolve()): EVAL.sha256_file(path) for path in (
        Path(__file__), root / "src/avito_candidates/dense.py", root / "src/avito_candidates/core.py",
        root / "src/avito_candidates/reranker.py", root / "src/avito_candidates/embedding_formats.py",
        root / "scripts/produce_document_hypotheses.py")}
    policy = research_prior_policy(artifact)
    config = dict(artifact["metadata"]["retrieval_config"])
    if config.get("embedding_document_format", "current") != "current":
        raise ValueError("This secondary ablation starts from the verified current-document V8 baseline")
    base = check_encoder_base(config["embedding_model"], manifest, args.encoder_base_reference)
    del artifact
    gc.collect()
    catalog = EVAL.protocol_file(args.protocol_dir, manifest, "benchmark_items")
    items, corpus_hash = load_catalog_documents(catalog)
    if corpus_hash != metadata["corpus_fingerprint"]:
        raise ValueError("Compact index inventory differs from the V8 source features")
    dense = DenseRetriever(items, model_name=config["embedding_model"], cache_dir=args.compact_cache,
        batch_size=args.query_batch_size, device=args.device,
        max_seq_length=128, params_words=config["embedding_params_words"],
        description_words=config["embedding_description_words"], document_format="compact",
        ef_search=config["ef_search"], require_cached_index=True)
    index_identity = completed_index_identity(dense, items, args.compact_cache)
    primary_index, primary_index_identity = current_index(dense, items, args.current_cache)
    corpus_ids = set(dense.ids)
    raw_representatives = None
    representative_source = EVAL.protocol_file(args.protocol_dir, manifest, "train")
    if args.mode == "train":
        needed = set()
        for batch in pq.ParquetFile(args.source).iter_batches(batch_size=args.row_batch_size, columns=["query_key_json"]):
            needed.update(batch.column(0).to_pylist())
        raw_representatives = recover_training_representatives(representative_source, needed, fit_texts,
            corpus_ids, batch_size=args.representative_batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("x", encoding="utf-8") as stream:
        stream.write("Secondary feature derivation is incomplete; do not fit or evaluate this output.\n")
    schema = pq.ParquetFile(args.source).schema_arrow
    schema = schema.append(pa.field(ADDED_FEATURES[0], pa.float32())).append(pa.field(ADDED_FEATURES[1], pa.float32()))
    rows, groups, pending = 0, 0, []
    start = time.monotonic()

    def write_batch(writer, pending):
        queries, pools, tables = zip(*pending)
        if any(not set(pool) <= corpus_ids for pool in pools):
            raise ValueError("Secondary source contains a candidate outside the completed compact inventory")
        results = score_both_document_indexes(dense, primary_index, queries, pools)
        if len(results) != len(pending):
            raise ValueError("Compact rescoring omitted a declared query context")
        for table, pool, (primary, compact) in zip(tables, pools, results):
            writer.write_table(append_cosine(table, pool, compact, primary_control=primary))

    with temporary.open("xb") as stream, pq.ParquetWriter(stream, schema, compression="zstd") as writer:
        for payload in iter_complete_groups(args.source, metadata, args.mode, args.row_batch_size):
            query, pool, table = payload
            if args.mode == "train" and normalize(query["search_query"]) not in fit_texts:
                raise ValueError("Secondary training context is outside the proven fit split")
            if args.mode == "tune" and normalize(query["search_query"]) in fit_texts:
                raise ValueError("Secondary Tune context overlaps the encoder's training texts")
            if args.mode == "train":
                raw = next(iter(set(table["query_key_json"].to_pylist())))
                representative = raw_representatives[raw]
                if query_text(representative) != query_text(query):
                    raise ValueError("Raw primary query and reconstructed key no longer render identically")
                payload = (representative, pool, table)
            pending.append(payload)
            rows += table.num_rows
            groups += 1
            if len(pending) == args.query_batch_size:
                write_batch(writer, pending)
                pending = []
                if groups % (args.query_batch_size * 10) == 0:
                    print(json.dumps({"mode": args.mode, "groups": groups, "rows": rows,
                        "seconds": round(time.monotonic() - start, 1)}), flush=True)
        if pending:
            write_batch(writer, pending)
    if rows != pq.ParquetFile(args.source).metadata.num_rows:
        raise ValueError("Secondary output omitted or added candidate rows")
    if (EVAL.sha256_file(args.source) != source_hash or EVAL.sha256_file(source_metadata_path) != metadata_hash
            or EVAL.sha256_file(args.reference_model) != reference_hash
            or check_encoder_base(config["embedding_model"], manifest, args.encoder_base_reference) != base):
        raise ValueError("Secondary source/model/base changed during derivation")
    for entry in (index_identity["index"], index_identity["metadata"],
                  primary_index_identity["index"], primary_index_identity["metadata"], *training_inputs.values()):
        if EVAL.sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Secondary index/training inputs changed during query-only rescoring")
    if any(EVAL.sha256_file(Path(path)) != expected for path, expected in source_closure.items()):
        raise ValueError("Secondary producer or input-rendering source changed")
    # Exclusive publication prevents a concurrent writer from being replaced.
    os.link(temporary, args.output)
    temporary.unlink()
    derived = dict(metadata)
    if args.mode == "train":
        derived["feature_version"] = 9
        source_priors = args.source.with_suffix(args.source.suffix + ".priors.joblib")
        with source_priors.open("rb") as source, priors_output.open("xb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        if EVAL.sha256_file(priors_output) != training_inputs["priors"]["sha256"]:
            raise ValueError("Copied OOF inference-prior bytes differ from the V8 source")
    else:
        derived["feature_schema_version"] = 9
        derived["cache_version"] = 3
        derived["source_reference_model_sha256"] = derived.pop("model_sha256")
    derived["feature_names"] = [*FEATURE_SCHEMAS[8], *ADDED_FEATURES]
    derived["secondary_dense_derivation"] = {
        "version": 1, "complete": True, "standalone_schema": 9, "production_schema_registered": False,
        "source": {"path": str(args.source.resolve()), "sha256": source_hash},
        "source_metadata": {"path": str(source_metadata_path.resolve()), "sha256": metadata_hash},
        "reference_model": {"path": str(args.reference_model.resolve()), "sha256": reference_hash},
        "protocol_sha256": protocol_hash, "source_feature_version": 8,
        "encoder_base": base, "compact_index": index_identity, "primary_index": primary_index_identity,
        "training_inputs": training_inputs, "source_sha256": source_closure,
        "query_format": "current query_text with saved query prompt, identical to primary V8",
        "query_vectors_shared_across_both_views": True, "primary_cosine_consistency_tolerance": .005,
        "raw_train_representative_rule": "last fit in-corpus raw row, matching read_fit_labels" if args.mode == "train" else None,
        "representative_source_sha256": manifest["files"]["train"]["sha256"],
        "representative_reader_batch_size": args.representative_batch_size,
        "primary_candidate_rows_unchanged": True, "original_columns_unchanged": True,
        "retrieval_performed": False, "priors_refitted": False, "research_prior_policy": policy,
        "rows": rows, "nonempty_groups": groups,
        "producer": {"path": str(Path(__file__).resolve()), "sha256": EVAL.sha256_file(Path(__file__))},
        "output_parquet_sha256": EVAL.sha256_file(args.output),
        "stage": "exploratory same-pool feature augmentation; not an independent test"}
    if args.mode == "tune":
        derived["cache_parquet_sha256"] = EVAL.sha256_file(args.output)
    EVAL.write_new_json(output_metadata, derived)
    marker.unlink()
    print(json.dumps({"complete": True, "output": str(args.output), "rows": rows,
                      "production_schema_registered": False}), flush=True)
    return derived


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "tune"], required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--reference-train-data", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact-cache", type=Path, required=True)
    parser.add_argument("--current-cache", type=Path, required=True)
    parser.add_argument("--encoder-base-reference", type=Path, required=True)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--row-batch-size", type=int, default=8192)
    parser.add_argument("--representative-batch-size", type=int, default=65536)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.query_batch_size <= 0 or args.row_batch_size <= 0 or args.representative_batch_size <= 0:
        parser.error("Query and row batch sizes must be positive")
    derive(args)


if __name__ == "__main__":
    main()
