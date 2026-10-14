"""Produce target-blind document-layout channels on the sealed Tune inventory.

Changing the document layout needs a separate corpus index: current vectors
cannot represent the newly retained description. A completed matching index
is reused; partial DenseRetriever checkpoints resume naturally. This script
never opens Tune/Audit relevance files or inserts a selected item. Queries
come only from the immutable Tune cache metadata.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
from pathlib import Path
import time

from avito_candidates.core import SEARCH_FIELDS
from avito_candidates.dense import (DenseRetriever, _fingerprint,
                                    local_checkpoint_content_fingerprint,
                                    offline_adapter_base_reference)
from avito_candidates.embedding_formats import FORMAT_VERSION
from avito_candidates.reranker import corpus_fingerprint
import evaluate_fresh_validation as EVAL
from evaluate_retrieval_hypotheses import candidate_metadata
import produce_retrieval_hypotheses as HELPER

QUERY_VARIANTS = ("current", "plain", "values", "no_category")
DOCUMENT_FORMATS = ("compact", "title_description")
DOCUMENT_FIELDS = ("item_id", "item_title_raw", "item_category_id",
                   "item_infm_params_text", "item_description_raw")


def iter_catalog_records(path, batch_size=1024):
    """Match CLI string conversion without materializing the full DataFrame.

Pandas converts a nullable integer column to float64 for the *whole* table.
Individual Arrow batches without that column's nulls would otherwise become
integers and produce different strings (114 vs 114.0). Global null counts
preserve the legacy conversion, including its numerical representation.
"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if batch_size <= 0:
        raise ValueError("Metadata batch size must be positive")
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    if any(pa.types.is_nested(field.type) for field in schema):
        raise ValueError("Catalog reader requires flat scalar metadata columns")
    integer_columns = [field.name for field in schema if pa.types.is_integer(field.type)]
    nullable_integers = set()
    unknown_null_counts = []
    for name in integer_columns:
        index = schema.get_field_index(name)
        counts = [parquet.metadata.row_group(row).column(index).statistics
                  for row in range(parquet.metadata.num_row_groups)]
        if any(stats is None or stats.null_count is None for stats in counts):
            unknown_null_counts.append(name)
        elif any(stats.null_count for stats in counts):
            nullable_integers.add(name)
    # Old or stats-disabled Parquet files still get the exact conversion.
    if unknown_null_counts:
        for batch in parquet.iter_batches(batch_size=batch_size, columns=unknown_null_counts):
            for name in unknown_null_counts:
                if batch.column(batch.schema.get_field_index(name)).null_count:
                    nullable_integers.add(name)
    for batch in parquet.iter_batches(batch_size=batch_size):
        frame = batch.to_pandas()
        for name in nullable_integers:
            frame[name] = frame[name].astype("float64")
        for record in frame.fillna("").astype(str).to_dict("records"):
            yield record


def load_catalog_documents(path, batch_size=1024):
    """Check the full feature fingerprint before retaining encoder fields."""
    import pyarrow as pa
    items = list(iter_catalog_records(path, batch_size))
    fingerprint = corpus_fingerprint(items, feature_version=4)
    # Replace one record at a time so the old and new complete inventories
    # never coexist. Metadata required by the encoder remains byte-identical.
    for index, item in enumerate(items):
        items[index] = {name: item.get(name, "") for name in DOCUMENT_FIELDS}
    gc.collect()
    pa.default_memory_pool().release_unused()
    return items, fingerprint


def check_encoder(manifest, config):
    reference = manifest.get("encoder", {})
    model_path = Path(config["embedding_model"]).resolve()
    if (model_path != Path(reference.get("path", "")).resolve()
            or local_checkpoint_content_fingerprint(model_path) != reference.get("content_sha256")):
        raise ValueError("Frozen encoder path or bytes differ from the sealed protocol")
    if config.get("embedding_max_length") != 128:
        raise ValueError("Document-layout experiment requires the unchanged 128-token window")
    return model_path


def write_encoder_base_reference(model_dir, training_metadata, output):
    """Capture the already-recorded training base, verified against local cache.

This is a binding for subsequent document experiments. It does not fabricate
evidence of which cached base a previously completed query run had loaded.
"""
    model_dir, training_metadata, output = map(Path, (model_dir, training_metadata, output))
    metadata = json.loads(training_metadata.read_text(encoding="utf-8"))
    current = offline_adapter_base_reference(model_dir)
    if (current.get("is_adapter") is not True or not current.get("base_revision")
            or current["base_repo"] != metadata.get("embedding_base_repo")
            or current["base_revision"] != metadata.get("embedding_base_revision")):
        raise ValueError("Current cached base differs from the recorded training base")
    content_hash = local_checkpoint_content_fingerprint(model_dir)
    if content_hash != metadata.get("embedding_checkpoint_sha256"):
        raise ValueError("Local adapter differs from the recorded training checkpoint")
    reference = {"reference_version": 1,
        "scope": "Frozen cached base binding for subsequent document-layout experiments",
        "encoder_path": str(model_dir.resolve()), "encoder_content_sha256": content_hash,
        **current, "training_metadata": {"path": str(training_metadata.resolve()),
            "sha256": HELPER.sha256(training_metadata)},
        "historical_query_run_base_capture": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(reference, ensure_ascii=False, indent=2) + "\n")
    return reference


def check_encoder_base(model_path, manifest, reference_path):
    reference_path = Path(reference_path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if (reference.get("reference_version") != 1 or reference.get("is_adapter") is not True
            or Path(reference.get("encoder_path", "")).resolve() != Path(model_path).resolve()
            or reference.get("encoder_content_sha256") != manifest["encoder"]["content_sha256"]):
        raise ValueError("Frozen base reference differs from the sealed encoder")
    expected = {key: reference.get(key) for key in ("is_adapter", "base_repo", "base_revision")}
    if not expected["base_repo"] or not expected["base_revision"]:
        raise ValueError("Frozen base reference must bind an actual cached commit")
    current = offline_adapter_base_reference(model_path)
    if current != expected:
        raise ValueError("Cached adapter base changed from the declared frozen reference")
    recorded = reference.get("training_metadata", {})
    training_path = Path(recorded.get("path", ""))
    if not training_path.is_file() or HELPER.sha256(training_path) != recorded.get("sha256"):
        raise ValueError("Recorded training-base metadata changed or is missing")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if (training.get("embedding_base_repo") != current["base_repo"]
            or training.get("embedding_base_revision") != current["base_revision"]):
        raise ValueError("Declared base differs from its recorded training provenance")
    return {"reference": {"path": str(reference_path.resolve()), "sha256": HELPER.sha256(reference_path)},
            **current}


def require_offline_environment():
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("Run the frozen-encoder experiment with HF_HUB_OFFLINE=1")


def rename_channels(rows, document_format, variants):
    """Keep document and query axes explicit in candidate channel names."""
    if document_format not in DOCUMENT_FORMATS:
        raise ValueError("Only explicit experimental document formats are permitted")
    expected = {"dense_" + variant for variant in variants}
    renamed = []
    for row in rows:
        if set(row) != expected:
            raise ValueError("Encoded query channels differ from the declared variants")
        renamed.append({"dense_" + document_format + "_" + variant: row["dense_" + variant]
                        for variant in variants})
    return renamed


def completed_index_identity(dense, items, cache_dir):
    fingerprint = _fingerprint(items, dense.model_name, dense.max_seq_length,
        dense.params_words, dense.description_words, dense.model_signature,
        document_format=dense.document_format)
    root = Path(cache_dir).resolve()
    index, metadata = root / (fingerprint + ".bin"), root / (fingerprint + ".json")
    if (not index.is_file() or not metadata.is_file()
            or json.loads(metadata.read_text(encoding="utf-8")) != dense._metadata()):
        raise ValueError("Experimental document index is not complete or metadata differs")
    return {"index": {"path": str(index), "sha256": HELPER.sha256(index)},
            "metadata": {"path": str(metadata), "sha256": HELPER.sha256(metadata)},
            "fingerprint": fingerprint}


def produce(args):
    require_offline_environment()
    root = args.protocol_dir.resolve()
    manifest, protocol_hash = EVAL.checked_protocol(root)
    cache_path, metadata_path = EVAL.phase_cache_paths(args.cache_dir, "tune")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cache_hash = EVAL.check_phase_cache_inputs(cache_path, metadata, "tune", protocol_hash,
                                              metadata.get("model_sha256"))
    metadata_hash = HELPER.sha256(metadata_path)
    fit_path = EVAL.protocol_file(root, manifest, "fit-texts.json")
    fit_texts = set(json.loads(fit_path.read_text(encoding="utf-8"))["texts"])
    groups = HELPER.selected_groups(metadata, fit_texts)
    sidecar = Path(str(args.output) + ".json")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("Use a new candidate output path; existing outputs are immutable")
    config = metadata["retrieval_config"]
    model_path = check_encoder(manifest, config)
    base_reference = check_encoder_base(model_path, manifest, args.encoder_base_reference)
    item_path = EVAL.protocol_file(root, manifest, "benchmark_items")
    items, inventory_fingerprint = load_catalog_documents(item_path)
    if inventory_fingerprint != metadata.get("corpus_fingerprint"):
        raise ValueError("Inventory feature fingerprint differs from the baseline cache")
    started = time.monotonic()
    dense = DenseRetriever(items, model_name=str(model_path), cache_dir=args.dense_cache,
        batch_size=args.batch_size, max_seq_length=config["embedding_max_length"],
        params_words=config["embedding_params_words"],
        description_words=config["embedding_description_words"],
        ef_search=config["ef_search"], device=args.device,
        document_format=args.document_format)
    index_identity = completed_index_identity(dense, items, args.dense_cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    open_fn = gzip.open if args.output.suffix == ".gz" else open
    # Exclusive creation also rejects a concurrent producer writing this path.
    compression_options = {"compresslevel": 1} if args.output.suffix == ".gz" else {}
    with open_fn(args.output, "xt", encoding="utf-8", **compression_options) as stream:
        for start in range(0, len(groups), args.batch_size):
            batch = groups[start:start + args.batch_size]
            queries = [dict(zip(SEARCH_FIELDS, group["query_key"])) for group in batch]
            channels = rename_channels(HELPER.dense_query_channels(
                dense, queries, args.query_variants, args.depth),
                args.document_format, args.query_variants)
            for group, ranked in zip(batch, channels):
                stream.write(json.dumps({"group_id": group["group_identity"],
                    "query_key": group["query_key"], "channels": ranked},
                    ensure_ascii=False, separators=(",", ":")) + "\n")
            if start == 0 or (start // args.batch_size + 1) % 10 == 0 or start + len(batch) == len(groups):
                print(json.dumps({"document_format": args.document_format,
                    "groups": start + len(batch), "total": len(groups),
                    "seconds": round(time.monotonic() - started, 1)}), flush=True)
    # Refuse provenance from a baseline/checkpoint that changed mid-run.
    if HELPER.sha256(metadata_path) != metadata_hash or HELPER.sha256(cache_path) != cache_hash:
        raise ValueError("Immutable baseline cache changed during document production")
    check_encoder(manifest, config)
    if check_encoder_base(model_path, manifest, args.encoder_base_reference) != base_reference:
        raise ValueError("Frozen encoder base declaration changed during document production")
    model_source = Path(__file__).resolve().parents[1] / "src/avito_candidates/embedding_formats.py"
    dense_source = Path(__file__).resolve().parents[1] / "src/avito_candidates/dense.py"
    result = candidate_metadata(args.output, root, args.cache_dir, Path(__file__),
        provenance_kind="frozen_encoder", extra={
            "kind": "document_layout", "document_format": args.document_format,
            "format_version": FORMAT_VERSION, "query_variants": args.query_variants,
            "depth": args.depth, "max_length": config["embedding_max_length"],
            "params_words": config["embedding_params_words"],
            "description_words": config["embedding_description_words"],
            "groups": len(groups), "seconds": time.monotonic() - started,
            "encoder_content_sha256": manifest["encoder"]["content_sha256"],
            "encoder_base_reference": base_reference,
            "hf_hub_offline": True,
            "separate_completed_index": index_identity,
            "helper_source": {"path": str(Path(HELPER.__file__).resolve()),
                              "sha256": HELPER.sha256(HELPER.__file__)},
            "formats_source": {"path": str(model_source), "sha256": HELPER.sha256(model_source)},
            "dense_source": {"path": str(dense_source), "sha256": HELPER.sha256(dense_source)},
            "inventory_reader": "global nullable-integer Pandas conversion in 1024-row Arrow batches; full fingerprint before encoder-field projection",
            "gzip_compresslevel": 1 if args.output.suffix == ".gz" else None,
            "tuning_labels_read": False})
    with sidecar.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"complete": True, "output": str(args.output), "metadata": str(sidecar)}), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dense-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--document-format", choices=DOCUMENT_FORMATS, required=True)
    parser.add_argument("--encoder-base-reference", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--depth", type=int, default=449)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-variants", choices=QUERY_VARIANTS, nargs="+",
                        default=["current", "plain", "values", "no_category"])
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or not 0 < args.depth <= 998:
        parser.error("Require positive batch size and depth between 1 and 998")
    if len(args.query_variants) != len(set(args.query_variants)):
        parser.error("Query variants must not repeat")
    produce(args)


if __name__ == "__main__":
    main()
