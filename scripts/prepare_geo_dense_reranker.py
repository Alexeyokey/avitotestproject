"""Prepare fit-only V8 rows for geography before the dense candidate cutoff.

The old encoder, catalog vectors, classifier folds and aggregate priors are
restored unchanged. Only fit query vectors are needed. Every history feature
subtracts the current query text; the category classifier uses its held-out
text fold. This script has no Tune/Audit target reader and never adds a positive
item to retrieval. Missing retrieved positives follow the existing sampler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from avito_candidates.core import SEARCH_FIELDS, normalize, query_key
import evaluate_fresh_validation as EVAL


GEO_CONFIG = {
    "dense_retrieval_policy": "exact-cosine-plus-location-log-prior-v1",
    "geo_dense_weight": .05,
    "geo_dense_smoothing_mass": 3.,
    "geo_dense_distance_scale_km": 40.,
    "geo_dense_probability_floor": 1e-7,
}


def validate_geo_config(config):
    if any(config.get(key) != value for key, value in GEO_CONFIG.items()):
        raise ValueError("Retrieval configuration does not identify the fixed geo-dense policy")
    return dict(GEO_CONFIG)


def group_id(key):
    value = json.dumps(list(key), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def selected_fit_keys(truth, representatives, fit_texts, maximum=30000):
    """Exactly the existing prepare_training_data hash order, including misses."""
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise ValueError("Expected a positive fit group budget")
    for key in truth:
        if (len(key) != len(SEARCH_FIELDS) or any(not isinstance(x, str) for x in key)
                or key[0] not in fit_texts or key != query_key(representatives[key])):
            raise ValueError("Fit truth contains a non-fit or inconsistent query key")
    return sorted(truth, key=lambda key: hashlib.sha256(repr(key).encode()).digest())[:maximum]


def _checked_files(root, manifest, names):
    if not set(names) <= set(manifest.get("files", {})):
        raise ValueError("Vector manifest omits required files")
    for name in names:
        if EVAL.sha256_file(root / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Frozen vector content differs: {name}")


def verify_fit_vectors(root, corpus_root, keys, corpus_ids, *, protocol_hash, metadata_hash):
    """Bind all reusable arrays to the complete, ordered 30k fit selection."""
    root, corpus_root = Path(root), Path(corpus_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("complete") is not True or manifest.get("phase") != "fit"
            or manifest.get("fit_only") is not True
            or manifest.get("tune_labels_opened") is not False
            or manifest.get("audit_labels_opened") is not False
            or manifest.get("protocol_sha256") != protocol_hash
            or manifest.get("metadata_sha256") != metadata_hash
            or manifest.get("inventory_reencoded") is not False):
        raise ValueError("Fit vector manifest is incomplete or belongs to another experiment")
    names = ("order.json", "queries.npy", "ann998.npy", "ann998-cosines.npy")
    _checked_files(root, manifest, names)
    order = json.loads((root / "order.json").read_text(encoding="utf-8"))
    if (order.get("query_keys") != [list(key) for key in keys]
            or order.get("query_group_ids") != [group_id(key) for key in keys]
            or order.get("corpus_item_ids") != corpus_ids):
        raise ValueError("Fit vectors do not match exact ordered query keys and catalog IDs")
    corpus_manifest = json.loads((corpus_root / "manifest.json").read_text(encoding="utf-8"))
    if (corpus_manifest.get("complete") is not True
            or corpus_manifest.get("inventory_reencoded") is not False
            or corpus_manifest.get("protocol_sha256") != protocol_hash):
        raise ValueError("Original corpus-vector manifest does not match the protocol")
    _checked_files(corpus_root, corpus_manifest, ("corpus.npy", "order.json"))
    if (manifest.get("corpus_vectors_sha256") != corpus_manifest["files"]["corpus.npy"]["sha256"]
            or not corpus_manifest.get("inventory_index_inputs")
            or manifest.get("inventory_index_inputs") != corpus_manifest["inventory_index_inputs"]
            or manifest.get("encoder_base") != corpus_manifest.get("encoder_base")):
        raise ValueError("Fit vectors are not bound to the original encoder and inventory index")
    corpus_order = json.loads((corpus_root / "order.json").read_text(encoding="utf-8"))
    if corpus_order.get("corpus_item_ids") != corpus_ids:
        raise ValueError("Original corpus vector order differs from the frozen inventory")
    arrays = {name: np.load(root / (name + ".npy"), mmap_mode="r")
              for name in ("queries", "ann998", "ann998-cosines")}
    arrays["corpus"] = np.load(corpus_root / "corpus.npy", mmap_mode="r")
    corpus, queries = arrays["corpus"], arrays["queries"]
    if (corpus.ndim != 2 or corpus.shape[0] != len(corpus_ids)
            or queries.shape != (len(keys), corpus.shape[1])
            or corpus.dtype != np.float32 or queries.dtype != np.float32):
        raise ValueError("Fit/corpus vector shape or dtype differs")
    expected = (len(keys), min(998, len(corpus_ids)))
    if arrays["ann998"].shape != expected or arrays["ann998-cosines"].shape != expected:
        raise ValueError("Fit supplemental ANN shape differs")
    if not np.issubdtype(arrays["ann998"].dtype, np.integer):
        raise ValueError("ANN corpus labels must have integer dtype")
    return manifest, corpus_manifest, arrays


class PreencodedFitRetriever:
    """Existing trainer interface, with ordered fit vectors and no encoder."""

    def __init__(self, hybrid, micro, prior, scorer, arrays, keys, item_ids, *,
                 micro_weight, shared_fn=None, compose_fn=None, ann_fn=None):
        if shared_fn is None or compose_fn is None or ann_fn is None:
            from evaluate_geo_dense_tune import raw_shared_routes, compose_result, ann_result
            shared_fn = shared_fn or raw_shared_routes
            compose_fn = compose_fn or compose_result
            ann_fn = ann_fn or ann_result
        self.hybrid, self.micro, self.prior, self.scorer = hybrid, micro, prior, scorer
        self.arrays, self.keys, self.item_ids = arrays, list(keys), list(item_ids)
        self.id_to_row = {item: row for row, item in enumerate(item_ids)}
        self.micro_weight = micro_weight
        self.shared_fn, self.compose_fn, self.ann_fn = shared_fn, compose_fn, ann_fn
        self.cursor = 0
        self.pool_sum, self.maximum_pool, self.minimum_pool = 0, 0, 998
        # make_retriever is shared with Tune, where this flag is deliberately
        # false. Training must restore exclusion before any query is fetched.
        self.hybrid.bm25.geo_leave_query_out = True

    def retrieve_batch_with_diagnostics(self, queries):
        start, end = self.cursor, self.cursor + len(queries)
        if [query_key(query) for query in queries] != self.keys[start:end]:
            raise ValueError("Training requests differ from frozen fit vector order")
        cosines = self.scorer.score(self.arrays["queries"][start:end])
        results = []
        for row, query, cosine in zip(range(start, end), queries, cosines):
            supplemental = self.ann_fn(self.arrays["ann998"][row],
                self.arrays["ann998-cosines"][row], self.item_ids)
            shared = self.shared_fn(query, self.hybrid, self.micro, supplemental)
            dense = self.prior.retrieve(cosine, query.get("search_location_id", ""),
                limit=449, weight=.05, exclude_query_text=normalize(query.get("search_query", "")))
            result = self.compose_fn(query, self.hybrid, shared, dense, cosine,
                                     self.id_to_row, self.micro_weight)
            pool = result[1]["candidate_pool"]
            if len(pool) > 998 or len(set(pool)) != len(pool):
                raise ValueError("Training pool exceeds 998 or contains duplicates")
            self.pool_sum += len(pool)
            self.maximum_pool = max(self.maximum_pool, len(pool))
            self.minimum_pool = min(self.minimum_pool, len(pool))
            results.append(result)
        if len(results) != len(queries):
            raise ValueError("Exact scorer omitted a training query")
        self.cursor = end
        return results


def check_old_training_groups(path, keys):
    """Old pool misses lack rows; surviving groups must be in this same selection."""
    import pyarrow.parquet as pq
    allowed = {group_id(key): key for key in keys}
    observed = set()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536,
            columns=["group_id", "query_key_json"]):
        for identity, serialized in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            if identity not in allowed or tuple(json.loads(serialized)) != allowed[identity]:
                raise ValueError("Old training cache contains groups outside the identical fit selection")
            observed.add(identity)
    if not observed:
        raise ValueError("Old training cache has no groups")
    return len(observed)


def run(args):
    from build_fresh_validation_protocol import load_manifest
    from cache_research_reranker import (validate_research_artifact, check_training_prior_file,
        research_prior_policy, source_snapshot, check_sources)
    from evaluate_geo_dense_tune import make_retriever
    from produce_document_hypotheses import iter_catalog_records, require_offline_environment
    from avito_candidates.reranker import (FeatureBuilder, V8_FEATURES,
        corpus_fingerprint, read_fit_labels, prepare_training_data)
    from avito_candidates.geo_dense import GeoDensePrior, ExactDenseScorer
    import joblib

    require_offline_environment()
    if args.batch_size <= 0 or args.max_train_queries != 30000:
        raise ValueError("This comparison uses exactly the old 30000-group budget and positive batches")
    output = args.output_data
    outputs = [output, output.with_suffix(output.suffix + ".json"),
               output.with_suffix(output.suffix + ".priors.joblib"),
               output.with_suffix(output.suffix + ".INCOMPLETE")]
    if any(path.exists() for path in outputs):
        raise ValueError("Use a fresh output path; completed or partial training data already exists")
    started = time.monotonic()
    # Parse the sealed manifest, then hash ONLY the requested fit/catalog files.
    # In particular, do not use checked_protocol, which hashes all label arms.
    manifest = load_manifest(args.protocol_dir, verify=False)
    protocol_hash = EVAL.sha256_file(args.protocol_dir / "manifest.json")
    fit_path = EVAL.protocol_file(args.protocol_dir, manifest, "fit_texts")
    train_path = EVAL.protocol_file(args.protocol_dir, manifest, "train")
    item_path = EVAL.protocol_file(args.protocol_dir, manifest, "benchmark_items")
    fit_texts = set(json.loads(fit_path.read_text(encoding="utf-8"))["texts"])
    model_hash, train_hash = map(EVAL.sha256_file, (args.source_model, args.source_train_data))
    old_metadata_path = args.source_train_data.with_suffix(args.source_train_data.suffix + ".json")
    metadata_hash = EVAL.sha256_file(old_metadata_path)
    old_metadata = json.loads(old_metadata_path.read_text(encoding="utf-8"))
    if old_metadata.get("stats", {}).get("sampled_queries") != 30000:
        raise ValueError("Source training cache does not declare the same 30000 selected groups")
    artifact = joblib.load(args.source_model)
    validate_research_artifact(artifact, fit_texts, manifest)
    EVAL.check_encoder_identity(artifact, manifest)
    EVAL.check_training_data(args.source_train_data, artifact, args.protocol_dir, manifest)
    prior_file_hash = check_training_prior_file(args.source_train_data, artifact)
    if artifact["metadata"] != old_metadata:
        raise ValueError("Source model and training metadata differ")
    source_policy = research_prior_policy(artifact)
    priors = artifact["priors"]
    classifier = priors["microcategory_classifier"]
    if classifier.metadata.get("crossfit_folds") != 3 or set(classifier.crossfit_models) != {0, 1, 2}:
        raise ValueError("Restore exactly the proven three grouped-OOF classifier models")
    geo, micro = priors["geo_associations"], priors["microcategory_prior"]
    if (getattr(geo, "query_counts", None) is None
            or not set(geo.query_counts) <= fit_texts
            or not set(micro.exact_counts) <= fit_texts):
        raise ValueError("Saved aggregate priors are not tracked fit-only query histories")
    config = {**old_metadata["retrieval_config"], **GEO_CONFIG}
    validate_geo_config(config)
    items = list(iter_catalog_records(item_path))
    if (old_metadata["feature_version"] != 8
            or corpus_fingerprint(items, feature_version=8) != old_metadata["corpus_fingerprint"]):
        raise ValueError("Frozen source model requires exactly the original V8 inventory")
    ids = [item["item_id"] for item in items]
    truth, representatives, fit_stats = read_fit_labels(train_path, set(ids),
        split="queries", fraction=.2, seed=42, batch_size=4096)
    keys = selected_fit_keys(truth, representatives, fit_texts, args.max_train_queries)
    if len(keys) != 30000:
        raise ValueError("Frozen fit partition no longer yields the original 30000 groups")
    old_surviving_groups = check_old_training_groups(args.source_train_data, keys)
    vector_manifest, corpus_manifest, arrays = verify_fit_vectors(args.fit_vector_dir,
        args.corpus_vector_dir, keys, ids, protocol_hash=protocol_hash, metadata_hash=metadata_hash)
    snapshot = source_snapshot()
    root = Path(__file__).resolve().parents[1]
    for path in (Path(__file__), root / "scripts/evaluate_geo_dense_tune.py",
                 root / "src/avito_candidates/geo_dense.py"):
        snapshot[str(path.resolve())] = EVAL.sha256_file(path)
    hybrid, channel = make_retriever(config, items, priors)
    builder = FeatureBuilder(items, config, microcategory_prior=micro, geo_associations=geo,
        microcategory_classifier=classifier, feature_names=V8_FEATURES,
        leave_query_out=True, leave_geography_query_out=True)
    prior = GeoDensePrior(items, geo, smoothing_mass=3., distance_scale_km=40., probability_floor=1e-7)
    scorer = ExactDenseScorer(arrays["corpus"], device=args.device)
    retriever = PreencodedFitRetriever(hybrid, channel, prior, scorer, arrays, keys, ids,
                                      micro_weight=config["microcategory_weight"])
    declaration = {"complete": False, "phase": "fit", "tune_labels_read": False,
        "audit_labels_read": False, "inventory_reencoded": False,
        "protocol_sha256": protocol_hash, "fit_texts_sha256": EVAL.sha256_file(fit_path),
        "fit_train_sha256": EVAL.sha256_file(train_path),
        "source_model_sha256": model_hash, "source_train_data_sha256": train_hash,
        "source_train_metadata_sha256": metadata_hash, "source_training_priors_sha256": prior_file_hash,
        "ordered_fit_keys_sha256": EVAL.canonical_sha256([list(key) for key in keys]),
        "selected_groups": len(keys), "old_surviving_training_groups": old_surviving_groups,
        "fit_vector_manifest_sha256": EVAL.sha256_file(args.fit_vector_dir / "manifest.json"),
        "corpus_vector_sha256": corpus_manifest["files"]["corpus.npy"]["sha256"],
        "embedding_checkpoint_sha256": old_metadata["embedding_checkpoint_sha256"],
        "geo_dense_policy": dict(GEO_CONFIG), "maximum_pool_size": 998,
        "negatives_per_query": 64, "fit_stats": fit_stats,
        "training_geo_dense_history": "leave-query-text-out",
        "source_prior_policy": source_policy, "source_sha256": snapshot}
    output.parent.mkdir(parents=True, exist_ok=True)
    marker = outputs[-1]
    marker.write_text(json.dumps(declaration, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"geo_dense_prepare": declaration}), flush=True)
    metadata = prepare_training_data(output, retriever, builder, truth, representatives,
        split="queries", fraction=.2, seed=42, negatives=64,
        batch_size=args.batch_size, max_queries=30000)
    if retriever.cursor != len(keys) or metadata["stats"]["sampled_queries"] != 30000:
        raise ValueError("Preparation did not consume the entire frozen fit selection")
    check_sources(snapshot)
    if (EVAL.sha256_file(args.source_model) != model_hash
            or EVAL.sha256_file(args.source_train_data) != train_hash
            or EVAL.sha256_file(old_metadata_path) != metadata_hash
            or research_prior_policy({"priors": priors}) != source_policy):
        raise ValueError("Source model, training data or restored priors changed during preparation")
    declaration.update(complete=True, elapsed_seconds=time.monotonic() - started,
        actual_maximum_pool_size=retriever.maximum_pool, actual_minimum_pool_size=retriever.minimum_pool,
        average_pool_size=retriever.pool_sum / len(keys))
    metadata["geo_dense_training"] = declaration
    outputs[1].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    marker.unlink()
    print(json.dumps({"geo_dense_prepare_complete": declaration, "stats": metadata["stats"]}), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol-dir", "source-model", "source-train-data", "fit-vector-dir",
                 "corpus-vector-dir", "output-data"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-train-queries", type=int, default=30000)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
