"""Add V6 dense-union features to a completed, fixed validation candidate cache.

This streams cached rows and never retrieves or encodes anything. Original
features and candidate order are preserved. Fit-side rows are not copied:
leave-query-text-out geography requires its own training-side cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from reranker_audit_provenance import artifact_prior_policy, _atomic_json, _file_sha256


EXTRA_FEATURES = ["full_dense_score", "full_dense_known"]
ALIGNMENT_PROTOCOL = "query-norm-recovered-from-hnsw-cosines-v1"
MIN_DENSE_PAIRS = 8
MAX_ALIGNMENT_RESIDUAL = 1e-5
ROW_IDENTITY_COLUMNS = ["group_id", "item_id", "label", "positive_local", "positive_seen"]
GROUP_IDENTITY_COLUMNS = [
    "group_id", "query_key", "query_text", "relevant_count", "local_relevant_count",
    "nonlocal_relevant_count", "seen_relevant_count", "unseen_relevant_count",
    "pool_size", "positives_in_pool", "baseline_hits",
]
ITEM_COLUMNS = (
    "item_id", "item_title_raw", "item_infm_params_text", "item_description_raw",
    "item_category_id", "item_location_id", "item_microcat_id", "item_price",
    "item_rating", "item_rating_reviews_count", "item_is_phone_hidden",
    "item_is_message_forbidden",
)


def _canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8") + b"\n"


def validate_rescore_config(source, target):
    """Only enabling union rescoring may differ; no quotas/depths/weights change."""
    if target.get("rescore_dense_union") is not True:
        raise ValueError("V6 target must enable rescore_dense_union")
    for config in (source, target):
        if config.get("rescore_dense_union", False) not in (False, True):
            raise ValueError("rescore_dense_union must be a boolean")
    before = {key: value for key, value in source.items() if key != "rescore_dense_union"}
    after = {key: value for key, value in target.items() if key != "rescore_dense_union"}
    if before != after:
        changed = sorted(key for key in before.keys() | after.keys()
                         if before.get(key) != after.get(key) or (key in before) != (key in after))
        raise ValueError(f"Retrieval settings changed beyond union rescoring: {changed}")


def validate_bridge_metadata(source, training, artifact):
    target = artifact["metadata"]
    old_features = source.get("feature_names")
    if source.get("feature_schema_version") != 4 or not isinstance(old_features, list) or len(old_features) != 75:
        raise ValueError("Source must use the original 75-feature V4 schema")
    expected = [*old_features, *EXTRA_FEATURES]
    if len(set(expected)) != len(expected):
        raise ValueError("Source features already contain dense-union or duplicate names")
    for name, metadata in (("training", training), ("model", target)):
        if metadata.get("feature_version") != 6 or metadata.get("feature_names") != expected:
            raise ValueError(f"{name} must use exactly source75 + full_dense_score + full_dense_known")
        validate_rescore_config(source["retrieval_config"], metadata["retrieval_config"])
        for key in ("split", "seed", "validation_fraction"):
            if metadata.get(key) != source.get(key):
                raise ValueError(f"{key} changed between source and {name}")
    for key in ("retrieval_config", "corpus_fingerprint", "training_prior_policy"):
        if training.get(key) != target.get(key):
            raise ValueError(f"Training/model {key} differ")
    policy = artifact_prior_policy(artifact)
    for key, value in policy.items():
        if key not in source:
            raise ValueError(f"Source inference policy is missing {key}; finalize its provenance first")
        if source[key] != value:
            raise ValueError(f"Source/model inference prior policy differs: {key}")
    return expected


def verify_corpus(source, target, data_dir, *, items=None, fingerprint_function=None):
    """Verify both schema-version fingerprints against the actual item inventory."""
    if fingerprint_function is None:
        from avito_candidates.reranker import corpus_fingerprint
        fingerprint_function = corpus_fingerprint
    if items is None:
        from avito_candidates.cli import read
        items = read(Path(data_dir) / "benchmark_items.parquet", ITEM_COLUMNS)
    old = fingerprint_function(items, feature_version=4)
    new = fingerprint_function(items, feature_version=6)
    if old != source.get("corpus_fingerprint") or new != target.get("corpus_fingerprint"):
        raise ValueError("Source/model corpus fingerprints do not match the actual inventory")
    return {"source_v4": old, "target_v6": new, "items": len(items)}


def derive_validation_batch(batch):
    """Validate finite cached dot scores and append a known=1 flag.

    Raw scores can exceed one when the originally encoded query was not
    exactly unit length. The cosine bound is checked after norm recovery.
    """
    import numpy as np
    import pyarrow as pa
    names = batch.schema.names
    if "full_dense_score" not in names:
        raise ValueError("Source cache has no full_dense_score column")
    scores = batch.column("full_dense_score").to_numpy(zero_copy_only=False)
    if not np.isfinite(scores).all():
        raise ValueError("Source full dense scores must be finite cosine-compatible dots for every row")
    table = pa.Table.from_batches([batch]) if isinstance(batch, pa.RecordBatch) else batch
    if "full_dense_known" in names:
        known = batch.column("full_dense_known").to_numpy(zero_copy_only=False)
        if not np.isfinite(known).all() or np.any(known != 1.0):
            raise ValueError("Source full_dense_known contains unknown rows")
        if batch.schema.field("full_dense_known").type != pa.float32():
            raise ValueError("full_dense_known must be float32")
        return table
    return table.append_column("full_dense_known", pa.array(np.ones(len(scores), dtype=np.float32)))


def align_full_dense_group(table, *, min_pairs=MIN_DENSE_PAIRS,
                          maximum_residual=MAX_ALIGNMENT_RESIDUAL):
    """Recover the encoded query norm from paired HNSW and cached dot scores.

    For unit stored vectors u and cached query q, old dot=u.q, while HNSW
    cosine=u.(q/||q||). Their ratio is ||q||. A single positive scalar must
    explain every usable dense-channel row before applying it to the union.
    No labels, model predictions, or held-out relevance affect this recovery.
    """
    import numpy as np
    import pyarrow as pa
    if not set(("group_id", "score_dense", "present_dense", "rank_dense")) <= set(table.schema.names):
        raise ValueError("Norm recovery requires dense-channel scores, presence and ranks")
    group_ids = table["group_id"].to_pylist()
    if not group_ids or len(set(group_ids)) != 1:
        raise ValueError("Norm recovery requires exactly one complete nonempty query group")
    known_table = derive_validation_batch(table)
    raw = known_table["full_dense_score"].to_numpy(zero_copy_only=False).astype(np.float64)
    cosines = known_table["score_dense"].to_numpy(zero_copy_only=False).astype(np.float64)
    present = known_table["present_dense"].to_numpy(zero_copy_only=False)
    ranks = known_table["rank_dense"].to_numpy(zero_copy_only=False)
    if (not np.isfinite(cosines).all() or not np.isfinite(present).all()
            or not np.isfinite(ranks).all() or np.any((present > 0) != (ranks > 0))):
        raise ValueError("Dense-channel score/presence/rank columns are inconsistent")
    pairs = (present > 0) & (ranks > 0) & (np.abs(cosines) > 1e-3)
    if int(pairs.sum()) < min_pairs:
        raise ValueError(f"Insufficient safe dense pairs for group {group_ids[0]}: {int(pairs.sum())}")
    ratios = raw[pairs] / cosines[pairs]
    norm = float(np.median(ratios))
    if not np.isfinite(norm) or norm <= 0 or np.any(ratios <= 0):
        raise ValueError("Recovered query norm must be positive and finite")
    residual = float(np.max(np.abs(raw[pairs] / norm - cosines[pairs])))
    if residual > maximum_residual:
        raise ValueError(f"Dense ratios are not constant for group {group_ids[0]}: residual={residual}")
    normalized = (raw / norm).astype(np.float32)
    if not np.isfinite(normalized).all() or np.any(np.abs(normalized) > 1.0001):
        raise ValueError("Recovered normalized full dense scores are invalid cosines")
    # Only the extra V6 signal changes. score_dense and every original V4
    # feature remain byte-for-byte the source values.
    aligned = known_table.set_column(known_table.schema.get_field_index("full_dense_score"),
                                     "full_dense_score", pa.array(normalized))
    return aligned, {"query_norm": norm, "dense_pairs": int(pairs.sum()),
                     "max_abs_cosine_residual_vs_hnsw": residual,
                     "ratio_min": float(ratios.min()), "ratio_max": float(ratios.max())}


def iter_complete_query_groups(batches, *, max_pool_size=998):
    """Carry at most one candidate group across arbitrary Parquet batch cuts."""
    import numpy as np
    import pyarrow as pa
    pending_id, pending, count = None, [], 0
    for batch in batches:
        ids = batch.column("group_id").to_numpy(zero_copy_only=False)
        if not len(ids):
            continue
        boundaries = np.concatenate(([0], np.flatnonzero(ids[1:] != ids[:-1]) + 1, [len(ids)]))
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            group_id = int(ids[start])
            if pending_id is not None and group_id != pending_id:
                if group_id < pending_id:
                    raise ValueError("Candidate groups are not contiguous and ordered")
                yield pa.concat_tables(pending)
                pending, count = [], 0
            pending_id = group_id
            part = batch.slice(int(start), int(stop - start))
            pending.append(pa.Table.from_batches([part]))
            count += len(part)
            if count > max_pool_size:
                raise ValueError(f"Candidate group exceeds {max_pool_size} rows")
    if pending:
        yield pa.concat_tables(pending)


class CandidatePoolDigest:
    """Schema-independent proof of group denominators and ordered candidate rows."""
    def __init__(self, metadata):
        self.metadata = metadata
        self.groups = metadata["groups"]
        self.digest = hashlib.sha256(b"avito-candidate-pool-digest-v1\0")
        self.counts = [[0, 0, 0, 0, 0] for _group in self.groups]
        self.rows = 0
        self.previous_group = -1
        for expected_id, group in enumerate(self.groups):
            if group.get("group_id") != expected_id:
                raise ValueError("Group IDs must be ordered and contiguous")
            if any(key not in group for key in GROUP_IDENTITY_COLUMNS):
                raise ValueError("Source group lacks identity/denominator fields")
            if group["relevant_count"] < 1 or group["pool_size"] < 0:
                raise ValueError("Invalid corpus denominator or pool size")
            if (group["local_relevant_count"] + group["nonlocal_relevant_count"] != group["relevant_count"]
                    or group["seen_relevant_count"] + group["unseen_relevant_count"] != group["relevant_count"]):
                raise ValueError("Slice denominators do not sum to the corpus total")
            self.digest.update(_canonical_bytes({key: group[key] for key in GROUP_IDENTITY_COLUMNS}))

    def update(self, batch):
        if not set(ROW_IDENTITY_COLUMNS + ["baseline_top50"]) <= set(batch.schema.names):
            raise ValueError("Source cache lacks candidate identity or baseline columns")
        columns = [batch.column(name).to_pylist() for name in ROW_IDENTITY_COLUMNS + ["baseline_top50"]]
        for group_id, item_id, label, local, seen, baseline in zip(*columns):
            if not isinstance(group_id, int) or not 0 <= group_id < len(self.groups):
                raise ValueError("Candidate row references an invalid group")
            if group_id < self.previous_group:
                raise ValueError("Candidate groups are not contiguous and ordered")
            if label not in (0, 1) or not isinstance(item_id, str):
                raise ValueError("Candidate identity has an invalid label/item ID")
            if not label and (local or seen):
                raise ValueError("Unobserved candidate has a positive slice flag")
            self.previous_group = group_id
            count = self.counts[group_id]
            count[0] += 1
            count[1] += int(label)
            count[2] += int(bool(local))
            count[3] += int(bool(seen))
            count[4] += int(label and baseline > 0)
            self.rows += 1
            self.digest.update(_canonical_bytes([group_id, item_id, int(label), bool(local), bool(seen)]))

    def finish(self):
        if self.rows != self.metadata.get("rows"):
            raise ValueError("Candidate row count differs from source manifest")
        for group, count in zip(self.groups, self.counts):
            if count[0] != group["pool_size"] or count[1] != group["positives_in_pool"]:
                raise ValueError(f"Pool size/positive count mismatch for group {group['group_id']}")
            if count[2] > group["local_relevant_count"] or count[3] > group["seen_relevant_count"]:
                raise ValueError("Pool positives exceed slice denominators")
            if count[4] != group["baseline_hits"]:
                raise ValueError("Baseline hit count differs from source manifest")
        return self.digest.hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stamp_source_digest(path, source, digest):
    if "candidate_pool_digest" in source and source["candidate_pool_digest"] != digest:
        raise ValueError("Existing source candidate_pool_digest differs from the streamed rows")
    if "candidate_pool_digest" not in source:
        _atomic_json(path, {**source, "candidate_pool_digest": digest})


def derive_validation_cache(source_cache_dir, train_data, reference_model, data_dir, output_dir,
                            *, batch_size=32768, items=None, fingerprint_function=None):
    """Derive only validation; audit producers build policy-matched fit-side rows later."""
    import joblib
    import pyarrow as pa
    import pyarrow.parquet as pq
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    source_dir, output_dir = Path(source_cache_dir).resolve(), Path(output_dir).resolve()
    if source_dir == output_dir:
        raise ValueError("Derived validation cache needs a separate output directory")
    source_path, source_json = source_dir / "validation.parquet", source_dir / "validation.json"
    train_data, reference_model = Path(train_data), Path(reference_model)
    if not source_path.is_file() or not source_json.is_file():
        raise ValueError("Source validation cache is incomplete")
    source = _read_json(source_json)
    if source.get("role") != "validation" or source.get("cache_version") != 2:
        raise ValueError("Source must be a completed V2 validation cache")
    train_json = train_data.with_suffix(train_data.suffix + ".json")
    training = _read_json(train_json)
    artifact = joblib.load(reference_model)
    features = validate_bridge_metadata(source, training, artifact)
    target_metadata = artifact["metadata"]
    train_schema = pq.ParquetFile(train_data).schema_arrow
    if not set(features + ["label", "weight"]) <= set(train_schema.names):
        raise ValueError("Training Parquet lacks the declared model features/labels/weights")
    if any(train_schema.field(name).type != pa.float32() for name in features):
        raise ValueError("Training features must be float32")
    provenance = artifact.get("provenance", {})
    for key, path in (("training_data_sha256", train_data), ("training_metadata_sha256", train_json)):
        if provenance.get(key) is not None and provenance[key] != _file_sha256(path):
            raise ValueError(f"Reference model {key} does not match the requested training input")
    corpus = verify_corpus(source, target_metadata, data_dir, items=items,
                           fingerprint_function=fingerprint_function)
    reference_hash = _file_sha256(reference_model)
    source_hash = _file_sha256(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path, target_json = output_dir / "validation.parquet", output_dir / "validation.json"
    if target_path.exists() or target_json.exists():
        if not (target_path.is_file() and target_json.is_file()):
            raise ValueError("Incomplete derived validation already exists")
        existing = _read_json(target_json)
        if (existing.get("model_sha256") != reference_hash or existing.get("feature_names") != features
                or existing.get("retrieval_config") != target_metadata["retrieval_config"]
                or existing.get("derivation", {}).get("source_validation_parquet_sha256") != source_hash
                or existing.get("derivation", {}).get("dense_score_alignment", {}).get("protocol") != ALIGNMENT_PROTOCOL
                or existing.get("validation_parquet_sha256") != _file_sha256(target_path)):
            raise ValueError("Existing derived cache differs from this source/model")
        _stamp_source_digest(source_json, source, existing["candidate_pool_digest"])
        return existing
    source_file = pq.ParquetFile(source_path)
    required = set(source["feature_names"] + ROW_IDENTITY_COLUMNS + ["full_dense_score"])
    if not required <= set(source_file.schema_arrow.names):
        raise ValueError("Source Parquet lacks declared features/identity/full dense scores")
    if any(source_file.schema_arrow.field(name).type != pa.float32()
           for name in source["feature_names"] + ["full_dense_score"]):
        raise ValueError("Source features and full dense score must be float32")
    schema = source_file.schema_arrow
    if "full_dense_known" not in schema.names:
        schema = schema.append(pa.field("full_dense_known", pa.float32()))
    source_digest, target_digest = CandidatePoolDigest(source), CandidatePoolDigest(source)
    alignment = {"protocol": ALIGNMENT_PROTOCOL, "minimum_required_dense_pairs": MIN_DENSE_PAIRS,
                 "allowed_max_abs_cosine_residual_vs_hnsw": MAX_ALIGNMENT_RESIDUAL,
                 "query_groups_aligned": 0, "query_norm_min": None, "query_norm_max": None,
                 "dense_pairs_min": None, "dense_pairs_max": None,
                 "max_abs_cosine_residual_vs_hnsw": 0.0,
                 "empty_groups": sum(group["pool_size"] == 0 for group in source["groups"])}
    partial = target_path.with_suffix(".partial.parquet")
    if partial.exists():
        raise ValueError(f"Incomplete prior derivation exists: {partial}")
    original_stat = source_path.stat()
    try:
        with pq.ParquetWriter(partial, schema, compression="zstd") as writer:
            for group_table in iter_complete_query_groups(source_file.iter_batches(batch_size=batch_size)):
                table, stats = align_full_dense_group(group_table)
                source_digest.update(group_table)
                target_digest.update(table)
                writer.write_table(table)
                alignment["query_groups_aligned"] += 1
                for field, value in (("query_norm", stats["query_norm"]),
                                     ("dense_pairs", stats["dense_pairs"])):
                    for suffix, operation in (("min", min), ("max", max)):
                        key = f"{field}_{suffix}"
                        old = alignment[key]
                        alignment[key] = value if old is None else operation(old, value)
                alignment["max_abs_cosine_residual_vs_hnsw"] = max(
                    alignment["max_abs_cosine_residual_vs_hnsw"], stats["max_abs_cosine_residual_vs_hnsw"])
        before, after = source_digest.finish(), target_digest.finish()
        if before != after:
            raise ValueError("Derived candidate identity/order/denominators differ from source")
        if "candidate_pool_digest" in source and source["candidate_pool_digest"] != before:
            raise ValueError("Source candidate_pool_digest does not match its rows")
        final_stat = source_path.stat()
        if (original_stat.st_size, original_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
            raise ValueError("Source validation changed while deriving rows")
        metadata = {
            **source, "feature_names": features, "feature_schema_version": 6,
            "retrieval_config": target_metadata["retrieval_config"],
            "corpus_fingerprint": target_metadata["corpus_fingerprint"],
            "model_reference": str(reference_model.resolve()), "model_sha256": reference_hash,
            "reference_artifact_sha256": reference_hash,
            **artifact_prior_policy(artifact),
            "source_full_dense_score_max_abs_difference_from_topk": source.get(
                "full_dense_score_max_abs_difference_from_topk"),
            "full_dense_score_max_abs_difference_from_topk": alignment[
                "max_abs_cosine_residual_vs_hnsw"],
            "training_prior_policy": target_metadata.get("training_prior_policy"),
            "candidate_pool_digest": before,
            "historical_holdout_reused": True,
            "derivation": {
                "version": 1, "source_cache_dir": str(source_dir),
                "source_validation_parquet_sha256": source_hash,
                "source_validation_metadata_sha256_before_digest": _file_sha256(source_json),
                "source_model_sha256": source["model_sha256"],
                "verified_inventory": corpus,
                "original_features_preserved": source["feature_names"],
                "candidate_rows_and_order_preserved": True,
                "dense_scores_reencoded": False, "fit_side_cache_copied": False,
                "dense_score_alignment": alignment,
            },
            "limitations": [*source.get("limitations", []),
                "V6 validation reuses the original frozen candidate pools and cosine scores.",
                "Cached query-vector norm is recovered from dense-channel HNSW cosines; only the extra full dense feature is normalized.",
                "Only the optional rescore_dense_union retrieval setting differs.",
                "Fit-side geography policy requires separately generated training-side rows."],
        }
        partial.replace(target_path)
        metadata["validation_parquet_sha256"] = _file_sha256(target_path)
        _atomic_json(target_json, metadata)
        _stamp_source_digest(source_json, source, before)
        return metadata
    finally:
        if partial.exists():
            partial.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache-dir", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32768)
    args = parser.parse_args(argv)
    result = derive_validation_cache(args.source_cache_dir, args.train_data, args.reference_model,
                                     args.data_dir, args.output_dir, batch_size=args.batch_size)
    print(json.dumps({key: result[key] for key in
                      ("rows", "pool_recall", "baseline_recall_at_50", "candidate_pool_digest")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
