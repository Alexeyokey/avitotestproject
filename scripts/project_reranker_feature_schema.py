"""Project grouped V6 training data to V4 for a matched feature ablation.

This removes only the two full-pool dense features. It never retrieves items,
encodes text, changes sampled rows, or rebuilds an embedding index. Both models
therefore receive the same query groups, selected/unobserved labels and weights.
The source remains unchanged; output data, JSON and priors must be new paths.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


DROPPED_FEATURES = ("full_dense_score", "full_dense_known")
ITEM_COLUMNS = (
    "item_id", "item_title_raw", "item_infm_params_text", "item_description_raw",
    "item_category_id", "item_location_id", "item_microcat_id", "item_price",
    "item_rating", "item_rating_reviews_count", "item_is_phone_hidden",
    "item_is_message_forbidden",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _core_contract():
    # Resolve the active code, rather than embedding another feature-name list.
    from avito_candidates.core import SEARCH_FIELDS, normalize
    from avito_candidates.reranker import (
        FEATURE_SCHEMAS, GROUP_COLUMNS, corpus_fingerprint,
    )
    return {
        "schemas": FEATURE_SCHEMAS, "group_columns": GROUP_COLUMNS,
        "fingerprint": corpus_fingerprint, "normalize": normalize,
        "query_key_length": len(SEARCH_FIELDS),
    }


def validate_metadata(metadata, contract):
    schemas = contract["schemas"]
    v4, v6 = tuple(schemas.get(4, ())), tuple(schemas.get(6, ()))
    if len(v4) != 75 or v6 != (*v4, *DROPPED_FEATURES) or len(set(v6)) != 77:
        raise ValueError("Active core must define V6 as V4's 75 features plus the two dense features")
    if metadata.get("feature_version") != 6 or tuple(metadata.get("feature_names", ())) != v6:
        raise ValueError("Source metadata does not match the active V6 feature schema")
    config = metadata.get("retrieval_config", {})
    if config.get("method") != "hybrid" or config.get("rescore_dense_union") is not True:
        raise ValueError("Source must be hybrid with measured dense-union rescoring")
    groups = [name for name, _dtype in contract["group_columns"]]
    if metadata.get("training_data_schema_version") != 2 or metadata.get("group_columns") != groups:
        raise ValueError("Projection requires explicit grouped training schema version 2")
    stats = metadata.get("stats", {})
    for name in ("queries_with_positive_in_pool", "positive_rows", "unobserved_rows"):
        value = stats.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"Missing or invalid source preparation statistic: {name}")
    if not isinstance(metadata.get("corpus_fingerprint"), str):
        raise ValueError("Source corpus fingerprint is missing")
    return v4, v6


def validate_schema(schema, metadata, contract):
    import pyarrow as pa

    v4, v6 = validate_metadata(metadata, contract)
    expected = [*(pa.field(name, pa.float32()) for name in v6),
                pa.field("label", pa.int8()), pa.field("weight", pa.float32()),
                *(pa.field(name, dtype) for name, dtype in contract["group_columns"])]
    if schema.names != [field.name for field in expected] or any(
            schema.field(field.name).type != field.type for field in expected):
        raise ValueError("Source Parquet must exactly match the V6 grouped column order and types")
    # Selecting arrays preserves values/types/column metadata without conversion.
    return pa.schema([field for field in schema if field.name not in DROPPED_FEATURES],
                     metadata=schema.metadata), v4


def verify_corpus(metadata, data_dir, contract, *, items=None):
    if items is None:
        from avito_candidates.cli import read
        items = read(Path(data_dir) / "benchmark_items.parquet", ITEM_COLUMNS)
    fingerprint = contract["fingerprint"]
    actual_v6 = fingerprint(items, feature_version=6)
    if metadata["corpus_fingerprint"] != actual_v6:
        raise ValueError("Source corpus fingerprint differs from the actual benchmark inventory")
    actual_v4 = fingerprint(items, feature_version=4)
    return actual_v4, {"source_v6": actual_v6, "target_v4": actual_v4, "items": len(items)}


class TrainingGroupValidator:
    """Check group boundaries across batches using only the current small group.

    The checks mirror the current core's validate_training_groups contract;
    training still invokes that authoritative function before LambdaRank fitting.
    Keeping at most one sampled group avoids loading the full Parquet in memory.
    """

    def __init__(self, metadata, contract):
        self.metadata, self.contract = metadata, contract
        self.previous, self.seen, self.rows = None, set(), []
        self.groups = self.positives = self.unobserved = 0

    def _finish_group(self):
        import numpy as np

        if not self.rows:
            return
        first = self.rows[0]
        text, key_json, relevant = first["query_text"], first["query_key_json"], first["relevant_count"]
        if (not isinstance(text, str) or not isinstance(key_json, str)
                or not isinstance(relevant, int) or relevant < 1
                or any((row["query_text"], row["query_key_json"], row["relevant_count"])
                       != (text, key_json, relevant) for row in self.rows)):
            raise ValueError("Ranking query metadata must be consistent within each group")
        try:
            key = json.loads(key_json)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid query-key JSON in ranking data") from error
        if (not isinstance(key, list) or len(key) != self.contract["query_key_length"]
                or any(not isinstance(value, str) for value in key)
                or self.contract["normalize"](key[0]) != text
                or hashlib.sha256(key_json.encode("utf-8")).hexdigest() != self.previous):
            raise ValueError("Ranking group ID, query key and normalized text differ")
        ids = [row["item_id"] for row in self.rows]
        if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("Ranking groups must contain distinct item IDs")
        n_positive = sum(row["label"] for row in self.rows)
        if not 0 < n_positive < len(self.rows) or n_positive > relevant:
            raise ValueError("Each ranking group requires selected and unobserved rows")
        for label in (0, 1):
            weights = np.array([row["weight"] for row in self.rows if row["label"] == label], dtype=np.float64)
            if (not np.isclose(weights.sum(), 1.0, atol=1e-5, rtol=0)
                    or not np.allclose(weights, 1 / len(weights), atol=1e-7, rtol=1e-5)):
                raise ValueError("Ranking source weights do not match per-query class totals")
        self.groups += 1
        self.positives += n_positive
        self.unobserved += len(self.rows) - n_positive
        self.rows.clear()

    def update(self, batch):
        columns = [name for name, _dtype in self.contract["group_columns"]] + ["label", "weight"]
        for row in batch.select(columns).to_pylist():
            group_id = row["group_id"]
            if not isinstance(group_id, str) or not group_id:
                raise ValueError("Every ranking row requires a group ID")
            if group_id != self.previous:
                self._finish_group()
                if group_id in self.seen:
                    raise ValueError("Ranking group IDs must form contiguous query runs")
                self.seen.add(group_id)
                self.previous = group_id
            self.rows.append(row)

    def finish(self):
        self._finish_group()
        stats = self.metadata["stats"]
        if (self.groups != stats["queries_with_positive_in_pool"]
                or self.positives != stats["positive_rows"]
                or self.unobserved != stats["unobserved_rows"]):
            raise ValueError("Group/label counts differ from preparation metadata")
        return {"groups": self.groups, "positive_rows": self.positives,
                "unobserved_rows": self.unobserved, "rows": self.positives + self.unobserved}


def validate_batch(batch, feature_names):
    import numpy as np

    for name in feature_names:
        values = batch.column(name).to_numpy(zero_copy_only=False)
        if not np.isfinite(values).all():
            raise ValueError(f"Source training feature must be finite: {name}")
    known = batch.column("full_dense_known").to_numpy(zero_copy_only=False)
    scores = batch.column("full_dense_score").to_numpy(zero_copy_only=False)
    if np.any(known != 1.0) or np.any(np.abs(scores) > 1.0001):
        raise ValueError("Every V6 row requires a measured valid dense cosine")
    labels = batch.column("label").to_numpy(zero_copy_only=False)
    weights = batch.column("weight").to_numpy(zero_copy_only=False)
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Source labels must be binary and weights finite and positive")


def _temporary_path(directory, suffix):
    fd, name = tempfile.mkstemp(prefix=".reranker-projection-", suffix=suffix, dir=directory)
    os.close(fd)
    return Path(name)


def project_training_data(source_path, target_path, data_dir, *, batch_size=32768,
                          contract=None, items=None):
    import pyarrow.parquet as pq

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("Batch size must be positive")
    source, target = Path(source_path).resolve(), Path(target_path).resolve()
    if source == target:
        raise ValueError("Projection requires a separate target path")
    source_json = source.with_suffix(source.suffix + ".json")
    source_priors = source.with_suffix(source.suffix + ".priors.joblib")
    target_json = target.with_suffix(target.suffix + ".json")
    target_priors = target.with_suffix(target.suffix + ".priors.joblib")
    source_files, target_files = [source, source_json, source_priors], [target, target_json, target_priors]
    if not all(path.is_file() for path in source_files):
        raise ValueError("Completed source data, metadata and inference priors are required")
    if any(path.exists() for path in target_files):
        raise FileExistsError("Target data, metadata or priors already exist")
    contract = _core_contract() if contract is None else contract
    source_hashes = {str(path): _sha256(path) for path in source_files}
    metadata = json.loads(source_json.read_text(encoding="utf-8"))
    parquet = pq.ParquetFile(source)
    output_schema, v4_features = validate_schema(parquet.schema_arrow, metadata, contract)
    v4_fingerprint, corpus_verification = verify_corpus(metadata, data_dir, contract, items=items)
    expected_rows = metadata["stats"]["positive_rows"] + metadata["stats"]["unobserved_rows"]
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError("Parquet row count differs from source preparation metadata")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporaries, committed = [], []
    try:
        temporary_data = _temporary_path(target.parent, ".parquet")
        temporaries.append(temporary_data)
        groups = TrainingGroupValidator(metadata, contract)
        with pq.ParquetWriter(temporary_data, output_schema, compression="zstd") as writer:
            for batch in parquet.iter_batches(batch_size=batch_size):
                validate_batch(batch, metadata["feature_names"])
                groups.update(batch)
                # No casting, sorting, resampling or recomputation occurs here.
                writer.write_batch(batch.select(output_schema.names))
        counts = groups.finish()
        if counts["rows"] != expected_rows:
            raise ValueError("Streamed row count differs from the source")
        for path in source_files:
            if _sha256(path) != source_hashes[str(path)]:
                raise ValueError("Source changed while projection was running")
        result = deepcopy(metadata)
        result["feature_version"] = 4
        result["feature_names"] = list(v4_features)
        result["corpus_fingerprint"] = v4_fingerprint
        result["retrieval_config"].pop("rescore_dense_union", None)
        result["feature_projection"] = {
            "version": 1, "source_feature_version": 6, "target_feature_version": 4,
            "dropped_features": list(DROPPED_FEATURES), "rows_unchanged": True,
            "source_data": str(source), "source_data_sha256": source_hashes[str(source)],
            "source_metadata_sha256": source_hashes[str(source_json)],
            "source_priors_sha256": source_hashes[str(source_priors)],
            "projected_data_sha256": _sha256(temporary_data),
            "corpus_verification": corpus_verification, "counts": counts,
        }
        temporary_priors = _temporary_path(target.parent, ".priors.joblib")
        temporaries.append(temporary_priors)
        shutil.copyfile(source_priors, temporary_priors)
        if _sha256(temporary_priors) != source_hashes[str(source_priors)]:
            raise ValueError("Copied inference priors differ from the source bytes")
        temporary_json = _temporary_path(target.parent, ".json")
        temporaries.append(temporary_json)
        temporary_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        # Hard links create exclusive final names on the same filesystem. JSON
        # is the completion marker and is installed last; existing files are
        # never replaced, including if another writer appears after the check.
        for temporary, destination in ((temporary_data, target), (temporary_priors, target_priors),
                                       (temporary_json, target_json)):
            os.link(temporary, destination)
            committed.append(destination)
        return result
    except BaseException:
        for path in committed:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporaries:
            path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--data-dir", default=Path("dataset"), type=Path)
    parser.add_argument("--batch-size", default=32768, type=int)
    args = parser.parse_args()
    result = project_training_data(args.source, args.target, args.data_dir, batch_size=args.batch_size)
    print(json.dumps({"target": str(args.target), "feature_version": result["feature_version"],
                      "feature_projection": result["feature_projection"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
