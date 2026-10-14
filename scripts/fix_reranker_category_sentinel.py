"""Derive a new V4 training cache with unspecified-category flags corrected.

Selection uses query metadata only: search_category == "0" means unspecified.
Only category_match/category_known are set to zero for those rows. Labels,
weights, other features, row order and serialized priors are preserved. No
retrieval, embedding or model fitting is performed; source files are untouched.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


CORRECTED_FEATURES = ("category_match", "category_known")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_from_core():
    from avito_candidates.core import normalize
    from avito_candidates.reranker import GROUP_COLUMNS, V4_FEATURES
    return {"feature_names": V4_FEATURES, "group_columns": GROUP_COLUMNS,
            "normalize": normalize}


def sidecars(path):
    path = Path(path)
    return path.with_suffix(path.suffix + ".json"), path.with_suffix(path.suffix + ".priors.joblib")


def correct_training_cache(source, target, *, batch_size=65536, contract=None):
    source, target = Path(source), Path(target)
    if batch_size <= 0:
        raise ValueError("Correction batch size must be positive")
    if source.resolve() == target.resolve():
        raise ValueError("Correction requires a new target, never the source")
    metadata_path, priors_path = sidecars(source)
    target_metadata, target_priors = sidecars(target)
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary_metadata = target_metadata.with_suffix(target_metadata.suffix + ".partial")
    temporary_priors = target_priors.with_suffix(target_priors.suffix + ".partial")
    outputs = (target, target_metadata, target_priors, temporary, temporary_metadata, temporary_priors)
    if any(path.exists() for path in outputs):
        raise ValueError("Correction target or partial sidecars already exist")
    if any(not path.is_file() for path in (source, metadata_path, priors_path)):
        raise ValueError("Complete source Parquet, metadata and priors are required")
    original_sha = {"parquet": sha256_file(source), "metadata": sha256_file(metadata_path),
                    "priors": sha256_file(priors_path)}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    contract = contract or contract_from_core()
    names = tuple(contract["feature_names"])
    if (metadata.get("feature_version") != 4 or tuple(metadata.get("feature_names", ())) != names
            or metadata.get("training_data_schema_version") != 2
            or metadata.get("group_columns") != [name for name, _dtype in contract["group_columns"]]
            or not set(CORRECTED_FEATURES) <= set(names)):
        raise ValueError("Correction requires the active grouped V4 feature schema")
    if "category_sentinel_correction" in metadata:
        raise ValueError("Source is already annotated with a category correction")
    parquet = pq.ParquetFile(source)
    expected = [*(pa.field(name, pa.float32()) for name in names),
                pa.field("label", pa.int8()), pa.field("weight", pa.float32()),
                *(pa.field(name, dtype) for name, dtype in contract["group_columns"])]
    if (parquet.schema_arrow.names != [field.name for field in expected]
            or any(parquet.schema_arrow.field(field.name).type != field.type for field in expected)):
        raise ValueError("Source columns, order and types differ from grouped V4")
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = affected_rows = 0
    affected_groups = set()
    changed_values = dict.fromkeys(CORRECTED_FEATURES, 0)
    with pq.ParquetWriter(temporary, parquet.schema_arrow, compression="zstd") as writer:
        for batch in parquet.iter_batches(batch_size=batch_size):
            keys = batch.column(parquet.schema_arrow.get_field_index("query_key_json")).to_pylist()
            groups = batch.column(parquet.schema_arrow.get_field_index("group_id")).to_pylist()
            texts = batch.column(parquet.schema_arrow.get_field_index("query_text")).to_pylist()
            mask, blank = [], []
            for serialized, group, text in zip(keys, groups, texts):
                key = json.loads(serialized)
                if (not isinstance(key, list) or len(key) != 5
                        or any(not isinstance(value, str) or value != contract["normalize"](value)
                               for value in key)
                        or json.dumps(key, ensure_ascii=False, separators=(",", ":")) != serialized
                        or hashlib.sha256(serialized.encode("utf-8")).hexdigest() != group
                        or key[0] != text):
                    raise ValueError("Source query metadata is not canonical or does not match its identity")
                selected = key[4] == "0"
                mask.append(selected)
                blank.append(key[4] == "")
                if selected:
                    affected_groups.add(group)
            selected_array = pa.array(mask, type=pa.bool_())
            selected_numpy = np.asarray(mask, dtype=np.bool_)
            blank_numpy = np.asarray(blank, dtype=np.bool_)
            for name in CORRECTED_FEATURES:
                index = parquet.schema_arrow.get_field_index(name)
                values = batch.column(index)
                numeric = values.to_numpy(zero_copy_only=False)
                if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
                    raise ValueError("Category flags must be finite binary features")
                if np.any(numeric[blank_numpy] != 0):
                    raise ValueError("Existing unspecified empty-category flags must already be zero")
                changed_values[name] += int(np.count_nonzero(numeric[selected_numpy]))
                corrected = pc.if_else(selected_array, pa.scalar(0.0, type=pa.float32()), values)
                batch = batch.set_column(index, batch.schema.field(index), corrected)
            writer.write_batch(batch)
            rows += batch.num_rows
            affected_rows += int(selected_numpy.sum())
    if rows != parquet.metadata.num_rows:
        raise ValueError("Correction changed the source row count")
    if any(sha256_file(path) != original_sha[name] for name, path in (
            ("parquet", source), ("metadata", metadata_path), ("priors", priors_path))):
        raise ValueError("Source files changed during correction")
    shutil.copyfile(priors_path, temporary_priors)
    if sha256_file(temporary_priors) != original_sha["priors"]:
        raise ValueError("Derived priors are not byte-for-byte identical")
    derived = deepcopy(metadata)
    derived["category_sentinel_correction"] = {
        "version": 1, "unknown_search_category_values": ["", "0"],
        "corrected_features": list(CORRECTED_FEATURES),
        "feature_definition": "category_match/category_known are zero for unspecified search category 0",
        "source_parquet_sha256": original_sha["parquet"],
        "source_metadata_sha256": original_sha["metadata"],
        "source_priors_sha256": original_sha["priors"],
        "source_parquet": str(source.resolve()),
        "producer_sha256": sha256_file(Path(__file__)),
        "derived_parquet_sha256": sha256_file(temporary),
        "rows": rows, "affected_rows": affected_rows,
        "affected_groups": len(affected_groups), "changed_values": changed_values,
        "selection_uses_labels": False, "other_columns_preserved": True,
        "priors_unchanged": True,
    }
    temporary_metadata.write_text(json.dumps(derived, ensure_ascii=False, indent=2,
                                             allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    temporary_priors.replace(target_priors)
    temporary_metadata.replace(target_metadata)  # Completion metadata is written last.
    return derived


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=65536)
    args = parser.parse_args()
    metadata = correct_training_cache(args.source, args.target, batch_size=args.batch_size)
    print(json.dumps(metadata["category_sentinel_correction"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
