"""Small Parquet fixtures for a matched V6/V4 training ablation."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from project_reranker_feature_schema import project_training_data, validate_metadata


BASE_FEATURES = tuple(f"feature_{i}" for i in range(75))
GROUP_COLUMNS = (("group_id", pa.string()), ("query_text", pa.string()),
                 ("query_key_json", pa.string()), ("item_id", pa.string()),
                 ("relevant_count", pa.int32()))


def fixture_fingerprint(items, *, feature_version):
    # Deliberately distinct across versions: the script must use both APIs.
    return f"v{feature_version}:" + hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


def contract():
    return {"schemas": {4: BASE_FEATURES, 6: (*BASE_FEATURES, "full_dense_score", "full_dense_known")},
            "group_columns": GROUP_COLUMNS, "fingerprint": fixture_fingerprint,
            "normalize": lambda text: text.strip().lower(), "query_key_length": 5}


class FeatureProjectionTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        source, target = root / "source.parquet", root / "v4" / "train.parquet"
        items = [{"item_id": f"{index:016x}", "item_title_raw": "ремонт"} for index in range(6)]
        features = list(contract()["schemas"][6])
        metadata = {
            "feature_version": 6, "feature_names": features,
            "training_data_schema_version": 2, "group_columns": [name for name, _dtype in GROUP_COLUMNS],
            "retrieval_config": {"method": "hybrid", "candidate_k": 449,
                                 "geo_candidate_k": 100, "location_bonus": 0.02,
                                 "rescore_dense_union": True, "embedding_model": "/unchanged/model"},
            "corpus_fingerprint": fixture_fingerprint(items, feature_version=6),
            "split": "queries", "seed": 42, "validation_fraction": 0.2,
            "training_prior_policy": {"microcategory": "leave-query-text-out", "geography": "leave-query-text-out"},
            "embedding_checkpoint_sha256": "verified-checkpoint",
            "embedding_base_revision": "verified-base", "extra_metadata": {"keep": ["exact", "unchanged"]},
            "stats": {"sampled_queries": 3, "queries_with_positive_in_pool": 2,
                      "positive_rows": 3, "unobserved_rows": 3},
        }
        columns = {name: pa.array(np.arange(6, dtype=np.float32) + index / 100)
                   for index, name in enumerate(BASE_FEATURES)}
        columns["full_dense_score"] = pa.array(np.array([0.9, 0.2, -0.1, 0.0, 0.6, 0.5], dtype=np.float32))
        columns["full_dense_known"] = pa.array(np.ones(6, dtype=np.float32))
        keys = [json.dumps([text, "city", "", "", ""], ensure_ascii=False, separators=(",", ":"))
                for text in ("ремонт", "доставка")]
        group_ids = [hashlib.sha256(key.encode()).hexdigest() for key in keys]
        columns.update({
            "label": pa.array([1, 0, 0, 1, 1, 0], type=pa.int8()),
            "weight": pa.array([1, 0.5, 0.5, 0.5, 0.5, 1], type=pa.float32()),
            "group_id": pa.array([group_ids[0]] * 3 + [group_ids[1]] * 3),
            "query_text": pa.array(["ремонт"] * 3 + ["доставка"] * 3),
            "query_key_json": pa.array([keys[0]] * 3 + [keys[1]] * 3),
            "item_id": pa.array([item["item_id"] for item in items]),
            "relevant_count": pa.array([2] * 3 + [3] * 3, type=pa.int32()),
        })
        table = pa.table(columns)
        pq.write_table(table, source, row_group_size=2)
        source.with_suffix(".parquet.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        source.with_suffix(".parquet.priors.joblib").write_bytes(b"opaque priors are copied without deserialization\0")
        return source, target, metadata, table, items

    def project(self, source, target, items, **kwargs):
        return project_training_data(source, target, "unused", contract=contract(), items=items, **kwargs)

    def write_table(self, source, table):
        pq.write_table(table, source, row_group_size=2)

    def test_only_two_features_removed_values_order_groups_and_priors_preserved(self):
        with TemporaryDirectory() as directory:
            source, target, original, table, items = self.fixture(directory)
            source_bytes = {suffix: source.with_suffix(suffix).read_bytes()
                            for suffix in (".parquet", ".parquet.json", ".parquet.priors.joblib")}
            result = self.project(source, target, items, batch_size=2)
            projected = pq.read_table(target)
            kept = [name for name in table.column_names if name not in ("full_dense_score", "full_dense_known")]
            self.assertTrue(projected.equals(table.select(kept)))
            self.assertEqual(projected.schema, table.select(kept).schema)
            self.assertEqual(result["feature_names"], list(BASE_FEATURES))
            self.assertEqual(result["feature_version"], 4)
            self.assertNotIn("rescore_dense_union", result["retrieval_config"])
            self.assertEqual(result["corpus_fingerprint"], fixture_fingerprint(items, feature_version=4))
            for key in original.keys() - {"feature_version", "feature_names", "retrieval_config", "corpus_fingerprint"}:
                self.assertEqual(result[key], original[key])
            self.assertEqual(result["retrieval_config"],
                             {key: value for key, value in original["retrieval_config"].items() if key != "rescore_dense_union"})
            self.assertEqual(target.with_suffix(".parquet.priors.joblib").read_bytes(), source_bytes[".parquet.priors.joblib"])
            for suffix, before in source_bytes.items():
                self.assertEqual(source.with_suffix(suffix).read_bytes(), before)
            self.assertEqual(result["feature_projection"]["counts"],
                             {"groups": 2, "positive_rows": 3, "unobserved_rows": 3, "rows": 6})
            self.assertTrue(result["feature_projection"]["rows_unchanged"])
            self.assertEqual(json.loads(target.with_suffix(".parquet.json").read_text(encoding="utf-8")), result)

    def test_projection_is_batch_invariant_including_group_spanning_boundary(self):
        with TemporaryDirectory() as directory:
            source, target, _meta, _table, items = self.fixture(directory)
            first = self.project(source, target, items, batch_size=1)
            other = target.with_name("other.parquet")
            second = self.project(source, other, items, batch_size=5)
            self.assertTrue(pq.read_table(target).equals(pq.read_table(other)))
            self.assertEqual(first["feature_projection"]["counts"], second["feature_projection"]["counts"])

    def test_existing_data_or_sidecar_is_never_overwritten(self):
        for suffix in (".parquet", ".parquet.json", ".parquet.priors.joblib"):
            with self.subTest(suffix=suffix), TemporaryDirectory() as directory:
                source, target, _meta, _table, items = self.fixture(directory)
                target.parent.mkdir()
                occupied = target.with_suffix(suffix)
                occupied.write_bytes(b"preserve me")
                with self.assertRaises(FileExistsError):
                    self.project(source, target, items)
                self.assertEqual(occupied.read_bytes(), b"preserve me")
                self.assertEqual(list(target.parent.iterdir()), [occupied])

    def test_source_path_cannot_be_target(self):
        with TemporaryDirectory() as directory:
            source, _target, _meta, _table, items = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, "separate target"):
                self.project(source, source, items)

    def test_missing_priors_or_wrong_corpus_rejected_before_output(self):
        with TemporaryDirectory() as directory:
            source, target, _meta, _table, items = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, "actual benchmark"):
                self.project(source, target, [*items, {"item_id": "extra"}])
            self.assertFalse(target.parent.exists())
            source.with_suffix(".parquet.priors.joblib").unlink()
            with self.assertRaisesRegex(ValueError, "Completed source"):
                self.project(source, target, items)

    def test_wrong_schema_order_version_and_rescore_flag_are_rejected(self):
        with TemporaryDirectory() as directory:
            source, target, metadata, table, items = self.fixture(directory)
            for patch in ({"feature_version": 4}, {"feature_names": list(reversed(metadata["feature_names"]))},
                          {"retrieval_config": {"method": "hybrid", "rescore_dense_union": False}}):
                with self.subTest(patch=patch), self.assertRaises(ValueError):
                    validate_metadata({**metadata, **patch}, contract())
            self.write_table(source, table.select(list(reversed(table.column_names))))
            with self.assertRaisesRegex(ValueError, "column order"):
                self.project(source, target, items)

    def test_invalid_dense_values_or_numeric_features_leave_no_partial_files(self):
        for column, values in (("full_dense_known", [0] * 6), ("full_dense_score", [1.5] * 6),
                               ("feature_0", [float("nan")] * 6)):
            with self.subTest(column=column), TemporaryDirectory() as directory:
                source, target, _meta, table, items = self.fixture(directory)
                bad = table.set_column(table.schema.get_field_index(column), column,
                                       pa.array(values, type=pa.float32()))
                self.write_table(source, bad)
                with self.assertRaises(ValueError):
                    self.project(source, target, items, batch_size=2)
                self.assertFalse(target.exists())
                self.assertEqual(list(target.parent.iterdir()), [])

    def test_group_query_identity_and_duplicate_items_are_checked(self):
        for column, values in (("group_id", ["incorrect"] * 6),
                               ("query_text", ["incorrect"] * 6),
                               ("item_id", ["duplicate"] * 6)):
            with self.subTest(column=column), TemporaryDirectory() as directory:
                source, target, _meta, table, items = self.fixture(directory)
                self.write_table(source, table.set_column(table.schema.get_field_index(column), column, pa.array(values)))
                with self.assertRaises(ValueError):
                    self.project(source, target, items)
                self.assertFalse(target.exists())

    def test_noncontiguous_groups_and_wrong_group_totals_are_rejected(self):
        with TemporaryDirectory() as directory:
            source, target, metadata, table, items = self.fixture(directory)
            # Repeat a complete valid first group after the valid second one.
            duplicated = pa.concat_tables([table, table.slice(0, 3)])
            metadata["stats"].update(positive_rows=4, unobserved_rows=5, queries_with_positive_in_pool=3)
            source.with_suffix(".parquet.json").write_text(json.dumps(metadata), encoding="utf-8")
            self.write_table(source, duplicated)
            with self.assertRaisesRegex(ValueError, "contiguous"):
                self.project(source, target, items, batch_size=2)
            self.assertFalse(target.exists())
        with TemporaryDirectory() as directory:
            source, target, metadata, _table, items = self.fixture(directory)
            metadata["stats"]["queries_with_positive_in_pool"] = 3
            source.with_suffix(".parquet.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Group/label counts"):
                self.project(source, target, items)

    def test_invalid_labels_weights_and_manifest_row_count_are_rejected(self):
        for column, values, dtype in (("label", [2] * 6, pa.int8()),
                                      ("weight", [0] * 6, pa.float32()),
                                      ("weight", [1] * 6, pa.float32())):
            with self.subTest(column=column, values=values), TemporaryDirectory() as directory:
                source, target, _meta, table, items = self.fixture(directory)
                self.write_table(source, table.set_column(table.schema.get_field_index(column), column, pa.array(values, type=dtype)))
                with self.assertRaises(ValueError):
                    self.project(source, target, items)
                self.assertFalse(target.exists())
        with TemporaryDirectory() as directory:
            source, target, metadata, _table, items = self.fixture(directory)
            metadata["stats"]["unobserved_rows"] = 4
            source.with_suffix(".parquet.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "row count"):
                self.project(source, target, items)

    def test_unsupported_active_contract_and_invalid_batch_sizes_are_rejected(self):
        with TemporaryDirectory() as directory:
            source, target, metadata, _table, items = self.fixture(directory)
            broken = deepcopy(contract())
            broken["schemas"][4] = tuple(reversed(BASE_FEATURES))
            with self.assertRaisesRegex(ValueError, "Active core"):
                validate_metadata(metadata, broken)
            for batch_size in (0, -1, True, 1.5):
                with self.subTest(batch_size=batch_size), self.assertRaisesRegex(ValueError, "Batch size"):
                    self.project(source, target, items, batch_size=batch_size)


if __name__ == "__main__":
    unittest.main()
