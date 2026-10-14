"""Category correction changes two feature columns, never label sampling."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fix_reranker_category_sentinel import (
    CORRECTED_FEATURES, contract_from_core, correct_training_cache, sha256_file, sidecars,
)


class CategorySentinelCorrectionTests(unittest.TestCase):
    def fixture(self, root, categories=("0", "", "service"), labels=(1, 0, 1, 0, 1, 0)):
        source = root / "train.parquet"
        contract = contract_from_core()
        names = tuple(contract["feature_names"])
        keys = [json.dumps([f"query-{index}", "location", "0", "", category],
                           ensure_ascii=False, separators=(",", ":"))
                for index, category in enumerate(categories)]
        ids = [hashlib.sha256(key.encode()).hexdigest() for key in keys]
        columns = {name: pa.array(np.arange(6, dtype=np.float32) + index / 100)
                   for index, name in enumerate(names)}
        columns["category_match"] = pa.array([1, 0, 0, 0, 1, 0], type=pa.float32())
        columns["category_known"] = pa.array([1, 1, 0, 0, 1, 1], type=pa.float32())
        columns.update({"label": pa.array(labels, type=pa.int8()),
                        "weight": pa.array([1.0] * 6, type=pa.float32()),
                        "group_id": pa.array([value for value in ids for _ in range(2)]),
                        "query_text": pa.array([f"query-{index}" for index in range(3) for _ in range(2)]),
                        "query_key_json": pa.array([value for value in keys for _ in range(2)]),
                        "item_id": pa.array([f"{index:016x}" for index in range(6)]),
                        "relevant_count": pa.array([1] * 6, type=pa.int32())})
        table = pa.table(columns).replace_schema_metadata({b"fixture": b"preserve schema metadata"})
        pq.write_table(table, source, row_group_size=2)
        metadata = {"feature_version": 4, "feature_names": list(names),
                    "training_data_schema_version": 2,
                    "group_columns": [name for name, _dtype in contract["group_columns"]],
                    "retrieval_config": {"method": "hybrid", "candidate_k": 449, "location_bonus": .02},
                    "corpus_fingerprint": "same-corpus", "split": "queries", "seed": 42,
                    "validation_fraction": .2,
                    "training_prior_policy": {"microcategory": "leave-query-text-out",
                                              "geography": "leave-query-text-out"},
                    "stats": {"sampled_queries": 3, "queries_with_positive_in_pool": 3,
                              "positive_rows": 3, "unobserved_rows": 3},
                    "embedding_checkpoint_sha256": "encoder-sha", "unrelated_metadata": {"keep": True}}
        metadata_path, priors_path = sidecars(source)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        priors_path.write_bytes(b"opaque exact prior bytes\0\1\2")
        return source, table, metadata

    def test_only_zero_category_flags_change_and_sources_and_priors_are_unchanged(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, original, metadata = self.fixture(root)
            inputs = (source, *sidecars(source))
            original_sha = [sha256_file(path) for path in inputs]
            target = root / "derived" / "train-corrected.parquet"
            derived = correct_training_cache(source, target, batch_size=1)
            corrected = pq.read_table(target)
            unchanged = [name for name in original.column_names if name not in CORRECTED_FEATURES]
            self.assertTrue(corrected.select(unchanged).equals(original.select(unchanged)))
            self.assertEqual(corrected.schema, original.schema)
            self.assertEqual(corrected["category_match"].to_pylist(), [0, 0, 0, 0, 1, 0])
            self.assertEqual(corrected["category_known"].to_pylist(), [0, 0, 0, 0, 1, 1])
            self.assertEqual([sha256_file(path) for path in inputs], original_sha)
            self.assertEqual(sidecars(target)[1].read_bytes(), sidecars(source)[1].read_bytes())
            for name, value in metadata.items():
                self.assertEqual(derived[name], value)
            marker = derived["category_sentinel_correction"]
            self.assertEqual(marker["affected_rows"], 2)
            self.assertEqual(marker["affected_groups"], 1)
            self.assertEqual(marker["changed_values"], {"category_match": 1, "category_known": 2})
            self.assertEqual(marker["derived_parquet_sha256"], sha256_file(target))
            for name, expected in zip(("source_parquet_sha256", "source_metadata_sha256", "source_priors_sha256"),
                                      original_sha):
                self.assertEqual(marker[name], expected)
            self.assertFalse(marker["selection_uses_labels"])
            self.assertTrue(marker["priors_unchanged"])

    def test_selection_does_not_follow_clicked_or_unlabelled_labels(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, original, _metadata = self.fixture(root, labels=(0, 1, 0, 1, 0, 1))
            target = root / "corrected.parquet"
            correct_training_cache(source, target, batch_size=3)
            result = pq.read_table(target)
            self.assertTrue(result["label"].equals(original["label"]))
            self.assertEqual(result["category_known"].to_pylist()[:2], [0.0, 0.0])

    def test_no_zero_query_groups_is_a_documented_noop_in_values(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, original, _metadata = self.fixture(root, categories=("known", "", "service"))
            target = root / "corrected.parquet"
            derived = correct_training_cache(source, target, batch_size=2)
            self.assertTrue(pq.read_table(target).equals(original))
            self.assertEqual(derived["category_sentinel_correction"]["affected_rows"], 0)
            self.assertEqual(derived["category_sentinel_correction"]["changed_values"],
                             {"category_match": 0, "category_known": 0})

    def test_output_exists_and_in_place_correction_are_refused(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, _original, _metadata = self.fixture(root)
            before = source.read_bytes()
            with self.assertRaisesRegex(ValueError, "never the source"):
                correct_training_cache(source, source)
            target = root / "corrected.parquet"
            target.write_bytes(b"keep existing output")
            with self.assertRaisesRegex(ValueError, "already exist"):
                correct_training_cache(source, target)
            self.assertEqual(target.read_bytes(), b"keep existing output")
            self.assertEqual(source.read_bytes(), before)

    def test_wrong_schema_and_noncanonical_query_metadata_are_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, original, metadata = self.fixture(root)
            altered = deepcopy(metadata)
            altered["feature_version"] = 5
            sidecars(source)[0].write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "active grouped V4"):
                correct_training_cache(source, root / "bad-schema.parquet")
            sidecars(source)[0].write_text(json.dumps(metadata), encoding="utf-8")
            malformed = original.set_column(original.schema.get_field_index("query_key_json"),
                                            original.schema.field("query_key_json"),
                                            pa.array(["[]"] * 6))
            pq.write_table(malformed, source)
            with self.assertRaisesRegex(ValueError, "canonical"):
                correct_training_cache(source, root / "bad-key.parquet")
            self.assertFalse((root / "bad-key.parquet").exists())


if __name__ == "__main__":
    unittest.main()
