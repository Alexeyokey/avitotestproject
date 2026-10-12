"""Validate recovery of saved query groups without importing ML libraries."""

import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "reranker_comparison", Path(__file__).with_name("train_reranker_comparison.py"))
COMPARISON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARISON)


class GroupReconstructionTests(unittest.TestCase):
    def test_multiple_positives_and_groups_keep_original_row_runs(self):
        labels = [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0]
        weights = [0.5, 0.5, 0.5, 0.5, 1, 0.5, 0.5,
                   1 / 3, 1 / 3, 1 / 3, 0.5, 0.5]
        sizes = COMPARISON.reconstruct_groups(
            labels, weights, expected_groups=3, expected_negatives=2,
            expected_positive_rows=6, expected_unobserved_rows=6)
        self.assertEqual(sizes, [4, 3, 5])
        self.assertEqual(sum(sizes), len(labels))

    def test_nonbinary_labels_and_invalid_weights_are_rejected(self):
        for labels in ([1, 2], [1, -1], [1, "0"], [1, 0.5]):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, "binary"):
                COMPARISON.reconstruct_groups(labels, [1, 1], expected_groups=1,
                                              expected_negatives=1)
        for weight in (0, -1, float("nan"), float("inf")):
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, "finite and positive"):
                COMPARISON.reconstruct_groups([1, 0], [1, weight],
                                              expected_groups=1, expected_negatives=1)

    def test_corrupted_order_and_missing_classes_are_rejected(self):
        cases = [([0, 1], [1, 1]), ([1, 1], [0.5, 0.5]),
                 ([1, 0, 1, 0, 0], [1, 0.5, 1, 0.5, 0.5])]
        for labels, weights in cases:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                COMPARISON.reconstruct_groups(labels, weights,
                                              expected_groups=1, expected_negatives=2)

    def test_class_weights_must_sum_to_one_and_be_uniform(self):
        with self.assertRaisesRegex(ValueError, "weight sum"):
            COMPARISON.reconstruct_groups([1, 0, 0], [1, 0.25, 0.25],
                                          expected_groups=1, expected_negatives=2)
        with self.assertRaisesRegex(ValueError, "uniform"):
            COMPARISON.reconstruct_groups([1, 1, 0, 0], [0.2, 0.8, 0.5, 0.5],
                                          expected_groups=1, expected_negatives=2)

    def test_group_count_and_saved_class_totals_must_match(self):
        common = dict(labels=[1, 0, 0], weights=[1, 0.5, 0.5], expected_negatives=2)
        with self.assertRaisesRegex(ValueError, "Recovered"):
            COMPARISON.reconstruct_groups(**common, expected_groups=2)
        with self.assertRaisesRegex(ValueError, "Positive row count"):
            COMPARISON.reconstruct_groups(**common, expected_groups=1, expected_positive_rows=2)
        with self.assertRaisesRegex(ValueError, "Unobserved row count"):
            COMPARISON.reconstruct_groups(**common, expected_groups=1, expected_unobserved_rows=3)

    def test_empty_or_unequal_columns_are_rejected(self):
        for labels, weights in (([], []), ([1, 0], [1])):
            with self.assertRaisesRegex(ValueError, "equal length"):
                COMPARISON.reconstruct_groups(labels, weights, expected_groups=1)


class MetadataTests(unittest.TestCase):
    def metadata(self):
        return {"feature_version": 4, "feature_names": [f"feature{i}" for i in range(75)],
                "split": "queries", "seed": 42, "validation_fraction": 0.2,
                "retrieval_config": {"method": "hybrid"}, "corpus_fingerprint": "sha",
                "stats": {"queries_with_positive_in_pool": 2,
                          "positive_rows": 3, "unobserved_rows": 128}}

    def test_original_outer_split_and_feature_count_are_required(self):
        metadata = self.metadata()
        names, stats = COMPARISON.validate_metadata(metadata, expected_groups=2)
        self.assertEqual(len(names), 75)
        self.assertEqual(stats["positive_rows"], 3)
        for key, value in (("split", "all"), ("seed", 43), ("feature_version", 5),
                           ("feature_names", ["query_id"] * 75)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                COMPARISON.validate_metadata({**metadata, key: value}, expected_groups=2)

    def test_metadata_group_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "group count"):
            COMPARISON.validate_metadata(self.metadata(), expected_groups=3)


if __name__ == "__main__":
    unittest.main()
