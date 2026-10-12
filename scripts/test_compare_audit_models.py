"""Evidence-selection and cluster invariants, without parquet or model imports."""

import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "compare_audit", Path(__file__).with_name("compare_audit_models.py"))
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def row(group_id, text, partition="tune", relevant=2, stage25=1, stage50=1):
    return {"group_id": group_id, "query_text": text, "query_key": [text, "location", "0", "", "category"],
            "partition": partition, "relevant_count": relevant, "pool_size": 100,
            "positives_in_pool": relevant, "baseline_hits": 1,
            "s25_all_hits": stage25, "s50_all_hits": stage50}


class TextClusterTests(unittest.TestCase):
    def test_resampled_macro_weights_query_groups_not_cluster_means(self):
        rows = [row(0, "А"), row(1, " а "), row(2, "а"), row(3, "Б")]
        clusters = COMPARE.query_text_clusters(rows)
        self.assertEqual(clusters, [[0, 1, 2], [3]])
        values = [0.0, 0.0, 0.0, 1.0]
        self.assertEqual(COMPARE.cluster_resample_group_mean(values, clusters, [0, 1]), 0.25)
        self.assertEqual(COMPARE.cluster_resample_group_mean(values, clusters, [0, 0]), 0.0)
        self.assertEqual(COMPARE.cluster_resample_group_mean(values, clusters, [1, 1]), 1.0)

    def test_normalization_matches_project(self):
        self.assertEqual(COMPARE.normalize_text("  СЧЁТЧИК "), "счетчик")


class QueryIdentityTests(unittest.TestCase):
    def test_rows_align_by_group_and_query_keys_are_checked(self):
        reference = [row(0, "телефон"), row(1, "часы")]
        candidate = [dict(reference[1]), dict(reference[0])]
        aligned = COMPARE.align_query_rows(reference, candidate)
        self.assertEqual([value["group_id"] for value in aligned], [0, 1])
        candidate[1] = {**candidate[1], "query_key": ["телефон", "different-location", "0", "", "category"]}
        with self.assertRaisesRegex(ValueError, "query_key differs"):
            COMPARE.align_query_rows(reference, candidate)

    def test_denominator_change_is_rejected(self):
        reference = [row(0, "телефон")]
        candidate = [{**reference[0], "relevant_count": 3}]
        with self.assertRaisesRegex(ValueError, "relevant_count differs"):
            COMPARE.align_query_rows(reference, candidate)

    def test_duplicate_ids_and_texts_across_partitions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate group_id"):
            COMPARE.align_query_rows([row(0, "a"), row(1, "b")], [row(0, "a"), row(0, "a")])
        rows = [row(0, "a"), row(1, "A", partition="confirmation")]
        with self.assertRaisesRegex(ValueError, "crosses tune"):
            COMPARE.align_query_rows(rows, rows)


class DenseRescoreCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.reference = {"retrieval_config": {"candidate_k": 449, "channel_quota": 0},
                          "candidate_pool_digest": "same-frozen-pool-digest",
                          "inference_prior_policy": {"inference_priors_sha256": "same-prior-hash",
                                                     "disable_microcategory_prior": False}}
        self.candidate = {**self.reference,
                          "retrieval_config": {**self.reference["retrieval_config"], "rescore_dense_union": True}}

    def test_safe_only_flag_difference_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "strict comparison"):
            COMPARE.verify_comparison_compatibility(self.reference, self.candidate)
        result = COMPARE.verify_comparison_compatibility(self.reference, self.candidate, allow_rescore_only=True)
        self.assertEqual(result["mode"], "dense_rescore_flag_only_with_immutable_pool")

    def test_missing_or_different_pool_digest_is_rejected(self):
        for changed in ({key: value for key, value in self.candidate.items() if key != "candidate_pool_digest"},
                        {**self.candidate, "candidate_pool_digest": "different"}):
            with self.assertRaisesRegex(ValueError, "candidate_pool_digest"):
                COMPARE.verify_comparison_compatibility(self.reference, changed, allow_rescore_only=True)

    def test_changed_quota_is_rejected_even_with_matching_hashes(self):
        changed = {**self.candidate, "retrieval_config": {**self.candidate["retrieval_config"], "channel_quota": 10}}
        with self.assertRaisesRegex(ValueError, "Only rescore_dense_union"):
            COMPARE.verify_comparison_compatibility(self.reference, changed, allow_rescore_only=True)

    def test_missing_changed_prior_or_disable_flag_is_rejected(self):
        cases = [{key: value for key, value in self.candidate.items() if key != "inference_prior_policy"},
                 {**self.candidate, "inference_prior_policy": {"inference_priors_sha256": "changed", "disable_microcategory_prior": False}},
                 {**self.candidate, "inference_prior_policy": {"inference_priors_sha256": "same-prior-hash", "disable_microcategory_prior": True}}]
        for changed in cases:
            with self.assertRaisesRegex(ValueError, "prior policy"):
                COMPARE.verify_comparison_compatibility(self.reference, changed, allow_rescore_only=True)


class TuningSelectionTests(unittest.TestCase):
    def test_stage_selection_does_not_peek_at_confirmation(self):
        rows = [row(0, "tune", stage25=2, stage50=1),
                row(1, "confirmation", partition="confirmation", stage25=0, stage50=2)]
        self.assertEqual(COMPARE.choose_stage_from_tune(rows, (25, 50)), (25, 1.0))
        rows[1]["s50_all_hits"] = 0
        self.assertEqual(COMPARE.choose_stage_from_tune(rows, (25, 50)), (25, 1.0))

    def test_ties_choose_fewer_trees_then_model_name(self):
        rows = [row(0, "tune", stage25=1, stage50=1)]
        self.assertEqual(COMPARE.choose_stage_from_tune(rows, (50, 25)), (25, 0.5))
        models = {"z": {"stage": 25, "tune_recall": 0.8},
                  "a": {"stage": 50, "tune_recall": 0.8}}
        self.assertEqual(COMPARE.choose_model_from_tune(models), "z")
        models["a"]["stage"] = 25
        self.assertEqual(COMPARE.choose_model_from_tune(models), "a")

    def test_ci_crossing_zero_cannot_establish_positive_evidence(self):
        self.assertEqual(COMPARE.ci_evidence(-0.01, 0.02), "uncertain_interval_includes_zero")
        self.assertEqual(COMPARE.ci_evidence(0.001, 0.02), "positive_paired_evidence")

    def test_reference_requires_available_histgb_prefix(self):
        models = {"hgb": {"report": {"model_class": "HistGradientBoostingClassifier"},
                          "stages": (25, 50, 100, 150)}}
        COMPARE.validate_reference_stage(models, "hgb", 50)
        with self.assertRaisesRegex(ValueError, "stage 250"):
            COMPARE.validate_reference_stage(models, "hgb", 250)
        models["hgb"]["report"]["model_class"] = "LGBMRanker"
        with self.assertRaisesRegex(ValueError, "HistGradientBoosting"):
            COMPARE.validate_reference_stage(models, "hgb", 50)

    def test_reference_scores_use_selected_prefix_not_hardcoded_final(self):
        scores = [row(0, "a", stage50=2), row(1, "b", partition="confirmation", stage50=1)]
        for value in scores:
            value["s150_all_hits"] = 0
        self.assertEqual(COMPARE.reference_stage_scores(scores, 50), [1.0, 0.5])
        self.assertEqual(COMPARE.reference_stage_scores(scores, 150), [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "lacks stage 100"):
            COMPARE.reference_stage_scores(scores, 100)


if __name__ == "__main__":
    unittest.main()
