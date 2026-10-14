"""Small retrieval-audit invariants; no model libraries or parquet are needed."""

import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "reranker_learning_curve_audit", Path(__file__).with_name("audit_reranker_learning_curve.py"))
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def group(pool_size, positives, relevant, baseline_hits=0, text="query", partition="tune", group_id=0):
    return {"group_id": group_id, "query_text": text, "partition": partition,
            "pool_size": pool_size, "positives_in_pool": positives,
            "relevant_count": relevant, "local_relevant_count": relevant,
            "nonlocal_relevant_count": 0, "baseline_hits": baseline_hits}


class RecallScoringTests(unittest.TestCase):
    def test_positive_outside_pool_stays_in_denominator(self):
        metadata = group(pool_size=2, positives=1, relevant=2, baseline_hits=1)
        result = AUDIT.score_group(metadata, [1, 0], [True, True], None,
                                   [True, True], [0.02, 0.01], {25: [0.9, 0.1]})
        self.assertEqual(result["s25_recall"], 0.5)
        metrics = AUDIT.summarize([result], 25)["slices"]["all"]
        self.assertEqual(metrics["reranker_macro_recall"], 0.5)
        self.assertEqual(metrics["pool_macro_recall"], 0.5)

    def test_top50_is_selected_from_entire_pool(self):
        labels = [0] * 100
        labels[80] = 1
        scores = [0.0] * 100
        scores[80] = 1.0
        metadata = group(pool_size=100, positives=1, relevant=1)
        result = AUDIT.score_group(metadata, labels, [True] * 100, None,
                                   [index < 50 for index in range(100)],
                                   [float(100 - index) for index in range(100)],
                                   {150: scores})
        self.assertEqual(result["baseline_all_hits"], 0)
        self.assertEqual(result["s150_all_hits"], 1)
        self.assertEqual(result["s150_recall"], 1.0)

    def test_rank_ties_match_production_rrf_then_original_order(self):
        self.assertEqual(AUDIT.top_indices([0.5, 0.5, 0.5], [0.01, 0.03, 0.03]), [1, 2, 0])

    def test_incomplete_pool_and_wrong_baseline_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete pool"):
            AUDIT.score_group(group(3, 1, 1), [1, 0], [True, True], None,
                              [False, False], [0.1, 0.0], {25: [0.5, 0.0]})
        with self.assertRaisesRegex(ValueError, "Baseline reproduction failed"):
            AUDIT.score_group(group(2, 1, 1, baseline_hits=1), [1, 0], [True, True], None,
                              [False, True], [0.1, 0.0], {25: [0.5, 0.0]})

    def test_metadata_only_empty_first_middle_last_groups_keep_zero_recall(self):
        groups = [group(0, 0, 2, group_id=0),
                  group(1, 1, 1, baseline_hits=1, group_id=1),
                  group(0, 0, 2, group_id=2),
                  group(1, 1, 1, baseline_hits=1, group_id=3),
                  group(0, 0, 2, group_id=4)]
        AUDIT.check_group_metadata(groups, validation=True)
        records = AUDIT.score_missing_zero_groups(groups, 0, 1, (25,), has_seen=False)
        records.append(AUDIT.score_group(groups[1], [1], [True], None, [True], [0.1], {25: [0.9]}))
        records.extend(AUDIT.score_missing_zero_groups(groups, 2, 3, (25,), has_seen=False))
        records.append(AUDIT.score_group(groups[3], [1], [True], None, [True], [0.1], {25: [0.9]}))
        records.extend(AUDIT.score_missing_zero_groups(groups, 4, 5, (25,), has_seen=False))
        self.assertEqual([record["group_id"] for record in records], [0, 1, 2, 3, 4])
        self.assertEqual([record["s25_recall"] for record in records], [0.0, 1.0, 0.0, 1.0, 0.0])
        self.assertEqual(AUDIT.summarize(records, 25)["slices"]["all"]["reranker_macro_recall"], 0.4)

    def test_missing_nonempty_pool_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "omitted a nonempty pool"):
            AUDIT.score_missing_zero_groups([group(1, 1, 1)], 0, 1, (25,), has_seen=False)

    def test_fit_side_report_separates_reranker_training_eligible_groups(self):
        eligible = AUDIT.score_group(group(1, 1, 1, baseline_hits=1), [1], [True], None,
                                     [True], [0.1], {25: [0.9]})
        skipped = AUDIT.score_group(group(0, 0, 1, group_id=1), [], [], None,
                                    [], [], {25: []})
        report = AUDIT.stage_report([eligible, skipped], (25,), {}, fit_side_sample=True)["25"]
        self.assertEqual(report["all_groups"]["slices"]["all"]["reranker_macro_recall"], 0.5)
        self.assertEqual(report["reranker_training_eligible_groups"]["groups"], 1)
        self.assertEqual(report["reranker_training_eligible_groups"]["slices"]["all"]["reranker_macro_recall"], 1.0)
        self.assertEqual(report["groups_without_retrieved_training_positive"]["groups"], 1)


class PriorPolicyTests(unittest.TestCase):
    def test_semantic_hash_and_disabled_feature_flag_must_match(self):
        from reranker_audit_provenance import artifact_prior_policy
        artifact = {"priors": {}, "metadata": {}}
        policy = artifact_prior_policy(artifact)
        self.assertEqual(AUDIT.ensure_prior_policy(artifact, policy, "test"), policy)
        with self.assertRaisesRegex(ValueError, "inference_priors_sha256 differs"):
            AUDIT.ensure_prior_policy(artifact, {**policy, "inference_priors_sha256": "changed"}, "test")
        with self.assertRaisesRegex(ValueError, "disable_microcategory_prior differs"):
            AUDIT.ensure_prior_policy({**artifact, "disable_microcategory_prior": True}, policy, "test")

    def test_missing_policy_and_wrong_fit_policy_fail_closed(self):
        from reranker_audit_provenance import artifact_prior_policy
        artifact = {"priors": {}, "metadata": {"training_prior_policy": {"geography": "leave-query-text-out"}}}
        policy = artifact_prior_policy(artifact)
        with self.assertRaisesRegex(ValueError, "lacks required prior policy"):
            AUDIT.ensure_prior_policy(artifact, {}, "test")
        with self.assertRaisesRegex(ValueError, "training_prior_policy differs"):
            AUDIT.ensure_prior_policy(artifact, policy, "test", fit_side=True)
        AUDIT.ensure_prior_policy(artifact, {**policy, "training_prior_policy": artifact["metadata"]["training_prior_policy"]},
                                 "test", fit_side=True)


class ValidationOnlyModeTests(unittest.TestCase):
    def test_default_requires_both_roles_and_opt_in_selects_validation_only(self):
        cache = Path("cache")
        self.assertEqual(set(AUDIT.audit_cache_paths(cache)), {"validation", "train_sample"})
        self.assertEqual(AUDIT.audit_cache_paths(cache, validation_only=True),
                         {"validation": cache / "validation.parquet"})

    def test_missing_comparable_fit_sample_is_explicit_and_training_loss_is_retained(self):
        report = {"datasets": {"validation": {"25": {"all_groups": {"slices": {"all": {"reranker_macro_recall": 0.8}}}}}},
                  "sampled_training_row_loss": {"applicable": True, "stages": {"25": {"weighted_log_loss": 0.1}}}}
        AUDIT.add_fit_side_gap_report(report, (25,))
        self.assertFalse(report["fit_side_sample_available"])
        self.assertEqual(report["train_minus_validation_recall_gap"], {})
        self.assertEqual(report["eligible_fit_groups_minus_validation_recall_gap"], {})
        self.assertIn("no comparable fit-side", report["fit_side_sample_note"])
        self.assertTrue(report["sampled_training_row_loss"]["applicable"])

    def test_compatible_fit_sample_still_produces_all_and_eligible_gaps(self):
        def metrics(value):
            return {"slices": {"all": {"reranker_macro_recall": value}}}
        report = {"datasets": {"validation": {"25": {"all_groups": metrics(0.8)}},
                               "train_sample": {"25": {"all_groups": metrics(0.9),
                                                           "reranker_training_eligible_groups": metrics(0.95)}}}}
        AUDIT.add_fit_side_gap_report(report, (25,))
        self.assertTrue(report["fit_side_sample_available"])
        self.assertAlmostEqual(report["train_minus_validation_recall_gap"]["25"], 0.1)
        self.assertAlmostEqual(report["eligible_fit_groups_minus_validation_recall_gap"]["25"], 0.15)


class QueryPartitionTests(unittest.TestCase):
    def test_same_normalized_text_has_same_hash_partition(self):
        self.assertEqual(AUDIT.query_text_partition("  Телефон   NOKIA "),
                         AUDIT.query_text_partition("телефон nokia"))
        self.assertEqual(AUDIT.query_text_partition("Счётчик"),
                         AUDIT.query_text_partition("счетчик"))

    def test_text_cannot_cross_tune_and_confirmation(self):
        groups = [group(2, 1, 1, text="Телефон", group_id=0),
                  group(2, 1, 1, text=" телефон ", partition="confirmation", group_id=1)]
        with self.assertRaisesRegex(ValueError, "both partitions"):
            AUDIT.check_group_metadata(groups, validation=True)

    def test_multiple_groups_for_one_text_are_allowed_in_same_partition(self):
        groups = [group(2, 1, 1, text="Телефон", group_id=0),
                  group(2, 1, 1, text="телефон", group_id=1)]
        AUDIT.check_group_metadata(groups, validation=True)


class ModelPrefixTests(unittest.TestCase):
    def test_unavailable_stages_are_not_mislabeled_and_final_is_included(self):
        self.assertEqual(AUDIT.model_stages(95, [25, 50, 100, 150]), (25, 50, 95))
        with self.assertRaises(ValueError):
            AUDIT.model_stages(150, [0, 25])

    def test_histgb_uses_actual_staged_predictions(self):
        class Model:
            consumed = 0
            def staged_predict_proba(self, matrix):
                for iteration in range(1, 6):
                    self.consumed += 1
                    yield [[0.0, iteration / 10]]
        model = Model()
        values = list(AUDIT.prefix_probabilities(model, [[1]], [2, 4]))
        self.assertEqual(values, [(2, [[0.0, 0.2]]), (4, [[0.0, 0.4]])])
        self.assertEqual(model.consumed, 4)

    def test_lightgbm_receives_requested_tree_count(self):
        class Model:
            booster_ = object()
            def predict_proba(self, matrix, *, num_iteration):
                return [[0.0, num_iteration / 100]]
        values = list(AUDIT.prefix_probabilities(Model(), [[1]], [25, 50]))
        self.assertEqual(values, [(25, [[0.0, 0.25]]), (50, [[0.0, 0.5]])])

    def test_lambdarank_keeps_one_dimensional_scores(self):
        class Model:
            booster_ = object()
            def predict(self, matrix, *, num_iteration):
                return [float(num_iteration), -float(num_iteration)]
            def predict_proba(self, *args, **kwargs):
                self.fail("A ranker must not be asked for probabilities")
        values = list(AUDIT.prefix_probabilities(Model(), [[1], [2]], [25, 50], is_ranker=True))
        self.assertEqual(values, [(25, [25.0, -25.0]), (50, [50.0, -50.0])])
        self.assertTrue(all(len(scores) == 2 and isinstance(scores[0], float)
                            for _, scores in values))
        artifact = {"model": Model(), "estimator": "lambdarank"}
        self.assertTrue(AUDIT.is_ranking_artifact(artifact))
        self.assertFalse(AUDIT.sampled_train_losses(Path("not-needed.parquet"), artifact,
                                                   (25, 50), 8)["applicable"])

    def test_real_lightgbm_booster_path_avoids_wrapper_feature_name_warnings(self):
        class Booster:
            def predict(self, matrix, *, num_iteration):
                return [float(num_iteration), -float(num_iteration)]
        class Model:
            booster_ = Booster()
            def predict(self, *args, **kwargs):
                raise AssertionError("Native booster prediction should bypass sklearn wrapper")
        outputs = list(AUDIT.prefix_probabilities(Model(), [[1], [2]], [25], is_ranker=True))
        self.assertEqual(outputs, [(25, [25.0, -25.0])])


if __name__ == "__main__":
    unittest.main()
