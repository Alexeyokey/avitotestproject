"""Focused tests for fixed-pool retraining comparison and prefix publication."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from avito_candidates.reranker import prediction_scores
from evaluate_geo_dense_retrained import (GEO_CONFIG, STAGES, assert_same_pool,
    choose_conservative_stage, metric_row, save_selected_artifact, top50_hits, validate_new_config,
    TuneFeatureCache)
import evaluate_fresh_validation as EVAL


def curve(values):
    return {str(stage): {"conditional_available": {"reranker": {"text_macro_recall_at_50": value}}}
            for stage, value in zip(STAGES, values)}


class GeoRetrainedTest(unittest.TestCase):
    def row(self):
        return {"group_identity": "x", "query_key": ["a", "", "", "", ""], "query_text": "a",
                "representative": True, "available_relevant_count": 2, "full_relevant_count": 3,
                "pool_size": 998, "pool_hits": 2, "baseline_hits": 1, "dense_hits": 2}

    def test_selection_prefers_smallest_within_point_two_percentage_point(self):
        self.assertEqual(choose_conservative_stage(curve([.8981, .899, .900, .8995])), 50)
        self.assertEqual(choose_conservative_stage(curve([.8979, .8980, .900, .8995])), 75)
        self.assertEqual(choose_conservative_stage(curve([.897, .8979, .900, .8995])), 100)

    def test_selection_requires_all_declared_finite_stages(self):
        incomplete = curve([.8, .81, .82, .83])
        del incomplete["50"]
        with self.assertRaisesRegex(ValueError, "every predeclared"):
            choose_conservative_stage(incomplete)
        with self.assertRaisesRegex(ValueError, "finite"):
            choose_conservative_stage(curve([.8, .81, np.nan, .83]))

    def test_only_declared_new_retrieval_keys_allowed(self):
        old = {"method": "hybrid", "location_bonus": .02}
        new = {**old, **GEO_CONFIG}
        validate_new_config(new, old)
        for key, value in (("geo_dense_weight", .02), ("location_bonus", .05), ("unknown", 1)):
            changed = {**new, key: value}
            with self.assertRaises(ValueError):
                validate_new_config(changed, old)

    def test_fixed_pool_rejects_any_count_or_context_change(self):
        original = self.row()
        assert_same_pool(original, original)
        for key, value in (("pool_size", 997), ("pool_hits", 1), ("baseline_hits", 0),
                           ("dense_hits", 1), ("available_relevant_count", 3), ("query_text", "b")):
            with self.assertRaises(ValueError):
                assert_same_pool({**original, key: value}, original)

    def test_invalid_overbudget_and_relevance_counts_rejected_even_if_both_match(self):
        for key, value in (("pool_size", 999), ("pool_hits", 3), ("baseline_hits", 3), ("dense_hits", 3)):
            row = {**self.row(), key: value}
            with self.assertRaisesRegex(ValueError, "Invalid fixed-pool"):
                assert_same_pool(row, row)

    def test_no_available_targets_keep_pool_and_full_denominator(self):
        group = {**self.row(), "available_relevant_count": 0}
        result = (["a"], {"candidate_pool": ["a", "b"], "channels": {"dense": ["b"]}})
        row = metric_row(group, result, set())
        self.assertEqual(row["full_relevant_count"], 3)
        self.assertEqual(row["pool_size"], 2)
        self.assertEqual([row[key] for key in ("pool_hits", "baseline_hits", "dense_hits")], [0, 0, 0])

    def test_reranking_matches_probability_rrf_pool_order_ties(self):
        pool = [str(i) for i in range(52)]
        scores = np.zeros(52)
        matrix = np.zeros((52, 2))
        matrix[51, 0] = 2
        scores[50] = 1
        # 50 and51 enter; the two latest zero-score ties (48,49) are displaced.
        self.assertEqual(top50_hits(scores, matrix, pool, {"50", "51", "48", "49"}), 2)
        with self.assertRaisesRegex(ValueError, "complete pool"):
            top50_hits(scores[:-1], matrix, pool, {"50"})

    def test_selected_artifact_retains_fitted_trees_and_runtime_prefix(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            rng = np.random.RandomState(11)
            matrix = rng.randn(120, 3).astype(np.float32)
            labels = (matrix[:, 0] + .2 * matrix[:, 1] > 0).astype(int)
            model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=3,
                early_stopping=False, random_state=42).fit(matrix, labels)
            artifact = {"model": model, "metadata": {"retrieval_config": dict(GEO_CONFIG)},
                        "training_configuration": {"fitted_iterations": 150}, "provenance": {"example": "fit-only"}}
            source = root / "full.joblib"
            joblib.dump(artifact, source)
            # Evaluation always starts from the saved training artifact. Some
            # NumPy structured tree dtypes normalize on their first load.
            artifact = joblib.load(source)
            report = root / "report.json"
            report.write_text(json.dumps({"full_tune_complete": True, "all_per_context_pool_metrics_reproduced": True,
                "new_capacity_curve": curve([.88, .89, .90, .899]), "selection_rule": "declared"}))
            declaration = root / "declaration.json"
            declaration.write_text("{}")
            output = save_selected_artifact(artifact, source, 100, report, declaration, root)
            selected = joblib.load(output["path"])
            self.assertEqual(selected["model"].n_iter_, 150)
            self.assertEqual(selected["selected_iteration"], 100)
            self.assertEqual(selected["training_configuration"]["selected_iteration"], 100)
            self.assertEqual(selected["metadata"], artifact["metadata"])
            np.testing.assert_array_equal(prediction_scores(selected["model"], matrix, selected_iteration=100),
                                          prediction_scores(model, matrix, selected_iteration=100))
            with self.assertRaises(FileExistsError):
                save_selected_artifact(artifact, source, 100, report, declaration, root)
            report.write_text(json.dumps({"full_tune_complete": False, "all_per_context_pool_metrics_reproduced": True,
                "new_capacity_curve": curve([.88, .89, .90, .899])}))
            with self.assertRaisesRegex(ValueError, "complete verified Tune"):
                save_selected_artifact(artifact, source, 100, report, declaration, root)

    def cache_record(self, key, pool, relevant, matrix, index):
        row = {"group_identity": key, "pool_size": len(pool), "available_relevant_count": len(relevant),
               "full_relevant_count": max(len(relevant), 1), "pool_hits": len(set(pool) & relevant),
               "ordered_pool_sha256": EVAL.canonical_sha256(pool)}
        return (row, pool, relevant, matrix, index)

    def cache_report(self, directory, records, *, corrupt=False):
        scores = directory / "group-scores.jsonl"
        with scores.open("w", encoding="utf-8") as stream:
            for record in records:
                row = dict(record[0])
                if corrupt and record[4] == 0:
                    row["pool_hits"] = 0
                stream.write(json.dumps({"new": row}) + "\n")
        report = directory / "report.json"
        report.write_text(json.dumps({"complete": True, "phase": "tune", "groups": len(records),
            "full_tune_complete": False, "group_scores_sha256": EVAL.sha256_file(scores)}))
        return report, scores

    def test_feature_cache_preserves_float32_pool_order_labels_and_zero_context(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            one = np.asarray([[.1, 2], [.2, 3], [.3, 4]], dtype=np.float32)
            two = np.asarray([[.4, 5], [.5, 6]], dtype=np.float32)
            records = [self.cache_record("q1", ["a", "b", "c"], {"b", "outside"}, one, 0),
                       self.cache_record("q0", ["a", "d"], set(), None, 1),
                       self.cache_record("q2", ["d", "a"], {"d"}, two, 2)]
            cache = TuneFeatureCache(root, feature_names=["x", "y"], provenance={"fixture": "fixed"})
            with cache:
                cache.write_batch(records[:2])
                cache.write_batch(records[2:])
            self.assertTrue(cache.marker.exists())
            self.assertFalse(cache.sidecar.exists())
            report, scores = self.cache_report(root, records)
            cache.finalize(report_path=report, score_path=scores)
            table = pq.read_table(cache.path)
            self.assertEqual(table.schema.field("x").type, pa.float32())
            self.assertEqual(table.column("group_identity").to_pylist(), ["q1", "q1", "q1", "q2", "q2"])
            self.assertEqual(table.column("item_id").to_pylist(), ["a", "b", "c", "d", "a"])
            self.assertEqual(table.column("available_positive").to_pylist(), [False, True, False, True, False])
            np.testing.assert_array_equal(np.column_stack([table[name].to_numpy() for name in ("x", "y")]),
                                          np.concatenate([one, two]))
            self.assertNotIn("label", table.schema.names)
            meta = json.loads(cache.sidecar.read_text())
            self.assertEqual((meta["rows"], meta["positive_rows"], meta["groups"], meta["exported_groups"]), (5, 2, 3, 2))
            self.assertEqual(meta["zero_availability_groups_skipped"], 1)
            self.assertEqual(meta["role"], "tune-only-never-training")
            self.assertFalse(meta["training_allowed"])
            self.assertEqual(meta["data"]["sha256"], EVAL.sha256_file(cache.path))
            self.assertEqual(meta["all_group_metadata_and_denominators"]["sha256"], EVAL.sha256_file(scores))
            self.assertFalse(cache.marker.exists())

    def test_feature_cache_cannot_complete_with_different_scoring_counts(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            records = [self.cache_record("q1", ["a", "b"], {"b"}, np.ones((2, 2), dtype=np.float32), 0)]
            with TuneFeatureCache(root, feature_names=["x", "y"], provenance={}) as cache:
                cache.write_batch(records)
            report, scores = self.cache_report(root, records, corrupt=True)
            with self.assertRaisesRegex(ValueError, "counts or ordered pool identities"):
                cache.finalize(report_path=report, score_path=scores)
            self.assertTrue(cache.marker.exists())
            self.assertFalse(cache.sidecar.exists())

    def test_feature_cache_rejects_silent_feature_precision_change(self):
        with tempfile.TemporaryDirectory() as raw:
            records = [self.cache_record("q1", ["a"], {"a"}, np.ones((1, 2), dtype=np.float64), 0)]
            with TuneFeatureCache(Path(raw), feature_names=["x", "y"], provenance={}) as cache:
                with self.assertRaisesRegex(ValueError, "actual finite float32"):
                    cache.write_batch(records)


if __name__ == "__main__":
    unittest.main()
