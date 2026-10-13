"""Small frozen-pool microcategory ranking checks, with no real training run."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import evaluate_microcategory_ranking as M


def item(value):
    return f"{value:016x}"


class DummyHGB:
    n_iter_ = 150

    def predict_proba(self, matrix):
        return np.column_stack((np.full(len(matrix), .5), np.full(len(matrix), .5)))

    def staged_predict_proba(self, matrix):
        for _ in range(self.n_iter_):
            yield self.predict_proba(matrix)


class DummyClassifier:
    def __init__(self):
        self.model = SimpleNamespace(fitted_texts=frozenset({"fit"}), classes=np.array(["A", "B"]))
        self._prediction_cache = {}
        self.metadata = {"fixture": True, "crossfit_folds": 0}

    def predict_categories_batch(self, queries, *, top_k, exclude_self):
        self._prediction_cache["discard"] = "not persisted"
        return [(("A", .9), ("B", .1)) for _ in queries]


def group(index, text, available, full=None, *, size=0, hits=0, baseline=0):
    key = [text, str(index), "0", "", "114"]
    full = available if full is None else full
    return {"group_id": index, "query_text": text, "query_key": key,
        "query_key_json": json.dumps(key, ensure_ascii=False, separators=(",", ":")),
        "group_identity": M.RETRIEVAL.identity_for_key(key), "pool_size": size,
        "positives_in_pool": hits, "baseline_hits": baseline, "relevant_count": len(available),
        "available_relevant_count": len(available), "full_relevant_count": len(full),
        "representative": True, "_available_ids": set(available), "one_word": True,
        "available_item_ids": available, "relevant_item_ids": full}


class MicrocategoryRankingTests(unittest.TestCase):
    def test_alpha_zero_preserves_probability_endpoints_and_tie_order(self):
        scores = [0., 0., .5, .5, 1., 1.]
        adjusted = M.adjusted_scores(scores, ["A"] * 6, {"A": .1}, 0.)
        self.assertEqual(adjusted, scores)
        self.assertEqual(M.EVAL.top_indices(adjusted, [1, 1, 1, 2, 1, 1], 6), [4, 5, 3, 2, 0, 1])

    def test_unknown_category_gets_declared_floor_without_looking_at_labels(self):
        result = M.adjusted_scores([.5, .5], ["known", "missing"], {"known": .8}, .1)
        self.assertAlmostEqual(result[0], .1 * np.log(.8))
        self.assertAlmostEqual(result[1], .1 * np.log(1e-5))
        self.assertGreater(result[0], result[1])

    def test_new_ranking_changes_top50_only_inside_existing_pool(self):
        items = [item(value) for value in range(51)]
        current = group(0, "cold", [items[-1], item(999)], size=51, hits=1)
        raw = {"items": items, "labels": [0] * 50 + [1], "hgb": [.5] * 51,
               "rrf": [.1] * 51, "baseline": [True] * 50 + [False]}
        categories = {value: "B" if value == items[-1] else "A" for value in items}
        result = M.score_group(current, raw, {"hgb_hits": 0, "pool_hits": 1, "baseline_hits": 0},
                               categories, {"A": .01, "B": .99})
        self.assertEqual(result["a0_hits"], 0)
        self.assertEqual(result["a0p1_hits"], 1)
        self.assertEqual(result["pool_hits"], 1)
        self.assertEqual(result["available_relevant_count"], 2)
        self.assertEqual(result["a0p1_selected_count"], 50)

    def test_alpha_zero_group_mismatch_fails_closed(self):
        current = group(0, "cold", [item(1)], size=1, hits=1, baseline=1)
        raw = {"items": [item(1)], "labels": [1], "hgb": [.5], "rrf": [.1], "baseline": [True]}
        with self.assertRaises(ValueError):
            M.score_group(current, raw, {"hgb_hits": 0, "pool_hits": 1, "baseline_hits": 1},
                          {item(1): "A"}, {"A": 1.})

    def test_cached_labels_or_pool_shape_cannot_change(self):
        current = group(0, "cold", [item(1)], size=1, hits=1, baseline=1)
        raw = {"items": [item(1)], "labels": [0], "hgb": [.5], "rrf": [.1], "baseline": [True]}
        with self.assertRaises(ValueError):
            M.score_group(current, raw, {"hgb_hits": 1, "pool_hits": 1, "baseline_hits": 1},
                          {item(1): "A"}, {"A": 1.})

    def test_classifier_own_text_is_forbidden_and_prediction_cache_is_disposable(self):
        classifier = DummyClassifier()
        result = M.full_category_probabilities(classifier, {"search_query": "cold"})
        self.assertEqual(result, {"A": .9, "B": .1})
        self.assertEqual(classifier._prediction_cache, {})
        with self.assertRaises(ValueError):
            M.full_category_probabilities(classifier, {"search_query": " FIT "})

    def test_classifier_must_cover_all_classes_with_normalized_finite_scores(self):
        classifier = DummyClassifier()
        for predicted in ((("A", .9),), (("A", .9), ("B", .9)), (("A", np.nan), ("B", .1))):
            with self.subTest(predicted=predicted), mock.patch.object(
                    classifier, "predict_categories_batch", return_value=[predicted]):
                with self.assertRaises(ValueError):
                    M.full_category_probabilities(classifier, {"search_query": "cold"})

    def test_selected_final_hgb_predicts_once_instead_of_iterating150_stages(self):
        model = DummyHGB()
        matrix = np.ones((3, 2), dtype=np.float32)
        with mock.patch.object(model, "staged_predict_proba", wraps=model.staged_predict_proba) as staged:
            self.assertEqual(M.selected_hgb_scores(model, matrix, 150).tolist(), [.5] * 3)
            staged.assert_not_called()
        self.assertEqual(M.selected_hgb_scores(model, matrix, 50).tolist(), [.5] * 3)

    def test_stream_handles_split_and_empty_groups_and_selected_subset(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tune.parquet"
            groups = [group(0, "cold", [item(1), item(999)], size=3, hits=1, baseline=1),
                      group(1, "empty", [item(4)]), group(2, "other", [], size=1)]
            columns = {"group_id": pa.array([0, 0, 0, 2], type=pa.int32()),
                "item_id": [item(1), item(2), item(3), item(9)], "label": [1, 0, 0, 0],
                "rrf_score": pa.array([.1, .09, .08, .1], type=pa.float32()),
                "baseline_top50": pa.array([1] * 4, type=pa.float32())}
            pq.write_table(pa.table(columns), path)
            metadata = {"feature_names": ["rrf_score", "baseline_top50"], "rows": 4}
            artifact = {"model": DummyHGB()}
            expected = [{"hgb_hits": 1, "pool_hits": 1, "baseline_hits": 1},
                        {"hgb_hits": 0, "pool_hits": 0, "baseline_hits": 0},
                        {"hgb_hits": 0, "pool_hits": 0, "baseline_hits": 0}]
            categories = {item(value): "A" for value in (1, 2, 3, 9)}
            rows = M.stream_scores(path, metadata, groups, artifact, DummyClassifier(), categories,
                                    expected, 150, {0, 1, 2}, batch_size=2)
            self.assertEqual([row["group_id"] for row in rows], [0, 1, 2])
            self.assertEqual(rows[0]["available_relevant_count"], 2)
            self.assertEqual(rows[1]["a0_hits"], 0)
            filtered = M.stream_scores(path, metadata, groups, artifact, DummyClassifier(), categories,
                                       expected, 150, {2}, batch_size=1)
            self.assertEqual([row["group_id"] for row in filtered], [2])

    def test_text_macro_and_cluster_bootstrap_do_not_weight_contexts_as_texts(self):
        rows = [{"query_text": "same", "available_relevant_count": 2, "full_relevant_count": 3,
                 "representative": True, "a0_hits": 1, "a0p1_hits": 2},
                {"query_text": "same", "available_relevant_count": 1, "full_relevant_count": 1,
                 "representative": False, "a0_hits": 1, "a0p1_hits": 1},
                {"query_text": "other", "available_relevant_count": 1, "full_relevant_count": 2,
                 "representative": True, "a0_hits": 0, "a0p1_hits": 0},
                {"query_text": "unavailable", "available_relevant_count": 0, "full_relevant_count": 1,
                 "representative": True, "a0_hits": 0, "a0p1_hits": 0}]
        result = M.paired_bootstrap(rows, "a0p1", replicates=20)
        self.assertEqual(result["query_text_clusters"], 2)
        self.assertEqual(result["delta"], .125)
        full = M.EVAL.aggregate(rows, "a0_hits", "full_relevant_count")
        self.assertEqual(full["groups"], 4)
        self.assertAlmostEqual(full["text_macro_recall_at_50"], (2 / 3) / 3)

    def test_tied_alpha_choice_prefers_zero_without_model_changes(self):
        current = group(0, "cold", [item(1)], size=1, hits=1, baseline=1)
        rows = [{**current, "pool_hits": 1, **{f"{M.alpha_key(alpha)}_hits": 1 for alpha in M.ALPHAS}}]
        report = M.summarize(rows, bootstrap_replicates=10)
        self.assertEqual(report["selected_alpha_on_tune"], 0.)
        self.assertEqual(report["variants"]["a0p25"]["conditional_available"]["text_macro_recall_at_50"], 1.)

    def test_completed_reference_checks_canonical_group_identity_and_scores_sha(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            current = group(0, "cold", [item(1)], size=1, hits=1, baseline=1)
            (root / "selection.json").write_text("{}", encoding="utf-8")
            outcome = {key: value for key, value in current.items() if not key.startswith("_")}
            outcome.update(s150_hits=1, pool_hits=1)
            scores = root / "tune-group-scores.jsonl"
            scores.write_text(json.dumps(outcome) + "\n", encoding="utf-8")
            report = {"phase": "tune", "report_version": 1, "selected_stage": 150,
                "policy": M.EVAL.POLICY, "cache_sha256": {"parquet": "x"},
                "group_scores_sha256": M.EVAL.sha256_file(scores)}
            (root / "tune-report.json").write_text(json.dumps(report), encoding="utf-8")
            with mock.patch.object(M.EVAL, "verify_selection", return_value=150):
                stage, outcomes, _, _ = M.read_completed_tune(root, [current],
                    {"cache_parquet_sha256": "x"}, "model", "protocol", "train")
                self.assertEqual(stage, 150)
                self.assertEqual(outcomes[0]["hgb_hits"], 1)
                scores.write_text(json.dumps({**outcome, "group_identity": "wrong"}) + "\n", encoding="utf-8")
                report["group_scores_sha256"] = M.EVAL.sha256_file(scores)
                (root / "tune-report.json").write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaises(ValueError):
                    M.read_completed_tune(root, [current], {"cache_parquet_sha256": "x"}, "model", "protocol", "train")

    def test_report_cannot_overwrite_existing_output(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text("old result", encoding="utf-8")
            with self.assertRaises(ValueError):
                M.evaluate(argparse.Namespace(output=output))


if __name__ == "__main__":
    unittest.main()
