"""Reranker contracts: disjoint query split, model fit and candidate-only output."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from avito_candidates.core import held_out, normalize, query_key
from avito_candidates.reranker import (
    FEATURES, FeatureBuilder, Reranker, fit_model, prepare_training_data,
    read_fit_labels,
)


class FakeRetriever:
    def __init__(self, items):
        self.ids = [item["item_id"] for item in items]

    def retrieve_with_diagnostics(self, query):
        # A deliberately weak retriever: all items are available, fixed order.
        return self.ids[:50], {"candidate_pool": self.ids,
                               "channels": {"title": self.ids}}


class RerankerTest(unittest.TestCase):
    def test_hybrid_features_use_bm25_and_dense_lists(self):
        items = [{"item_id": "a", "item_title_raw": "ремонт", "item_location_id": "1"},
                 {"item_id": "b", "item_title_raw": "монтаж", "item_location_id": "2"}]
        config = {"method": "hybrid", "candidate_k": 2, "bm25_weight": 1.0,
                  "dense_weight": 1.0, "title_weight": 2.0, "params_weight": 0.0,
                  "description_weight": 0.0, "stem_title_weight": 0.0,
                  "stem_params_weight": 0.0, "rrf_k": 60, "location_bonus": 0.1}
        details = {"candidate_pool": ["a", "b"],
                   "channels": {"title": ["a", "b"], "dense": ["b", "a"]}}
        matrix = FeatureBuilder(items, config).transform(
            {"search_query": "ремонт", "search_location_id": "1"}, details, ["a"])
        self.assertEqual(matrix.shape, (2, len(FEATURES)))
        self.assertEqual(matrix[0, FEATURES.index("location_match")], 1)
        self.assertGreater(matrix[0, FEATURES.index("rank_title")],
                           matrix[1, FEATURES.index("rank_title")])
        self.assertGreater(matrix[1, FEATURES.index("rank_dense")],
                           matrix[0, FEATURES.index("rank_dense")])

    def test_dominant_location_priority_survives_model_scores(self):
        items = [
            {"item_id": "remote", "item_location_id": "2"},
            {"item_id": "local", "item_location_id": "1"},
        ]
        config = {"method": "dense", "candidate_k": 2}
        query = {"search_query": "ремонт", "search_location_id": "1"}
        result = (["remote", "local"], {"candidate_pool": ["remote", "local"],
                                          "channels": {"dense": ["remote", "local"]}})

        class FixedModel:
            def predict_proba(self, matrix):
                import numpy as np
                return np.array([[0.1, 0.9], [0.9, 0.1]])

        ranker = object.__new__(Reranker)
        ranker.builder = FeatureBuilder(items, config)
        ranker.model = FixedModel()
        ranker.location_first = True
        self.assertEqual(ranker.rank_batch([query], [result], limit=1), [["local"]])

    def test_fit_split_excludes_all_variants_of_heldout_text(self):
        rows = []
        for index in range(30):
            text = f"услуга {index}"
            rows.extend([
                {"search_query": text, "search_location_id": "1", "item_id": "a"},
                {"search_query": text.upper(), "search_location_id": "2", "item_id": "b"},
            ])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.parquet"
            pd.DataFrame(rows).to_parquet(path)
            truth, _reps, stats = read_fit_labels(
                path, {"a", "b"}, split="queries", fraction=0.3, seed=42,
                batch_size=7)
        fit_texts = {key[0] for key in truth}
        expected = {normalize(row["search_query"]) for row in rows
                    if not held_out(normalize(row["search_query"]), 0.3, 42)}
        self.assertEqual(fit_texts, expected)
        self.assertEqual(stats["fit_rows"], 2 * len(expected))
        self.assertTrue(all(not held_out(text, 0.3, 42) for text in fit_texts))

    def test_prepare_fit_and_rerank_from_same_pool(self):
        items = [
            {"item_id": "a", "item_title_raw": "ремонт", "item_location_id": "1"},
            {"item_id": "b", "item_title_raw": "ремонт", "item_location_id": "2"},
            {"item_id": "c", "item_title_raw": "доставка", "item_location_id": "3"},
            {"item_id": "d", "item_title_raw": "строительство", "item_location_id": "4"},
        ]
        config = {"method": "bm25", "candidate_k": 4, "title_weight": 1.0,
                  "params_weight": 0.0, "description_weight": 0.0,
                  "stem_title_weight": 0.0, "stem_params_weight": 0.0,
                  "location_bonus": 0.0, "bm25_rrf_k": 30}
        truth, reps = {}, {}
        for index in range(80):
            location = "1" if index % 2 else "2"
            query = {"search_query": f"ремонт {index}", "search_location_id": location}
            key = query_key(query)
            truth[key] = {"a" if location == "1" else "b"}
            reps[key] = query
        with TemporaryDirectory() as directory:
            data = Path(directory) / "train.parquet"
            model_path = Path(directory) / "model.joblib"
            metadata = prepare_training_data(
                data, FakeRetriever(items), FeatureBuilder(items, config), truth, reps,
                split="queries", fraction=0.2, seed=42, negatives=3, batch_size=9)
            self.assertEqual(metadata["stats"]["positive_rows"], 80)
            self.assertEqual(metadata["stats"]["unobserved_rows"], 240)
            report = fit_model(data, model_path, max_iter=30)
            self.assertEqual(report["training_rows"], 320)
            ranker = Reranker(model_path, items, config,
                              evaluation_split=("queries", 0.2, 42))
            query = {"search_query": "ремонт новый", "search_location_id": "2"}
            result = FakeRetriever(items).retrieve_with_diagnostics(query)
            prediction = ranker.rank_batch([query], [result], limit=2)[0]
            self.assertEqual(prediction[0], "b")
            self.assertEqual(len(prediction), 2)
            self.assertTrue(set(prediction) <= set(result[1]["candidate_pool"]))
            with self.assertRaisesRegex(ValueError, "retrieval settings"):
                Reranker(model_path, items, {**config, "candidate_k": 5})
            with self.assertRaisesRegex(ValueError, "held-out query-text"):
                Reranker(model_path, items, config, evaluation_split=("all", 0.2, 42))
            with self.assertRaisesRegex(ValueError, "corpus"):
                Reranker(model_path, [*items[:-1], {**items[-1], "item_title_raw": "new"}], config)


if __name__ == "__main__":
    unittest.main()
