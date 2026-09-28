"""Проверки разбиения запросов, обучения и выбора только из пула кандидатов."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from avito_candidates.core import held_out, normalize, query_key
from avito_candidates.reranker import (
    FEATURES, FeatureBuilder, Reranker, fit_model, prepare_training_data,
    read_fit_labels, sample_training_ids,
)


class FakeRetriever:
    def __init__(self, items):
        self.ids = [item["item_id"] for item in items]

    def retrieve_with_diagnostics(self, query):
        # Заведомо слабый поиск: доступны все объявления в неизменном порядке.
        return self.ids[:50], {"candidate_pool": self.ids,
                               "channels": {"title": self.ids}}


class RerankerTest(unittest.TestCase):
    def test_hard_negatives_include_multiple_channels(self):
        baseline = ["positive", *[f"local-{index}" for index in range(50)]]
        channels = {"title": baseline,
                    "title_geo": [f"geo-{index}" for index in range(20)],
                    "dense": [f"dense-{index}" for index in range(20)]}
        pool = list(dict.fromkeys([*baseline, *channels["title_geo"], *channels["dense"]]))
        chosen = sample_training_ids(pool, baseline, channels, {"positive"}, 64, 42)
        self.assertEqual(chosen[0], "positive")
        self.assertTrue(any(item_id.startswith("geo-") for item_id in chosen[:33]))
        self.assertTrue(any(item_id.startswith("dense-") for item_id in chosen[:33]))

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

    def test_geographic_channel_uses_actual_fused_score(self):
        items = [{"item_id": "a", "item_location_id": "1"}]
        config = {"method": "bm25", "candidate_k": 1, "title_weight": 1.0,
                  "params_weight": 0.0, "description_weight": 0.0,
                  "stem_title_weight": 0.0, "stem_params_weight": 0.0,
                  "location_bonus": 0.0, "bm25_rrf_k": 30}
        details = {"candidate_pool": ["a"],
                   "channels": {"title": ["a"], "title_geo": ["a"]},
                   "rrf_scores": {"a": 1.25 / 31}}
        matrix = FeatureBuilder(items, config).transform(
            {"search_query": "ремонт", "search_location_id": "1"}, details, ["a"])
        self.assertAlmostEqual(matrix[0, FEATURES.index("rrf_score")], 1.25 / 31)
        self.assertEqual(matrix[0, FEATURES.index("channel_count")], 2)
        self.assertGreater(matrix[0, FEATURES.index("rank_title_geo")], 0)
        self.assertEqual(matrix[0, FEATURES.index("geo_channel_count")], 1)

    def test_raw_scores_geography_and_filter_compatibility(self):
        items = [
            {"item_id": "a", "item_location_id": "city", "item_rating": "4.8",
             "item_infm_params_text": "Тип услуги Маникюр, педикюр Вид услуги Красота, здоровье"},
            {"item_id": "b", "item_location_id": "other", "item_rating": "3.5",
             "item_infm_params_text": "Тип услуги Ремонт обуви Вид услуги Бытовые услуги"},
        ]
        config = {"method": "hybrid", "candidate_k": 2, "bm25_weight": 1.0,
                  "dense_weight": 1.0, "title_weight": 1.0, "params_weight": 1.0,
                  "description_weight": 0.0, "stem_title_weight": 0.0,
                  "stem_params_weight": 0.0, "rrf_k": 60, "location_bonus": 0.0}
        details = {"candidate_pool": ["a", "b"],
                   "channels": {"params_geo": ["a"], "dense": ["b", "a"]},
                   "channel_scores": {"params_geo": {"a": 7.25},
                                      "dense": {"b": 0.91, "a": 0.73}},
                   "geo_locations": ("city",)}
        query = {"search_query": "маникюр", "search_location_id": "city",
                 "search_infm_params_text": "Тип услуги Маникюр, педикюр "
                                            "Рейтинг пользователя 4 звезды и выше"}
        matrix = FeatureBuilder(items, config).transform(query, details, ["b"])
        feature = lambda name: matrix[:, FEATURES.index(name)]
        self.assertEqual(feature("score_params_geo").tolist(), [7.25, 0.0])
        self.assertAlmostEqual(feature("score_dense")[0], 0.73, places=5)
        self.assertEqual(feature("geo_destination_match").tolist(), [1.0, 0.0])
        self.assertGreater(feature("filter_params_bigram_coverage")[0],
                           feature("filter_params_bigram_coverage")[1])
        self.assertEqual(feature("rating_filter_match").tolist(), [1.0, 0.0])

    def test_model_can_promote_nonlocal_candidate_from_same_pool(self):
        items = [
            {"item_id": "remote", "item_location_id": "2"},
            {"item_id": "local", "item_location_id": "1"},
        ]
        config = {"method": "hybrid", "candidate_k": 2, "bm25_weight": 1.0,
                  "dense_weight": 1.0, "title_weight": 1.0, "params_weight": 0.0,
                  "description_weight": 0.0, "stem_title_weight": 0.0,
                  "stem_params_weight": 0.0, "rrf_k": 60, "location_bonus": 0.1}
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
        self.assertEqual(ranker.rank_batch([query], [result], limit=1), [["remote"]])

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
