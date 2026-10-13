"""Проверки разбиения запросов, обучения и выбора только из пула кандидатов."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import pyarrow.parquet as pq
import json
from collections import Counter
from unittest.mock import patch

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
    def test_unspecified_zero_category_has_same_features_as_missing_category(self):
        from avito_candidates.reranker import V4_FEATURES
        items = [{"item_id": "zero", "item_category_id": "0"},
                 {"item_id": "known", "item_category_id": "service"},
                 {"item_id": "missing", "item_category_id": ""}]
        config = {"method": "bm25", "candidate_k": 3, "title_weight": 1.0,
                  "params_weight": 0.0, "description_weight": 0.0,
                  "stem_title_weight": 0.0, "stem_params_weight": 0.0,
                  "location_bonus": 0.0, "bm25_rrf_k": 30}
        details = {"candidate_pool": ["zero", "known", "missing"],
                   "channels": {"title": ["zero", "known", "missing"]}}
        builder = FeatureBuilder(items, config, feature_names=V4_FEATURES)
        missing = builder.transform({"search_query": "repair"}, details, ["zero"])
        for sentinel in ("0", 0):
            matrix = builder.transform({"search_query": "repair", "search_category": sentinel},
                                       details, ["zero"])
            self.assertTrue((matrix == missing).all())
            for name in ("category_match", "category_known"):
                self.assertEqual(matrix[:, V4_FEATURES.index(name)].tolist(), [0.0, 0.0, 0.0])
        known = builder.transform({"search_query": "repair", "search_category": "service"},
                                  details, ["zero"])
        self.assertEqual(known[:, V4_FEATURES.index("category_match")].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(known[1, V4_FEATURES.index("category_known")], 1.0)
        self.assertEqual(known[2, V4_FEATURES.index("category_known")], 0.0)

    def test_geo_features_use_only_remaining_query_text_history(self):
        from avito_candidates.geography import LocationAssociations
        associations = LocationAssociations(
            {"own", "other"},
            counts={"region": Counter({"own": 20, "other": 5})},
            totals=Counter({"region": 25}),
            query_counts={"ремонт": {"region": Counter({"own": 20})}})
        items = [{"item_id": item_id, "item_location_id": item_id}
                 for item_id in ("own", "other")]
        config = {"method": "bm25", "candidate_k": 2, "title_weight": 1.0,
                  "params_weight": 0.0, "description_weight": 0.0,
                  "stem_title_weight": 0.0, "stem_params_weight": 0.0,
                  "location_bonus": 0.0, "bm25_rrf_k": 30, "geo_min_history": 20}
        details = {"candidate_pool": ["own", "other"],
                   "channels": {"title": ["own", "other"]}}
        query = {"search_query": "РЕМОНТ", "search_location_id": "region"}
        ordinary = FeatureBuilder(items, config, geo_associations=associations).transform(
            query, details, ["own"])
        training = FeatureBuilder(items, config, geo_associations=associations,
                                  leave_geography_query_out=True).transform(
            query, details, ["own"])
        self.assertAlmostEqual(ordinary[0, FEATURES.index("geo_click_probability")], 0.8)
        for feature in ("geo_click_probability", "geo_click_lift", "geo_click_rank"):
            self.assertEqual(training[:, FEATURES.index(feature)].tolist(), [0.0, 0.0])

    def test_preparation_preserves_group_identity_without_using_ids_as_features(self):
        items = [{"item_id": f"item-{index}", "item_title_raw": "ремонт"}
                 for index in range(66)]
        config = {"method": "bm25", "candidate_k": 66, "title_weight": 1.0,
                  "params_weight": 0.0, "description_weight": 0.0,
                  "stem_title_weight": 0.0, "stem_params_weight": 0.0,
                  "location_bonus": 0.0, "bm25_rrf_k": 30}
        queries = [{"search_query": text, "search_location_id": location}
                   for text, location in ((" РЕМОНТ ", "one"), ("ремонт", "two"))]
        representatives = {query_key(query): query for query in queries}
        truth = {key: {"item-0", "not-in-pool"} for key in representatives}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.parquet"
            metadata = prepare_training_data(
                path, FakeRetriever(items), FeatureBuilder(items, config), truth,
                representatives, split="queries", fraction=0.2, seed=42,
                negatives=64, batch_size=1)
            rows = pq.read_table(path).to_pylist()
            self.assertEqual(len(rows), 130)
            self.assertEqual({row["query_text"] for row in rows}, {"ремонт"})
            self.assertEqual(len({row["group_id"] for row in rows}), 2)
            self.assertEqual({row["relevant_count"] for row in rows}, {2})
            for group_id in {row["group_id"] for row in rows}:
                group = [row for row in rows if row["group_id"] == group_id]
                self.assertEqual(sum(row["label"] for row in group), 1)
                self.assertEqual(len(group) - sum(row["label"] for row in group), 64)
                key = tuple(json.loads(group[0]["query_key_json"]))
                self.assertIn(key, truth)
                self.assertTrue(all(row["label"] == int(row["item_id"] in truth[key])
                                    for row in group))
            self.assertEqual(metadata["training_data_schema_version"], 2)
            self.assertTrue(set(metadata["group_columns"]).isdisjoint(metadata["feature_names"]))
            actual_read = pq.read_table
            with patch("avito_candidates.reranker.pq.read_table", wraps=actual_read) as read_table:
                fit_model(path, Path(directory) / "model.joblib", max_iter=2)
            self.assertEqual(read_table.call_args.kwargs["columns"],
                             [*FEATURES, "label", "weight"])

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

    def test_top50_microcategory_share_and_v4_schema(self):
        from avito_candidates.reranker import V4_FEATURES

        items = [{"item_id": item_id, "item_microcat_id": microcat}
                 for item_id, microcat in (("a", "x"), ("b", "x"),
                                           ("c", "y"), ("d", "x"))]
        config = {"method": "bm25", "candidate_k": 4,
                  "title_weight": 1.0, "params_weight": 0.0,
                  "description_weight": 0.0, "stem_title_weight": 0.0,
                  "stem_params_weight": 0.0, "location_bonus": 0.0,
                  "bm25_rrf_k": 30}
        details = {"candidate_pool": ["a", "b", "c", "d"],
                   "channels": {"title": ["a", "b", "c", "d"]}}
        baseline = ["a", "b", "c"]
        matrix = FeatureBuilder(items, config).transform(
            {}, details, baseline, ["a", "c", "d"])
        for actual, expected in zip(
                matrix[:, FEATURES.index("microcat_top50_share")],
                [2 / 3, 1 / 3, 2 / 3]):
            self.assertAlmostEqual(actual, expected)
        old_matrix = FeatureBuilder(items, config, feature_names=V4_FEATURES).transform(
            {}, details, baseline, ["a", "c", "d"])
        self.assertEqual(old_matrix.shape, (3, len(V4_FEATURES)))

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
