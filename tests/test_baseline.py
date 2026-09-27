"""Проверки текстового поиска и формата выдачи."""
import unittest

from avito_candidates.baseline import BM25Index, Baseline, stem_text
from avito_candidates.core import validate_answers


class BaselineTest(unittest.TestCase):
    def test_top50_and_export(self):
        items = [{"item_id": f"{i:016x}", "item_title_raw": f"услуга ремонт номер {i}",
                  "item_infm_params_text": "мастер"} for i in range(60)]
        query = {"query_id": "A" * 16, "search_query": "ремонт"}
        model = Baseline(items)
        result = model.retrieve(query)
        self.assertEqual(len(result), 50)
        self.assertEqual(result, model.retrieve(query))
        validate_answers([{"query_id": query["query_id"], "answer": " ".join(result)}],
                         [query["query_id"]], [x["item_id"] for x in items])

    def test_lexical_signal_without_history(self):
        items = [{"item_id": "0" * 16, "item_title_raw": "автоподбор осмотр автомобиля"},
                 {"item_id": "1" * 16, "item_title_raw": "монтаж видеодомофонов"}]
        model = Baseline(items)
        self.assertEqual(model.retrieve({"search_query": "монтаж видеодомофонов"})[0], "1" * 16)

    def test_each_field_can_find_an_item(self):
        items = [
            {"item_id": "0" * 16, "item_title_raw": "автоподбор"},
            {"item_id": "1" * 16, "item_infm_params_text": "автоподбор"},
            {"item_id": "2" * 16, "item_description_raw": "автоподбор"},
        ]
        query = {"search_query": "автоподбор"}
        for weight, expected in (
            ({"title_weight": 1, "params_weight": 0, "description_weight": 0}, "0" * 16),
            ({"title_weight": 0, "params_weight": 1, "description_weight": 0}, "1" * 16),
            ({"title_weight": 0, "params_weight": 0, "description_weight": 1}, "2" * 16),
        ):
            self.assertEqual(Baseline(items, **weight).retrieve(query, 1), [expected])

    def test_diagnostics_include_all_field_candidates_before_top50(self):
        items = [
            {"item_id": "0" * 16, "item_title_raw": "автоподбор"},
            {"item_id": "1" * 16, "item_infm_params_text": "автоподбор"},
            {"item_id": "2" * 16, "item_description_raw": "автоподбор"},
        ]
        model = Baseline(items, candidate_k=2)
        prediction, diagnostics = model.retrieve_with_diagnostics(
            {"search_query": "автоподбор"}, limit=1
        )
        self.assertEqual(len(prediction), 1)
        self.assertEqual(
            set(diagnostics["candidate_pool"]), {item["item_id"] for item in items}
        )
        self.assertEqual(model.retrieve({"search_query": "автоподбор"}, 1), prediction)

    def test_no_matches(self):
        model = Baseline([{"item_id": "0" * 16, "item_title_raw": "ремонт телевизоров"}])
        self.assertEqual(model.retrieve({"search_query": "автоподбор"}), [])
        self.assertEqual(model.retrieve({"search_query": ""}), [])

    def test_stem_channel_finds_a_new_candidate_without_replacing_exact_search(self):
        self.assertEqual(stem_text("Ремонт Ёлки"), stem_text("ремонта елка"))
        items = [
            {"item_id": "0" * 16, "item_title_raw": "монтаж домофонов"},
            {"item_id": "1" * 16, "item_title_raw": "ремонта квартир"},
        ]
        query = {"search_query": "ремонт квартиры"}
        exact = Baseline(items, title_weight=1, params_weight=0,
                         description_weight=0, candidate_k=2)
        stemmed = Baseline(items, title_weight=1, params_weight=0,
                           description_weight=0, stem_title_weight=0.5,
                           candidate_k=2)
        self.assertEqual(exact.retrieve(query), [])
        self.assertEqual(stemmed.ranked_lists(query, 2)[0][0], "title")
        self.assertEqual(stemmed.ranked_lists(query, 2)[1][0], "title_stem")
        self.assertEqual(stemmed.retrieve(query), ["1" * 16])

    def test_stemmed_parameters_are_a_separate_channel(self):
        items = [{"item_id": "2" * 16, "item_infm_params_text": "установки домофонов"}]
        model = Baseline(items, title_weight=0, params_weight=1,
                         description_weight=0, stem_params_weight=0.5)
        channels = model.ranked_lists({"search_query": "установка домофона"}, 1)
        self.assertEqual(channels[0], ("params", [], 1))
        self.assertEqual(channels[1], ("params_stem", ["2" * 16], 0.5))

    def test_location_bonus_promotes_local_candidate_without_hard_filter(self):
        items = [
            {"item_id": "0" * 16, "item_title_raw": "ремонт",
             "item_location_id": "other"},
            {"item_id": "1" * 16, "item_title_raw": "ремонт",
             "item_location_id": "local"},
        ]
        query = {"search_query": "ремонт", "search_location_id": "local"}
        model = Baseline(items, title_weight=1, params_weight=0,
                         description_weight=0, candidate_k=2,
                         channel_quota=0, location_bonus=0.1)
        self.assertEqual(model.retrieve(query, 1), ["1" * 16])
        self.assertEqual(model.retrieve({"search_query": "ремонт"}, 1), ["0" * 16])
        self.assertEqual(set(model.retrieve(query, 2)), {"0" * 16, "1" * 16})

    def test_length_normalization(self):
        ids = ["0" * 16, "1" * 16]
        texts = ["ремонт авто шин дисков", "ремонт"]
        self.assertEqual(BM25Index(ids, texts, k1=1.5, b=0.75).retrieve("ремонт", 2), ids[::-1])
        # Без поправки на длину оценки равны; сохраняется порядок корпуса.
        self.assertEqual(BM25Index(ids, texts, k1=1.5, b=0).retrieve("ремонт", 2), ids)

    def test_term_frequency_and_query_repetitions(self):
        ids = ["0" * 16, "1" * 16]
        model = BM25Index(ids, ["ремонт авто", "ремонт ремонт"], k1=1.5, b=0.75)
        self.assertEqual(model.retrieve("ремонт", 2), ids[::-1])
        self.assertEqual(model.retrieve("ремонт ремонт", 2), model.retrieve("ремонт", 2))

    def test_empty_corpus_and_parameters(self):
        for texts in [[], ["..."]]:
            self.assertEqual(BM25Index(["0"][:len(texts)], texts, k1=1.5, b=0.75)
                             .retrieve("ремонт", 50), [])
        for settings in [{"k1": 0}, {"k1": float("nan")}, {"b": 2}]:
            with self.assertRaises(ValueError):
                BM25Index([], [], k1=settings.get("k1", 1.5), b=settings.get("b", 0.75))

    def test_scores_match_bm25_formula(self):
        # Два объявления длиной 1 и 3, слово «ремонт» встречается в обоих.
        import math
        import numpy as np
        model = BM25Index(["0" * 16, "1" * 16], ["ремонт", "ремонт ремонт авто"],
                          k1=1.5, b=0.75)
        term = model.vectorizer.vocabulary_["ремонт"]
        expected = [math.log1p(0.5 / 2.5) * tf * 2.5 /
                    (tf + 1.5 * (0.25 + 0.75 * length / 2))
                    for tf, length in [(1, 1), (2, 3)]]
        np.testing.assert_allclose(model.index[:, term].toarray().ravel(), expected, rtol=1e-6)
