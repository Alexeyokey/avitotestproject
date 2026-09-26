"""Проверки текстового поиска и формата выдачи."""
import unittest

from avito_candidates.baseline import Baseline
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

    def test_no_matches(self):
        model = Baseline([{"item_id": "0" * 16, "item_title_raw": "ремонт телевизоров"}])
        self.assertEqual(model.retrieve({"search_query": "автоподбор"}), [])
        self.assertEqual(model.retrieve({"search_query": ""}), [])

    def test_length_normalization(self):
        items = [{"item_id": "0" * 16, "item_title_raw": "ремонт авто шин дисков"},
                 {"item_id": "1" * 16, "item_title_raw": "ремонт"}]
        query = {"search_query": "ремонт"}
        self.assertEqual(Baseline(items).retrieve(query), ["1" * 16, "0" * 16])
        # Без поправки на длину оценки равны; сохраняется порядок корпуса.
        self.assertEqual(Baseline(items, b=0).retrieve(query), ["0" * 16, "1" * 16])

    def test_term_frequency_and_query_repetitions(self):
        items = [{"item_id": "0" * 16, "item_title_raw": "ремонт авто"},
                 {"item_id": "1" * 16, "item_title_raw": "ремонт ремонт"}]
        model = Baseline(items)
        self.assertEqual(model.retrieve({"search_query": "ремонт"}), ["1" * 16, "0" * 16])
        self.assertEqual(model.retrieve({"search_query": "ремонт ремонт"}),
                         model.retrieve({"search_query": "ремонт"}))

    def test_empty_corpus_and_parameters(self):
        for items in [[], [{"item_id": "0" * 16, "item_title_raw": "..."}]]:
            self.assertEqual(Baseline(items).retrieve({"search_query": "ремонт"}), [])
        for settings in [{"k1": 0}, {"k1": float("nan")}, {"b": 2}]:
            with self.assertRaises(ValueError):
                Baseline([], **settings)

    def test_scores_match_bm25_formula(self):
        # Два объявления длиной 1 и 3, слово «ремонт» встречается в обоих.
        import math
        import numpy as np
        model = Baseline([{"item_id": "0" * 16, "item_title_raw": "ремонт"},
                          {"item_id": "1" * 16, "item_title_raw": "ремонт ремонт авто"}])
        term = model.vectorizer.vocabulary_["ремонт"]
        expected = [math.log1p(0.5 / 2.5) * tf * 2.5 /
                    (tf + 1.5 * (0.25 + 0.75 * length / 2))
                    for tf, length in [(1, 1), (2, 3)]]
        np.testing.assert_allclose(model.index[:, term].toarray().ravel(), expected, rtol=1e-6)
