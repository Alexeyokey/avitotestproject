"""Проверки текстового поиска и формата выдачи."""
import unittest

from avito_candidates.baseline import BM25Index, Baseline
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

    def test_no_matches(self):
        model = Baseline([{"item_id": "0" * 16, "item_title_raw": "ремонт телевизоров"}])
        self.assertEqual(model.retrieve({"search_query": "автоподбор"}), [])
        self.assertEqual(model.retrieve({"search_query": ""}), [])

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
