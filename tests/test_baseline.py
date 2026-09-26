"""Small end-to-end checks for training signals and the export contract."""
import unittest

from avito_candidates.baseline import Baseline
from avito_candidates.core import validate_answers


class BaselineTest(unittest.TestCase):
    def test_history_corpus_boundary_and_export(self):
        items = [{"item_id": f"{i:016x}", "item_title_raw": f"услуга ремонт номер {i}",
                  "item_infm_params_text": "мастер"} for i in range(60)]
        chosen = items[-1]["item_id"]
        query = {"query_id": "A" * 16, "search_query": "неизвестныйзапрос"}
        train = [{**query, "item_id": chosen}, {**query, "item_id": "f" * 16}]
        model = Baseline(items, train)
        result = model.retrieve(query)
        self.assertEqual(result[0], chosen)
        self.assertEqual(len(result), 50)
        self.assertEqual(result, model.retrieve(query))
        validate_answers([{"query_id": query["query_id"], "answer": " ".join(result)}],
                         [query["query_id"]], [x["item_id"] for x in items])

    def test_lexical_signal_without_history(self):
        items = [{"item_id": "0" * 16, "item_title_raw": "автоподбор осмотр автомобиля"},
                 {"item_id": "1" * 16, "item_title_raw": "монтаж видеодомофонов"}]
        model = Baseline(items, [])
        self.assertEqual(model.retrieve({"search_query": "монтаж видеодомофонов"})[0], "1" * 16)
