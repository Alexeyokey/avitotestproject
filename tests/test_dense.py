import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from avito_candidates.dense import (
    DEFAULT_EMBEDDING_MODEL, DenseRetriever, _fingerprint, encoding_input,
    item_text, query_text,
)


class DenseHelpersTest(unittest.TestCase):
    def test_octen_uses_its_model_prompts_without_e5_prefixes(self):
        self.assertEqual(DEFAULT_EMBEDDING_MODEL, "Octen/Octen-Embedding-0.6B")
        self.assertEqual(
            encoding_input(["ремонт квартир"], DEFAULT_EMBEDDING_MODEL, "query"),
            (["ремонт квартир"], {"prompt_name": "query"}),
        )
        self.assertEqual(
            encoding_input(["услуги мастера"], DEFAULT_EMBEDDING_MODEL, "document"),
            (["услуги мастера"], {"prompt_name": "document"}),
        )

    def test_e5_override_keeps_legacy_prefixes(self):
        self.assertEqual(
            encoding_input(["ремонт квартир"], "intfloat/multilingual-e5-small", "query"),
            (["query: ремонт квартир"], {}),
        )
        self.assertEqual(
            encoding_input(["услуги мастера"], "intfloat/multilingual-e5-small", "document"),
            (["passage: услуги мастера"], {}),
        )

    def test_octen_prompts_are_used_for_indexing_and_search(self):
        calls = []

        class FakeModel:
            def __init__(self, _name, **_kwargs):
                pass

            def get_embedding_dimension(self):
                return 3

            def encode(self, texts, **kwargs):
                calls.append((texts, kwargs))
                return np.ones((len(texts), 3), dtype=np.float32)

        class FakeIndex:
            def __init__(self, **_kwargs):
                pass

            def init_index(self, **_kwargs):
                pass

            def add_items(self, _vectors, _labels):
                pass

            def set_ef(self, _value):
                pass

            def save_index(self, path):
                Path(path).write_bytes(b"fake")

            def knn_query(self, _vector, k):
                return np.zeros((1, k), dtype=int), np.zeros((1, k))

        class FakeHnsw:
            Index = FakeIndex

        with TemporaryDirectory() as directory, patch(
            "avito_candidates.dense._load_dependencies",
            return_value=(FakeHnsw, FakeModel),
        ):
            model = DenseRetriever(
                [{"item_id": "a", "item_title_raw": "ремонт"}],
                cache_dir=directory,
            )
            self.assertEqual(model.retrieve({"search_query": "ремонт"}, 1), ["a"])

        self.assertEqual(calls[0][1]["prompt_name"], "document")
        self.assertEqual(calls[1][1]["prompt_name"], "query")
        self.assertFalse(calls[0][0][0].startswith("passage: "))
        self.assertFalse(calls[1][0][0].startswith("query: "))

    def test_item_text_balances_all_fields(self):
        self.assertEqual(
            item_text({
                "item_title_raw": "  РЕМОНТ Ёлок ",
                "item_category_id": "114",
                "item_infm_params_text": " выезд срочно недорого ",
                "item_description_raw": " большой опыт работы ",
            }, params_words=2, description_words=2),
            "заголовок: ремонт елок категория: 114 параметры: выезд срочно "
            "описание: большой опыт",
        )

    def test_query_text_uses_filters_and_nonzero_category(self):
        self.assertEqual(
            query_text({
                "search_query": " Монтаж ",
                "search_infm_params_text": " Вид услуги Домофоны ",
                "search_category": "114",
            }),
            "запрос: монтаж фильтры: вид услуги домофоны категория: 114",
        )
        self.assertEqual(
            query_text({"search_query": "монтаж", "search_category": "0"}),
            "запрос: монтаж",
        )

    def test_fingerprint_changes_with_corpus_or_model(self):
        items = [{"item_id": "a", "item_title_raw": "ремонт"}]
        original = _fingerprint(items, "model-a", 128)
        self.assertNotEqual(original, _fingerprint(items, "model-b"))
        self.assertNotEqual(original, _fingerprint(
            [{"item_id": "a", "item_title_raw": "монтаж"}], "model-a"
        ))
        self.assertNotEqual(original, _fingerprint(items, "model-a", 256))
        self.assertNotEqual(original, _fingerprint(items, "model-a", 128, 20, 48))
        self.assertNotEqual(original, _fingerprint(items, "model-a", 128, 40, 20))


if __name__ == "__main__":
    unittest.main()
