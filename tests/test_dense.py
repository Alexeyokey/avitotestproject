import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from avito_candidates.dense import (
    DEFAULT_EMBEDDING_MODEL, DenseRetriever, _fingerprint,
    _local_model_signature, encoding_input,
    item_text, query_text,
)


class DenseHelpersTest(unittest.TestCase):
    def test_query_batches_encode_once_per_batch_and_keep_order(self):
        calls = []

        class FakeModel:
            def encode(self, texts, **kwargs):
                calls.append((list(texts), kwargs))
                return np.array([
                    [1.0, 0.0] if "первый" in text else [0.0, 1.0]
                    for text in texts
                ], dtype=np.float32)

        class FakeIndex:
            def set_ef(self, _value):
                pass

            def knn_query(self, vectors, k):
                labels = np.array([
                    [0, 1] if vector[0] else [1, 0]
                    for vector in vectors
                ], dtype=int)
                distances = np.tile([0.1, 0.4], (len(vectors), 1))
                return labels[:, :k], distances[:, :k]

        retriever = object.__new__(DenseRetriever)
        retriever.ids = ["a", "b"]
        retriever.batch_size = 2
        retriever.model_name = DEFAULT_EMBEDDING_MODEL
        retriever.model = FakeModel()
        retriever.index = FakeIndex()
        retriever.ef_search = 10
        queries = [
            {"search_query": "первый"},
            {"search_query": ""},
            {"search_query": "второй"},
            {"search_query": "первый снова"},
        ]
        self.assertEqual(
            retriever.retrieve_batch(queries, 2),
            [["a", "b"], [], ["b", "a"], ["a", "b"]],
        )
        scored = retriever.retrieve_batch_with_scores(queries, 2)
        self.assertEqual(scored[0][0], ["a", "b"])
        self.assertAlmostEqual(scored[0][1]["a"], 0.9)
        self.assertAlmostEqual(scored[0][1]["b"], 0.6)
        self.assertEqual(scored[1], ([], {}))
        self.assertEqual([len(texts) for texts, _kwargs in calls], [2, 1, 2, 1])
        self.assertTrue(all(kwargs["batch_size"] == 2 for _texts, kwargs in calls))
        self.assertTrue(all(kwargs["prompt_name"] == "query" for _texts, kwargs in calls))

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

    def test_local_finetuned_model_keeps_saved_query_and_document_prompts(self):
        class SavedOcten:
            prompts = {"query": "поиск: ", "document": "объявление: "}

        for role in ("query", "document"):
            self.assertEqual(
                encoding_input(["ремонт"], "/models/octen-finetuned", role, SavedOcten()),
                (["ремонт"], {"prompt_name": role}),
            )
        self.assertEqual(
            encoding_input(["ремонт"], "/models/model-without-prompts", "query",
                           object()),
            (["ремонт"], {}),
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

    def test_local_checkpoint_change_invalidates_dense_index(self):
        class SavedModel:
            prompts = {"query": "поиск: ", "document": "объявление: "}

        items = [{"item_id": "a", "item_title_raw": "ремонт"}]
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            weights = checkpoint / "model.safetensors"
            weights.write_bytes(b"first")
            first = _local_model_signature(str(checkpoint), SavedModel())
            weights.write_bytes(b"second-version")
            second = _local_model_signature(str(checkpoint), SavedModel())
            self.assertNotEqual(first, second)
            self.assertNotEqual(
                _fingerprint(items, str(checkpoint), model_signature=first),
                _fingerprint(items, str(checkpoint), model_signature=second),
            )


if __name__ == "__main__":
    unittest.main()
