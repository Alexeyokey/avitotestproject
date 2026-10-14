import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from avito_candidates.dense import (
    DEFAULT_EMBEDDING_MODEL, DenseRetriever, _fingerprint,
    _local_model_signature, encoding_input,
    item_text, query_text, document_text,
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

    def test_legacy_fingerprint_has_not_changed_with_default_format(self):
        items = [{"item_id": "a", "item_title_raw": "ремонт", "item_category_id": "114",
                  "item_infm_params_text": "вид услуги x", "item_description_raw": "description"}]
        # Captured from the pre-extension version, including a local signature.
        self.assertEqual(_fingerprint(items, "model-a", 128, 40, 48, "sig"),
                         "aa647f0df73e0e8a8354")
        self.assertEqual(_fingerprint(items, "model-a", 128, 40, 48, "sig", "current"),
                         "aa647f0df73e0e8a8354")

    def test_document_variants_get_separate_content_bound_indexes(self):
        items = [{"item_id": "a", "item_title_raw": "ремонт", "item_category_id": "114",
                  "item_infm_params_text": "вид услуги телефоны", "item_description_raw": "description"}]
        variants = {variant: _fingerprint(items, "model", document_format=variant)
                    for variant in ("current", "compact", "title_description")}
        self.assertEqual(len(set(variants.values())), 3)
        changed = [{**items[0], "item_description_raw": "different"}]
        self.assertNotEqual(variants["compact"], _fingerprint(changed, "model", document_format="compact"))
        # Changing implementation format version must invalidate experimental
        # indexes, even if this particular row happens to render identically.
        with patch("avito_candidates.embedding_formats.FORMAT_VERSION", 999):
            self.assertNotEqual(variants["compact"], _fingerprint(items, "model", document_format="compact"))
            self.assertEqual(variants["current"], _fingerprint(items, "model"))

    def test_document_helper_matches_declared_layout(self):
        from avito_candidates.embedding_formats import item_format
        item = {"item_title_raw": "t", "item_infm_params_text": "вид услуги v", "item_description_raw": "a b c"}
        for variant in ("current", "compact", "title_description"):
            self.assertEqual(document_text(item, document_format=variant, description_words=2),
                             item_format(item, variant, description_words=2))
        with self.assertRaises(ValueError):
            document_text(item, document_format="invalid")

    def test_legacy_index_reused_and_changed_format_cannot_silently_reuse_it(self):
        encoded = []
        loaded = []

        class FakeModel:
            prompts = {"query": "Q:", "document": "D:"}

            def __init__(self, _name, **kwargs):
                pass

            def get_embedding_dimension(self):
                return 2

            def encode(self, texts, **kwargs):
                encoded.append((list(texts), kwargs))
                return np.ones((len(texts), 2), dtype=np.float32)

        class FakeIndex:
            def __init__(self, **kwargs):
                pass

            def init_index(self, **kwargs):
                pass

            def add_items(self, vectors, labels):
                pass

            def set_ef(self, value):
                pass

            def save_index(self, path):
                Path(path).write_bytes(b"fake")

            def load_index(self, path, **kwargs):
                loaded.append(path)

        class FakeHnsw:
            Index = FakeIndex

        items = [{"item_id": "a", "item_title_raw": "title", "item_infm_params_text": "вид услуги service",
                  "item_description_raw": "description"}]
        with TemporaryDirectory() as directory, patch("avito_candidates.dense._load_dependencies", return_value=(FakeHnsw, FakeModel)):
            current = DenseRetriever(items, model_name="fake-octen", cache_dir=directory)
            old_metadata = current._metadata()
            self.assertNotIn("document_format", old_metadata)
            self.assertNotIn("document_format_version", old_metadata)
            DenseRetriever(items, model_name="fake-octen", cache_dir=directory, require_cached_index=True)
            self.assertEqual(len(encoded), 1)
            self.assertEqual(len(loaded), 1)
            with self.assertRaises(FileNotFoundError):
                DenseRetriever(items, model_name="fake-octen", cache_dir=directory, document_format="compact", require_cached_index=True)
            self.assertEqual(len(encoded), 1)
            compact = DenseRetriever(items, model_name="fake-octen", cache_dir=directory, document_format="compact")
            self.assertEqual(len(encoded), 2)
            self.assertEqual(encoded[1][0], [document_text(items[0], document_format="compact")])
            self.assertEqual(encoded[1][1]["prompt_name"], "document")
            self.assertEqual(compact._metadata()["document_format"], "compact")
            self.assertEqual(len(list(Path(directory).glob("*.bin"))), 2)


if __name__ == "__main__":
    unittest.main()
