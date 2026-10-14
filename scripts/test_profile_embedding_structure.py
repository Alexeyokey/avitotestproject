"""Structural tests need no encoder weights, tokenizer download or GPU."""
import types
import unittest
import json
import tempfile
from pathlib import Path

from profile_embedding_structure import HashSample, item_layout, query_layout, token_profile, load_texts


class StructureTest(unittest.TestCase):
    def test_current_item_order_and_word_limits(self):
        text, spans = item_layout({"item_title_raw": "Ёлка  Ремонт", "item_category_id": "114",
                                   "item_infm_params_text": "a b c", "item_description_raw": "d e f"},
                                  "current", params_words=2, description_words=1)
        self.assertEqual(text, " заголовок: елка ремонт категория: 114 параметры: a b описание: d")
        self.assertEqual(text[slice(*spans["params"])], "a b")
        self.assertEqual(text[slice(*spans["description"])], "d")

    def test_query_prompt_concatenation_matches_training(self):
        text, spans = query_layout({"search_query": "ремонт", "search_infm_params_text": "тип услуги x",
                                    "search_category": "114"}, "current", "Query:")
        self.assertEqual(text, "Query:запрос: ремонт фильтры: тип услуги x категория: 114")
        self.assertEqual(text[slice(*spans["query"])], "ремонт")

    def test_unknown_category_is_omitted(self):
        self.assertEqual(query_layout({"search_query": "a", "search_category": "0"}, "current", "Q:")[0],
                         "Q:запрос: a")

    def test_plain_query_has_no_field_labels(self):
        self.assertEqual(query_layout({"search_query": "A", "search_infm_params_text": "filter"}, "plain", "Q:")[0], "Q:a")

    def test_offsets_and_special_tokens_use_actual_window(self):
        encoding = types.SimpleNamespace(ids=[9, 1, 2, 3], offsets=[(0, 0), (0, 2), (2, 4), (4, 6)])
        p = token_profile(encoding, {"a": (0, 4), "b": (4, 6), "empty": (6, 6)}, 2)
        self.assertEqual(p["a"], {"full": 2, "kept": 1, "fraction": .5})
        self.assertEqual(p["b"]["kept"], 0)
        self.assertIsNone(p["empty"]["fraction"])
        self.assertTrue(p["truncated"])

    def test_actual_truncation_reserves_terminal_special_token(self):
        full = types.SimpleNamespace(ids=[1, 2, 3, 9], offsets=[(0, 2), (2, 4), (4, 6), (0, 0)])
        retained = types.SimpleNamespace(ids=[1, 2, 9], offsets=[(0, 2), (2, 4), (0, 0)])
        p = token_profile(full, {"content": (0, 6)}, 3, retained)
        self.assertEqual(p["content"], {"full": 3, "kept": 2, "fraction": 2 / 3})

    def test_hash_sample_order_independent_and_deduplicated(self):
        first, second = HashSample(3, "salt"), HashSample(3, "salt")
        for i in range(30):
            first.offer(str(i), {"i": i})
            first.offer(str(i), {"i": i})
        for i in reversed(range(30)):
            second.offer(str(i), {"i": i})
        self.assertEqual(first.result(), second.result())
        self.assertEqual(len(first.result()), 3)

    def test_description_variant_moves_content_before_params(self):
        text, spans = item_layout({"item_title_raw": "t", "item_infm_params_text": "p",
                                   "item_description_raw": "d"}, "description_before_params")
        self.assertLess(spans["description"][0], spans["params"][0])

    def test_named_text_manifest_uses_text_values_not_metadata_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "texts.json"
            path.write_text(json.dumps({"texts": ["a", "b"], "count": 2}))
            self.assertEqual(load_texts(path), {"a", "b"})
            path.write_text(json.dumps({"count": 2}))
            with self.assertRaises(ValueError):
                load_texts(path)


if __name__ == "__main__":
    unittest.main()
