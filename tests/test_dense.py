import unittest

from avito_candidates.dense import _fingerprint, item_text


class DenseHelpersTest(unittest.TestCase):
    def test_item_text_uses_title_and_parameters(self):
        self.assertEqual(
            item_text({"item_title_raw": "  РЕМОНТ Ёлок ", "item_infm_params_text": " Выезд  "}),
            "ремонт елок выезд",
        )

    def test_fingerprint_changes_with_corpus_or_model(self):
        items = [{"item_id": "a", "item_title_raw": "ремонт"}]
        original = _fingerprint(items, "model-a")
        self.assertNotEqual(original, _fingerprint(items, "model-b"))
        self.assertNotEqual(original, _fingerprint(
            [{"item_id": "a", "item_title_raw": "монтаж"}], "model-a"
        ))


if __name__ == "__main__":
    unittest.main()
