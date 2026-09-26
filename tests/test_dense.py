import unittest

from avito_candidates.dense import _fingerprint, item_text


class DenseHelpersTest(unittest.TestCase):
    def test_item_text_balances_all_fields(self):
        self.assertEqual(
            item_text({
                "item_title_raw": "  РЕМОНТ Ёлок ",
                "item_infm_params_text": " выезд срочно недорого ",
                "item_description_raw": " большой опыт работы ",
            }, params_words=2, description_words=2),
            "заголовок: ремонт елок параметры: выезд срочно описание: большой опыт",
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
