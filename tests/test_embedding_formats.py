import unittest

from avito_candidates.dense import item_text, query_text
from avito_candidates.embedding_formats import extract_filter_values, item_format, query_format


class FormatTest(unittest.TestCase):
    def test_current_exactly_delegates_saved_format(self):
        row = {"search_query": "Ёлка", "search_category": "114", "search_infm_params_text": "a"}
        self.assertEqual(query_format(row), query_text(row))
        item = {"item_title_raw": "A", "item_category_id": "114"}
        self.assertEqual(item_format(item, "current"), item_text(item))

    def test_long_key_prefix_and_bounds(self):
        values = extract_filter_values("Тип услуги автосервиса Кузовной ремонт и покраска Вид услуги Автосервис, аренда Тип услуги Автосервис Рейтинг пользователя от 4")
        self.assertEqual(values, "кузовной ремонт и покраска ; автосервис, аренда ; автосервис")

    def test_unknown_free_text_is_preserved(self):
        self.assertEqual(extract_filter_values("Бренд оборудования Qwen"), "бренд оборудования qwen")
        self.assertEqual(extract_filter_values("Срочно Вид услуги Ремонт"), "срочно ; ремонт")

    def test_empty_values_unicode_and_duplicate_values(self):
        self.assertEqual(extract_filter_values("Вид услуги Тип услуги Ёлки \t Услуга Елки"), "елки")
        self.assertEqual(extract_filter_values("Вид услуги Тип услуги"), "")
        self.assertEqual(extract_filter_values(None), "")

    def test_nonsemantic_numeric_fields_are_not_added(self):
        self.assertEqual(extract_filter_values("Рейтинг пользователя от 4 Стоимость 500"), "")
        self.assertEqual(extract_filter_values("Вид услуги Ремонт Место оказания услуг улица 7 График работы от 27000 Тип услуги Телефоны"), "ремонт ; телефоны")

    def test_no_category_omits_macro_ids_without_losing_filters(self):
        row = {"search_query": "a", "search_category": "114", "search_infm_params_text": "x"}
        self.assertEqual(query_format(row, "no_category"), "запрос: a фильтры: x")
        self.assertEqual(query_format({**row, "search_category": "0"}, "current"), "запрос: a фильтры: x")

    def test_values_use_query_content_and_semantic_values(self):
        row = {"search_query": "Ремонт", "search_infm_params_text": "Вид услуги Автосервис"}
        self.assertEqual(query_format(row, "values"), "ремонт | автосервис")
        self.assertEqual(query_format({"search_query": "Ремонт"}, "values"), "ремонт")
        self.assertEqual(query_format({"search_query": "|a|", "search_infm_params_text": "|b|"}, "values"), "|a| | |b|")

    def test_compact_reduces_labels_and_preserves_description(self):
        item = {"item_title_raw": "Ремонт ноутбука", "item_infm_params_text": "Вид услуги Компьютерная помощь Место оказания услуг улица 7 Тип услуги Компьютеры",
                "item_description_raw": "Замена матрицы с гарантией", "item_category_id": "114"}
        self.assertEqual(item_format(item), "ремонт ноутбука. компьютерная помощь ; компьютеры. замена матрицы с гарантией")
        self.assertEqual(item_format(item, "compact", params_words=0, description_words=2), "ремонт ноутбука. замена матрицы")

    def test_variant_and_word_limit_guards(self):
        with self.assertRaises(ValueError):
            query_format({}, "unknown")
        with self.assertRaises(ValueError):
            item_format({}, "unknown")
        with self.assertRaises(ValueError):
            item_format({}, params_words=-1)


if __name__ == "__main__":
    unittest.main()
