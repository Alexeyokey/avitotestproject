"""The soft service classifier must never see the query's own target category."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from avito_candidates.baseline import BM25Index
from avito_candidates.microcategory import (
    MicrocategoryChannel, SoftMicrocategoryClassifier,
    fit_soft_microcategory_classifier, microcategory_query_text,
)
from avito_candidates.reranker import corpus_fingerprint


FINGERPRINT = "a" * 64
SOURCE = "b" * 64


def row(text, category, filters="", macro="0"):
    return dict(search_query=text, item_microcat_id=category,
                search_infm_params_text=filters, search_category=macro)


def fit(rows, *, folds=0):
    return SoftMicrocategoryClassifier.fit_rows(rows,
        allowed_fit_texts={record["search_query"] for record in rows},
        corpus_fingerprint=FINGERPRINT, source_sha256=SOURCE,
        word_min_df=1, char_min_df=1, word_max_features=300,
        char_max_features=300, crossfit_folds=folds, max_iter=8)


class ClassifierTest(unittest.TestCase):
    def test_explicit_allowlist_removes_heldout_targets(self):
        model = SoftMicrocategoryClassifier.fit_rows([
            row("ремонт обуви", "shoes"), row("ремонт машины", "auto"),
            row("секретная проверка", "heldout-only")],
            allowed_fit_texts={"РЕМОНТ ОБУВИ", "ремонт машины"},
            corpus_fingerprint=FINGERPRINT, source_sha256=SOURCE,
            word_min_df=1, char_min_df=1, max_iter=5)
        self.assertNotIn("heldout-only", model.metadata["classes"])
        self.assertEqual(model.metadata["excluded_nonfit_rows"], 1)
        self.assertEqual(model.metadata["fit_texts_count"], 2)
        self.assertNotIn("секретная проверка", model.model.fitted_texts)

    def test_contexts_categories_and_repeated_clicks_share_one_text_weight(self):
        records = [row("ремонт", "shoes", '{"type":"shoe"}')] * 10
        records += [row("ремонт", "auto", '{"type":"car"}'),
                    row("ремонт", "moto", '{"type":"car"}'),
                    row("уборка", "cleaning")]
        model = fit(records)
        self.assertEqual(model.metadata["accepted_fit_rows"], 13)
        self.assertEqual(model.metadata["training_contexts"], 3)
        self.assertEqual(model.metadata["training_examples"], 4)
        self.assertAlmostEqual(model.metadata["text_balanced_total_weight"], 2)

    def test_fitted_text_fails_closed_without_crossfit_even_with_new_filters(self):
        model = fit([row("ремонт обуви", "shoes"), row("ремонт авто", "auto")])
        for exclude in (False, True):
            with self.assertRaisesRegex(ValueError, "held-out cross-fit"):
                model.predict_categories(row("РЕМОНТ ОБУВИ", "ignored", '{"brand":"new"}'),
                                         exclude_self=exclude)

    def test_crossfit_excludes_all_contexts_of_query_text(self):
        texts = {}
        for i in range(100):
            text = f"уникальная услуга {i}"
            texts.setdefault(SoftMicrocategoryClassifier.fold(text, 2, 42), text)
        first, second = texts[0], texts[1]
        model = fit([row(first, "own-only", '{"type":"one"}'),
                     row(first, "own-only", '{"type":"two"}'),
                     row(second, "other")], folds=2)
        result = model.predict_categories(row(first.upper(), "ignored"), exclude_self=True)
        self.assertEqual(result, (("other", 1.0),))
        self.assertNotIn(first, model.crossfit_models[0].fitted_texts)
        self.assertNotIn("own-only", model.crossfit_models[0].classes)
        with self.assertRaises(ValueError):
            model.predict_categories(row(first, "ignored"), exclude_self=False)

    def test_query_values_are_parsed_and_serialization_order_is_stable(self):
        first = row("РЕМОНТ", "not-a-feature", '{"type":["машина","мото"],"a":true}')
        second = row("ремонт", "different-label", '{"a":true,"type":["мото","машина"]}')
        self.assertEqual(microcategory_query_text(first), microcategory_query_text(second))
        features = microcategory_query_text(first)
        self.assertIn("машина", features)
        self.assertIn("true", features)
        self.assertNotIn("not-a-feature", features)
        self.assertNotIn("category_0", features)
        self.assertNotIn("{", features)
        self.assertEqual(microcategory_query_text(row("ремонт", "x", macro="0")),
                         microcategory_query_text(row("ремонт", "x", macro="")))

    def test_batch_matches_single_and_probabilities_cover_multiple_classes(self):
        model = fit([row("ремонт обуви", "shoes"), row("ремонт авто", "auto")])
        queries = [row("ремонтирование ботинок", "ignored"), row("автомобиль", "ignored")]
        batch = model.predict_categories_batch(queries, top_k=2)
        self.assertEqual(batch, [model.predict_categories(query, top_k=2) for query in queries])
        for result in batch:
            self.assertEqual({category for category, _prob in result}, {"shoes", "auto"})
            self.assertAlmostEqual(sum(prob for _category, prob in result), 1, places=6)
        # Batch top-5 precomputation also supplies top-1/3 comparisons without
        # repeating singleton TF-IDF transforms inside retrieval loops.
        model.model.probabilities = lambda documents: self.fail("cached prefixes recomputed")
        self.assertEqual(model.predict_categories(queries[0], top_k=1), batch[0][:1])

    def test_save_load_checks_source_allowlist_corpus_and_bytes(self):
        records = [row("ремонт обуви", "shoes"), row("ремонт авто", "auto")]
        model = fit(records)
        allowed = {record["search_query"] for record in records}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            model.save(path)
            loaded = SoftMicrocategoryClassifier.load(path, allowed_fit_texts=allowed,
                corpus_fingerprint=FINGERPRINT, source_sha256=SOURCE)
            self.assertEqual(model.metadata, loaded.metadata)
            query = row("ремонтирование ботинок", "ignored")
            self.assertEqual(model.predict_categories(query), loaded.predict_categories(query))
            for modifications in ({"allowed_fit_texts": {"wrong"}},
                                  {"corpus_fingerprint": "c" * 64},
                                  {"source_sha256": "d" * 64}):
                settings = dict(allowed_fit_texts=allowed, corpus_fingerprint=FINGERPRINT,
                                source_sha256=SOURCE)
                settings.update(modifications)
                with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                    SoftMicrocategoryClassifier.load(path, **settings)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                model.save(path)
            with path.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                SoftMicrocategoryClassifier.load(path, allowed_fit_texts=allowed,
                    corpus_fingerprint=FINGERPRINT, source_sha256=SOURCE)

    def test_parquet_fit_uses_actual_microcat_field_and_source_hash(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.parquet"
            pd.DataFrame([row("ремонт обуви", "shoes"),
                          row("ремонт авто", "auto"), row("проверка", "secret")]).to_parquet(path)
            model = fit_soft_microcategory_classifier(path,
                allowed_fit_texts={"ремонт обуви", "ремонт авто"},
                corpus_fingerprint=FINGERPRINT, batch_size=1,
                word_min_df=1, char_min_df=1, max_iter=5)
            self.assertEqual(model.metadata["accepted_fit_rows"], 2)
            self.assertEqual(len(model.metadata["source_sha256"]), 64)
            self.assertNotIn("secret", model.metadata["classes"])


class StubClassifier:
    def __init__(self, items):
        self.metadata = {"corpus_fingerprint": corpus_fingerprint(items, feature_version=4)}

    def predict_categories(self, query, *, top_k=3, exclude_self=False):
        return (("shoes", .6), ("auto", .4))[:top_k]


class CountingIndex(BM25Index):
    def __init__(self, ids, texts):
        super().__init__(ids, texts, k1=1, b=.2)
        self.calls = 0

    def _score(self, text):
        self.calls += 1
        self.last_query = text
        return super()._score(text)


class ChannelTest(unittest.TestCase):
    def setUp(self):
        self.items = [
            dict(item_id=f"{i:016x}", item_title_raw=title,
                 item_description_raw=title, item_microcat_id=category,
                 item_location_id=location)
            for i, (title, category, location) in enumerate([
                ("ремонт ремонт ремонт", "wrong", "city"),
                ("ремонт обуви", "shoes", "city"),
                ("ремонт ботинок", "shoes", "other"),
                ("ремонт авто", "auto", "other"),
                ("ремонт автомобилей", "auto", "city"),
                ("мытье обуви", "shoes", "city"),
            ])]
        self.ids = [item["item_id"] for item in self.items]
        self.index = CountingIndex(self.ids, [item["item_title_raw"] for item in self.items])
        self.classifier = StubClassifier(self.items)

    def channel(self, **settings):
        return MicrocategoryChannel(self.classifier, self.items,
            indexes={"title": self.index}, top_categories=2, min_per_category=1, **settings)

    def test_masks_microcategory_before_topk_and_preserves_external_base(self):
        channel = self.channel()
        existing = [self.ids[0]]
        added, _scores, diagnostics = channel.retrieve({"search_query": "ремонт"}, 2,
                                                     existing_ids=existing, field="title")
        self.assertEqual(len(added), 2)
        self.assertEqual({self.items[self.ids.index(item_id)]["item_microcat_id"] for item_id in added},
                         {"shoes", "auto"})
        self.assertNotIn(self.ids[0], added)
        self.assertIn(self.ids[0], [*existing, *added])
        self.assertEqual(diagnostics["base_pool_size"], 1)

    def test_budget_fills_only_unused_slots_and_has_no_repeated_ids(self):
        channel = self.channel(max_pool_size=3)
        existing = [self.ids[0], self.ids[1]]
        added, _scores, _diagnostics = channel.retrieve({"search_query": "ремонт"}, 100,
                                                      existing_ids=existing)
        self.assertEqual(len(set([*existing, *added])), 3)
        self.assertFalse(set(added) & set(existing))
        empty, _scores, _diagnostics = channel.retrieve({"search_query": "ремонт"}, 100,
                                                       existing_ids=[*existing, *added])
        self.assertEqual(empty, [])
        with self.assertRaisesRegex(ValueError, "already exceeds"):
            channel.retrieve({"search_query": "ремонт"}, 2, existing_ids=self.ids[:4])

    def test_all_routes_reuse_one_field_score_pass_and_precomputed_arrays(self):
        channel = self.channel()
        query = {"search_query": "ремонт"}
        routes, _scores, _details = channel.ranked_lists_with_scores(query, 3)
        self.assertEqual(self.index.calls, 1)
        self.assertEqual({name for name, _ids, _weight in routes}, {"micro_title", "micro_combined"})
        arrays = channel._scores(query)
        channel.ranked_lists_with_scores(query, 3, lexical_scores=arrays)
        self.assertEqual(self.index.calls, 2)
        with self.assertRaisesRegex(ValueError, "Precomputed lexical scores"):
            channel.ranked_lists_with_scores(query, 3, lexical_scores={"title": np.asarray([1])})

    def test_parsed_filter_values_can_retrieve_without_query_lexical_match(self):
        added, _scores, _details = self.channel().retrieve(
            {"search_query": "неизвестное", "search_infm_params_text": '{"type":"обуви"}'}, 4)
        self.assertTrue(added)
        self.assertIn(self.ids[1], added)

    def test_lexical_input_matches_shared_prototype_filter_parser(self):
        from avito_candidates.embedding_formats import extract_filter_values
        query = {"search_query": "РЕМОНТ", "search_infm_params_text":
                 "вид услуги ремонт обуви рейтинг пользователя от 4,5"}
        before_classifier_text = microcategory_query_text(query)
        arrays = self.channel()._scores(query)
        expected_query = (query["search_query"] + " " + extract_filter_values(
            query["search_infm_params_text"])).strip()
        self.assertEqual(self.index.last_query, expected_query)
        self.assertNotIn("рейтинг", expected_query)
        self.assertNotIn("4,5", expected_query)
        expected_scores, _matched = self.index._score(expected_query)
        np.testing.assert_array_equal(arrays["title"], expected_scores)
        self.assertEqual(microcategory_query_text(query), before_classifier_text)

    def test_modified_corpus_and_unverified_prediction_fail_closed(self):
        changed = [dict(item) for item in self.items]
        changed[0]["item_microcat_id"] = "modified"
        with self.assertRaisesRegex(ValueError, "corpus differs"):
            MicrocategoryChannel(self.classifier, changed, indexes={"title": self.index})
        with self.assertRaisesRegex(ValueError, "differ from classifier"):
            self.channel().retrieve({"search_query": "ремонт"}, 3,
                                    predicted_categories=(("wrong", 1.0),))


if __name__ == "__main__":
    unittest.main()
