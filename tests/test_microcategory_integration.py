"""Opt-in retrieval must protect the existing pool and preserve old identities."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from avito_candidates import cli
from avito_candidates.core import held_out, normalize
from avito_candidates.hybrid import HybridRetriever
from avito_candidates.microcategory import SoftMicrocategoryClassifier
from avito_candidates.reranker import retrieval_config


def arguments(**overrides):
    base = dict(command="evaluate", method="hybrid", data_dir=Path("unused"),
        split="queries", seed=42, validation_fraction=.2, train_batch_size=3,
        k1=None, b=None, title_k1=1., title_b=.2, title_weight=0.,
        params_k1=1.2, params_b=.7, params_weight=0., description_k1=1.,
        description_b=.8, description_weight=.5, stem_title_weight=0.,
        stem_params_weight=0., location_bonus=0., bm25_rrf_k=30,
        bm25_channel_quota=0, geo_candidate_k=0, geo_top_locations=3,
        geo_include_related=False, geo_min_history=20, geo_weight=.25,
        candidate_k=449, embedding_model="fake", dense_cache=Path("unused-cache"),
        embedding_batch_size=64, embedding_max_length=128,
        embedding_params_words=40, embedding_description_words=48,
        device="cpu", ef_search=1000, rrf_k=60, bm25_weight=1.,
        dense_weight=1., channel_quota=0)
    base.update(overrides)
    return SimpleNamespace(**base)


class FakeBM25:
    geo_candidate_k = 0
    geo_include_related = False
    def __init__(self, ids):
        self.ids = ids
    def ranked_lists(self, query, limit):
        return [("description", self.ids[:limit], 1.)]


class FakeDense:
    def __init__(self, ids):
        self.ids = ids
    def retrieve(self, query, limit):
        return self.ids[:limit]
    def retrieve_batch(self, queries, limit):
        return [self.ids[:limit] for query in queries]


class FakeMicro:
    top_categories = 3
    def __init__(self, ids):
        self.ids, self.calls, self.batch_calls = ids, [], []
        self.classifier = self
    def predict_categories_batch(self, queries, **kwargs):
        self.batch_calls.append((queries, kwargs))
    def retrieve(self, query, limit, **kwargs):
        self.calls.append((limit, kwargs))
        ids = self.ids[:limit]
        return ids, {item_id: .3 for item_id in ids}, {}


class HybridMicroIntegrationTest(unittest.TestCase):
    def test_adds_only_free_slots_without_renormalizing_base_weights(self):
        base = [f"b{i}" for i in range(997)]
        micro = FakeMicro(["added", "excess"])
        model = HybridRetriever(FakeBM25(base), FakeDense([]), candidate_k=1000,
            microcategory_channel=micro, microcategory_candidate_k=100,
            microcategory_field="title", microcategory_weight=.25)
        _prediction, details = model.retrieve_with_diagnostics({"search_query": "test"})
        self.assertEqual(details["candidate_pool"], [*base, "added"])
        self.assertEqual(micro.calls[0][0], 998)
        self.assertEqual(micro.calls[0][1]["existing_ids"], ())
        self.assertTrue(micro.calls[0][1]["exclude_self"])
        self.assertEqual(details["channel_weights"]["description"], 1.)
        self.assertEqual(details["channel_weights"]["micro_title"], .25)

    def test_excess_base_fails_only_when_micro_is_enabled(self):
        base = [f"b{i}" for i in range(999)]
        ordinary = HybridRetriever(FakeBM25(base), FakeDense([]), candidate_k=1000)
        self.assertEqual(len(ordinary.retrieve_with_diagnostics({})[1]["candidate_pool"]), 999)
        extended = HybridRetriever(FakeBM25(base), FakeDense([]), candidate_k=1000,
            microcategory_channel=FakeMicro(["added"]), microcategory_candidate_k=10)
        with self.assertRaisesRegex(ValueError, "Protected hybrid pool exceeds"):
            extended.retrieve_with_diagnostics({})

    def test_bad_channel_cannot_return_duplicate_items(self):
        for ids in (["base", "base"], ["new", "new"]):
            model = HybridRetriever(FakeBM25(["base"]), FakeDense([]), candidate_k=1,
                microcategory_channel=FakeMicro(ids), microcategory_candidate_k=10)
            with self.assertRaisesRegex(ValueError, "violated"):
                model.retrieve_with_diagnostics({})

    def test_novel_prefix_matches_prototype_when_base_consumes_category_guarantees(self):
        from avito_candidates.baseline import BM25Index
        from avito_candidates.microcategory import MicrocategoryChannel
        from avito_candidates.reranker import corpus_fingerprint
        items = [dict(item_id=f"{i:016x}", item_title_raw="ремонт услуги",
                      item_microcat_id="a" if i < 3 else "b") for i in range(6)]
        ids = [item["item_id"] for item in items]
        class Classifier:
            metadata = {"corpus_fingerprint": corpus_fingerprint(items, feature_version=4)}
            def predict_categories(self, query, *, top_k, exclude_self):
                return (("a", .9), ("b", .1))[:top_k]
        channel = MicrocategoryChannel(Classifier(), items,
            indexes={"title": BM25Index(ids, [item["item_title_raw"] for item in items], k1=1, b=.2)},
            top_categories=2, min_per_category=1)
        query, base = {"search_query": "ремонт"}, [ids[0], ids[3]]
        raw, _scores, _diag = channel.retrieve(query, 998, field="title", exclude_self=True)
        expected = [item_id for item_id in raw if item_id not in base][:2]
        old_pool, _scores, _diag = channel.retrieve(query, 2, field="title",
                                                   existing_ids=base, exclude_self=True)
        self.assertEqual(expected, [ids[1], ids[2]])
        self.assertEqual(old_pool, [ids[1], ids[4]])
        model = HybridRetriever(FakeBM25(base), FakeDense([]), candidate_k=10,
            microcategory_channel=channel, microcategory_candidate_k=2, microcategory_field="title")
        _prediction, details = model.retrieve_with_diagnostics(query)
        self.assertEqual(details["channels"]["micro_title"], expected)
        self.assertEqual(details["candidate_pool"], [*base, *expected])

    def test_batch_warms_classifier_and_matches_single_retrieval(self):
        micro = FakeMicro(["micro"])
        model = HybridRetriever(FakeBM25(["base"]), FakeDense(["dense"]), candidate_k=2,
            microcategory_channel=micro, microcategory_candidate_k=10)
        queries = [{"search_query": "one"}, {"search_query": "two"}]
        batch = model.retrieve_batch_with_diagnostics(queries)
        self.assertEqual(len(micro.batch_calls), 1)
        self.assertTrue(micro.batch_calls[0][1]["exclude_self"])
        self.assertEqual(batch, [model.retrieve_with_diagnostics(query) for query in queries])

    def test_micro_fills_capacity_before_optional_related_cities(self):
        class Related(FakeBM25):
            geo_include_related = True
            geo_candidate_k = 5
            def related_geo_lists_with_scores(self, query, limit):
                self.fail_if_called = True
                return [("description_geo_related", ["related"], 0.)], {
                    "description_geo_related": {"related": 1.}}
        bm25 = Related([f"base{i}" for i in range(997)])
        model = HybridRetriever(bm25, FakeDense([]), candidate_k=1000,
            microcategory_channel=FakeMicro(["micro"]), microcategory_candidate_k=10)
        _prediction, details = model.retrieve_with_diagnostics({})
        self.assertEqual(len(details["candidate_pool"]), 998)
        self.assertIn("micro", details["candidate_pool"])
        self.assertNotIn("related", details["candidate_pool"])
        self.assertFalse(hasattr(bm25, "fail_if_called"))


class CliMicroIntegrationTest(unittest.TestCase):
    def test_off_defaults_leave_legacy_configuration_exactly_unchanged(self):
        old = retrieval_config(arguments())
        new = retrieval_config(arguments(embedding_document_format="current",
            microcategory_model=Path("unused"), microcategory_candidate_k=0,
            microcategory_field="combined", microcategory_weight=.25,
            microcategory_top_categories=3))
        self.assertEqual(old, new)
        self.assertNotIn("embedding_document_format", new)
        self.assertFalse(any(key.startswith("microcategory_") for key in new))

    def test_active_model_and_document_template_bind_retrieval_identity(self):
        with TemporaryDirectory() as directory:
            model = Path(directory) / "model.joblib"
            model.write_bytes(b"first model")
            args = arguments(microcategory_model=model, microcategory_candidate_k=175,
                microcategory_field="title", microcategory_weight=.25,
                microcategory_top_categories=3, embedding_document_format="compact")
            before = retrieval_config(args)
            self.assertEqual(before["embedding_document_format"], "compact")
            self.assertEqual(before["microcategory_selection_policy"], "top998-then-novel-prefix")
            self.assertEqual(before["microcategory_model_sha256"], hashlib.sha256(b"first model").hexdigest())
            model.write_bytes(b"changed model")
            self.assertNotEqual(before, retrieval_config(args))

    def test_fit_allowlist_groups_normalized_texts_and_forbids_pairs(self):
        fit_text = next(f"ремонт {i}" for i in range(100)
                        if not held_out(f"ремонт {i}", .2, 42))
        cold = next(f"ремонт {i}" for i in range(100)
                    if held_out(f"ремонт {i}", .2, 42))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.parquet"
            pd.DataFrame({"search_query": [fit_text, fit_text.upper(), cold]}).to_parquet(path)
            selected = cli.microcategory_fit_texts(path, split="queries", fraction=.2, seed=42, batch_size=1)
            self.assertEqual(selected, {fit_text})
            with self.assertRaises(ValueError):
                cli.microcategory_fit_texts(path, split="pairs", fraction=.2, seed=42)

    def make_dataset(self, directory):
        root = Path(directory)
        first = next(f"ремонт обуви {i}" for i in range(100)
                     if not held_out(f"ремонт обуви {i}", .2, 42))
        second = next(f"ремонт авто {i}" for i in range(100)
                      if not held_out(f"ремонт авто {i}", .2, 42))
        cold = next(f"секретная проверка {i}" for i in range(100)
                    if held_out(f"секретная проверка {i}", .2, 42))
        fields = ("item_id", "item_title_raw", "item_infm_params_text", "item_description_raw",
                  "item_category_id", "item_location_id", "item_microcat_id", "item_price",
                  "item_rating", "item_rating_reviews_count", "item_is_phone_hidden", "item_is_message_forbidden")
        items = [{field: "" for field in fields} for _ in range(3)]
        for item, category, index in zip(items, ("shoes", "auto", "heldout-only"), range(3)):
            item.update(item_id=f"{index:016x}", item_microcat_id=category,
                        item_title_raw="ремонт", item_description_raw="ремонт")
        pd.DataFrame(items).to_parquet(root / "benchmark_items.parquet")
        pd.DataFrame([dict(search_query=text, item_microcat_id=item["item_microcat_id"],
                           search_infm_params_text="", search_category="0")
                      for text, item in zip((first, second, cold), items)]).to_parquet(root / "train.parquet")
        model = root / "soft.joblib"
        with patch("sys.argv", ["avito", "fit-microcategory", "--data-dir", str(root),
                                 "--split", "queries", "--microcategory-model", str(model),
                                 "--output", str(root / "fit.json")]), patch.object(cli, "save_json"):
            cli.main()
        return root, model, items, cold

    def test_fit_command_excludes_cold_labels_and_records_training_policy(self):
        with TemporaryDirectory() as directory:
            root, model, _items, _cold = self.make_dataset(directory)
            metadata = json.loads(model.with_suffix(".joblib.json").read_text())["metadata"]
            self.assertEqual(metadata["fit_split"], "queries")
            self.assertEqual(metadata["validation_fraction"], .2)
            self.assertEqual(metadata["crossfit_folds"], 0)
            self.assertNotIn("heldout-only", metadata["classes"])

    def test_prepare_requires_oof_before_loading_dense_model(self):
        with TemporaryDirectory() as directory:
            root, model, items, _cold = self.make_dataset(directory)
            args = arguments(command="prepare-reranker", data_dir=root,
                microcategory_model=model, microcategory_candidate_k=10,
                microcategory_top_categories=3, microcategory_field="title", microcategory_weight=.25)
            with patch("avito_candidates.dense.DenseRetriever") as dense:
                with self.assertRaisesRegex(ValueError, "grouped OOF"):
                    cli.build_retriever(args, items)
                dense.assert_not_called()

    def test_predict_reuses_saved_fit_split_instead_of_all_train_rows(self):
        with TemporaryDirectory() as directory:
            root, model, items, cold = self.make_dataset(directory)
            args = arguments(command="predict", data_dir=root, split="all", seed=999,
                microcategory_model=model, microcategory_candidate_k=10,
                microcategory_top_categories=3, microcategory_field="title", microcategory_weight=.25)
            with patch("avito_candidates.dense.DenseRetriever") as dense:
                retriever = cli.build_retriever(args, items)
                self.assertNotIn(normalize(cold), retriever.microcategory_channel.classifier.model.fitted_texts)
                self.assertEqual(retriever.microcategory_channel.classifier.metadata["fit_texts_count"], 2)
                self.assertEqual(dense.call_args.kwargs["document_format"], "current")

    def test_changed_source_fails_before_loading_dense_model(self):
        with TemporaryDirectory() as directory:
            root, model, items, _cold = self.make_dataset(directory)
            records = pd.read_parquet(root / "train.parquet")
            records.loc[0, "item_microcat_id"] = "changed"
            records.to_parquet(root / "train.parquet")
            args = arguments(data_dir=root, microcategory_model=model, microcategory_candidate_k=10,
                microcategory_top_categories=3, microcategory_field="title", microcategory_weight=.25)
            with patch("avito_candidates.dense.DenseRetriever") as dense:
                with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                    cli.build_retriever(args, items)
                dense.assert_not_called()

    def test_parser_rejects_pairs_and_nonhybrid_enabled_channel(self):
        variants = (["fit-microcategory", "--microcategory-model", "unused"],
                    ["evaluate", "--method", "dense", "--microcategory-candidate-k", "5",
                     "--microcategory-model", "unused"],
                    ["evaluate", "--method", "hybrid", "--microcategory-candidate-k", "1000",
                     "--microcategory-model", "unused"])
        for variant in variants:
            with patch("sys.argv", ["avito", *variant]), patch("sys.stderr"):
                with self.assertRaises(SystemExit) as error:
                    cli.main()
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
