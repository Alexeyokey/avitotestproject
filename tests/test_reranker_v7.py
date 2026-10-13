"""V7 learns fit-only microcategory signals while keeping the V4 prefix exact."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pyarrow.parquet as pq

from avito_candidates import cli
from avito_candidates.core import query_key
from avito_candidates.microcategory import SoftMicrocategoryClassifier
from avito_candidates.reranker import (
    FEATURES, LEGACY_FEATURES, V4_FEATURES, V6_FEATURES, V7_FEATURES,
    FeatureBuilder, Reranker, corpus_fingerprint, fit_model,
    inference_microcategory_classifier, prepare_training_data,
)


def config():
    return {"method": "hybrid", "candidate_k": 2, "bm25_weight": 1., "dense_weight": 1.,
            "title_weight": 0., "params_weight": 0., "description_weight": 1.,
            "stem_title_weight": 0., "stem_params_weight": 0., "rrf_k": 60,
            "location_bonus": .02, "microcategory_candidate_k": 100,
            "microcategory_field": "title", "microcategory_weight": .25}


ITEMS = [dict(item_id=category, item_title_raw="repair", item_microcat_id=category)
         for category in ("a", "b", "unknown")]
DETAILS = {"candidate_pool": ["a", "b", "unknown"],
           "channels": {"description": ["a"], "dense": ["b"], "micro_title": ["b", "unknown"]},
           "channel_scores": {"micro_title": {"b": 4., "unknown": 1.}},
           "rrf_scores": {"a": .01, "b": .02, "unknown": .003}}


class DistributionClassifier:
    def __init__(self, items=ITEMS, probabilities=None):
        self.distribution = probabilities or (("a", .5), ("b", .3), ("c", .1), ("d", .1))
        self.model = SimpleNamespace(classes=np.asarray([category for category, _prob in self.distribution]))
        self.crossfit_models = {}
        self.metadata = {"crossfit_folds": 3,
                         "corpus_fingerprint": corpus_fingerprint(items, feature_version=4)}
        self._prediction_cache = {"prewarmed": ["not-for-serialization"]}
        self.calls = []
    def predict_categories(self, query, *, top_k, exclude_self):
        self.calls.append((top_k, exclude_self))
        self._prediction_cache["large-distribution"] = self.distribution
        return self.distribution[:top_k]


def actual_classifier(items=ITEMS):
    texts = {}
    for index in range(100):
        text = f"разный запрос {index}"
        texts.setdefault(SoftMicrocategoryClassifier.fold(text, 3, 42), text)
    rows = [dict(search_query=text, item_microcat_id=category,
                 search_infm_params_text="", search_category="0")
            for category, text in zip(("a", "b", "c"), (texts[0], texts[1], texts[2]))]
    classifier = SoftMicrocategoryClassifier.fit_rows(rows,
        allowed_fit_texts={row["search_query"] for row in rows},
        corpus_fingerprint=corpus_fingerprint(items, feature_version=4), source_sha256="b" * 64,
        crossfit_folds=3, word_max_features=100, char_max_features=100,
        word_min_df=1, char_min_df=1, max_iter=2)
    return classifier, texts


class TinyScorer:
    n_iter_ = 1
    def fit(self, matrix, labels, sample_weight=None):
        self.seen_shape = matrix.shape
        return self
    def get_params(self):
        return {"max_iter": 1}
    def predict_proba(self, matrix):
        positives = np.clip(matrix[:, V7_FEATURES.index("microcat_classifier_probability")], 0, 1)
        return np.column_stack((1 - positives, positives))


class V7FeatureTest(unittest.TestCase):
    def builder(self, classifier=None, **kwargs):
        return FeatureBuilder(ITEMS, config(), feature_names=V7_FEATURES,
            microcategory_classifier=classifier or DistributionClassifier(), **kwargs)

    def test_v4_prefix_is_exact_and_eight_signals_are_measured(self):
        classifier = DistributionClassifier()
        old = FeatureBuilder(ITEMS, config(), feature_names=V4_FEATURES).transform({}, DETAILS, ["a"])
        new = self.builder(classifier).transform({}, DETAILS, ["a"])
        self.assertEqual(len(V7_FEATURES), len(V4_FEATURES) + 8)
        np.testing.assert_array_equal(new[:, :len(V4_FEATURES)], old)
        self.assertEqual(new[:, V7_FEATURES.index("microcat_classifier_known")].tolist(), [1., 1., 0.])
        self.assertAlmostEqual(new[0, V7_FEATURES.index("microcat_classifier_probability")], .5)
        self.assertEqual(new[:, V7_FEATURES.index("microcat_classifier_rank")].tolist(), [1., .5, 0.])
        self.assertAlmostEqual(new[0, V7_FEATURES.index("microcat_classifier_top3_mass")], .9)
        entropy = new[0, V7_FEATURES.index("microcat_classifier_entropy")]
        self.assertGreater(entropy, 0)
        self.assertLess(entropy, 1)
        self.assertEqual(new[:, V7_FEATURES.index("present_microcategory_channel")].tolist(), [0., 1., 1.])
        self.assertAlmostEqual(new[1, V7_FEATURES.index("rank_microcategory_channel")], 1 / 61)
        self.assertEqual(new[:, V7_FEATURES.index("score_microcategory_channel")].tolist(), [0., 4., 1.])
        self.assertEqual(classifier.calls, [(4, True)])
        self.assertEqual(classifier._prediction_cache, {})

    def test_seen_text_uses_heldout_fold_even_when_prior_leaveout_flag_is_false(self):
        classifier, texts = actual_classifier()
        builder = self.builder(classifier, leave_query_out=False)
        matrix = builder.transform({"search_query": texts[0].upper()}, DETAILS, ["a"])
        # Category a occurs only in this query's own fit records, so its fold
        # model cannot assign a measured class probability to candidate a.
        self.assertEqual(matrix[0, V7_FEATURES.index("microcat_classifier_known")], 0)
        self.assertEqual(matrix[0, V7_FEATURES.index("microcat_classifier_probability")], 0)
        self.assertNotIn(texts[0], classifier.crossfit_models[0].fitted_texts)
        self.assertEqual(classifier._prediction_cache, {})

    def test_known_zero_probability_differs_from_unknown_class(self):
        classifier = DistributionClassifier(probabilities=(("a", 1.), ("b", 0.)))
        matrix = self.builder(classifier).transform({}, DETAILS, ["a"])
        self.assertEqual(matrix[1, V7_FEATURES.index("microcat_classifier_probability")], 0)
        self.assertEqual(matrix[1, V7_FEATURES.index("microcat_classifier_known")], 1)
        self.assertEqual(matrix[2, V7_FEATURES.index("microcat_classifier_known")], 0)
        self.assertEqual(matrix[0, V7_FEATURES.index("microcat_classifier_entropy")], 0)
        self.assertTrue(np.isfinite(matrix).all())

    def test_one_class_entropy_and_empty_pool_are_finite(self):
        classifier = DistributionClassifier(probabilities=(("a", 1.),))
        builder = self.builder(classifier)
        matrix = builder.transform({}, DETAILS, ["a"])
        self.assertEqual(matrix[0, V7_FEATURES.index("microcat_classifier_entropy")], 0)
        empty = builder.transform({}, {"candidate_pool": [], "channels": {}}, [])
        self.assertEqual(empty.shape, (0, len(V7_FEATURES)))
        self.assertEqual(classifier._prediction_cache, {})

    def test_schema_gate_requires_oof_and_correct_corpus(self):
        with self.assertRaisesRegex(ValueError, "V7 requires hybrid"):
            FeatureBuilder(ITEMS, config(), feature_names=V7_FEATURES)
        classifier = DistributionClassifier()
        classifier.metadata["crossfit_folds"] = 0
        with self.assertRaisesRegex(ValueError, "grouped OOF"):
            self.builder(classifier)
        classifier.metadata["crossfit_folds"] = 3
        classifier.metadata["corpus_fingerprint"] = "wrong"
        with self.assertRaisesRegex(ValueError, "different item corpus"):
            self.builder(classifier)

    def test_nonfinite_incomplete_probabilities_fail_and_clear_cache(self):
        for distribution in ((("a", .2), ("b", .3)), (("a", float("nan")),),
                             (("a", 1.1), ("b", -.1)), (("a", .5), ("a", .5))):
            classifier = DistributionClassifier(probabilities=distribution)
            with self.assertRaisesRegex(ValueError, "complete finite"):
                self.builder(classifier).transform({}, DETAILS, ["a"])
            self.assertEqual(classifier._prediction_cache, {})

    def test_micro_triplet_requires_matching_field_and_measured_scores(self):
        variants = (
            {**DETAILS, "channel_scores": {}},
            {**DETAILS, "channel_scores": {"micro_title": {"b": 4., "unknown": float("nan")}}},
            {**DETAILS, "channel_scores": {"micro_title": {"b": 4., "unknown": None}}},
            {**DETAILS, "channels": {**DETAILS["channels"], "micro_params": ["a"]}},
        )
        for details in variants:
            with self.assertRaisesRegex(ValueError, "V7"):
                self.builder().transform({}, details, ["a"])

    def test_old_schemas_ignore_new_classifier_and_signal_maps(self):
        for schema in (LEGACY_FEATURES, V4_FEATURES, FEATURES, V6_FEATURES):
            cfg = {**config(), "rescore_dense_union": True}
            details = {**DETAILS, "full_dense_scores": {"a": .2, "b": .8, "unknown": 0.}}
            classifier = DistributionClassifier()
            before = FeatureBuilder(ITEMS, cfg, feature_names=schema).transform({}, details, ["a"])
            after = FeatureBuilder(ITEMS, cfg, feature_names=schema,
                microcategory_classifier=classifier).transform({}, details, ["a"])
            np.testing.assert_array_equal(before, after)
            self.assertEqual(classifier.calls, [])


class V7PersistenceTest(unittest.TestCase):
    def test_inference_wrapper_shares_weights_but_drops_prediction_cache(self):
        classifier, _texts = actual_classifier()
        classifier._prediction_cache["many-query-tuples"] = (("a", .5),)
        inference = inference_microcategory_classifier(classifier)
        self.assertIs(inference.model, classifier.model)
        self.assertIs(inference.crossfit_models, classifier.crossfit_models)
        self.assertEqual(inference._prediction_cache, {})
        self.assertTrue(classifier._prediction_cache)

    def test_prepared_priors_and_fitted_artifact_roundtrip_preserve_features(self):
        classifier, texts = actual_classifier()
        builder = FeatureBuilder(ITEMS, config(), feature_names=V7_FEATURES,
            microcategory_classifier=classifier, leave_query_out=True)
        class Retrieval:
            def retrieve_batch_with_diagnostics(self, queries):
                return [(["a"], DETAILS) for _query in queries]
        query = {"search_query": texts[0]}
        key = query_key(query)
        cold = {"search_query": "совершенно новая услуга"}
        expected = builder.transform(cold, DETAILS, ["a"])
        with TemporaryDirectory() as directory:
            data, model = Path(directory) / "features.parquet", Path(directory) / "model.joblib"
            metadata = prepare_training_data(data, Retrieval(), builder,
                {key: {"a"}}, {key: query}, split="queries", fraction=.2, seed=42,
                negatives=2, batch_size=1)
            self.assertEqual(metadata["feature_version"], 7)
            self.assertEqual(metadata["training_prior_policy"]["microcategory_classifier"], "grouped-out-of-fold")
            saved_prior = joblib.load(data.with_suffix(".parquet.priors.joblib"))["microcategory_classifier"]
            self.assertEqual(saved_prior._prediction_cache, {})
            np.testing.assert_array_equal(FeatureBuilder(ITEMS, config(), feature_names=V7_FEATURES,
                microcategory_classifier=saved_prior).transform(cold, DETAILS, ["a"]), expected)
            with patch("avito_candidates.reranker.HistGradientBoostingClassifier", return_value=TinyScorer()):
                report = fit_model(data, model, max_iter=1)
            artifact = joblib.load(model)
            self.assertEqual(artifact["priors"]["microcategory_classifier"]._prediction_cache, {})
            reranker = Reranker(model, ITEMS, config(), evaluation_split=("queries", .2, 42))
            np.testing.assert_array_equal(reranker.builder.transform(cold, DETAILS, ["a"]), expected)
            self.assertEqual(artifact["model"].seen_shape[1], len(V7_FEATURES))
            self.assertEqual(report["training_rows"], pq.read_table(data).num_rows)
            artifact["priors"].pop("microcategory_classifier")
            joblib.dump(artifact, model)
            with self.assertRaisesRegex(ValueError, "matching saved OOF"):
                Reranker(model, ITEMS, config())

    def test_cli_rejects_v7_without_active_microcategory_model(self):
        with patch("sys.argv", ["avito", "prepare-reranker", "--data-dir", "unused",
                                 "--split", "queries", "--method", "hybrid",
                                 "--reranker-feature-version", "7"]), patch("sys.stderr"):
            with self.assertRaises(SystemExit) as error:
                cli.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
