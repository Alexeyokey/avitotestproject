"""V8 adds measured full-pool cosine to the exact 83-column V7 prefix."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from avito_candidates import cli
from avito_candidates.core import query_key
from avito_candidates.reranker import (
    V7_FEATURES, V8_FEATURES, FeatureBuilder, Reranker, fit_model, prepare_training_data,
)
from test_reranker_v7 import ITEMS, DETAILS, DistributionClassifier, TinyScorer, actual_classifier, config


def measured():
    return {**DETAILS, "full_dense_scores": {"a": .4, "b": .9, "unknown": 0.}}


class V8Test(unittest.TestCase):
    def builder(self):
        return FeatureBuilder(ITEMS, {**config(), "rescore_dense_union": True},
            feature_names=V8_FEATURES, microcategory_classifier=DistributionClassifier())

    def test_v7_prefix_is_identical_and_lexical_only_candidate_gets_real_cosine(self):
        previous = FeatureBuilder(ITEMS, config(), feature_names=V7_FEATURES,
            microcategory_classifier=DistributionClassifier()).transform({}, DETAILS, ["a"])
        new = self.builder().transform({}, measured(), ["a"])
        self.assertEqual(len(V8_FEATURES), 85)
        np.testing.assert_array_equal(previous, new[:, :len(V7_FEATURES)])
        self.assertEqual(new[0, V8_FEATURES.index("score_dense")], 0.)
        self.assertAlmostEqual(new[0, V8_FEATURES.index("full_dense_score")], .4)
        self.assertEqual(new[:, V8_FEATURES.index("full_dense_known")].tolist(), [1., 1., 1.])
        self.assertEqual(new[2, -2:].tolist(), [0., 1.])

    def test_missing_or_invalid_scores_are_rejected_outside_training_sample(self):
        for scores in ({"a": .4}, {"a": .4, "b": .9, "unknown": float("nan")},
                       {"a": .4, "b": .9, "unknown": 1.1}):
            with self.assertRaisesRegex(ValueError, "full dense scores"):
                self.builder().transform({}, {**DETAILS, "full_dense_scores": scores},
                                         ["a"], candidate_ids=["a"])

    def test_v8_requires_rescore_and_empty_pool_has_valid_shape(self):
        with self.assertRaisesRegex(ValueError, "V8 requires"):
            FeatureBuilder(ITEMS, config(), feature_names=V8_FEATURES,
                           microcategory_classifier=DistributionClassifier())
        empty = self.builder().transform({}, {"candidate_pool": [], "channels": {},
                                              "full_dense_scores": {}}, [])
        self.assertEqual(empty.shape, (0, 85))

    def test_prepare_fit_load_preserves_v8_and_saved_classifier(self):
        classifier, texts = actual_classifier()
        cfg = {**config(), "rescore_dense_union": True}
        builder = FeatureBuilder(ITEMS, cfg, feature_names=V8_FEATURES,
            microcategory_classifier=classifier, leave_query_out=True)
        details = measured()
        class Retrieval:
            def retrieve_batch_with_diagnostics(self, queries):
                return [(["a"], details) for query in queries]
        query = {"search_query": texts[0]}
        key = query_key(query)
        cold = {"search_query": "полностью новый контекст"}
        expected = builder.transform(cold, details, ["a"])
        with TemporaryDirectory() as directory:
            data, model = Path(directory) / "features.parquet", Path(directory) / "model.joblib"
            meta = prepare_training_data(data, Retrieval(), builder,
                {key: {"a"}}, {key: query}, split="queries", fraction=.2, seed=42,
                negatives=2, batch_size=1)
            self.assertEqual(meta["feature_version"], 8)
            self.assertEqual(meta["training_prior_policy"]["microcategory_classifier"], "grouped-out-of-fold")
            with patch("avito_candidates.reranker.HistGradientBoostingClassifier", return_value=TinyScorer()):
                fit_model(data, model, max_iter=1)
            saved = joblib.load(model)
            self.assertEqual(saved["priors"]["microcategory_classifier"]._prediction_cache, {})
            restored = Reranker(model, ITEMS, cfg)
            np.testing.assert_array_equal(restored.builder.transform(cold, details, ["a"]), expected)

    def test_cli_rejects_v8_without_full_pool_rescoring(self):
        with patch("sys.argv", ["avito", "prepare-reranker", "--data-dir", "unused",
                                 "--method", "hybrid", "--split", "queries",
                                 "--microcategory-model", "unused", "--microcategory-candidate-k", "100",
                                 "--reranker-feature-version", "8"]), patch("sys.stderr"):
            with self.assertRaises(SystemExit) as error:
                cli.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
