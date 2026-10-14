"""Focused safeguards for the new, frozen-model dense retrieval experiment."""
import copy
from types import SimpleNamespace
import unittest

import numpy as np

from avito_candidates.reranker import FeatureBuilder, V8_FEATURES, corpus_fingerprint
from evaluate_geo_dense_tune import ann_result, metric_row, shared_variant_matrices


class ConstantClassifier:
    def __init__(self, items):
        self.metadata = {"crossfit_folds": 3,
                         "corpus_fingerprint": corpus_fingerprint(items, feature_version=4)}
        self.model = SimpleNamespace(classes=["x", "y"])
        self._prediction_cache = {}

    def predict_categories(self, query, top_k, exclude_self):
        return [("x", .7), ("y", .3)]


class GeoTuneTest(unittest.TestCase):
    def fixture(self):
        items = [{"item_id": name, "item_title_raw": "ремонт " + name,
                  "item_infm_params_text": "марка kia", "item_description_raw": "ремонт авто быстро",
                  "item_category_id": "114", "item_location_id": "loc" if name in "ac" else "other",
                  "item_microcat_id": "x" if name in "ab" else "y"}
                 for name in "abcde"]
        config = {"method": "hybrid", "microcategory_candidate_k": 998,
                  "rescore_dense_union": True, "rrf_k": 60, "bm25_weight": 1., "dense_weight": 1.,
                  "title_weight": 0., "params_weight": 0., "description_weight": .5,
                  "stem_title_weight": 0., "stem_params_weight": 0., "microcategory_field": "combined"}
        builder = FeatureBuilder(items, config, feature_names=V8_FEATURES,
                                 microcategory_classifier=ConstantClassifier(items))
        query = {"search_query": "ремонт", "search_location_id": "loc", "search_category": "114"}
        cos = dict(zip("abcde", [.8, .7, .6, .5, .4]))
        def result(pool, dense, micro, baseline):
            channels = {"description": ["a", "b"], "description_geo": ["a"],
                        "dense": dense, "micro_combined": micro,
                        "supplemental_fill": micro, "supplemental_dense_current": micro[:1]}
            scores = {"description": {"a": 4., "b": 2.}, "description_geo": {"a": 4.},
                      "dense": {item: cos[item] for item in dense},
                      "micro_combined": {item: .5 for item in micro},
                      "supplemental_fill": {item: .02 for item in micro},
                      "supplemental_dense_current": {item: cos[item] for item in micro[:1]}}
            return baseline, {"candidate_pool": pool, "channels": channels, "channel_scores": scores,
                "geo_locations": ["loc"], "rrf_scores": {item: .1 / (i+1) for i, item in enumerate(pool)},
                "full_dense_scores": {item: cos[item] for item in pool}}
        results = {"one": result(list("abcd"), list("bc"), ["d"], ["a", "c"]),
                   "two": result(list("abed"), ["e", "a"], ["d", "b"], ["e", "b"])}
        return builder, query, results

    def test_shared_features_are_bit_identical_to_full_production_transform(self):
        builder, query, results = self.fixture()
        matrices, difference = shared_variant_matrices(builder, query, results, check_parity=True)
        self.assertEqual(difference, 0.)
        for name, (baseline, details) in results.items():
            np.testing.assert_array_equal(matrices[name], builder.transform(query, details, baseline))
        self.assertNotEqual(matrices["one"][1, V8_FEATURES.index("rank_dense")],
                            matrices["two"][1, V8_FEATURES.index("rank_dense")])
        # Geo-selected item e keeps raw measured cosine .4, not a log-prior score.
        self.assertAlmostEqual(matrices["two"][2, V8_FEATURES.index("score_dense")], .4)

    def test_shared_features_reject_changed_lexical_search(self):
        builder, query, results = self.fixture()
        results["two"][1]["channels"]["description"] = ["b", "a"]
        with self.assertRaisesRegex(ValueError, "fixed lexical"):
            shared_variant_matrices(builder, query, results)

    def test_shared_features_reject_changed_geo_metadata(self):
        builder, query, results = self.fixture()
        results["two"][1]["geo_locations"] = ["other"]
        with self.assertRaisesRegex(ValueError, "geographic lexical"):
            shared_variant_matrices(builder, query, results)

    def test_ann_control_rejects_invalid_labels_and_nonfinite_scores(self):
        for labels, scores in (([1, 1], [.4, .5]), ([-1], [.5]), ([2], [.5]), ([0], [np.nan])):
            with self.assertRaises(ValueError):
                ann_result(labels, scores, ["a", "b"])
        self.assertEqual(ann_result([1, 0], [.7, .3], ["a", "b"]),
                         (["b", "a"], {"b": .7, "a": .3}))

    def test_metric_keeps_complete_denominator_when_positive_is_outside_pool(self):
        _, _, results = self.fixture()
        group = {"group_identity": "q", "query_text": "ремонт", "available_relevant_count": 2,
                 "full_relevant_count": 3, "representative": True}
        row = metric_row(group, results["one"], {"a", "z"}, ["a"])
        self.assertEqual(row["available_relevant_count"], 2)
        self.assertEqual(row["full_relevant_count"], 3)
        self.assertEqual((row["pool_hits"], row["s100_hits"]), (1, 1))

    def test_zero_availability_remains_in_full_truth(self):
        _, _, results = self.fixture()
        group = {"group_identity": "q", "query_text": "ремонт", "available_relevant_count": 0,
                 "full_relevant_count": 2, "representative": False}
        row = metric_row(group, results["one"], set(), [])
        self.assertEqual((row["pool_hits"], row["baseline_hits"], row["s100_hits"]), (0, 0, 0))
        self.assertEqual(row["full_relevant_count"], 2)
        self.assertEqual(row["pool_size"], 4)


if __name__ == "__main__":
    unittest.main()
