"""New priors must use fit rows only and keep global retrieval available."""

from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from avito_candidates.core import held_out, normalize
from avito_candidates.baseline import Baseline
from avito_candidates.geography import LocationAssociations
from avito_candidates.hybrid import HybridRetriever
from avito_candidates.microcategory import MicrocategoryPrior, fit_microcategory_prior


class NewSignalsTest(unittest.TestCase):
    def test_microcategory_prior_excludes_heldout_query_text(self):
        fit_text = next(f"ремонт обуви {i}" for i in range(1000)
                        if not held_out(normalize(f"ремонт обуви {i}"), 0.2, 42))
        heldout_text = next(f"секретная услуга {i}" for i in range(1000)
                            if held_out(normalize(f"секретная услуга {i}"), 0.2, 42))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.parquet"
            pd.DataFrame([
                {"search_query": fit_text, "item_microcat_id": "shoes"},
                {"search_query": heldout_text, "item_microcat_id": "secret"},
            ]).to_parquet(path)
            prior = fit_microcategory_prior(path, split="queries", fraction=0.2,
                                            seed=42, batch_size=1)
        self.assertEqual(prior.features(fit_text, "shoes")[0], 1.0)
        self.assertEqual(prior.features(fit_text, "shoes", exclude_self=True),
                         (0.0, 0.0, 0.0))
        self.assertEqual(prior.features(heldout_text, "secret"), (0.0, 0.0, 0.0))

    def test_token_prior_removes_the_training_texts_own_category(self):
        prior = MicrocategoryPrior(
            {"ремонт обуви": Counter({"shoes": 1})},
            {"ремонт": Counter({"shoes": 1.0, "other": 5.0})},
            Counter({"ремонт": 6}),
        )
        self.assertGreater(prior.features("ремонт", "shoes")[1], 0)
        self.assertEqual(prior.features("ремонт обуви", "shoes",
                                        exclude_self=True)[1], 0)

    def test_related_locations_expand_exact_city_without_removing_it(self):
        counts = defaultdict(Counter, {
            "city": Counter({"neighbor": 20, "city": 100, "far": 10}),
            "region": Counter({"neighbor": 20, "far": 10}),
        })
        associations = LocationAssociations(
            {"city", "neighbor", "far"}, counts=counts,
            totals=Counter({"city": 130, "region": 30}),
        )
        self.assertEqual(associations.destinations("city", max_locations=2), ("city",))
        self.assertEqual(associations.destinations(
            "city", max_locations=2, include_related=True), ("city", "neighbor"))
        self.assertEqual(associations.destinations(
            "region", max_locations=2, include_related=True), ("neighbor", "far"))

        items = [
            {"item_id": "exact", "item_location_id": "city",
             "item_title_raw": "", "item_infm_params_text": "",
             "item_description_raw": "ремонт"},
            {"item_id": "related", "item_location_id": "neighbor",
             "item_title_raw": "", "item_infm_params_text": "",
             "item_description_raw": "ремонт"},
        ]
        bm25 = Baseline(items, title_weight=0, params_weight=0,
                        description_weight=1, geo_candidate_k=1,
                        geo_top_locations=2, geo_include_related=True,
                        geo_associations=associations)
        query = {"search_query": "ремонт", "search_location_id": "city"}
        self.assertEqual(bm25.geo_locations(query), ("city",))
        self.assertEqual(bm25.related_geo_locations(query), ("neighbor",))
        related, _scores = bm25.related_geo_lists_with_scores(query, 1)
        self.assertEqual(related[0][1], ["related"])

    def test_related_geo_uses_only_free_pool_places(self):
        class BM25:
            geo_include_related = True
            geo_candidate_k = 100

            def ranked_lists_with_scores(self, query, limit):
                return [
                    ("description", [f"i{i}" for i in range(449)], 1.0),
                    ("description_geo", [f"i{i}" for i in range(449, 549)], 0.25),
                ], {}

            def related_geo_lists_with_scores(self, query, limit):
                return [("description_geo_related", ["extra"], 0.0)], {
                    "description_geo_related": {"extra": 1.0}}

            def geo_locations(self, query):
                return ("city",)

        class Dense:
            def __init__(self, start):
                self.start = start

            def retrieve_with_scores(self, query, limit):
                return [f"i{i}" for i in range(self.start, self.start + 449)], {}

        query = {"search_query": "ремонт"}
        full = HybridRetriever(BM25(), Dense(549), candidate_k=449)
        _ranking, full_details = full.retrieve_with_diagnostics(query)
        self.assertEqual(len(full_details["candidate_pool"]), 998)
        self.assertNotIn("extra", full_details["candidate_pool"])

        overlap = HybridRetriever(BM25(), Dense(548), candidate_k=449)
        _ranking, overlap_details = overlap.retrieve_with_diagnostics(query)
        self.assertEqual(len(overlap_details["candidate_pool"]), 998)
        self.assertIn("extra", overlap_details["candidate_pool"])
        self.assertTrue(set(full_details["candidate_pool"]) - {"i997"}
                        <= set(overlap_details["candidate_pool"]))


if __name__ == "__main__":
    unittest.main()
