"""Geography must affect retrieval before topK without changing cosine features."""

from collections import Counter
import unittest

import numpy as np

from avito_candidates.geography import LocationAssociations
from avito_candidates.geo_dense import GeoDensePrior, ExactDenseScorer, catalog_location_centers


def item(identifier, location, lat=55.0, lon=37.0):
    return {"item_id": identifier, "item_location_id": location,
            "item_latitude": lat, "item_longitude": lon}


def associations(counts, query_counts=None):
    return LocationAssociations(
        {location for destinations in counts.values() for location in destinations},
        counts={location: Counter(destinations) for location, destinations in counts.items()},
        totals=Counter({location: sum(destinations.values()) for location, destinations in counts.items()}),
        query_counts=query_counts,
    )


class GeoDensePriorTest(unittest.TestCase):
    def test_local_item_outside_semantic_topk_is_recovered_by_global_selection(self):
        items = [item("foreign0", "far", 60, 60), item("foreign1", "far", 60, 60),
                 item("foreign2", "far", 60, 60), item("local", "home")]
        geo = GeoDensePrior(items, associations({"home": {"home": 99, "far": 1}}))
        cosines = np.array([0.9, 0.85, 0.8, 0.7], dtype=np.float32)
        self.assertEqual(geo.retrieve(cosines, "home", limit=2, weight=0)[0], ["foreign0", "foreign1"])
        ids, scores = geo.retrieve(cosines, "home", limit=2, weight=0.1)
        self.assertEqual(ids, ["local", "foreign0"])
        self.assertAlmostEqual(scores["local"], 0.7, places=6)
        np.testing.assert_array_equal(cosines, np.array([0.9, 0.85, 0.8, 0.7], dtype=np.float32))

    def test_no_fit_history_falls_back_even_for_known_catalog_city(self):
        items = [item("a", "known"), item("b", "elsewhere")]
        geo = GeoDensePrior(items, associations({"another": {"known": 1}}))
        for location in ("unknown", "known", "", None):
            np.testing.assert_array_equal(geo.log_prior(location), [0, 0])
            self.assertEqual(geo.retrieve([0.1, 0.9], location, 1, 0.1)[0], ["b"])

    def test_query_text_exclusion_removes_own_fit_evidence(self):
        history = associations({"region": {"a": 100, "b": 10}},
                               {"own text": {"region": Counter({"a": 100})}})
        geo = GeoDensePrior([item("ia", "a"), item("ib", "b")], history)
        self.assertEqual(geo.retrieve([0.5, 0.5], "region", 1, 0.05)[0], ["ia"])
        self.assertEqual(geo.retrieve([0.5, 0.5], "region", 1, 0.05,
                                     exclude_query_text="OWN TEXT")[0], ["ib"])
        self.assertEqual(history.history("region")[1], 110)

    def test_prior_reads_only_provided_fit_associations(self):
        items = [item("a", "fit_city"), item("b", "heldout_city")]
        fit_only = associations({"region": {"fit_city": 10}})
        geo = GeoDensePrior(items, fit_only)
        ids, _ = geo.retrieve([0.5, 0.5], "region", 1, 0.1)
        self.assertEqual(ids, ["a"])
        self.assertNotIn("heldout_city", fit_only.counts["region"])
        self.assertFalse(geo.log_prior("region").flags.writeable)

    def test_distance_smoothing_prefers_near_unobserved_destination(self):
        items = [item("home", "home"), item("near", "near", 55.01, 37.01),
                 item("far", "far", 60, 60)]
        geo = GeoDensePrior(items, associations({"home": {"home": 1}}))
        prior = geo.log_prior("home")
        self.assertGreater(prior[1], prior[2])
        self.assertTrue(np.isfinite(prior).all())

    def test_region_center_is_fit_weighted_and_missing_coordinates_are_finite(self):
        items = [item("a", "a"), item("b", "b", 60, 60),
                 {"item_id": "missing", "item_location_id": ""}]
        geo = GeoDensePrior(items, associations({"region": {"a": 10}}))
        prior = geo.log_prior("region")
        self.assertGreater(prior[0], prior[1])
        self.assertTrue(np.isfinite(prior).all())

    def test_ties_at_cutoff_use_item_id_not_corpus_order(self):
        geo = GeoDensePrior([item("z", "x"), item("a", "x"), item("b", "x")], associations({}))
        self.assertEqual(geo.retrieve([0.5, 0.5, 0.5], "", 2, 0)[0], ["a", "b"])
        self.assertEqual(geo.retrieve([0.5, 0.5, 0.5], "", 99, 0)[0], ["a", "b", "z"])
        self.assertEqual(geo.retrieve([0.5, 0.5, 0.5], "", 0, 0)[0], [])

    def test_rejects_incomplete_corpus_scores_and_nonfinite_data(self):
        geo = GeoDensePrior([item("a", "x"), item("b", "y")], associations({}))
        for scores in ([0.5], [0.5, float("nan")], [1.2, 0.1]):
            with self.assertRaises(ValueError):
                geo.rank(scores, "x")
        for weight in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                geo.rank([0.1, 0.2], "x", weight=weight)
        with self.assertRaises(ValueError):
            geo.rank([0.1, 0.2], "x", limit=-1)
        with self.assertRaises(ValueError):
            GeoDensePrior([item("a", "x"), item("a", "y")], associations({}))

    def test_inconsistent_fit_counts_fail_instead_of_silent_prior_corruption(self):
        history = associations({"x": {"x": 10}})
        history.totals["x"] = 100
        geo = GeoDensePrior([item("a", "x")], history)
        with self.assertRaisesRegex(ValueError, "disagree"):
            geo.log_prior("x")

    def test_catalog_median_discards_bad_coordinates(self):
        centers = catalog_location_centers([item("a", "x", 10, 20), item("b", "x", 30, 40),
                                            item("c", "x", float("nan"), 40), item("d", "x", 100, 40)])
        np.testing.assert_allclose(centers["x"], [20, 30])


class ExactDenseScorerTest(unittest.TestCase):
    def test_saved_unit_vectors_share_memory_and_keep_original_values(self):
        corpus = np.array([[1.00001, 0], [0, 0.99999]], dtype=np.float32)
        corpus.setflags(write=False)
        scorer = ExactDenseScorer(corpus, item_block_size=1)
        self.assertTrue(np.shares_memory(corpus, scorer.corpus))
        np.testing.assert_array_equal(scorer.corpus, corpus)
        np.testing.assert_array_equal(scorer._norms, [1, 1])

    def test_all_blocks_are_scored_and_last_item_can_win(self):
        corpus = np.array([[2, 0], [3, 0], [1, 1], [0, 7]], dtype=np.float32)
        scorer = ExactDenseScorer(corpus, item_block_size=2)
        scores = scorer.score([[0, 5], [3, 0]])
        np.testing.assert_allclose(scores, [[0, 0, 1 / np.sqrt(2), 1],
                                            [1, 1, 1 / np.sqrt(2), 0]], atol=1e-6)
        self.assertEqual(int(np.argmax(scores[0])), 3)
        np.testing.assert_array_equal(corpus, [[2, 0], [3, 0], [1, 1], [0, 7]])

    def test_query_batch_and_empty_shapes(self):
        scorer = ExactDenseScorer([[1, 0]])
        self.assertEqual(scorer.score(np.empty((0, 2))).shape, (0, 1))
        empty = ExactDenseScorer(np.empty((0, 2)))
        self.assertEqual(empty.score([[1, 0]]).shape, (1, 0))
        for vectors in ([[0, 0]], [[float("nan"), 1]], [[1, 0, 0]]):
            with self.assertRaises(ValueError):
                scorer.score(vectors)

    def test_zero_and_nonfinite_corpus_rejected(self):
        for vectors in ([[0, 0]], [[float("inf"), 1]], [1, 2]):
            with self.assertRaises(ValueError):
                ExactDenseScorer(vectors)


if __name__ == "__main__":
    unittest.main()
