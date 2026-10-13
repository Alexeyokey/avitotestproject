"""Composite production fill must match the frozen identity-only evaluator."""
import unittest
from unittest.mock import patch

import numpy as np

from avito_candidates import cli
from avito_candidates.baseline import BM25Index
from avito_candidates.hybrid import HybridRetriever
from avito_candidates.reranker import corpus_fingerprint
from avito_candidates.supplemental import ROUTES, select_supplemental, SupplementalFill
from evaluate_retrieval_hypotheses import ordered_novel


def raw_scores(routes):
    return {name: {item_id: .5 for item_id in ids} for name, ids in routes.items()}


class CompositeSelectionTest(unittest.TestCase):
    def test_exact_evaluator_order_including_protected_raw_rank_gaps(self):
        routes = dict(zip(ROUTES, (["base", "b", "x"], ["x", "base", "a"],
                                  ["b", "a"], ["base", "a", "x"])))
        expected = ordered_novel(routes, dict.fromkeys(ROUTES, 1), {"base"})
        selected, retained, scores, selection_scores = select_supplemental(
            routes, raw_scores(routes), ["base"])
        self.assertEqual(selected, expected)
        self.assertEqual(retained[ROUTES[0]], ["b", "x"])
        for name, ids in retained.items():
            self.assertEqual(set(scores[name]), set(ids))
        self.assertEqual(set(selection_scores), set(selected))

    def test_id_ties_and_capacity_keep_all_protected_geo_ids(self):
        protected = [f"protected{i}" for i in range(996)]
        routes = dict(zip(ROUTES, (["z"], ["a"], ["b"], ["c"])))
        selected, _retained, _scores, _selection = select_supplemental(routes, raw_scores(routes), protected)
        self.assertEqual(selected, ["a", "b"])
        self.assertEqual(len(set([*protected, *selected])), 998)
        self.assertEqual(select_supplemental(routes, raw_scores(routes), protected + ["x", "y"])[0], [])

    def test_bad_raw_routes_or_scores_fail_closed(self):
        routes = dict(zip(ROUTES, (["a"], ["b"], ["c"], ["d"])))
        scores = raw_scores(routes)
        for changed_routes, changed_scores in (
            ({**routes, ROUTES[0]: ["a", "a"]}, scores),
            (routes, {**scores, ROUTES[0]: {}}),
            (routes, {**scores, ROUTES[0]: {"a": float("nan")}}),
        ):
            with self.assertRaises(ValueError):
                select_supplemental(changed_routes, changed_scores, [])
        with self.assertRaises(ValueError):
            select_supplemental(routes, scores, list(range(999)))


class Classifier:
    def __init__(self, items):
        self.metadata = {"corpus_fingerprint": corpus_fingerprint(items, feature_version=4)}
    def predict_categories(self, query, *, top_k, exclude_self):
        return (("a", .6), ("b", .3), ("c", .1))[:top_k]
    def predict_categories_batch(self, queries, **kwargs):
        return [self.predict_categories(query, **kwargs) for query in queries]


class Dense:
    document_format = "current"
    def __init__(self, ids):
        self.ids, self.calls = ids, []
    def retrieve_batch_with_scores(self, queries, limit):
        self.calls.append(limit)
        # A larger request deliberately changes its first ID. Prefixing this
        # result would delete a valid protected base candidate.
        ids = [self.ids[0]] if limit != 998 else list(reversed(self.ids))
        return [(ids, {item_id: .5 for item_id in ids}) for query in queries]
    def retrieve_with_scores(self, query, limit):
        return self.retrieve_batch_with_scores([query], limit)[0]


class Base:
    geo_candidate_k = 1
    geo_include_related = False
    def __init__(self, ids):
        self.ids = ids
    def ranked_lists_with_scores(self, query, limit):
        return [("description", [self.ids[1]], 1.), ("description_geo", [self.ids[2]], .25)], {
            "description": {self.ids[1]: 1.}, "description_geo": {self.ids[2]: 1.}}
    def geo_locations(self, query):
        return ("exact-city",)


class CompositeIntegrationTest(unittest.TestCase):
    def setup_components(self):
        from avito_candidates.microcategory import MicrocategoryChannel
        items = [dict(item_id=f"{i:016x}", item_title_raw="ремонт обуви",
                      item_infm_params_text="ремонт", item_description_raw="ремонт",
                      item_microcat_id=("a", "b", "c")[i % 3]) for i in range(10)]
        ids = [item["item_id"] for item in items]
        indexes = {name: BM25Index(ids, [item.get(field, "") for item in items], k1=1, b=.2)
                   for name, field in (("title", "item_title_raw"), ("params", "item_infm_params_text"),
                                       ("description", "item_description_raw"))}
        micro = MicrocategoryChannel(Classifier(items), items, indexes=indexes, top_categories=3)
        dense = Dense(ids)
        return ids, micro, dense

    def test_batch_preserves_separate_native_base_and_deep_requests(self):
        ids, micro, dense = self.setup_components()
        model = HybridRetriever(Base(ids), dense, candidate_k=449,
            microcategory_channel=micro, microcategory_candidate_k=998,
            microcategory_field="combined", supplemental_fill="micro-typed-dense-v1")
        query = {"search_query": "ремонт"}
        _prediction, details = model.retrieve_batch_with_diagnostics([query])[0]
        self.assertEqual(dense.calls, [449, 998])
        self.assertEqual(details["channels"]["dense"], [ids[0]])
        self.assertEqual(details["candidate_pool"][:3], [ids[1], ids[2], ids[0]])
        self.assertEqual(set(details["candidate_pool"]), set(ids))
        self.assertEqual(details["geo_locations"], ("exact-city",))
        for name, route in details["channels"].items():
            self.assertEqual(set(details["channel_scores"][name]), set(route))
        self.assertIn("micro_combined", details["channels"])
        # Independent retrieval of the raw routes and the pure evaluator must
        # produce precisely the same appended candidate order.
        selected = SupplementalFill(micro, dense).retrieve(query, ids[:3],
            dense_result=(list(reversed(ids)), {item_id: .5 for item_id in ids}))[0]
        self.assertEqual(details["candidate_pool"][3:], selected)

    def test_mode_guards_reject_wrong_micro_policy_or_document_format(self):
        ids, micro, dense = self.setup_components()
        with self.assertRaises(ValueError):
            HybridRetriever(Base(ids), dense, microcategory_channel=micro,
                microcategory_candidate_k=100, supplemental_fill="micro-typed-dense-v1")
        dense.document_format = "compact"
        with self.assertRaises(ValueError):
            SupplementalFill(micro, dense)

    def test_cli_guard_requires_explicit_composite_micro_settings(self):
        with patch("sys.argv", ["avito", "evaluate", "--data-dir", "unused", "--method", "hybrid",
                                 "--supplemental-fill", "micro-typed-dense-v1"]), patch("sys.stderr"):
            with self.assertRaises(SystemExit) as error:
                cli.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
