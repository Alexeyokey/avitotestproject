import unittest

from avito_candidates.hybrid import (HybridRetriever, rank_location_bonus_grid,
                                     reciprocal_rank_fusion)


class FakeRetriever:
    def __init__(self, values):
        self.values = values
        self.limits = []

    def retrieve(self, _query, limit=50):
        self.limits.append(limit)
        return self.values[:limit]


class FakeBM25:
    def __init__(self, channels):
        self.channels = channels
        self.limits = []

    def ranked_lists(self, _query, limit):
        self.limits.append(limit)
        return [
            (name, values[:limit], weight)
            for name, values, weight in self.channels
        ]


class HybridTest(unittest.TestCase):
    def test_real_bm25_geo_channel_enters_hybrid_pool(self):
        from avito_candidates.baseline import Baseline

        items = [{"item_id": f"{index:016x}", "item_title_raw": "ремонт",
                  "item_location_id": "local" if index == 4 else "other"}
                 for index in range(5)]
        bm25 = Baseline(items, title_weight=1, params_weight=0,
                        description_weight=0, candidate_k=1,
                        geo_candidate_k=1, channel_quota=0)
        model = HybridRetriever(bm25, FakeRetriever([]), candidate_k=1,
                                channel_quota=0, location_bonus=0)
        _top, details = model.retrieve_with_diagnostics(
            {"search_query": "ремонт", "search_location_id": "local"}, limit=1)
        self.assertEqual(set(details["candidate_pool"]),
                         {items[0]["item_id"], items[4]["item_id"]})
        self.assertEqual(details["geo_locations"], ("local",))
        self.assertIn(items[4]["item_id"], details["channel_scores"]["title_geo"])

    def test_dense_similarity_reaches_hybrid_diagnostics(self):
        class ScoredDense(FakeRetriever):
            def retrieve_batch_with_scores(self, queries, limit=50):
                return [(self.values[:limit], {"b": 0.83}) for _query in queries]

        model = HybridRetriever(FakeBM25([]), ScoredDense(["b"]),
                                candidate_k=1, bm25_weight=0, dense_weight=1)
        _top, details = model.retrieve_batch_with_diagnostics([{"search_query": "тест"}])[0]
        self.assertEqual(details["channel_scores"]["dense"]["b"], 0.83)

    def test_location_sweep_matches_independent_rrf_for_each_bonus(self):
        channels = {"title": ["a", "b", "b", "c"],
                    "params": ["c", "a"], "dense": ["d", "c", "b"]}
        weights = {"title": 0.6, "params": 0.4, "dense": 1.0}
        locations = {"a": "1", "b": "2", "c": "1", "d": "2"}
        bonuses = [0, 0.001, 0.01, 0.033, 0.1]
        sweep = rank_location_bonus_grid(
            channels, weights, locations, "1", bonuses, rank_constant=60, limit=3
        )
        for bonus in bonuses:
            expected = reciprocal_rank_fusion(
                [(ids, weights[name]) for name, ids in channels.items()],
                limit=3, rank_constant=60, channel_quota=0,
                score_boosts={item_id: bonus for item_id in ("a", "c")},
            )
            self.assertEqual(sweep[bonus], expected)

    def test_batched_hybrid_matches_single_query_fusion(self):
        class BatchDense(FakeRetriever):
            def __init__(self, values):
                super().__init__(values)
                self.batch_calls = []

            def retrieve_batch(self, queries, limit=50):
                self.batch_calls.append((len(queries), limit))
                return [self.values[:limit] for _query in queries]

        bm25 = FakeBM25([
            ("title", ["a", "shared"], 2.0),
            ("params", ["b", "shared"], 1.0),
        ])
        dense = BatchDense(["shared", "c"])
        model = HybridRetriever(bm25, dense, candidate_k=10, channel_quota=0)
        queries = [{"search_query": "один"}, {"search_query": "два"}]
        batched = model.retrieve_batch_with_diagnostics(queries, limit=2)
        singles = [model.retrieve_with_diagnostics(query, limit=2) for query in queries]
        self.assertEqual(batched, singles)
        self.assertEqual(dense.batch_calls, [(2, 10)])

    def test_rrf_rewards_results_from_both_channels(self):
        result = reciprocal_rank_fusion([
            (["a", "b", "c"], 1.0),
            (["c", "d", "a"], 1.0),
        ], limit=4, channel_quota=0)
        self.assertEqual(result[:2], ["a", "c"])
        self.assertEqual(set(result), {"a", "b", "c", "d"})

    def test_channel_quota_keeps_top_results(self):
        first = [f"shared-{i}" for i in range(20)]
        second = ["dense-only"] + first
        result = reciprocal_rank_fusion(
            [(first, 2.0), (second, 0.1)], limit=5, channel_quota=1
        )
        self.assertIn("dense-only", result)
        self.assertIn("shared-0", result)

    def test_score_boost_changes_selection_only_within_retrieved_pool(self):
        result = reciprocal_rank_fusion(
            [(["a", "b"], 1.0)], limit=1, channel_quota=0,
            score_boosts={"b": 0.1, "unseen": 100.0},
        )
        self.assertEqual(result, ["b"])

    def test_returned_rrf_scores_match_deduplicated_ranks_and_boosts(self):
        result, scores = reciprocal_rank_fusion(
            [(["a", "a", "b"], 2.0), (["b", "a"], 1.0)],
            limit=1, rank_constant=10, channel_quota=0,
            score_boosts={"b": 0.02, "absent": 99.0}, return_scores=True,
        )
        self.assertEqual(result, ["b"])
        self.assertAlmostEqual(scores["a"], 2 / 11 + 1 / 12)
        self.assertAlmostEqual(scores["b"], 2 / 12 + 1 / 11 + 0.02)
        self.assertNotIn("absent", scores)

    def test_hybrid_requests_extended_candidate_pool(self):
        bm25 = FakeBM25([
            ("title", ["a", "b"], 2.0),
            ("params", ["c"], 1.0),
        ])
        dense = FakeRetriever(["b", "c"])
        model = HybridRetriever(bm25, dense, candidate_k=100, channel_quota=0)
        self.assertEqual(set(model.retrieve({}, limit=3)), {"a", "b", "c"})
        self.assertEqual(bm25.limits, [100])
        self.assertEqual(dense.limits, [100])

    def test_hybrid_fuses_bm25_fields_without_intermediate_merge(self):
        bm25 = FakeBM25([
            ("title", ["shared", "title"], 2.0),
            ("params", ["shared", "params"], 1.0),
            ("description", ["description", "shared"], 0.5),
        ])
        dense = FakeRetriever(["dense", "shared"])
        model = HybridRetriever(bm25, dense, candidate_k=10, channel_quota=0)
        result = model.retrieve({}, limit=5)
        self.assertEqual(result[0], "shared")
        self.assertEqual(
            set(result),
            {"shared", "title", "params", "description", "dense"},
        )

    def test_diagnostics_keep_union_before_top50_cutoff(self):
        bm25 = FakeBM25([
            ("title", ["a", "shared"], 2.0),
            ("params", ["b", "shared"], 1.0),
        ])
        dense = FakeRetriever(["c", "shared"])
        model = HybridRetriever(bm25, dense, candidate_k=10)
        prediction, diagnostics = model.retrieve_with_diagnostics({}, limit=2)
        self.assertEqual(len(prediction), 2)
        self.assertEqual(set(diagnostics["candidate_pool"]), {"a", "b", "c", "shared"})
        self.assertEqual(diagnostics["channels"]["title"], ["a", "shared"])
        self.assertEqual(diagnostics["channels"]["dense"], ["c", "shared"])
        self.assertEqual(diagnostics["prediction_without_quota"], prediction)
        self.assertEqual(len(diagnostics["prediction_with_quota_10"]), 2)
        self.assertEqual(set(diagnostics["rrf_scores"]), set(diagnostics["candidate_pool"]))


if __name__ == "__main__":
    unittest.main()
