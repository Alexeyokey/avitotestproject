import unittest

from avito_candidates.hybrid import HybridRetriever, reciprocal_rank_fusion


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


if __name__ == "__main__":
    unittest.main()
