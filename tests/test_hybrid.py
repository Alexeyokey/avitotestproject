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


if __name__ == "__main__":
    unittest.main()
