"""Mock-only tests for optional union-candidate cosine features."""

import unittest

import numpy as np

from avito_candidates.dense import DenseRetriever, query_text
from avito_candidates.hybrid import HybridRetriever


class QueryEncoder:
    prompts = {"query": "search prompt: ", "document": "document prompt: "}

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        if kwargs.get("prompt_name") != "query":
            raise AssertionError("Rescoring must never encode corpus/document text")
        return np.array([[0.0, 1.0] if "второй" in text else [1.0, 0.0]
                         for text in texts], dtype=np.float32)


class ExistingIndex:
    vectors = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype=np.float32)

    def __init__(self):
        self.item_reads = []
        self.search_calls = 0

    def get_items(self, labels):
        self.item_reads.append(list(labels))
        return self.vectors[labels]

    def add_items(self, *_args, **_kwargs):
        raise AssertionError("Rescoring must not rebuild or add to the index")

    def set_ef(self, _value):
        pass

    def knn_query(self, vectors, k):
        self.search_calls += 1
        scores = vectors @ self.vectors.T
        labels = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        distances = 1 - np.take_along_axis(scores, labels, axis=1)
        return labels, distances


def dense_retriever():
    retriever = object.__new__(DenseRetriever)
    retriever.ids = ["dense-only", "bm25-only", "second-query"]
    retriever.dimension = 2
    retriever.batch_size = 2
    retriever.model_name = "/models/octen-local-adapter"
    retriever.model = QueryEncoder()
    retriever.index = ExistingIndex()
    retriever.ef_search = 10
    return retriever


class BM25OnlyCandidate:
    geo_candidate_k = 0

    def ranked_lists_with_scores(self, _query, _limit):
        return [("description", ["bm25-only"], 1.0)], {"description": {"bm25-only": 7.0}}


class DenseCandidateScoringTests(unittest.TestCase):
    def test_reads_only_supplied_vectors_and_uses_saved_octen_query_prompt(self):
        dense = dense_retriever()
        query = {"search_query": "первый", "search_infm_params_text": "фильтр", "search_category": "12"}
        result = dense.score_candidates_batch([query], [["bm25-only", "dense-only"]])
        self.assertAlmostEqual(result[0]["bm25-only"], 0.6, places=6)
        self.assertEqual(result[0]["dense-only"], 1.0)
        self.assertEqual(dense.index.item_reads, [[1, 0]])
        self.assertEqual(dense.index.search_calls, 0)
        texts, kwargs = dense.model.calls[0]
        self.assertEqual(texts, [query_text(query)])
        self.assertEqual(kwargs["prompt_name"], "query")
        self.assertTrue(kwargs["normalize_embeddings"])

    def test_batches_skip_empty_pools_and_preserve_order(self):
        dense = dense_retriever()
        queries = [{"search_query": "первый"}, {}, {"search_query": "второй"},
                   {"search_query": "первый снова"}]
        scores = dense.score_candidates_batch(
            queries, [["bm25-only"], [], ["bm25-only"], ["dense-only"]])
        self.assertEqual(scores[1], {})
        self.assertAlmostEqual(scores[0]["bm25-only"], 0.6, places=6)
        self.assertAlmostEqual(scores[2]["bm25-only"], 0.8, places=6)
        self.assertEqual(scores[3]["dense-only"], 1.0)
        self.assertEqual([len(texts) for texts, _kwargs in dense.model.calls], [2, 1])
        dense.score_candidates_batch([queries[0]], [["dense-only"]])
        self.assertEqual(len(dense._id_to_label), 3)

    def test_bad_pool_or_batch_is_rejected_before_encoding(self):
        dense = dense_retriever()
        for queries, pools, message in (
            ([{}], [], "batch sizes"),
            ([{}], [["unknown"]], "unknown item ID"),
            ([{}], [[None]], "unknown item ID"),
            ([{}], [["dense-only", "dense-only"]], "duplicate"),
            ([{}], [["dense-only"] * 999], "exceeds"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                dense.score_candidates_batch(queries, pools)
        self.assertEqual(dense.model.calls, [])
        self.assertEqual(dense.index.item_reads, [])

    def test_empty_pool_requires_no_encode_and_invalid_vectors_are_rejected(self):
        dense = dense_retriever()
        self.assertEqual(dense.score_candidates_batch([{}], [[]]), [{}])
        self.assertEqual(dense.model.calls, [])
        dense.model.encode = lambda *_args, **_kwargs: np.array([[np.nan, 0.0]])
        with self.assertRaisesRegex(ValueError, "Invalid query vectors"):
            dense.score_candidates_batch([{"search_query": "test"}], [["dense-only"]])


class HybridUnionScoringTests(unittest.TestCase):
    def test_optional_feature_scores_bm25_only_candidate_without_changing_baseline(self):
        query = {"search_query": "первый"}
        plain = HybridRetriever(BM25OnlyCandidate(), dense_retriever(), candidate_k=1)
        dense = dense_retriever()
        enhanced = HybridRetriever(BM25OnlyCandidate(), dense, candidate_k=1,
                                   rescore_dense_union=True)
        baseline = plain.retrieve_batch_with_diagnostics([query])[0]
        top, details = enhanced.retrieve_batch_with_diagnostics([query])[0]
        full_scores = details.pop("full_dense_scores")
        self.assertEqual((top, details), baseline)
        self.assertEqual(set(full_scores), {"bm25-only", "dense-only"})
        self.assertAlmostEqual(full_scores["bm25-only"], 0.6, places=6)
        self.assertNotIn("bm25-only", details["channel_scores"]["dense"])
        self.assertEqual(details["channel_scores"]["description"]["bm25-only"], 7.0)
        self.assertTrue(all(kwargs["prompt_name"] == "query"
                            for _texts, kwargs in dense.model.calls))

    def test_batch_and_single_keep_identical_pools_and_scores(self):
        queries = [{"search_query": "первый"}, {"search_query": "второй"}]
        batch_model = HybridRetriever(BM25OnlyCandidate(), dense_retriever(), candidate_k=1,
                                      rescore_dense_union=True)
        single_model = HybridRetriever(BM25OnlyCandidate(), dense_retriever(), candidate_k=1,
                                       rescore_dense_union=True)
        batch = batch_model.retrieve_batch_with_diagnostics(queries, limit=1)
        single = [single_model.retrieve_with_diagnostics(query, limit=1) for query in queries]
        self.assertEqual(batch, single)
        self.assertEqual(single_model.retrieve(queries[0], limit=1), batch[0][0])

    def test_off_is_default_and_enabled_requires_complete_scorer(self):
        class UnsupportedDense:
            pass
        HybridRetriever(BM25OnlyCandidate(), UnsupportedDense(), dense_weight=0)
        with self.assertRaisesRegex(ValueError, "requires score_candidates_batch"):
            HybridRetriever(BM25OnlyCandidate(), UnsupportedDense(), dense_weight=0,
                            rescore_dense_union=True)
        dense = dense_retriever()
        dense.score_candidates_batch = lambda _queries, _pools: [{}]
        model = HybridRetriever(BM25OnlyCandidate(), dense, candidate_k=1,
                                rescore_dense_union=True)
        with self.assertRaisesRegex(ValueError, "every candidate"):
            model.retrieve_with_diagnostics({"search_query": "первый"})


if __name__ == "__main__":
    unittest.main()
