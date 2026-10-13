"""Partial selection must reproduce the complete (-score, row) ordering."""
import unittest

import numpy as np

from avito_candidates.baseline import BM25Index


def bare_index(size):
    index = BM25Index.__new__(BM25Index)
    index.ids = [f"{row:016x}" for row in range(size)]
    return index


def reference(index, scores, candidates, limit, allowed=None):
    if scores is None or limit == 0:
        return [], {}
    if allowed is not None:
        candidates = candidates[allowed[candidates]]
    order = np.lexsort((candidates, -scores[candidates]))[:limit]
    rows = candidates[order]
    return [index.ids[row] for row in rows], {
        index.ids[row]: float(scores[row]) for row in rows}


class PartialBM25TopTest(unittest.TestCase):
    def test_large_unordered_ties_equal_full_sort(self):
        rng = np.random.default_rng(20261013)
        size = 100000
        index = bare_index(size)
        scores = rng.integers(0, 9, size=size).astype(np.float32) / 10
        candidates = rng.permutation(size)
        for limit in (1, 50, 449, 998, 49999, size, size + 1):
            expected = reference(index, scores, candidates, limit)
            self.assertEqual(index._top(scores, candidates, limit, return_scores=True), expected)

    def test_all_tied_boundary_retains_smallest_row_numbers(self):
        size = 100000
        index = bare_index(size)
        scores = np.full(size, .125, dtype=np.float32)
        candidates = np.arange(size - 1, -1, -1, dtype=np.int64)
        actual = index._top(scores, candidates, 998)
        self.assertEqual(actual, index.ids[:998])

    def test_allowed_mask_is_applied_before_partition(self):
        rng = np.random.default_rng(24)
        index = bare_index(4000)
        scores = rng.integers(0, 15, size=4000).astype(np.float32)
        candidates = rng.permutation(4000).astype(np.int32)
        for count in (0, 1, 10, 449, 450, 1000, 4000):
            allowed = np.zeros(4000, dtype=bool)
            allowed[rng.choice(4000, count, replace=False)] = True
            for limit in (0, 1, 449, 4000, 4001):
                self.assertEqual(index._top(scores, candidates, limit, allowed, True),
                                 reference(index, scores, candidates, limit, allowed))

    def test_nonfinite_fallback_preserves_existing_numpy_order(self):
        index = bare_index(7)
        scores = np.asarray([0., np.inf, 3., np.nan, -np.inf, -0., 3.], dtype=np.float32)
        candidates = np.asarray([6, 5, 4, 3, 2, 1, 0])
        for limit in (1, 3, 6, 7, 10):
            self.assertEqual(index._top(scores, candidates, limit, return_scores=True)[0],
                             reference(index, scores, candidates, limit)[0])

    def test_empty_no_scores_and_invalid_mask_contracts(self):
        index = bare_index(3)
        scores = np.asarray([0., 1., 1.])
        self.assertEqual(index._top(scores, np.asarray([], dtype=int), 1), [])
        self.assertEqual(index._top(None, None, 10, return_scores=True), ([], {}))
        self.assertEqual(index._top(scores, np.arange(3), 0, return_scores=True), ([], {}))
        with self.assertRaisesRegex(ValueError, "match the index size"):
            index._top(scores, np.arange(3), 1, np.asarray([True]))


if __name__ == "__main__":
    unittest.main()
