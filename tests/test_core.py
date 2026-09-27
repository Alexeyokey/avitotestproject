import unittest
from avito_candidates.core import recall_at_k, split_rows, validate_answers, query_key


class ContractsTest(unittest.TestCase):
    def test_macro_recall(self):
        self.assertAlmostEqual(recall_at_k({"a": ["x"], "b": ["z"]},
                                          {"a": {"x", "y"}, "b": {"z"}}), .75)

    def test_duplicate_pairs_never_cross_split(self):
        rows = [{"search_query": "ремонт", "item_id": str(i)} for i in range(100)] * 2
        fit, valid = split_rows(rows)
        self.assertTrue(fit and valid)
        self.assertFalse({(query_key(x), x["item_id"]) for x in fit} &
                         {(query_key(x), x["item_id"]) for x in valid})

    def test_query_holdout(self):
        rows = [{"search_query": str(i), "item_id": str(j)} for i in range(100) for j in range(3)]
        fit, valid = split_rows(rows, mode="queries")
        self.assertFalse({x["search_query"] for x in fit} & {x["search_query"] for x in valid})

    def test_all_uses_every_row_and_custom_fraction_changes_partial_size(self):
        rows = [{"search_query": str(i), "item_id": str(i)} for i in range(1000)]
        fit_all, valid_all = split_rows(rows, mode="all")
        self.assertEqual(fit_all, [])
        self.assertEqual(valid_all, rows)
        fit_small, valid_small = split_rows(rows, mode="queries", fraction=0.1)
        fit_large, valid_large = split_rows(rows, mode="queries", fraction=0.5)
        self.assertEqual(len(fit_small) + len(valid_small), len(rows))
        self.assertEqual(len(fit_large) + len(valid_large), len(rows))
        self.assertLess(len(valid_small), len(valid_large))
        self.assertTrue({r["search_query"] for r in valid_small}.issubset(
            {r["search_query"] for r in valid_large}
        ))

    def test_submission_contract(self):
        q, iid = "A" * 16, "0" * 16
        validate_answers([{"query_id": q, "answer": iid}], [q], [iid])
        for answer in [iid + " " + iid, "F" * 16, "1" * 16, iid + "  "]:
            with self.assertRaises(ValueError):
                validate_answers([{"query_id": q, "answer": answer}], [q], [iid])
        with self.assertRaises(ValueError):
            validate_answers([], [q], [iid])


if __name__ == "__main__":
    unittest.main()
