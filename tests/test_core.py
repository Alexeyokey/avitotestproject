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
