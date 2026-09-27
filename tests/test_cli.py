"""CLI-level checks for batched evaluation without loading an embedding model."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from avito_candidates import cli
from avito_candidates.core import query_key, split_rows


class EvaluationBatchTest(unittest.TestCase):
    def test_dense_evaluation_batches_queries_without_changing_recall(self):
        train = [
            {"search_query": f"услуга {index}", "item_id": f"item-{index}"}
            for index in range(3)
        ]
        items = [{"item_id": row["item_id"]} for row in train]
        truth = {query_key(row): {row["item_id"]} for row in train}
        representatives = {query_key(row): row for row in train}

        class FakeDense:
            def __init__(self):
                self.calls = []

            def retrieve_batch(self, queries, limit):
                self.calls.append((len(queries), limit))
                return [[query["item_id"]] for query in queries]

        model = FakeDense()
        args = [
            "avito", "evaluate", "--data-dir", "unused", "--split", "all",
            "--method", "dense", "--embedding-batch-size", "2", "--candidate-k", "20",
        ]
        with patch("sys.argv", args), patch.object(
            cli, "read", return_value=items
        ), patch.object(
            cli, "read_validation", return_value=(truth, representatives, 0, 3, 3)
        ), patch.object(cli, "build_retriever", return_value=model), patch.object(
            cli, "save_json"
        ) as save_json:
            cli.main()

        self.assertEqual(model.calls, [(2, 50), (1, 50)])
        report = save_json.call_args.args[1]
        self.assertEqual(report["evaluated_queries"], 3)
        self.assertEqual(report["recall_at_50"], 1.0)

    def test_streamed_validation_matches_existing_split(self):
        rows = [
            {"search_query": "Ремонт", "item_id": "a"},
            {"search_query": "ремонт", "item_id": "b"},
            {"search_query": "монтаж", "item_id": "c"},
            {"search_query": "монтаж", "item_id": "c"},
            {"search_query": "покраска", "item_id": "outside"},
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.parquet"
            pd.DataFrame(rows).to_parquet(path)
            for mode in ("pairs", "queries", "all"):
                _fit, valid = split_rows(rows, mode=mode, fraction=0.5, seed=42)
                truth, _representatives, fit_count, valid_count, in_corpus = (
                    cli.read_validation(path, {"a", "b", "c"}, mode=mode,
                                        fraction=0.5, seed=42, batch_size=2)
                )
                expected = {}
                for row in valid:
                    if row["item_id"] in {"a", "b", "c"}:
                        expected.setdefault(query_key(row), set()).add(row["item_id"])
                self.assertEqual(dict(truth), expected)
                self.assertEqual((fit_count, valid_count), (len(rows) - len(valid), len(valid)))
                self.assertEqual(in_corpus, sum(row["item_id"] in {"a", "b", "c"}
                                                for row in valid))


if __name__ == "__main__":
    unittest.main()
