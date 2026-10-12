"""Проверки пакетной оценки без загрузки модели эмбеддингов."""

import unittest
import csv
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from avito_candidates import cli
from avito_candidates.core import query_key, split_rows
from avito_candidates.hybrid import reciprocal_rank_fusion


class EvaluationBatchTest(unittest.TestCase):
    def test_hybrid_location_sweep_reuses_one_retrieval_and_matches_current_bonus(self):
        ids = [f"item-{index:02d}" for index in range(60)]
        items = [{"item_id": item_id, "item_location_id": "1" if index == 59 else "2"}
                 for index, item_id in enumerate(ids)]
        queries = [
            {"search_query": "первый", "search_location_id": "1"},
            {"search_query": "второй", "search_location_id": "2"},
        ]
        truth = {query_key(query): {ids[59 if index == 0 else 0]}
                 for index, query in enumerate(queries)}
        representatives = {query_key(query): query for query in queries}

        class FakeHybrid:
            calls = 0

            def retrieve_batch_with_diagnostics(self, queries_batch):
                self.calls += 1
                results = []
                for query in queries_batch:
                    channels = {"title": ids, "dense": ids}
                    ranked = [(item_ids, 1.0) for item_ids in channels.values()]
                    location = query["search_location_id"]
                    boosts = {item_id: 0.1 for item_id in ids
                              if items[ids.index(item_id)]["item_location_id"] == location}
                    prediction, scores = reciprocal_rank_fusion(
                        ranked, rank_constant=60, channel_quota=0,
                        score_boosts=boosts, return_scores=True)
                    details = {
                        "candidate_pool": ids, "channels": channels,
                        "channel_weights": {"title": 1.0, "dense": 1.0},
                        "rrf_scores": scores,
                        "prediction_without_quota": prediction,
                        "prediction_with_quota_10": reciprocal_rank_fusion(
                            ranked, rank_constant=60, channel_quota=10,
                            score_boosts=boosts),
                    }
                    results.append((prediction, details))
                return results

        model = FakeHybrid()
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        output_path = Path(directory.name) / "hybrid.json"
        args = ["avito", "evaluate", "--data-dir", "unused", "--split", "all",
                "--method", "hybrid", "--candidate-k", "60",
                "--embedding-batch-size", "2", "--location-bonus", "0.1",
                "--embedding-max-length", "128", "--ef-search", "1000",
                "--sweep-location-bonus", "--output", str(output_path)]
        with (
            patch("sys.argv", args),
            patch.object(cli, "read", return_value=items),
            patch.object(cli, "read_validation",
                         return_value=(truth, representatives, 0, 2, 2)),
            patch.object(cli, "build_retriever", return_value=model),
            patch.object(cli, "save_json") as save_json,
        ):
            cli.main()
        report = save_json.call_args.args[1]
        self.assertEqual(report["retrieval_config"]["ef_search"], 1000)
        self.assertEqual(report["embedding_max_length"], 128)
        self.assertEqual(report["ef_search"], 1000)
        self.assertEqual(report["microcategory_exact_bonus"], 0.0)
        self.assertIsNone(report["embedding_checkpoint_sha256"])
        # Reproduction inherits the saved search depth instead of silently
        # replacing ef=1000 with an unrelated command-line default of 300.
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_reranker_new_octen.py"
        spec = importlib.util.spec_from_file_location("train_reranker_run_test", script_path)
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        source = {**report, "split": "queries"}
        common = dict(data_dir=Path("dataset"), checkpoint=Path("checkpoint"),
                      device="cuda", batch_size=32)
        restored = script.retrieval_arguments(source, **common)
        self.assertEqual(restored[restored.index("--ef-search") + 1], "1000")
        self.assertEqual(restored[restored.index("--embedding-max-length") + 1], "128")
        override = script.retrieval_arguments(source, **common, ef_search=1200)
        self.assertEqual(override[override.index("--ef-search") + 1], "1200")
        legacy = {name: value for name, value in source.items()
                  if name not in {"retrieval_config", "ef_search", "embedding_max_length"}}
        legacy_arguments = script.retrieval_arguments(legacy, **common)
        self.assertEqual(legacy_arguments[legacy_arguments.index("--ef-search") + 1], "300")
        sweep = report["retrieval_diagnostics"]["location_bonus_sweep"]
        rows = {row["bonus"]: row for row in sweep["results"]}
        self.assertEqual(model.calls, 1)
        self.assertEqual(rows[0]["recall_at_50"], 0.5)
        self.assertEqual(rows[0.1]["recall_at_50"], report["recall_at_50"])
        self.assertEqual(report["recall_at_50"], 1.0)
        scores_report = report["retrieval_diagnostics"]["rrf_scores"]
        self.assertEqual(scores_report["rows"], 100)
        with Path(scores_report["file"]).open(encoding="utf-8", newline="") as stream:
            saved = list(csv.DictReader(stream))
        first_query_top = next(row for row in saved
                               if row["search_query"] == "первый" and row["rank"] == "1")
        self.assertEqual(first_query_top["item_id"], ids[59])
        self.assertAlmostEqual(float(first_query_top["rrf_score"]), 2 / 120 + 0.1)

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
        self.assertIsNone(report["retrieval_diagnostics"]["rrf_scores"])

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
