import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import evaluate_fresh_validation as EVAL
from evaluate_research_reranker import (aligned_rows, paired_pipeline_bootstrap,
    named_summary, check_research_cache, check_identity, main)
from evaluate_research_reranker import (check_frozen_reference,
    FROZEN_REFERENCE_MODEL_SHA256, FROZEN_REFERENCE_TRAIN_SHA256)


class FakeHGB:
    n_iter_ = 150

    def staged_predict_proba(self, matrix):
        quality = matrix[:, 2]
        for _ in range(self.n_iter_):
            yield np.column_stack([1 - quality, quality])


class PipelineComparisonTest(unittest.TestCase):
    def row(self, identity="a", text="query", *, available=2, full=3, hits=1):
        return {"group_id": 0, "group_identity": identity, "query_text": text,
            "query_key": [text, "1", "0", "", "114"], "representative": True,
            "available_relevant_count": available, "full_relevant_count": full,
            "s150_hits": hits, "pool_hits": hits}

    def test_alignment_uses_canonical_identity_and_full_truth(self):
        old = [self.row("a"), self.row("b", "other")]
        new = [self.row("b", "other"), self.row("a")]
        new[0]["group_id"] = 99
        self.assertEqual(len(aligned_rows(new, old)), 2)
        changed = copy.deepcopy(new)
        changed[1]["available_relevant_count"] = 1
        with self.assertRaisesRegex(ValueError, "denominator"):
            aligned_rows(changed, old)
        with self.assertRaises(ValueError):
            aligned_rows([new[0]], old)
        with self.assertRaises(ValueError):
            aligned_rows([new[0], new[0]], old)

    def test_bootstrap_clusters_text_not_context_and_drops_only_zero_conditional_denominator(self):
        old = [self.row("a1", "a", available=1, full=1, hits=0),
               self.row("a2", "a", available=1, full=1, hits=0),
               self.row("b", "b", available=1, full=1, hits=0),
               self.row("c", "c", available=0, full=1, hits=0)]
        new = copy.deepcopy(old)
        new[0]["s150_hits"] = 1
        pairs = aligned_rows(new, old)
        result = paired_pipeline_bootstrap(pairs, "s150_hits", "s150_hits",
            "available_relevant_count", replicates=100)
        self.assertEqual(result["query_text_clusters"], 2)
        self.assertEqual(result["delta"], .25)
        full = paired_pipeline_bootstrap(pairs, "s150_hits", "s150_hits",
            "full_relevant_count", replicates=100)
        self.assertEqual(full["query_text_clusters"], 3)
        self.assertAlmostEqual(full["delta"], 1 / 6)

    def test_positive_denominator_is_never_derived_from_retrieved_positives(self):
        row = self.row(hits=1)
        result = paired_pipeline_bootstrap([(row, self.row(hits=0))], "s150_hits", "s150_hits",
            "available_relevant_count", replicates=10)
        self.assertEqual(result["delta"], .5)
        with self.assertRaises(ValueError):
            paired_pipeline_bootstrap([(self.row(hits=3), self.row(hits=0))], "s150_hits", "s150_hits",
                "available_relevant_count", replicates=10)

    def test_each_hgb_scores_its_own_pool_and_preserves_empty_groups(self):
        features = ["rrf_score", "baseline_top50", "quality"]
        a, b = "0000000000000001", "0000000000000002"
        negatives = [f"{i:016x}" for i in range(3, 53)]
        old_ids, new_ids = negatives + [a], [a, b] + negatives
        empty = {"group_id": 1, "group_identity": "empty", "query_text": "empty",
            "query_key": ["empty", "2", "0", "", "114"], "representative": True,
            "full_relevant_count": 1, "available_relevant_count": 0, "pool_size": 0,
            "positives_in_pool": 0, "baseline_hits": 0, "local_relevant_count": 0,
            "nonlocal_relevant_count": 0, "_available_ids": set()}
        outputs = []
        with tempfile.TemporaryDirectory() as directory:
            for name, items in (("old", old_ids), ("new", new_ids)):
                labels = [int(item in {a, b}) for item in items]
                baseline = [i < 50 for i in range(len(items))]
                group = {**self.row(), "pool_size": len(items), "positives_in_pool": sum(labels),
                    "baseline_hits": sum(label for label, chosen in zip(labels, baseline) if chosen),
                    "local_relevant_count": 2, "nonlocal_relevant_count": 0, "_available_ids": {a, b}}
                quality = ([.5] * len(items) if name == "old" else
                           [1. if item in {a, b} else .01 for item in items])
                path = Path(directory) / (name + ".parquet")
                pq.write_table(pa.table({"rrf_score": pa.array([1. / (i + 1) for i in range(len(items))], pa.float32()),
                    "baseline_top50": pa.array(list(map(float, baseline)), pa.float32()),
                    "quality": pa.array(quality, pa.float32()), "group_id": [0] * len(items),
                    "item_id": items, "label": labels, "positive_local": [True] * len(items)}), path)
                rows = EVAL.score_cache(path, {"feature_names": features}, [group, empty],
                    {"model": FakeHGB()}, (150,), batch_size=17)
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[1]["s150_hits"], 0)
                outputs.append(rows)
        old, new = outputs
        pairs = aligned_rows(new, old)
        self.assertEqual(old[0]["s150_hits"], 0)
        self.assertEqual(new[0]["s150_hits"], 2)
        self.assertEqual(paired_pipeline_bootstrap(pairs, "pool_hits", "pool_hits",
            "available_relevant_count", replicates=10)["delta"], .5)
        summary = named_summary(new, 150)
        self.assertEqual(summary["conditional_available"]["reranker"]["text_macro_recall_at_50"], 1.)
        self.assertAlmostEqual(summary["full_truth"]["reranker"]["text_macro_recall_at_50"], 1 / 3)
        self.assertIn("text_macro_candidate_recall", summary["conditional_available"]["candidate_pool"])

    def test_v7_cache_classifier_and_source_policy_cannot_be_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.py"
            path.write_text("source")
            policy = {"research_priors_sha256": "prior"}
            artifact = {"metadata": {"training_prior_policy": {"microcategory_classifier": "grouped-out-of-fold"}}}
            metadata = {"research_cache_version": 1, "research_train_data_sha256": "training",
                **policy, "training_prior_policy": artifact["metadata"]["training_prior_policy"],
                "research_source_sha256": {str(path): EVAL.sha256_file(path)}}
            with patch("evaluate_research_reranker.research_prior_policy", return_value=policy):
                check_research_cache(metadata, artifact, "training")
                with self.assertRaises(ValueError):
                    check_research_cache({**metadata, "research_priors_sha256": "other"}, artifact, "training")
                path.write_text("changed")
                with self.assertRaises(ValueError):
                    check_research_cache(metadata, artifact, "training")

    def test_string_paths_are_verified_without_ignoring_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model"
            path.write_bytes(b"model")
            identity = {"model": {"path": str(path), "sha256": EVAL.sha256_file(path)}}
            check_identity(identity)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                check_identity(identity)

    def test_original_model_and_training_data_are_the_exact_frozen_pair(self):
        with patch("evaluate_research_reranker.EVAL.sha256_file",
                   side_effect=[FROZEN_REFERENCE_MODEL_SHA256, FROZEN_REFERENCE_TRAIN_SHA256]):
            check_frozen_reference("original", "original-train")
        with patch("evaluate_research_reranker.EVAL.sha256_file", return_value="wrong-model"):
            with self.assertRaisesRegex(ValueError, "byte-identical"):
                check_frozen_reference("different", "original-train")

    def test_audit_and_wrong_reference_stage_have_no_cli_path(self):
        base = ["--protocol-dir", "protocol", "--new-model", "new", "--new-train-data", "new-train",
                "--new-cache-dir", "new-cache", "--old-model", "old", "--old-train-data", "old-train",
                "--old-cache-dir", "old-cache", "--output-dir", "report"]
        with patch("evaluate_research_reranker.evaluate") as evaluate:
            for extra in (["--phase", "audit"], ["--old-stage", "100"], ["--primary-stage", "50"]):
                with self.assertRaises(SystemExit):
                    main(base + extra)
            evaluate.assert_not_called()
            main(base)
            self.assertEqual(evaluate.call_args.args[0].new_stages, [75, 100, 150])


if __name__ == "__main__":
    unittest.main()
