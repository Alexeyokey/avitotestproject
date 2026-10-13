"""Publication proof and selected-prefix runtime use tiny synthetic artifacts."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import joblib
import numpy as np

import publish_fresh_reranker as PUBLISH
import evaluate_fresh_validation as EVAL


class FakeHistModel:
    n_iter_ = 150

    def __init__(self):
        self._predictors = list(range(150))

    def staged_predict_proba(self, matrix):
        for stage in range(1, 151):
            values = [0.9, 0.1] if stage <= 50 else [0.1, 0.9]
            positive = np.resize(values, len(matrix))
            yield np.column_stack([1 - positive, positive])

    def predict_proba(self, matrix):
        positive = np.resize([0.1, 0.9], len(matrix))
        return np.column_stack([1 - positive, positive])


def score_row(text, stage, hits, baseline=1):
    return {"query_text": text, "available_relevant_count": 1,
            "full_relevant_count": 2, "representative": True,
            "pool_size": 2, "pool_hits": 1, "baseline_hits": baseline,
            f"s{stage}_hits": hits}


class FreshPublicationTests(unittest.TestCase):
    def fixture(self, root):
        from avito_candidates.reranker import V4_FEATURES, corpus_fingerprint
        protocol = root / "protocol"
        protocol.mkdir()
        manifest_path = protocol / "manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        train = root / "train-corrected.parquet"
        train.write_bytes(b"opaque proven training fixture")
        train_hash = EVAL.sha256_file(train)
        items = [{"item_id": "a" * 16}, {"item_id": "b" * 16}]
        config = {"method": "bm25", "candidate_k": 2, "title_weight": 1.0,
                  "params_weight": 0.0, "description_weight": 0.0,
                  "stem_title_weight": 0.0, "stem_params_weight": 0.0,
                  "location_bonus": 0.0, "bm25_rrf_k": 30}
        artifact = {"model": FakeHistModel(), "estimator": "histgb",
                    "metadata": {"feature_version": 4, "feature_names": list(V4_FEATURES),
                                 "retrieval_config": config, "split": "queries", "seed": 42,
                                 "validation_fraction": .2,
                                 "corpus_fingerprint": corpus_fingerprint(items, feature_version=4),
                                 "category_sentinel_correction": {"version": 1,
                                                                   "derived_parquet_sha256": train_hash}},
                    "priors": {"opaque_fixture": {"counts": [2, 3], "array": np.arange(3)}},
                    "provenance": {"training_data_sha256": train_hash, "keep_original": True},
                    "training_configuration": {"fitted_iterations": 150, "parameters": {"keep": 1}},
                    "selection_status": "unselected", "selected_iteration": None}
        model = root / "unselected.joblib"
        joblib.dump(artifact, model)
        model_hash, protocol_hash = EVAL.sha256_file(model), EVAL.sha256_file(manifest_path)
        evaluation_dir = root / "evaluation"
        evaluation_dir.mkdir()
        reports = {str(stage): EVAL.summarize([score_row("tune text", stage, int(stage == 50))], stage)
                   for stage in EVAL.TUNE_STAGES}
        tune_report = evaluation_dir / "tune-report.json"
        tune_report.write_text(json.dumps({"stages": reports}), encoding="utf-8")
        cache_paths = {"parquet": root / "tune.parquet", "metadata": root / "tune.json"}
        for path in cache_paths.values():
            path.write_bytes(b"opaque completed tune cache")
        hashes = {name: EVAL.sha256_file(path) for name, path in cache_paths.items()}
        selection = EVAL.make_selection(protocol_hash, model_hash, hashes, EVAL.sha256_file(tune_report),
                                         50, reports, train_hash, cache_paths)
        (evaluation_dir / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
        # Audit is worse than baseline deliberately: publication must not call
        # this a confirmed gain or select a different capacity from audit.
        rows = [score_row("audit text", 50, 0)]
        scores = evaluation_dir / "audit-group-scores.jsonl"
        scores.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        audit = {"report_version": 1, "phase": "audit", "policy": EVAL.POLICY,
                 "protocol_sha256": protocol_hash, "model_sha256": model_hash,
                 "train_data_sha256": train_hash, "encoder_content_sha256": "encoder-proof",
                 "selected_stage": 50, "stages": {"50": EVAL.summarize(rows, 50)},
                 "group_scores_sha256": EVAL.sha256_file(scores),
                 "scope": "Synthetic cold cohort, not official benchmark"}
        (evaluation_dir / "audit-report.json").write_text(json.dumps(audit), encoding="utf-8")
        manifest = {"stats": {"audit": {"groups": 1}}, "corpus": {"size": 200000}}
        return {"protocol": protocol, "evaluation_dir": evaluation_dir,
                "model": model, "train": train, "artifact": artifact, "items": items,
                "manifest": manifest, "protocol_hash": protocol_hash, "audit": audit,
                "scores": scores, "output": root / "selected.joblib"}

    def publish(self, fixture, **patches):
        with mock.patch.object(EVAL, "checked_protocol", return_value=(
                fixture["manifest"], fixture["protocol_hash"])), \
                mock.patch.object(EVAL, "check_encoder_identity", return_value="encoder-proof"), \
                mock.patch.object(EVAL, "check_training_data", side_effect=lambda path, *_args:
                                  EVAL.sha256_file(path)):
            return PUBLISH.publish(fixture["protocol"], fixture["evaluation_dir"], fixture["model"],
                                   fixture["train"], fixture["output"])

    def write_audit(self, fixture, audit):
        (fixture["evaluation_dir"] / "audit-report.json").write_text(json.dumps(audit), encoding="utf-8")

    def test_preserves_all_predictors_priors_and_provenance_and_runtime_honors_prefix(self):
        from avito_candidates.reranker import Reranker
        with TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            original_hash = EVAL.sha256_file(fixture["model"])
            result = self.publish(fixture)
            selected = joblib.load(fixture["output"])
            self.assertEqual(selected["model"]._predictors, list(range(150)))
            self.assertEqual(selected["model"].n_iter_, 150)
            self.assertEqual(selected["metadata"], fixture["artifact"]["metadata"])
            self.assertEqual(selected["provenance"], fixture["artifact"]["provenance"])
            np.testing.assert_array_equal(selected["priors"]["opaque_fixture"]["array"], np.arange(3))
            self.assertEqual(selected["priors"]["opaque_fixture"]["counts"], [2, 3])
            self.assertEqual(selected["selected_iteration"], 50)
            self.assertEqual(selected["selection_status"], PUBLISH.STATUS)
            self.assertFalse(selected["fresh_selection"]["official_benchmark_validated"])
            self.assertFalse(selected["fresh_selection"]["audit_used_for_capacity_selection"])
            self.assertNotIn("confirmed_gain", selected["fresh_selection"])
            self.assertEqual(result["sha256"], EVAL.sha256_file(fixture["output"]))
            self.assertEqual(EVAL.sha256_file(fixture["model"]), original_hash)
            details = {"candidate_pool": ["a" * 16, "b" * 16],
                       "channels": {"title": ["a" * 16, "b" * 16]}}
            inputs = [(["a" * 16, "b" * 16], details)]
            raw = Reranker(fixture["model"], fixture["items"], fixture["artifact"]["metadata"]["retrieval_config"])
            chosen = Reranker(fixture["output"], fixture["items"], fixture["artifact"]["metadata"]["retrieval_config"])
            self.assertEqual(raw.rank_batch([{}], inputs, limit=1), [["b" * 16]])
            self.assertEqual(chosen.rank_batch([{}], inputs, limit=1), [["a" * 16]])

    def test_missing_or_wrong_audit_cannot_publish(self):
        mutations = ("missing", "phase", "model", "protocol", "train", "stage", "extra_stage", "summary")
        for mutation in mutations:
            with self.subTest(mutation=mutation), TemporaryDirectory() as directory:
                fixture = self.fixture(Path(directory))
                report = deepcopy(fixture["audit"])
                if mutation == "missing":
                    (fixture["evaluation_dir"] / "audit-report.json").unlink()
                elif mutation in {"model", "protocol", "train"}:
                    report[{"model": "model_sha256", "protocol": "protocol_sha256",
                            "train": "train_data_sha256"}[mutation]] = "wrong"
                elif mutation == "phase":
                    report["phase"] = "tune"
                elif mutation == "stage":
                    report["selected_stage"] = 25
                elif mutation == "extra_stage":
                    report["stages"]["150"] = report["stages"]["50"]
                elif mutation == "summary":
                    report["stages"]["50"]["conditional_available"]["reranker"]["text_macro_recall_at_50"] = .99
                if mutation != "missing":
                    self.write_audit(fixture, report)
                with self.assertRaises(ValueError):
                    self.publish(fixture)
                self.assertFalse(fixture["output"].exists())

    def test_scores_hash_full_cohort_and_stage_are_verified(self):
        for mutation in ("bytes", "empty", "extra_stage"):
            with self.subTest(mutation=mutation), TemporaryDirectory() as directory:
                fixture = self.fixture(Path(directory))
                report = deepcopy(fixture["audit"])
                if mutation == "bytes":
                    fixture["scores"].write_text("changed", encoding="utf-8")
                else:
                    row = score_row("audit text", 50, 0)
                    if mutation == "extra_stage":
                        row["s150_hits"] = 1
                    fixture["scores"].write_text("" if mutation == "empty" else json.dumps(row) + "\n",
                                                 encoding="utf-8")
                    report["group_scores_sha256"] = EVAL.sha256_file(fixture["scores"])
                    self.write_audit(fixture, report)
                with self.assertRaises(ValueError):
                    self.publish(fixture)
                self.assertFalse(fixture["output"].exists())

    def test_tune_proof_and_category_correction_are_mandatory(self):
        for mutation in ("tune_cache", "category", "selected_model"):
            with self.subTest(mutation=mutation), TemporaryDirectory() as directory:
                fixture = self.fixture(Path(directory))
                if mutation == "tune_cache":
                    (Path(directory) / "tune.parquet").write_bytes(b"changed tune")
                else:
                    artifact = fixture["artifact"]
                    if mutation == "category":
                        artifact["metadata"]["category_sentinel_correction"] = {}
                    else:
                        artifact["selected_iteration"] = 25
                    joblib.dump(artifact, fixture["model"])
                with self.assertRaises(ValueError):
                    self.publish(fixture)
                self.assertFalse(fixture["output"].exists())

    def test_existing_output_is_preserved_and_atomic_race_does_not_overwrite(self):
        with TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            fixture["output"].write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                self.publish(fixture)
            self.assertEqual(fixture["output"].read_bytes(), b"keep")
            target = Path(directory) / "raced.joblib"

            def race(_temporary, destination):
                Path(destination).write_bytes(b"created by another publisher")
                raise FileExistsError("raced")

            with mock.patch.object(PUBLISH.os, "link", side_effect=race):
                with self.assertRaises(FileExistsError):
                    PUBLISH.publish_new(target, {"tiny": True})
            self.assertEqual(target.read_bytes(), b"created by another publisher")
            self.assertEqual(list(Path(directory).glob("raced.joblib.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
