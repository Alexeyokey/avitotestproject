"""Frozen-regression guards use synthetic files/targets, never real Audit GT."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import evaluate_fresh_validation as EVAL
from freeze_research_selection import (REGRESSION_POLICY, REQUIRED_FILES, identity,
    validate_tune_report, verify_frozen_selection, freeze, regression_source_paths,
    select_dense_index, verify_dense_index)
from cache_frozen_research_audit import check_regression_cache, check_loaded_dense_index, produce
from evaluate_frozen_research_audit import (public_train_control, assert_targets_in_public_train,
    source_summary, annotate_known_slices, compare_rows, evaluate, main)


class FrozenRegressionTest(unittest.TestCase):
    def tune_report(self):
        return {"report_version": 1, "phase": "tune", "exploratory": True,
            "audit_labels_read": False, "selection_performed": False, "old_stage": 150,
            "new_capacity_curve": {"100": {"conditional_available": {
                "reranker": {"text_macro_recall_at_50": .8}}}}}

    def selection_fixture(self, root):
        files = {}
        for name in REQUIRED_FILES:
            path = root / name
            path.write_text(name)
            files[name] = identity(path)
        report = {**self.tune_report(), "protocol_sha256": files["protocol_manifest"]["sha256"],
                  "group_scores_sha256": files["tune_group_scores"]["sha256"]}
        for prefix, key in (("new", "new_pipeline_identity"), ("old", "old_pipeline_identity")):
            report[key] = {field: files[f"{prefix}_{suffix}"] for field, suffix in (
                ("model", "model"), ("training_data", "train_data"), ("cache", "tune_cache"),
                ("metadata", "tune_metadata"))}
        Path(files["tune_report"]["path"]).write_text(json.dumps(report))
        files["tune_report"] = identity(files["tune_report"]["path"])
        source = root / "producer.py"
        source.write_text("frozen producer")
        source_snapshot = {str(source): EVAL.sha256_file(source)}
        source_snapshot.update({str(path.resolve()): EVAL.sha256_file(path) for path in regression_source_paths()})
        dense_root = root / "dense"
        dense_root.mkdir()
        fingerprint = "1" * 20
        binary, dense_metadata = dense_root / (fingerprint + ".bin"), dense_root / (fingerprint + ".json")
        binary.write_bytes(b"synthetic dense index")
        dense_metadata.write_text("{}")
        dense_index = {"directory": str(dense_root), "fingerprint": fingerprint,
            "binary": identity(binary), "metadata": identity(dense_metadata)}
        files["dense_index_binary"], files["dense_index_metadata"] = dense_index["binary"], dense_index["metadata"]
        payload = {"selection_version": 1, "policy": dict(REGRESSION_POLICY), "selected_stage": 100,
            "selected_tune_text_macro_recall": .8, "files": files,
            "source_sha256": source_snapshot, "dense_index": dense_index,
            "retrieval_config": {"method": "hybrid"}, "research_prior_policy": {"priors": "fixed"}}
        path = root / "selection.json"
        self.write_selection(path, payload)
        return path, payload

    def write_selection(self, path, payload):
        payload = {key: value for key, value in payload.items() if key != "selection_sha256"}
        payload["selection_sha256"] = EVAL.canonical_sha256(payload)
        path.write_text(json.dumps(payload))

    def test_stage_must_come_from_completed_tune_only_report(self):
        report = self.tune_report()
        self.assertEqual(validate_tune_report(report, 100), .8)
        for altered, stage in (({**report, "audit_labels_read": True}, 100), (report, 75),
                               (report, 200), ({**report, "phase": "audit"}, 100)):
            with self.assertRaises(ValueError):
                validate_tune_report(altered, stage)

    def test_frozen_closure_checks_no_audit_target_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path, payload = self.selection_fixture(Path(directory))
            with patch("freeze_research_selection.FROZEN_REFERENCE_MODEL_SHA256",
                       payload["files"]["old_model"]["sha256"]), \
                 patch("evaluate_fresh_validation.read_phase_labels", side_effect=AssertionError("No Audit targets")):
                verified = verify_frozen_selection(path)
                self.assertEqual(verified["selected_stage"], 100)
                self.assertFalse(verified["policy"]["independent_fresh_test"])
                self.assertTrue(verified["policy"]["previous_baseline_audit_observed"])

    def test_freeze_writes_exclusively_from_completed_report_without_audit_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture_selection, initial = self.selection_fixture(root)
            files = initial["files"]
            report_path = Path(files["tune_report"]["path"])
            report = json.loads(report_path.read_text())
            (root / "manifest.json").write_bytes(Path(files["protocol_manifest"]["path"]).read_bytes())
            (root / "research-tune-group-scores.jsonl").write_bytes(Path(files["tune_group_scores"]["path"]).read_bytes())
            data = Path(files["new_train_data"]["path"])
            data.with_suffix(data.suffix + ".json").write_text("train metadata")
            data.with_suffix(data.suffix + ".priors.joblib").write_text("train priors")
            micro = Path(files["microcategory_model"]["path"])
            micro.with_suffix(micro.suffix + ".json").write_text("micro metadata")
            base = {"reference": files["encoder_base_reference"], "base_revision": "frozen"}
            cache_metadata = {"research_encoder_base": base, "research_source_sha256": initial["source_sha256"]}
            metadata_path = Path(files["new_tune_metadata"]["path"])
            metadata_path.write_text(json.dumps(cache_metadata))
            report["new_pipeline_identity"]["metadata"] = identity(metadata_path)
            report_path.write_text(json.dumps(report))
            fit = root / "fit-texts.json"
            fit.write_text(json.dumps({"texts": ["fit"]}))
            old_meta = Path(files["old_audit_metadata"]["path"])
            old_meta.write_text("{}")
            config = {"embedding_model": "frozen-model", "microcategory_model": str(micro)}
            artifact = {"metadata": {"retrieval_config": config},
                "model": SimpleNamespace(n_iter_=150, staged_predict_proba=lambda matrix: ())}
            manifest = {"sources": {"train": files["public_train_inventory"]}}
            args = SimpleNamespace(output=root / "new-selection.json", tune_report=report_path,
                protocol_dir=root, old_audit_cache_dir=root, dense_cache=root / "dense",
                stage=100, reason="selected using Tune only")
            with patch("freeze_research_selection.EVAL.checked_protocol", return_value=(manifest, files["protocol_manifest"]["sha256"])), \
                 patch("freeze_research_selection.EVAL.protocol_file", return_value=fit), \
                 patch("freeze_research_selection.check_frozen_reference"), \
                 patch("joblib.load", return_value=artifact), \
                 patch("freeze_research_selection.validate_research_artifact"), \
                 patch("freeze_research_selection.EVAL.check_encoder_identity"), \
                 patch("freeze_research_selection.EVAL.check_training_data"), \
                 patch("freeze_research_selection.check_training_prior_file"), \
                 patch("freeze_research_selection.check_research_cache"), \
                 patch("freeze_research_selection.check_encoder_base", return_value=base), \
                 patch("freeze_research_selection.research_prior_policy", return_value={"fixed": True}), \
                 patch("freeze_research_selection.EVAL.phase_cache_paths", return_value=(Path(files["old_audit_cache"]["path"]), old_meta)), \
                 patch("freeze_research_selection.EVAL.check_phase_cache_inputs"), \
                 patch("freeze_research_selection.select_dense_index", return_value=initial["dense_index"]), \
                 patch("freeze_research_selection.FROZEN_REFERENCE_MODEL_SHA256", files["old_model"]["sha256"]), \
                 patch("evaluate_fresh_validation.read_phase_labels") as targets:
                frozen = freeze(args)
                self.assertEqual(frozen["selected_stage"], 100)
                self.assertFalse(frozen["policy"]["independent_fresh_test"])
                targets.assert_not_called()
                with self.assertRaises(FileExistsError):
                    freeze(args)

    def test_resealed_unreported_stage_and_missing_closure_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, payload = self.selection_fixture(Path(directory))
            with patch("freeze_research_selection.FROZEN_REFERENCE_MODEL_SHA256",
                       payload["files"]["old_model"]["sha256"]):
                self.write_selection(path, {**payload, "selected_stage": 150})
                with self.assertRaisesRegex(ValueError, "absent"):
                    verify_frozen_selection(path)
                altered = copy.deepcopy(payload)
                del altered["files"]["public_train_inventory"]
                self.write_selection(path, altered)
                with self.assertRaisesRegex(ValueError, "required material"):
                    verify_frozen_selection(path)

    def test_model_source_or_report_mutation_cannot_preserve_seal(self):
        for key in ("new_model", "tune_report", "old_audit_cache"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                path, payload = self.selection_fixture(Path(directory))
                Path(payload["files"][key]["path"]).write_text("changed")
                with self.assertRaisesRegex(ValueError, "Frozen input"):
                    verify_frozen_selection(path)
        with tempfile.TemporaryDirectory() as directory:
            path, payload = self.selection_fixture(Path(directory))
            source = next(iter(payload["source_sha256"]))
            Path(source).write_text("changed")
            with self.assertRaisesRegex(ValueError, "source changed"):
                verify_frozen_selection(path)

    def regression_cache(self, root):
        path = root / "audit.parquet"
        pq.write_table(pa.table({"rrf_score": pa.array([.1], pa.float32()),
            "baseline_top50": pa.array([1.], pa.float32()), "group_id": [0],
            "item_id": ["0000000000000001"], "label": [1], "positive_local": [True],
            "positive_seen": [False]}), path)
        selection = {"files": {"protocol_manifest": {"sha256": "protocol"},
                                "new_model": {"sha256": "model"}},
                     "selected_stage": 100, "selection_sha256": "selection",
                     "dense_index": {"fingerprint": "frozen-index"},
                     "source_sha256": {str((Path(__file__).parent / "cache_frozen_research_audit.py").resolve()): "producer"}}
        meta = {"cache_version": 2, "role": "validation", "fresh_partition": "audit",
            "fresh_validation_protocol_sha256": "protocol", "model_sha256": "model",
            "historical_holdout_reused": True, "full_truth_preserved": True,
            "frozen_regression_cache_version": 1, "previous_baseline_audit_observed": True,
            "independent_fresh_test": False, "purpose": "frozen regression", "frozen_selected_stage": 100,
            "regression_policy": REGRESSION_POLICY, "regression_selection_sha256": "selection",
            "regression_selection_file_sha256": "selection-file", "feature_names": ["rrf_score", "baseline_top50"],
            "producer_sha256": "producer", "regression_dense_index": selection["dense_index"],
            "groups": [{"group_id": 0, "pool_size": 1}], "rows": 1,
            "cache_parquet_sha256": EVAL.sha256_file(path)}
        return path, meta, selection

    def test_observed_audit_cache_requires_honest_regression_flags_and_frozen_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            path, meta, selection = self.regression_cache(Path(directory))
            check_regression_cache(path, meta, selection, "selection-file")
            for key, value in (("historical_holdout_reused", False), ("independent_fresh_test", True),
                               ("frozen_selected_stage", 150), ("previous_baseline_audit_observed", False)):
                with self.assertRaises(ValueError):
                    check_regression_cache(path, {**meta, key: value}, selection, "selection-file")

    def test_regression_integrity_rejects_oversize_rows_and_corrupt_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path, meta, selection = self.regression_cache(Path(directory))
            for changed in ({**meta, "groups": [{"group_id": 0, "pool_size": 999}]},
                            {**meta, "rows": 2}, {**meta, "cache_parquet_sha256": "changed"}):
                with self.assertRaises(ValueError):
                    check_regression_cache(path, changed, selection, "selection-file")

    def test_both_drivers_check_selection_before_any_audit_target_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(selection=Path(directory) / "missing", output_dir=Path(directory),
                                   protocol_dir=Path(directory))
            with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}), \
                 patch("cache_frozen_research_audit.verify_frozen_selection", side_effect=ValueError("unsealed")), \
                 patch("evaluate_fresh_validation.read_phase_labels") as targets:
                with self.assertRaises(ValueError):
                    produce(args)
                targets.assert_not_called()
            with patch("evaluate_frozen_research_audit.verify_frozen_selection", side_effect=ValueError("unsealed")), \
                 patch("evaluate_fresh_validation.read_phase_labels") as targets:
                with self.assertRaises(ValueError):
                    evaluate(args)
                targets.assert_not_called()

    def test_source_membership_is_global_metadata_only_and_preserves_overlap(self):
        a, b, c = "0000000000000001", "0000000000000002", "0000000000000003"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, catalog = root / "train.parquet", root / "items.parquet"
            pq.write_table(pa.table({"item_id": [a, a, b]}), train)
            pq.write_table(pa.table({"item_id": [a, b, c], "item_microcat_id": [10, 20, 30]}), catalog)
            manifest = {"sources": {"train": identity(train)},
                        "files": {"benchmark_items": {"path": "items.parquet", "sha256": EVAL.sha256_file(catalog)}}}
            selection = {"files": {"public_train_inventory": identity(train)}}
            with patch("evaluate_fresh_validation.read_phase_labels", side_effect=AssertionError("No targets")):
                ids, excluded, categories, details = public_train_control(manifest, root, selection)
            self.assertEqual(ids, {a, b})
            self.assertEqual(excluded, {c})
            self.assertEqual(categories[a], "10")
            self.assertEqual(details["retained_catalog_items"], 2)
            assert_targets_in_public_train({"g": {"relevant_item_ids": [a, b]}}, ids)
            with self.assertRaises(ValueError):
                assert_targets_in_public_train({"g": {"relevant_item_ids": [c]}}, ids)

    def test_source_control_reselects_top50_without_changing_gt_or_original_scores(self):
        positive, missed = "0000000000000001", "0000000000000002"
        negatives = [f"{i:016x}" for i in range(3, 53)]
        items = negatives + [positive]
        group = {"group_id": 0, "group_identity": "g", "query_text": "q", "representative": True,
            "full_relevant_count": 2, "available_relevant_count": 2,
            "local_relevant_count": 0, "nonlocal_relevant_count": 2, "pool_size": 51,
            "positives_in_pool": 1, "baseline_hits": 0, "_available_ids": {positive, missed}}
        scores = [.8] * 50 + [.1]
        row = EVAL.score_cached_group(group, items, [0] * 50 + [1], [False] * 51, None,
            [True] * 50 + [False], [1.] * 51, {100: scores}, original_benchmark_ids={negatives[0]})
        self.assertEqual(scores[-1], .1)
        self.assertEqual(row["s100_hits"], 0)
        self.assertEqual(row["s100_train_source_only_hits"], 1)
        result = source_summary([row], 100)
        self.assertEqual(result["conditional_available"]["reranker"]["text_macro_recall_at_50"], .5)
        self.assertEqual(row["available_relevant_count"], 2)

    def test_microcategory_known_slice_is_not_top3_prediction(self):
        row = {"group_identity": "g"}
        labels = {"g": {"available_item_ids": ["a", "b"]}}
        annotate_known_slices([row], labels, {"a": "10", "b": "20"}, {"10"})
        self.assertFalse(row["all_available_gt_microcategories_known"])
        self.assertTrue(row["any_available_gt_microcategory_known"])

    def test_no_audit_stage_sweep_cli_option_exists(self):
        base = ["--protocol-dir", "protocol", "--selection", "selection", "--cache-dir", "cache",
                "--output-dir", "report"]
        with patch("evaluate_frozen_research_audit.evaluate") as run:
            with self.assertRaises(SystemExit):
                main(base + ["--stage", "150"])
            run.assert_not_called()
            main(base)
            self.assertFalse(hasattr(run.call_args.args[0], "stage"))


if __name__ == "__main__":
    unittest.main()
