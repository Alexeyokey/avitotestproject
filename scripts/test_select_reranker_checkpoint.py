"""Prefix selection uses tiny pickleable stubs; no fitting or data loading."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import joblib

from reranker_audit_provenance import artifact_prior_policy
from select_reranker_checkpoint import (
    LEGACY_V4_TRAINING_PRIOR_POLICY,
    available_prefixes, select_checkpoint, validate_audit_selection,
    validate_capacity_transfer,
)


class HistGradientBoostingClassifier:
    n_iter_ = 150
    classes_ = [0, 1]
    def __init__(self, *, fit_marker="cold", iterations=150):
        self.n_iter_ = iterations
        self.n_features_in_ = 1
        self.params = {"max_iter": iterations, "random_state": 42,
                       "learning_rate": .1, "max_leaf_nodes": 31,
                       "l2_regularization": 1., "early_stopping": False}
        self._predictors = [{"fit": fit_marker, "stage": i} for i in range(iterations)]
        self._baseline_prediction = [0.]
        self._bin_mapper = {"fit": fit_marker}
    def get_params(self, deep=False):
        return dict(self.params)
    def staged_predict_proba(self, _matrix):
        raise AssertionError("Selection must not run model inference")


class Booster:
    def current_iteration(self):
        return 150


class LGBMClassifier:
    booster_ = Booster()
    n_estimators_ = 999
    def predict_proba(self, _matrix, **_kwargs):
        raise AssertionError("Selection must not run model inference")


class LGBMRanker:
    booster_ = Booster()
    def predict(self, _matrix, **_kwargs):
        raise AssertionError("Selection must not run model inference")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CheckpointSelectionTest(unittest.TestCase):
    def make_fixture(self, directory, *, legacy=False):
        root = Path(directory)
        metadata = {"split": "queries", "validation_fraction": 0.2, "seed": 42,
                    "feature_version": 4, "feature_names": ["rrf_score"],
                    "corpus_fingerprint": "same-inventory",
                    "retrieval_config": {"method": "hybrid", "candidate_k": 449}}
        artifact = {"model": HistGradientBoostingClassifier(), "metadata": metadata,
                    "priors": {}, "estimator": "histgb",
                    "training_configuration": {"fitted_iterations": 150,
                                               "selection_status": "unselected",
                                               "selected_iteration": None}}
        train = root / "train.parquet"
        train.write_bytes(b"tiny opaque training bytes")
        train_json = train.with_suffix(".parquet.json")
        train_json.write_text(json.dumps(metadata), encoding="utf-8")
        if not legacy:
            artifact["provenance"] = {"training_data": str(train),
                                      "training_data_sha256": sha(train),
                                      "training_metadata_sha256": sha(train_json),
                                      "training_priors_sha256": None}
        model = root / "source.joblib"
        joblib.dump(artifact, model)
        validation = {}
        for stage, recall in ((25, 0.8), (50, 0.85), (150, 0.84)):
            validation[str(stage)] = {
                "tune": {"slices": {"all": {"reranker_macro_recall": recall}}},
                "confirmation": {"slices": {"all": {"reranker_macro_recall": recall - 0.01}}},
            }
        report = {"model": str(model), "sha256": sha(model),
                  "model_class": "HistGradientBoostingClassifier", "stages": [25, 50, 150],
                  "prefix_score_type": "positive_class_probability",
                  "retrieval_config": metadata["retrieval_config"],
                  "inference_prior_policy": artifact_prior_policy(artifact),
                  "datasets": {"validation": validation}, "historical_holdout_reused": True,
                  "selection": {"stage_selected_on_tuning_texts": 50,
                                "selection_metric": "tune macro Recall@50 over complete pools",
                                "tune": validation["50"]["tune"],
                                "confirmation": validation["50"]["confirmation"],
                                "confirmation_caveat": "Historical holdout was previously inspected."}}
        report_path = root / "audit.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return model, artifact, train, report_path, report

    def make_transfer_target(self, directory, source, *, legacy=False):
        root = Path(directory)
        target = deepcopy(source)
        target["model"] = HistGradientBoostingClassifier(fit_marker="all-data", iterations=75)
        target["metadata"].update(split="all", validation_fraction=1.0)
        # The all-data history can legitimately differ from the cold history.
        target["priors"] = {"microcategory_prior": SimpleNamespace(
            exact_counts={"ремонт": {"service": 12}}, token_counts={}, token_totals={})}
        target["training_configuration"]["fitted_iterations"] = 75
        train = root / "final-train.parquet"
        train.write_bytes(b"tiny distinct all-data training bytes")
        metadata_path = train.with_suffix(".parquet.json")
        metadata_path.write_text(json.dumps(target["metadata"]), encoding="utf-8")
        prior_path = train.with_suffix(".parquet.priors.joblib")
        joblib.dump(target["priors"], prior_path)
        if legacy:
            target.pop("provenance", None)
        else:
            target["provenance"] = {"training_data": str(train),
                                    "training_data_sha256": sha(train),
                                    "training_metadata_sha256": sha(metadata_path),
                                    "training_priors_sha256": sha(prior_path)}
        path = root / "final-model.joblib"
        joblib.dump(target, path)
        return path, target

    def test_selection_preserves_fitted_model_metadata_and_source_bytes(self):
        with TemporaryDirectory() as directory:
            model, original, _train, report_path, _report = self.make_fixture(directory)
            source_bytes = model.read_bytes()
            output = Path(directory) / "selected.joblib"
            result = select_checkpoint(report_path, output)
            selected = joblib.load(output)
            self.assertEqual(result["selected_iteration"], 50)
            self.assertTrue(result["training_provenance_verified"])
            self.assertEqual(selected["model"].n_iter_, 150)
            self.assertEqual(selected["metadata"], original["metadata"])
            self.assertEqual(selected["priors"], original["priors"])
            self.assertEqual(selected["provenance"], original["provenance"])
            self.assertEqual(selected["training_configuration"]["fitted_iterations"], 150)
            self.assertEqual(selected["training_configuration"]["selected_iteration"], 50)
            selection = selected["selection_provenance"]
            self.assertEqual(selection["audit_report_sha256"], sha(report_path))
            self.assertEqual(selection["source_artifact_sha256"], sha(model))
            self.assertTrue(selection["historical_holdout_reused"])
            self.assertIn("Historical", selection["confirmation_caveat"])
            self.assertEqual(model.read_bytes(), source_bytes)

    def test_legacy_provenance_requires_explicit_switch_and_keeps_caveat(self):
        with TemporaryDirectory() as directory:
            _model, original, _train, report_path, _report = self.make_fixture(directory, legacy=True)
            output = Path(directory) / "selected.joblib"
            with self.assertRaisesRegex(ValueError, "allow-legacy-provenance"):
                select_checkpoint(report_path, output)
            self.assertFalse(output.exists())
            result = select_checkpoint(report_path, output, allow_legacy_provenance=True)
            self.assertFalse(result["training_provenance_verified"])
            selected = joblib.load(output)
            self.assertEqual(selected["metadata"], original["metadata"])
            self.assertFalse(selected["selection_provenance"]["training_provenance_verification"]["verified"])

    def test_report_hash_class_stage_and_tune_choice_are_guarded(self):
        with TemporaryDirectory() as directory:
            model, artifact, _train, _report_path, report = self.make_fixture(directory)
            for case in ("sha", "class", "unavailable", "not-audited", "not-best", "summary"):
                changed = deepcopy(report)
                if case == "sha":
                    changed["sha256"] = "different"
                elif case == "class":
                    changed["model_class"] = "LGBMClassifier"
                elif case == "unavailable":
                    changed["stages"].append(200)
                elif case == "not-audited":
                    changed["selection"]["stage_selected_on_tuning_texts"] = 75
                elif case == "not-best":
                    changed["selection"]["stage_selected_on_tuning_texts"] = 25
                else:
                    changed["selection"]["confirmation"] = None
                with self.subTest(case=case), self.assertRaises(ValueError):
                    validate_audit_selection(changed, artifact, sha(model))

    def test_changed_training_bytes_cannot_be_overridden_by_legacy_switch(self):
        with TemporaryDirectory() as directory:
            _model, _artifact, train, report_path, _report = self.make_fixture(directory)
            train.write_bytes(b"changed training input")
            with self.assertRaisesRegex(ValueError, "training_data_sha256"):
                select_checkpoint(report_path, Path(directory) / "selected.joblib",
                                  allow_legacy_provenance=True)

    def test_output_overwrite_is_refused(self):
        with TemporaryDirectory() as directory:
            _model, _artifact, _train, report_path, _report = self.make_fixture(directory)
            output = Path(directory) / "selected.joblib"
            output.write_bytes(b"preserve existing artifact")
            with self.assertRaises(FileExistsError):
                select_checkpoint(report_path, output)
            self.assertEqual(output.read_bytes(), b"preserve existing artifact")

    def test_lightgbm_uses_actual_booster_iteration_count(self):
        self.assertEqual(available_prefixes(LGBMClassifier()), 150)
        self.assertEqual(available_prefixes(LGBMRanker()), 150)

    def test_capacity_transfer_preserves_final_weights_histories_and_metadata(self):
        with TemporaryDirectory() as directory:
            cold, source, _train, report_path, _report = self.make_fixture(directory)
            final, target = self.make_transfer_target(directory, source)
            cold_bytes, final_bytes = cold.read_bytes(), final.read_bytes()
            output = Path(directory) / "selected-final.joblib"
            result = select_checkpoint(report_path, output, transfer_to_model=final)
            selected = joblib.load(output)
            self.assertEqual(selected["model"]._predictors, target["model"]._predictors)
            self.assertEqual(selected["metadata"], target["metadata"])
            self.assertEqual(selected["provenance"], target["provenance"])
            self.assertEqual(artifact_prior_policy(selected), artifact_prior_policy(target))
            self.assertEqual(selected["training_configuration"]["fitted_iterations"], 75)
            self.assertEqual(selected["selected_iteration"], 50)
            self.assertEqual(selected["selection_status"], "capacity_transferred_from_tune")
            provenance = selected["selection_provenance"]
            self.assertEqual(provenance["source_artifact_sha256"], sha(cold))
            self.assertEqual(provenance["target_artifact_sha256"], sha(final))
            self.assertEqual(provenance["available_iterations"], 75)
            self.assertFalse(provenance["final_weights_directly_evaluated"])
            self.assertTrue(provenance["capacity_selection"]["transferred"])
            self.assertIn("not directly evaluated", provenance["confirmation_caveat"])
            self.assertTrue(result["capacity_transferred"])
            self.assertTrue(result["training_provenance_verified"])
            self.assertEqual(cold.read_bytes(), cold_bytes)
            self.assertEqual(final.read_bytes(), final_bytes)

    def test_transfer_checks_cold_report_sha_before_target(self):
        with TemporaryDirectory() as directory:
            _cold, source, _train, report_path, report = self.make_fixture(directory)
            final, _target = self.make_transfer_target(directory, source)
            report["sha256"] = "tampered"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            output = Path(directory) / "selected-final.joblib"
            with self.assertRaisesRegex(ValueError, "model SHA256"):
                select_checkpoint(report_path, output, transfer_to_model=final)
            self.assertFalse(output.exists())

    def test_legacy_transfer_target_requires_explicit_allowance(self):
        with TemporaryDirectory() as directory:
            _cold, source, _train, report_path, _report = self.make_fixture(directory)
            final, target = self.make_transfer_target(directory, source, legacy=True)
            output = Path(directory) / "selected-final.joblib"
            with self.assertRaisesRegex(ValueError, "allow-legacy-provenance"):
                select_checkpoint(report_path, output, transfer_to_model=final)
            self.assertFalse(output.exists())
            result = select_checkpoint(report_path, output, transfer_to_model=final,
                                       allow_legacy_provenance=True)
            self.assertFalse(result["training_provenance_verified"])
            selected = joblib.load(output)
            self.assertEqual(selected["metadata"], target["metadata"])
            self.assertTrue(selected["selection_provenance"]["cold_source_training_provenance_verification"]["verified"])

    def test_transfer_rejects_incompatible_or_copied_fits(self):
        with TemporaryDirectory() as directory:
            _cold, source, _train, _report, _contents = self.make_fixture(directory)
            _final, target = self.make_transfer_target(directory, source)
            changes = {
                "split": lambda a: a["metadata"].update(split="queries"),
                "seed": lambda a: a["metadata"].update(seed=43),
                "config": lambda a: a["metadata"]["retrieval_config"].update(candidate_k=500),
                "features": lambda a: a["metadata"].update(feature_names=["other"]),
                "schema": lambda a: a["metadata"].update(feature_version=6),
                "corpus": lambda a: a["metadata"].update(corpus_fingerprint="different"),
                "prior-policy": lambda a: a.update(disable_microcategory_prior=True),
                "estimator": lambda a: a.update(estimator="lightgbm"),
                "class": lambda a: a.update(model=LGBMClassifier()),
                "hyperparams": lambda a: a["model"].params.update(learning_rate=.05),
                "small-prefix": lambda a: a.update(model=HistGradientBoostingClassifier(fit_marker="all-data", iterations=25)),
                "copied-fit": lambda a: a.update(model=deepcopy(source["model"])),
            }
            for name, mutate in changes.items():
                changed = deepcopy(target)
                mutate(changed)
                with self.subTest(change=name), self.assertRaises(ValueError):
                    validate_capacity_transfer(source, changed, 50)

    def test_known_legacy_policy_is_recovered_only_from_report_with_explicit_allowance(self):
        with TemporaryDirectory() as directory:
            cold, source, _train, report_path, report = self.make_fixture(directory, legacy=True)
            policy = dict(LEGACY_V4_TRAINING_PRIOR_POLICY)
            report["training_prior_policy"] = policy
            report_path.write_text(json.dumps(report), encoding="utf-8")
            modern_source = deepcopy(source)
            modern_source["metadata"]["training_prior_policy"] = policy
            final, target = self.make_transfer_target(directory, modern_source)
            output = Path(directory) / "selected-final.joblib"
            with self.assertRaisesRegex(ValueError, "allow-legacy-provenance"):
                validate_capacity_transfer(source, target, 50, audit_report=report)
            with self.assertRaisesRegex(ValueError, "training_prior_policy differs"):
                validate_capacity_transfer(source, target, 50, allow_legacy_provenance=True)
            result = select_checkpoint(report_path, output, transfer_to_model=final,
                                       allow_legacy_provenance=True)
            self.assertTrue(result["training_provenance_verified"])
            selected = joblib.load(output)
            compatibility = selected["selection_provenance"]["capacity_selection"]["compatibility"]
            evidence = compatibility["source_training_prior_policy_evidence"]
            self.assertEqual(evidence["policy"], policy)
            self.assertEqual(evidence["status"], "inferred_from_frozenproducer_cache")
            self.assertFalse(evidence["historical_training_policy_verified"])
            self.assertNotIn("training_prior_policy", joblib.load(cold)["metadata"])
            self.assertEqual(selected["metadata"], target["metadata"])

    def test_legacy_recovery_rejects_geography_loo_and_unknown_cache_policy(self):
        with TemporaryDirectory() as directory:
            _cold, source, _train, _report_path, report = self.make_fixture(directory, legacy=True)
            report["training_prior_policy"] = dict(LEGACY_V4_TRAINING_PRIOR_POLICY)
            modern_source = deepcopy(source)
            modern_source["metadata"]["training_prior_policy"] = {
                "microcategory": "leave-query-text-out", "geography": "leave-query-text-out"}
            _final, target = self.make_transfer_target(directory, modern_source)
            with self.assertRaisesRegex(ValueError, "training_prior_policy differs"):
                validate_capacity_transfer(source, target, 50, audit_report=report,
                                           allow_legacy_provenance=True)
            report["training_prior_policy"]["geography"] = "leave-query-text-out"
            with self.assertRaisesRegex(ValueError, "known frozen V4"):
                validate_capacity_transfer(source, target, 50, audit_report=report,
                                           allow_legacy_provenance=True)


if __name__ == "__main__":
    unittest.main()
