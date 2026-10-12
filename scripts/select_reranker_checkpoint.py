"""Publish a new artifact using the tree prefix selected by an audit's tune split.

The fitted model, original training metadata and priors are preserved. Updated
project inference must honor selected_iteration; the frozen baseline runtime
used for retrospective audits does not necessarily support this deployment.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from numbers import Integral
import os
from pathlib import Path
import tempfile

from reranker_audit_provenance import (
    artifact_prior_policy, inference_prior_fingerprint, _file_sha256, _reference_path,
)


MODEL_ESTIMATORS = {
    "HistGradientBoostingClassifier": "histgb",
    "LGBMClassifier": "lightgbm",
    "LGBMRanker": "lambdarank",
}
HISTORICAL_CAVEAT = "Historical validation was previously inspected; confirmation is not a new untouched holdout."
TRANSFER_CAVEAT = "Only tree capacity is transferred from the cold audit; the final all-data model weights and histories were not directly evaluated by that audit."
LEGACY_V4_TRAINING_PRIOR_POLICY = {"microcategory": "leave-query-text-out",
                                   "geography": "full-fit-history"}


def available_prefixes(model):
    """Use fitted iteration counts and supported prediction APIs, never max_iter."""
    name = type(model).__name__
    if name not in MODEL_ESTIMATORS:
        raise ValueError(f"Unsupported selected-prefix model class: {name}")
    if name == "HistGradientBoostingClassifier":
        if not callable(getattr(model, "staged_predict_proba", None)):
            raise ValueError("HGB model has no staged probability API")
        count = getattr(model, "n_iter_", None)
    else:
        booster = getattr(model, "booster_", None)
        if booster is None or not callable(getattr(booster, "current_iteration", None)):
            raise ValueError("LightGBM model has no fitted booster iteration count")
        prediction = "predict" if name == "LGBMRanker" else "predict_proba"
        if not callable(getattr(model, prediction, None)):
            raise ValueError(f"LightGBM model has no {prediction} API")
        count = booster.current_iteration()
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
        raise ValueError("Cannot verify a positive fitted iteration count")
    return int(count)


def validate_audit_selection(report, artifact, source_hash):
    """Bind selection to the exact audited artifact and tune-selected stage."""
    if report.get("sha256") != source_hash:
        raise ValueError("Audit report model SHA256 differs from source artifact")
    model = artifact["model"]
    name = type(model).__name__
    if report.get("model_class") != name:
        raise ValueError("Audit report model class differs from the source artifact")
    available = available_prefixes(model)
    expected = MODEL_ESTIMATORS[name]
    if artifact.get("estimator", expected) != expected:
        raise ValueError("Artifact estimator and model class differ")
    score_type = "raw_ranking_score" if expected == "lambdarank" else "positive_class_probability"
    if report.get("prefix_score_type") != score_type:
        raise ValueError("Audit prefix score type differs from the model class")
    if expected != "lambdarank" and list(getattr(model, "classes_", [0, 1])) != [0, 1]:
        raise ValueError("Selected classifier must score classes [0, 1]")
    metadata = artifact["metadata"]
    if metadata.get("split") != "queries" or not 0 < metadata.get("validation_fraction", 0) < 1:
        raise ValueError("Selected audit artifacts must use an independent query-text split")
    if report.get("retrieval_config") != metadata.get("retrieval_config"):
        raise ValueError("Audit report retrieval configuration differs from training")
    if report.get("inference_prior_policy") != artifact_prior_policy(artifact):
        raise ValueError("Audit report inference-prior policy differs from the artifact")
    if ("model_training_prior_policy" in report
            and report["model_training_prior_policy"] != metadata.get("training_prior_policy")):
        raise ValueError("Audit report training prior policy differs from the artifact")
    stages = report.get("stages")
    if (not isinstance(stages, list) or not stages
            or any(isinstance(stage, bool) or not isinstance(stage, Integral)
                   or not 1 <= stage <= available for stage in stages)
            or stages != sorted(set(stages))):
        raise ValueError("Audit stages must be unique ordered available prefixes")
    selection = report.get("selection", {})
    stage = selection.get("stage_selected_on_tuning_texts")
    if isinstance(stage, bool) or not isinstance(stage, Integral) or stage not in stages:
        raise ValueError("Selected iteration is not an audited available stage")
    validation = report.get("datasets", {}).get("validation", {})
    tune_scores = {}
    for candidate in stages:
        try:
            score = validation[str(candidate)]["tune"]["slices"]["all"]["reranker_macro_recall"]
        except (KeyError, TypeError) as error:
            raise ValueError("Audit lacks tune Recall for an available stage") from error
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= score <= 1:
            raise ValueError("Invalid tune Recall in audit report")
        tune_scores[candidate] = score
    best = min(stages, key=lambda candidate: (-tune_scores[candidate], candidate))
    if stage != best:
        raise ValueError("Report selected iteration is not the best tune Recall prefix")
    chosen = validation[str(stage)]
    if selection.get("tune") != chosen["tune"] or selection.get("confirmation") != chosen.get("confirmation"):
        raise ValueError("Selection summaries do not match the audited selected stage")
    return int(stage), available


def verify_training_provenance(artifact, source_model, *, allow_legacy=False):
    """Verify recorded input hashes, allowing only explicit hashless legacy use."""
    provenance = artifact.get("provenance")
    if not provenance:
        if not allow_legacy:
            raise ValueError("Training provenance is absent; use --allow-legacy-provenance for a documented legacy artifact")
        return {"verified": False, "legacy_provenance_explicitly_allowed": True,
                "caveat": "Original artifact did not record training-input hashes; saved artifact SHA is verified, but original training inputs cannot be independently verified."}
    if not isinstance(provenance, Mapping):
        raise ValueError("Artifact training provenance must be a mapping")
    required = ("training_data", "training_data_sha256", "training_metadata_sha256")
    if any(not provenance.get(key) for key in required):
        raise ValueError("Recorded training provenance is incomplete; legacy allowance cannot override partial provenance")
    source_model = Path(source_model).resolve()
    data = _reference_path(source_model.parent, provenance["training_data"])
    metadata_path = data.with_suffix(data.suffix + ".json")
    verified = {}
    for key, path in (("training_data_sha256", data), ("training_metadata_sha256", metadata_path)):
        actual = _file_sha256(path)
        if actual != provenance[key]:
            raise ValueError(f"Recorded {key} differs from the current training input")
        verified[key] = actual
    training_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if training_metadata != artifact["metadata"]:
        raise ValueError("Verified training metadata differs from the artifact's training metadata")
    prior_hash = provenance.get("training_priors_sha256")
    priors = artifact.get("priors") or {}
    if prior_hash is not None:
        import joblib
        prior_path = data.with_suffix(data.suffix + ".priors.joblib")
        actual = _file_sha256(prior_path)
        if actual != prior_hash:
            raise ValueError("Recorded training_priors_sha256 differs from the current prior input")
        if inference_prior_fingerprint(joblib.load(prior_path)) != inference_prior_fingerprint(priors):
            raise ValueError("Verified training priors differ from the artifact's inference priors")
        verified["training_priors_sha256"] = actual
    elif any(value is not None for value in priors.values()):
        raise ValueError("Inference histories exist but training-prior provenance is missing")
    return {"verified": True, "training_data": str(data), **verified,
            "training_prior_policy": artifact["metadata"].get("training_prior_policy")}


def _fitted_prefix_fingerprint(model, stage):
    """Detect a copied cold fit without executing inference or inspecting data."""
    import joblib
    if type(model).__name__ == "HistGradientBoostingClassifier":
        if any(not hasattr(model, name) for name in
               ("_predictors", "_baseline_prediction", "_bin_mapper")):
            raise ValueError("Cannot verify distinct fitted HGB state for capacity transfer")
        if len(model._predictors) < stage:
            raise ValueError("Actual HGB predictor state lacks the selected prefix")
        state = {"predictors": model._predictors[:stage],
                 "baseline_prediction": model._baseline_prediction,
                 "bin_mapper": model._bin_mapper}
    else:
        booster = model.booster_
        if not callable(getattr(booster, "dump_model", None)):
            raise ValueError("Cannot verify distinct fitted LightGBM state for capacity transfer")
        state = booster.dump_model(num_iteration=stage)
    return joblib.hash(state)


def _transfer_training_prior_policy(source, report, *, allow_legacy=False):
    """Use only the known frozen V4 producer policy for an explicit legacy recovery.

    Cache provenance establishes how the old audit features were produced; it
    does not independently establish how the historical training rows were
    originally prepared. The recovery and this limitation remain recorded.
    """
    metadata = source["metadata"]
    policy = metadata.get("training_prior_policy")
    if policy is not None:
        return policy, {"policy": policy, "status": "declared_in_source_metadata"}
    cached = (report or {}).get("training_prior_policy")
    if cached is None:
        return None, {"policy": None, "status": "not_recorded"}
    if not allow_legacy:
        raise ValueError("Legacy training-prior policy recovery requires --allow-legacy-provenance")
    if metadata.get("feature_version") != 4 or cached != LEGACY_V4_TRAINING_PRIOR_POLICY:
        raise ValueError("Legacy policy recovery supports only the known frozen V4 microcategory-LOO/full-history geography policy")
    if source.get("provenance"):
        raise ValueError("Frozen producer policy recovery is only permitted for a hashless legacy source artifact")
    return dict(cached), {"policy": dict(cached),
                          "status": "inferred_from_frozenproducer_cache",
                          "source": "audit_report.training_prior_policy",
                          "historical_training_policy_verified": False,
                          "caveat": "The known frozen producer/cache policy is used for compatibility; historical training-row policy was not independently verified."}


def validate_capacity_transfer(source, target, stage, *, audit_report=None,
                               allow_legacy_provenance=False):
    """Permit capacity transfer only between compatible, independently fitted models."""
    import joblib
    if not isinstance(target, Mapping) or "metadata" not in target or "model" not in target:
        raise ValueError("Transfer target must be a saved reranker artifact")
    cold_model, final_model = source["model"], target["model"]
    name = type(cold_model).__name__
    if type(final_model) is not type(cold_model):
        raise ValueError("Capacity transfer requires the exact same model class")
    estimator = MODEL_ESTIMATORS[name]
    if target.get("estimator", estimator) != estimator:
        raise ValueError("Transfer target estimator differs from the model class")
    available = available_prefixes(final_model)
    if stage > available:
        raise ValueError("Transfer target lacks the selected available prefix")
    if estimator != "lambdarank" and list(getattr(final_model, "classes_", [])) != [0, 1]:
        raise ValueError("Transfer target classifier must score classes [0, 1]")
    before, after = source["metadata"], target["metadata"]
    if after.get("split") != "all":
        raise ValueError("Capacity transfer target must use split=all")
    for field in ("seed", "feature_version", "feature_names", "retrieval_config",
                  "corpus_fingerprint"):
        if before.get(field) != after.get(field):
            raise ValueError(f"Capacity transfer {field} differs between cold and final models")
    policy, policy_evidence = _transfer_training_prior_policy(
        source, audit_report, allow_legacy=allow_legacy_provenance)
    if policy != after.get("training_prior_policy"):
        raise ValueError("Capacity transfer training_prior_policy differs between cold and final models")
    features = before.get("feature_names")
    if not isinstance(features, list) or not features or len(set(features)) != len(features):
        raise ValueError("Capacity transfer requires a declared unique feature schema")
    for model in (cold_model, final_model):
        if getattr(model, "n_features_in_", None) != len(features):
            raise ValueError("Capacity transfer fitted feature count differs from metadata")
    if artifact_prior_policy(source)["disable_microcategory_prior"] != artifact_prior_policy(target)["disable_microcategory_prior"]:
        raise ValueError("Capacity transfer microcategory-prior enable policy differs")
    parameters = []
    capacity_key = "max_iter" if estimator == "histgb" else "n_estimators"
    for model in (cold_model, final_model):
        if not callable(getattr(model, "get_params", None)):
            raise ValueError("Capacity transfer cannot verify model hyperparameters")
        params = dict(model.get_params(deep=False))
        capacity = params.pop(capacity_key, None)
        if isinstance(capacity, bool) or not isinstance(capacity, Integral) or capacity < stage:
            raise ValueError("Declared model capacity lacks the selected prefix")
        parameters.append(params)
    if joblib.hash(parameters[0]) != joblib.hash(parameters[1]):
        raise ValueError("Capacity transfer model hyperparameters differ beyond iteration capacity")
    fingerprints = [_fitted_prefix_fingerprint(model, stage) for model in (cold_model, final_model)]
    if fingerprints[0] == fingerprints[1]:
        raise ValueError("Capacity transfer target copies the audited fitted prefix; it is not a distinct all-data fit")
    return available, {"model_class": name, "estimator": estimator,
                       "hyperparameters_except_capacity": parameters[0],
                       "capacity_parameter": capacity_key,
                       "source_training_prior_policy_evidence": policy_evidence,
                       "source_fitted_prefix_fingerprint": fingerprints[0],
                       "target_fitted_prefix_fingerprint": fingerprints[1]}


def _publish_new_artifact(path, artifact):
    """Publish a complete joblib with exclusive creation, even under a race."""
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite selected artifact: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        joblib.dump(artifact, temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        # link() creates the target atomically and refuses an existing name;
        # model bytes are complete before the deployment path becomes visible.
        os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def select_checkpoint(audit_report, output, source_model=None, *, allow_legacy_provenance=False,
                      transfer_to_model=None):
    """Save a new prefix-selected artifact without retraining or modifying trees."""
    import joblib
    report_path, output = Path(audit_report).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite selected artifact: {output}")
    report_hash = _file_sha256(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reference = source_model or report.get("model")
    if not reference:
        raise ValueError("Audit report lacks a model path; pass --source-model")
    source = _reference_path(report_path.parent, reference)
    if source == output:
        raise ValueError("Selected artifact must use a new path")
    source_hash = _file_sha256(source)
    artifact = joblib.load(source)
    if not isinstance(artifact, Mapping) or "metadata" not in artifact or "model" not in artifact:
        raise ValueError("Expected a saved reranker artifact with model and metadata")
    stage, available = validate_audit_selection(report, artifact, source_hash)
    audited_source_available = available
    training = verify_training_provenance(artifact, source,
                                         allow_legacy=allow_legacy_provenance)
    deployment, deployment_source, deployment_hash = artifact, source, source_hash
    deployment_training, transfer = training, None
    if transfer_to_model is not None:
        deployment_source = _reference_path(report_path.parent, transfer_to_model)
        if deployment_source in (source, output):
            raise ValueError("Capacity transfer must use a distinct target fit and new output path")
        deployment_hash = _file_sha256(deployment_source)
        if deployment_hash == source_hash:
            raise ValueError("Capacity transfer target is identical to the cold source artifact")
        deployment = joblib.load(deployment_source)
        available, transfer = validate_capacity_transfer(
            artifact, deployment, stage, audit_report=report,
            allow_legacy_provenance=allow_legacy_provenance)
        deployment_training = verify_training_provenance(
            deployment, deployment_source, allow_legacy=allow_legacy_provenance)
    status = "capacity_transferred_from_tune" if transfer else "selected_on_tune"
    selected = dict(deployment)
    selected.update(selected_iteration=stage, selection_status=status)
    selected["selection_provenance"] = {
        "version": 1, "source_model": str(source), "source_artifact_sha256": source_hash,
        "audit_report": str(report_path), "audit_report_sha256": report_hash,
        "model_class": type(artifact["model"]).__name__, "available_iterations": available,
        "selected_iteration": stage, "audited_stages": report["stages"],
        "selection_metric": report["selection"].get("selection_metric"),
        "selection": report["selection"],
        "historical_holdout_reused": bool(report.get("historical_holdout_reused")),
        "confirmation_caveat": (report["selection"].get("confirmation_caveat")
                                or HISTORICAL_CAVEAT),
        "training_provenance_verification": deployment_training,
        "inference_prior_policy": artifact_prior_policy(deployment),
        "candidate_pool_digest": report.get("candidate_pool_digest"),
        "requires_selected_iteration_inference": True,
    }
    if transfer:
        selected["selection_provenance"].update(
            version=2, target_model=str(deployment_source),
            target_artifact_sha256=deployment_hash,
            cold_source_training_provenance_verification=training,
            audited_source_available_iterations=audited_source_available,
            audited_source_inference_prior_policy=artifact_prior_policy(artifact),
            final_weights_directly_evaluated=False,
            capacity_selection={"transferred": True, "selected_iteration": stage,
                                "audited_cold_model": str(source),
                                "audited_cold_artifact_sha256": source_hash,
                                "final_fit_model": str(deployment_source),
                                "final_fit_artifact_sha256": deployment_hash,
                                "compatibility": transfer, "caveat": TRANSFER_CAVEAT})
        selected["selection_provenance"]["confirmation_caveat"] += " " + TRANSFER_CAVEAT
    if isinstance(deployment.get("training_configuration"), Mapping):
        selected["training_configuration"] = {
            **deployment["training_configuration"], "selected_iteration": stage,
            "selection_status": status}
    _publish_new_artifact(output, selected)
    return {"output": str(output), "sha256": _file_sha256(output),
            "selected_iteration": stage, "available_iterations": available,
            "audit_report_sha256": report_hash,
            "training_provenance_verified": deployment_training["verified"],
            "capacity_transferred": bool(transfer), "target_artifact_sha256": deployment_hash}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-report", "--report", type=Path, required=True)
    parser.add_argument("--source-model", "--model", type=Path)
    parser.add_argument("--transfer-to-model", type=Path,
                        help="Transfer only the audited tree capacity to a compatible distinct split=all fit")
    parser.add_argument("--output", "--output-model", type=Path, required=True)
    parser.add_argument("--allow-legacy-provenance", action="store_true")
    args = parser.parse_args(argv)
    result = select_checkpoint(args.audit_report, args.output, args.source_model,
                               allow_legacy_provenance=args.allow_legacy_provenance,
                               transfer_to_model=args.transfer_to_model)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
