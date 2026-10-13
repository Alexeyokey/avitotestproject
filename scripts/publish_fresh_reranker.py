"""Publish a new runtime-selected HGB only after its sealed fresh audit is complete.

The full fitted predictor remains intact. Reranker uses selected_iteration for
inference; publication never chooses capacity from audit results or claims an
official-benchmark improvement. Inputs and earlier artifacts are not changed.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import re
import tempfile

import joblib

import evaluate_fresh_validation as evaluation


STATUS = "selected-on-fresh-tune; sealed-audit-complete"
SCOPE = ("Fresh cold-query tail cohort and the protocol's sampled catalog only. "
         "This artifact is not validated for the original official benchmark corpus.")


def checked_audit(evaluation_dir, manifest, protocol_hash, model_hash, train_hash,
                  encoder_hash, stage):
    root = Path(evaluation_dir).resolve()
    report_path = root / "audit-report.json"
    scores_path = root / "audit-group-scores.jsonl"
    if not report_path.is_file() or not scores_path.is_file():
        raise ValueError("Complete audit report and group scores are required before publication")
    report_hash = evaluation.sha256_file(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("report_version") != 1 or report.get("phase") != "audit"
            or report.get("policy") != evaluation.POLICY
            or report.get("protocol_sha256") != protocol_hash
            or report.get("model_sha256") != model_hash
            or report.get("train_data_sha256") != train_hash
            or report.get("encoder_content_sha256") != encoder_hash
            or report.get("selected_stage") != stage
            or not isinstance(report.get("stages"), dict)
            or set(report["stages"]) != {str(stage)}
            or not isinstance(report.get("scope"), str) or not report["scope"]):
        raise ValueError("Audit identity, policy or stages differ from the sealed tune selection")
    scores_hash = evaluation.sha256_file(scores_path)
    if report.get("group_scores_sha256") != scores_hash:
        raise ValueError("Audit group score bytes differ from the completed report")
    rows = []
    with scores_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            stages = {int(match.group(1)) for key in row
                      if (match := re.match(r"s([0-9]+)_", key))}
            if stages != {stage}:
                raise ValueError("Audit group scores contain an unselected or missing stage")
            rows.append(row)
    expected_groups = manifest.get("stats", {}).get("audit", {}).get("groups")
    if (not isinstance(expected_groups, int) or isinstance(expected_groups, bool)
            or expected_groups < 1 or len(rows) != expected_groups):
        raise ValueError("Completed audit must retain every protocol query group")
    # Recompute only recorded summaries; no model prediction or label loading.
    if report["stages"][str(stage)] != evaluation.summarize(rows, stage):
        raise ValueError("Audit summary does not match its complete sealed group scores")
    return report, report_path, report_hash, scores_path, scores_hash


def publish_new(path, artifact):
    """Publish complete bytes atomically with no overwrite, including a race."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Selected artifact already exists: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        joblib.dump(artifact, temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        # Unlike replace(), link() cannot overwrite a name created after the
        # initial check. The same publication contract is used by our selector.
        os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def publish(protocol_dir, evaluation_dir, model, train_data, output):
    protocol_dir, evaluation_dir = Path(protocol_dir), Path(evaluation_dir).resolve()
    model, train_data, output = map(lambda path: Path(path).resolve(), (model, train_data, output))
    if output == model or output.exists():
        raise FileExistsError("Publication requires a new output path; source artifact is preserved")
    manifest, protocol_hash = evaluation.checked_protocol(protocol_dir)
    model_hash = evaluation.sha256_file(model)
    artifact = joblib.load(model)
    if (not isinstance(artifact, Mapping) or "model" not in artifact or "metadata" not in artifact
            or artifact.get("estimator", "histgb") != "histgb"
            or artifact.get("selected_iteration") is not None
            or evaluation.maximum_iterations(artifact["model"]) < 150):
        raise ValueError("Publication requires the unselected fresh HGB fitted to 150 iterations")
    encoder_hash = evaluation.check_encoder_identity(artifact, manifest)
    train_hash = evaluation.check_training_data(train_data, artifact, protocol_dir, manifest)
    correction = artifact["metadata"].get("category_sentinel_correction", {})
    if correction.get("version") != 1 or correction.get("derived_parquet_sha256") != train_hash:
        raise ValueError("Fresh fitted model must use its proven category-corrected training cache")
    selection_path = evaluation_dir / "selection.json"
    if not selection_path.is_file():
        raise ValueError("Tune selection seal is required before publication")
    selection_hash = evaluation.sha256_file(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    stage = evaluation.verify_selection(selection, protocol_hash, model_hash, evaluation_dir, train_hash)
    if stage > evaluation.maximum_iterations(artifact["model"]):
        raise ValueError("Selected prefix exceeds the fitted model")
    _report, report_path, report_hash, scores_path, scores_hash = checked_audit(
        evaluation_dir, manifest, protocol_hash, model_hash, train_hash, encoder_hash, stage)
    # Prevent publishing a result if its source or selection changed mid-check.
    for path, expected in ((model, model_hash), (selection_path, selection_hash),
                           (report_path, report_hash), (scores_path, scores_hash),
                           (train_data, train_hash)):
        if evaluation.sha256_file(path) != expected:
            raise ValueError("Publication inputs changed while their proof was being verified")
    selected = dict(artifact)
    selected.update(selected_iteration=stage, selection_status=STATUS)
    if isinstance(artifact.get("training_configuration"), Mapping):
        selected["training_configuration"] = {**artifact["training_configuration"],
                                               "selected_iteration": stage,
                                               "selection_status": STATUS}
    selected["fresh_selection"] = {
        "version": 1, "scope": SCOPE, "official_benchmark_validated": False,
        "capacity_chosen_from": "fresh tune only", "audit_used_for_capacity_selection": False,
        "requires_selected_iteration_inference": True,
        "selected_iteration": stage, "available_iterations": evaluation.maximum_iterations(artifact["model"]),
        "source_model": str(model), "source_model_sha256": model_hash,
        "protocol_manifest": str((protocol_dir / "manifest.json").resolve()),
        "protocol_sha256": protocol_hash, "encoder_content_sha256": encoder_hash,
        "train_data": str(train_data), "train_data_sha256": train_hash,
        "selection": str(selection_path), "selection_sha256": selection_hash,
        "tune_report": str(evaluation_dir / "tune-report.json"),
        "tune_report_sha256": selection["tune_report_sha256"],
        "audit_report": str(report_path), "audit_report_sha256": report_hash,
        "audit_group_scores": str(scores_path), "audit_group_scores_sha256": scores_hash,
        "producer_sha256": evaluation.sha256_file(Path(__file__)),
        "corpus": manifest.get("corpus", {}),
    }
    publish_new(output, selected)
    return {"output": str(output), "sha256": evaluation.sha256_file(output),
            "selected_iteration": stage, "selection_status": STATUS, "scope": SCOPE}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(publish(**vars(args)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
