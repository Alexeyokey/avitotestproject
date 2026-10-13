"""Seal a Tune-chosen pipeline for one frozen regression on the observed Audit.

This is a checksum seal, not a cryptographic signature. The previous Audit
was already inspected for the old baseline, so this cannot create a new
independent test. It prevents the new pipeline/stage from changing before
the regression check and binds every material input and producer source.
This script never parses Audit target IDs and never launches a cache job.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import evaluate_fresh_validation as EVAL
from cache_research_reranker import (validate_research_artifact, research_prior_policy,
                                    check_training_prior_file, check_sources)
from evaluate_research_reranker import (check_frozen_reference, check_research_cache,
                                       FROZEN_REFERENCE_MODEL_SHA256)
from produce_document_hypotheses import check_encoder_base

REGRESSION_POLICY = {
    "version": 1, "purpose": "frozen regression",
    "previous_baseline_audit_observed": True, "independent_fresh_test": False,
    "audit_use": "one frozen selected new prefix versus byte-identical original HGB150",
    "capacity_selection": "explicit choice among stages already reported on Tune",
    "retuning_after_audit": False, "new_stage_sweep_on_audit": False,
    "gt_policy": "full available and unavailable targets preserved",
    "maximum_candidate_budget": 998,
}
REQUIRED_FILES = {"protocol_manifest", "tune_report", "tune_group_scores",
    "new_model", "new_train_data", "new_tune_cache", "new_tune_metadata",
    "old_model", "old_train_data", "old_tune_cache", "old_tune_metadata",
    "encoder_base_reference", "new_train_metadata", "new_train_priors",
    "microcategory_model", "microcategory_metadata", "old_audit_cache",
    "old_audit_metadata", "public_train_inventory", "dense_index_binary",
    "dense_index_metadata"}


def regression_source_paths():
    """Bind imported scoring/denominator/retrieval code, not only entry points.

    The old Tune producer snapshot is retained verbatim. This extra closure
    belongs solely to the new regression wrapper; it changes no old source.
    """
    root = Path(__file__).resolve().parents[1]
    scripts = ("freeze_research_selection.py", "cache_frozen_research_audit.py",
        "evaluate_frozen_research_audit.py", "cache_research_reranker.py",
        "evaluate_research_reranker.py", "evaluate_fresh_validation.py",
        "audit_reranker_learning_curve.py", "cache_reranker_audit.py",
        "reranker_audit_provenance.py", "build_fresh_validation_protocol.py",
        "produce_document_hypotheses.py", "produce_retrieval_hypotheses.py",
        "evaluate_retrieval_hypotheses.py")
    return tuple(sorted({*(root / "scripts" / name for name in scripts),
                         *(root / "src/avito_candidates").rglob("*.py")}))


def check_regression_sources(snapshot):
    if not isinstance(snapshot, dict) or not snapshot:
        raise ValueError("Frozen regression selection must bind producer sources")
    required = {str(path.resolve()) for path in regression_source_paths()}
    if not required <= set(snapshot):
        raise ValueError("Frozen regression source closure lacks scoring/denominator/retrieval code")
    check_sources(snapshot)


def select_dense_index(directory, config, catalog, *, fingerprint=None, records=None, cache_version=None):
    """Choose one completed index by config AND actual ordered document input.

    Reuse the existing fingerprint utility without loading an encoder. A saved
    model signature is enough to identify candidates here; the producer also
    verifies the actually restored encoder's signature before opening targets.
    Multiple matching completed pairs fail closed instead of choosing by mtime.
    """
    if fingerprint is None or cache_version is None:
        from avito_candidates.dense import _fingerprint, _CACHE_VERSION
        fingerprint = _fingerprint if fingerprint is None else fingerprint
        cache_version = _CACHE_VERSION if cache_version is None else cache_version
    if records is None:
        from produce_document_hypotheses import iter_catalog_records
        records = iter_catalog_records
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError("Frozen regression needs its completed dense-index directory")
    document_format = config.get("embedding_document_format", "current")
    expected = {"version": cache_version, "model": config["embedding_model"],
        "max_seq_length": config["embedding_max_length"],
        "params_words": config["embedding_params_words"],
        "description_words": config["embedding_description_words"]}
    matches = []
    for metadata_path in sorted(directory.glob("*.json")):
        if not re.fullmatch(r"[0-9a-f]{20}", metadata_path.stem):
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            # An unrelated generic-cache entry cannot become the selected pair.
            continue
        if (not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected.items())
                or metadata.get("document_format", "current") != document_format):
            continue
        signature = metadata.get("model_signature", "")
        if not isinstance(signature, str):
            raise ValueError("Completed dense-index model signature is invalid")
        count = [0]
        def documents():
            for row in records(catalog):
                count[0] += 1
                yield row
        calculated = fingerprint(documents(), config["embedding_model"],
            config["embedding_max_length"], config["embedding_params_words"],
            config["embedding_description_words"], signature, document_format=document_format)
        binary = metadata_path.with_suffix(".bin")
        if (calculated == metadata_path.stem and binary.is_file()
                and isinstance(metadata.get("items"), int) and not isinstance(metadata.get("items"), bool)
                and metadata["items"] == count[0] and count[0] > 0
                and isinstance(metadata.get("dimension"), int)
                and not isinstance(metadata.get("dimension"), bool) and metadata["dimension"] > 0):
            matches.append({"directory": str(directory), "fingerprint": calculated,
                            "binary": identity(binary), "metadata": identity(metadata_path)})
    if len(matches) != 1:
        raise ValueError("Frozen regression requires exactly one completed dense index matching config/corpus")
    return matches[0]


def verify_dense_index(selection, directory=None):
    """Bind only the selected completed pair; unrelated cache files are allowed."""
    index = selection.get("dense_index")
    if (not isinstance(index, dict) or not isinstance(index.get("fingerprint"), str)
            or not re.fullmatch(r"[0-9a-f]{20}", index["fingerprint"])):
        raise ValueError("Frozen selection lacks its exact completed dense-index identity")
    root = Path(index.get("directory", "")).resolve()
    if directory is not None and Path(directory).resolve() != root:
        raise ValueError("Dense-cache path differs from the frozen selected index")
    for kind, suffix in (("binary", ".bin"), ("metadata", ".json")):
        entry = index.get(kind)
        if (not isinstance(entry, dict) or entry != selection["files"].get(f"dense_index_{kind}")
                or Path(entry.get("path", "")).resolve() != root / (index["fingerprint"] + suffix)):
            raise ValueError("Frozen dense-index pair path or material-file binding differs")
        verify_files({f"dense_index_{kind}": entry})
    return index


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": EVAL.sha256_file(path)}


def validate_tune_report(report, stage):
    if (report.get("report_version") != 1 or report.get("phase") != "tune"
            or report.get("exploratory") is not True or report.get("audit_labels_read") is not False
            or report.get("selection_performed") is not False
            or report.get("old_stage") != 150):
        raise ValueError("Freeze requires the completed exploratory Tune pipeline report")
    if isinstance(stage, bool) or stage not in (75, 100, 150):
        raise ValueError("Selected stage must be an explicitly reported 75/100/150 Tune prefix")
    curve = report.get("new_capacity_curve", {})
    if str(stage) not in curve:
        raise ValueError("Selected stage is absent from the completed Tune curve")
    metric = curve[str(stage)]["conditional_available"]["reranker"]["text_macro_recall_at_50"]
    if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not 0 <= metric <= 1:
        raise ValueError("Selected stage lacks a valid Tune text-macro Recall")
    return metric


def verify_files(files):
    if not isinstance(files, dict) or not files:
        raise ValueError("Frozen selection needs its complete file closure")
    for name, entry in files.items():
        if (not isinstance(entry, dict) or not isinstance(entry.get("path"), str)
                or not Path(entry["path"]).is_file()
                or EVAL.sha256_file(Path(entry["path"])) != entry.get("sha256")):
            raise ValueError(f"Frozen input changed or is missing: {name}")


def verify_frozen_selection(path):
    path = Path(path)
    selection = json.loads(path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in selection.items() if key != "selection_sha256"}
    if (selection.get("selection_version") != 1 or selection.get("policy") != REGRESSION_POLICY
            or selection.get("selection_sha256") != EVAL.canonical_sha256(payload)):
        raise ValueError("Frozen regression selection checksum or policy differs")
    verify_files(selection.get("files"))
    check_regression_sources(selection.get("source_sha256"))
    files = selection["files"]
    if not REQUIRED_FILES <= set(files):
        raise ValueError("Frozen selection lacks a required material input")
    verify_dense_index(selection)
    report = json.loads(Path(files["tune_report"]["path"]).read_text(encoding="utf-8"))
    metric = validate_tune_report(report, selection.get("selected_stage"))
    if (selection.get("selected_tune_text_macro_recall") != metric
            or report.get("protocol_sha256") != files["protocol_manifest"]["sha256"]
            or report.get("group_scores_sha256") != files["tune_group_scores"]["sha256"]
            or files["old_model"]["sha256"] != FROZEN_REFERENCE_MODEL_SHA256):
        raise ValueError("Frozen stage/report/protocol/reference identity differs")
    for prefix, key in (("new", "new_pipeline_identity"), ("old", "old_pipeline_identity")):
        for field, suffix in (("model", "model"), ("training_data", "train_data"),
                              ("cache", "tune_cache"), ("metadata", "tune_metadata")):
            if report[key][field] != files[f"{prefix}_{suffix}"]:
                raise ValueError("Frozen model/cache differs from the completed Tune report")
    return selection


def freeze(args):
    if args.output.exists():
        raise FileExistsError("Frozen selection is immutable; do not overwrite it")
    report_path = args.tune_report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metric = validate_tune_report(report, args.stage)
    manifest, protocol_hash = EVAL.checked_protocol(args.protocol_dir)
    if report.get("protocol_sha256") != protocol_hash:
        raise ValueError("Tune report uses a different frozen validation protocol")
    files = {"protocol_manifest": identity(args.protocol_dir / "manifest.json"),
             "tune_report": identity(report_path),
             "tune_group_scores": identity(report_path.parent / "research-tune-group-scores.jsonl")}
    if files["tune_group_scores"]["sha256"] != report.get("group_scores_sha256"):
        raise ValueError("Completed Tune group-score bytes changed")
    for prefix, key in (("new", "new_pipeline_identity"), ("old", "old_pipeline_identity")):
        for field, suffix in (("model", "model"), ("training_data", "train_data"),
                              ("cache", "tune_cache"), ("metadata", "tune_metadata")):
            entry = report[key][field]
            if identity(entry["path"]) != entry:
                raise ValueError("Tune model/cache bytes changed before the regression freeze")
            files[f"{prefix}_{suffix}"] = entry
    check_frozen_reference(files["old_model"]["path"], files["old_train_data"]["path"])
    import joblib
    artifact = joblib.load(files["new_model"]["path"])
    metadata = json.loads(Path(files["new_tune_metadata"]["path"]).read_text(encoding="utf-8"))
    fit_path = EVAL.protocol_file(args.protocol_dir, manifest, "fit_texts")
    fit_texts = set(json.loads(fit_path.read_text(encoding="utf-8"))["texts"])
    validate_research_artifact(artifact, fit_texts, manifest)
    EVAL.check_encoder_identity(artifact, manifest)
    EVAL.check_training_data(files["new_train_data"]["path"], artifact, args.protocol_dir, manifest)
    check_training_prior_file(files["new_train_data"]["path"], artifact)
    check_research_cache(metadata, artifact, files["new_train_data"]["sha256"])
    if args.stage > EVAL.maximum_iterations(artifact["model"]):
        raise ValueError("Frozen selected prefix is absent from the saved HGB model")
    base = metadata["research_encoder_base"]
    reference = Path(base["reference"]["path"])
    if check_encoder_base(artifact["metadata"]["retrieval_config"]["embedding_model"], manifest, reference) != base:
        raise ValueError("Frozen cached adapter base differs from the completed Tune run")
    files["encoder_base_reference"] = identity(reference)
    data = Path(files["new_train_data"]["path"])
    files["new_train_metadata"] = identity(data.with_suffix(data.suffix + ".json"))
    files["new_train_priors"] = identity(data.with_suffix(data.suffix + ".priors.joblib"))
    config = artifact["metadata"]["retrieval_config"]
    files["microcategory_model"] = identity(config["microcategory_model"])
    micro_path = Path(config["microcategory_model"])
    files["microcategory_metadata"] = identity(micro_path.with_suffix(micro_path.suffix + ".json"))
    old_cache, old_metadata = EVAL.phase_cache_paths(args.old_audit_cache_dir, "audit")
    old_meta = json.loads(old_metadata.read_text(encoding="utf-8"))
    # Metadata/Parquet-integrity checks do not parse the Audit GT file.
    EVAL.check_phase_cache_inputs(old_cache, old_meta, "audit", protocol_hash,
                                 FROZEN_REFERENCE_MODEL_SHA256)
    files["old_audit_cache"], files["old_audit_metadata"] = identity(old_cache), identity(old_metadata)
    public_train = manifest["sources"]["train"]
    files["public_train_inventory"] = identity(public_train["path"])
    if files["public_train_inventory"]["sha256"] != public_train["sha256"]:
        raise ValueError("Original public training source changed before the regression freeze")
    catalog = EVAL.protocol_file(args.protocol_dir, manifest, "benchmark_items")
    dense_index = select_dense_index(args.dense_cache, config, catalog)
    files["dense_index_binary"], files["dense_index_metadata"] = dense_index["binary"], dense_index["metadata"]
    source = dict(metadata["research_source_sha256"])
    for path in regression_source_paths():
        source[str(path.resolve())] = EVAL.sha256_file(path)
    check_regression_sources(source)
    selection = {"selection_version": 1, "policy": REGRESSION_POLICY,
        "selected_stage": args.stage, "selected_tune_text_macro_recall": metric,
        "selection_reason": args.reason, "files": files, "source_sha256": source,
        "retrieval_config": config, "research_prior_policy": research_prior_policy(artifact),
        "dense_index": dense_index,
        "encoder_base_reference": base,
        "source_read_policy": "Audit labels not parsed while sealing; old Audit cache identity checked only"}
    selection["selection_sha256"] = EVAL.canonical_sha256(selection)
    verify_files(files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    EVAL.write_new_json(args.output, selection)
    verify_frozen_selection(args.output)
    print(json.dumps({"frozen": True, "stage": args.stage, "selection": str(args.output),
                      "independent_fresh_test": False}), flush=True)
    return selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--tune-report", type=Path, required=True)
    parser.add_argument("--old-audit-cache-dir", type=Path, required=True)
    parser.add_argument("--dense-cache", type=Path, required=True)
    parser.add_argument("--stage", type=int, choices=[75, 100, 150], required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    freeze(args)


if __name__ == "__main__":
    main()
