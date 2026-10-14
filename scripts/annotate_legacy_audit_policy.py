"""Create a separate audit report with evidenced legacy V4 prior-policy recovery.

This verifies a frozen producer's static code and its fit-cache manifest. It
does not establish how historical training rows were actually prepared, so
the recovered policy is labelled as an inference, not verified train history.
No training rows, model inference, or existing reports are modified.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import tempfile

from reranker_audit_provenance import _file_sha256, artifact_prior_policy
from select_reranker_checkpoint import LEGACY_V4_TRAINING_PRIOR_POLICY, validate_audit_selection


def _dump(node):
    return ast.dump(node, include_attributes=False)


def _expression(source):
    return ast.parse(source, mode="eval").body


def _method(module, class_name, method_name):
    classes = [node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    if len(classes) != 1:
        raise ValueError(f"Frozen producer lacks unique {class_name}")
    methods = [node for node in classes[0].body
               if isinstance(node, ast.FunctionDef) and node.name == method_name]
    if len(methods) != 1:
        raise ValueError(f"Frozen producer lacks unique {class_name}.{method_name}")
    return methods[0]


def _assignment(method, target, expression):
    expected = _dump(_expression(expression))
    matches = [node for node in ast.walk(method) if isinstance(node, ast.Assign)
               and any(isinstance(name, ast.Name) and name.id == target for name in node.targets)
               and _dump(node.value) == expected]
    if len(matches) != 1:
        raise ValueError(f"Frozen producer does not establish expected {target} calculation")
    return matches[0].lineno


def verify_frozen_producer(frozen_source):
    """Verify the known old full-geo/LOO-micro pattern without importing it."""
    root = Path(frozen_source).resolve()
    package = root / "avito_candidates" if (root / "avito_candidates").is_dir() else root
    modules, evidence = {}, {}
    for name in ("cli", "reranker", "geography", "microcategory"):
        path = package / f"{name}.py"
        source = path.read_text(encoding="utf-8")
        modules[name] = ast.parse(source, filename=str(path))
        evidence[name] = {"path": str(path), "sha256": _file_sha256(path)}
    cli = modules["cli"]
    branch_test = _dump(_expression('args.command == "prepare-reranker"'))
    branches = [node for node in ast.walk(cli)
                if isinstance(node, ast.If) and _dump(node.test) == branch_test
                and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                        and call.func.id == "prepare_training_data" for call in ast.walk(node))]
    if len(branches) != 1:
        raise ValueError("Frozen CLI lacks unique prepare-reranker preparation branch")
    branch = branches[0]
    builders = [node for node in ast.walk(branch) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "builder" for target in node.targets)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "FeatureBuilder"]
    if len(builders) != 1 or not any(keyword.arg == "leave_query_out"
                                  and isinstance(keyword.value, ast.Constant)
                                  and keyword.value.value is True for keyword in builders[0].value.keywords):
        raise ValueError("Frozen preparation does not set builder leave_query_out=True")
    calls = [node for node in ast.walk(branch) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "prepare_training_data"]
    if len(calls) != 1 or len(calls[0].args) < 3 or _dump(calls[0].args[2]) != _dump(ast.Name(id="builder", ctx=ast.Load())):
        raise ValueError("Frozen preparation does not pass its LOO builder to training preparation")
    evidence["cli"].update(prepare_branch_line=branch.lineno,
                           builder_leave_query_out_true_line=builders[0].lineno)
    init = _method(modules["reranker"], "FeatureBuilder", "__init__")
    transform = _method(modules["reranker"], "FeatureBuilder", "transform")
    if any(argument.arg in {"leave_geography_query_out", "exclude_query_text"}
           for argument in ast.walk(init) if isinstance(argument, ast.arg)):
        raise ValueError("Frozen FeatureBuilder exposes geography query exclusion")
    total_line = _assignment(transform, "click_total", "associations.totals[location] if associations else 0")
    counts_line = _assignment(transform, "click_counts",
                              "associations.counts[location] if associations and click_total >= 20 else {}")
    feature_calls = [node for node in ast.walk(transform) if isinstance(node, ast.Call)
                     and _dump(node.func) == _dump(_expression("self.microcategory_prior.features"))
                     and any(keyword.arg == "exclude_self"
                             and _dump(keyword.value) == _dump(_expression("self.leave_query_out"))
                             for keyword in node.keywords)]
    if len(feature_calls) != 1:
        raise ValueError("Frozen microcategory features do not use builder leave_query_out")
    forbidden = {"exclude_query_text", "leave_geography_query_out", "query_counts"}
    for name in ("reranker", "geography"):
        if any((isinstance(node, ast.Name) and node.id in forbidden)
               or (isinstance(node, ast.Attribute) and node.attr in forbidden)
               or (isinstance(node, ast.arg) and node.arg in forbidden)
               or (isinstance(node, ast.keyword) and node.arg in forbidden)
               for node in ast.walk(modules[name])):
            raise ValueError("Frozen producer contains a geography self-query exclusion mechanism")
    evidence["reranker"].update(full_geo_total_line=total_line, full_geo_counts_line=counts_line,
                                microcategory_exclude_self_line=feature_calls[0].lineno)
    geo = _method(modules["geography"], "LocationAssociations", "destinations")
    for expression in ("self.totals[location]", "self.counts[location]"):
        if not any(_dump(node) == _dump(_expression(expression)) for node in ast.walk(geo)):
            raise ValueError("Frozen geography destinations do not use full aggregate history")
    evidence["geography"]["full_history_destinations_line"] = geo.lineno
    scores = _method(modules["microcategory"], "MicrocategoryPrior", "scores")
    adjusted_line = _assignment(scores, "adjusted", "count - own.get(category, 0.0) if exclude_self else count")
    total_line = _assignment(scores, "total", "self.token_totals[token] - bool(exclude_self and own)")
    evidence["microcategory"].update(self_contribution_subtraction_line=adjusted_line,
                                    self_total_subtraction_line=total_line)
    return evidence


def _publish_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite policy-annotated report: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", mode="w", encoding="utf-8", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def annotate_legacy_policy(audit_report, frozen_source, fit_cache_metadata, reference_model,
                           output, *, expected_model_sha256):
    import joblib
    report_path, fit_path, model_path, output = map(
        lambda path: Path(path).resolve(), (audit_report, fit_cache_metadata, reference_model, output))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite policy-annotated report: {output}")
    if output in (report_path, fit_path, model_path):
        raise ValueError("Policy recovery needs a new separate report path")
    original_hash = _file_sha256(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    actual_hash = _file_sha256(model_path)
    if not (expected_model_sha256 == actual_hash == report.get("sha256") == fit.get("model_sha256")):
        raise ValueError("Expected model, audit and frozen fit-cache SHA256 must match exactly")
    artifact = joblib.load(model_path)
    validate_audit_selection(report, artifact, actual_hash)
    metadata = artifact["metadata"]
    if (metadata.get("feature_version") != 4 or metadata.get("training_prior_policy") is not None
            or artifact.get("provenance") or report.get("training_prior_policy") is not None):
        raise ValueError("Policy recovery is only for a hashless V4 source/report with missing policy")
    if (fit.get("cache_version") != 2 or fit.get("role") != "train_sample"
            or fit.get("feature_schema_version") != 4 or fit.get("leave_query_out") is not True
            or fit.get("leave_geography_query_out") not in (None, False)
            or fit.get("training_prior_policy") is not None):
        raise ValueError("Fit-cache metadata does not establish the old V4 microcategory-LOO/full-geo producer")
    for field in ("feature_names", "retrieval_config", "corpus_fingerprint", "split", "seed", "validation_fraction"):
        if fit.get(field) != metadata.get(field):
            raise ValueError(f"Frozen fit-cache {field} differs from the audited artifact")
    for field, value in artifact_prior_policy(artifact).items():
        if fit.get(field) != value:
            raise ValueError(f"Frozen fit-cache inference-prior policy differs: {field}")
    modules = verify_frozen_producer(frozen_source)
    annotated = {**report, "training_prior_policy": dict(LEGACY_V4_TRAINING_PRIOR_POLICY),
                 "training_prior_policy_evidence": {
                     "status": "inferred_from_frozenproducer_cache",
                     "policy": dict(LEGACY_V4_TRAINING_PRIOR_POLICY),
                     "historical_training_policy_verified": False,
                     "audited_model_sha256": actual_hash,
                     "original_audit_report": {"path": str(report_path), "sha256": original_hash},
                     "fit_cache_metadata": {"path": str(fit_path), "sha256": _file_sha256(fit_path),
                                             "leave_query_out": True,
                                             "leave_geography_query_out": fit.get("leave_geography_query_out")},
                     "frozen_producer_modules": modules,
                     "caveat": "Static frozen producer and fit-cache flags establish the inferred compatibility policy; the original historical training rows were not independently verified."}}
    _publish_new_json(output, annotated)
    if _file_sha256(report_path) != original_hash:
        raise ValueError("Original report changed while policy evidence was annotated")
    return {"output": str(output), "sha256": _file_sha256(output),
            "original_audit_report_sha256": original_hash, "model_sha256": actual_hash,
            "status": "inferred_from_frozenproducer_cache"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--frozen-source", type=Path, required=True)
    parser.add_argument("--fit-cache-metadata", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(annotate_legacy_policy(args.audit_report, args.frozen_source,
                     args.fit_cache_metadata, args.reference_model, args.output,
                     expected_model_sha256=args.expected_model_sha256), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
