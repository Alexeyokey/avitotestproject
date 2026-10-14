"""Two one-factor HGB comparisons on the immutable cached geographic Tune pool.

Primary comparisons fix all models at 50 trees. The four-stage capacity curves
are secondary diagnostics; they cannot silently replace the primary endpoint.
No retrieval, encoder loading, training or Audit access occurs. All original
stage predictions must reproduce per-context Recall before any new result is
published. Tune was already consumed and remains exploratory.
"""
from __future__ import annotations

import argparse
import copy
import inspect
import json
from pathlib import Path
import time

import numpy as np

from avito_candidates.reranker import V8_FEATURES
from cache_research_reranker import research_prior_policy
from evaluate_geo_dense_retrained import STAGES, choose_conservative_stage, identity
from evaluate_research_reranker import named_summary, paired_pipeline_bootstrap
from summarize_geo_dense_retraining import (checked_identity, validate_row, verify_feature_cache,
    verify_selected_model, verify_summary)
import evaluate_fresh_validation as EVAL


VARIANTS = {"maxfeatures08": {"max_features": .8}, "leaves15": {"max_leaf_nodes": 15}}
PRIMARY_STAGE = 50
TIE_POLICY = "model probability descending, RRF score descending, original pool position ascending"


def check_source_identity(snapshot):
    """JSON/source snapshots carry string paths; the hash helper needs Path."""
    for raw_path, digest in snapshot.items():
        path = Path(raw_path)
        if not path.is_file() or EVAL.sha256_file(path) != digest:
            raise ValueError("Evaluation implementation changed while scoring")


def validate_variant(artifact, source, name):
    """Bind the changed estimator to the exact same fitted training inputs."""
    if name not in VARIANTS:
        raise ValueError("Only the two predeclared separate variants are allowed")
    expected = {**source["model"].get_params(), **VARIANTS[name]}
    if (artifact.get("estimator") != "histgb" or artifact.get("selected_iteration") is not None
            or EVAL.maximum_iterations(artifact["model"]) != 150
            or artifact["model"].get_params() != expected
            or artifact.get("metadata") != source.get("metadata")
            or artifact.get("training_configuration", {}).get("parameters") != expected):
        raise ValueError("Variant must change exactly one declared HGB parameter and preserve training metadata")
    for key in ("training_data_sha256", "training_metadata_sha256", "training_priors_sha256", "training_prior_policy"):
        if artifact.get("provenance", {}).get(key) != source.get("provenance", {}).get(key):
            raise ValueError("Regularization variant changed training data, weights or prior provenance")
    if research_prior_policy(artifact) != research_prior_policy(source):
        raise ValueError("Regularization variant changed saved inference priors")


def read_reference(report_path, scores_path, expected_count):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if (not report.get("complete") or not report.get("full_tune_complete") or report.get("phase") != "tune"
            or report.get("audit_labels_read") is not False or report.get("selected_stage") != 50
            or report.get("groups") != expected_count
            or report.get("group_scores_sha256") != EVAL.sha256_file(scores_path)):
        raise ValueError("Reference must be the completed geographic HGB50 Tune selection")
    rows = []
    with Path(scores_path).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            row = record["new"]
            if record["group_identity"] != row["group_identity"]:
                raise ValueError("Reference wrapper and context identity differ")
            validate_row(row, STAGES, dense=True)
            rows.append(row)
    if len(rows) != expected_count or len({row["group_identity"] for row in rows}) != len(rows):
        raise ValueError("Reference omits or repeats Tune contexts")
    for stage in STAGES:
        verify_summary(rows, report["new_capacity_curve"][str(stage)], stage)
    return report, rows


def rank_hits(scores, rrf, labels):
    scores, rrf, labels = np.asarray(scores), np.asarray(rrf), np.asarray(labels)
    if scores.ndim != 1 or scores.shape != rrf.shape or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Scoring columns must cover a finite complete pool")
    selected = np.lexsort((np.arange(len(scores)), -rrf, -scores))[:50]
    return int(labels[selected].sum())


def score_one_group(reference, items, labels, rrf, baseline_mask, predictions, relevant):
    """Check candidate identities and labels, then reproduce the frozen source."""
    size = reference["pool_size"]
    labels, rrf, baseline_mask = np.asarray(labels), np.asarray(rrf), np.asarray(baseline_mask)
    if (size != len(items) or size > 998 or len(set(items)) != size
            or EVAL.canonical_sha256(items) != reference["ordered_pool_sha256"]
            or labels.shape != (size,) or rrf.shape != (size,) or baseline_mask.shape != (size,)
            or not np.isfinite(rrf).all() or not np.isin(baseline_mask, [0., 1.]).all()):
        raise ValueError("Cached complete pool identity, order or feature columns differ")
    expected_labels = np.asarray([item in relevant for item in items], dtype=bool)
    if (len(relevant) != reference["available_relevant_count"] or not np.array_equal(labels, expected_labels)
            or int(labels.sum()) != reference["pool_hits"]
            or int(labels[baseline_mask.astype(bool)].sum()) != reference["baseline_hits"]
            or int(baseline_mask.sum()) > 50):
        raise ValueError("Cached positive membership or fixed Recall counts differ from Tune truth")
    if set(predictions) != {"source", *VARIANTS}:
        raise ValueError("Cached evaluation requires the source and both separate variants")
    output = {}
    for name, by_stage in predictions.items():
        if set(by_stage) != set(STAGES):
            raise ValueError("Cached scoring lacks a predeclared HGB stage")
        row = dict(reference)
        for stage, scores in by_stage.items():
            row[f"s{stage}_hits"] = rank_hits(scores, rrf, labels)
            if name == "source" and row[f"s{stage}_hits"] != reference[f"s{stage}_hits"]:
                raise ValueError(f"Original HGB stage{stage} no longer reproduces saved per-context hits")
        if name != "source":
            output[name] = row
    return output


def stream_scores(cache_path, feature_names, references, models, targets, *, batch_size=32768):
    """Carry groups across arbitrary Parquet batches; never load the full cache."""
    import pyarrow.parquet as pq
    eligible = [row for row in references if row["available_relevant_count"]]
    expected_identity = [row["group_identity"] for row in eligible]
    predictions_by_group = {name: {} for name in VARIANTS}
    current = None
    current_index = 0
    started, scored_rows = time.monotonic(), 0
    positions = {name: index for index, name in enumerate(feature_names)}

    def begin(key):
        if current_index >= len(eligible) or key != expected_identity[current_index]:
            raise ValueError("Feature cache groups are missing, repeated, unknown or reordered")
        return {"key": key, "items": [], "labels": [], "rrf": [], "baseline": [],
                "predictions": {name: {stage: [] for stage in STAGES} for name in models}}

    def flush(pending):
        nonlocal current_index
        reference = eligible[current_index]
        observed = score_one_group(reference, pending["items"], pending["labels"], pending["rrf"],
            pending["baseline"], pending["predictions"], set(targets[pending["key"]]["available_item_ids"]))
        for name, row in observed.items():
            predictions_by_group[name][pending["key"]] = row
        current_index += 1

    parquet = pq.ParquetFile(cache_path)
    columns = list(feature_names) + ["group_identity", "item_id", "available_positive"]
    for number, batch in enumerate(parquet.iter_batches(batch_size=batch_size, columns=columns)):
        matrix = np.column_stack([batch.column(name).to_numpy(zero_copy_only=False) for name in feature_names])
        if matrix.dtype != np.float32 or not np.isfinite(matrix).all():
            raise ValueError("Frozen numeric feature cache must remain finite float32")
        predictions = {name: dict(EVAL.prefix_scores(model, matrix, STAGES)) for name, model in models.items()}
        keys, items = batch.column("group_identity").to_pylist(), batch.column("item_id").to_pylist()
        labels = batch.column("available_positive").to_numpy(zero_copy_only=False)
        # Contiguous query runs avoid a Python append for each model/stage/row.
        starts = [0, *(index for index in range(1, len(keys)) if keys[index] != keys[index - 1]), len(keys)]
        for left, right in zip(starts, starts[1:]):
            key = keys[left]
            if current is None:
                current = begin(key)
            elif current["key"] != key:
                flush(current)
                current = begin(key)
            current["items"].extend(items[left:right])
            current["labels"].extend(labels[left:right].tolist())
            current["rrf"].extend(matrix[left:right, positions["rrf_score"]].tolist())
            current["baseline"].extend(matrix[left:right, positions["baseline_top50"]].tolist())
            for name, by_stage in predictions.items():
                for stage, values in by_stage.items():
                    current["predictions"][name][stage].extend(values[left:right].tolist())
        scored_rows += len(batch)
        if number == 0 or number % 8 == 0:
            elapsed = time.monotonic() - started
            print(json.dumps({"feature_rows": scored_rows, "total_feature_rows": parquet.metadata.num_rows,
                "complete_available_groups": current_index, "seconds": round(elapsed, 1),
                "estimated_remaining_seconds": round(elapsed * (parquet.metadata.num_rows - scored_rows) / scored_rows, 1)}), flush=True)
    if current is not None:
        flush(current)
    if current_index != len(eligible):
        raise ValueError("Feature cache omitted eligible Tune groups")
    if scored_rows != sum(row["pool_size"] for row in eligible):
        raise ValueError("Feature cache row count does not equal complete eligible pools")
    result = {name: [] for name in VARIANTS}
    for reference in references:
        key = reference["group_identity"]
        for name in VARIANTS:
            if reference["available_relevant_count"]:
                row = predictions_by_group[name][key]
            else:
                row = dict(reference)
                if any(row[f"s{stage}_hits"] or row["pool_hits"] or row["baseline_hits"] for stage in STAGES):
                    raise ValueError("Zero-availability context cannot contain positive hits")
                row.update({f"s{stage}_hits": 0 for stage in STAGES})
            result[name].append(row)
    return result, {"rows": scored_rows, "available_groups": current_index,
        "all_groups": len(references), "zero_availability_groups": len(references) - current_index,
        "source_stages_reproduced_per_context": list(STAGES), "pool_id_order_and_labels_verified": True}


def save_secondary_model(artifact, source_path, variant, stage, report_path, output_dir):
    import joblib
    from avito_candidates.fitted_fingerprint import fitted_estimator_fingerprint
    destination = Path(output_dir) / f"hgb-geo005-{variant}-selected{stage}.joblib"
    if destination.exists():
        raise FileExistsError("Secondary model output already exists")
    selected = dict(artifact)
    selected["selected_iteration"] = stage
    selected["selection_status"] = "secondary-capacity-selected-on-consumed-tune; no-audit"
    selected["training_configuration"] = {**artifact["training_configuration"],
        "selected_iteration": stage, "selection_status": selected["selection_status"]}
    fingerprint = fitted_estimator_fingerprint(artifact["model"])
    selected["regularization_selection"] = {"variant": variant, "stage": stage, "primary_stage": 50,
        "source_model": identity(source_path), "report": identity(report_path),
        "fitted_estimator_sha256": fingerprint,
        "selection_rule": "secondary smallest 50/75/100/150 within 0.002 of best variant Tune Recall",
        "scope": "Secondary exploratory capacity artifact; does not replace the primary fixed50 comparison"}
    with destination.open("xb") as stream:
        joblib.dump(selected, stream)
    loaded = joblib.load(destination)
    if (loaded.get("selected_iteration") != stage or loaded["metadata"] != artifact["metadata"]
            or fitted_estimator_fingerprint(loaded["model"]) != fingerprint):
        raise ValueError("Secondary artifact failed reload verification")
    envelope = {**selected["regularization_selection"], "model": identity(destination),
                "training_provenance": selected["provenance"], "retrieval_config": selected["metadata"]["retrieval_config"]}
    EVAL.write_new_json(destination.with_suffix(destination.suffix + ".json"), envelope)
    return envelope["model"]


def evaluate(args):
    import joblib
    from build_fresh_validation_protocol import load_manifest
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh evaluation subdirectory")
    args.output_dir.mkdir(parents=True)
    plan = json.loads(args.experiment_plan.read_text(encoding="utf-8"))
    if (plan.get("variants") != VARIANTS or plan.get("primary_stage") != 50
            or plan.get("primary_baseline_stage") != 50 or plan.get("secondary_stages") != list(STAGES)
            or plan.get("fitted_trees") != 150 or plan.get("scoring_order") != TIE_POLICY
            or plan.get("tune_groups") != 12103 or plan.get("available_tune_groups") != 4750
            or plan.get("maximum_pool_size") != 998 or plan.get("training_features") != 85):
        raise ValueError("Experiment plan differs from the declared two fixed50 comparisons")
    names = ("source_model", "current_selected_model", "previous_report", "previous_group_scores",
             "training_data", "training_metadata", "cached_tune_data", "cached_tune_metadata")
    paths = {name: checked_identity(plan[name], name) for name in names}
    source = joblib.load(paths["source_model"])
    if (source.get("estimator") != "histgb" or source.get("selected_iteration") is not None
            or EVAL.maximum_iterations(source["model"]) != 150 or tuple(source["metadata"]["feature_names"]) != V8_FEATURES
            or source["provenance"]["training_data_sha256"] != plan["training_data"]["sha256"]
            or source["provenance"]["training_metadata_sha256"] != plan["training_metadata"]["sha256"]):
        raise ValueError("Frozen source is not the same geographic HGB150 training artifact")
    report, reference_rows = read_reference(paths["previous_report"], paths["previous_group_scores"], plan["tune_groups"])
    if report["input_identities"]["new_model"] != plan["source_model"]:
        raise ValueError("Reference results were scored with another fitted source model")
    selected_identity, _ = verify_selected_model(paths["previous_report"], report, 50)
    if {key: selected_identity[key] for key in ("path", "sha256")} != plan["current_selected_model"]:
        raise ValueError("Current selected50 model identity differs from predeclared baseline")
    cache = verify_feature_cache(paths["previous_report"], report, reference_rows, source["metadata"])
    if (cache["data"] != plan["cached_tune_data"] or cache["metadata"] != plan["cached_tune_metadata"]
            or cache["rows"] != 4740500 or cache["exported_groups"] != 4750):
        raise ValueError("Cached Tune scope differs from the fixed 4750 complete available pools")
    variant_paths = {"maxfeatures08": args.maxfeatures_model, "leaves15": args.leaves_model}
    artifacts = {name: joblib.load(path) for name, path in variant_paths.items()}
    for name, artifact in artifacts.items():
        validate_variant(artifact, source, name)
        experiment = artifact.get("provenance", {}).get("regularization_experiment", {})
        if (experiment.get("variant") != name or experiment.get("only_changed_parameters") != VARIANTS[name]
                or experiment.get("inputs", {}).get("experiment_plan") != identity(args.experiment_plan)
                or experiment.get("inputs", {}).get("source_model") != plan["source_model"]
                or experiment.get("inputs", {}).get("train_data") != plan["training_data"]
                or experiment.get("tune_labels_read") is not False or experiment.get("audit_labels_read") is not False
                or experiment.get("primary_evaluation_stage") != 50):
            raise ValueError("Variant fitting provenance does not match the predeclared fit-only experiment")
    manifest = load_manifest(args.protocol_dir, verify=False)
    protocol_hash = EVAL.sha256_file(args.protocol_dir / "manifest.json")
    if protocol_hash != report["protocol_sha256"]:
        raise ValueError("Tune label protocol differs from the fixed feature cache")
    source_files = {str(path.resolve()): EVAL.sha256_file(path) for path in (
        Path(__file__), Path(inspect.getfile(verify_feature_cache)), Path(inspect.getfile(EVAL.prefix_scores)),
        Path(inspect.getfile(paired_pipeline_bootstrap)))}
    inputs = {**{name: plan[name] for name in names}, "experiment_plan": identity(args.experiment_plan),
              **{name: identity(path) for name, path in variant_paths.items()}}
    declaration = {"phase": "tune", "audit_labels_read": False, "retrieval_rebuilt": False,
        "inventory_reencoded": False, "training_performed_by_evaluator": False, "stages": list(STAGES),
        "primary_stage": 50, "primary_baseline_stage": 50, "secondary_tolerance": .002,
        "secondary_selection_rule": "smallest declared stage within 0.002 of best Tune Recall per variant",
        "variants": VARIANTS, "scoring_order": TIE_POLICY, "inputs": inputs,
        "source_sha256": source_files, "protocol_sha256": protocol_hash,
        "bootstrap_replicates": args.bootstrap_replicates, "bootstrap_seed": 20261013,
        "maximum_pool_size": 998, "current_model_replaced": False,
        "interpretation": "Prespecified fixed50 comparisons plus separate exploratory capacity curves on reused Tune"}
    EVAL.write_new_json(args.output_dir / "declaration.json", declaration)
    targets = EVAL.read_phase_labels(args.protocol_dir, manifest, "tune")
    if set(targets) != {row["group_identity"] for row in reference_rows}:
        raise ValueError("Cached groups do not cover every Tune target context")
    for row in reference_rows:
        target = targets[row["group_identity"]]
        if any(row[key] != target[key] for key in ("query_key", "query_text", "representative",
                                                  "available_relevant_count", "full_relevant_count")):
            raise ValueError("Complete Tune labels and cached context denominators differ")
    started = time.monotonic()
    models = {"source": source["model"], **{name: artifact["model"] for name, artifact in artifacts.items()}}
    rows, parity = stream_scores(paths["cached_tune_data"], list(V8_FEATURES), reference_rows, models,
                                targets, batch_size=args.batch_size)
    curves = {name: {str(stage): named_summary(values, stage) for stage in STAGES} for name, values in rows.items()}
    secondary_stages = {name: choose_conservative_stage(curve) for name, curve in curves.items()}
    comparisons = {}
    for name, values in rows.items():
        comparisons[name] = {}
        for stage in STAGES:
            result = paired_pipeline_bootstrap(list(zip(values, reference_rows)), f"s{stage}_hits", "s50_hits",
                "available_relevant_count", replicates=args.bootstrap_replicates, seed=20261013)
            result["comparison"] = f"{name} HGB{stage} minus current geographic HGB50"
            comparisons[name][str(stage)] = result
    score_path = args.output_dir / "group-scores.jsonl"
    with score_path.open("x", encoding="utf-8") as stream:
        for index, baseline in enumerate(reference_rows):
            stream.write(json.dumps({"group_identity": baseline["group_identity"], "baseline": baseline,
                "variants": {name: values[index] for name, values in rows.items()}},
                ensure_ascii=False, separators=(",", ":")) + "\n")
    for name, info in inputs.items():
        checked_identity(info, "unchanged " + name)
    check_source_identity(source_files)
    final = {**declaration, "complete": True, "full_tune_complete": True,
        "groups": len(reference_rows), "cache_parity": parity, "feature_cache": cache,
        "baseline_capacity_curve": report["new_capacity_curve"], "baseline": named_summary(reference_rows, 50),
        "primary_fixed50": {name: {"summary": curves[name]["50"],
                                  "paired_vs_current50": comparisons[name]["50"]} for name in VARIANTS},
        "secondary_capacity_curves": curves, "secondary_selected_stages": secondary_stages,
        "secondary_paired_vs_current50": comparisons,
        "group_scores_sha256": EVAL.sha256_file(score_path), "seconds": time.monotonic() - started,
        "limitations": ["Two nominal bootstrap comparisons on previously consumed Tune are exploratory, not independent confirmation.",
            "Secondary prefix selection cannot be interpreted as the primary one-parameter fixed50 comparison.",
            "All candidates, complete denominators and available-positive memberships are identical across models.",
            "Unclicked training candidates remain unlabelled; this is not an official submission result."]}
    final_path = args.output_dir / "report.json"
    EVAL.write_new_json(final_path, final)
    selected = ({name: save_secondary_model(artifact, variant_paths[name], name, secondary_stages[name],
        final_path, args.output_dir) for name, artifact in artifacts.items()} if args.save_selected_models else {})
    print(json.dumps({"complete": True, "report": str(final_path), "primary_stage": 50,
                      "secondary_selected_models": selected}), flush=True)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment-plan", "protocol-dir", "maxfeatures-model", "leaves-model", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--save-selected-models", action="store_true")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.bootstrap_replicates) < 1:
        parser.error("Batch size and bootstrap replicates must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
