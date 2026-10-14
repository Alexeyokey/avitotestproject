"""Fit two matched HGB regularization experiments on existing geo0.05 rows.

Each variant changes exactly one parameter of the saved 150-tree estimator.
The cache, row order, features, labels, per-query class weights and all learned
retrieval priors remain identical. No Tune/Audit cache or labels are opened.
Training is sequential; checkpoint evaluation belongs to a separate script.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import time


VARIANTS = {"maxfeatures08": {"max_features": .8}, "leaves15": {"max_leaf_nodes": 15}}
PRIMARY_STAGE = 50
SECONDARY_STAGES = (50, 75, 100, 150)
DEFAULT_ROOT = Path("artifacts/geo-dense-reranker-2026-10-14")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matched_parameters(parameters, variant):
    """Reject drift instead of silently resetting an accidental second change."""
    if variant not in VARIANTS:
        raise ValueError("Unknown regularization variant")
    expected = {"max_iter": 150, "early_stopping": False, "random_state": 42,
                "max_features": 1., "max_leaf_nodes": 31, "warm_start": False}
    if any(parameters.get(key) != value for key, value in expected.items()):
        raise ValueError("Source parameters differ from the frozen 150-tree reference")
    changed = {**parameters, **VARIANTS[variant]}
    differences = {key for key in parameters if parameters[key] != changed[key]}
    if differences != set(VARIANTS[variant]):
        raise ValueError("Each experiment must change exactly its one declared parameter")
    return changed


def validate_fit_metadata(metadata):
    """A Tune candidate cache is never an acceptable supervised training input."""
    from avito_candidates.reranker import V8_FEATURES
    from cache_research_reranker import RESEARCH_POLICY
    from prepare_geo_dense_reranker import validate_geo_config
    if any(metadata.get(key) not in (None, "fit", "train", "training")
           for key in ("role", "phase", "fresh_partition")):
        raise ValueError("Only original fit training data is allowed; Tune/Audit caches are forbidden")
    if (metadata.get("feature_version") != 8
            or tuple(metadata.get("feature_names", ())) != V8_FEATURES
            or metadata.get("training_data_schema_version") != 2
            or metadata.get("split") != "queries" or metadata.get("seed") != 42
            or metadata.get("validation_fraction") != .2
            or metadata.get("training_prior_policy") != RESEARCH_POLICY):
        raise ValueError("Fit cache schema, split or exclusion policy differs")
    evidence = metadata.get("geo_dense_training", {})
    if (evidence.get("complete") is not True or evidence.get("phase") != "fit"
            or evidence.get("tune_labels_read") is not False
            or evidence.get("audit_labels_read") is not False
            or evidence.get("inventory_reencoded") is not False
            or evidence.get("selected_groups") != 30000
            or evidence.get("maximum_pool_size") != 998
            or metadata.get("stats", {}).get("sampled_queries") != 30000):
        raise ValueError("Expected completed, fit-only geo0.05 training provenance")
    validate_geo_config(metadata.get("retrieval_config", {}))
    return tuple(metadata["feature_names"])


def validate_source(source, metadata, hashes):
    from importlib.metadata import version
    from sklearn.ensemble import HistGradientBoostingClassifier
    from avito_candidates.reranker import fitted_iterations
    from cache_research_reranker import research_prior_policy
    if (source.get("estimator") != "histgb"
            or not isinstance(source.get("model"), HistGradientBoostingClassifier)
            or fitted_iterations(source["model"]) != 150
            or source.get("selected_iteration") is not None
            or source.get("selection_status") != "unselected"
            or source.get("disable_microcategory_prior", False)):
        raise ValueError("Use the unselected, complete, original 150-tree HGB source artifact")
    if source.get("metadata") != metadata:
        raise ValueError("Source model metadata does not match the exact supplied fit cache")
    provenance = source.get("provenance", {})
    mapping = {"training_data_sha256": "train_data", "training_metadata_sha256": "train_metadata",
               "training_priors_sha256": "train_priors"}
    if any(provenance.get(field) != hashes[key] for field, key in mapping.items()):
        raise ValueError("Fit data, metadata or priors are not the source model's proven inputs")
    parameters = source["model"].get_params(deep=False)
    if source.get("training_configuration", {}).get("parameters") != parameters:
        raise ValueError("Source estimator parameters differ from its training declaration")
    if provenance.get("package_versions", {}).get("scikit-learn") != version("scikit-learn"):
        raise ValueError("Matched regularization requires the source scikit-learn version")
    matched_parameters(parameters, "maxfeatures08")
    policy = research_prior_policy(source)
    return parameters, policy


def load_fit_arrays(path, feature_names, stats):
    """One float32 feature matrix; release Arrow buffers before tree fitting."""
    import numpy as np
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=[*feature_names, "label", "weight"])
    rows = len(table)
    if rows != stats["positive_rows"] + stats["unobserved_rows"]:
        raise ValueError("Saved training row count differs from metadata")
    matrix = np.empty((rows, len(feature_names)), dtype=np.float32)
    for column, name in enumerate(feature_names):
        matrix[:, column] = table[name].to_numpy()
    labels = table["label"].to_numpy().copy()
    weights = table["weight"].to_numpy().copy()
    del table
    gc.collect()
    if (not np.isin(labels, (0, 1)).all() or len(np.unique(labels)) != 2
            or int(labels.sum()) != stats["positive_rows"]
            or not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError("Fit labels or unchanged per-query class weights are invalid")
    known = feature_names.index("full_dense_known")
    score = feature_names.index("full_dense_score")
    for start in range(0, rows, 32768):
        block = matrix[start:start + 32768]
        if (not np.isfinite(block).all() or np.any(block[:, known] != 1)
                or np.any(np.abs(block[:, score]) > 1.0001)):
            raise ValueError("Fit features require measured, finite raw dense cosines")
    return matrix, labels, weights


def new_envelope(source, model, declaration):
    """Share immutable fitted priors without duplicating their coefficient arrays."""
    artifact = dict(source)
    artifact.update(model=model, selection_status="unselected", selected_iteration=None)
    for key in ("selection", "selection_provenance", "tune_selection", "checkpoint_selection"):
        artifact.pop(key, None)
    provenance = copy.deepcopy(source["provenance"])
    provenance["regularization_experiment"] = declaration
    artifact["provenance"] = provenance
    configuration = copy.deepcopy(source["training_configuration"])
    configuration.update(parameters=model.get_params(deep=False), fitted_iterations=int(model.n_iter_),
        early_stopping=False, selection_status="unselected", selected_iteration=None,
        regularization_variant=declaration["variant"])
    artifact["training_configuration"] = configuration
    return artifact


def run(args):
    import joblib
    from sklearn.base import clone
    from threadpoolctl import threadpool_limits
    from cache_research_reranker import research_prior_policy

    data = args.train_data
    metadata_path = data.with_suffix(data.suffix + ".json")
    priors_path = data.with_suffix(data.suffix + ".priors.joblib")
    if data.with_suffix(data.suffix + ".INCOMPLETE").exists():
        raise ValueError("Fit cache still has an incomplete preparation marker")
    inputs = {"source_model": args.source_model, "train_data": data,
              "train_metadata": metadata_path, "train_priors": priors_path,
              "producer": Path(__file__).resolve(),
              "experiment_plan": args.output_dir / "experiment-plan.json"}
    hashes = {name: sha256_file(path) for name, path in inputs.items()}
    plan = json.loads(inputs["experiment_plan"].read_text(encoding="utf-8"))
    if (plan.get("variants") != VARIANTS or plan.get("primary_stage") != PRIMARY_STAGE
            or plan.get("primary_baseline_stage") != PRIMARY_STAGE
            or plan.get("secondary_stages") != list(SECONDARY_STAGES)
            or plan.get("fitted_trees") != 150 or plan.get("early_stopping") is not False
            or plan.get("seed") != 42
            or any(plan.get(name, {}).get("sha256") != hashes[key] for name, key in
                   (("source_model", "source_model"), ("training_data", "train_data"),
                    ("training_metadata", "train_metadata")))):
        raise ValueError("Experiment plan differs from the declared matched tests or frozen training inputs")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = validate_fit_metadata(metadata)
    source = joblib.load(args.source_model)
    parameters, prior_policy = validate_source(source, metadata, hashes)
    variants = list(VARIANTS) if args.variant == "both" else [args.variant]
    for variant in variants:
        stem = "hgb-geo005-v8-" + variant + "-150"
        if any((args.output_dir / (stem + suffix)).exists()
               for suffix in (".joblib", ".json", ".INCOMPLETE", ".joblib.partial")):
            raise ValueError("Use fresh variant outputs; refusing to overwrite a completed or partial run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix, labels, weights = load_fit_arrays(data, feature_names, metadata["stats"])
    if plan.get("training_rows") != len(labels) or plan.get("training_features") != len(feature_names):
        raise ValueError("Training matrix shape differs from the predeclared experiment plan")
    results = []
    for variant in variants:
        stem = "hgb-geo005-v8-" + variant + "-150"
        model_path = args.output_dir / (stem + ".joblib")
        marker = args.output_dir / (stem + ".INCOMPLETE")
        expected = matched_parameters(parameters, variant)
        model = clone(source["model"]).set_params(**VARIANTS[variant])
        if model.get_params(deep=False) != expected:
            raise ValueError("Estimator clone did not preserve the matched parameter set")
        declaration = {"variant": variant, "only_changed_parameters": VARIANTS[variant],
            "source_parameters": parameters, "new_parameters": expected,
            "inputs": {key: {"path": str(path.resolve()), "sha256": hashes[key]}
                       for key, path in inputs.items()},
            "feature_names": list(feature_names), "training_rows": len(labels),
            "positive_rows": int(labels.sum()), "weights": "unchanged original per-query class weights",
            "row_order": "unchanged saved fit parquet order", "training_phase": "fit",
            "tune_labels_read": False, "audit_labels_read": False,
            "retrieval_rerun": False, "encoder_retrained": False,
            "priors_refitted": False, "source_prior_policy": prior_policy,
            "primary_evaluation_stage": PRIMARY_STAGE,
            "secondary_evaluation_stages": list(SECONDARY_STAGES),
            "selection": "unselected; fixed primary stage50 must be evaluated separately",
            "fit_execution": "sequential variants, 4 CPU threads"}
        marker.write_text(json.dumps(declaration, indent=2), encoding="utf-8")
        print(json.dumps({"regularization_fit_started": declaration}), flush=True)
        started = time.monotonic()
        with threadpool_limits(limits=4):
            model.fit(matrix, labels, sample_weight=weights)
        if int(model.n_iter_) != 150 or model.get_params(deep=False) != expected:
            raise ValueError("Fitting did not preserve the declared 150-tree experiment")
        if (source["model"].get_params(deep=False) != parameters
                or research_prior_policy(source) != prior_policy):
            raise ValueError("Frozen reference estimator or priors changed")
        if any(sha256_file(path) != hashes[key] for key, path in inputs.items()):
            raise ValueError("Source/input bytes changed during fitting")
        declaration["fit_seconds"] = time.monotonic() - started
        artifact = new_envelope(source, model, declaration)
        temporary = model_path.with_suffix(model_path.suffix + ".partial")
        joblib.dump(artifact, temporary)
        temporary.replace(model_path)
        report = {"complete": True, "model": str(model_path.resolve()),
                  "model_sha256": sha256_file(model_path), "experiment": declaration}
        (args.output_dir / (stem + ".json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
        marker.unlink()
        print(json.dumps({"regularization_fit_complete": report}), flush=True)
        results.append(report)
        del artifact, model
        gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    fit = command.add_parser("fit", help="Fit matched independent variants sequentially")
    fit.add_argument("--source-model", type=Path, default=DEFAULT_ROOT / "hgb-geo005-v8-150.joblib")
    fit.add_argument("--train-data", type=Path, default=DEFAULT_ROOT / "train-geo005-v8.parquet")
    fit.add_argument("--variant", choices=(*VARIANTS, "both"), required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
