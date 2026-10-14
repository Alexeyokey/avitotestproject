"""Fit matched pointwise and grouped reranker candidates on the saved v4 rows.

Imports for parquet and model fitting stay inside main so group validation is
testable without a GPU, LightGBM, or the project's Python environment. This
script does not select a winner: the outer, query-text-excluded evaluation cache
must be scored separately, and it is a historically reused holdout.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time


COMPARISON_VERSION = 1
DEFAULT_GROUPS = 19648
DEFAULT_UNOBSERVED_PER_GROUP = 64


def reconstruct_groups(labels, weights, *, expected_groups,
                       expected_negatives=DEFAULT_UNOBSERVED_PER_GROUP,
                       expected_positive_rows=None, expected_unobserved_rows=None,
                       weight_tolerance=1e-5):
    """Recover query runs from the original positive-first sampled row order.

    The writer emits all retrieved positives, then the sampled unobserved rows
    for each query. A 0->1 transition therefore begins the next query. Without
    IDs there is no independent way to prove identity; exact class weights,
    negative counts, and saved aggregate counts are required safeguards. Never
    shuffle or repartition these rows before reconstructing the runs.
    """
    if len(labels) != len(weights) or not len(labels):
        raise ValueError("Labels and weights must be nonempty and have equal length")
    if expected_groups <= 0 or expected_negatives <= 0:
        raise ValueError("Expected group and unobserved counts must be positive")
    if not math.isfinite(weight_tolerance) or weight_tolerance <= 0:
        raise ValueError("Weight tolerance must be positive and finite")

    sizes = []
    positives = negatives = 0
    positive_weight = negative_weight = 0.0
    all_positive_rows = all_unobserved_rows = 0
    positive_weights, negative_weights = [], []
    previous = None

    def finish_group():
        if positives == 0 or negatives == 0:
            raise ValueError("Each query run requires positives followed by unobserved rows")
        if negatives != expected_negatives:
            raise ValueError(
                f"Group {len(sizes)} has {negatives} unobserved rows; "
                f"expected {expected_negatives}. Row order may be corrupted.")
        for total, values, count in ((positive_weight, positive_weights, positives),
                                     (negative_weight, negative_weights, negatives)):
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=weight_tolerance):
                raise ValueError("Each group/class weight sum must be approximately one")
            expected_weight = 1.0 / count
            if any(not math.isclose(value, expected_weight, rel_tol=weight_tolerance,
                                    abs_tol=weight_tolerance * expected_weight)
                   for value in values):
                raise ValueError("Rows within a group/class must have uniform reciprocal weights")
        sizes.append(positives + negatives)

    for offset, (label, raw_weight) in enumerate(zip(labels, weights)):
        if label not in (0, 1) or isinstance(label, (str, bytes)):
            raise ValueError(f"Labels must be binary 0/1; invalid row {offset}")
        label = int(label)
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid row weight at row {offset}") from error
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Row weights must be finite and positive; invalid row {offset}")
        if previous is None and label != 1:
            raise ValueError("First row must be positive; positive-first ordering is required")
        if previous == 0 and label == 1:
            finish_group()
            positives = negatives = 0
            positive_weight = negative_weight = 0.0
            positive_weights, negative_weights = [], []
        if label:
            positives += 1
            all_positive_rows += 1
            positive_weight += weight
            positive_weights.append(weight)
        else:
            negatives += 1
            all_unobserved_rows += 1
            negative_weight += weight
            negative_weights.append(weight)
        previous = label
    finish_group()

    if len(sizes) != expected_groups:
        raise ValueError(f"Recovered {len(sizes)} groups; expected {expected_groups}")
    if expected_positive_rows is not None and all_positive_rows != expected_positive_rows:
        raise ValueError("Positive row count differs from training metadata")
    if expected_unobserved_rows is not None and all_unobserved_rows != expected_unobserved_rows:
        raise ValueError("Unobserved row count differs from training metadata")
    return sizes


def validate_metadata(metadata, *, expected_groups):
    """Require the historical outer split and the original 75 v4 features."""
    if metadata.get("feature_version") != 4:
        raise ValueError("Comparison requires feature schema v4")
    names = metadata.get("feature_names", [])
    if (len(names) != 75 or len(set(names)) != 75
            or any(not isinstance(name, str) for name in names)
            or set(names) & {"query_id", "item_id", "label", "weight"}):
        raise ValueError("Expected 75 distinct numeric v4 feature names without identifiers")
    if (metadata.get("split") != "queries" or metadata.get("seed") != 42
            or metadata.get("validation_fraction") != 0.2):
        raise ValueError("Comparison requires the original query-text split, seed 42, fraction 0.2")
    if not metadata.get("retrieval_config") or not metadata.get("corpus_fingerprint"):
        raise ValueError("Retrieval config and corpus fingerprint are required")
    stats = metadata.get("stats", {})
    if stats.get("queries_with_positive_in_pool") != expected_groups:
        raise ValueError("Expected group count differs from training metadata")
    for name in ("positive_rows", "unobserved_rows"):
        if not isinstance(stats.get(name), int) or stats[name] <= 0:
            raise ValueError(f"Missing positive integer training statistic: {name}")
    return names, stats


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json_new(path, payload):
    # Exclusive creation refuses accidental replacement of previous evidence.
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path,
                        help="Defaults to TRAIN_DATA.parquet.json")
    parser.add_argument("--priors", type=Path,
                        help="Defaults to TRAIN_DATA.parquet.priors.joblib")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-trees", type=int, default=400)
    parser.add_argument("--expected-groups", type=int, default=DEFAULT_GROUPS)
    parser.add_argument("--unobserved-per-group", type=int,
                        default=DEFAULT_UNOBSERVED_PER_GROUP)
    parser.add_argument("--models", choices=("both", "classifier", "ranker"), default="both")
    args = parser.parse_args(argv)
    if not 1 <= args.max_trees <= 400:
        parser.error("--max-trees must be between 1 and 400")

    metadata_path = args.metadata or args.train_data.with_suffix(args.train_data.suffix + ".json")
    priors_path = args.priors or args.train_data.with_suffix(args.train_data.suffix + ".priors.joblib")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    names, stats = validate_metadata(metadata, expected_groups=args.expected_groups)
    if not priors_path.is_file():
        parser.error(f"v4 training priors must exist: {priors_path}")

    modes = ("classifier", "ranker") if args.models == "both" else (args.models,)
    paths = {mode: args.output_dir / f"{mode}-{args.max_trees}.joblib" for mode in modes}
    summary_path = args.output_dir / f"comparison-{args.models}-{args.max_trees}.json"
    for path in (*paths.values(), *(p.with_suffix(".json") for p in paths.values()), summary_path):
        if path.exists() or path.with_suffix(path.suffix + ".partial").exists():
            parser.error(f"Refusing to overwrite existing output: {path}")

    # Heavy libraries are imported only for the explicitly requested fit.
    import joblib
    import lightgbm
    from lightgbm import LGBMClassifier, LGBMRanker
    import numpy as np
    import pyarrow
    import pyarrow.parquet as pq
    import sklearn

    table = pq.read_table(args.train_data, columns=[*names, "label", "weight"])
    labels = table["label"].to_numpy()
    weights = table["weight"].to_numpy()
    sizes = reconstruct_groups(
        labels, weights, expected_groups=args.expected_groups,
        expected_negatives=args.unobserved_per_group,
        expected_positive_rows=stats["positive_rows"],
        expected_unobserved_rows=stats["unobserved_rows"],
    )
    matrix = np.column_stack([table[name].to_numpy() for name in names]).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Saved v4 feature matrix must contain only finite values")
    del table
    priors = joblib.load(priors_path)
    if (not isinstance(priors, dict)
            or not {"microcategory_prior", "geo_associations"}.issubset(priors)):
        raise ValueError("v4 priors artifact is incomplete")

    hashes = {"training_data_sha256": digest(args.train_data),
              "training_metadata_sha256": digest(metadata_path),
              "priors_sha256": digest(priors_path),
              "training_script_sha256": digest(Path(__file__))}
    versions = {"lightgbm": lightgbm.__version__, "numpy": np.__version__,
                "pyarrow": pyarrow.__version__, "scikit_learn": sklearn.__version__}
    common = dict(n_estimators=args.max_trees, num_leaves=31, min_child_samples=50,
                  learning_rate=0.05, reg_lambda=1.0, n_jobs=4, random_state=42,
                  deterministic=True, force_col_wise=True, verbosity=-1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for mode in modes:
        start = time.monotonic()
        if mode == "classifier":
            parameters = {**common, "objective": "binary"}
            model = LGBMClassifier(**parameters)
            # Preserve the original equal positive/unobserved class mass per query.
            fit_kwargs = {"sample_weight": weights, "feature_name": names}
            estimator = "lightgbm"
            weighting = "original reciprocal class weights, class sum one per query"
        else:
            parameters = {**common, "objective": "lambdarank", "label_gain": [0, 1],
                          "lambdarank_norm": True, "lambdarank_truncation_level": 60}
            model = LGBMRanker(**parameters)
            # Group-normalised pairwise ranking uses the recovered query runs.
            # Truncation 60 covers the top-50 boundary. Pointwise class weights
            # are deliberately not reused: they define a different training goal.
            fit_kwargs = {"group": sizes, "feature_name": names}
            estimator = "lambdarank"
            weighting = "group-normalised LambdaRank; no pointwise class weights"
        print(json.dumps({"phase": "fit", "model": mode, "rows": len(labels),
                          "groups": len(sizes), "max_trees": args.max_trees}), flush=True)
        model.fit(matrix, labels, **fit_kwargs)
        configuration = {
            "comparison_version": COMPARISON_VERSION, "parameters": parameters,
            "row_weighting": weighting, "training_rows": len(labels),
            "groups": len(sizes), "positive_rows": stats["positive_rows"],
            "unobserved_rows": stats["unobserved_rows"],
            "unobserved_per_group": args.unobserved_per_group,
            "group_reconstruction": "positive-first runs separated by label 0->1, checked against weights and metadata",
            "internal_validation": None, "random_row_split": False,
            "selection_status": "unselected", "early_stopping": False,
            "comparison_evaluation": "historically reused outer query-text-excluded seed-42 cache; score separately",
            "limitations": [
                "Training rows lack query text/IDs; no internal cross-validation is possible from this artifact.",
                "Group identity is inferred from writer ordering and cannot be independently proved without IDs.",
                "Outer validation was inspected during previous model selection; this is a retrospective audit.",
                "Labels distinguish observed choices from unobserved candidates, not exhaustive relevance.",
                "Sampled 64 unobserved rows per group differ from complete inference candidate pools.",
                "NDCG-oriented LambdaRank must be selected by measured Recall@50, not training loss.",
            ],
            "package_versions": versions, "source_paths": {"train": str(args.train_data),
                "metadata": str(metadata_path), "priors": str(priors_path)}, **hashes,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "fit_seconds": time.monotonic() - start,
            "fitted_trees": model.booster_.num_trees(),
        }
        artifact = {"model": model, "metadata": metadata, "priors": priors,
                    "estimator": estimator, "disable_microcategory_prior": False,
                    "comparison_version": COMPARISON_VERSION,
                    "selection_status": "unselected", "training_configuration": configuration,
                    **hashes}
        target = paths[mode]
        temporary = target.with_suffix(target.suffix + ".partial")
        # Reserve the temporary path before dumping; final files are unique per run.
        with temporary.open("xb"):
            pass
        joblib.dump(artifact, temporary)
        if target.exists():
            raise FileExistsError(f"Output appeared during fitting: {target}")
        temporary.replace(target)
        report = {"model_path": str(target), "model_sha256": digest(target),
                  "estimator": estimator, "selection_status": "unselected",
                  "training_configuration": configuration}
        write_json_new(target.with_suffix(".json"), report)
        outputs.append(report)
        print(json.dumps({"phase": "saved", "model": mode, "path": str(target),
                          "fit_seconds": round(configuration["fit_seconds"], 2),
                          "trees": configuration["fitted_trees"]}), flush=True)
    write_json_new(summary_path, {"comparison_version": COMPARISON_VERSION,
                                 "selection_status": "unselected", "models": outputs})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
