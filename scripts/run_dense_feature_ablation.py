"""Sequential, matched reranker experiment after grouped V6 preparation.

All models see identical sampled rows. Validation reuses immutable candidate
pools and saved cosines; no item embeddings are rebuilt. Prefix selection uses
only tuning texts. Historical confirmation is explicitly not an untouched test.
Run from the project checkout with PYTHONPATH=src and HF_HUB_OFFLINE=1.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command, *, dry_run=False):
    print(shlex.join(map(str, command)), flush=True)
    if not dry_run:
        subprocess.run(list(map(str, command)), check=True)


def fit(data, model, *, estimator, iterations, dry_run):
    if model.exists() and not dry_run:
        import joblib
        from avito_candidates.reranker import fitted_iterations
        artifact = joblib.load(model)
        provenance = artifact.get("provenance", {})
        config = artifact.get("training_configuration", {})
        if (provenance.get("training_data_sha256") != sha256(data)
                or provenance.get("training_metadata_sha256") != sha256(data.with_suffix(data.suffix + ".json"))
                or provenance.get("training_priors_sha256") != sha256(data.with_suffix(data.suffix + ".priors.joblib"))
                or config.get("estimator") != estimator
                or config.get("fitted_iterations") != iterations
                or fitted_iterations(artifact["model"]) != iterations
                or type(artifact["model"]).__name__ != {
                    "histgb": "HistGradientBoostingClassifier", "lambdarank": "LGBMRanker"
                }[estimator]):
            raise ValueError(f"Existing model differs from this experiment: {model}")
        print(f"Reusing verified fit {model}", flush=True)
        return
    run([sys.executable, "-m", "avito_candidates.cli", "fit-reranker",
         "--reranker-data", data, "--reranker-model", model,
         "--reranker-estimator", estimator, "--reranker-max-iter", iterations],
        dry_run=dry_run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-train", type=Path, required=True)
    parser.add_argument("--source-cache-dir", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--baseline-train", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    env = os.environ
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("HF_HUB_OFFLINE", "1")
    scripts = Path(__file__).resolve().parent
    output = args.output_dir
    v4_train = output / "v4-training" / "train.parquet"
    if not args.dry_run:
        for path in (args.v6_train, args.v6_train.with_suffix(args.v6_train.suffix + ".json"),
                     args.source_cache_dir / "validation.parquet", args.baseline_model):
            if not path.is_file():
                raise FileNotFoundError(f"Required completed input is missing: {path}")
        output.mkdir(parents=True, exist_ok=True)
    # Run in the actual ML environment after retrieval has released its memory.
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
        dry_run=args.dry_run)
    run([sys.executable, "-m", "unittest", "discover", "-s", "scripts",
         "-p", "test_*.py", "-q"], dry_run=args.dry_run)
    if not v4_train.exists():
        run([sys.executable, scripts / "project_reranker_feature_schema.py",
             "--source", args.v6_train, "--target", v4_train, "--data-dir", args.data_dir],
            dry_run=args.dry_run)
    elif not args.dry_run:
        import json
        manifest = json.loads(v4_train.with_suffix(v4_train.suffix + ".json").read_text())
        projection = manifest.get("feature_projection", {})
        checks = {
            "source_data_sha256": args.v6_train,
            "source_metadata_sha256": args.v6_train.with_suffix(args.v6_train.suffix + ".json"),
            "source_priors_sha256": args.v6_train.with_suffix(args.v6_train.suffix + ".priors.joblib"),
            "projected_data_sha256": v4_train,
        }
        if (any(projection.get(key) != sha256(path) for key, path in checks.items())
                or projection.get("source_priors_sha256") != sha256(v4_train.with_suffix(v4_train.suffix + ".priors.joblib"))):
            raise ValueError("Existing V4 projection differs from its verified source rows/metadata/priors")

    model_paths = {}
    for schema, data in (("v4", v4_train), ("v6", args.v6_train)):
        for estimator, iterations in (("histgb", 150), ("lambdarank", 400)):
            name = f"{schema}-{estimator}"
            model = output / "models" / f"{name}.joblib"
            fit(data, model, estimator=estimator, iterations=iterations, dry_run=args.dry_run)
            model_paths[name] = model

    v6_cache = output / "v6-cache"
    run([sys.executable, scripts / "derive_full_dense_audit_cache.py",
         "--source-cache-dir", args.source_cache_dir, "--train-data", args.v6_train,
         "--reference-model", model_paths["v6-histgb"], "--data-dir", args.data_dir,
         "--output-dir", v6_cache], dry_run=args.dry_run)
    common = ["--stages", 25, 50, 75, 100, 125, 150, 250, 400]
    # The new baseline audit binds the newly computed candidate-pool digest.
    # A previous report's missing digest is never inferred from its hit counts.
    run([sys.executable, scripts / "audit_reranker_learning_curve.py",
         "--cache-dir", args.source_cache_dir, "--model", f"hgb={args.baseline_model}",
         "--train-data", args.baseline_train, "--sampled-train-logloss",
         *common, "--output-dir", output / "baseline-curves"], dry_run=args.dry_run)
    for schema, data, cache in (("v4", v4_train, args.source_cache_dir),
                                ("v6", args.v6_train, v6_cache)):
        # Do not reuse the old fit-side cache for geography cross-fitting.
        run([sys.executable, scripts / "audit_reranker_learning_curve.py",
             "--cache-dir", cache, "--validation-only", "--train-data", data,
             "--sampled-train-logloss", "--model", f"{schema}-histgb={model_paths[f'{schema}-histgb']}",
             "--model", f"{schema}-lambdarank={model_paths[f'{schema}-lambdarank']}",
             *common, "--output-dir", output / f"{schema}-curves"], dry_run=args.dry_run)
    run([sys.executable, scripts / "compare_audit_models.py",
         "--audit-dir", output / "baseline-curves", "--audit-dir", output / "v4-curves",
         "--audit-dir", output / "v6-curves", "--reference-name", "hgb", "--reference-stage", 50,
         "--allow-rescore-only-config-difference", "--output", output / "paired-vs-hgb50.json"],
        dry_run=args.dry_run)
    print("Completed matched ablation; inspect confirmation intervals before deployment.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
