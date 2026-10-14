"""Publish the fixed-capacity final model, predict and assemble an offline bundle.

The final model was fitted on all available training groups after capacity was
chosen on the queries split. Its own weights have no held-out score. This script
records that distinction, reads back the actual CSV, and preserves older files.
Run from the source checkout with PYTHONPATH=src in the recorded ML environment.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from export_offline_solution import portable_config_args, sha256


def run(command):
    import shlex
    print(shlex.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fitted-model", type=Path, required=True)
    parser.add_argument("--audited-model", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--paired-report", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--hub-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    if args.embedding_batch_size < 1:
        parser.error("Embedding batch size must be positive")
    root, scripts = Path.cwd().resolve(), Path(__file__).resolve().parent
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Release output must be a new directory: {output}")
    for path in (args.fitted_model, args.audited_model, args.audit_report, args.paired_report,
                 root / "requirements-research-2026-10-12.txt"):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True)
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    selected = output / "model.joblib"
    run([sys.executable, scripts / "select_reranker_checkpoint.py",
         "--audit-report", args.audit_report, "--source-model", args.audited_model,
         "--transfer-to-model", args.fitted_model, "--allow-legacy-provenance",
         "--output", selected])

    import joblib
    import pyarrow.parquet as pq
    from avito_candidates.core import validate_answers
    artifact = joblib.load(selected)
    metadata = artifact["metadata"]
    config = metadata["retrieval_config"]
    checkpoint = Path(config["embedding_model"])
    answer = output / "answer.csv"
    arguments = portable_config_args(config, checkpoint=checkpoint)
    prediction = [sys.executable, "-m", "avito_candidates.cli", "predict", *arguments,
                  "--data-dir", args.data_dir, "--reranker-model", selected,
                  "--device", args.device, "--embedding-batch-size", args.embedding_batch_size,
                  "--answer", answer]
    run(prediction)
    # Verify the bytes submitted to the judge, including exact headers and IDs.
    with answer.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=",")
        if reader.fieldnames != ["query_id", "answer"]:
            raise ValueError("The submitted CSV must have exactly query_id,answer")
        rows = list(reader)
    query_ids = pq.read_table(args.data_dir / "benchmark_queries.parquet", columns=["query_id"])["query_id"].to_pylist()
    item_ids = pq.read_table(args.data_dir / "benchmark_items.parquet", columns=["item_id"])["item_id"].to_pylist()
    validate_answers(rows, query_ids, set(item_ids))
    if any(not isinstance(row["query_id"], str) or len(row["query_id"]) != 16
           or row["answer"] != " ".join(row["answer"].split()) for row in rows):
        raise ValueError("IDs or the answer's single-space representation are invalid")

    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    for source, name in ((args.audit_report, "cold-validation-report.json"),
                         (args.paired_report, "paired-comparison.json")):
        shutil.copyfile(source, output / name)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "answer_sha256": sha256(answer), "answer_queries": len(rows),
        "answer_columns": ["query_id", "answer"], "encoding": "UTF-8",
        "delimiter": ",", "all_csv_requirements_validated_on_written_file": True,
        "minimum_items_per_query": min(len(row["answer"].split()) for row in rows),
        "maximum_items_per_query": max(len(row["answer"].split()) for row in rows),
        "model_sha256": sha256(selected), "fitted_model_sha256": sha256(args.fitted_model),
        "selected_iteration": artifact["selected_iteration"],
        "training_stats": metadata["stats"], "training_prior_policy": metadata["training_prior_policy"],
        "retrieval_config": config, "embedding_checkpoint_sha256": metadata["embedding_checkpoint_sha256"],
        "embedding_base_revision": metadata["embedding_base_revision"],
        "capacity_selection": artifact["selection_provenance"],
        "validation_score_applies_to_cold_split_model_only": True,
        "final_weights_directly_evaluated": False, "new_hidden_recall": None,
        "cold_validation_selection": audit["selection"],
        "datasets": {name: sha256(args.data_dir / name) for name in (
            "train.parquet", "benchmark_items.parquet", "benchmark_queries.parquet")},
        "prediction_command": list(map(str, prediction)),
        "inference_environment": {"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"]},
    }
    (output / "release-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    run([sys.executable, scripts / "export_offline_solution.py", "--project-root", root,
         "--reranker-model", selected, "--embedding-checkpoint", checkpoint,
         "--hub-cache", args.hub_cache, "--output-dir", output / "offline-solution",
         "--requirements", root / "requirements-research-2026-10-12.txt", "--answer", answer])
    print(json.dumps({"answer": str(answer), "queries": len(rows),
                      "answer_sha256": report["answer_sha256"], "offline_bundle": str(output / "offline-solution")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
