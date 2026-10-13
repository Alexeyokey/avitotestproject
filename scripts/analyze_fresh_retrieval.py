"""Post-completion descriptive route analysis of a sealed fresh-validation cache.

No retrieval, model scoring or capacity selection occurs here. A completed phase
report, its group scores and all identity/hash checks are required before labels
or candidate columns are processed. Results belong in a new diagnostic JSON,
separate from the report and tune selection seal.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import evaluate_fresh_validation as EVAL
from reranker_audit_provenance import _reference_path


ROUTE_COLUMNS = ("present_description", "present_dense", "geo_channel_count")
ROUTES = ("description", "dense", "geo", "description_or_dense", "description_and_dense",
          "description_only", "dense_only", "geo_unique_additions", "other_routes", "full_pool")


def v4_feature_schema():
    from avito_candidates.reranker import FEATURE_SCHEMAS
    return tuple(FEATURE_SCHEMAS[4])


def read_completed_phase(evaluation_dir, phase):
    """Check closure before reading any protocol labels or candidate contents."""
    root = Path(evaluation_dir)
    path = root / f"{phase}-report.json"
    if not path.is_file():
        raise ValueError("Route analysis requires a completed phase report first")
    report = json.loads(path.read_text(encoding="utf-8"))
    stage = report.get("selected_stage")
    if (report.get("report_version") != 1 or report.get("phase") != phase
            or report.get("policy") != EVAL.POLICY or isinstance(stage, bool)
            or stage not in EVAL.TUNE_STAGES):
        raise ValueError("Completed phase identity, policy or selected prefix is invalid")
    expected_stages = {str(stage)} if phase == "audit" else {str(value) for value in EVAL.TUNE_STAGES}
    if set(report.get("stages", {})) != expected_stages:
        raise ValueError("Completed phase stages differ; audit must contain its selected prefix only")
    scores_path = root / f"{phase}-group-scores.jsonl"
    if (not scores_path.is_file()
            or EVAL.sha256_file(scores_path) != report.get("group_scores_sha256")):
        raise ValueError("Completed group scores are missing or changed")
    return report, path, scores_path


def read_completed_groups(path, groups):
    """Bind previously scored outcomes to complete ordered protocol groups."""
    completed = []
    with Path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= len(groups):
                raise ValueError("Completed group scores contain extra groups")
            row = json.loads(line)
            group = groups[index]
            for field in ("group_id", "protocol_group_id", "query_text", "pool_size",
                          "full_relevant_count", "available_relevant_count", "representative"):
                if row.get(field) != group.get(field):
                    raise ValueError(f"Completed group-score identity differs: {field}")
            hits = row.get("pool_hits")
            if (isinstance(hits, bool) or not isinstance(hits, int)
                    or hits != group.get("positives_in_pool")
                    or not 0 <= hits <= group["available_relevant_count"]):
                raise ValueError("Completed positive pool count differs from cache metadata")
            completed.append({"pool_hits": hits, "pool_size": row["pool_size"]})
    if len(completed) != len(groups):
        raise ValueError("Completed group scores omitted protocol groups")
    return completed


def stream_route_stats(path, metadata, groups, completed, *, batch_size=65536,
                       expected_features=None):
    """Accumulate complete route counts/hits across arbitrary Parquet batches."""
    import numpy as np
    import pyarrow.parquet as pq

    if batch_size < 1 or not groups or len(groups) != len(completed):
        raise ValueError("Route analysis requires groups, completed scores and a positive batch size")
    expected = tuple(expected_features) if expected_features is not None else v4_feature_schema()
    if (metadata.get("feature_schema_version") != 4
            or tuple(metadata.get("feature_names", ())) != expected
            or not set(ROUTE_COLUMNS) <= set(expected)):
        raise ValueError("Route analysis requires the exact V4 feature names and route columns")
    parquet = pq.ParquetFile(path)
    columns = ["group_id", "label", *ROUTE_COLUMNS]
    if not set(columns) <= set(parquet.schema_arrow.names):
        raise ValueError("V4 candidate cache lacks exact route columns")
    count = len(groups)
    totals = {route: np.zeros(count, dtype=np.int64) for route in ROUTES}
    positives = {route: np.zeros(count, dtype=np.int64) for route in ROUTES}
    observed_rows = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        ids = batch.column("group_id").to_numpy(zero_copy_only=False)
        labels = batch.column("label").to_numpy(zero_copy_only=False)
        if (not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or np.any(ids >= count)
                or not np.isin(labels, (0, 1)).all()):
            raise ValueError("Candidate group IDs or binary labels are invalid")
        raw = [batch.column(name).to_numpy(zero_copy_only=False) for name in ROUTE_COLUMNS]
        if (any(not np.isfinite(values).all() for values in raw)
                or not np.isin(raw[0], (0, 1)).all() or not np.isin(raw[1], (0, 1)).all()
                or np.any(raw[2] < 0) or np.any(raw[2] != np.floor(raw[2]))):
            raise ValueError("V4 route-presence flags or geography counts are invalid")
        description, dense, geo = raw[0].astype(bool), raw[1].astype(bool), raw[2] > 0
        union = description | dense
        masks = {"description": description, "dense": dense, "geo": geo,
                 "description_or_dense": union, "description_and_dense": description & dense,
                 "description_only": description & ~dense, "dense_only": dense & ~description,
                 "geo_unique_additions": geo & ~union, "other_routes": ~(union | geo),
                 "full_pool": np.ones(len(ids), dtype=bool)}
        relevant = labels.astype(bool)
        for route, mask in masks.items():
            totals[route] += np.bincount(ids[mask], minlength=count)
            positives[route] += np.bincount(ids[mask & relevant], minlength=count)
        observed_rows += len(ids)
    if observed_rows != metadata.get("rows"):
        raise ValueError("Parquet rows differ from completed cache metadata")
    rows = []
    for index, (group, scored) in enumerate(zip(groups, completed)):
        if (int(totals["full_pool"][index]) != group["pool_size"]
                or scored["pool_size"] != group["pool_size"]
                or int(positives["full_pool"][index]) != scored["pool_hits"]):
            raise ValueError("Full-pool size or positive hits differ from completed group scores")
        row = {key: value for key, value in group.items() if not key.startswith("_")}
        for route in ROUTES:
            hits = int(positives[route][index])
            if hits > group["available_relevant_count"]:
                raise ValueError("Route positives exceed the sealed available GT denominator")
            row[f"{route}_count"] = int(totals[route][index])
            row[f"{route}_hits"] = hits
        d, n = row["description_count"], row["dense_count"]
        intersection, union_size = row["description_and_dense_count"], row["description_or_dense_count"]
        row["description_dense_jaccard"] = intersection / union_size if union_size else None
        row["description_overlap_fraction"] = intersection / d if d else None
        row["dense_overlap_fraction"] = intersection / n if n else None
        row["description_dense_overlap_coefficient"] = intersection / min(d, n) if min(d, n) else None
        rows.append(row)
    return rows


def describe_stat(rows, field):
    by_text, values = defaultdict(list), []
    for row in rows:
        value = row[field]
        if value is not None:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Route summary statistic is not finite")
            values.append(value)
            by_text[row["query_text"]].append(value)
    return {"groups_with_defined_value": len(values), "undefined_groups": len(rows) - len(values),
            "texts_with_defined_value": len(by_text),
            "group_mean": sum(values) / len(values) if values else None,
            "text_macro_mean": (sum(sum(values) / len(values) for values in by_text.values())
                                / len(by_text) if by_text else None)}


def summarize_routes(rows):
    if not rows:
        raise ValueError("Route analysis needs the complete cohort, including empty pools")
    def candidate_recall(route, denominator):
        # Reuse the evaluator's denominator/averaging rules, but do not call
        # channel-depth candidate coverage a final top-50 metric.
        values = EVAL.aggregate(rows, f"{route}_hits", denominator)
        return {key.replace("_recall_at_50", "_candidate_recall"): value for key, value in values.items()}
    recalls = {
        route: {"conditional_available": candidate_recall(route, "available_relevant_count"),
                "full_truth": candidate_recall(route, "full_relevant_count"),
                "candidate_count": describe_stat(rows, f"{route}_count")}
        for route in ROUTES
    }
    return {
        "candidate_recall": recalls,
        "recall_scope": "Coverage at each saved channel's candidate depth, not final Recall@50",
        "description_dense_overlap": {
            "definitions": {"intersection_count": "number of candidates present in both channels",
                            "jaccard": "intersection / description-dense union",
                            "description_overlap_fraction": "intersection / description candidates",
                            "dense_overlap_fraction": "intersection / dense candidates",
                            "overlap_coefficient": "intersection / min(description,dense)",
                            "undefined": "zero denominators are null; empty pools remain in counts"},
            "intersection_count": describe_stat(rows, "description_and_dense_count"),
            "jaccard": describe_stat(rows, "description_dense_jaccard"),
            "description_overlap_fraction": describe_stat(rows, "description_overlap_fraction"),
            "dense_overlap_fraction": describe_stat(rows, "dense_overlap_fraction"),
            "overlap_coefficient": describe_stat(rows, "description_dense_overlap_coefficient"),
        },
        "cohort": {"groups": len(rows), "texts": len({row["query_text"] for row in rows}),
                   "empty_pools": sum(row["pool_size"] == 0 for row in rows),
                   "groups_without_available_gt": sum(row["available_relevant_count"] == 0 for row in rows),
                   "full_relevant_pairs": sum(row["full_relevant_count"] for row in rows),
                   "available_relevant_pairs": sum(row["available_relevant_count"] for row in rows)},
        "interpretation": "Candidate recall uses complete sealed GT denominators, never positives found in a pool. Geo unique additions exclude description/dense union; other routes are reported explicitly.",
    }


def analyze_completed(phase, protocol_dir, cache_dir, evaluation_dir, output, *, batch_size=65536):
    output = Path(output)
    if output.exists():
        raise ValueError("Diagnostic output already exists; choose a new JSON path")
    report, report_path, scores_path = read_completed_phase(evaluation_dir, phase)
    manifest, protocol_hash = EVAL.checked_protocol(protocol_dir)
    model_hash = report.get("model_sha256")
    if (report.get("protocol_sha256") != protocol_hash or not isinstance(model_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", model_hash)):
        raise ValueError("Completed report protocol or model identity differs")
    selection_path = Path(evaluation_dir) / "selection.json"
    if not selection_path.is_file():
        raise ValueError("Completed phase has no immutable tune selection")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = EVAL.verify_selection(selection, protocol_hash, model_hash,
                                      evaluation_dir, report.get("train_data_sha256"))
    if selected != report["selected_stage"]:
        raise ValueError("Completed report prefix differs from the tune seal")
    path, metadata_path = EVAL.phase_cache_paths(cache_dir, phase)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_hashes = report.get("cache_sha256", {})
    parquet_hash = EVAL.check_phase_cache_inputs(path, metadata, phase, protocol_hash, model_hash)
    if (expected_hashes.get("parquet") != parquet_hash
            or expected_hashes.get("metadata") != EVAL.sha256_file(metadata_path)):
        raise ValueError("Completed report cache identities differ from current bytes")
    model_path = _reference_path(Path(cache_dir).resolve(), metadata.get("model_reference", ""))
    if EVAL.sha256_file(model_path) != model_hash:
        raise ValueError("Completed model artifact bytes changed")
    if metadata.get("feature_schema_version") != 4 or tuple(metadata.get("feature_names", ())) != v4_feature_schema():
        raise ValueError("Posthoc retrieval diagnostics require the exact V4 schema")
    labels = EVAL.read_phase_labels(protocol_dir, manifest, phase)
    groups = EVAL.bind_cache_groups(metadata, labels, phase, protocol_hash, model_hash)
    completed = read_completed_groups(scores_path, groups)
    rows = stream_route_stats(path, metadata, groups, completed, batch_size=batch_size)
    diagnostic = {"diagnostic_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "phase": phase, "type": "posthoc_descriptive_retrieval_routes",
                  "completed_report_sha256": EVAL.sha256_file(report_path),
                  "completed_group_scores_sha256": EVAL.sha256_file(scores_path),
                  "selection_sha256": EVAL.sha256_file(selection_path),
                  "protocol_sha256": protocol_hash, "model_sha256": model_hash,
                  "cache_sha256": {"parquet": parquet_hash, "metadata": EVAL.sha256_file(metadata_path)},
                  "route_columns": list(ROUTE_COLUMNS), **summarize_routes(rows),
                  "limitations": ["No model rescoring, new retrieval, capacity selection or pipeline change.",
                                  "Candidate recall differs from final top-50 Recall.",
                                  "Fresh inventory/tail cohort differs from the official benchmark.",
                                  "These posthoc descriptions do not confirm an optimization; tuning from them would consume the heldout evidence."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(diagnostic, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return diagnostic


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("tune", "audit"), required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=65536)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    diagnostic = analyze_completed(**vars(args))
    print(json.dumps({"output": str(args.output), "phase": args.phase,
                      "groups": diagnostic["cohort"]["groups"], "type": diagnostic["type"]}))


if __name__ == "__main__":
    main()
