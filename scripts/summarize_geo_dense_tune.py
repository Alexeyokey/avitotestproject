"""Verify and summarize the completed, full Tune geography experiment.

This reader never loads Audit labels, selects a model, or changes evaluation
artifacts. It independently reconstructs the displayed metrics from saved
per-context hit counts and checks that every arm has identical denominators.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


ARMS = ("exact_cosine", "geo_0.02", "geo_0.05", "geo_0.1")
EXPECTED_GROUPS = 12103
DENOMINATOR_FIELDS = ("query_key", "query_text", "available_relevant_count",
                      "full_relevant_count", "representative")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mean_recall(rows, hit_key, denominator="available_relevant_count"):
    clusters = defaultdict(list)
    contexts = []
    for row in rows:
        count = row[denominator]
        hits = row[hit_key]
        require(isinstance(count, int) and count >= 0 and isinstance(hits, int)
                and 0 <= hits <= count, "Invalid Recall numerator or denominator")
        if count:
            value = hits / count
            clusters[row["query_text"]].append(value)
            contexts.append(value)
    require(bool(contexts), "No evaluable contexts")
    return (sum(sum(values) / len(values) for values in clusters.values()) / len(clusters),
            sum(contexts) / len(contexts), len(contexts), len(clusters))


def close(actual, expected, label):
    require(math.isfinite(actual) and math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12),
            f"Saved aggregate disagrees with group scores: {label}")


def verify_aggregate(rows, saved, *, dense=False):
    coverage = saved["coverage"]
    sizes = [row["pool_size"] for row in rows]
    require(all(isinstance(size, int) and 0 <= size <= 998 for size in sizes),
            "A candidate pool exceeds 998 or has an invalid size")
    require(coverage["groups"] == len(rows) and coverage["maximum_pool_size"] == max(sizes),
            "Saved cohort or maximum pool size differs")
    close(coverage["average_pool_size"], sum(sizes) / len(sizes), "average pool size")
    for key, field in (("available_relevant_pairs", "available_relevant_count"),
                       ("full_relevant_pairs", "full_relevant_count")):
        require(coverage[key] == sum(row[field] for row in rows), "Saved relevance pair total differs")
    require(coverage["groups_without_available_target"] == sum(not row["available_relevant_count"] for row in rows),
            "Zero-availability contexts were omitted")
    for scope, denominator in (("conditional_available", "available_relevant_count"),
                               ("full_truth", "full_relevant_count")):
        for channel, hits, suffix in (("candidate_pool", "pool_hits", "candidate_recall"),
                                       ("reranker", "s100_hits", "recall_at_50"),
                                       ("baseline_rrf", "baseline_hits", "recall_at_50")):
            text, group, n, texts = mean_recall(rows, hits, denominator)
            recorded = saved[scope][channel]
            require(recorded["groups"] == n and recorded["texts"] == texts, "Saved metric cohort differs")
            close(recorded["text_macro_" + suffix], text, scope + "/" + channel)
            close(recorded["group_macro_" + suffix], group, scope + "/" + channel)
    if dense:
        close(saved["dense_recall"]["text_macro_recall_at_50"], mean_recall(rows, "dense_hits")[0], "dense449 recall")


def compact(name, saved):
    available = saved["conditional_available"]
    return {"variant": name,
            "pool_recall": available["candidate_pool"]["text_macro_candidate_recall"],
            "hgb_recall_at_50": available["reranker"]["text_macro_recall_at_50"],
            "rrf_recall_at_50": available["baseline_rrf"]["text_macro_recall_at_50"],
            "group_macro_hgb_recall_at_50": available["reranker"]["group_macro_recall_at_50"],
            "average_pool_size": saved["coverage"]["average_pool_size"],
            "maximum_pool_size": saved["coverage"]["maximum_pool_size"],
            "dense_recall_at_449": saved.get("dense_recall", {}).get("text_macro_recall_at_50")}


def summarize(report_path, baseline_report_path, baseline_scores_path):
    report, baseline_report = read_json(report_path), read_json(baseline_report_path)
    require(report.get("complete") is True and report.get("phase") == "tune"
            and report.get("groups") == EXPECTED_GROUPS and report.get("group_limit") == 0,
            "Only a completed full 12103-context Tune report may be summarized")
    require(report.get("audit_labels_read") is False and report.get("maximum_pool_size") == 998
            and report.get("reranker_retrained") is False and report.get("model_iteration") == 100,
            "Experiment scope differs from the declared frozen-model comparison")
    require(report.get("weights") == [0., .02, .05, .1] and report.get("primary") == "geo_0.05"
            and set(report["variants"]) == set(ARMS), "Declared experiment arms differ")
    require(baseline_report.get("phase") == "tune"
            and baseline_report["protocol_sha256"] == report["protocol_sha256"], "Baseline protocol differs")
    metadata_info = baseline_report["new_pipeline_identity"]["metadata"]
    require(sha256(metadata_info["path"]) == metadata_info["sha256"], "Baseline metadata changed")
    metadata = read_json(metadata_info["path"])
    groups = metadata["groups"]
    expected_ids = [g["group_identity"] for g in groups]
    require(len(expected_ids) == EXPECTED_GROUPS and len(set(expected_ids)) == EXPECTED_GROUPS,
            "Baseline metadata does not contain the full unique Tune cohort")
    require(sha256(baseline_scores_path) == baseline_report["group_scores_sha256"], "Baseline scores changed")
    old = {}
    with Path(baseline_scores_path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)["new"]
            key = row["group_identity"]
            require(key not in old, "Duplicate baseline context")
            old[key] = row
    require(set(old) == set(expected_ids), "Baseline scores omit or add Tune contexts")
    score_path = Path(report_path).parent / "group-scores.jsonl"
    require(sha256(score_path) == report["group_scores_sha256"], "Experiment group scores changed")
    rows = {name: [] for name in ARMS}
    observed_ids = []
    with score_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            key = record["group_identity"]
            observed_ids.append(key)
            require(key in old and set(record["variants"]) == set(ARMS), "Unknown context or arm")
            for name, row in record["variants"].items():
                require(row["group_identity"] == key and all(row[field] == old[key][field] for field in DENOMINATOR_FIELDS),
                        "An experiment arm changes the baseline query or Recall denominator")
                rows[name].append(row)
    require(observed_ids == expected_ids, "Experiment does not cover all Tune contexts in original order")
    baseline_rows = [old[key] for key in expected_ids]
    for group, row in zip(groups, baseline_rows):
        require(all(group[field] == row[field] for field in DENOMINATOR_FIELDS), "Baseline metadata and scores differ")
    verify_aggregate(baseline_rows, report["baseline"])
    for name in ARMS:
        verify_aggregate(rows[name], report["variants"][name], dense=True)
    table = [compact("baseline_v8_ann", report["baseline"])] + [compact(name, report["variants"][name]) for name in ARMS]
    primary_intervals = {}
    for comparison, reference in (("paired_vs_baseline", report["baseline"]),
                                   ("paired_vs_exact_cosine", report["variants"]["exact_cosine"])):
        intervals = report[comparison]["geo_0.05"]
        primary_intervals[comparison] = {}
        for metric, channel, key in (("candidate_pool_recall", "candidate_pool", "text_macro_candidate_recall"),
                                    ("reranker_recall_at_50", "reranker", "text_macro_recall_at_50"),
                                    ("rrf_recall_at_50", "baseline_rrf", "text_macro_recall_at_50")):
            interval = intervals[metric]
            delta = report["variants"]["geo_0.05"]["conditional_available"][channel][key] - reference["conditional_available"][channel][key]
            close(interval["delta"], delta, comparison + "/" + metric)
            lo, hi = interval["percentile_95_interval"]
            require(math.isfinite(lo) and math.isfinite(hi) and lo <= hi
                    and interval["bootstrap_unit"] == "normalized query text", "Invalid paired interval")
            primary_intervals[comparison][metric] = {"delta_percentage_points": 100 * delta,
                "percentile_95_interval_percentage_points": [100 * lo, 100 * hi],
                "query_text_clusters": interval["query_text_clusters"], "replicates": interval["replicates"]}
    return {"verified": True, "groups": EXPECTED_GROUPS,
        "metric": "Macro average over normalized query texts, available-positive denominator",
        "dense_depth": 449, "baseline_dense_recall": "Not recorded in saved baseline; left unavailable, not replaced with exact cosine",
        "table": table, "primary_weight": .05, "primary_intervals": primary_intervals,
        "coverage": report["baseline"]["coverage"],
        "feature_parity_groups": report["feature_parity_groups"],
        "maximum_feature_parity_difference": report["maximum_feature_parity_difference"],
        "interpretation": "Consumed Tune is exploratory. Same frozen HGB100; no reranker retraining or independent Audit confirmation.",
        "inputs": {str(Path(path).resolve()): sha256(path) for path in
                   (report_path, baseline_report_path, baseline_scores_path, score_path, metadata_info["path"])}}


def render(summary):
    lines = ["Full Tune: 12103 contexts; identical complete denominators; every pool <=998.",
             "Recall values are text-macro percentages among available positives.",
             "variant              pool%    HGB50%    RRF50%    dense449%    avg_pool    max_pool"]
    for row in summary["table"]:
        dense = "n/a" if row["dense_recall_at_449"] is None else f'{100 * row["dense_recall_at_449"]:.5f}'
        lines.append(f'{row["variant"]:20} {100 * row["pool_recall"]:8.5f} {100 * row["hgb_recall_at_50"]:9.5f} '
                     f'{100 * row["rrf_recall_at_50"]:9.5f} {dense:>12} {row["average_pool_size"]:11.3f} {row["maximum_pool_size"]:11}')
    lines += ["", "Primary geography weight0.05; deltas in percentage points, paired95% text-bootstrap intervals:"]
    for comparison, metrics in summary["primary_intervals"].items():
        lines.append(comparison)
        for name, values in metrics.items():
            lo, hi = values["percentile_95_interval_percentage_points"]
            lines.append(f'  {name}: {values["delta_percentage_points"]:+.5f} [{lo:+.5f}, {hi:+.5f}]')
    lines += ["", summary["interpretation"], summary["baseline_dense_recall"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "baseline-report", "baseline-scores", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "Use a new summary output directory")
    result = summarize(args.report, args.baseline_report, args.baseline_scores)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text = render(result)
    (args.output_dir / "SUMMARY.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
