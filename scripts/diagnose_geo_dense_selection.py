"""Post-hoc paired ranking comparisons on completed Tune results only.

This creates no predictions or fitted parameters. Confidence intervals describe
exploratory comparisons selected after viewing Tune, not an independent test.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from summarize_geo_dense_tune import ARMS, EXPECTED_GROUPS, require, sha256, summarize


def run(experiment_dir, baseline_report, baseline_scores, *, replicates=10000, seed=20261014):
    experiment_dir = Path(experiment_dir)
    report_path = experiment_dir / "full-tune" / "report.json"
    scores_path = report_path.parent / "group-scores.jsonl"
    verified = summarize(report_path, baseline_report, baseline_scores)
    old = {}
    with Path(baseline_scores).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)["new"]
            old[row["group_identity"]] = row
    groups = []
    with scores_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            rows = {**record["variants"], "baseline_v8": old[record["group_identity"]]}
            for row in rows.values():
                available, full = row["available_relevant_count"], row["full_relevant_count"]
                require(0 <= available <= full, "Available denominator exceeds full truth")
                require(0 <= row["pool_hits"] <= min(available, row["pool_size"]), "Impossible pool hit count")
                require(0 <= row["pool_size"] < 1000, "Candidate budget exceeded")
                for hit in ("baseline_hits", "s100_hits"):
                    require(0 <= row[hit] <= min(50, available), "Impossible top50 hit count")
                if "dense_hits" in row:
                    require(0 <= row["dense_hits"] <= min(449, available), "Impossible dense449 hit count")
            groups.append(rows)
    require(len(groups) == EXPECTED_GROUPS, "Incomplete full Tune results")
    comparisons = (
        ("geo002_rrf_minus_geo002_hgb", "geo_0.02", "baseline_hits", "geo_0.02", "s100_hits"),
        ("geo002_rrf_minus_baseline_v8_hgb", "geo_0.02", "baseline_hits", "baseline_v8", "s100_hits"),
        ("geo002_rrf_minus_geo005_hgb", "geo_0.02", "baseline_hits", "geo_0.05", "s100_hits"),
    )
    by_text = defaultdict(list)
    for rows in groups:
        row = rows["geo_0.02"]
        count = row["available_relevant_count"]
        if not count:
            continue
        by_text[row["query_text"]].append([
            (rows[left][left_hit] / count, rows[right][right_hit] / count)
            for _, left, left_hit, right, right_hit in comparisons])
    # Normalize within each text first; repeated locations/filter contexts do
    # not give a frequently occurring query text extra bootstrap weight.
    text_values = np.asarray([
        [[math.fsum(row[i][j] for row in entries) / len(entries) for j in (0, 1)]
         for i in range(len(comparisons))]
        for _, entries in sorted(by_text.items())], dtype=np.float64)
    differences = text_values[:, :, 0] - text_values[:, :, 1]
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, len(comparisons)), dtype=np.float64)
    for start in range(0, replicates, 128):
        count = min(128, replicates - start)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        samples[start:start + count] = differences[indices].mean(axis=1)
    intervals = np.quantile(samples, [.025, .975], axis=0)
    results = {}
    for i, (name, left, left_hit, right, right_hit) in enumerate(comparisons):
        lo, hi = map(float, intervals[:, i])
        results[name] = {
            "left": {"pipeline": left, "hit_counter": left_hit,
                     "text_macro_recall_at_50": float(text_values[:, i, 0].mean())},
            "right": {"pipeline": right, "hit_counter": right_hit,
                      "text_macro_recall_at_50": float(text_values[:, i, 1].mean())},
            "delta": float(differences[:, i].mean()),
            "delta_percentage_points": float(100 * differences[:, i].mean()),
            "percentile_95_interval": [lo, hi],
            "percentile_95_interval_percentage_points": [100 * lo, 100 * hi],
            "interval_excludes_zero": lo > 0 or hi < 0,
            "query_text_clusters": len(differences),
            "improved_texts": int((differences[:, i] > 0).sum()),
            "worsened_texts": int((differences[:, i] < 0).sum()),
            "equal_texts": int((differences[:, i] == 0).sum()),
        }
    return {"complete": True, "phase": "tune", "audit_accessed": False,
        "new_experiments_run": False, "groups": EXPECTED_GROUPS,
        "groups_with_available_target": sum(bool(rows["geo_0.02"]["available_relevant_count"]) for rows in groups),
        "bootstrap_unit": "normalized query text", "replicates": replicates, "seed": seed,
        "random_generator": "numpy.default_rng PCG64; shared paired resamples across comparisons",
        "quantile_method": "numpy.quantile default linear, 2.5 and 97.5 percentiles",
        "denominator": "All available positives, including positives missing from every candidate pool",
        "all_12103_groups_and_denominators_verified": verified["verified"],
        "all_five_pipelines_maximum_pool_size": max(row["pool_size"] for rows in groups for row in rows.values()),
        "comparisons": results,
        "interpretation": "Exploratory comparisons chosen after viewing consumed Tune. Intervals are not corrected for model/weight selection or multiple comparisons and are not independent confirmation.",
        "inputs": {str(Path(path).resolve()): sha256(path) for path in
                   (report_path, scores_path, baseline_report, baseline_scores)},
        "source_sha256": sha256(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment-dir", "baseline-report", "baseline-scores"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    require(args.replicates >= 2000, "Use at least2000 resamples")
    output = args.experiment_dir / "selection-diagnostics.json"
    require(not output.exists(), "Selection diagnostics already exist")
    result = run(args.experiment_dir, args.baseline_report, args.baseline_scores,
                 replicates=args.replicates, seed=args.seed)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(output), "comparisons": result["comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
