"""Compare held-out Recall@50 after matching observable benchmark query mix.

This uses *only* benchmark query attributes, never hidden labels. The matched
number is diagnostic: held-out positives are still selected by their presence
in benchmark_items, so it is not an unbiased leaderboard estimate.

Example:
  python scripts/benchmark_like_validation.py --data-dir dataset \
    --report old=artifacts/old-eval.rrf.csv \
    --report new=artifacts/new-eval.rrf.csv --output artifacts/validation-mix.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random

import pyarrow.parquet as pq

from avito_candidates.cli import read_validation
from avito_candidates.core import held_out, normalize, query_key


def signature(row, locations):
    """Observable query properties that differ between benchmark and holdout."""
    text = normalize(row.get("search_query", ""))
    return (
        str(row.get("search_location_id", "")) not in locations,
        bool(row.get("search_infm_params_text", "")),
        len(text.split()) <= 1,
    )


def profile(rows, locations, fit_texts):
    rows = list(rows)
    local_counts = [locations.get(str(row.get("search_location_id", "")), 0)
                    for row in rows]
    return {
        "queries": len(rows),
        "text_seen_in_fit_fraction": sum(normalize(row.get("search_query", ""))
                                         in fit_texts for row in rows) / len(rows),
        "no_exact_location_fraction": sum(count == 0 for count in local_counts) / len(rows),
        "same_location_corpus_items_median": sorted(local_counts)[len(rows) // 2],
        "has_filters_fraction": sum(bool(row.get("search_infm_params_text", ""))
                                    for row in rows) / len(rows),
        "one_word_fraction": sum(len(normalize(row.get("search_query", "")).split()) <= 1
                                 for row in rows) / len(rows),
    }


def load_predictions(path):
    predictions = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            predictions[query_key(row)].add(row["item_id"])
    return predictions


def score_report(predictions, truth, representatives, target_counts, locations,
                 fit_items):
    if set(predictions) != set(truth):
        raise ValueError("Prediction query set differs from held-out labels")
    validation_counts = Counter(signature(row, locations) for row in representatives.values())
    unsupported = [stratum for stratum, count in target_counts.items()
                   if count and not validation_counts[stratum]]
    if unsupported:
        raise ValueError(f"Benchmark query strata absent in validation: {unsupported}")
    total, matched_sum, matched_mass, matched_sq = 0.0, 0.0, 0.0, 0.0
    strata_scores = defaultdict(list)
    for key, positives in truth.items():
        row = representatives[key]
        group = signature(row, locations)
        recall = len(predictions[key] & positives) / len(positives)
        weight = target_counts[group] / validation_counts[group]
        total += recall
        matched_sum += weight * recall
        matched_mass += weight
        matched_sq += weight * weight
        strata_scores[group].append(recall)
    unseen_queries = [key for key, positives in truth.items()
                      if all(item_id not in fit_items for item_id in positives)]
    seen_pairs = [(key, item_id) for key, positives in truth.items()
                  for item_id in positives if item_id in fit_items]
    unseen_pairs = [(key, item_id) for key, positives in truth.items()
                    for item_id in positives if item_id not in fit_items]
    return {
        "ordinary_recall_at_50": total / len(truth),
        "query_mix_matched_recall_at_50": matched_sum / matched_mass,
        "effective_sample_size": matched_mass ** 2 / matched_sq,
        "unseen_positive_only_queries": len(unseen_queries),
        "unseen_positive_only_recall_at_50": sum(
            len(predictions[key] & truth[key]) / len(truth[key])
            for key in unseen_queries) / len(unseen_queries),
        "seen_positive_pair_hit_fraction": sum(
            item_id in predictions[key] for key, item_id in seen_pairs) / len(seen_pairs),
        "unseen_positive_pair_hit_fraction": sum(
            item_id in predictions[key] for key, item_id in unseen_pairs) / len(unseen_pairs),
        "strata": [
            {"no_exact_location": group[0], "has_filters": group[1],
             "one_word": group[2], "benchmark_queries": target_counts[group],
             "heldout_queries": validation_counts[group],
             "heldout_recall_at_50": sum(scores) / len(scores)}
            for group, scores in sorted(strata_scores.items())
        ],
    }


def paired_comparison(old_predictions, new_predictions, truth, representatives,
                      target_counts, locations, *, seed, replicates=1000):
    """Cluster the paired Recall difference by query text, not individual rows."""
    validation_counts = Counter(signature(row, locations)
                                for row in representatives.values())
    clusters = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for key, positives in truth.items():
        row = representatives[key]
        old = len(old_predictions[key] & positives) / len(positives)
        new = len(new_predictions[key] & positives) / len(positives)
        delta = new - old
        weight = target_counts[signature(row, locations)] / validation_counts[
            signature(row, locations)]
        entry = clusters[normalize(row.get("search_query", ""))]
        entry[0] += delta
        entry[1] += weight * delta
        entry[2] += weight
        entry[3] += 1
    groups = list(clusters.values())
    rng = random.Random(seed)
    ordinary, matched = [], []
    for _ in range(replicates):
        sampled = rng.choices(groups, k=len(groups))
        ordinary.append(sum(x[0] for x in sampled) / sum(x[3] for x in sampled))
        matched.append(sum(x[1] for x in sampled) / sum(x[2] for x in sampled))
    ordinary.sort()
    matched.sort()
    return {
        "query_text_clusters": len(groups),
        "delta_recall_at_50": sum(x[0] for x in groups) / sum(x[3] for x in groups),
        "delta_query_mix_matched_recall_at_50":
            sum(x[1] for x in groups) / sum(x[2] for x in groups),
        "cluster_bootstrap_95_interval": [ordinary[25], ordinary[975]],
        "matched_cluster_bootstrap_95_interval": [matched[25], matched[975]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--report", action="append", required=True,
                        help="NAME=path/to/evaluate.rrf.csv; repeat to compare runs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    item_table = pq.read_table(args.data_dir / "benchmark_items.parquet",
                               columns=["item_id", "item_location_id"])
    item_ids = set(item_table["item_id"].to_pylist())
    locations = Counter(str(value) for value in item_table["item_location_id"].to_pylist())
    benchmark = pq.read_table(args.data_dir / "benchmark_queries.parquet", columns=[
        "search_query", "search_location_id", "search_infm_params_text",
    ]).to_pylist()
    truth, representatives, fit_rows, heldout_rows, heldout_in_corpus = read_validation(
        args.data_dir / "train.parquet", item_ids, mode="queries",
        fraction=args.validation_fraction, seed=args.seed, batch_size=65536,
    )
    fit_texts, fit_items = set(), set()
    for batch in pq.ParquetFile(args.data_dir / "train.parquet").iter_batches(
        batch_size=65536, columns=["search_query", "item_id"]
    ):
        for raw_text, item_id in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            text = normalize(raw_text or "")
            if not held_out(text, args.validation_fraction, args.seed):
                fit_texts.add(text)
                fit_items.add(item_id)
    targets = Counter(signature(row, locations) for row in benchmark)
    positive_items = [item_id for positives in truth.values() for item_id in positives]
    result = {
        "split": "queries", "seed": args.seed,
        "benchmark_query_profile": profile(benchmark, locations, fit_texts),
        "heldout_query_profile": profile(representatives.values(), locations, fit_texts),
        "heldout_rows": heldout_rows, "heldout_in_corpus_fraction": heldout_in_corpus / heldout_rows,
        "corpus_items_seen_in_fit_fraction": len(item_ids & fit_items) / len(item_ids),
        "heldout_positive_items_seen_in_fit_fraction":
            sum(item_id in fit_items for item_id in positive_items) / len(positive_items),
        "limitation": "Matching observed query mix cannot correct positive-item selection bias.",
        "reports": {},
    }
    predictions_by_name = {}
    for specification in args.report:
        if "=" not in specification:
            parser.error("--report must be NAME=PATH")
        name, filename = specification.split("=", 1)
        predictions_by_name[name] = load_predictions(Path(filename))
        result["reports"][name] = score_report(
            predictions_by_name[name], truth, representatives, targets,
            locations, fit_items)
    if len(predictions_by_name) == 2:
        old_name, new_name = predictions_by_name
        result["paired_comparison"] = {
            "old": old_name, "new": new_name,
            **paired_comparison(predictions_by_name[old_name],
                                predictions_by_name[new_name], truth,
                                representatives, targets, locations, seed=args.seed),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, report in result["reports"].items():
        print(f"{name}: ordinary={report['ordinary_recall_at_50']:.6f} "
              f"matched={report['query_mix_matched_recall_at_50']:.6f} "
              f"unseen_items={report['unseen_positive_only_recall_at_50']:.6f} "
              f"effective_n={report['effective_sample_size']:.0f}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
