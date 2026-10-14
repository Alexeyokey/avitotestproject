"""Paired evidence from saved, compact reranker-audit scores; no model inference.

    python scripts/compare_audit_models.py \
      --audit-dir artifacts/reranker-audit \
      --audit-dir artifacts/reranker-audit-extended \
      --output artifacts/reranker-audit-comparison.json

Stages and the candidate model are selected only using tuning query texts.
Confirmation scores are then reported with paired confidence intervals. The
historical validation set was previously inspected, so confirmation is not a
new untouched test. Bootstrap clusters are normalized query texts; the point
estimate and each replicate retain the original macro average over query groups.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re


STRATUM_KEYS = ("no_exact_location", "has_filters", "one_word")


def normalize_text(value: str) -> str:
    return " ".join(str(value).lower().replace("ё", "е").split())


def align_query_rows(reference: list[dict], candidate: list[dict]) -> list[dict]:
    """Reorder by reference group ID and reject changed query identities or pools."""
    if len(reference) != len(candidate):
        raise ValueError("Compared score files have different query-group counts")
    indexed = {}
    for row in candidate:
        group_id = int(row["group_id"])
        if group_id in indexed:
            raise ValueError("Duplicate group_id in candidate score file")
        indexed[group_id] = row
    if len({int(row["group_id"]) for row in reference}) != len(reference):
        raise ValueError("Duplicate group_id in reference score file")
    aligned = []
    text_partitions = {}
    for original in reference:
        group_id = int(original["group_id"])
        if group_id not in indexed:
            raise ValueError(f"Candidate lacks group_id={group_id}")
        row = indexed[group_id]
        if normalize_text(original["query_text"]) != normalize_text(row["query_text"]):
            raise ValueError(f"Query text differs for group_id={group_id}")
        for key in ("relevant_count", "partition", "pool_size", "positives_in_pool", "baseline_hits"):
            if original.get(key) != row.get(key):
                raise ValueError(f"{key} differs for group_id={group_id}")
        if "query_key" in original and "query_key" in row:
            if tuple(map(normalize_text, original["query_key"])) != tuple(map(normalize_text, row["query_key"])):
                raise ValueError(f"query_key differs for group_id={group_id}")
        if int(row["relevant_count"]) < 1:
            raise ValueError("Recall denominator must be positive")
        if row["partition"] not in {"tune", "confirmation"}:
            raise ValueError("Expected tune/confirmation partition")
        text = normalize_text(row["query_text"])
        previous = text_partitions.setdefault(text, row["partition"])
        if previous != row["partition"]:
            raise ValueError("One query text crosses tune and confirmation")
        for key in STRATUM_KEYS:
            if key in original and key in row and bool(original[key]) != bool(row[key]):
                raise ValueError(f"Query stratum differs for group_id={group_id}: {key}")
        aligned.append(row)
    return aligned


def stage_recall(row: dict, stage: int) -> float:
    hits = int(row[f"s{stage}_all_hits"])
    denominator = int(row["relevant_count"])
    if denominator < 1 or hits < 0 or hits > denominator:
        raise ValueError("Invalid Recall numerator or corpus denominator")
    value = hits / denominator
    saved = row.get(f"s{stage}_recall")
    if saved is not None and not math.isclose(value, float(saved), abs_tol=1e-12):
        raise ValueError("Saved Recall differs from full-corpus denominator")
    return value


def available_stages(rows: list[dict]) -> tuple[int, ...]:
    if not rows:
        raise ValueError("Empty score file")
    stages = tuple(sorted(int(match.group(1)) for key in rows[0]
                          if (match := re.fullmatch(r"s(\d+)_all_hits", key))))
    if not stages:
        raise ValueError("No model-prefix hit columns in score file")
    for row in rows:
        if any(f"s{stage}_all_hits" not in row for stage in stages):
            raise ValueError("Model-prefix columns differ between query rows")
    return stages


def choose_stage_from_tune(rows: list[dict], stages: tuple[int, ...]) -> tuple[int, float]:
    """Never inspect confirmation values during prefix selection."""
    tuning = [row for row in rows if row["partition"] == "tune"]
    if not tuning:
        raise ValueError("Stage selection requires tuning groups")
    best_stage, best_score = None, -math.inf
    for stage in sorted(stages):
        score = sum(stage_recall(row, stage) for row in tuning) / len(tuning)
        if score > best_score + 1e-12:
            best_stage, best_score = stage, score
    return int(best_stage), best_score


def choose_model_from_tune(selections: dict) -> str:
    """Ties use fewer fitted prefix iterations, then a stable model name."""
    if not selections:
        raise ValueError("No candidate models")
    best_score = max(value["tune_recall"] for value in selections.values())
    tied = [name for name, value in selections.items()
            if math.isclose(value["tune_recall"], best_score, rel_tol=0, abs_tol=1e-12)]
    return min(tied, key=lambda name: (selections[name]["stage"], name))


def validate_reference_stage(models: dict, name: str, stage: int) -> None:
    if (name not in models
            or models[name]["report"]["model_class"] != "HistGradientBoostingClassifier"
            or stage not in models[name]["stages"]):
        raise ValueError(f"Reference must identify a HistGradientBoosting model with stage {stage}")


def reference_stage_scores(rows: list[dict], stage: int) -> list[float]:
    """Extract the requested reference prefix without substituting its final stage."""
    if stage not in available_stages(rows):
        raise ValueError(f"Reference score file lacks stage {stage}")
    return [stage_recall(row, stage) for row in rows]


def verify_comparison_compatibility(reference: dict, candidate: dict,
                                    *, allow_rescore_only: bool = False) -> dict:
    """Permit one optional dense-rescore flag only with immutable pool proof."""
    reference_config = reference.get("retrieval_config")
    candidate_config = candidate.get("retrieval_config")
    if not isinstance(reference_config, dict) or not isinstance(candidate_config, dict):
        raise ValueError("Comparison reports require retrieval_config mappings")
    reference_policy = reference.get("inference_prior_policy")
    candidate_policy = candidate.get("inference_prior_policy")
    required_policy = {"inference_priors_sha256", "disable_microcategory_prior"}
    policies_present = reference_policy is not None or candidate_policy is not None
    if policies_present:
        if (not isinstance(reference_policy, dict) or not isinstance(candidate_policy, dict)
                or not required_policy <= reference_policy.keys()
                or not required_policy <= candidate_policy.keys()):
            raise ValueError("Both reports need complete inference prior policy")
        for key in required_policy:
            if reference_policy[key] != candidate_policy[key]:
                raise ValueError(f"Inference prior policy differs: {key}")
    if reference_config == candidate_config:
        return {"mode": "strict_config_match", "inference_policy_compared": policies_present}
    if not allow_rescore_only:
        raise ValueError("Retrieval configuration differs; strict comparison requires equality")
    left, right = dict(reference_config), dict(candidate_config)
    reference_rescore = left.pop("rescore_dense_union", False)
    candidate_rescore = right.pop("rescore_dense_union", False)
    if (left != right or type(reference_rescore) is not bool or type(candidate_rescore) is not bool
            or reference_rescore == candidate_rescore):
        raise ValueError("Only rescore_dense_union=False/missing versus True may differ")
    digest = reference.get("candidate_pool_digest")
    if not isinstance(digest, str) or not digest or digest != candidate.get("candidate_pool_digest"):
        raise ValueError("Dense-rescore comparison requires identical nonempty candidate_pool_digest")
    if (not policies_present or not isinstance(reference_policy["inference_priors_sha256"], str)
            or not reference_policy["inference_priors_sha256"]):
        raise ValueError("Dense-rescore comparison requires identical nonempty inference prior hashes and disable flags")
    return {"mode": "dense_rescore_flag_only_with_immutable_pool",
            "candidate_pool_digest": digest,
            "inference_prior_policy": reference_policy,
            "reference_rescore_dense_union": reference_rescore,
            "candidate_rescore_dense_union": candidate_rescore}


def query_text_clusters(rows: list[dict]) -> list[list[int]]:
    clusters = defaultdict(list)
    for index, row in enumerate(rows):
        clusters[normalize_text(row["query_text"])].append(index)
    return list(clusters.values())


def cluster_resample_group_mean(values: list[float], clusters: list[list[int]], draw: list[int]) -> float:
    """Keep query-group macro weighting when a sampled text has several groups."""
    numerator, denominator = 0.0, 0
    for cluster_index in draw:
        indices = clusters[cluster_index]
        numerator += sum(values[index] for index in indices)
        denominator += len(indices)
    if not denominator:
        raise ValueError("Bootstrap draw has no query groups")
    return numerator / denominator


def ci_evidence(lower: float, upper: float) -> str:
    if lower > 0:
        return "positive_paired_evidence"
    if upper < 0:
        return "negative_paired_evidence"
    return "uncertain_interval_includes_zero"


def paired_cluster_bootstrap(rows: list[dict], candidate: list[float], reference: list[float],
                             replicates: int, seed: int) -> dict:
    import numpy as np
    if len(rows) != len(candidate) or len(rows) != len(reference) or not rows:
        raise ValueError("Paired comparison requires aligned nonempty score vectors")
    differences = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    clusters = query_text_clusters(rows)
    sums = np.asarray([differences[indices].sum() for indices in clusters])
    counts = np.asarray([len(indices) for indices in clusters], dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = []
    # Bounded batches avoid a replicates*texts allocation on memory-limited WSL.
    for start in range(0, replicates, 64):
        draw = rng.integers(0, len(clusters), size=(min(64, replicates - start), len(clusters)))
        estimates.extend((sums[draw].sum(axis=1) / counts[draw].sum(axis=1)).tolist())
    lower, upper = map(float, np.quantile(estimates, [0.025, 0.975]))
    return {"query_groups": len(rows), "query_text_clusters": len(clusters),
            "candidate_recall": float(np.mean(candidate)), "reference_recall": float(np.mean(reference)),
            "delta": float(differences.mean()), "delta_ci95": [lower, upper],
            "bootstrap_replicates": replicates, "seed": seed,
            "evidence": ci_evidence(lower, upper),
            "method": "Paired bootstrap of normalized query-text clusters; macro average remains over query groups."}


def matched_mix_bootstrap(rows: list[dict], candidate: list[float], reference: list[float],
                          benchmark_strata: list[dict], replicates: int, seed: int) -> dict | None:
    import numpy as np
    if (not rows or not benchmark_strata
            or not all(all(key in row for key in STRATUM_KEYS) for row in rows)):
        return None
    targets = defaultdict(int)
    for entry in benchmark_strata:
        targets[tuple(bool(entry[key]) for key in STRATUM_KEYS)] += int(entry["queries"])
    present = {tuple(bool(row[key]) for key in STRATUM_KEYS) for row in rows}
    active = sorted(key for key, value in targets.items() if value > 0 and key in present)
    total_target = sum(targets.values())
    covered_target = sum(targets[key] for key in active)
    if not active or not total_target:
        return None
    lookup = {key: index for index, key in enumerate(active)}
    weights = np.asarray([targets[key] for key in active], dtype=np.float64)
    weights /= weights.sum()
    row_strata = [lookup.get(tuple(bool(row[key]) for key in STRATUM_KEYS)) for row in rows]
    differences = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    clusters = query_text_clusters(rows)
    sums = np.zeros((len(clusters), len(active)), dtype=np.float64)
    counts = np.zeros_like(sums)
    for cluster_index, indices in enumerate(clusters):
        for row_index in indices:
            stratum = row_strata[row_index]
            if stratum is not None:
                sums[cluster_index, stratum] += differences[row_index]
                counts[cluster_index, stratum] += 1
    point = float(np.dot(weights, sums.sum(axis=0) / counts.sum(axis=0)))
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, replicates, 32):
        draw = rng.integers(0, len(clusters), size=(min(32, replicates - start), len(clusters)))
        sampled_sums, sampled_counts = sums[draw].sum(axis=1), counts[draw].sum(axis=1)
        valid = (sampled_counts > 0).all(axis=1)
        estimates.extend(((sampled_sums[valid] / sampled_counts[valid]) @ weights).tolist())
    if len(estimates) < 2:
        return {"delta": point, "delta_ci95": None,
                "covered_target_fraction": covered_target / total_target,
                "evidence": "insufficient_bootstrap_stratum_coverage",
                "valid_bootstrap_replicates": len(estimates)}
    lower, upper = map(float, np.quantile(estimates, [0.025, 0.975]))
    candidate_means, reference_means = [], []
    for key in active:
        indices = [index for index, row in enumerate(rows)
                   if tuple(bool(row[name]) for name in STRATUM_KEYS) == key]
        candidate_means.append(sum(candidate[index] for index in indices) / len(indices))
        reference_means.append(sum(reference[index] for index in indices) / len(indices))
    return {"candidate_recall": float(np.dot(weights, candidate_means)),
            "reference_recall": float(np.dot(weights, reference_means)),
            "delta": point, "delta_ci95": [lower, upper],
            "covered_target_fraction": covered_target / total_target,
            "target_queries": total_target, "covered_target_queries": covered_target,
            "valid_bootstrap_replicates": len(estimates), "requested_bootstrap_replicates": replicates,
            "seed": seed, "evidence": ci_evidence(lower, upper),
            "method": "Paired query-text cluster bootstrap with fixed benchmark stratum weights.",
            "coverage_note": "Target strata absent from this partition are excluded; replicates missing a covered stratum are discarded."}


def resolve_saved_path(value: str, audit_dir: Path) -> Path:
    path = Path(value)
    for candidate in (path, audit_dir / path.name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Saved audit file does not exist: {value}")


def load_audits(audit_dirs: list[Path]) -> dict:
    import pyarrow.parquet as pq
    models = {}
    for audit_dir in audit_dirs:
        manifest = json.loads((audit_dir / "report.json").read_text(encoding="utf-8"))
        for name, saved_report in manifest["reports"].items():
            report_path = resolve_saved_path(saved_report, audit_dir)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            score_path = resolve_saved_path(report["per_query_scores"]["validation"], audit_dir)
            label = name
            if label in models:
                if report.get("sha256") == models[label]["report"].get("sha256"):
                    continue
                label = f"{name}@{audit_dir.name}"
                if label in models:
                    raise ValueError(f"Ambiguous model name: {label}")
            # These are compact per-query scores, never the candidate feature caches.
            rows = pq.read_table(score_path).to_pylist()
            models[label] = {"report": report, "report_path": str(report_path),
                             "score_path": str(score_path), "rows": rows,
                             "stages": available_stages(rows)}
    return models


def comparison_partitions(rows: list[dict], candidate: list[float], reference: list[float],
                           benchmark_strata: list[dict], replicates: int, seed: int) -> dict:
    output = {}
    for partition in ("all_validation", "tune", "confirmation"):
        indices = [index for index, row in enumerate(rows)
                   if partition == "all_validation" or row["partition"] == partition]
        if not indices:
            continue
        subset = [rows[index] for index in indices]
        left, right = [candidate[index] for index in indices], [reference[index] for index in indices]
        evidence = paired_cluster_bootstrap(subset, left, right, replicates, seed)
        matched = matched_mix_bootstrap(subset, left, right, benchmark_strata, replicates, seed)
        if matched is not None:
            evidence["benchmark_query_mix"] = matched
        output[partition] = evidence
    return output


def final_stage_overfit_evidence(model: dict, rows: list[dict], selected_stage: int,
                                 benchmark_strata: list[dict], replicates: int, seed: int) -> dict:
    final_stage = max(model["stages"])
    selected = [stage_recall(row, selected_stage) for row in rows]
    final = [stage_recall(row, final_stage) for row in rows]
    comparisons = comparison_partitions(rows, final, selected, benchmark_strata, replicates, seed)
    confirmation = comparisons.get("confirmation", {})
    interval = confirmation.get("delta_ci95")
    flag = bool(final_stage > selected_stage and interval is not None and interval[1] < 0)
    row_losses = model["report"].get("sampled_training_row_loss", {})
    losses = row_losses.get("stages", {})
    first_loss = losses.get(str(selected_stage), {}).get("weighted_log_loss")
    last_loss = losses.get(str(final_stage), {}).get("weighted_log_loss")
    gaps = model["report"].get("train_minus_validation_recall_gap", {})
    return {"selected_stage": selected_stage, "final_stage": final_stage,
            "final_minus_selected_stage": comparisons,
            "possible_excess_trees_on_confirmation": flag,
            "selected_stage_weighted_training_log_loss": first_loss,
            "final_stage_weighted_training_log_loss": last_loss,
            "training_log_loss_decreased": (last_loss < first_loss if first_loss is not None and last_loss is not None else None),
            "selected_stage_train_minus_validation_recall_gap": gaps.get(str(selected_stage)),
            "final_stage_train_minus_validation_recall_gap": gaps.get(str(final_stage)),
            "interpretation": ("More trees hurt confirmation Recall with an interval below zero; possible overfitting or excess capacity on this reused holdout."
                               if flag else "These prefix comparisons do not establish overfitting; inspect paired confirmation intervals and the training curve."),
            "caveat": "Training loss or a train/validation gap alone is not proof of overfitting; confirmation is historical validation already used in earlier work."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit-dir", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-name", help="Optional label of the HistGradientBoosting reference")
    parser.add_argument("--reference-stage", type=int, default=150,
                        help="Saved prefix of the HistGradientBoosting reference (default: 150)")
    parser.add_argument("--allow-rescore-only-config-difference", action="store_true",
                        help="Allow only rescore_dense_union=False/missing versus True when pool digests and inference prior policy match")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args(argv)
    if args.bootstrap_replicates < 2:
        parser.error("At least two bootstrap replicates are required")
    if args.reference_stage < 1:
        parser.error("Reference stage must be positive")
    models = load_audits(args.audit_dir)
    references = [name for name, model in models.items()
                  if model["report"]["model_class"] == "HistGradientBoostingClassifier"
                  and args.reference_stage in model["stages"]]
    if args.reference_name:
        if args.reference_name not in references:
            parser.error(f"--reference-name must identify a HistGradientBoosting model with stage {args.reference_stage}")
        reference_name = args.reference_name
    elif references:
        reference_name = min(references, key=lambda name: (max(models[name]["stages"]) != args.reference_stage, name))
    else:
        parser.error(f"Audits lack a HistGradientBoosting stage-{args.reference_stage} reference")
    validate_reference_stage(models, reference_name, args.reference_stage)
    reference_model = models[reference_name]
    rows = reference_model["rows"]
    # Validate the reference itself as well as every candidate.
    rows = align_query_rows(rows, rows)
    reference_scores = reference_stage_scores(rows, args.reference_stage)
    baseline = [int(row["baseline_hits"]) / int(row["relevant_count"]) for row in rows]
    benchmark_strata = reference_model["report"].get("benchmark_strata", [])
    selections, comparisons = {}, {}
    for name, model in models.items():
        candidate_rows = align_query_rows(rows, model["rows"])
        compatibility = verify_comparison_compatibility(
            reference_model["report"], model["report"],
            allow_rescore_only=args.allow_rescore_only_config_difference)
        stage, tune_recall = choose_stage_from_tune(candidate_rows, model["stages"])
        selections[name] = {"stage": stage, "tune_recall": tune_recall,
                            "model_class": model["report"]["model_class"],
                            "report_path": model["report_path"], "score_path": model["score_path"]}
        candidate_scores = [stage_recall(row, stage) for row in candidate_rows]
        reference_comparison = comparison_partitions(
            rows, candidate_scores, reference_scores, benchmark_strata, args.bootstrap_replicates, args.seed)
        comparisons[name] = {
            "selected_stage": stage,
            "comparison_compatibility": compatibility,
            "selected_stage_vs_reference": reference_comparison,
            f"selected_stage_vs_hgb{args.reference_stage}": reference_comparison,
            "selected_stage_vs_baseline": comparison_partitions(
                rows, candidate_scores, baseline, benchmark_strata, args.bootstrap_replicates, args.seed),
            "overfit_prefix_evidence": final_stage_overfit_evidence(
                model, candidate_rows, stage, benchmark_strata, args.bootstrap_replicates, args.seed)}
        confirmation = comparisons[name]["selected_stage_vs_reference"].get("confirmation", {})
        print(f"{name}: tune-selected stage={stage}, tune Recall={tune_recall:.6f}, "
              f"confirmation delta vs HGB{args.reference_stage}={confirmation.get('delta')}, "
              f"CI={confirmation.get('delta_ci95')}", flush=True)
    chosen = choose_model_from_tune(selections)
    chosen_confirmation = comparisons[chosen]["selected_stage_vs_reference"].get("confirmation", {})
    output = {
        "audit_dirs": [str(path) for path in args.audit_dir],
        "reference": {"name": reference_name, "stage": args.reference_stage,
                      "model_sha256": reference_model["report"].get("sha256")},
        "bootstrap_replicates": args.bootstrap_replicates, "seed": args.seed,
        "allow_rescore_only_config_difference": args.allow_rescore_only_config_difference,
        "historical_validation_reused": True, "fresh_confirmation_holdout": False,
        "selection_method": "Select each prefix and then the model solely on tune macro Recall@50. Ties prefer fewer prefix iterations, then the model name.",
        "tune_selected_model": chosen, "selections": selections,
        "selected_model_confirmation_evidence": chosen_confirmation.get("evidence"),
        "automatic_winner_declared": False,
        "reference_vs_baseline": comparison_partitions(
            rows, reference_scores, baseline, benchmark_strata, args.bootstrap_replicates, args.seed),
        "models": comparisons,
        "caveats": ["Historical validation queries have already informed prior experiments; this confirmation split is not a new untouched test.",
                    "Tune and all-validation intervals describe selected candidates and are affected by selection. Confirmation estimates are reported separately.",
                    "An interval that includes zero leaves the model improvement uncertain; tuning rank alone does not establish a winner.",
                    "Bootstrap clusters preserve all query groups for the same normalized text and retain macro averaging over query groups.",
                    "Clicked examples are known positives; sampled unobserved examples are not proven irrelevant. Training row loss is not Recall or proof of overfitting."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved comparison: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
