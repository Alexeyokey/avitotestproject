"""Compare a new V7/V8 pipeline with original HGB150 on the same Tune texts.

Each model is scored on its own SHA-bound cached pool/features, so candidate
pool changes do not masquerade as a same-pool reranker ablation. Complete GT
denominators, zero-availability groups and text-cluster bootstrap are retained.
Tune was consumed by retrieval hypothesis selection: these are exploratory
comparisons, not an untouched Audit, selection seal or official benchmark.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import evaluate_fresh_validation as EVAL
from cache_research_reranker import (validate_research_artifact, research_prior_policy,
                                    check_training_prior_file, check_sources)
from produce_document_hypotheses import check_encoder_base

FROZEN_REFERENCE_MODEL_SHA256 = "f19f86df4ac76b3f8721768f3b3e048a9a97382bb970bb7ea40572b2ef910363"
FROZEN_REFERENCE_TRAIN_SHA256 = "afce6ff62d031fd360ed583f900f73c6b3859f5c16d335f6c47a5d8471e967c6"


def check_frozen_reference(model, training_data):
    if (EVAL.sha256_file(Path(model)) != FROZEN_REFERENCE_MODEL_SHA256
            or EVAL.sha256_file(Path(training_data)) != FROZEN_REFERENCE_TRAIN_SHA256):
        raise ValueError("Original comparison must use the byte-identical corrected HGB150 and its corrected training cache")


def check_research_cache(metadata, artifact, train_hash):
    if (metadata.get("research_cache_version") != 1
            or metadata.get("research_train_data_sha256") != train_hash):
        raise ValueError("Research cache lacks its new model's sealed training provenance")
    expected = research_prior_policy(artifact)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Research cache V7 classifier or aggregate priors differ")
    if metadata.get("training_prior_policy") != artifact["metadata"].get("training_prior_policy"):
        raise ValueError("Research cache training prior policy differs from the new model")
    snapshot = metadata.get("research_source_sha256")
    if not isinstance(snapshot, dict) or not snapshot:
        raise ValueError("Research cache must seal its feature/retrieval source files")
    check_sources(snapshot)
    return expected


def aligned_rows(new_rows, old_rows):
    """Align canonical query identities; local integer IDs may differ."""
    def keyed(rows):
        result = {}
        for row in rows:
            key = row["group_identity"]
            if key in result:
                raise ValueError("Repeated canonical query group in comparison")
            result[key] = row
        return result
    new, old = keyed(new_rows), keyed(old_rows)
    if set(new) != set(old):
        raise ValueError("Pipelines do not cover identical complete Tune groups")
    pairs = []
    for key in sorted(new):
        left, right = new[key], old[key]
        for name in ("query_key", "query_text", "full_relevant_count",
                     "available_relevant_count", "representative"):
            if left.get(name) != right.get(name):
                raise ValueError("Canonical query context or complete Recall denominator differs")
        pairs.append((left, right))
    return pairs


def paired_pipeline_bootstrap(pairs, new_hits, old_hits, denominator, *, replicates=2000, seed=20261013):
    by_text = defaultdict(list)
    for new, old in pairs:
        count = new[denominator]
        if count != old[denominator]:
            raise ValueError("Paired Recall denominators differ")
        if not count:
            continue
        if not 0 <= new[new_hits] <= count or not 0 <= old[old_hits] <= count:
            raise ValueError("Paired Recall numerator exceeds complete truth")
        by_text[EVAL.normalize_query_text(new["query_text"])].append((new[new_hits] - old[old_hits]) / count)
    differences = [sum(values) / len(values) for values in by_text.values()]
    if not differences or replicates < 1:
        raise ValueError("Bootstrap requires usable text clusters and positive replicates")
    rng = random.Random(seed)
    sampled = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences)
                     for _ in range(replicates))
    low, high = int(.025 * (replicates - 1)), int(.975 * (replicates - 1))
    return {"comparison": "new pipeline minus original HGB150 pipeline",
            "denominator": denominator, "bootstrap_unit": "normalized query text",
            "query_text_clusters": len(differences), "replicates": replicates, "seed": seed,
            "delta": sum(differences) / len(differences),
            "percentile_95_interval": [sampled[low], sampled[high]],
            "interpretation": "Exploratory Tune interval after hypothesis selection, not independent confirmation"}


def named_summary(rows, stage):
    summary = EVAL.summarize(rows, stage)
    for scope in ("conditional_available", "full_truth"):
        pool = summary[scope].pop("pool")
        summary[scope]["candidate_pool"] = {
            name.replace("_recall_at_50", "_candidate_recall"): value for name, value in pool.items()}
    for values in summary["slices"].values():
        if "pool" in values:
            pool = values.pop("pool")
            values["candidate_pool"] = {
                name.replace("_recall_at_50", "_candidate_recall"): value for name, value in pool.items()}
    return summary


def checked_model_and_cache(model, training_data, cache_dir, protocol_dir, manifest, protocol_hash,
                            *, research, stages, fit_texts):
    import joblib
    model_hash = EVAL.sha256_file(model)
    artifact = joblib.load(model)
    if (artifact.get("estimator", "histgb") != "histgb"
            or any(not isinstance(stage, int) or not 1 <= stage <= EVAL.maximum_iterations(artifact["model"])
                   for stage in stages)):
        raise ValueError("Explicit comparison stages must be available HGB iterations")
    if artifact.get("selected_iteration") is not None:
        raise ValueError("Use complete fitted HGB artifacts for explicit exploratory prefix comparisons")
    EVAL.check_encoder_identity(artifact, manifest)
    train_hash = EVAL.check_training_data(training_data, artifact, protocol_dir, manifest)
    path, metadata_path = EVAL.phase_cache_paths(cache_dir, "tune")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    EVAL.ensure_compatible(artifact["metadata"], metadata, "research V7" if research else "original HGB150")
    EVAL.ensure_prior_policy(artifact, metadata, "research V7" if research else "original HGB150")
    if research:
        validate_research_artifact(artifact, fit_texts, manifest)
        prior_hash = check_training_prior_file(training_data, artifact)
        if metadata.get("research_training_priors_file_sha256") != prior_hash:
            raise ValueError("Research cache training-prior file differs")
        check_research_cache(metadata, artifact, train_hash)
        base = metadata.get("research_encoder_base", {})
        reference = base.get("reference", {})
        if (not reference.get("path")
                or EVAL.sha256_file(Path(reference["path"])) != reference.get("sha256")
                or check_encoder_base(artifact["metadata"]["retrieval_config"]["embedding_model"],
                                      manifest, Path(reference["path"])) != base):
            raise ValueError("Research cache frozen adapter base differs")
    else:
        if artifact["metadata"].get("feature_version") != 4:
            raise ValueError("Original reference must be the frozen V4 HGB150 model")
    cache_hash = EVAL.check_phase_cache_inputs(path, metadata, "tune", protocol_hash, model_hash)
    identity = {"model": {"path": str(Path(model).resolve()), "sha256": model_hash},
                "training_data": {"path": str(Path(training_data).resolve()), "sha256": train_hash},
                "cache": {"path": str(path.resolve()), "sha256": cache_hash},
                "metadata": {"path": str(metadata_path.resolve()), "sha256": EVAL.sha256_file(metadata_path)}}
    return artifact, metadata, path, identity


def check_identity(identity):
    if any(EVAL.sha256_file(Path(value["path"])) != value["sha256"] for value in identity.values()):
        raise ValueError("Comparison artifact/cache inputs changed during evaluation")


def evaluate(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "research-tune-report.json"
    score_path = args.output_dir / "research-tune-group-scores.jsonl"
    if output.exists() or score_path.exists():
        raise FileExistsError("Exploratory report paths are immutable; choose a new output directory")
    manifest, protocol_hash = EVAL.checked_protocol(args.protocol_dir)
    fit_path = EVAL.protocol_file(args.protocol_dir, manifest, "fit_texts")
    fit_texts = set(json.loads(fit_path.read_text(encoding="utf-8"))["texts"])
    check_frozen_reference(args.old_model, args.old_train_data)
    new = checked_model_and_cache(args.new_model, args.new_train_data, args.new_cache_dir,
        args.protocol_dir, manifest, protocol_hash, research=True, stages=args.new_stages,
        fit_texts=fit_texts)
    old = checked_model_and_cache(args.old_model, args.old_train_data, args.old_cache_dir,
        args.protocol_dir, manifest, protocol_hash, research=False, stages=[args.old_stage],
        fit_texts=fit_texts)
    if new[1]["corpus_fingerprint"] != old[1]["corpus_fingerprint"]:
        raise ValueError("Pipeline comparison catalogs differ")
    check_identity(new[3]); check_identity(old[3])
    # The only target partition parsed by this evaluator is Tune.
    labels = EVAL.read_phase_labels(args.protocol_dir, manifest, "tune")
    new_groups = EVAL.bind_cache_groups(new[1], labels, "tune", protocol_hash, new[3]["model"]["sha256"])
    old_groups = EVAL.bind_cache_groups(old[1], labels, "tune", protocol_hash, old[3]["model"]["sha256"])
    from cache_research_reranker import check_cold_groups
    check_cold_groups(list(labels.values()), fit_texts)
    new_rows = EVAL.score_cache(new[2], new[1], new_groups, new[0], tuple(args.new_stages), args.batch_size)
    old_rows = EVAL.score_cache(old[2], old[1], old_groups, old[0], (args.old_stage,), args.batch_size)
    pairs = aligned_rows(new_rows, old_rows)
    curve = {str(stage): named_summary(new_rows, stage) for stage in args.new_stages}
    comparisons = {}
    for stage in args.new_stages:
        comparisons[str(stage)] = {
            "conditional_available_recall_at_50": paired_pipeline_bootstrap(pairs, f"s{stage}_hits",
                f"s{args.old_stage}_hits", "available_relevant_count", replicates=args.bootstrap_replicates),
            "full_truth_recall_at_50": paired_pipeline_bootstrap(pairs, f"s{stage}_hits",
                f"s{args.old_stage}_hits", "full_relevant_count", replicates=args.bootstrap_replicates)}
    pool = {scope: paired_pipeline_bootstrap(pairs, "pool_hits", "pool_hits", denominator,
        replicates=args.bootstrap_replicates) for scope, denominator in (
            ("conditional_available_candidate_recall", "available_relevant_count"),
            ("full_truth_candidate_recall", "full_relevant_count"))}
    for value in pool.values():
        value["comparison"] = "new candidate pool minus original candidate pool"
    losses = (EVAL.sampled_train_losses(args.new_train_data, new[0], tuple(args.new_stages), args.batch_size)
              if args.with_training_loss else None)
    check_identity(new[3]); check_identity(old[3]); check_sources(new[1]["research_source_sha256"])
    if EVAL.sha256_file(args.protocol_dir / "manifest.json") != protocol_hash:
        raise ValueError("Comparison protocol changed during scoring")
    with score_path.open("x", encoding="utf-8") as stream:
        for left, right in pairs:
            stream.write(json.dumps({"group_identity": left["group_identity"],
                "query_text": left["query_text"], "new": left, "old": right},
                ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {"report_version": 1, "phase": "tune", "exploratory": True,
        "audit_labels_read": False, "selection_performed": False,
        "scope": "Previously consumed Tune cold-text cohort; different saved pools on one fixed catalog",
        "protocol_sha256": protocol_hash, "new_pipeline_identity": new[3], "old_pipeline_identity": old[3],
        "primary_new_stage": args.primary_stage, "old_stage": args.old_stage,
        "new_capacity_curve": curve, "original_pipeline": named_summary(old_rows, args.old_stage),
        "paired_reranker_comparisons": comparisons, "paired_candidate_pool_comparison": pool,
        "new_training_row_loss": losses, "group_scores_sha256": EVAL.sha256_file(score_path),
        "maximum_candidate_budget": 998,
        "interpretation": ["A pipeline comparison changes both candidates and learned features; it is not a same-pool feature ablation.",
            "Tune confidence intervals after trying hypotheses do not provide independent confirmation.",
            "A declining Tune capacity curve while training loss decreases suggests possible overfitting, not a proof.",
            "Binary training loss is not percentage error or Recall."]}
    EVAL.write_new_json(output, report)
    print(json.dumps({"complete": True, "report": str(output), "primary_stage": args.primary_stage}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["tune"], default="tune")
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--new-model", type=Path, required=True)
    parser.add_argument("--new-train-data", type=Path, required=True)
    parser.add_argument("--new-cache-dir", type=Path, required=True)
    parser.add_argument("--old-model", type=Path, required=True)
    parser.add_argument("--old-train-data", type=Path, required=True)
    parser.add_argument("--old-cache-dir", type=Path, required=True)
    parser.add_argument("--old-stage", type=int, default=150)
    parser.add_argument("--new-stages", type=int, nargs="+", default=[75, 100, 150])
    parser.add_argument("--primary-stage", type=int, default=150)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--with-training-loss", action="store_true")
    args = parser.parse_args(argv)
    if (args.batch_size <= 0 or args.bootstrap_replicates <= 0 or args.old_stage != 150
            or not args.new_stages or len(set(args.new_stages)) != len(args.new_stages)
            or any(stage < 1 for stage in args.new_stages) or args.primary_stage not in args.new_stages):
        parser.error("Use positive batches/bootstrap, original HGB150, unique positive new stages containing primary stage")
    evaluate(args)


if __name__ == "__main__":
    main()
