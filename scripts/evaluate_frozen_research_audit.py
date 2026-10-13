"""One frozen new prefix versus original HGB150 on the observed Audit.

No capacity sweep or retuning is available. The source-controlled view filters
already-scored candidates to IDs occurring anywhere in original public train;
scores/features/priors and complete target denominators remain unchanged.
This differs from the earlier filter excluding original benchmark IDs, since
train/benchmark inventory overlap is retained here. This is a regression
diagnostic on previously observed targets, never an independent fresh test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import evaluate_fresh_validation as EVAL
from cache_frozen_research_audit import check_regression_cache
from cache_research_reranker import (validate_research_artifact, research_prior_policy,
                                    check_training_prior_file, check_cold_groups)
from evaluate_research_reranker import (aligned_rows, paired_pipeline_bootstrap,
    named_summary, check_frozen_reference)
from freeze_research_selection import verify_frozen_selection, REGRESSION_POLICY
from produce_document_hypotheses import iter_catalog_records, check_encoder_base


def public_train_control(manifest, protocol_dir, selection):
    """Read item metadata only, before opening Audit target IDs."""
    import pyarrow.parquet as pq
    entry = selection["files"]["public_train_inventory"]
    if (entry["sha256"] != manifest["sources"]["train"]["sha256"]
            or EVAL.sha256_file(Path(entry["path"])) != entry["sha256"]):
        raise ValueError("Public-train source-control inventory changed")
    train_ids = set()
    for batch in pq.ParquetFile(entry["path"]).iter_batches(batch_size=16384, columns=["item_id"]):
        train_ids.update(batch.column(0).to_pylist())
    if not train_ids or any(not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{16}", item) for item in train_ids):
        raise ValueError("Public-train source-control IDs are invalid")
    catalog = EVAL.protocol_file(protocol_dir, manifest, "benchmark_items")
    # Match the original feature reader's string conversion, including any
    # nullable integer category representation, without retaining descriptions.
    categories = {row["item_id"]: row.get("item_microcat_id", "") for row in iter_catalog_records(catalog)}
    return train_ids, set(categories) - train_ids, categories, {
        "source": entry, "train_unique_items": len(train_ids), "catalog_items": len(categories),
        "retained_catalog_items": len(set(categories) & train_ids),
        "predicate": "item_id occurs anywhere in original public train.parquet",
        "not_the_old_predicate": "Earlier source control excluded original benchmark IDs; overlapping train/benchmark IDs remain here",
        "features_and_scores_recomputed": False, "gt_denominator_restricted": False}


def assert_targets_in_public_train(labels, train_ids):
    for group in labels.values():
        if not set(group["relevant_item_ids"]) <= train_ids:
            raise ValueError("Frozen regression full GT includes an item outside the public-train source control")


def source_summary(rows, stage):
    sources = {"reranker": f"s{stage}_train_source_only_hits",
               "baseline_rrf": "train_source_only_baseline_hits", "candidate_pool": "train_source_only_pool_hits"}
    if any(row["train_source_only_pool_hits"] != row["pool_hits"] for row in rows):
        raise ValueError("Public-train source restriction removed a known positive")
    def metrics(denominator):
        result = {name: EVAL.aggregate(rows, hits, denominator) for name, hits in sources.items()}
        result["candidate_pool"] = {key.replace("_recall_at_50", "_candidate_recall"): value
                                    for key, value in result["candidate_pool"].items()}
        return result
    return {"conditional_available": metrics("available_relevant_count"),
            "full_truth": metrics("full_relevant_count"),
            "pool_statistics": {"average_size": sum(row["train_source_only_pool_size"] for row in rows) / len(rows),
                                "maximum_size": max(row["train_source_only_pool_size"] for row in rows)},
            "internal_score_field_prefix": "train_source_only",
            "ranking": "Re-select top50 from original model scores/RRF within retained candidates"}


def annotate_known_slices(rows, labels, categories, known_classes):
    for row in rows:
        available = labels[row["group_identity"]]["available_item_ids"]
        known = [categories[item] in known_classes for item in available]
        row["all_available_gt_microcategories_known"] = bool(known) and all(known)
        row["any_available_gt_microcategory_known"] = any(known)


def compare_rows(new_rows, old_rows, stage, bootstrap_replicates):
    pairs = aligned_rows(new_rows, old_rows)
    output = {}
    for scope, denominator in (("conditional_available", "available_relevant_count"),
                               ("full_truth", "full_relevant_count")):
        output[scope] = {
            "recall_at_50": paired_pipeline_bootstrap(pairs, f"s{stage}_hits", "s150_hits", denominator,
                                                       replicates=bootstrap_replicates),
            "candidate_pool_recall": paired_pipeline_bootstrap(pairs, "pool_hits", "pool_hits", denominator,
                                                                replicates=bootstrap_replicates),
            "public_train_only_recall_at_50": paired_pipeline_bootstrap(pairs,
                f"s{stage}_train_source_only_hits", "s150_train_source_only_hits", denominator,
                replicates=bootstrap_replicates)}
        for value in output[scope].values():
            value["interpretation"] = "Frozen regression on previously observed Audit; not an independent confidence claim"
    return output


def evaluate(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "frozen-regression-report.json"
    scores_path = args.output_dir / "frozen-regression-group-scores.jsonl"
    if output.exists() or scores_path.exists():
        raise FileExistsError("Frozen regression has already produced outputs; do not repeat or overwrite")
    selection = verify_frozen_selection(args.selection)
    selection_file_sha = EVAL.sha256_file(args.selection)
    files, stage = selection["files"], selection["selected_stage"]
    manifest, protocol_hash = EVAL.checked_protocol(args.protocol_dir)
    if protocol_hash != files["protocol_manifest"]["sha256"]:
        raise ValueError("Frozen Audit protocol differs from the Tune-only selection")
    import joblib
    new_artifact = joblib.load(files["new_model"]["path"])
    old_artifact = joblib.load(files["old_model"]["path"])
    check_frozen_reference(files["old_model"]["path"], files["old_train_data"]["path"])
    fit_path = EVAL.protocol_file(args.protocol_dir, manifest, "fit_texts")
    fit_texts = set(json.loads(fit_path.read_text(encoding="utf-8"))["texts"])
    validate_research_artifact(new_artifact, fit_texts, manifest)
    check_training_prior_file(files["new_train_data"]["path"], new_artifact)
    for prefix, artifact in (("new", new_artifact), ("old", old_artifact)):
        EVAL.check_encoder_identity(artifact, manifest)
        EVAL.check_training_data(files[f"{prefix}_train_data"]["path"], artifact, args.protocol_dir, manifest)
        if artifact.get("estimator", "histgb") != "histgb":
            raise ValueError("Frozen regression compares saved HGB prefixes only")
    if stage > EVAL.maximum_iterations(new_artifact["model"]) or EVAL.maximum_iterations(old_artifact["model"]) < 150:
        raise ValueError("Frozen comparison stage is unavailable")
    if (new_artifact["metadata"]["retrieval_config"] != selection["retrieval_config"]
            or research_prior_policy(new_artifact) != selection["research_prior_policy"]):
        raise ValueError("Frozen new model configuration/priors changed")
    base = check_encoder_base(new_artifact["metadata"]["retrieval_config"]["embedding_model"], manifest,
                              Path(files["encoder_base_reference"]["path"]))
    if base != selection["encoder_base_reference"]:
        raise ValueError("Frozen adapter base changed")
    new_cache, new_meta_path = EVAL.phase_cache_paths(args.cache_dir, "audit")
    new_meta = json.loads(new_meta_path.read_text(encoding="utf-8"))
    new_metadata_hash = EVAL.sha256_file(new_meta_path)
    new_cache_hash = check_regression_cache(new_cache, new_meta, selection, selection_file_sha)
    old_cache, old_meta_path = Path(files["old_audit_cache"]["path"]), Path(files["old_audit_metadata"]["path"])
    old_meta = json.loads(old_meta_path.read_text(encoding="utf-8"))
    EVAL.check_phase_cache_inputs(old_cache, old_meta, "audit", protocol_hash, files["old_model"]["sha256"])
    for artifact, metadata in ((new_artifact, new_meta), (old_artifact, old_meta)):
        EVAL.ensure_compatible(artifact["metadata"], metadata, "frozen regression")
        EVAL.ensure_prior_policy(artifact, metadata, "frozen regression")
    if any(new_meta.get(key) != value for key, value in selection["research_prior_policy"].items()):
        raise ValueError("Frozen Audit feature cache classifier/priors changed")
    # Inventory membership is computed globally before parsing query targets.
    train_ids, excluded_catalog_ids, categories, source_identity = public_train_control(manifest, args.protocol_dir, selection)
    verify_frozen_selection(args.selection)
    EVAL.checked_protocol(args.protocol_dir)
    if EVAL.sha256_file(args.selection) != selection_file_sha:
        raise ValueError("Frozen selection changed before opening Audit target IDs")
    labels = EVAL.read_phase_labels(args.protocol_dir, manifest, "audit")
    check_cold_groups(list(labels.values()), fit_texts)
    assert_targets_in_public_train(labels, train_ids)
    new_groups = EVAL.bind_cache_groups(new_meta, labels, "audit", protocol_hash, files["new_model"]["sha256"])
    old_groups = EVAL.bind_cache_groups(old_meta, labels, "audit", protocol_hash, files["old_model"]["sha256"])
    # The generic score helper filters candidates with IDs outside this supplied
    # set. Here it receives non-train IDs, not the original benchmark IDs.
    new_rows = EVAL.score_cache(new_cache, new_meta, new_groups, new_artifact, (stage,), args.batch_size,
                               original_benchmark_ids=excluded_catalog_ids)
    old_rows = EVAL.score_cache(old_cache, old_meta, old_groups, old_artifact, (150,), args.batch_size,
                               original_benchmark_ids=excluded_catalog_ids)
    known = set(new_artifact["metadata"]["microcategory_classifier_metadata"]["classes"])
    annotate_known_slices(new_rows, labels, categories, known)
    annotate_known_slices(old_rows, labels, categories, known)
    comparison = compare_rows(new_rows, old_rows, stage, args.bootstrap_replicates)
    known_slices = {}
    for key in ("all_available_gt_microcategories_known", "any_available_gt_microcategory_known"):
        for value in (False, True):
            left = [row for row in new_rows if row[key] == value]
            right = [row for row in old_rows if row[key] == value]
            if left and any(row["available_relevant_count"] > 0 for row in left):
                known_slices[f"{key}={str(value).lower()}"] = {
                    "new": named_summary(left, stage), "old": named_summary(right, 150),
                    "paired": compare_rows(left, right, stage, args.bootstrap_replicates)}
    verify_frozen_selection(args.selection)
    EVAL.checked_protocol(args.protocol_dir)
    if (EVAL.sha256_file(args.selection) != selection_file_sha or EVAL.sha256_file(new_cache) != new_cache_hash
            or EVAL.sha256_file(new_meta_path) != new_metadata_hash):
        raise ValueError("Frozen regression sources changed during scoring")
    if check_encoder_base(new_artifact["metadata"]["retrieval_config"]["embedding_model"], manifest,
                          Path(files["encoder_base_reference"]["path"])) != base:
        raise ValueError("Frozen base reference changed during regression scoring")
    with scores_path.open("x", encoding="utf-8") as stream:
        for left, right in aligned_rows(new_rows, old_rows):
            stream.write(json.dumps({"group_identity": left["group_identity"], "new": left, "old": right},
                ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {"report_version": 1, **REGRESSION_POLICY, "phase": "audit", "selected_stage": stage,
        "selection_sha256": selection["selection_sha256"], "selection_file_sha256": selection_file_sha,
        "frozen_selection": {"path": str(args.selection.resolve()), "sha256": selection_file_sha},
        "dense_index": selection["dense_index"],
        "protocol_sha256": protocol_hash, "new_audit_cache_sha256": new_cache_hash,
        "new_audit_metadata": {"path": str(new_meta_path.resolve()), "sha256": new_metadata_hash},
        "old_audit_cache": files["old_audit_cache"], "new_pipeline": named_summary(new_rows, stage),
        "old_hgb150_pipeline": named_summary(old_rows, 150), "paired_comparison": comparison,
        "public_train_source_control": {"identity": source_identity, "new": source_summary(new_rows, stage),
                                         "old": source_summary(old_rows, 150)},
        "microcategory_known_gt_slices": known_slices,
        "microcategory_slice_definition": "Known means GT item's metadata microcategory is in the fit-only global classifier class set, not that top3 predicts it",
        "group_scores_sha256": EVAL.sha256_file(scores_path),
        "limitations": ["Audit targets were previously observed for the old baseline; this is a frozen regression, not a new independent test.",
            "The new stage/pipeline was selected on Tune only; do not retune from this report.",
            "Public-train restriction reduces competition and keeps prior normalization/features frozen; it does not prove source-cue causality.",
            "Microcategory GT slices are posthoc diagnostics and cannot be used for hyperparameter selection."]}
    EVAL.write_new_json(output, report)
    print(json.dumps({"complete": True, "report": str(output), "purpose": "frozen regression",
                      "independent_fresh_test": False}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.bootstrap_replicates <= 0:
        parser.error("Batch and bootstrap counts must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
