"""Publish a Tune-sealed V8 candidate after the completed frozen regression.

This records a runtime prefix without removing fitted trees or changing priors.
It never predicts, reads Audit target IDs, chooses capacity from Audit, changes
earlier artifacts or declares a new official-benchmark winner.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re

STATUS = "selected-on-research-tune; frozen-regression-complete"
SCOPE = ("Candidate for the protocol's fresh sampled catalog and cold-query cohort. "
         "Audit was previously observed; frozen regression is not an independent test. "
         "The original official benchmark has not validated this artifact.")


def publication_contract():
    """Reuse the established seal, cache, summary and atomic-publication guards."""
    import evaluate_fresh_validation as evaluation
    from freeze_research_selection import verify_frozen_selection, REGRESSION_POLICY
    from cache_frozen_research_audit import check_regression_cache
    from cache_research_reranker import research_prior_policy, check_training_prior_file
    from evaluate_research_reranker import named_summary, check_frozen_reference
    from produce_document_hypotheses import check_encoder_base
    from publish_fresh_reranker import publish_new
    return dict(evaluation=evaluation, verify_selection=verify_frozen_selection,
        policy=REGRESSION_POLICY, check_regression_cache=check_regression_cache,
        named_summary=named_summary, research_prior_policy=research_prior_policy,
        check_training_priors=check_training_prior_file, check_reference=check_frozen_reference,
        check_base=check_encoder_base, publish_new=publish_new)


def _integer(value, *, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _context_inventory(metadata, expected_count, evaluation):
    groups = metadata.get("groups")
    if not isinstance(groups, list) or len(groups) != expected_count:
        raise ValueError("Completed cache must include every protocol Audit context")
    contexts = {}
    for index, group in enumerate(groups):
        key = group.get("query_key") if isinstance(group, dict) else None
        if (not isinstance(key, list) or len(key) != 5 or any(not isinstance(value, str) for value in key)
                or not _integer(group.get("group_id")) or group["group_id"] != index):
            raise ValueError("Cache context keys or contiguous local group IDs are invalid")
        key_json = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
        identity = hashlib.sha256(key_json.encode("utf-8")).hexdigest()
        if (group.get("group_identity") != identity or identity in contexts
                or group.get("query_text") != evaluation.normalize_query_text(key[0])
                or not isinstance(group.get("representative"), bool)
                or not _integer(group.get("full_relevant_count"), minimum=1)
                or not _integer(group.get("available_relevant_count"))
                or group["available_relevant_count"] > group["full_relevant_count"]
                or not _integer(group.get("pool_size")) or group["pool_size"] > 998):
            raise ValueError("Repeated or invalid canonical cache context/denominator")
        contexts[identity] = group
    return contexts


def _checked_rows(records, metadata, contexts, stage):
    rows, seen = [], set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Regression group scores require paired context records")
        identity = record.get("group_identity")
        row = record.get("row")
        if identity in seen or identity not in contexts or not isinstance(row, dict):
            raise ValueError("Regression group scores have duplicate, unknown or missing contexts")
        seen.add(identity)
        group = contexts[identity]
        if row.get("group_identity") != identity:
            raise ValueError("Paired regression wrapper and scored context identity differ")
        # Stored scoring rows retain the cache's local identity/denominators.
        # Nothing is inferred from target item IDs or from model predictions.
        for name in ("group_id", "query_key", "query_text", "representative", "full_relevant_count",
                     "available_relevant_count", "pool_size", "local_relevant_count",
                     "nonlocal_relevant_count", "seen_relevant_count", "unseen_relevant_count",
                     "has_filters", "no_exact_location", "one_word"):
            if name in group and row.get(name) != group[name]:
                raise ValueError(f"Regression row differs from the completed cache: {name}")
        stages = {int(match.group(1)) for key in row if (match := re.match(r"s([0-9]+)_", key))}
        if stages != {stage}:
            raise ValueError("Regression scores contain an unselected or missing prefix")
        available, size = group["available_relevant_count"], group["pool_size"]
        for name, maximum in (("pool_hits", min(available, size)),
                              ("baseline_hits", min(available, size, 50)),
                              (f"s{stage}_hits", min(available, size, 50))):
            if not _integer(row.get(name)) or row[name] > maximum:
                raise ValueError("Regression Recall numerator exceeds its recorded denominator/pool")
        if (row["pool_hits"] != group.get("positives_in_pool")
                or row["baseline_hits"] != group.get("baseline_hits")
                or row[f"s{stage}_hits"] > row["pool_hits"]):
            raise ValueError("Regression pool/baseline counts differ from completed cache metadata")
        rows.append(row)
    if seen != set(contexts):
        raise ValueError("Regression scores omit a protocol context, including zero-availability groups")
    # The evaluator summarized score_cache's contiguous local-group order,
    # then wrote paired rows in canonical-identity order. Restore each cache's
    # original order so floating-point accumulation matches the report exactly.
    return sorted(rows, key=lambda row: row["group_id"])


def checked_regression(protocol_dir, regression_dir, manifest, selection, selection_file_sha, contract):
    evaluation = contract["evaluation"]
    report_path = Path(regression_dir).resolve() / "frozen-regression-report.json"
    scores_path = report_path.with_name("frozen-regression-group-scores.jsonl")
    if not report_path.is_file() or not scores_path.is_file():
        raise ValueError("Complete frozen-regression report and paired group-score file are required")
    report_sha, scores_sha = evaluation.sha256_file(report_path), evaluation.sha256_file(scores_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stage, files = selection["selected_stage"], selection["files"]
    if (report.get("report_version") != 1 or report.get("phase") != "audit"
            or any(report.get(key) != value for key, value in contract["policy"].items())
            or report.get("selected_stage") != stage
            or report.get("selection_sha256") != selection["selection_sha256"]
            or report.get("selection_file_sha256") != selection_file_sha
            or report.get("frozen_selection") != {"path": selection["_path"], "sha256": selection_file_sha}
            or report.get("protocol_sha256") != files["protocol_manifest"]["sha256"]
            or report.get("dense_index") != selection["dense_index"]
            or report.get("old_audit_cache") != files["old_audit_cache"]
            or report.get("group_scores_sha256") != scores_sha):
        raise ValueError("Frozen regression identity, flags, prefix, index or scored bytes differ from the seal")
    new_entry = report.get("new_audit_metadata", {})
    if not isinstance(new_entry, dict) or not isinstance(new_entry.get("path"), str):
        raise ValueError("Completed regression lacks the exact new Audit cache metadata identity")
    new_cache, new_meta_path = evaluation.phase_cache_paths(Path(new_entry["path"]).parent, "audit")
    if str(new_meta_path.resolve()) != new_entry["path"] or evaluation.sha256_file(new_meta_path) != new_entry.get("sha256"):
        raise ValueError("New completed Audit cache metadata bytes/path differ from the report")
    new_meta = json.loads(new_meta_path.read_text(encoding="utf-8"))
    cache_sha = contract["check_regression_cache"](new_cache, new_meta, selection, selection_file_sha)
    if cache_sha != report.get("new_audit_cache_sha256"):
        raise ValueError("New completed Audit cache bytes differ from the regression report")
    old_cache, old_meta_path = (Path(files[key]["path"]) for key in ("old_audit_cache", "old_audit_metadata"))
    old_meta = json.loads(old_meta_path.read_text(encoding="utf-8"))
    evaluation.check_phase_cache_inputs(old_cache, old_meta, "audit", files["protocol_manifest"]["sha256"],
                                       files["old_model"]["sha256"])
    if (any(new_meta.get(key) != value for key, value in selection["research_prior_policy"].items())
            or new_meta.get("retrieval_config") != selection["retrieval_config"]):
        raise ValueError("Completed new cache retrieval settings or saved inference priors differ from the seal")
    stats = manifest.get("stats", {}).get("audit", {})
    count = stats.get("groups")
    if not _integer(count, minimum=1):
        raise ValueError("Protocol lacks a positive complete Audit context count")
    inventories = [_context_inventory(meta, count, evaluation) for meta in (new_meta, old_meta)]
    if set(inventories[0]) != set(inventories[1]):
        raise ValueError("New/old completed caches do not cover the same Audit context IDs")
    for identity, group in inventories[0].items():
        if any(group.get(name) != inventories[1][identity].get(name) for name in (
                "query_key", "query_text", "full_relevant_count", "available_relevant_count", "representative")):
            raise ValueError("New/old canonical contexts or complete denominators differ")
    pairs = []
    with scores_path.open(encoding="utf-8") as stream:
        for line in stream:
            pair = json.loads(line)
            if not isinstance(pair, dict):
                raise ValueError("Regression scores require paired context dictionaries")
            pairs.append(pair)
    rows = [_checked_rows([{"group_identity": pair.get("group_identity"), "row": pair.get(key)} for pair in pairs],
                          meta, inventory, capacity) for key, meta, inventory, capacity in (
                              ("new", new_meta, inventories[0], stage), ("old", old_meta, inventories[1], 150))]
    texts = {row["query_text"] for row in rows[0]}
    text_path = evaluation.protocol_file(protocol_dir, manifest, "audit-texts.json")
    text_payload = json.loads(text_path.read_text(encoding="utf-8"))
    expected_texts = text_payload.get("texts")
    if (not isinstance(expected_texts, list) or len(expected_texts) != len(set(expected_texts))
            or set(expected_texts) != texts or len(texts) != stats.get("texts")):
        raise ValueError("Completed regression text/context coverage differs from protocol text inventory")
    by_text = defaultdict(list)
    for row in rows[0]:
        by_text[row["query_text"]].append(row)
    salt = manifest.get("split", {}).get("representative_salt")
    if not isinstance(salt, str) or not salt:
        raise ValueError("Protocol lacks its label-blind representative selection salt")
    for variants in by_text.values():
        if sum(row["representative"] for row in variants) != 1:
            raise ValueError("Regression must preserve exactly one label-blind representative per text")
        expected = min(variants, key=lambda row: hashlib.sha256(json.dumps(
            [salt, json.dumps(row["query_key"], ensure_ascii=False, separators=(",", ":"))],
            ensure_ascii=False, separators=(",", ":")).encode("utf-8")).digest())
        if not expected["representative"]:
            raise ValueError("Regression representative differs from the label-blind protocol rule")
    for name, data, capacity in (("new_pipeline", rows[0], stage), ("old_hgb150_pipeline", rows[1], 150)):
        if report.get(name) != contract["named_summary"](data, capacity):
            raise ValueError("Recorded named_summary differs from the complete paired score rows")
    proof = {"report": {"path": str(report_path), "sha256": report_sha},
             "group_scores": {"path": str(scores_path), "sha256": scores_sha},
             "new_cache": {"path": str(new_cache.resolve()), "sha256": cache_sha},
             "new_cache_metadata": dict(new_entry), "old_cache": dict(files["old_audit_cache"]),
             "old_cache_metadata": dict(files["old_audit_metadata"]),
             "context_count": count, "query_text_count": len(texts),
             "context_ids_sha256": evaluation.canonical_sha256(sorted(inventories[0]))}
    return report, proof, new_meta


def publish(protocol_dir, selection, regression_dir, output, *, contract=None):
    contract = publication_contract() if contract is None else contract
    evaluation = contract["evaluation"]
    protocol_dir, selection_path, output = (Path(value).resolve() for value in (protocol_dir, selection, output))
    if output.exists():
        raise FileExistsError("Publication requires a new path; previous runtime models stay unchanged")
    sealed = contract["verify_selection"](selection_path)
    seal_sha = evaluation.sha256_file(selection_path)
    selection = {**sealed, "_path": str(selection_path)}
    stage, files = selection["selected_stage"], selection["files"]
    if isinstance(stage, bool) or stage not in (75, 100, 150):
        raise ValueError("Only the Tune-sealed available prefix may be published")
    model = Path(files["new_model"]["path"])
    if output == model.resolve() or output == Path(files["old_model"]["path"]).resolve():
        raise FileExistsError("Raw and previous best model paths cannot be publication targets")
    manifest, protocol_sha = evaluation.checked_protocol(protocol_dir)
    if (protocol_sha != files["protocol_manifest"]["sha256"]
            or str((protocol_dir / "manifest.json").resolve()) != files["protocol_manifest"]["path"]):
        raise ValueError("Publication protocol path/content differs from the Tune seal")
    report, proof, new_meta = checked_regression(protocol_dir, regression_dir, manifest, selection, seal_sha, contract)
    # Load a trusted fitted artifact only after every report/group/cache proof is
    # complete. This path never loads Audit targets or invokes model prediction.
    import joblib
    artifact = joblib.load(model)
    if (not isinstance(artifact, Mapping) or artifact.get("estimator", "histgb") != "histgb"
            or artifact.get("selected_iteration") is not None or artifact.get("metadata", {}).get("feature_version") != 8
            or evaluation.maximum_iterations(artifact["model"]) != 150):
        raise ValueError("Publication requires the original unselected V8 HGB150 fitted artifact")
    if (artifact["metadata"].get("retrieval_config") != selection["retrieval_config"]
            or contract["research_prior_policy"](artifact) != selection["research_prior_policy"]):
        raise ValueError("New fitted model retrieval configuration or priors differ from the Tune seal")
    evaluation.ensure_compatible(artifact["metadata"], new_meta, "published research V8")
    evaluation.ensure_prior_policy(artifact, new_meta, "published research V8")
    encoder_sha = evaluation.check_encoder_identity(artifact, manifest)
    train_sha = evaluation.check_training_data(files["new_train_data"]["path"], artifact, protocol_dir, manifest)
    if train_sha != files["new_train_data"]["sha256"]:
        raise ValueError("New fitted model training data differs from the sealed identity")
    contract["check_training_priors"](files["new_train_data"]["path"], artifact)
    contract["check_reference"](files["old_model"]["path"], files["old_train_data"]["path"])
    if contract["check_base"](selection["retrieval_config"]["embedding_model"], manifest,
            Path(files["encoder_base_reference"]["path"])) != selection["encoder_base_reference"]:
        raise ValueError("Published model encoder base differs from the frozen identity")
    # Reverify the full seal, including producer sources and material byte hashes.
    if contract["verify_selection"](selection_path) != sealed or evaluation.sha256_file(selection_path) != seal_sha:
        raise ValueError("Tune seal changed during publication proof checking")
    for entry in (proof["report"], proof["group_scores"], proof["new_cache"], proof["new_cache_metadata"]):
        if evaluation.sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Completed regression inputs changed during publication checks")
    selected = dict(artifact)
    selected.update(selected_iteration=stage, selection_status=STATUS)
    if isinstance(artifact.get("training_configuration"), Mapping):
        selected["training_configuration"] = {**artifact["training_configuration"],
            "selected_iteration": stage, "selection_status": STATUS}
    selected["research_selection"] = {
        "version": 1, "scope": SCOPE, "publication_role": "candidate-only",
        "official_benchmark_validated": False, "audit_independent": False,
        "automatic_deployment_selection": False, "capacity_chosen_from": "sealed research Tune only",
        "audit_used_for_capacity_selection": False, "requires_selected_iteration_inference": True,
        "selected_iteration": stage, "available_iterations": 150,
        "source_model": dict(files["new_model"]), "protocol_manifest": dict(files["protocol_manifest"]),
        "training_data": dict(files["new_train_data"]), "encoder_content_sha256": encoder_sha,
        "frozen_selection": {"path": str(selection_path), "sha256": seal_sha},
        "selection_sha256": selection["selection_sha256"],
        "tune_report": dict(files["tune_report"]), "tune_group_scores": dict(files["tune_group_scores"]),
        "frozen_regression": proof, "regression_policy": dict(contract["policy"]),
        "new_pipeline_summary": report["new_pipeline"], "old_hgb150_summary": report["old_hgb150_pipeline"],
        "corpus": manifest.get("corpus", {}), "producer_sha256": evaluation.sha256_file(Path(__file__)),
    }
    contract["publish_new"](output, selected)
    return {"output": str(output), "sha256": evaluation.sha256_file(output),
            "selected_iteration": stage, "selection_status": STATUS, "publication_role": "candidate-only",
            "official_benchmark_validated": False, "audit_independent": False, "scope": SCOPE}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--regression-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(publish(**vars(args)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
