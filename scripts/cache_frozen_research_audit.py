"""Cache one checksum-frozen new pipeline on the previously observed Audit.

This is regression checking, not a fresh test or another selection arm. The
completed Tune report/model/train/source closure is verified before any Audit
target IDs are parsed. The saved classifier and histories are never refitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from avito_candidates.core import SEARCH_FIELDS, normalize
from avito_candidates.reranker import Reranker, corpus_fingerprint
from cache_reranker_audit import ITEM_COLUMNS, build_cache, digest, query_order
from cache_research_reranker import (validate_research_artifact, research_prior_policy,
    check_training_prior_file, check_cold_groups, build_saved_retriever)
from freeze_research_selection import verify_frozen_selection, verify_dense_index, REGRESSION_POLICY
from produce_document_hypotheses import (iter_catalog_records, check_encoder_base,
                                         require_offline_environment)
import evaluate_fresh_validation as EVAL


def check_regression_cache(path, metadata, selection, selection_file_sha):
    """Strict observed-Audit integrity guard; no fresh-test flag substitution."""
    import pyarrow.parquet as pq
    expected = {"cache_version": 2, "role": "validation", "fresh_partition": "audit",
        "fresh_validation_protocol_sha256": selection["files"]["protocol_manifest"]["sha256"],
        "model_sha256": selection["files"]["new_model"]["sha256"],
        "historical_holdout_reused": True, "full_truth_preserved": True,
        "frozen_regression_cache_version": 1,
        "previous_baseline_audit_observed": True, "independent_fresh_test": False,
        "purpose": "frozen regression", "frozen_selected_stage": selection["selected_stage"],
        "regression_policy": REGRESSION_POLICY,
        "regression_selection_sha256": selection["selection_sha256"],
        "regression_selection_file_sha256": selection_file_sha,
        "regression_dense_index": selection["dense_index"],
        "producer_sha256": selection["source_sha256"].get(str(Path(__file__).resolve()))}
    if not expected["producer_sha256"]:
        raise ValueError("Frozen regression closure lacks its cache producer")
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Observed-Audit regression cache or its frozen selection differs")
    features, groups = metadata.get("feature_names"), metadata.get("groups")
    if (not isinstance(features, list) or not features or any(not isinstance(name, str) for name in features)
            or len(set(features)) != len(features)
            or not {"rrf_score", "baseline_top50"} <= set(features)
            or not isinstance(groups, list) or not groups):
        raise ValueError("Regression cache lacks its complete feature schema/groups")
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError("Regression cache group metadata is invalid")
        size = group.get("pool_size")
        if (isinstance(group.get("group_id"), bool) or not isinstance(group.get("group_id"), int)
                or group.get("group_id") != index
                or isinstance(size, bool) or not isinstance(size, int)
                or not 0 <= size <= 998):
            raise ValueError("Regression cache group order or candidate budget differs")
    rows = metadata.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows != sum(group["pool_size"] for group in groups):
        raise ValueError("Regression cache row total differs from its complete pools")
    parquet_hash = digest(path)
    if parquet_hash != metadata.get("cache_parquet_sha256"):
        raise ValueError("Regression cache Parquet bytes changed")
    parquet = pq.ParquetFile(path)
    required = set(features) | {"group_id", "item_id", "label", "positive_local", "positive_seen"}
    if (parquet.metadata.num_rows != rows or not required <= set(parquet.schema_arrow.names)
            or any(str(parquet.schema_arrow.field(name).type) != "float" for name in features)):
        raise ValueError("Regression Parquet rows or feature schema differ")
    return parquet_hash


def check_loaded_dense_index(retriever, items, selection, *, fingerprint=None):
    """Confirm the restored runtime model uses exactly the pre-target sealed pair."""
    if fingerprint is None:
        from avito_candidates.dense import _fingerprint
        fingerprint = _fingerprint
    dense = retriever.dense
    actual = fingerprint(items, dense.model_name, dense.max_seq_length, dense.params_words,
                         dense.description_words, dense.model_signature,
                         document_format=dense.document_format)
    if actual != selection["dense_index"]["fingerprint"]:
        raise ValueError("Restored encoder/index fingerprint differs from the pre-Audit selected pair")


def produce(args):
    require_offline_environment()
    selection = verify_frozen_selection(args.selection)
    verify_dense_index(selection, args.dense_cache)
    selection_sha = digest(args.selection)
    files = selection["files"]
    manifest, protocol_hash = EVAL.checked_protocol(args.protocol_dir)
    if protocol_hash != files["protocol_manifest"]["sha256"]:
        raise ValueError("Regression protocol differs from the pre-Audit selection")
    import joblib
    import pyarrow.parquet as pq
    model_path, training_path = Path(files["new_model"]["path"]), Path(files["new_train_data"]["path"])
    artifact = joblib.load(model_path)
    fit_path = EVAL.protocol_file(args.protocol_dir, manifest, "fit_texts")
    fit_texts = set(json.loads(fit_path.read_text(encoding="utf-8"))["texts"])
    validate_research_artifact(artifact, fit_texts, manifest)
    EVAL.check_encoder_identity(artifact, manifest)
    EVAL.check_training_data(training_path, artifact, args.protocol_dir, manifest)
    check_training_prior_file(training_path, artifact)
    metadata, policy = artifact["metadata"], research_prior_policy(artifact)
    config = metadata["retrieval_config"]
    if config != selection["retrieval_config"] or policy != selection["research_prior_policy"]:
        raise ValueError("New retrieval or inference priors changed after freezing")
    reference_path = Path(files["encoder_base_reference"]["path"])
    if check_encoder_base(config["embedding_model"], manifest, reference_path) != selection["encoder_base_reference"]:
        raise ValueError("Cached encoder base changed after the Tune-only freeze")
    item_path = EVAL.protocol_file(args.protocol_dir, manifest, "benchmark_items")
    items = [{field: row.get(field, "") for field in ITEM_COLUMNS} for row in iter_catalog_records(item_path)]
    if corpus_fingerprint(items, feature_version=metadata["feature_version"]) != metadata["corpus_fingerprint"]:
        raise ValueError("Frozen regression inventory differs before opening targets")
    del artifact
    reranker = Reranker(model_path, items, config, evaluation_split=("queries", .2, 42),
                        selected_iteration=selection["selected_stage"])
    priors = {"microcategory_classifier": reranker.builder.microcategory_classifier,
              "microcategory_prior": reranker.builder.microcategory_prior,
              "geo_associations": reranker.builder.geo_associations}
    if research_prior_policy({"priors": priors}) != policy:
        raise ValueError("Restored frozen model has different inference priors")
    retriever = build_saved_retriever(config, items, priors, dense_cache=args.dense_cache,
                                     batch_size=args.batch_size, device=args.device)
    check_loaded_dense_index(retriever, items, selection)
    # The final closure check is immediately before parsing Audit target IDs.
    verify_frozen_selection(args.selection)
    EVAL.checked_protocol(args.protocol_dir)
    if digest(args.selection) != selection_sha:
        raise ValueError("Frozen selection file changed before opening Audit targets")
    groups = list(EVAL.read_phase_labels(args.protocol_dir, manifest, "audit").values())
    check_cold_groups(groups, fit_texts)
    items_by_id = {item["item_id"]: item for item in items}
    ids = set(items_by_id)
    fit_items = set()
    train_source = EVAL.protocol_file(args.protocol_dir, manifest, "train")
    for batch in pq.ParquetFile(train_source).iter_batches(batch_size=4096, columns=["search_query", "item_id"]):
        for row in batch.to_pylist():
            if normalize(row["search_query"] or "") in fit_texts:
                fit_items.add(str(row["item_id"]))
    truth, representatives, additions = {}, {}, {}
    for group in groups:
        key = tuple(group["query_key"])
        full, available = set(group["relevant_item_ids"]), set(group["available_item_ids"])
        if available != full & ids:
            raise ValueError("Frozen Audit target availability differs from the selected inventory")
        truth[key], representatives[key] = available, dict(zip(SEARCH_FIELDS, key))
        additions[key] = {"group_identity": group["group_id"], "full_relevant_count": len(full),
                          "available_relevant_count": len(available), "representative": group["representative"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extra = {"fresh_validation_protocol_sha256": protocol_hash, "fresh_partition": "audit", **policy,
        "producer_sha256": digest(Path(__file__)), "full_truth_preserved": True,
        "frozen_regression_cache_version": 1, "previous_baseline_audit_observed": True,
        "independent_fresh_test": False, "purpose": "frozen regression",
        "regression_policy": REGRESSION_POLICY, "frozen_selected_stage": selection["selected_stage"],
        "regression_selection_sha256": selection["selection_sha256"],
        "regression_selection_file_sha256": selection_sha,
        "regression_dense_index": selection["dense_index"]}
    result = build_cache(args.output_dir / "audit.parquet", truth=truth, representatives=representatives,
        keys=sorted(truth, key=query_order), builder=reranker.builder, retriever=retriever,
        items_by_id=items_by_id, fit_items=fit_items, benchmark_strata=[], model_metadata=metadata,
        model_reference=model_path, model_hash=files["new_model"]["sha256"], batch_size=args.batch_size,
        role="validation", max_pool_size=998, partitioner=lambda _text: "audit",
        historical_holdout_reused=True, extra_group_metadata=additions, extra_metadata=extra,
        allow_empty_relevance=True)
    verify_frozen_selection(args.selection)
    EVAL.checked_protocol(args.protocol_dir)
    if digest(args.selection) != selection_sha:
        raise ValueError("Frozen selection changed during regression retrieval")
    check_regression_cache(args.output_dir / "audit.parquet", result, selection, selection_sha)
    print(json.dumps({"complete": True, "purpose": "frozen regression", "independent_fresh_test": False,
                      "cache": str(args.output_dir / "audit.parquet")}), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dense-cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("Batch size must be positive")
    produce(args)


if __name__ == "__main__":
    main()
