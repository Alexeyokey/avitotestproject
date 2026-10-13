"""Cache a new V7/V8 reranker's complete Tune pools under explicit saved priors.

This experimental adapter does not change the old fresh cache/evaluator. It
accepts only the frozen seed42 fit protocol, grouped-OOF microcategory models
and the unchanged encoder. Retrieval and features share the classifier saved
inside the reranker; no history or classifier is fitted at inference. Audit
is deliberately unavailable after retrieval hypotheses have consumed Tune.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from avito_candidates.core import SEARCH_FIELDS, held_out, normalize
from avito_candidates.reranker import Reranker, FEATURE_SCHEMAS, corpus_fingerprint
from avito_candidates.fitted_fingerprint import (fitted_estimator_fingerprint,
    classifier_semantic_fingerprint, FITTED_FINGERPRINT_VERSION)
from cache_reranker_audit import ITEM_COLUMNS, build_cache, digest, query_order
from produce_document_hypotheses import (iter_catalog_records, check_encoder_base,
                                         require_offline_environment)
from reranker_audit_provenance import artifact_prior_policy
import evaluate_fresh_validation as EVAL

RESEARCH_POLICY = {"microcategory": "leave-query-text-out",
                   "geography": "leave-query-text-out",
                   "microcategory_classifier": "grouped-out-of-fold"}


def research_prior_policy(artifact):
    """Extend, rather than bypass, aggregate priors with stable fitted semantics."""
    classifier = artifact.get("priors", {}).get("microcategory_classifier")
    if classifier is None:
        raise ValueError("V7 inference requires its saved microcategory classifier")
    classifier_hash = classifier_semantic_fingerprint(classifier)
    legacy = artifact_prior_policy(artifact)
    return {**legacy, "research_classifier_fingerprint_version": FITTED_FINGERPRINT_VERSION,
            "research_classifier_semantic_sha256": classifier_hash,
            "research_priors_sha256": EVAL.canonical_sha256({**legacy,
                "classifier_fingerprint_version": FITTED_FINGERPRINT_VERSION,
                "classifier_semantic_sha256": classifier_hash})}


def validate_research_artifact(artifact, fit_texts, manifest):
    from avito_candidates.microcategory import _text_digest
    metadata = artifact.get("metadata", {})
    version = metadata.get("feature_version")
    if (version not in (7, 8)
            or tuple(metadata.get("feature_names", ())) != FEATURE_SCHEMAS.get(version)
            or metadata.get("split") != "queries" or metadata.get("seed") != 42
            or metadata.get("validation_fraction") != .2
            or metadata.get("training_prior_policy") != RESEARCH_POLICY):
        raise ValueError("Research reranker must be V7/V8 with seed42 self-excluding and grouped-OOF priors")
    if artifact.get("disable_microcategory_prior", False):
        raise ValueError("This V7 research protocol retains its original token-prior features")
    config = metadata.get("retrieval_config", {})
    if version == 8 and config.get("rescore_dense_union") is not True:
        raise ValueError("V8 requires measured full-pool dense cosine rescoring")
    if (config.get("method") != "hybrid" or config.get("microcategory_candidate_k", 0) <= 0
            or config.get("microcategory_selection_policy") != "top998-then-novel-prefix"):
        raise ValueError("Research V7 requires an explicit budgeted microcategory retrieval configuration")
    check_supplemental_config(config)
    classifier = artifact.get("priors", {}).get("microcategory_classifier")
    if classifier is None or classifier.metadata != metadata.get("microcategory_classifier_metadata"):
        raise ValueError("Saved V7 classifier or its metadata differs from fitting")
    expected = {"source_sha256": manifest["files"]["train"]["sha256"],
                "corpus_fingerprint": metadata.get("corpus_fingerprint"),
                "allowed_fit_texts_sha256": _text_digest(fit_texts), "seed": 42}
    if any(classifier.metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved classifier crosses the allowed fit partition or inventory")
    fitted = set(classifier.model.fitted_texts)
    if not fitted or not fitted <= fit_texts:
        raise ValueError("Saved classifier fitted query texts are not fit-only")
    if _text_digest(fitted) != classifier.metadata.get("observed_fit_texts_sha256"):
        raise ValueError("Saved classifier observed training text inventory changed")
    folds = classifier.metadata.get("crossfit_folds")
    if (isinstance(folds, bool) or not isinstance(folds, int) or folds < 2
            or set(classifier.crossfit_models) != set(range(folds))):
        raise ValueError("V7 classifier must preserve every grouped-OOF fold")
    for fold, model in classifier.crossfit_models.items():
        texts = set(model.fitted_texts)
        if (not texts <= fitted or any(classifier.fold(text, folds, 42) == fold for text in texts)):
            raise ValueError("Classifier OOF model contains its own held-out query-text fold")
    micro_path = Path(config.get("microcategory_model", ""))
    if not micro_path.is_file() or digest(micro_path) != config.get("microcategory_model_sha256"):
        raise ValueError("Saved classifier file provenance changed or is missing")
    envelope = micro_path.with_suffix(micro_path.suffix + ".json")
    if (not envelope.is_file() or json.loads(envelope.read_text(encoding="utf-8")).get("metadata")
            != classifier.metadata):
        raise ValueError("Saved classifier manifest differs from the embedded inference classifier")
    return classifier


def check_supplemental_config(config):
    mode = config.get("supplemental_fill", "off")
    if mode == "off":
        return mode
    if (mode != "micro-typed-dense-v1" or config.get("supplemental_raw_depth") != 998
            or config.get("supplemental_rank_constant") != 60
            or config.get("supplemental_selection_policy") != "raw-ranks-equal-route-novel-rrf-id-ties"
            or config.get("microcategory_candidate_k") != 998
            or config.get("microcategory_top_categories") != 3
            or config.get("microcategory_field") != "combined"
            or config.get("embedding_document_format", "current") != "current"):
        raise ValueError("Composite supplemental policy differs from the verified prototype")
    return mode


def check_training_prior_file(train_data, artifact):
    import joblib
    path = Path(train_data).with_suffix(Path(train_data).suffix + ".priors.joblib")
    expected = artifact.get("provenance", {}).get("training_priors_sha256")
    if not expected or not path.is_file() or digest(path) != expected:
        raise ValueError("V7 model's proven training-prior file changed or is missing")
    priors = joblib.load(path)
    wrapped = {"priors": priors,
               "disable_microcategory_prior": artifact.get("disable_microcategory_prior", False)}
    if research_prior_policy(wrapped) != research_prior_policy(artifact):
        raise ValueError("V7 inference classifier/priors differ from its feature-preparation objects")
    return expected


def check_cold_groups(groups, fit_texts):
    identities = set()
    for group in groups:
        text = normalize(group["query_text"])
        identity = group["group_id"]
        if (text != group["query_text"] or text in fit_texts
                or not held_out(text, .2, 42) or identity in identities):
            raise ValueError("Tune groups must be unique cold query texts outside encoder and classifier fit")
        identities.add(identity)


def source_snapshot():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "scripts/cache_reranker_audit.py",
             root / "scripts/produce_document_hypotheses.py"]
    paths.extend(root / "src/avito_candidates" / name for name in (
        "reranker.py", "baseline.py", "hybrid.py", "dense.py", "microcategory.py",
        "embedding_formats.py", "geography.py", "fitted_fingerprint.py"))
    return {str(path.resolve()): digest(path) for path in paths}


def check_sources(snapshot):
    if any(not Path(path).is_file() or digest(path) != expected for path, expected in snapshot.items()):
        raise ValueError("Research feature/retrieval producer source changed")


def build_saved_retriever(config, items, priors, *, dense_cache, batch_size, device):
    """Restore saved objects directly; this function has no fit call or train path."""
    from avito_candidates.baseline import Baseline
    from avito_candidates.dense import DenseRetriever
    from avito_candidates.hybrid import HybridRetriever
    from avito_candidates.microcategory import MicrocategoryChannel
    classifier = priors["microcategory_classifier"]
    supplemental = check_supplemental_config(config)
    kwargs = {f"{field}_{name}": config[f"{field}_{name}"]
              for field in ("title", "params", "description") for name in ("k1", "b", "weight")}
    kwargs.update({key: config[key] for key in ("stem_title_weight", "stem_params_weight",
                                               "location_bonus", "candidate_k")})
    kwargs.update(rank_constant=config["bm25_rrf_k"], channel_quota=config["bm25_channel_quota"],
        geo_candidate_k=config.get("geo_candidate_k", 0),
        geo_top_locations=config.get("geo_top_locations", 3),
        geo_include_related=config.get("geo_include_related", False),
        geo_min_history=config.get("geo_min_history", 20), geo_weight=config.get("geo_weight", .25),
        geo_associations=priors.get("geo_associations"), geo_leave_query_out=False)
    if kwargs["geo_candidate_k"] and kwargs["geo_associations"] is None:
        raise ValueError("Inference geography must use the saved fit-only associations")
    bm25 = Baseline(items, **kwargs)
    dense = DenseRetriever(items, model_name=config["embedding_model"], cache_dir=dense_cache,
        batch_size=batch_size, max_seq_length=config["embedding_max_length"],
        params_words=config["embedding_params_words"], description_words=config["embedding_description_words"],
        ef_search=config["ef_search"], document_format=config.get("embedding_document_format", "current"),
        device=device, require_cached_index=True)
    channel = MicrocategoryChannel(classifier, items,
        indexes={"title": bm25.title_index, "params": bm25.params_index, "description": bm25.description_index},
        field_weights={"title": 2.0, "params": 1.0, "description": .5},
        top_categories=config["microcategory_top_categories"], max_pool_size=998)
    return HybridRetriever(bm25, dense, candidate_k=config["candidate_k"], rank_constant=config["rrf_k"],
        bm25_weight=config["bm25_weight"], dense_weight=config["dense_weight"],
        channel_quota=config["channel_quota"], location_bonus=config["location_bonus"],
        rescore_dense_union=config.get("rescore_dense_union", False), microcategory_channel=channel,
        microcategory_candidate_k=config["microcategory_candidate_k"], microcategory_field=config["microcategory_field"],
        microcategory_weight=config["microcategory_weight"], microcategory_exclude_self=True,
        supplemental_fill=supplemental)


def produce(args):
    require_offline_environment()
    import joblib
    import pyarrow.parquet as pq
    manifest, protocol_hash = EVAL.checked_protocol(args.protocol_dir)
    model_hash, training_hash = digest(args.model), digest(args.train_data)
    snapshot = source_snapshot()
    artifact = joblib.load(args.model)
    fit_path = EVAL.protocol_file(args.protocol_dir, manifest, "fit_texts")
    fit_texts = set(json.loads(fit_path.read_text(encoding="utf-8"))["texts"])
    validate_research_artifact(artifact, fit_texts, manifest)
    EVAL.check_encoder_identity(artifact, manifest)
    EVAL.check_training_data(args.train_data, artifact, args.protocol_dir, manifest)
    prior_file_hash = check_training_prior_file(args.train_data, artifact)
    metadata = artifact["metadata"]
    config = metadata["retrieval_config"]
    base_reference = check_encoder_base(config["embedding_model"], manifest, args.encoder_base_reference)
    policy = research_prior_policy(artifact)
    item_path = EVAL.protocol_file(args.protocol_dir, manifest, "benchmark_items")
    items = [{field: row.get(field, "") for field in ITEM_COLUMNS}
             for row in iter_catalog_records(item_path)]
    if corpus_fingerprint(items, feature_version=metadata["feature_version"]) != metadata["corpus_fingerprint"]:
        raise ValueError("Research model inventory differs before opening Tune targets")
    # Reranker restores both saved priors and the V7 classifier. Keeping only
    # one deserialized artifact avoids duplicating classifier coefficient RAM.
    del artifact
    reranker = Reranker(args.model, items, config, evaluation_split=("queries", .2, 42))
    classifier = reranker.builder.microcategory_classifier
    priors = {"microcategory_classifier": classifier,
              "microcategory_prior": reranker.builder.microcategory_prior,
              "geo_associations": reranker.builder.geo_associations}
    if research_prior_policy({"priors": priors, "disable_microcategory_prior": False}) != policy:
        raise ValueError("Restored inference priors differ from the proven artifact")
    if digest(args.model) != model_hash:
        raise ValueError("Research model changed during provenance validation")
    # Only now parse Tune truth. Audit has no code path in this script.
    groups = list(EVAL.read_phase_labels(args.protocol_dir, manifest, "tune").values())
    check_cold_groups(groups, fit_texts)
    items_by_id = {item["item_id"]: item for item in items}
    corpus_ids = set(items_by_id)
    dataset_train = EVAL.protocol_file(args.protocol_dir, manifest, "train")
    fit_items = set()
    for batch in pq.ParquetFile(dataset_train).iter_batches(batch_size=4096, columns=["search_query", "item_id"]):
        for row in batch.to_pylist():
            if normalize(row["search_query"] or "") in fit_texts:
                fit_items.add(str(row["item_id"]))
    retriever = build_saved_retriever(config, items, priors, dense_cache=args.dense_cache,
                                     batch_size=args.batch_size, device=args.device)
    truth, representatives, additions = {}, {}, {}
    for group in groups:
        key = tuple(group["query_key"])
        relevant, available = set(group["relevant_item_ids"]), set(group["available_item_ids"])
        if available != relevant & corpus_ids:
            raise ValueError("Tune target availability differs from the frozen catalog")
        truth[key], representatives[key] = available, dict(zip(SEARCH_FIELDS, key))
        additions[key] = {"group_identity": group["group_id"], "full_relevant_count": len(relevant),
                          "available_relevant_count": len(available), "representative": group["representative"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extra = {"fresh_validation_protocol_sha256": protocol_hash, "fresh_partition": "tune",
             **policy, "producer_sha256": digest(Path(__file__)), "full_truth_preserved": True,
             "research_cache_version": 1, "research_source_sha256": snapshot,
             "research_encoder_base": base_reference, "research_train_data_sha256": training_hash,
             "research_training_priors_file_sha256": prior_file_hash,
             "research_scope": "Tune-only new V7/V8 pool; hypotheses already selected on this arm"}
    result = build_cache(args.output_dir / "tune.parquet", truth=truth, representatives=representatives,
        keys=sorted(truth, key=query_order), builder=reranker.builder, retriever=retriever,
        items_by_id=items_by_id, fit_items=fit_items, benchmark_strata=[], model_metadata=metadata,
        model_reference=args.model, model_hash=model_hash, batch_size=args.batch_size,
        role="validation", max_pool_size=998, partitioner=lambda _text: "tune",
        historical_holdout_reused=False, extra_group_metadata=additions, extra_metadata=extra,
        allow_empty_relevance=True)
    check_sources(snapshot)
    if (digest(args.model) != model_hash or digest(args.train_data) != training_hash
            or digest(args.protocol_dir / "manifest.json") != protocol_hash
            or check_encoder_base(config["embedding_model"], manifest, args.encoder_base_reference) != base_reference):
        raise ValueError("Research cache source/model/base changed during retrieval")
    print(json.dumps({"complete": True, "groups": len(result["groups"]),
                      "output": str(args.output_dir / "tune.parquet")}), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["tune"], default="tune")
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dense-cache", type=Path, required=True)
    parser.add_argument("--encoder-base-reference", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("Batch size must be positive")
    produce(args)


if __name__ == "__main__":
    main()
