"""Tune-only, frozen-model experiment: geography before dense top-K.

No fitting, corpus encoding, Audit access or production-config mutation occurs.
The existing V8 model remains a deliberately unchanged reranker. Its features
retain raw cosine, while its dense ranks reflect the proposed retrieval order.
Weights are declared before reading targets; .05 is the primary intervention.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import time

import numpy as np

from avito_candidates.core import SEARCH_FIELDS, normalize
from avito_candidates.reranker import (FeatureBuilder, V8_FEATURES,
    corpus_fingerprint, prediction_scores)
from avito_candidates.supplemental import select_supplemental
from cache_research_reranker import (check_cold_groups, check_supplemental_config,
    validate_research_artifact, research_prior_policy, check_training_prior_file,
    check_sources)
from produce_document_hypotheses import iter_catalog_records, check_encoder_base
from evaluate_research_reranker import named_summary, paired_pipeline_bootstrap
import evaluate_fresh_validation as EVAL

WEIGHTS = (0., .02, .05, .1)
PRIMARY = "geo_0.05"
DYNAMIC = ("rrf_score", "pool_rank", "baseline_top50", "channel_count",
           "rank_dense", "present_dense", "score_dense",
           "rank_microcategory_channel", "present_microcategory_channel",
           "score_microcategory_channel")


def verify_vector_inputs(root, metadata_path, metadata, protocol_hash):
    """Bind target-blind vectors and ANN controls to the original Tune order."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("complete") is not True or manifest.get("phase") != "tune"
            or manifest.get("target_blind") is not True
            or manifest.get("inventory_reencoded") is not False
            or manifest.get("protocol_sha256") != protocol_hash
            or manifest.get("metadata_sha256") != EVAL.sha256_file(metadata_path)):
        raise ValueError("Vector manifest is not a completed target-blind Tune extraction")
    required = {"corpus.npy", "queries.npy", "ann449.npy", "ann998.npy",
                "ann449-cosines.npy", "ann998-cosines.npy", "order.json"}
    if not required <= set(manifest.get("files", {})):
        raise ValueError("Vector manifest omits a required control array")
    for name, info in manifest["files"].items():
        if Path(name).name != name or EVAL.sha256_file(root / name) != info["sha256"]:
            raise ValueError("Vector input content hash differs")
    order = json.loads((root / "order.json").read_text(encoding="utf-8"))
    if order["query_group_ids"] != [g["group_identity"] for g in metadata["groups"]]:
        raise ValueError("Query vectors differ from complete Tune metadata order")
    arrays = {name: np.load(root / (name + ".npy"), mmap_mode="r")
              for name in ("corpus", "queries", "ann449", "ann998",
                           "ann449-cosines", "ann998-cosines")}
    count, dimension = arrays["corpus"].shape
    groups = len(metadata["groups"])
    if (count != len(order["corpus_item_ids"])
            or len(set(order["corpus_item_ids"])) != count
            or arrays["queries"].shape != (groups, dimension)):
        raise ValueError("Vector shapes or corpus IDs differ")
    for depth in (449, 998):
        labels, cosines = arrays[f"ann{depth}"], arrays[f"ann{depth}-cosines"]
        if labels.shape != (groups, min(depth, count)) or cosines.shape != labels.shape:
            raise ValueError("ANN control shape differs")
    return manifest, order, arrays


def make_retriever(config, items, priors):
    """Restore BM25/micro objects without loading any encoder or HNSW index."""
    from avito_candidates.baseline import Baseline
    from avito_candidates.hybrid import HybridRetriever
    from avito_candidates.microcategory import MicrocategoryChannel
    check_supplemental_config(config)
    if (config.get("candidate_k") != 449 or config.get("geo_candidate_k") != 100
            or config.get("geo_include_related", False)
            or config.get("channel_quota") != 0 or config.get("location_bonus") != .02):
        raise ValueError("Intervention requires the fixed V8 desc449+dense449+geo100 configuration")
    kwargs = {f"{field}_{name}": config[f"{field}_{name}"]
              for field in ("title", "params", "description")
              for name in ("k1", "b", "weight")}
    kwargs.update({key: config[key] for key in ("stem_title_weight", "stem_params_weight",
        "location_bonus", "candidate_k", "geo_candidate_k", "geo_top_locations",
        "geo_min_history", "geo_weight")})
    kwargs.update(rank_constant=config["bm25_rrf_k"], channel_quota=config["bm25_channel_quota"],
                  geo_associations=priors["geo_associations"], geo_leave_query_out=False)
    bm25 = Baseline(items, **kwargs)
    micro = MicrocategoryChannel(priors["microcategory_classifier"], items,
        indexes={"title": bm25.title_index, "params": bm25.params_index,
                 "description": bm25.description_index},
        field_weights={"title": 2., "params": 1., "description": .5},
        top_categories=3, max_pool_size=998)
    # Only _fuse_with_diagnostics is used; the GPU scorer supplies full cosines.
    hybrid = HybridRetriever(bm25, None, candidate_k=449, rank_constant=config["rrf_k"],
        bm25_weight=config["bm25_weight"], dense_weight=config["dense_weight"],
        channel_quota=0, location_bonus=config["location_bonus"])
    return hybrid, micro


def raw_shared_routes(query, hybrid, micro, supplemental_dense):
    """Compute all invariant lexical and category searches once per context."""
    lists, measured = hybrid.bm25.ranked_lists_with_scores(query, hybrid.candidate_k)
    total = sum(weight for _name, _ids, weight in lists)
    base = [(name, ids, hybrid.bm25_weight * weight / total)
            for name, ids, weight in lists]
    lexical = micro._scores(query)
    routes, scores = {}, {}
    for field in ("title", "params"):
        ids, values = micro.indexes[field]._top(lexical[field],
            np.flatnonzero(lexical[field] > 0), 998, return_scores=True)
        name = "supplemental_" + field + "_values"
        routes[name], scores[name] = ids, values
    ids, values, _ = micro.retrieve(query, 998, existing_ids=(), field="combined",
                                   lexical_scores=lexical, exclude_self=True)
    routes["micro_combined"], scores["micro_combined"] = ids, values
    routes["supplemental_dense_current"], scores["supplemental_dense_current"] = supplemental_dense
    return base, measured, routes, scores


def compose_result(query, hybrid, shared, dense_result, cosine, id_to_row, micro_weight):
    """Replace dense449, then apply the unchanged production supplemental fill."""
    base, base_scores, routes, raw_scores = shared
    ids, measured = dense_result
    channels = [*base, ("dense", ids, hybrid.dense_weight)]
    scores = {**base_scores, "dense": measured}
    protected = {item for _name, values, _weight in channels for item in values}
    added, retained, retained_scores, fill_scores = select_supplemental(
        routes, raw_scores, protected, maximum_pool_size=998)
    if added:
        channels.append(("supplemental_fill", added, 0.))
        scores["supplemental_fill"] = fill_scores
    for name, values in retained.items():
        if values:
            channels.append((name, values, micro_weight))
            scores[name] = retained_scores[name]
    baseline, details = hybrid._fuse_with_diagnostics(query, (channels, scores), 50)
    pool = details["candidate_pool"]
    if len(pool) > 998 or len(pool) != len(set(pool)):
        raise ValueError("Intervention exceeded the per-query candidate budget")
    details["full_dense_scores"] = {item: float(cosine[id_to_row[item]]) for item in pool}
    return baseline, details


def shared_variant_matrices(builder, query, results, *, check_parity=False):
    """Compute invariant features once, patch only the ten varying V8 columns.

The temporary union is a feature-computation batch, never a retrieval pool.
Each actual result is independently capped at998. Runtime parity checks use
the unmodified production FeatureBuilder on complete actual variant pools.
"""
    if tuple(builder.feature_names) != V8_FEATURES:
        raise ValueError("Shared feature optimization is limited to exact V8 schema")
    first = next(iter(results.values()))[1]
    union = list(dict.fromkeys(item for _, d in results.values() for item in d["candidate_pool"]))
    full_cosine = {item: value for _, d in results.values() for item, value in d["full_dense_scores"].items()}
    fixed_channels = {name: values for name, values in first["channels"].items()
                      if name != "dense" and name != "micro_combined"
                      and not name.startswith("supplemental_")}
    fixed_scores = {name: first["channel_scores"][name] for name in fixed_channels}
    for _, d in results.values():
        if d["geo_locations"] != first["geo_locations"]:
            raise ValueError("Dense intervention unexpectedly changed geographic lexical features")
        for name, values in fixed_channels.items():
            if d["channels"].get(name) != values or d["channel_scores"].get(name) != fixed_scores[name]:
                raise ValueError("Dense intervention unexpectedly changed a fixed lexical channel")
    synthetic = {"candidate_pool": union, "channels": fixed_channels,
        "channel_scores": fixed_scores, "rrf_scores": dict.fromkeys(union, 0.),
        "geo_locations": first["geo_locations"], "full_dense_scores": full_cosine}
    invariant = builder.transform(query, synthetic, [])
    position = {item: i for i, item in enumerate(union)}
    columns = {name: V8_FEATURES.index(name) for name in DYNAMIC}
    output, maximum_difference = {}, 0.
    rank_constant = builder.config["rrf_k"]
    for name, (baseline, details) in results.items():
        pool = details["candidate_pool"]
        matrix = invariant[[position[item] for item in pool]].copy()
        ranks = {channel: {item: i for i, item in enumerate(ids, 1)}
                 for channel, ids in details["channels"].items()}
        dense_ranks, micro_ranks = ranks.get("dense", {}), ranks.get("micro_combined", {})
        dense_scores = details["channel_scores"].get("dense", {})
        micro_scores = details["channel_scores"].get("micro_combined", {})
        baseline_set = set(baseline)
        for row, item in enumerate(pool):
            dense_rank, micro_rank = dense_ranks.get(item, 0), micro_ranks.get(item, 0)
            values = (details["rrf_scores"][item], 1 / (row + 2), float(item in baseline_set),
                sum(item in rank for rank in ranks.values()),
                1 / (rank_constant + dense_rank) if dense_rank else 0., float(bool(dense_rank)),
                dense_scores.get(item, 0.),
                1 / (rank_constant + micro_rank) if micro_rank else 0., float(bool(micro_rank)),
                micro_scores.get(item, 0.))
            for feature, value in zip(DYNAMIC, values):
                matrix[row, columns[feature]] = value
        if check_parity:
            reference = builder.transform(query, details, baseline)
            difference = float(np.max(np.abs(reference - matrix))) if matrix.size else 0.
            maximum_difference = max(maximum_difference, difference)
            if not np.array_equal(reference, matrix):
                raise ValueError(f"Shared feature optimization differs from production for {name}: {difference}")
        output[name] = matrix
    return output, maximum_difference


def ann_result(labels, values, item_ids):
    labels, values = np.asarray(labels), np.asarray(values)
    if (labels.ndim != 1 or values.shape != labels.shape or not np.isfinite(values).all()
            or np.any(labels < 0) or np.any(labels >= len(item_ids))
            or len(np.unique(labels)) != len(labels)):
        raise ValueError("Invalid frozen ANN control")
    ids = [item_ids[int(label)] for label in labels]
    return ids, dict(zip(ids, map(float, values)))


def metric_row(group, result, relevant, prediction):
    baseline, details = result
    row = {key: value for key, value in group.items()
           if not key.startswith("_") and key not in {"relevant_item_ids", "available_item_ids"}}
    row.update(pool_size=len(details["candidate_pool"]),
               pool_hits=len(set(details["candidate_pool"]) & relevant),
               baseline_hits=len(set(baseline) & relevant),
               s100_hits=len(set(prediction) & relevant))
    return row


def checked_baseline(args, manifest, protocol_hash, artifact):
    """Use already completed V8 Tune results, binding their cached inputs."""
    metadata_path = args.cache_dir / "tune.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    identity = report["new_pipeline_identity"]
    if (report.get("phase") != "tune" or report.get("protocol_sha256") != protocol_hash
            or identity["metadata"]["sha256"] != EVAL.sha256_file(metadata_path)
            or report.get("group_scores_sha256") != EVAL.sha256_file(args.baseline_scores)
            or artifact.get("selected_iteration") != 100):
        raise ValueError("Baseline is not the previously completed same-model V8 Tune result")
    EVAL.ensure_compatible(artifact["metadata"], metadata, "frozen V8")
    check_sources(metadata["research_source_sha256"])
    if research_prior_policy(artifact)["research_priors_sha256"] != metadata["research_priors_sha256"]:
        raise ValueError("Frozen reranker's priors differ from baseline cache")
    rows = {}
    with args.baseline_scores.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)["new"]
            key = row["group_identity"]
            if key in rows:
                raise ValueError("Repeated baseline Tune group")
            rows[key] = row
    if set(rows) != {g["group_identity"] for g in metadata["groups"]}:
        raise ValueError("Baseline omits complete Tune groups")
    # The selected artifact was published from this full150 estimator at100.
    # Compare actual fitted tree semantics, not merely model names.
    import joblib
    from avito_candidates.fitted_fingerprint import fitted_estimator_fingerprint
    source_path = Path(identity["model"]["path"])
    if EVAL.sha256_file(source_path) != identity["model"]["sha256"]:
        raise ValueError("Baseline fitted model changed")
    original = joblib.load(source_path)
    if fitted_estimator_fingerprint(original["model"]) != fitted_estimator_fingerprint(artifact["model"]):
        raise ValueError("Selected100 and baseline150 do not share identical fitted trees")
    return metadata, rows


def evaluate(args):
    import joblib
    import inspect
    from avito_candidates.geo_dense import GeoDensePrior, ExactDenseScorer
    if args.output_dir.exists():
        raise FileExistsError("Use a new experiment output directory")
    args.output_dir.mkdir(parents=True)
    manifest, protocol_hash = EVAL.checked_protocol(args.protocol_dir)
    artifact = joblib.load(args.model)
    model_hash = EVAL.sha256_file(args.model)
    fit_texts = set(json.loads(EVAL.protocol_file(args.protocol_dir, manifest, "fit_texts").read_text())["texts"])
    validate_research_artifact(artifact, fit_texts, manifest)
    EVAL.check_encoder_identity(artifact, manifest)
    check_training_prior_file(args.train_data, artifact)
    metadata, old_rows = checked_baseline(args, manifest, protocol_hash, artifact)
    config = copy.deepcopy(artifact["metadata"]["retrieval_config"])
    base_identity = check_encoder_base(config["embedding_model"], manifest, args.encoder_base_reference)
    vector_manifest, order, arrays = verify_vector_inputs(args.vector_dir,
        args.cache_dir / "tune.json", metadata, protocol_hash)
    experiment_sources = {str(path.resolve()): EVAL.sha256_file(path)
        for path in (Path(__file__), Path(inspect.getfile(GeoDensePrior)))}
    # Save the experiment specification before targets are parsed or scored.
    declaration = {"phase": "tune", "audit_labels_read": False, "weights": list(WEIGHTS),
        "primary": PRIMARY, "maximum_pool_size": 998, "dense_k": 449,
        "geo_prior": {"smoothing_mass": 3., "distance_scale_km": 40., "probability_floor": 1e-7},
        "model_sha256": model_hash, "model_iteration": 100, "reranker_retrained": False,
        "production_retrieval_config": config, "intervention": "Replace only base dense449; unchanged ANN998 supplemental route",
        "feature_semantics": "Dense ranks use geo score; measured dense scores remain raw cosine",
        "encoder_base": base_identity, "vector_manifest_sha256": EVAL.sha256_file(args.vector_dir / "manifest.json"),
        "protocol_sha256": protocol_hash, "group_limit": args.limit_groups,
        "experiment_source_sha256": experiment_sources,
        "interpretation": "Exploratory consumed Tune; unchanged reranker evaluates a new retrieval distribution"}
    EVAL.write_new_json(args.output_dir / "declaration.json", declaration)
    labels = EVAL.read_phase_labels(args.protocol_dir, manifest, "tune")
    check_cold_groups(list(labels.values()), fit_texts)
    groups = metadata["groups"][:args.limit_groups or None]
    if any(not normalize(group["query_key"][0]) for group in groups):
        raise ValueError("This dense intervention requires nonempty query text, matching frozen ANN behavior")
    item_path = EVAL.protocol_file(args.protocol_dir, manifest, "benchmark_items")
    items = list(iter_catalog_records(item_path))
    if [item["item_id"] for item in items] != order["corpus_item_ids"]:
        raise ValueError("Corpus vectors and metadata have different row order")
    if corpus_fingerprint(items, feature_version=8) != metadata["corpus_fingerprint"]:
        raise ValueError("Frozen model corpus identity changed")
    item_ids = order["corpus_item_ids"]
    id_to_row = {item: i for i, item in enumerate(item_ids)}
    corpus_ids = set(id_to_row)
    priors = artifact["priors"]
    prior = GeoDensePrior(items, priors["geo_associations"], smoothing_mass=3.,
                         distance_scale_km=40., probability_floor=1e-7)
    scorer = ExactDenseScorer(arrays["corpus"], device=args.device)
    builder = FeatureBuilder(items, config, microcategory_prior=priors["microcategory_prior"],
        geo_associations=priors["geo_associations"], microcategory_classifier=priors["microcategory_classifier"],
        feature_names=V8_FEATURES)
    hybrid, micro = make_retriever(config, items, priors)
    variants = ["exact_cosine", *(f"geo_{weight:g}" for weight in WEIGHTS[1:])]
    all_rows = {name: [] for name in variants}
    baseline_rows, parity_groups, parity_max, reference_parity_groups = [], 0, 0., 0
    started = time.monotonic()
    output_scores = args.output_dir / "group-scores.jsonl"
    with output_scores.open("x", encoding="utf-8") as stream:
        for start in range(0, len(groups), args.batch_size):
            batch = groups[start:start + args.batch_size]
            cosines = scorer.score(arrays["queries"][start:start + len(batch)])
            for offset, (group, cosine) in enumerate(zip(batch, cosines)):
                index = start + offset
                query = dict(zip(SEARCH_FIELDS, group["query_key"]))
                target = labels[group["group_identity"]]
                relevant = set(target["available_item_ids"])
                if (set(target["relevant_item_ids"]) & corpus_ids != relevant
                        or len(relevant) != group["available_relevant_count"]
                        or len(target["relevant_item_ids"]) != group["full_relevant_count"]):
                    raise ValueError("Complete Tune relevance denominators differ")
                ann998 = ann_result(arrays["ann998"][index], arrays["ann998-cosines"][index], item_ids)
                shared = raw_shared_routes(query, hybrid, micro, ann998)
                results = {}
                for name, weight in zip(variants, WEIGHTS):
                    dense_result = prior.retrieve(cosine, query.get("search_location_id", ""),
                        limit=449, weight=weight, exclude_query_text=group["query_text"])
                    results[name] = compose_result(query, hybrid, shared, dense_result, cosine,
                                                   id_to_row, config["microcategory_weight"])
                old = old_rows[group["group_identity"]]
                # A query with no available positive has exactly zero hits for
                # every selection. Its full-truth zero and pool size are kept;
                # expensive reranking cannot change that mathematical result.
                if relevant:
                    check = parity_groups < args.parity_groups or index % 500 == 0
                    matrices, difference = shared_variant_matrices(builder, query, results, check_parity=check)
                    if check:
                        parity_groups += 1
                        parity_max = max(parity_max, difference)
                    if reference_parity_groups < args.parity_groups:
                        ann449 = ann_result(arrays["ann449"][index], arrays["ann449-cosines"][index], item_ids)
                        reference = compose_result(query, hybrid, shared, ann449, cosine,
                                                   id_to_row, config["microcategory_weight"])
                        reference_matrix = builder.transform(query, reference[1], reference[0])
                        p = prediction_scores(artifact["model"], reference_matrix, selected_iteration=100)
                        ranked = np.lexsort((np.arange(len(p)), -reference_matrix[:, 0], -p))[:50]
                        predicted = [reference[1]["candidate_pool"][i] for i in ranked]
                        observed = metric_row(group, reference, relevant, predicted)
                        if any(observed[key] != old[key] for key in ("pool_size", "pool_hits", "baseline_hits", "s100_hits")):
                            raise ValueError("Reconstructed unchanged ANN V8 differs from its saved baseline metrics")
                        reference_parity_groups += 1
                    joined = np.concatenate([matrices[name] for name in variants])
                    probabilities = prediction_scores(artifact["model"], joined, selected_iteration=100)
                else:
                    matrices, probabilities = {}, np.empty(0)
                cursor, row_output = 0, {}
                for name in variants:
                    baseline, details = results[name]
                    if relevant:
                        count = len(details["candidate_pool"])
                        scores = probabilities[cursor:cursor + count]
                        order50 = np.lexsort((np.arange(count), -matrices[name][:, 0], -scores))[:50]
                        prediction = [details["candidate_pool"][i] for i in order50]
                        cursor += count
                    else:
                        prediction = []
                    row = metric_row(group, (baseline, details), relevant, prediction)
                    row["dense_hits"] = len(set(details["channels"]["dense"]) & relevant)
                    row_output[name] = row
                    all_rows[name].append(row)
                baseline_rows.append(old)
                stream.write(json.dumps({"group_identity": group["group_identity"], "variants": row_output},
                                        ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            done = start + len(batch)
            if start == 0 or done % 128 < args.batch_size or done == len(groups):
                elapsed = time.monotonic() - started
                print(json.dumps({"groups": done, "total": len(groups), "seconds": round(elapsed, 1),
                    "estimated_remaining_seconds": round(elapsed * (len(groups) - done) / done, 1),
                    "feature_parity_groups": parity_groups}), flush=True)
    summaries, comparisons, exact_comparisons = {}, {}, {}
    for name, rows in all_rows.items():
        summaries[name] = named_summary(rows, 100)
        summaries[name]["dense_recall"] = EVAL.aggregate(rows, "dense_hits", "available_relevant_count")
        pairs = list(zip(rows, baseline_rows))
        comparisons[name] = {metric: paired_pipeline_bootstrap(pairs, hits, hits, denominator,
            replicates=args.bootstrap_replicates) for metric, hits, denominator in (
                ("candidate_pool_recall", "pool_hits", "available_relevant_count"),
                ("reranker_recall_at_50", "s100_hits", "available_relevant_count"),
                ("rrf_recall_at_50", "baseline_hits", "available_relevant_count"),
                ("full_truth_recall_at_50", "s100_hits", "full_relevant_count"))}
        for result in comparisons[name].values():
            result["comparison"] = name + " minus frozen V8 selected100"
        if name != "exact_cosine":
            exact_pairs = list(zip(rows, all_rows["exact_cosine"]))
            exact_comparisons[name] = {metric: paired_pipeline_bootstrap(exact_pairs, hits, hits, denominator,
                replicates=args.bootstrap_replicates) for metric, hits, denominator in (
                    ("candidate_pool_recall", "pool_hits", "available_relevant_count"),
                    ("reranker_recall_at_50", "s100_hits", "available_relevant_count"),
                    ("rrf_recall_at_50", "baseline_hits", "available_relevant_count"))}
            for result in exact_comparisons[name].values():
                result["comparison"] = name + " minus exact cosine, isolating preselection geography"
    check_sources(metadata["research_source_sha256"])
    check_sources(experiment_sources)
    if EVAL.sha256_file(args.model) != model_hash:
        raise ValueError("Frozen reranker changed during evaluation")
    report = {**declaration, "complete": True, "groups": len(groups),
        "baseline": named_summary(baseline_rows, 100), "variants": summaries,
        "paired_vs_baseline": comparisons, "feature_parity_groups": parity_groups,
        "paired_vs_exact_cosine": exact_comparisons,
        "baseline_reference_parity_groups": reference_parity_groups,
        "maximum_feature_parity_difference": parity_max,
        "seconds": time.monotonic() - started, "group_scores_sha256": EVAL.sha256_file(output_scores),
        "source_sha256": experiment_sources,
        "scope": "Same current vectors, same frozen HGB100, Tune only; retrieval intervention without retraining",
        "no_available_target_reranking_skipped": True}
    EVAL.write_new_json(args.output_dir / "report.json", report)
    print(json.dumps({"complete": True, "report": str(args.output_dir / "report.json")}), flush=True)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["tune"], default="tune")
    for name in ("protocol-dir", "model", "train-data", "cache-dir", "baseline-report",
                 "baseline-scores", "vector-dir", "encoder-base-reference", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit-groups", type=int, default=0)
    p.add_argument("--parity-groups", type=int, default=32)
    p.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = p.parse_args(argv)
    if min(args.batch_size, args.parity_groups, args.bootstrap_replicates) < 1 or args.limit_groups < 0:
        p.error("Positive batch/parity/bootstrap and nonnegative metadata-prefix limit required")
    evaluate(args)


if __name__ == "__main__":
    main()
