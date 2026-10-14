"""Independently verify the completed geo-dense reranker Tune experiment.

No training, retrieval, encoder loading or Audit access occurs. Metrics and
capacity comparisons are reconstructed from per-context counts; the selected
artifact is read back only after the full evaluation and publication finish.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import re


STAGES = (50, 75, 100, 150)
EXPECTED_GROUPS = 12103
TOLERANCE = .002
GEO_CONFIG = {
    "dense_retrieval_policy": "exact-cosine-plus-location-log-prior-v1",
    "geo_dense_weight": .05, "geo_dense_smoothing_mass": 3.,
    "geo_dense_distance_scale_km": 40., "geo_dense_probability_floor": 1e-7,
}
CONTEXT = ("group_identity", "query_key", "query_text", "representative",
           "full_relevant_count", "available_relevant_count")
POOL_COUNTS = ("pool_size", "pool_hits", "baseline_hits", "dense_hits")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def normalized(value):
    return " ".join(str(value).lower().replace("ё", "е").split())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_identity(info, label):
    path = Path(info["path"])
    require(path.is_file() and sha256(path) == info["sha256"], f"Changed {label}")
    return path


def close(actual, expected, label):
    if expected is None:
        require(actual is None, f"Expected undefined aggregate: {label}")
    else:
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                f"Aggregate disagrees with saved context counts: {label}")


def aggregate(rows, hits, denominator):
    """Compute text/group/representative/micro Recall without evaluator helpers."""
    clusters, values, representatives = defaultdict(list), [], []
    hit_sum = denominator_sum = 0
    for row in rows:
        count, found = row[denominator], row[hits]
        require(integer(count) and integer(found) and found <= count, "Invalid Recall counts")
        if not count:
            continue
        value = found / count
        clusters[normalized(row["query_text"])].append(value)
        values.append(value)
        hit_sum += found
        denominator_sum += count
        if row["representative"]:
            representatives.append(value)
    return {
        "groups": len(values), "texts": len(clusters), "representative_groups": len(representatives),
        "text_macro_recall_at_50": (sum(sum(v) / len(v) for v in clusters.values()) / len(clusters)
                                   if clusters else None),
        "group_macro_recall_at_50": sum(values) / len(values) if values else None,
        "representative_macro_recall_at_50": (sum(representatives) / len(representatives)
                                             if representatives else None),
        "micro_recall_at_50": hit_sum / denominator_sum if denominator_sum else None,
    }


def verify_summary(rows, saved, stage):
    for scope, denominator in (("conditional_available", "available_relevant_count"),
                               ("full_truth", "full_relevant_count")):
        for channel, hits in (("candidate_pool", "pool_hits"), ("baseline_rrf", "baseline_hits"),
                              ("reranker", f"s{stage}_hits")):
            actual = aggregate(rows, hits, denominator)
            for key, value in actual.items():
                saved_key = key.replace("recall_at_50", "candidate_recall") if channel == "candidate_pool" else key
                close(saved[scope][channel][saved_key], value, f"{scope}/{channel}/{key}")
    coverage = {
        "groups": len(rows), "texts": len({row["query_text"] for row in rows}),
        "groups_without_available_target": sum(not row["available_relevant_count"] for row in rows),
        "full_relevant_pairs": sum(row["full_relevant_count"] for row in rows),
        "available_relevant_pairs": sum(row["available_relevant_count"] for row in rows),
        "group_macro_available_fraction": sum(row["available_relevant_count"] / row["full_relevant_count"]
                                              for row in rows) / len(rows),
        "maximum_pool_size": max(row["pool_size"] for row in rows),
        "average_pool_size": sum(row["pool_size"] for row in rows) / len(rows),
    }
    for key, value in coverage.items():
        close(saved["coverage"][key], value, "coverage/" + key)
    # Verify descriptive slices too, without reading any held-out item IDs.
    for name, subset in saved.get("slices", {}).items():
        if "=" not in name:
            continue
        field, flag = name.split("=", 1)
        require(field in ("has_filters", "no_exact_location", "one_word") and flag in ("true", "false"),
                "Unexpected descriptive slice")
        selected = [row for row in rows if bool(row[field]) == (flag == "true")]
        for channel, hits in (("candidate_pool", "pool_hits"), ("baseline_rrf", "baseline_hits"),
                              ("reranker", f"s{stage}_hits")):
            recorded = subset[channel]
            for key, value in aggregate(selected, hits, "available_relevant_count").items():
                saved_key = key.replace("recall_at_50", "candidate_recall") if channel == "candidate_pool" else key
                close(recorded[saved_key], value, f"slice/{name}/{channel}/{key}")
    return coverage


def validate_row(row, stages, *, dense):
    key = row["query_key"]
    require(isinstance(key, list) and len(key) == 5 and all(isinstance(v, str) and v == normalized(v) for v in key),
            "Invalid canonical query key")
    serialized = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    require(row["group_identity"] == hashlib.sha256(serialized.encode()).hexdigest()
            and row["query_text"] == key[0] and isinstance(row["representative"], bool),
            "Invalid query identity, normalization or representative flag")
    available, full, size = row["available_relevant_count"], row["full_relevant_count"], row["pool_size"]
    require(integer(full, 1) and integer(available) and available <= full and integer(size) and size <= 998,
            "Invalid complete denominators or pool budget")
    require(integer(row["pool_hits"]) and row["pool_hits"] <= min(available, size), "Invalid pool hits")
    for name in ("baseline_hits", *(f"s{stage}_hits" for stage in stages)):
        require(integer(row[name]) and row[name] <= min(50, row["pool_hits"]), "Invalid top50 hits")
    if dense:
        require(integer(row["dense_hits"]) and row["dense_hits"] <= min(449, row["pool_hits"]), "Invalid dense449 hits")


def choose_stage(scores):
    require(set(scores) == set(STAGES) and all(math.isfinite(x) and 0 <= x <= 1 for x in scores.values()),
            "Every declared capacity needs valid Recall")
    best = max(scores.values())
    return min(stage for stage, score in scores.items() if best - score <= TOLERANCE + 1e-12)


def paired_capacity_bootstrap(rows, left, right, *, replicates=10000, seed=20261014):
    """Resample normalized query texts, never individual contexts or item rows."""
    import numpy as np
    require(integer(replicates, 1), "Positive bootstrap replicate count required")
    clusters = defaultdict(list)
    for row in rows:
        count = row["available_relevant_count"]
        if count:
            clusters[normalized(row["query_text"])].append((row[f"s{left}_hits"] - row[f"s{right}_hits"]) / count)
    values = np.asarray([sum(v) / len(v) for _, v in sorted(clusters.items())], dtype=np.float64)
    require(len(values) > 0, "No available-positive text clusters")
    rng, means = np.random.default_rng(seed), np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 128):
        stop = min(start + 128, replicates)
        means[start:stop] = values[rng.integers(len(values), size=(stop - start, len(values)))].mean(axis=1)
    lo, hi = map(float, np.quantile(means, [.025, .975]))
    return {"comparison": f"HGB{left} minus HGB{right}; same geo0.05 pool",
            "delta": float(values.mean()), "percentile_95_interval": [lo, hi],
            "zero_in_interval": lo <= 0 <= hi, "bootstrap_unit": "normalized query text",
            "query_text_clusters": len(values), "replicates": replicates, "seed": seed,
            "interpretation": "Post-selection diagnostic on consumed Tune; not independent confirmation"}


def verify_selected_model(report_path, report, selected_stage):
    """Reload the published prefix and compare its actual fitted estimator."""
    import joblib
    from avito_candidates.fitted_fingerprint import fitted_estimator_fingerprint
    model_path = report_path.parent / f"hgb-geo005-selected{selected_stage}.joblib"
    envelope = read_json(model_path.with_suffix(model_path.suffix + ".json"))
    checked_identity(envelope["model"], "published model")
    require(Path(envelope["model"]["path"]).resolve() == model_path.resolve(), "Published model path differs")
    artifact = joblib.load(model_path)
    selection = artifact["geo_dense_selection"]
    require(artifact.get("selected_iteration") == selected_stage
            and artifact["training_configuration"].get("selected_iteration") == selected_stage
            and artifact["model"].n_iter_ == 150 and artifact.get("estimator") == "histgb",
            "Published model does not preserve the selected runtime prefix and full HGB150")
    require(selection["report"]["sha256"] == sha256(report_path)
            and selection["source_model"] == report["input_identities"]["new_model"]
            and selection["stage_tolerance"] == TOLERANCE
            and selection["selected_stage"] == selected_stage, "Published selection provenance differs")
    checked_identity(selection["declaration"], "selection declaration")
    actual = fitted_estimator_fingerprint(artifact["model"])
    require(actual == selection["fitted_estimator_sha256"], "Published fitted estimator fingerprint differs")
    metadata, provenance = artifact["metadata"], artifact["provenance"]
    require(all(metadata["retrieval_config"].get(k) == v for k, v in GEO_CONFIG.items()), "Published geo policy differs")
    train_path = checked_identity(report["input_identities"]["train_data"], "new training data")
    metadata_path = train_path.with_suffix(train_path.suffix + ".json")
    require(metadata == read_json(metadata_path)
            and sha256(metadata_path) == provenance["training_metadata_sha256"]
            and report["input_identities"]["train_data"]["sha256"] == provenance["training_data_sha256"],
            "Published model and training metadata differ")
    require(all(envelope.get(k) == v for k, v in selection.items()), "Published envelope differs from embedded selection")
    del artifact
    gc.collect()
    source_path = checked_identity(report["input_identities"]["new_model"], "full fitted HGB source")
    source = joblib.load(source_path)
    require(source.get("selected_iteration") is None and source["metadata"] == metadata
            and fitted_estimator_fingerprint(source["model"]) == actual, "Selected artifact changed the fitted trees")
    del source
    gc.collect()
    return {"path": str(model_path.resolve()), "sha256": sha256(model_path), "selected_iteration": selected_stage,
            "complete_fitted_iterations": 150, "independent_reload_verified": True,
            "fitted_estimator_sha256": actual}, metadata


def verify_feature_cache(report_path, report, rows, training_metadata):
    """Check optional diagnostic-cache hashes, footer/schema and group metadata.

    The large Parquet file is hashed once. No feature matrix, encoder or model
    is loaded here, and Tune rows never enter any training-data reader.
    """
    descriptor = report.get("feature_cache")
    require(bool(descriptor) == bool(report.get("save_feature_cache", False)),
            "Declared feature-cache export and report descriptor disagree")
    if descriptor is None:
        return {"present": False}
    import pyarrow as pa
    import pyarrow.parquet as pq
    data_path, metadata_path = Path(descriptor["data_path"]), Path(descriptor["metadata_path"])
    expected_data = report_path.parent / "tune-features.parquet"
    require(descriptor.get("role") == "tune-only-never-training"
            and data_path.resolve() == expected_data.resolve()
            and metadata_path.resolve() == data_path.with_suffix(data_path.suffix + ".json").resolve(),
            "Diagnostic cache role or output paths differ")
    marker = data_path.with_suffix(data_path.suffix + ".INCOMPLETE")
    require(not marker.exists() and metadata_path.is_file(), "Diagnostic feature cache is incomplete")
    metadata = read_json(metadata_path)
    require(metadata.get("complete") is True and metadata.get("full_tune_complete") is True
            and metadata.get("role") == "tune-only-never-training" and metadata.get("phase") == "tune"
            and metadata.get("training_allowed") is False and metadata.get("audit_labels_read") is False,
            "Diagnostic cache scope is not completed Tune-only nontraining data")
    require(checked_identity(metadata["data"], "diagnostic Parquet cache").resolve() == data_path.resolve(),
            "Diagnostic cache data identity points to a different file")
    expected_report = {"path": str(report_path.resolve()), "sha256": sha256(report_path)}
    expected_scores = {"path": str((report_path.parent / "group-scores.jsonl").resolve()),
                       "sha256": report["group_scores_sha256"]}
    require(metadata["evaluation_report"] == expected_report
            and metadata["all_group_metadata_and_denominators"] == expected_scores,
            "Diagnostic cache is not bound to this complete evaluation")
    expected_groups = [{"group_identity": row["group_identity"], "pool_size": row["pool_size"],
        "available_relevant_count": row["available_relevant_count"],
        "full_relevant_count": row["full_relevant_count"], "ordered_pool_sha256": row["ordered_pool_sha256"],
        "exported_rows": row["pool_size"] if row["available_relevant_count"] else 0,
        "exported_positive_rows": row["pool_hits"] if row["available_relevant_count"] else 0} for row in rows]
    require(metadata.get("groups_in_evaluation_order") == expected_groups,
            "Diagnostic cache group identities, pool hashes or denominators changed")
    count = sum(group["exported_rows"] for group in expected_groups)
    positives = sum(group["exported_positive_rows"] for group in expected_groups)
    eligible = sum(bool(row["available_relevant_count"]) for row in rows)
    require(metadata.get("rows") == count and metadata.get("positive_rows") == positives
            and metadata.get("groups") == len(rows) and metadata.get("exported_groups") == eligible
            and metadata.get("zero_availability_groups_skipped") == len(rows) - eligible,
            "Diagnostic cache recorded counts disagree with complete evaluation")
    feature_names = training_metadata["feature_names"]
    require(metadata.get("feature_names") == feature_names and metadata.get("feature_dtype") == "float32"
            and metadata.get("identity_columns") == ["group_identity", "item_id"]
            and metadata.get("positive_column") == "available_positive",
            "Diagnostic cache feature or identity schema differs from the reranker")
    parquet = pq.ParquetFile(data_path)
    expected_schema = pa.schema([(name, pa.float32()) for name in feature_names] + [
        ("group_identity", pa.string()), ("item_id", pa.string()), ("available_positive", pa.bool_())])
    require(parquet.metadata.num_rows == count and parquet.schema_arrow.equals(expected_schema, check_metadata=False)
            and "label" not in parquet.schema_arrow.names and "weight" not in parquet.schema_arrow.names,
            "Diagnostic cache Parquet row count or schema differs")
    provenance = metadata["provenance"]
    require(provenance.get("protocol_sha256") == report["protocol_sha256"]
            and provenance.get("input_identities") == report["input_identities"]
            and provenance.get("vector_manifest") == report["vector_manifest"]
            and provenance.get("retrieval_config") == training_metadata["retrieval_config"]
            and provenance.get("source_sha256") == report["experiment_source_sha256"]
            and provenance.get("research_prior_policy") == training_metadata["geo_dense_training"]["source_prior_policy"],
            "Diagnostic cache models, vectors, retrieval policy or priors differ")
    return {"present": True, "verified": True, "role": "tune-only-never-training", "training_allowed": False,
            "data": dict(metadata["data"]), "metadata": {"path": str(metadata_path.resolve()), "sha256": sha256(metadata_path)},
            "rows": count, "positive_rows": positives, "exported_groups": eligible,
            "zero_availability_groups_skipped": len(rows) - eligible,
            "parquet_row_groups": parquet.metadata.num_row_groups,
            "validation_scope": "File SHA256, Parquet footer/schema, complete ordered group metadata; numeric feature matrices not reread"}


def summarize(report_path, *, replicates=10000):
    report_path = Path(report_path)
    report = read_json(report_path)
    require(report.get("complete") is True and report.get("full_tune_complete") is True
            and report.get("phase") == "tune" and report.get("group_limit") == 0
            and report.get("groups") == EXPECTED_GROUPS and report.get("audit_labels_read") is False
            and report.get("reranker_retrained") is True
            and report.get("all_per_context_pool_metrics_reproduced") is True,
            "Only the completed full Tune retraining report is accepted")
    require(report.get("stages") == list(STAGES) and report.get("primary_stage") == 100
            and report.get("selection_tolerance") == TOLERANCE and report.get("geo_dense_config") == GEO_CONFIG
            and report.get("maximum_pool_size") == 998, "Declared experiment or selection policy differs")
    inputs = report["input_identities"]
    previous_report_path = checked_identity(inputs["previous_geo_report"], "previous geo report")
    previous_scores_path = checked_identity(inputs["previous_geo_scores"], "previous geo scores")
    baseline_report_path = checked_identity(inputs["baseline_report"], "V8 baseline report")
    baseline_scores_path = checked_identity(inputs["baseline_scores"], "V8 baseline scores")
    previous_report, baseline_report = read_json(previous_report_path), read_json(baseline_report_path)
    require(previous_report["protocol_sha256"] == baseline_report["protocol_sha256"] == report["protocol_sha256"],
            "Tune protocols differ")
    require(previous_report["group_scores_sha256"] == sha256(previous_scores_path)
            and baseline_report["group_scores_sha256"] == sha256(baseline_scores_path), "Baseline score provenance differs")
    metadata_path = checked_identity(baseline_report["new_pipeline_identity"]["metadata"], "Tune group metadata")
    groups = read_json(metadata_path)["groups"]
    expected = [row["group_identity"] for row in groups]
    require(len(expected) == len(set(expected)) == EXPECTED_GROUPS, "Incomplete or duplicate canonical Tune cohort")
    baseline, previous = {}, {}
    for path, mapping, extract in ((baseline_scores_path, baseline, lambda r: r["new"]),
                                  (previous_scores_path, previous, lambda r: r["variants"]["geo_0.05"])):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = extract(json.loads(line))
                key = row["group_identity"]
                require(key not in mapping, "Duplicate reference context")
                mapping[key] = row
        require(set(mapping) == set(expected), "Reference has missing or additional contexts")
    scores_path = report_path.parent / "group-scores.jsonl"
    require(sha256(scores_path) == report["group_scores_sha256"], "Final per-context scores changed")
    rows, old_same, old_v8, observed = [], [], [], []
    with scores_path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            require(index < len(groups), "Extra final Tune row")
            record = json.loads(line)
            row, key, group = record["new"], record["group_identity"], groups[index]
            require(key == row["group_identity"] == expected[index], "Final Tune rows are missing, repeated or reordered")
            require(all(row[field] == group[field] for field in CONTEXT), "Final context or denominator differs")
            require(record["old_same_pool"] == previous[key] and record["old_v8"] == baseline[key],
                    "Embedded old scores differ from independent original files")
            require(all(row[field] == previous[key][field] for field in (*CONTEXT, *POOL_COUNTS)),
                    "New model changed prior geo pool metrics or denominators")
            require(all(row[field] == baseline[key][field] for field in CONTEXT), "Baseline denominators differ")
            validate_row(row, STAGES, dense=True)
            validate_row(previous[key], (100,), dense=True)
            validate_row(baseline[key], (100,), dense=False)
            require(re.fullmatch(r"[0-9a-f]{64}", row.get("ordered_pool_sha256", "")), "Missing future pool identity hash")
            rows.append(row); old_same.append(previous[key]); old_v8.append(baseline[key]); observed.append(key)
    require(observed == expected, "Not all 12103 contexts were scored")
    require(set(report["new_capacity_curve"]) == {str(s) for s in STAGES}, "Capacity curve is incomplete")
    for stage in STAGES:
        verify_summary(rows, report["new_capacity_curve"][str(stage)], stage)
    verify_summary(old_same, report["old_same_pool_hgb100"], 100)
    coverage = verify_summary(old_v8, report["old_v8_hgb100"], 100)
    dense = aggregate(rows, "dense_hits", "available_relevant_count")
    for key, value in dense.items():
        close(report["dense_recall"][key], value, "dense449/" + key)
    scores = {stage: aggregate(rows, f"s{stage}_hits", "available_relevant_count")["text_macro_recall_at_50"]
              for stage in STAGES}
    selected = choose_stage(scores)
    require(selected == report["selected_stage"], "Published capacity does not follow the declared selection rule")
    references = {"old_same_pool_hgb100": (old_same, "s100_hits"),
                  "old_v8_hgb100": (old_v8, "s100_hits"), "same_pool_rrf": (rows, "baseline_hits")}
    checked_deltas = {}
    for stage in STAGES:
        for name, (reference, hits) in references.items():
            delta = scores[stage] - aggregate(reference, hits, "available_relevant_count")["text_macro_recall_at_50"]
            saved = report["paired_comparisons"][str(stage)][name]
            close(saved["delta"], delta, f"paired/{stage}/{name}")
            lo, hi = saved["percentile_95_interval"]
            require(math.isfinite(lo) and math.isfinite(hi) and lo <= hi
                    and saved["bootstrap_unit"] == "normalized query text" and saved["query_text_clusters"] == dense["texts"],
                    "Invalid saved text-cluster confidence interval")
            if stage == selected:
                checked_deltas[name] = {"delta": delta, "percentile_95_interval": [lo, hi],
                                       "replicates": saved["replicates"]}
    selected_model, training_metadata = verify_selected_model(report_path, report, selected)
    feature_cache = verify_feature_cache(report_path, report, rows, training_metadata)
    stats = training_metadata["stats"]
    require(stats["sampled_queries"] == 30000 and 0 < stats["queries_with_positive_in_pool"] <= 30000,
            "Unexpected fit query budget")
    losses = report.get("training_row_loss")
    require(isinstance(losses, dict) and losses.get("applicable") is True
            and losses["rows"] == stats["positive_rows"] + stats["unobserved_rows"]
            and set(losses["stages"]) == {str(s) for s in STAGES}, "Missing or inconsistent complete train-loss curve")
    require(math.isclose(losses["total_weight"], 2 * stats["queries_with_positive_in_pool"], rel_tol=1e-6),
            "Training weights are not preserved per-query class weights")
    for values in losses["stages"].values():
        require(all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values.values()), "Invalid train loss")
    capacity = {f"selected{selected}_minus150": paired_capacity_bootstrap(rows, selected, 150, replicates=replicates)}
    if selected != 100:
        capacity["primary100_minus150"] = paired_capacity_bootstrap(rows, 100, 150, replicates=replicates)
    diagnostic = capacity[f"selected{selected}_minus150"]
    decreasing_loss = losses["stages"]["150"]["weighted_log_loss"] < losses["stages"][str(selected)]["weighted_log_loss"]
    possible_capacity_overfit = selected != 150 and decreasing_loss and scores[selected] > scores[150]
    table = [{"name": "old_v8_hgb100", "recall_at_50": aggregate(old_v8, "s100_hits", "available_relevant_count")["text_macro_recall_at_50"],
              "pool_recall": aggregate(old_v8, "pool_hits", "available_relevant_count")["text_macro_recall_at_50"]},
             {"name": "old_hgb100_on_geo005", "recall_at_50": aggregate(old_same, "s100_hits", "available_relevant_count")["text_macro_recall_at_50"],
              "pool_recall": aggregate(rows, "pool_hits", "available_relevant_count")["text_macro_recall_at_50"]},
             {"name": "rrf_on_geo005", "recall_at_50": aggregate(rows, "baseline_hits", "available_relevant_count")["text_macro_recall_at_50"],
              "pool_recall": aggregate(rows, "pool_hits", "available_relevant_count")["text_macro_recall_at_50"]}]
    table.extend({"name": f"retrained_hgb{stage}", "recall_at_50": scores[stage],
                  "pool_recall": table[-1]["pool_recall"], "selected": stage == selected,
                  "weighted_train_log_loss": losses["stages"][str(stage)]["weighted_log_loss"]} for stage in STAGES)
    return {"verified": True, "phase": "tune", "groups": len(rows), "available_groups": dense["groups"],
        "available_texts": dense["texts"], "metric": "text-macro Recall@50, conditional on available positives",
        "full_truth_selected_recall": aggregate(rows, f"s{selected}_hits", "full_relevant_count"),
        "coverage": {**coverage, "average_pool_size": sum(r["pool_size"] for r in rows) / len(rows),
                     "maximum_pool_size": max(r["pool_size"] for r in rows)},
        "all_contexts_and_denominators_verified": True, "all_per_context_pool_metrics_verified": True,
        "candidate_ids_parity_limit": "Prior run stored counts, not pool IDs; no claim of an independent prior ID-by-ID proof",
        "table": table, "selected_stage": selected, "primary_stage": 100,
        "selection_rule": "Smallest of50/75/100/150 within0.002 of best Tune text-macro Recall",
        "selected_model": selected_model, "training_stats": stats, "training_row_loss": losses,
        "diagnostic_feature_cache": feature_cache,
        "selected_paired_comparisons": checked_deltas, "capacity_diagnostics": capacity,
        "overfitting_check": {"weighted_train_loss_falls_selected_to150": decreasing_loss,
            "tune_recall_falls_selected_to150": scores[selected] > scores[150],
            "possible_capacity_overfit_pattern": possible_capacity_overfit,
            "selected_minus150_interval_excludes_zero": not diagnostic["zero_in_interval"],
            "interpretation": "Lower train loss with lower Tune Recall is consistent with excess capacity, not proof; consumed Tune cannot rule out wider overfitting"},
        "inventory_reencoded": False, "audit_labels_opened": False,
        "limitations": ["Tune was consumed by retrieval research and capacity selection; not independent confirmation.",
                        "This is the sampled200000-item catalog, not the official submission catalog or external score.",
                        "Unclicked sampled candidates are unlabelled, not proved irrelevant; binary loss is not percentage error.",
                        "The selected model needs the geo0.05 retrieval route and matching corpus, not the original ANN pipeline."],
        "inputs": {str(path.resolve()): sha256(path) for path in (report_path, scores_path, previous_report_path,
                    previous_scores_path, baseline_report_path, baseline_scores_path, metadata_path)}}


def render(summary):
    lines = ["Переобучение HGB на географическом dense-пуле: независимая проверка",
             f"Полный Tune: {summary['groups']} контекста; метрика на {summary['available_groups']} контекстах / {summary['available_texts']} текстах с доступными положительными объявлениями.",
             "Recall усреднён по нормализованным текстам запросов. Полные знаменатели и пропуски проверены.",
             f"Средний пул: {summary['coverage']['average_pool_size']:.3f}; максимум: {summary['coverage']['maximum_pool_size']}.",
             "", "Вариант                         Recall пула%  Recall@50%  train log loss"]
    for row in summary["table"]:
        loss = f"{row['weighted_train_log_loss']:.6f}" if "weighted_train_log_loss" in row else "—"
        mark = " [выбран]" if row.get("selected") else ""
        lines.append(f"{row['name']:32} {100*row['pool_recall']:10.5f} {100*row['recall_at_50']:11.5f} {loss:>15}{mark}")
    lines += ["", "Выбор: самый короткий префикс в пределах 0,2 п.п. от лучшего на Tune.",
              "Изменения выбранного префикса, п.п.; исследовательские 95% интервалы:"]
    for name, value in summary["selected_paired_comparisons"].items():
        lo, hi = value["percentile_95_interval"]
        lines.append(f"  {name}: {100*value['delta']:+.5f} [{100*lo:+.5f}; {100*hi:+.5f}]")
    lines += ["", "Контроль сложности: парный bootstrap по текстам, 10000 повторов по умолчанию:"]
    for name, value in summary["capacity_diagnostics"].items():
        lo, hi = value["percentile_95_interval"]
        lines.append(f"  {name}: {100*value['delta']:+.5f} п.п. [{100*lo:+.5f}; {100*hi:+.5f}], n={value['replicates']}")
    if summary["overfitting_check"]["possible_capacity_overfit_pattern"]:
        lines.append("К 150 деревьям train loss снижается, а Tune Recall падает: это признак возможной избыточной сложности, не доказательство переобучения.")
    else:
        lines.append("По сравнению выбранного префикса с 150 деревьями сочетания падающего train loss и падающего Tune Recall не выявлено; это не исключает переобучение вообще.")
    if not summary["overfitting_check"]["selected_minus150_interval_excludes_zero"]:
        lines.append("Интервал разницы с 150 деревьями включает ноль: явное преимущество меньшей сложности не установлено.")
    stats = summary["training_stats"]
    lines += ["", f"Обучение: {stats['sampled_queries']} выбранных групп; {stats['queries_with_positive_in_pool']} групп с положительными в пуле; {stats['positive_rows']} положительных + {stats['unobserved_rows']} неразмеченных строк.",
              f"Модель повторно загружена и проверена: {summary['selected_model']['path']}",
              f"Выбранный runtime-префикс: {summary['selected_stage']}; сохранены все 150 обученных деревьев."]
    cache = summary.get("diagnostic_feature_cache", {})
    if cache.get("present"):
        lines += [f"Диагностический кеш Tune проверен: {cache['rows']} строк / {cache['exported_groups']} групп; использовать для обучения нельзя.",
                  cache["data"]["path"]]
    lines += ["", "Ограничения:", *["- " + value for value in summary["limitations"]]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "Use a new summary output directory")
    result = summarize(args.report, replicates=args.bootstrap_replicates)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    text = render(result)
    (args.output_dir / "SUMMARY.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
