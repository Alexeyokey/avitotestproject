"""Audit saved reranker prefixes using complete, fixed candidate pools.

Run in the project's offline Python environment after the feature-cache process
has exited. This script never retrieves candidates or trains a model. Example:

    python scripts/audit_reranker_learning_curve.py \
      --cache-dir artifacts/reranker-audit-cache \
      --train-data artifacts/reranker-microgeo-v4-k449-geo100/train.parquet \
      --model hgb=artifacts/reranker-microgeo-v4-k449-geo100/model.joblib \
      --model lightgbm=artifacts/reranker-microgeo-v4-k449-geo100/lightgbm-150.joblib \
      --output-dir artifacts/reranker-audit

The train_sample cache must contain the *whole* pool for each sampled training
query. Row losses on the separate, sampled train.parquet are optional and are
not retrieval Recall. Recall denominators include positives outside the pool.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


DEFAULT_STAGES = (25, 50, 100, 150)
SLICE_DENOMINATORS = {
    "all": "relevant_count",
    "local": "local_relevant_count",
    "nonlocal": "nonlocal_relevant_count",
    "seen": "seen_relevant_count",
    "unseen": "unseen_relevant_count",
}
STRATUM_KEYS = ("no_exact_location", "has_filters", "one_word")


def normalize_query_text(query_text: str) -> str:
    """Use the project's query-text normalization without importing its models."""
    return " ".join(str(query_text).lower().replace("ё", "е").split())


def query_text_partition(query_text: str, seed: int = 20261012) -> str:
    """Assign every normalized query text to one deterministic half."""
    normalized = normalize_query_text(query_text)
    # Match avito_candidates.core.held_out(text, fraction=.5, seed=seed).
    value = json.dumps([seed, normalized], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return "tune" if int.from_bytes(digest[:8], "big") / 2**64 < 0.5 else "confirmation"


def model_stages(max_iterations: int, requested: Iterable[int]) -> tuple[int, ...]:
    """Evaluate requested available prefixes and always include the final model."""
    requested = tuple(int(value) for value in requested)
    if max_iterations < 1 or any(value < 1 for value in requested):
        raise ValueError("Model iterations and requested stages must be positive")
    return tuple(sorted({value for value in requested if value <= max_iterations}
                        | {int(max_iterations)}))


def prefix_probabilities(model: Any, matrix: Any, stages: Iterable[int], *, is_ranker: bool = False):
    """Yield actual prefix outputs: class probabilities or one-dimensional rank scores."""
    stages = tuple(sorted(set(stages)))
    if not stages:
        return
    if is_ranker:
        if not hasattr(model, "booster_"):
            raise TypeError("LambdaRank prefix scoring requires a LightGBM booster")
        for stage in stages:
            yield stage, lightgbm_prefix_output(model, matrix, stage, is_ranker=True)
    elif callable(getattr(model, "staged_predict_proba", None)):
        wanted = set(stages)
        found = set()
        for stage, prediction in enumerate(model.staged_predict_proba(matrix), 1):
            if stage in wanted:
                found.add(stage)
                yield stage, prediction
            if stage >= stages[-1]:
                break
        if found != wanted:
            raise ValueError(f"Model did not produce requested stages: {wanted - found}")
    elif hasattr(model, "booster_"):
        for stage in stages:
            yield stage, lightgbm_prefix_output(model, matrix, stage, is_ranker=False)
    else:
        raise TypeError("Supported models are HistGradientBoostingClassifier, LGBMClassifier and LGBMRanker")


def lightgbm_prefix_output(model: Any, matrix: Any, stage: int, *, is_ranker: bool):
    """Use real boosters directly, avoiding sklearn's ndarray feature-name warning."""
    native_predict = getattr(model.booster_, "predict", None)
    if callable(native_predict):
        values = native_predict(matrix, num_iteration=stage)
        if is_ranker:
            return values
        import numpy as np
        probabilities = np.asarray(values)
        if probabilities.ndim != 1:
            raise ValueError("Binary LightGBM booster must return one probability per row")
        return np.column_stack((1 - probabilities, probabilities))
    # Test doubles and compatible wrappers may not expose native Booster.predict.
    if is_ranker:
        return model.predict(matrix, num_iteration=stage)
    return model.predict_proba(matrix, num_iteration=stage)


def is_ranking_artifact(artifact: dict) -> bool:
    return (artifact.get("estimator") == "lambdarank"
            or type(artifact["model"]).__name__ == "LGBMRanker")


def prefix_scores(model: Any, matrix: Any, stages: Iterable[int], *, is_ranker: bool = False):
    """Extract one score per candidate, validating ranker/classifier output shapes."""
    import numpy as np
    for stage, prediction in prefix_probabilities(model, matrix, stages, is_ranker=is_ranker):
        values = np.asarray(prediction)
        if is_ranker:
            if values.ndim != 1:
                raise ValueError("LambdaRank predictions must have one-dimensional shape")
            scores = values
        else:
            if values.ndim != 2 or values.shape[1] != 2:
                raise ValueError("Binary classifier predictions must have two probability columns")
            scores = values[:, 1]
        if len(scores) != len(matrix):
            raise ValueError("Prediction count differs from candidate row count")
        if not np.isfinite(scores).all():
            raise ValueError("Model prefix produced a non-finite score")
        yield stage, scores


def top_indices(scores: Iterable[float], rrf_scores: Iterable[float], limit: int = 50):
    """Match Reranker.rank_batch: probability, RRF score, original pool order."""
    scores, rrf_scores = list(scores), list(rrf_scores)
    if len(scores) != len(rrf_scores):
        raise ValueError("Scores and candidate pool lengths differ")
    if any(not math.isfinite(float(value)) for value in scores + rrf_scores):
        raise ValueError("A reranking score is not finite")
    return sorted(range(len(scores)),
                  key=lambda index: (-scores[index], -rrf_scores[index], index))[:limit]


def score_group(group: dict, labels: list[int], local: list[bool], seen: list[bool] | None,
                baseline: list[bool], rrf_scores: list[float],
                predictions: dict[int, list[float]], limit: int = 50) -> dict:
    """Score a complete query pool; never infer its denominator from pool labels."""
    count = len(labels)
    if not (count == len(local) == len(baseline) == len(rrf_scores)):
        raise ValueError("Candidate columns have different lengths")
    if count != int(group["pool_size"]):
        raise ValueError(f"Incomplete pool for group {group['group_id']}: {count} rows")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Labels must be 0 or 1")
    if seen is not None and len(seen) != count:
        raise ValueError("Seen-positive column has a different length")
    positives = sum(labels)
    if positives != int(group["positives_in_pool"]):
        raise ValueError(f"Positive count mismatch for group {group['group_id']}")
    if positives > int(group["relevant_count"]):
        raise ValueError("Pool contains more positives than the corpus denominator")
    if sum(bool(value) for value in baseline) > limit:
        raise ValueError("Baseline has more than 50 candidates")
    baseline_hits = sum(label for label, selected in zip(labels, baseline) if selected)
    if baseline_hits != int(group["baseline_hits"]):
        raise ValueError(f"Baseline reproduction failed for group {group['group_id']}: "
                         f"{baseline_hits} != {group['baseline_hits']}")

    row = {key: group[key] for key in (
        "group_id", "query_text", "partition", "relevant_count", "pool_size",
        "positives_in_pool", "baseline_hits")}
    if "query_key" in group:
        row["query_key"] = list(group["query_key"])
    for denominator in SLICE_DENOMINATORS.values():
        if denominator in group:
            row[denominator] = int(group[denominator])
    for key in STRATUM_KEYS:
        if key in group:
            row[key] = bool(group[key])
    masks = {
        "all": [True] * count,
        "local": [bool(value) for value in local],
        "nonlocal": [not value for value in local],
    }
    if seen is not None:
        masks["seen"] = [bool(value) for value in seen]
        masks["unseen"] = [not value for value in seen]
    for kind, mask in masks.items():
        pool_hits = sum(label for label, present in zip(labels, mask) if present)
        denominator = row.get(SLICE_DENOMINATORS[kind])
        if denominator is not None and pool_hits > denominator:
            raise ValueError(f"Invalid {kind} denominator for group {group['group_id']}")
        row[f"pool_{kind}_hits"] = pool_hits
        row[f"baseline_{kind}_hits"] = sum(
            label for label, present, selected in zip(labels, mask, baseline)
            if present and selected)
    for stage, scores in predictions.items():
        selected = top_indices(scores, rrf_scores, limit)
        for kind, mask in masks.items():
            row[f"s{stage}_{kind}_hits"] = sum(labels[index] for index in selected
                                              if mask[index])
        denominator = int(group["relevant_count"])
        row[f"s{stage}_recall"] = (row[f"s{stage}_all_hits"] / denominator
                                    if denominator else None)
    return row


def summarize(rows: list[dict], stage: int) -> dict:
    """Macro retrieval metrics, conditional on a nonempty slice denominator."""
    slices = {}
    for kind, denominator_name in SLICE_DENOMINATORS.items():
        usable = [row for row in rows if row.get(denominator_name, 0) > 0]
        if not usable or any(f"s{stage}_{kind}_hits" not in row for row in usable):
            continue
        denominators = [row[denominator_name] for row in usable]
        total_positives = sum(denominators)
        result = {"queries_with_positives": len(usable), "corpus_positive_rows": total_positives}
        for source, prefix in (("reranker", f"s{stage}"),
                               ("baseline", "baseline"), ("pool", "pool")):
            hits = [row[f"{prefix}_{kind}_hits"] for row in usable]
            result[f"{source}_macro_recall"] = sum(
                hit / denominator for hit, denominator in zip(hits, denominators)) / len(usable)
            result[f"{source}_micro_recall"] = sum(hits) / total_positives
            result[f"{source}_positive_hits"] = sum(hits)
        slices[kind] = result
    return {"groups": len(rows),
            "query_texts": len({normalize_query_text(row["query_text"]) for row in rows}),
            "average_pool_size": sum(row["pool_size"] for row in rows) / len(rows) if rows else None,
            "maximum_pool_size": max((row["pool_size"] for row in rows), default=0),
            "slices": slices}


def check_group_metadata(groups: list[dict], *, validation: bool) -> None:
    partitions = {}
    for expected_id, group in enumerate(groups):
        if int(group["group_id"]) != expected_id:
            raise ValueError("Group metadata IDs must be ordered and contiguous from zero")
        if validation:
            partition = group["partition"]
            if partition not in {"tune", "confirmation"}:
                raise ValueError("Validation partition must be tune or confirmation")
            # A text may have multiple location/filter groups; all stay together.
            text = normalize_query_text(group["query_text"])
            previous = partitions.setdefault(text, partition)
            if previous != partition:
                raise ValueError(f"Query text occurs in both partitions: {text!r}")
        if int(group["relevant_count"]) < 1:
            raise ValueError("Recall audit groups must have corpus-accessible positives")
        for a, b in (("local_relevant_count", "nonlocal_relevant_count"),
                     ("seen_relevant_count", "unseen_relevant_count")):
            if a in group and b in group and int(group[a]) + int(group[b]) != int(group["relevant_count"]):
                raise ValueError(f"Slice denominators do not sum to total for group {expected_id}")


def benchmark_mix_metrics(rows: list[dict], stage: int, benchmark_strata: list[dict]) -> dict | None:
    """Match validation strata to benchmark query mix, without benchmark labels."""
    if not rows or not benchmark_strata or not all(key in rows[0] for key in STRATUM_KEYS):
        return None
    strata = defaultdict(list)
    for row in rows:
        strata[tuple(bool(row[key]) for key in STRATUM_KEYS)].append(row)
    target_queries = sum(int(entry["queries"]) for entry in benchmark_strata)
    if not target_queries:
        return None
    covered = 0
    values = {"reranker_macro_recall": 0.0, "baseline_macro_recall": 0.0,
              "pool_macro_recall": 0.0}
    for target in benchmark_strata:
        subset = strata.get(tuple(bool(target[key]) for key in STRATUM_KEYS), [])
        if not subset:
            continue
        weight = int(target["queries"])
        covered += weight
        metrics = summarize(subset, stage)["slices"]["all"]
        for key in values:
            values[key] += weight * metrics[key]
    return {"target_queries": target_queries, "target_queries_in_covered_strata": covered,
            "covered_target_fraction": covered / target_queries,
            **{key: value / covered if covered else None for key, value in values.items()},
            "note": "Weights use benchmark query attributes only; uncovered strata are excluded."}


def read_metadata(path: Path) -> dict:
    # Audit caches use validation.json; existing training caches use
    # train.parquet.json. Accept both without changing either producer.
    candidates = (path.with_suffix(path.suffix + ".json"), path.with_suffix(".json"))
    for metadata_path in candidates:
        if metadata_path.is_file():
            return json.loads(metadata_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Metadata not found: {candidates[0]} or {candidates[1]}")


def ensure_compatible(model_metadata: dict, cache_metadata: dict, model_name: str) -> None:
    if model_metadata.get("feature_names") != cache_metadata.get("feature_names"):
        raise ValueError(f"Feature names differ for {model_name}")
    for key in ("retrieval_config", "corpus_fingerprint", "split", "seed", "validation_fraction"):
        if key in model_metadata and key in cache_metadata and model_metadata[key] != cache_metadata[key]:
            raise ValueError(f"{key} differs between cache and {model_name}")


def ensure_prior_policy(artifact: dict, cache_metadata: dict, model_name: str,
                        *, fit_side: bool = False) -> dict:
    """Feature values must come from the same semantic inference histories."""
    from reranker_audit_provenance import artifact_prior_policy
    actual = artifact_prior_policy(artifact)
    for key, value in actual.items():
        if key not in cache_metadata:
            raise ValueError(f"Cache lacks required prior policy: {key}")
        if cache_metadata[key] != value:
            raise ValueError(f"{key} differs between cache and {model_name}")
    if fit_side:
        # New leave-query-out training policies alter fit-side features while
        # retaining the same inference histories. Their training caches cannot
        # be replaced by the old full-history fit-side cache.
        expected_training = artifact.get("metadata", {}).get("training_prior_policy")
        cached_training = cache_metadata.get("training_prior_policy")
        if expected_training != cached_training:
            raise ValueError(f"training_prior_policy differs between fit-side cache and {model_name}")
    return actual


def score_missing_zero_groups(groups: list[dict], first_id: int, next_observed_id: int,
                              stages: tuple[int, ...], *, has_seen: bool) -> list[dict]:
    """Metadata-only empty pools contribute zero Recall instead of disappearing."""
    if next_observed_id < first_id or next_observed_id > len(groups):
        raise ValueError("Observed cache group IDs are out of order")
    result = []
    for group_id in range(first_id, next_observed_id):
        if int(groups[group_id]["pool_size"]) != 0:
            raise ValueError(f"Cache omitted a nonempty pool: group_id={group_id}")
        result.append(score_group(groups[group_id], [], [], [] if has_seen else None,
                                  [], [], {stage: [] for stage in stages}))
    return result


def audit_cache(path: Path, metadata: dict, artifact: dict, stages: tuple[int, ...],
                batch_size: int) -> list[dict]:
    """Stream predictions in row batches, carrying one partial pool across batches."""
    import numpy as np
    import pyarrow.parquet as pq

    features = metadata["feature_names"]
    groups = metadata["groups"]
    check_group_metadata(groups, validation=path.stem == "validation")
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    required = set(features) | {"label", "group_id", "item_id", "positive_local"}
    if not required <= names:
        raise ValueError(f"Cache lacks columns: {sorted(required - names)}")
    if "baseline_top50" not in features:
        raise ValueError("baseline_top50 feature is required for reproduction checks")
    if any("seen_relevant_count" in group for group in groups) and "positive_seen" not in names:
        raise ValueError("Seen/unseen metadata requires the positive_seen row column")
    columns = features + ["label", "group_id", "item_id", "positive_local"]
    if "positive_seen" in names:
        columns.append("positive_seen")
    feature_indices = {name: index for index, name in enumerate(features)}
    records = []
    pending_id = None
    pending = None
    expected_id = 0

    def complete_pending():
        nonlocal expected_id
        if pending_id != expected_id:
            raise ValueError(f"Cache group order mismatch: {pending_id}, expected {expected_id}")
        if len(set(pending["items"])) != len(pending["items"]):
            raise ValueError(f"Duplicate item in query pool {pending_id}")
        records.append(score_group(groups[pending_id], pending["labels"], pending["local"],
                                   pending["seen"], pending["baseline"], pending["rrf"],
                                   pending["scores"]))
        expected_id += 1

    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        matrix = np.column_stack([batch.column(name).to_numpy(zero_copy_only=False)
                                  for name in features]).astype(np.float32, copy=False)
        if artifact.get("disable_microcategory_prior"):
            for name in ("microcat_token_probability", "microcat_token_rank"):
                if name not in feature_indices:
                    raise ValueError("Disabled prior is absent from feature schema")
                matrix[:, feature_indices[name]] = 0
        predictions = dict(prefix_scores(artifact["model"], matrix, stages,
                                         is_ranker=is_ranking_artifact(artifact)))
        group_ids = batch.column("group_id").to_numpy(zero_copy_only=False)
        boundaries = np.concatenate(([0], np.flatnonzero(group_ids[1:] != group_ids[:-1]) + 1,
                                     [len(group_ids)]))
        labels = batch.column("label").to_numpy(zero_copy_only=False)
        local = batch.column("positive_local").to_numpy(zero_copy_only=False)
        seen = (batch.column("positive_seen").to_numpy(zero_copy_only=False)
                if "positive_seen" in columns else None)
        items = batch.column("item_id").to_pylist()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            group_id = int(group_ids[start])
            if group_id < 0 or group_id >= len(groups):
                raise ValueError(f"Group ID {group_id} has no metadata")
            if pending_id != group_id:
                if pending_id is not None:
                    complete_pending()
                records.extend(score_missing_zero_groups(
                    groups, expected_id, group_id, stages, has_seen=seen is not None))
                expected_id = group_id
                pending_id = group_id
                pending = {"items": [], "labels": [], "local": [],
                           "seen": [] if seen is not None else None,
                           "baseline": [], "rrf": [], "scores": {stage: [] for stage in stages}}
            pending["items"].extend(items[start:end])
            pending["labels"].extend(labels[start:end].tolist())
            pending["local"].extend(local[start:end].tolist())
            if seen is not None:
                pending["seen"].extend(seen[start:end].tolist())
            pending["baseline"].extend((matrix[start:end, feature_indices["baseline_top50"]] > 0).tolist())
            pending["rrf"].extend(matrix[start:end, feature_indices["rrf_score"]].tolist())
            for stage in stages:
                pending["scores"][stage].extend(predictions[stage][start:end].tolist())
        if len(records) and len(records) // 250 != (len(records) - len(boundaries) + 1) // 250:
            print(f"{path.stem}: {len(records)}/{len(groups)} complete groups", flush=True)
    if pending_id is not None:
        complete_pending()
    records.extend(score_missing_zero_groups(
        groups, expected_id, len(groups), stages, has_seen="positive_seen" in columns))
    expected_id = len(groups)
    if expected_id != len(groups):
        raise ValueError(f"Cache has {expected_id} groups, metadata has {len(groups)}")
    return records


def sampled_train_losses(path: Path, artifact: dict, stages: tuple[int, ...], batch_size: int) -> dict:
    """Optional row loss on clicked versus sampled unobserved training candidates."""
    if is_ranking_artifact(artifact):
        return {"applicable": False,
                "meaning": "LambdaRank emits raw ranking scores, not binary probabilities; binary log loss is not applicable."}
    import numpy as np
    import pyarrow.parquet as pq

    features = artifact["metadata"]["feature_names"]
    parquet = pq.ParquetFile(path)
    has_weight = "weight" in parquet.schema_arrow.names
    columns = features + ["label"] + (["weight"] if has_weight else [])
    sums = {stage: {"loss": 0.0, "weighted_loss": 0.0} for stage in stages}
    row_count, weight_sum = 0, 0.0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        matrix = np.column_stack([batch.column(name).to_numpy(zero_copy_only=False)
                                  for name in features]).astype(np.float32, copy=False)
        if artifact.get("disable_microcategory_prior"):
            for name in ("microcat_token_probability", "microcat_token_rank"):
                matrix[:, features.index(name)] = 0
        labels = batch.column("label").to_numpy(zero_copy_only=False)
        weights = (batch.column("weight").to_numpy(zero_copy_only=False)
                   if has_weight else np.ones(len(labels), dtype=np.float64))
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Training labels are not binary")
        for stage, scores in prefix_scores(artifact["model"], matrix, stages):
            probabilities = np.clip(scores, 1e-12, 1 - 1e-12)
            losses = -(labels * np.log(probabilities) + (1 - labels) * np.log1p(-probabilities))
            sums[stage]["loss"] += float(losses.sum())
            sums[stage]["weighted_loss"] += float(np.dot(weights, losses))
        row_count += len(labels)
        weight_sum += float(weights.sum())
    if not row_count or not weight_sum:
        raise ValueError("Sampled training file is empty or has zero total weight")
    return {"applicable": True, "rows": row_count, "total_weight": weight_sum,
            "meaning": "Binary loss on clicked versus sampled unobserved rows; this is not Recall or relevance accuracy.",
            "stages": {str(stage): {"unweighted_log_loss": values["loss"] / row_count,
                                    "weighted_log_loss": values["weighted_loss"] / weight_sum}
                       for stage, values in sums.items()}}


def write_scores(path: Path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def stage_report(rows: list[dict], stages: tuple[int, ...], metadata: dict,
                 *, fit_side_sample: bool = False) -> dict:
    partitions = {name: [row for row in rows if row["partition"] == name]
                  for name in ("tune", "confirmation")}
    report = {}
    for stage in stages:
        stage_value = {"all_groups": summarize(rows, stage)}
        if fit_side_sample:
            eligible = [row for row in rows if row["positives_in_pool"] > 0]
            ineligible = [row for row in rows if row["positives_in_pool"] == 0]
            stage_value["sample_description"] = "Fit-side sample; not every sampled group was eligible for reranker fitting."
            stage_value["reranker_training_eligible_groups"] = summarize(eligible, stage)
            stage_value["groups_without_retrieved_training_positive"] = summarize(ineligible, stage)
        matched = benchmark_mix_metrics(rows, stage, metadata.get("benchmark_strata", []))
        if matched is not None:
            stage_value["benchmark_mix"] = matched
        for partition, subset in partitions.items():
            if subset:
                stage_value[partition] = summarize(subset, stage)
                matched = benchmark_mix_metrics(subset, stage, metadata.get("benchmark_strata", []))
                if matched is not None:
                    stage_value[f"{partition}_benchmark_mix"] = matched
        report[str(stage)] = stage_value
    return report


def maximum_iterations(model: Any) -> int:
    if callable(getattr(model, "staged_predict_proba", None)):
        return int(model.n_iter_)
    if hasattr(model, "booster_"):
        return int(model.booster_.current_iteration())
    raise TypeError(f"Unsupported model class: {type(model).__name__}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_model(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not path:
        raise argparse.ArgumentTypeError("Model must be NAME=PATH; NAME uses letters, numbers, _ or -")
    return name, Path(path)


def audit_cache_paths(cache_dir: Path, *, validation_only: bool = False) -> dict[str, Path]:
    """Fit-side cache is omitted only by an explicit evaluation mode."""
    roles = ("validation",) if validation_only else ("train_sample", "validation")
    return {name: cache_dir / f"{name}.parquet" for name in roles}


def add_fit_side_gap_report(model_report: dict, stages: tuple[int, ...]) -> None:
    """Missing comparable fit-side data is explicit, never borrowed from legacy."""
    validation = model_report["datasets"]["validation"]
    fit = model_report["datasets"].get("train_sample")
    model_report["fit_side_sample_available"] = fit is not None
    if fit is None:
        model_report["fit_side_sample_note"] = (
            "Validation-only audit: no comparable fit-side sample was supplied. "
            "Legacy full-history samples cannot measure a model fitted with a different "
            "training-prior policy. Optional fitted-training row loss uses the model's "
            "own --train-data and is not retrieval Recall.")
        model_report["train_minus_validation_recall_gap"] = {}
        model_report["eligible_fit_groups_minus_validation_recall_gap"] = {}
        model_report["train_minus_validation_gap_definition"] = "Not computed: no comparable fit-side sample."
        return
    model_report["fit_side_sample_note"] = (
        "Fit-side sample includes groups skipped by reranker fitting when no positive "
        "was retrieved. The embedding encoder was trained on fit-side texts; gaps "
        "alone are not proof of reranker overfitting.")
    model_report["train_minus_validation_recall_gap"] = {
        str(stage): fit[str(stage)]["all_groups"]["slices"]["all"]["reranker_macro_recall"]
        - validation[str(stage)]["all_groups"]["slices"]["all"]["reranker_macro_recall"]
        for stage in stages}
    model_report["train_minus_validation_gap_definition"] = "Legacy field name: overall fit-side sample minus validation Recall, including fit-side groups ineligible for reranker fitting. Not evidence of overfitting by itself."
    model_report["eligible_fit_groups_minus_validation_recall_gap"] = {}
    for stage in stages:
        value = (fit[str(stage)]["reranker_training_eligible_groups"]["slices"]
                 .get("all", {}).get("reranker_macro_recall"))
        model_report["eligible_fit_groups_minus_validation_recall_gap"][str(stage)] = (
            value - validation[str(stage)]["all_groups"]["slices"]["all"]["reranker_macro_recall"]
            if value is not None else None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--validation-only", action="store_true",
                        help="Evaluate validation without a fit-side cache; default requires both caches")
    parser.add_argument("--train-data", type=Path,
                        help="Sampled training parquet; read only with --sampled-train-logloss")
    parser.add_argument("--model", type=parse_model, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", type=int, default=list(DEFAULT_STAGES))
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--sampled-train-logloss", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if len({name for name, _ in args.model}) != len(args.model):
        parser.error("Model names must be unique")
    if args.sampled_train_logloss and args.train_data is None:
        parser.error("--sampled-train-logloss requires --train-data")

    import joblib
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = audit_cache_paths(args.cache_dir, validation_only=args.validation_only)
    for path in cache_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required audit cache is missing: {path}. Use --validation-only explicitly to omit the fit-side cache.")
    metadata = {name: read_metadata(path) for name, path in cache_paths.items()}
    mandatory_prior_keys = {"inference_priors_sha256", "disable_microcategory_prior"}
    if any(not mandatory_prior_keys <= value.keys() for value in metadata.values()):
        # The cache finalizer verifies completed Parquet footers and the exact
        # producer model hash before annotating manifests. It never changes rows.
        from reranker_audit_provenance import finalize_cache_provenance
        result = finalize_cache_provenance(args.cache_dir, validation_only=args.validation_only)
        print(f"Finalized cache inference provenance: {result['inference_priors_sha256']}", flush=True)
        metadata = {name: read_metadata(path) for name, path in cache_paths.items()}
    for name, value in metadata.items():
        check_group_metadata(value["groups"], validation=name == "validation")
    if "train_sample" in metadata and set(normalize_query_text(group["query_text"]) for group in metadata["train_sample"]["groups"]) & set(
            normalize_query_text(group["query_text"]) for group in metadata["validation"]["groups"]):
        raise ValueError("Training sample and validation query texts overlap")
    report_paths = {}
    selections = {}
    for name, model_path in args.model:
        print(f"Auditing {name}: {model_path}", flush=True)
        artifact = joblib.load(model_path)
        if not isinstance(artifact, dict) or "model" not in artifact or "metadata" not in artifact:
            raise ValueError("Expected saved reranker artifact with model and metadata")
        for value in metadata.values():
            ensure_compatible(artifact["metadata"], value, name)
        policies = {dataset: ensure_prior_policy(artifact, value, name,
                                                fit_side=dataset == "train_sample")
                    for dataset, value in metadata.items()}
        if args.train_data is not None:
            ensure_compatible(artifact["metadata"], read_metadata(args.train_data), name)
        ranker = is_ranking_artifact(artifact)
        if not ranker:
            classes = list(getattr(artifact["model"], "classes_", [0, 1]))
            if classes != [0, 1]:
                raise ValueError("Model probability columns must correspond to classes [0, 1]")
        stages = model_stages(maximum_iterations(artifact["model"]), args.stages)
        model_report = {"model": str(model_path), "sha256": sha256_file(model_path),
                        "model_class": type(artifact["model"]).__name__, "stages": list(stages),
                        "prefix_score_type": "raw_ranking_score" if ranker else "positive_class_probability",
                        "inference_prior_policy": policies["validation"],
                        "historical_holdout_reused": True,
                        "validation_only": args.validation_only,
                        "interpretation": "Recall uses complete fixed pools and corpus-positive denominators. When a compatible fit-side sample is supplied, it includes groups skipped by reranker fitting because no positive was retrieved. The embedding encoder was fitted on fit-side texts. A fit-side/validation gap alone does not prove overfitting. Tune and confirmation texts are separated now, but the historical holdout has previously been inspected.",
                        "retrieval_config": artifact["metadata"]["retrieval_config"],
                        "benchmark_strata": metadata["validation"].get("benchmark_strata", []),
                        "datasets": {}, "per_query_scores": {}}
        for field in ("candidate_pool_digest", "training_prior_policy"):
            if field in metadata["validation"]:
                model_report[field] = metadata["validation"][field]
        if "training_prior_policy" in artifact["metadata"]:
            model_report["model_training_prior_policy"] = artifact["metadata"]["training_prior_policy"]
        for dataset, path in cache_paths.items():
            rows = audit_cache(path, metadata[dataset], artifact, stages, args.batch_size)
            score_path = args.output_dir / f"{name}-{dataset}-scores.parquet"
            write_scores(score_path, rows)
            model_report["per_query_scores"][dataset] = str(score_path)
            model_report["datasets"][dataset] = stage_report(
                rows, stages, metadata[dataset], fit_side_sample=dataset == "train_sample")
            del rows
        validation = model_report["datasets"]["validation"]
        if any("tune" not in validation[str(stage)] for stage in stages):
            raise ValueError("No tuning groups are available")
        chosen_stage = min(stages, key=lambda stage: (
            -validation[str(stage)]["tune"]["slices"]["all"]["reranker_macro_recall"], stage))
        chosen = validation[str(chosen_stage)]
        model_report["selection"] = {
            "stage_selected_on_tuning_texts": chosen_stage,
            "selection_metric": "tune macro Recall@50 over complete pools",
            "tune": chosen["tune"], "confirmation": chosen.get("confirmation"),
            "confirmation_caveat": "Historical validation was previously used for experiments; this is not a new untouched holdout."}
        add_fit_side_gap_report(model_report, stages)
        if hasattr(artifact["model"], "booster_"):
            booster = artifact["model"].booster_
            importance = booster.feature_importance(importance_type="gain", iteration=chosen_stage)
            model_report["top_feature_importances_gain"] = sorted(
                ({"feature": feature, "gain": float(value)}
                 for feature, value in zip(artifact["metadata"]["feature_names"], importance)),
                key=lambda entry: -entry["gain"])[:25]
        if args.sampled_train_logloss:
            model_report["sampled_training_row_loss"] = sampled_train_losses(
                args.train_data, artifact, stages, args.batch_size)
        path = args.output_dir / f"{name}-report.json"
        path.write_text(json.dumps(model_report, ensure_ascii=False, indent=2), encoding="utf-8")
        report_paths[name] = str(path)
        selections[name] = model_report["selection"]
        final_recall = validation[str(stages[-1])]["all_groups"]["slices"]["all"]["reranker_macro_recall"]
        confirmation = chosen.get("confirmation", {}).get("slices", {}).get("all", {}).get("reranker_macro_recall")
        print(f"{name}: final all-validation Recall@50={final_recall:.6f}; "
              f"tune-selected stage={chosen_stage}; confirmation={confirmation}", flush=True)
        del artifact
    summary = {"reports": report_paths, "selections": selections,
               "validation_only": args.validation_only,
               "historical_holdout_reused": True,
               "notes": ["Recall denominators include all accessible corpus positives, including those absent from the pool.",
                         "Local/nonlocal and seen/unseen macro metrics include only groups with positives in that slice.",
                         "Every compared prefix uses the same candidate pools; baseline hits are independently reproduced.",
                         "Per-query Parquet files preserve query_text for paired, text-cluster bootstrap comparisons."]}
    (args.output_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
