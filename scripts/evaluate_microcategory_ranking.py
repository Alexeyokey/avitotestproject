"""Explore soft microcategory reranking on the existing fresh Tune pools.

No new candidates, features, estimator fitting or encoder calls occur here.
The frozen HGB prefix selected by the earlier Tune protocol is combined with
a separate train-only microcategory classifier. Every original pool remains
unchanged. Alpha zero reproduces the completed HGB outcome for EVERY group,
including its probability/RRF/original-pool-order tie rule.

Preregistered scores: logit(HGB probability) + alpha*log(max(class score,1e-5)),
alpha in (0,.05,.1,.25). Alpha zero uses the original probabilities directly,
so clipping/logarithms cannot change the reference tie order. This is an
exploratory Tune grid, not an independently validated improvement. Audit
labels/reports are not opened or used for alpha selection.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import inspect
import math
from pathlib import Path
import random
import sys
import time

import evaluate_fresh_validation as EVAL
import evaluate_retrieval_hypotheses as RETRIEVAL


ALPHAS = (0., .05, .1, .25)
MICRO_FLOOR = 1e-5
LOGIT_EPSILON = 1e-12
POLICY = {"version": 1, "phase": "tune", "alphas": list(ALPHAS), "limit": 50,
          "score": "logit(HGB probability) + alpha*log(max(microcategory classifier score,1e-5))",
          "hgb_logit_clip": LOGIT_EPSILON, "microcategory_floor": MICRO_FLOOR,
          "alpha_zero": "original HGB probability, RRF score, original-pool-order tie rule",
          "selection_metric": "conditional_available_text_macro_recall_at_50",
          "tie_rule": "smallest alpha within1e-12 of the maximum",
          "bootstrap_unit": "normalized query text", "bootstrap_replicates": 2000,
          "bootstrap_seed": 20261013,
          "scope": "exploratory Tune grid after the HGB capacity was chosen; no independent Audit selection"}


def alpha_key(alpha):
    return "a" + format(float(alpha), ".8g").replace(".", "p")


def selected_hgb_scores(model, matrix, stage):
    """Use the equivalent final predictor when possible, avoiding150 links/batch."""
    import numpy as np
    if stage == EVAL.maximum_iterations(model) and callable(getattr(model, "predict_proba", None)):
        prediction = np.asarray(model.predict_proba(matrix))
        if prediction.shape != (len(matrix), 2):
            raise ValueError("Frozen HGB must return two probability columns")
        scores = prediction[:, 1]
    else:
        scores = dict(EVAL.prefix_scores(model, matrix, (stage,)))[stage]
    if (len(scores) != len(matrix) or not np.isfinite(scores).all()
            or np.any(scores < 0) or np.any(scores > 1)):
        raise ValueError("Frozen HGB produced invalid class probabilities")
    return scores


def full_category_probabilities(classifier, query):
    """Infer only cold contexts and discard the all-class prediction cache.

    Keeping209 classes for every12k context would needlessly inflate memory.
    This classifier instance is dedicated to the experiment; clearing its
    disposable prediction cache cannot alter model weights or provenance.
    """
    from avito_candidates.core import normalize
    text = normalize(query.get("search_query", "") or "")
    if text in classifier.model.fitted_texts:
        raise ValueError("Cold Tune ranking must not use a classifier fitted to its own text")
    classes = {str(value) for value in classifier.model.classes}
    if not classes:
        raise ValueError("Microcategory classifier has no classes")
    predicted = classifier.predict_categories_batch([query], top_k=len(classes), exclude_self=False)[0]
    classifier._prediction_cache.clear()
    if (len(predicted) != len(classes) or len({name for name, _ in predicted}) != len(classes)
            or {name for name, _ in predicted} != classes
            or any(not math.isfinite(score) or not 0 <= score <= 1 for _, score in predicted)
            or abs(sum(score for _, score in predicted) - 1) > 1e-5):
        raise ValueError("Microcategory predictions must cover every class with normalized finite scores")
    return dict(predicted)


def adjusted_scores(hgb, categories, probabilities, alpha):
    """Apply the declared scoring rule without reading relevance labels."""
    if (len(hgb) != len(categories) or alpha not in ALPHAS
            or any(not math.isfinite(float(score)) or not 0 <= score <= 1 for score in hgb)):
        raise ValueError("Microcategory ranking shape, HGB probabilities or alpha is invalid")
    if alpha == 0:
        return list(hgb)
    output = []
    for score, category in zip(hgb, categories):
        category_score = probabilities.get(category, 0.)
        if not math.isfinite(category_score) or not 0 <= category_score <= 1:
            raise ValueError("Invalid microcategory class score")
        clipped = min(1 - LOGIT_EPSILON, max(LOGIT_EPSILON, float(score)))
        output.append(math.log(clipped / (1 - clipped))
                      + alpha * math.log(max(MICRO_FLOOR, category_score)))
    return output


def score_group(group, raw, expected, categories_by_id, probabilities):
    items, labels, hgb, rrf = (raw[name] for name in ("items", "labels", "hgb", "rrf"))
    if (len(items) != group["pool_size"] or len(set(items)) != len(items)
            or not all(len(values) == len(items) for values in (labels, hgb, rrf, raw["baseline"]))
            or any(item not in categories_by_id for item in items)):
        raise ValueError("Microcategory ranking requires an unchanged complete original pool")
    expected_labels = [int(item in group["_available_ids"]) for item in items]
    if (labels != expected_labels or sum(labels) != group["positives_in_pool"]
            or sum(label for label, chosen in zip(labels, raw["baseline"]) if chosen) != group["baseline_hits"]):
        raise ValueError("Saved pool labels or original RRF hits differ from sealed Tune GT")
    row = {key: value for key, value in group.items() if not key.startswith("_")}
    row.update(pool_hits=sum(labels), baseline_hits=group["baseline_hits"])
    categories = [categories_by_id[item] for item in items]
    for alpha in ALPHAS:
        scores = adjusted_scores(hgb, categories, probabilities, alpha)
        selected = EVAL.top_indices(scores, rrf, POLICY["limit"])
        row[f"{alpha_key(alpha)}_hits"] = sum(labels[index] for index in selected)
        row[f"{alpha_key(alpha)}_selected_count"] = len(selected)
    if row["a0_hits"] != expected["hgb_hits"]:
        raise ValueError(f"Alpha-zero HGB reproduction differs for group{group['group_id']}")
    if row["pool_hits"] != expected["pool_hits"] or row["baseline_hits"] != expected["baseline_hits"]:
        raise ValueError("Original pool/RRF reproduction differs from the completed Tune outcomes")
    return row


def read_completed_tune(evaluation_dir, groups, metadata, model_hash, protocol_hash, train_hash):
    """Read only the completed Tune seal/outcomes; never the Audit report."""
    root = Path(evaluation_dir)
    selection_path = root / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    stage = EVAL.verify_selection(selection, protocol_hash, model_hash, root, train_hash)
    report_path = root / "tune-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("phase") != "tune" or report.get("report_version") != 1
            or report.get("selected_stage") != stage or report.get("policy") != EVAL.POLICY
            or report.get("cache_sha256", {}).get("parquet") != metadata["cache_parquet_sha256"]):
        raise ValueError("Completed Tune report differs from the fixed cache/selection")
    scores_path = root / "tune-group-scores.jsonl"
    if EVAL.sha256_file(scores_path) != report.get("group_scores_sha256"):
        raise ValueError("Completed Tune group scores are missing or changed")
    outcomes = []
    with scores_path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= len(groups):
                raise ValueError("Completed Tune scores contain extra groups")
            row, group = json.loads(line), groups[index]
            if any(row.get(name) != group.get(name) for name in (
                    "group_id", "group_identity", "query_key", "query_text", "pool_size",
                    "full_relevant_count", "available_relevant_count", "representative")):
                raise ValueError("Completed Tune group identity differs from the saved cache")
            hits = row.get(f"s{stage}_hits")
            if (isinstance(hits, bool) or not isinstance(hits, int)
                    or not 0 <= hits <= group["available_relevant_count"]):
                raise ValueError("Completed selected HGB hits are invalid")
            outcomes.append({"hgb_hits": hits, "pool_hits": row["pool_hits"],
                             "baseline_hits": row["baseline_hits"]})
    if len(outcomes) != len(groups):
        raise ValueError("Completed Tune scores omitted groups")
    return stage, outcomes, report, {"selection": EVAL.sha256_file(selection_path),
        "report": EVAL.sha256_file(report_path), "group_scores": EVAL.sha256_file(scores_path)}


def stream_scores(path, metadata, groups, artifact, classifier, categories_by_id,
                  completed, stage, selected_ids, *, batch_size=16384):
    """Predict in row batches, carry one pool, infer class scores only at flush."""
    import numpy as np
    import pyarrow.parquet as pq
    from avito_candidates.core import SEARCH_FIELDS
    features = metadata["feature_names"]
    positions = {name: index for index, name in enumerate(features)}
    columns = [*features, "group_id", "item_id", "label"]
    parquet = pq.ParquetFile(path)
    if batch_size < 1 or not set(columns) <= set(parquet.schema_arrow.names):
        raise ValueError("Candidate scoring columns or batch size are invalid")
    rows, pending, pending_id, next_id, observed = [], None, None, 0, 0
    started = time.monotonic()

    def make_row(group, raw):
        if group["group_id"] not in selected_ids:
            return
        query = dict(zip(SEARCH_FIELDS, group["query_key"]))
        probabilities = full_category_probabilities(classifier, query) if raw["items"] else {}
        rows.append(score_group(group, raw, completed[group["group_id"]], categories_by_id, probabilities))
        if len(rows) % 512 == 0:
            print(f"Microcategory Tune ranking groups:{len(rows)}/{len(selected_ids)}; "
                  f"elapsed={time.monotonic() - started:.1f}s", file=sys.stderr, flush=True)

    def empty_before(stop):
        nonlocal next_id
        while next_id < stop:
            group = groups[next_id]
            if group["pool_size"]:
                raise ValueError("Cache omitted a nonempty original pool")
            make_row(group, {"items": [], "labels": [], "hgb": [], "rrf": [], "baseline": []})
            next_id += 1

    def flush():
        nonlocal next_id
        if pending_id != next_id:
            raise ValueError("Original pools are not contiguous and ordered")
        if pending is not None:
            make_row(groups[pending_id], pending)
        next_id += 1

    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        ids = batch.column("group_id").to_numpy(zero_copy_only=False)
        if (not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0)
                or np.any(ids >= len(groups)) or np.any(ids[1:] < ids[:-1])):
            raise ValueError("Original candidate group IDs are invalid")
        # Score only selected rows during a cheap screen. Group IDs stay in
        # original order, so the same pools cross batch boundaries unchanged.
        wanted = np.isin(ids, list(selected_ids))
        matrix = np.column_stack([batch.column(name).to_numpy(zero_copy_only=False)
                                  for name in features]).astype(np.float32, copy=False)
        if not np.isfinite(matrix).all():
            raise ValueError("Original HGB features are not finite")
        if artifact.get("disable_microcategory_prior"):
            for name in ("microcat_token_probability", "microcat_token_rank"):
                matrix[:, positions[name]] = 0
        predicted = np.empty(len(matrix), dtype=np.float64)
        predicted[wanted] = selected_hgb_scores(artifact["model"], matrix[wanted], stage) if np.any(wanted) else []
        items, labels = (batch.column(name).to_pylist() for name in ("item_id", "label"))
        cuts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
        for start, end in zip(cuts[:-1], cuts[1:]):
            group_id = int(ids[start])
            if group_id != pending_id:
                if pending_id is not None:
                    flush()
                empty_before(group_id)
                if group_id != next_id:
                    raise ValueError("Original groups repeat or are out of order")
                pending_id = group_id
                pending = ({"items": [], "labels": [], "hgb": [], "rrf": [], "baseline": []}
                           if group_id in selected_ids else None)
            if pending is not None:
                left, right = int(start), int(end)
                pending["items"].extend(items[left:right])
                pending["labels"].extend(labels[left:right])
                pending["hgb"].extend(predicted[left:right].tolist())
                pending["rrf"].extend(matrix[left:right, positions["rrf_score"]].tolist())
                pending["baseline"].extend((matrix[left:right, positions["baseline_top50"]] > 0).tolist())
        observed += len(ids)
    if pending_id is not None:
        flush()
    empty_before(len(groups))
    if observed != metadata["rows"] or len(rows) != len(selected_ids):
        raise ValueError("Original row or selected-group counts differ from the sealed cache")
    return rows


def paired_bootstrap(rows, key, *, replicates=2000, seed=20261013):
    by_text = defaultdict(list)
    for row in rows:
        denominator = row["available_relevant_count"]
        if denominator:
            by_text[EVAL.normalize_query_text(row["query_text"])].append(
                (row[f"{key}_hits"] - row["a0_hits"]) / denominator)
    if not by_text or replicates < 1:
        raise ValueError("Microcategory bootstrap needs conditional-available texts")
    deltas = [sum(values) / len(values) for values in by_text.values()]
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(replicates))
    return {"comparison": f"{key} minus frozen selected HGB(alpha0)",
            "metric": POLICY["selection_metric"], "query_text_clusters": len(deltas),
            "replicates": replicates, "seed": seed, "delta": sum(deltas) / len(deltas),
            "percentile_95_interval": [draws[int(.025 * (replicates - 1))], draws[int(.975 * (replicates - 1))]],
            "scope": "Exploratory Tune alpha grid; no independent confidence after choosing an alpha"}


def summarize(rows, *, bootstrap_replicates=2000):
    if not rows:
        raise ValueError("Microcategory ranking has no Tune groups")
    summaries = {}
    for alpha in ALPHAS:
        key = alpha_key(alpha)
        result = {"alpha": alpha, "conditional_available": EVAL.aggregate(
            rows, f"{key}_hits", "available_relevant_count"), "full_truth": EVAL.aggregate(
            rows, f"{key}_hits", "full_relevant_count"), "paired_vs_hgb": paired_bootstrap(
            rows, key, replicates=bootstrap_replicates), "slices": {}}
        for field in ("one_word", "has_filters", "no_exact_location"):
            for value in (False, True):
                subset = [row for row in rows if bool(row.get(field)) == value]
                if subset:
                    result["slices"][f"{field}={str(value).lower()}"] = {
                        "variant": EVAL.aggregate(subset, f"{key}_hits", "available_relevant_count"),
                        "frozen_hgb": EVAL.aggregate(subset, "a0_hits", "available_relevant_count")}
        summaries[key] = result
    metric = POLICY["selection_metric"].removeprefix("conditional_available_")
    scores = {key: report["conditional_available"][metric] for key, report in summaries.items()}
    if any(value is None or not math.isfinite(value) for value in scores.values()):
        raise ValueError("Alpha comparison requires available target texts")
    best = max(scores.values())
    winner = min(alpha for alpha in ALPHAS if abs(scores[alpha_key(alpha)] - best) <= 1e-12)
    return {"selected_alpha_on_tune": winner, "variants": summaries,
            "reference_rrf": {"conditional_available": EVAL.aggregate(rows, "baseline_hits", "available_relevant_count"),
                              "full_truth": EVAL.aggregate(rows, "baseline_hits", "full_relevant_count")},
            "pool": {"conditional_available": RETRIEVAL.recall_summary(rows, "pool_hits", "available_relevant_count", candidate=True),
                     "full_truth": RETRIEVAL.recall_summary(rows, "pool_hits", "full_relevant_count", candidate=True),
                     "average_size": sum(row["pool_size"] for row in rows) / len(rows),
                     "maximum_size": max(row["pool_size"] for row in rows)},
            "coverage": {"groups": len(rows), "texts": len({row["query_text"] for row in rows}),
                "groups_without_available_target": sum(row["available_relevant_count"] == 0 for row in rows),
                "available_relevant_pairs": sum(row["available_relevant_count"] for row in rows),
                "full_relevant_pairs": sum(row["full_relevant_count"] for row in rows)}}


def evaluate(args):
    scores_path = args.output.with_suffix(".groups.jsonl")
    if args.output.exists() or scores_path.exists():
        raise ValueError("Microcategory ranking output already exists; choose distinct paths")
    manifest, protocol_hash = EVAL.checked_protocol(args.protocol_dir)
    model_hash = EVAL.sha256_file(args.model)
    import joblib
    artifact = joblib.load(args.model)
    if artifact.get("estimator", "histgb") != "histgb" or artifact.get("selected_iteration") is not None:
        raise ValueError("Use the raw frozen HGB artifact referenced by the completed Tune cache")
    EVAL.check_encoder_identity(artifact, manifest)
    train_hash = EVAL.check_training_data(args.train_data, artifact, args.protocol_dir, manifest)
    cache_path, metadata_path = EVAL.phase_cache_paths(args.cache_dir, "tune")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    EVAL.ensure_compatible(artifact["metadata"], metadata, "frozen HGB microcategory experiment")
    prior_policy = EVAL.ensure_prior_policy(artifact, metadata, "frozen HGB microcategory experiment")
    cache_hash = EVAL.check_phase_cache_inputs(cache_path, metadata, "tune", protocol_hash, model_hash)
    stage, completed, reference_report, reference_hashes = read_completed_tune(
        args.evaluation_dir, metadata["groups"], metadata, model_hash, protocol_hash, train_hash)
    catalog_path = EVAL.protocol_file(args.protocol_dir, manifest, "benchmark-items.json")
    import pyarrow.parquet as pq
    rows = pq.read_table(catalog_path, columns=["item_id", "item_microcat_id"]).to_pylist()
    categories = {row["item_id"]: str(row.get("item_microcat_id") or "") for row in rows}
    if len(categories) != len(rows) or len(categories) != manifest["corpus"]["size"]:
        raise ValueError("Research item/microcategory mapping is incomplete or repeated")
    fit_path = EVAL.protocol_file(args.protocol_dir, manifest, "fit-texts.json")
    fit_texts = set(json.loads(fit_path.read_text(encoding="utf-8"))["texts"])
    from avito_candidates.microcategory import SoftMicrocategoryClassifier
    classifier = SoftMicrocategoryClassifier.load(args.microcategory_model,
        allowed_fit_texts=fit_texts, corpus_fingerprint=metadata["corpus_fingerprint"],
        source_sha256=manifest["files"]["train"]["sha256"])
    if any(EVAL.normalize_query_text(group["query_text"]) in fit_texts for group in metadata["groups"]):
        raise ValueError("Tune contains a query permitted in classifier training")
    classifier_hash = EVAL.sha256_file(args.microcategory_model)
    classifier_source_path = Path(inspect.getfile(SoftMicrocategoryClassifier))
    classifier_source_hash = EVAL.sha256_file(classifier_source_path)
    classifier_metadata_path = args.microcategory_model.with_suffix(args.microcategory_model.suffix + ".json")
    classifier_metadata_hash = EVAL.sha256_file(classifier_metadata_path)
    before = {"model": model_hash, "cache_metadata": EVAL.sha256_file(metadata_path),
              "microcategory_model": classifier_hash, "microcategory_metadata": classifier_metadata_hash,
              "microcategory_code": classifier_source_hash,
              "evaluator": EVAL.sha256_file(Path(__file__))}
    # All label/byte/provenance guards precede parsing the Tune relevance arm.
    labels = EVAL.read_phase_labels(args.protocol_dir, manifest, "tune")
    groups = EVAL.bind_cache_groups(metadata, labels, "tune", protocol_hash, model_hash)
    texts = RETRIEVAL.choose_texts(groups, args.text_limit)
    selected_ids = {group["group_id"] for group in groups if group["query_text"] in texts}
    scored = stream_scores(cache_path, metadata, groups, artifact, classifier, categories, completed,
                           stage, selected_ids, batch_size=args.batch_size)
    result = summarize(scored, bootstrap_replicates=args.bootstrap_replicates)
    if args.text_limit == 0:
        for scope in ("conditional_available", "full_truth"):
            reproduced = result["variants"]["a0"][scope]
            reference = reference_report["stages"][str(stage)][scope]["reranker"]
            if any(reproduced[key] != reference.get(key) for key in reproduced):
                raise ValueError("Alpha-zero full-Tune metrics differ from the completed selected HGB")
    closure = {"model": EVAL.sha256_file(args.model), "cache_metadata": EVAL.sha256_file(metadata_path),
               "microcategory_model": EVAL.sha256_file(args.microcategory_model),
               "microcategory_metadata": EVAL.sha256_file(classifier_metadata_path),
               "microcategory_code": EVAL.sha256_file(classifier_source_path),
               "evaluator": EVAL.sha256_file(Path(__file__))}
    if closure != before or EVAL.sha256_file(args.protocol_dir / "manifest.json") != protocol_hash:
        raise ValueError("Microcategory ranking input bytes changed during scoring")
    report = {"report_version": 1, "phase": "tune", "complete": True, "policy": POLICY,
        "created_utc": datetime.now(timezone.utc).isoformat(), "selected_hgb_stage": stage,
        "protocol_sha256": protocol_hash, "cache_sha256": {"parquet": cache_hash, "metadata": before["cache_metadata"]},
        "hgb_model": {"path": str(args.model.resolve()), "sha256": model_hash},
        "hgb_train_data_sha256": train_hash, "inference_prior_policy": prior_policy,
        "microcategory_model": {"path": str(args.microcategory_model.resolve()), "sha256": classifier_hash,
            "metadata_sha256": classifier_metadata_hash, "source_sha256": classifier_source_hash,
            "training_metadata": classifier.metadata},
        "completed_tune_reference_sha256": reference_hashes,
        "evaluator_sha256": before["evaluator"],
        "text_selection": {"limit": args.text_limit, "salt": RETRIEVAL.TEXT_SELECTION_SALT,
                           "texts_sha256": EVAL.canonical_sha256(sorted(texts))},
        "alpha_zero_matches_every_completed_hgb_group": True,
        "results": result, "limitations": [
            "Tune has already selected HGB capacity; this additional alpha grid is exploratory.",
            "The completed Audit arm is never opened, scored or used to choose alpha.",
            "Microcategory classifier outputs are soft class scores, not calibrated relevance probabilities.",
            "No new candidates are retrieved and candidate Recall is unchanged.",
            "Train-only classifier inference on cold Tune needs no self-label crossfit; fitting any future reranker requires grouped OOF predictions.",
            "A Tune-selected alpha is not an independent result or a released model update.",
            "The sampled cold research catalog differs from the official benchmark."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("x", encoding="utf-8") as stream:
        for row in scored:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    report["group_scores_sha256"] = EVAL.sha256_file(scores_path)
    EVAL.write_new_json(args.output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--microcategory-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args(argv)
    try:
        report = evaluate(args)
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report["results"], ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
