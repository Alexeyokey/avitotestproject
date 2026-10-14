"""Evaluate a fit-only retrained HGB on the already fixed geo0.05 Tune pool.

The experiment changes learned reranker trees, never retrieval, embeddings,
targets, features or denominators. Every rebuilt pool must reproduce the prior
completed geography experiment's per-context measurements. Tune has already
been used for research; prefix selection here is exploratory, not a new test.
No Audit labels are parsed and no ordinary production model is overwritten.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import inspect
import json
import math
from pathlib import Path
import time

import numpy as np

from avito_candidates.core import SEARCH_FIELDS, normalize
from avito_candidates.reranker import FeatureBuilder, V8_FEATURES, corpus_fingerprint, prediction_scores
from cache_research_reranker import (check_cold_groups, check_sources, check_training_prior_file,
    research_prior_policy, validate_research_artifact)
from evaluate_geo_dense_tune import (ann_result, checked_baseline, compose_result, make_retriever,
    raw_shared_routes, verify_vector_inputs)
from evaluate_research_reranker import named_summary, paired_pipeline_bootstrap
from produce_document_hypotheses import check_encoder_base, iter_catalog_records
import evaluate_fresh_validation as EVAL


STAGES = (50, 75, 100, 150)
PRIMARY_STAGE = 100
SELECTION_TOLERANCE = .002  # 0.2 percentage point, declared before scoring.
GEO_CONFIG = {
    "dense_retrieval_policy": "exact-cosine-plus-location-log-prior-v1",
    "geo_dense_weight": .05, "geo_dense_smoothing_mass": 3.,
    "geo_dense_distance_scale_km": 40., "geo_dense_probability_floor": 1e-7,
}
CONTEXT_FIELDS = ("group_identity", "query_key", "query_text", "representative",
                  "available_relevant_count", "full_relevant_count")
POOL_FIELDS = ("pool_size", "pool_hits", "baseline_hits", "dense_hits")


class TuneFeatureCache:
    """Bounded-memory diagnostic cache; never a source of training examples.

Only contexts with an available target need learned scoring. Every context,
including skipped zero-availability contexts, is bound in the sidecar through
the completed per-context evaluation file. Labels intentionally have neither
the training column name ``label`` nor training row weights.
"""
    def __init__(self, directory, *, feature_names, provenance):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.path = Path(directory) / "tune-features.parquet"
        self.sidecar = self.path.with_suffix(self.path.suffix + ".json")
        self.marker = self.path.with_suffix(self.path.suffix + ".INCOMPLETE")
        self.feature_names = tuple(feature_names)
        reserved = {"group_identity", "item_id", "available_positive"}
        if (not self.feature_names or len(set(self.feature_names)) != len(self.feature_names)
                or reserved.intersection(self.feature_names)):
            raise ValueError("Diagnostic feature cache requires unique numeric feature names")
        if any(path.exists() for path in (self.path, self.sidecar, self.marker)):
            raise FileExistsError("Diagnostic feature cache outputs are immutable")
        self.provenance = copy.deepcopy(provenance)
        self.rows = self.positive_rows = self.eligible_groups = 0
        self.groups = []
        self.seen = set()
        self.closed = False
        self.marker.write_text(json.dumps({"complete": False, "role": "tune-only-never-training"}), encoding="utf-8")
        schema = pa.schema([(name, pa.float32()) for name in self.feature_names] + [
            ("group_identity", pa.string()), ("item_id", pa.string()), ("available_positive", pa.bool_())])
        self.writer = pq.ParquetWriter(self.path, schema, compression="zstd", use_dictionary=False)

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def close(self):
        if not self.closed:
            self.writer.close()
            self.closed = True

    def write_batch(self, records):
        import pyarrow as pa
        if self.closed:
            raise ValueError("Cannot append to a closed diagnostic cache")
        matrices, identities, item_ids, positives = [], [], [], []
        for row, pool, relevant, matrix, _index in records:
            key = row["group_identity"]
            if key in self.seen or len(pool) != row["pool_size"] or len(set(pool)) != len(pool):
                raise ValueError("Diagnostic cache context repeats or has a malformed full pool")
            if len(relevant) != row["available_relevant_count"]:
                raise ValueError("Diagnostic cache relevance denominator differs")
            self.seen.add(key)
            self.groups.append({"group_identity": key, "pool_size": len(pool),
                "available_relevant_count": row["available_relevant_count"],
                "full_relevant_count": row["full_relevant_count"],
                "ordered_pool_sha256": EVAL.canonical_sha256(pool),
                "exported_rows": len(pool) if relevant else 0,
                "exported_positive_rows": len(set(pool) & relevant) if relevant else 0})
            if not relevant:
                if matrix is not None:
                    raise ValueError("Zero-availability groups must not add diagnostic feature rows")
                continue
            values = np.asarray(matrix)
            if (values.dtype != np.float32 or values.shape != (len(pool), len(self.feature_names))
                    or not np.isfinite(values).all()):
                raise ValueError("Cache must preserve the actual finite float32 full-pool features")
            matrices.append(values)
            identities.extend([key] * len(pool))
            item_ids.extend(pool)
            labels = [item in relevant for item in pool]
            positives.extend(labels)
            self.rows += len(pool)
            self.positive_rows += sum(labels)
            self.eligible_groups += 1
        if matrices:
            matrix = np.concatenate(matrices)
            arrays = [pa.array(matrix[:, index], type=pa.float32()) for index in range(len(self.feature_names))]
            arrays.extend((pa.array(identities, type=pa.string()), pa.array(item_ids, type=pa.string()),
                           pa.array(positives, type=pa.bool_())))
            table = pa.Table.from_arrays(arrays, schema=self.writer.schema)
            self.writer.write_table(table, row_group_size=len(table))

    def finalize(self, *, report_path, score_path):
        """Mark complete only after the entire evaluation report is saved."""
        import pyarrow.parquet as pq
        self.close()
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        if (report.get("complete") is not True or report.get("phase") != "tune"
                or report.get("groups") != len(self.groups)
                or report.get("group_scores_sha256") != EVAL.sha256_file(score_path)):
            raise ValueError("Diagnostic cache needs a complete matching Tune evaluation report")
        actual_groups = []
        with Path(score_path).open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)["new"]
                actual_groups.append({"group_identity": row["group_identity"], "pool_size": row["pool_size"],
                    "available_relevant_count": row["available_relevant_count"],
                    "full_relevant_count": row["full_relevant_count"],
                    "ordered_pool_sha256": row["ordered_pool_sha256"],
                    "exported_rows": row["pool_size"] if row["available_relevant_count"] else 0,
                    "exported_positive_rows": row["pool_hits"] if row["available_relevant_count"] else 0})
        if (actual_groups != self.groups or pq.ParquetFile(self.path).metadata.num_rows != self.rows
                or self.rows != sum(group["exported_rows"] for group in self.groups)
                or self.positive_rows != sum(group["exported_positive_rows"] for group in self.groups)):
            raise ValueError("Diagnostic cache counts or ordered pool identities do not match scoring")
        sidecar = {"complete": True, "role": "tune-only-never-training", "phase": "tune",
            "training_allowed": False, "audit_labels_read": False,
            "full_tune_complete": report.get("full_tune_complete", False),
            "feature_names": list(self.feature_names), "feature_dtype": "float32",
            "identity_columns": ["group_identity", "item_id"], "positive_column": "available_positive",
            "positive_meaning": "Membership in the fixed Tune available-positive set; never training labels",
            "rows": self.rows, "positive_rows": self.positive_rows, "groups": len(self.groups),
            "exported_groups": self.eligible_groups,
            "zero_availability_groups_skipped": len(self.groups) - self.eligible_groups,
            "data": identity(self.path), "evaluation_report": identity(report_path),
            "all_group_metadata_and_denominators": identity(score_path),
            "groups_in_evaluation_order": self.groups, "provenance": self.provenance}
        EVAL.write_new_json(self.sidecar, sidecar)
        self.marker.unlink()
        return identity(self.sidecar)


def choose_conservative_stage(curve):
    """Smallest declared prefix within 0.2pp of best observed text-macro Recall."""
    if set(curve) != {str(stage) for stage in STAGES}:
        raise ValueError("Selection requires every predeclared checkpoint")
    scores = {int(stage): summary["conditional_available"]["reranker"]["text_macro_recall_at_50"]
              for stage, summary in curve.items()}
    if any(not isinstance(value, (float, int)) or not math.isfinite(value) for value in scores.values()):
        raise ValueError("Selection requires finite conditional text-macro Recall")
    best = max(scores.values())
    return min(stage for stage, value in scores.items() if best - value <= SELECTION_TOLERANCE + 1e-12)


def validate_new_config(new, old):
    if any(new.get(key) != value for key, value in GEO_CONFIG.items()):
        raise ValueError("New reranker requires the fixed geo0.05 full-corpus dense policy")
    if {key: value for key, value in new.items() if key not in GEO_CONFIG} != old:
        raise ValueError("Retraining may change only the declared geography configuration keys")


def assert_same_context(left, right):
    for name in CONTEXT_FIELDS:
        if left.get(name) != right.get(name):
            raise ValueError(f"Tune context or complete Recall denominator changed: {name}")


def assert_same_pool(row, previous):
    assert_same_context(row, previous)
    for name in POOL_FIELDS:
        if row.get(name) != previous.get(name):
            raise ValueError(f"Rebuilt geo0.05 pool differs from completed experiment: {name}")
    available = row["available_relevant_count"]
    if (row["pool_size"] > 998 or not 0 <= row["pool_hits"] <= min(available, row["pool_size"])
            or not 0 <= row["baseline_hits"] <= min(50, row["pool_hits"])
            or not 0 <= row["dense_hits"] <= min(449, row["pool_hits"])):
        raise ValueError("Invalid fixed-pool budget or positive counts")


def checked_previous_geo(args, metadata, protocol_hash, source_model_hash):
    report = json.loads(args.previous_geo_report.read_text(encoding="utf-8"))
    if (report.get("complete") is not True or report.get("phase") != "tune"
            or report.get("protocol_sha256") != protocol_hash or report.get("group_limit") != 0
            or report.get("model_sha256") != source_model_hash or report.get("model_iteration") != 100
            or report.get("reranker_retrained") is not False or report.get("primary") != "geo_0.05"
            or report.get("maximum_pool_size") != 998 or report.get("dense_k") != 449
            or report.get("geo_prior") != {"smoothing_mass": 3., "distance_scale_km": 40., "probability_floor": 1e-7}
            or report.get("group_scores_sha256") != EVAL.sha256_file(args.previous_geo_scores)
            or report.get("vector_manifest_sha256") != EVAL.sha256_file(args.vector_dir / "manifest.json")):
        raise ValueError("Prior geo0.05 baseline is not the complete frozen same-vector Tune experiment")
    check_sources(report["source_sha256"])
    rows = []
    with args.previous_geo_scores.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            row = record["variants"]["geo_0.05"]
            if record["group_identity"] != row["group_identity"]:
                raise ValueError("Prior geographic score wrapper identity differs")
            rows.append(row)
    if len(rows) != len(metadata["groups"]) or len(rows) != report.get("groups"):
        raise ValueError("Prior geographic baseline omits complete Tune contexts")
    for row, group in zip(rows, metadata["groups"]):
        assert_same_context(row, group)
    # Recompute from individual counts rather than trusting the report's totals.
    if named_summary(rows, 100) != {key: value for key, value in report["variants"]["geo_0.05"].items()
                                    if key != "dense_recall"}:
        raise ValueError("Prior geographic aggregate does not match its scored rows")
    return rows


def metric_row(group, result, relevant):
    baseline, details = result
    row = {key: value for key, value in group.items()
           if not key.startswith("_") and key not in {"relevant_item_ids", "available_item_ids"}}
    row.update(pool_size=len(details["candidate_pool"]),
               ordered_pool_sha256=EVAL.canonical_sha256(details["candidate_pool"]),
               pool_hits=len(set(details["candidate_pool"]) & relevant),
               baseline_hits=len(set(baseline) & relevant),
               dense_hits=len(set(details["channels"]["dense"]) & relevant))
    return row


def top50_hits(scores, matrix, pool, relevant):
    scores = np.asarray(scores)
    if scores.shape != (len(pool),) or matrix.shape[0] != len(pool) or not np.isfinite(scores).all():
        raise ValueError("Reranker scores do not cover the finite complete pool")
    indices = np.lexsort((np.arange(len(pool)), -matrix[:, 0], -scores))[:50]
    return len({pool[int(index)] for index in indices} & relevant)


def identity(path):
    return {"path": str(Path(path).resolve()), "sha256": EVAL.sha256_file(path)}


def save_selected_artifact(artifact, source_path, selected_stage, report_path, declaration_path, output_dir):
    """Publish a separate Tune-selected prefix; keep the complete fitted estimator."""
    import joblib
    from avito_candidates.fitted_fingerprint import fitted_estimator_fingerprint
    if selected_stage not in STAGES or EVAL.maximum_iterations(artifact["model"]) != 150:
        raise ValueError("Selected model must retain its complete fitted 150-tree source")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (not report.get("full_tune_complete") or not report.get("all_per_context_pool_metrics_reproduced")
            or selected_stage != choose_conservative_stage(report["new_capacity_curve"])):
        raise ValueError("Publication requires a complete verified Tune comparison and declared selection")
    status = "selected-on-consumed-research-tune; no-new-audit"
    selected = dict(artifact)
    selected.update(selection_status=status, selected_iteration=selected_stage)
    selected["training_configuration"] = {**artifact.get("training_configuration", {}),
        "selection_status": status, "selected_iteration": selected_stage}
    selected["geo_dense_selection"] = {
        "version": 1, "source_model": identity(source_path),
        "report": identity(report_path), "declaration": identity(declaration_path),
        "selected_stage": selected_stage, "primary_stage": PRIMARY_STAGE,
        "stage_tolerance": SELECTION_TOLERANCE, "selection_rule": report["selection_rule"],
        "required_geo_dense_config": GEO_CONFIG,
        "fitted_estimator_sha256": fitted_estimator_fingerprint(artifact["model"]),
        "scope": "Exploratory Tune selection on the fixed fresh sampled catalog; not official benchmark validation",
    }
    destination = Path(output_dir) / f"hgb-geo005-selected{selected_stage}.joblib"
    manifest_path = destination.with_suffix(destination.suffix + ".json")
    if destination.exists() or manifest_path.exists():
        raise FileExistsError("Selected model output is immutable")
    with destination.open("xb") as stream:
        joblib.dump(selected, stream)
    loaded = joblib.load(destination)
    if (loaded.get("selected_iteration") != selected_stage
            or loaded.get("metadata") != artifact["metadata"]
            or fitted_estimator_fingerprint(loaded["model"]) != selected["geo_dense_selection"]["fitted_estimator_sha256"]
            or loaded.get("geo_dense_selection") != selected["geo_dense_selection"]):
        raise ValueError("Published prefix failed independent reload verification")
    EVAL.write_new_json(manifest_path, {**selected["geo_dense_selection"], "model": identity(destination),
        "selected_iteration": selected_stage, "selection_status": status,
        "retrieval_config": artifact["metadata"]["retrieval_config"], "training_provenance": artifact["provenance"]})
    return identity(destination)


def evaluate(args):
    import joblib
    from avito_candidates.geo_dense import GeoDensePrior, ExactDenseScorer
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory")
    args.output_dir.mkdir(parents=True)
    # Verify only files requested below. Even byte-hashing unrelated Audit
    # targets is unnecessary for this Tune-only experiment.
    from build_fresh_validation_protocol import load_manifest
    manifest = load_manifest(args.protocol_dir, verify=False)
    protocol_hash = EVAL.sha256_file(args.protocol_dir / "manifest.json")
    source, new = joblib.load(args.source_model), joblib.load(args.new_model)
    fit_texts = set(json.loads(EVAL.protocol_file(args.protocol_dir, manifest, "fit_texts").read_text())["texts"])
    for artifact, train_data in ((source, args.source_train_data), (new, args.train_data)):
        validate_research_artifact(artifact, fit_texts, manifest)
        EVAL.check_encoder_identity(artifact, manifest)
        EVAL.check_training_data(train_data, artifact, args.protocol_dir, manifest)
        check_training_prior_file(train_data, artifact)
    if (new.get("estimator", "histgb") != "histgb" or new.get("selected_iteration") is not None
            or EVAL.maximum_iterations(new["model"]) != 150):
        raise ValueError("New comparison requires an unselected complete HGB150 estimator")
    if research_prior_policy(new) != research_prior_policy(source):
        raise ValueError("Retraining changed frozen inference priors")
    validate_new_config(new["metadata"]["retrieval_config"], source["metadata"]["retrieval_config"])
    if (new["metadata"].get("corpus_fingerprint") != source["metadata"].get("corpus_fingerprint")
            or new["metadata"].get("feature_names") != list(V8_FEATURES)):
        raise ValueError("New reranker changed the catalog or V8 feature schema")
    training = new["metadata"].get("geo_dense_training", {})
    expected_training = {"complete": True, "phase": "fit", "tune_labels_read": False,
        "audit_labels_read": False, "inventory_reencoded": False, "protocol_sha256": protocol_hash,
        "source_model_sha256": EVAL.sha256_file(args.source_model),
        "source_train_data_sha256": EVAL.sha256_file(args.source_train_data),
        "fit_texts_sha256": EVAL.sha256_file(EVAL.protocol_file(args.protocol_dir, manifest, "fit_texts")),
        "fit_train_sha256": manifest["files"]["train"]["sha256"], "selected_groups": 30000,
        "geo_dense_policy": GEO_CONFIG, "maximum_pool_size": 998, "negatives_per_query": 64,
        "training_geo_dense_history": "leave-query-text-out", "source_prior_policy": research_prior_policy(source)}
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise ValueError("New training must seal its complete fit-only geographic preparation")
    check_sources(training["source_sha256"])
    metadata, old_rows = checked_baseline(args, manifest, protocol_hash, source)
    previous_rows = checked_previous_geo(args, metadata, protocol_hash, EVAL.sha256_file(args.source_model))
    config = copy.deepcopy(new["metadata"]["retrieval_config"])
    base_identity = check_encoder_base(config["embedding_model"], manifest, args.encoder_base_reference)
    vector_manifest, order, arrays = verify_vector_inputs(args.vector_dir, args.cache_dir / "tune.json", metadata, protocol_hash)
    if training.get("corpus_vector_sha256") != vector_manifest["files"]["corpus.npy"]["sha256"]:
        raise ValueError("New reranker training and Tune must use the identical corpus vectors")
    input_paths = {name: getattr(args, name) for name in ("source_model", "new_model", "train_data",
        "source_train_data", "baseline_report", "baseline_scores", "previous_geo_report", "previous_geo_scores")}
    inputs = {name: identity(path) for name, path in input_paths.items()}
    sources = {str(path.resolve()): EVAL.sha256_file(path) for path in (
        Path(__file__), Path(inspect.getfile(GeoDensePrior)), Path(inspect.getfile(compose_result)),
        Path(inspect.getfile(FeatureBuilder)), Path(inspect.getfile(paired_pipeline_bootstrap)))}
    declaration = {"phase": "tune", "audit_labels_read": False, "reranker_retrained": True,
        "stages": list(STAGES), "primary_stage": PRIMARY_STAGE,
        "selection_rule": "smallest predeclared stage within 0.002 absolute text-macro Recall of best Tune stage",
        "selection_tolerance": SELECTION_TOLERANCE, "geo_dense_config": GEO_CONFIG,
        "maximum_pool_size": 998, "dense_k": 449, "source_stage": 100,
        "protocol_sha256": protocol_hash, "encoder_base": base_identity,
        "vector_manifest": identity(args.vector_dir / "manifest.json"), "input_identities": inputs,
        "experiment_source_sha256": sources, "group_limit": args.limit_groups,
        "save_feature_cache": args.save_feature_cache,
        "interpretation": "Previously consumed Tune, fit-only training, exploratory checkpoint selection; no independent confirmation"}
    declaration_path = args.output_dir / "declaration.json"
    EVAL.write_new_json(declaration_path, declaration)
    labels = EVAL.read_phase_labels(args.protocol_dir, manifest, "tune")
    check_cold_groups(list(labels.values()), fit_texts)
    groups = metadata["groups"][:args.limit_groups or None]
    if any(not normalize(group["query_key"][0]) for group in groups):
        raise ValueError("The established full-Tune vector cohort requires nonempty query texts")
    items = list(iter_catalog_records(EVAL.protocol_file(args.protocol_dir, manifest, "benchmark_items")))
    if ([item["item_id"] for item in items] != order["corpus_item_ids"]
            or corpus_fingerprint(items, feature_version=8) != metadata["corpus_fingerprint"]):
        raise ValueError("Geographic model vectors/catalog row identity differs")
    item_ids = order["corpus_item_ids"]
    id_to_row = {item: index for index, item in enumerate(item_ids)}
    corpus_ids = set(item_ids)
    priors = new["priors"]
    prior = GeoDensePrior(items, priors["geo_associations"], smoothing_mass=3.,
                         distance_scale_km=40., probability_floor=1e-7)
    scorer = ExactDenseScorer(arrays["corpus"], device=args.device)
    builder = FeatureBuilder(items, config, microcategory_prior=priors["microcategory_prior"],
        geo_associations=priors["geo_associations"], microcategory_classifier=priors["microcategory_classifier"],
        feature_names=V8_FEATURES)
    hybrid, micro = make_retriever(config, items, priors)
    rows, same_pool_rows, baseline_rows, old_model_checks = [], [], [], 0
    started = time.monotonic()
    output_scores = args.output_dir / "group-scores.jsonl"
    feature_cache = (TuneFeatureCache(args.output_dir, feature_names=V8_FEATURES,
        provenance={"input_identities": inputs, "protocol_sha256": protocol_hash,
            "vector_manifest": declaration["vector_manifest"], "retrieval_config": config,
            "research_prior_policy": research_prior_policy(new), "source_sha256": sources})
        if args.save_feature_cache else None)
    with output_scores.open("x", encoding="utf-8") as stream, (feature_cache if feature_cache else nullcontext()):
        for start in range(0, len(groups), args.batch_size):
            batch = groups[start:start + args.batch_size]
            cosines = scorer.score(arrays["queries"][start:start + len(batch)])
            pending, matrices = [], []
            for offset, (group, cosine) in enumerate(zip(batch, cosines)):
                index = start + offset
                query = dict(zip(SEARCH_FIELDS, group["query_key"]))
                target = labels[group["group_identity"]]
                relevant = set(target["available_item_ids"])
                if (set(target["relevant_item_ids"]) & corpus_ids != relevant
                        or len(relevant) != group["available_relevant_count"]
                        or len(target["relevant_item_ids"]) != group["full_relevant_count"]):
                    raise ValueError("Tune full and available denominators differ from fixed catalog")
                ann998 = ann_result(arrays["ann998"][index], arrays["ann998-cosines"][index], item_ids)
                shared = raw_shared_routes(query, hybrid, micro, ann998)
                dense = prior.retrieve(cosine, query.get("search_location_id", ""),
                    limit=449, weight=.05, exclude_query_text=group["query_text"])
                result = compose_result(query, hybrid, shared, dense, cosine, id_to_row, config["microcategory_weight"])
                row = metric_row(group, result, relevant)
                assert_same_pool(row, previous_rows[index])
                matrix = None
                if relevant:
                    matrix = builder.transform(query, result[1], result[0])
                    if not np.isfinite(matrix).all():
                        raise ValueError("New fixed-pool features must be finite")
                    if old_model_checks < args.parity_groups or index % 500 == 0:
                        old_predictions = prediction_scores(source["model"], matrix, selected_iteration=100)
                        if top50_hits(old_predictions, matrix, result[1]["candidate_pool"], relevant) != previous_rows[index]["s100_hits"]:
                            raise ValueError("Frozen oldHGB100 no longer reproduces the previous same-pool result")
                        old_model_checks += 1
                    matrices.append(matrix)
                pending.append((row, result[1]["candidate_pool"], relevant, matrix, index))
            probabilities = dict(EVAL.prefix_scores(new["model"], np.concatenate(matrices), STAGES)) if matrices else {}
            cursor = 0
            for row, pool, relevant, matrix, index in pending:
                for stage in STAGES:
                    row[f"s{stage}_hits"] = (top50_hits(probabilities[stage][cursor:cursor + len(pool)],
                        matrix, pool, relevant) if relevant else 0)
                if relevant:
                    cursor += len(pool)
                rows.append(row)
                same_pool_rows.append(previous_rows[index])
                baseline_rows.append(old_rows[row["group_identity"]])
                stream.write(json.dumps({"group_identity": row["group_identity"], "new": row,
                    "old_same_pool": previous_rows[index], "old_v8": baseline_rows[-1]},
                    ensure_ascii=False, separators=(",", ":")) + "\n")
            if feature_cache:
                feature_cache.write_batch(pending)
            stream.flush()
            done = start + len(batch)
            if start == 0 or done % 128 < args.batch_size or done == len(groups):
                elapsed = time.monotonic() - started
                print(json.dumps({"groups": done, "total": len(groups), "seconds": round(elapsed, 1),
                    "estimated_remaining_seconds": round(elapsed * (len(groups) - done) / done, 1),
                    "old_model_parity_groups": old_model_checks}), flush=True)
    curve = {str(stage): named_summary(rows, stage) for stage in STAGES}
    selected_stage = choose_conservative_stage(curve)
    comparisons = {}
    for stage in STAGES:
        comparisons[str(stage)] = {}
        for name, reference, old_hits in (("old_same_pool_hgb100", same_pool_rows, "s100_hits"),
                ("old_v8_hgb100", baseline_rows, "s100_hits"), ("same_pool_rrf", rows, "baseline_hits")):
            value = paired_pipeline_bootstrap(list(zip(rows, reference)), f"s{stage}_hits", old_hits,
                "available_relevant_count", replicates=args.bootstrap_replicates)
            value["comparison"] = f"new geo0.05 HGB{stage} minus {name}"
            comparisons[str(stage)][name] = value
    losses = (EVAL.sampled_train_losses(args.train_data, new, STAGES, args.loss_batch_size)
              if args.with_training_loss else None)
    check_sources(metadata["research_source_sha256"])
    check_sources(sources)
    check_sources(training["source_sha256"])
    for name, path in input_paths.items():
        if identity(path) != inputs[name]:
            raise ValueError(f"Evaluation input changed: {name}")
    if EVAL.sha256_file(args.protocol_dir / "manifest.json") != protocol_hash:
        raise ValueError("Frozen protocol changed during Tune evaluation")
    report = {**declaration, "complete": True, "full_tune_complete": args.limit_groups == 0,
        "groups": len(rows), "all_per_context_pool_metrics_reproduced": True,
        "pool_identity_evidence": "Same retrieval source, vectors and priors; all context counts and RRF hits match. Prior run did not save pool IDs, so exact historical ID equality cannot be directly checked. New ordered pool hashes are saved.",
        "old_model_parity_groups": old_model_checks,
        "new_capacity_curve": curve, "selected_stage": selected_stage,
        "old_same_pool_hgb100": named_summary(same_pool_rows, 100),
        "old_v8_hgb100": named_summary(baseline_rows, 100),
        "dense_recall": EVAL.aggregate(rows, "dense_hits", "available_relevant_count"),
        "paired_comparisons": comparisons, "training_row_loss": losses,
        "group_scores_sha256": EVAL.sha256_file(output_scores), "seconds": time.monotonic() - started,
        "feature_cache": ({"data_path": str(feature_cache.path.resolve()),
            "metadata_path": str(feature_cache.sidecar.resolve()), "role": "tune-only-never-training"}
            if feature_cache else None),
        "no_available_target_reranking_skipped": True,
        "interpretation_notes": ["Training loss is not percentage error or retrieval Recall.",
            "Falling train loss and falling Tune Recall across capacities suggests overfitting, not proof.",
            "Selection uses previously consumed Tune; intervals after exploration are not independent confirmation.",
            "The selected artifact requires its saved geo0.05 pool and cannot replace the old ANN model directly."]}
    report_path = args.output_dir / "report.json"
    EVAL.write_new_json(report_path, report)
    selected = (save_selected_artifact(new, args.new_model, selected_stage, report_path, declaration_path, args.output_dir)
                if not args.limit_groups else None)
    cache_metadata = feature_cache.finalize(report_path=report_path, score_path=output_scores) if feature_cache else None
    print(json.dumps({"complete": True, "report": str(report_path), "selected_model": selected,
                      "selected_stage": selected_stage, "feature_cache_metadata": cache_metadata}), flush=True)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["tune"], default="tune")
    for name in ("source-model", "new-model", "train-data", "source-train-data", "protocol-dir", "cache-dir",
                 "baseline-report", "baseline-scores", "previous-geo-report", "previous-geo-scores",
                 "vector-dir", "encoder-base-reference", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit-groups", type=int, default=0)
    p.add_argument("--parity-groups", type=int, default=32)
    p.add_argument("--bootstrap-replicates", type=int, default=2000)
    p.add_argument("--loss-batch-size", type=int, default=16384)
    p.add_argument("--with-training-loss", action="store_true")
    p.add_argument("--save-feature-cache", action="store_true",
                   help="Save diagnostic full-pool features for available-positive Tune contexts; never use for training")
    args = p.parse_args(argv)
    if min(args.batch_size, args.parity_groups, args.bootstrap_replicates, args.loss_batch_size) < 1 or args.limit_groups < 0:
        p.error("Use positive batches/parity/bootstrap and a nonnegative metadata-prefix limit")
    evaluate(args)


if __name__ == "__main__":
    main()
