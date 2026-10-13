"""Select a reranker prefix on fresh tune texts, then open the sealed audit once.

Only a complete saved candidate cache is scored. This script does not retrieve,
train, or read old validation labels. The corpus is a label-blind sampled
inventory, so its conditional Recall is not the official benchmark metric.
Tune considers the preregistered HGB prefixes; audit has no prefix-grid option
and accepts only the stage sealed by tune. Every unavailable target remains in
the separate full-truth denominators, including queries with no available target.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re

from audit_reranker_learning_curve import (
    ensure_compatible, ensure_prior_policy, maximum_iterations,
    normalize_query_text, prefix_scores, sampled_train_losses, sha256_file,
    top_indices,
)


TUNE_STAGES = (25, 50, 75, 100, 150)
POLICY = {
    "version": 1, "tune_stages": list(TUNE_STAGES), "limit": 50,
    "selection_metric": "conditional_available_text_macro_recall_at_50",
    "tie_rule": "smallest stage within 1e-12 of the maximum",
    "audit_comparisons": ["selected_stage", "baseline_rrf"],
    "bootstrap_unit": "normalized query text", "bootstrap_seed": 20261013,
    "bootstrap_replicates": 2000,
}
SLICE_DENOMINATORS = {
    "local_positive": "local_relevant_count",
    "nonlocal_positive": "nonlocal_relevant_count",
    "item_seen_in_allowed_fit": "seen_relevant_count",
    "item_unseen_in_allowed_fit": "unseen_relevant_count",
}


def canonical_sha256(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def checked_protocol(protocol_dir):
    from build_fresh_validation_protocol import load_manifest
    path = Path(protocol_dir) / "manifest.json"
    # Check the incomplete marker and every sealed file byte hash. Hashing the
    # audit file does not parse or score its labels; only read_phase_labels does.
    manifest = load_manifest(protocol_dir, verify=True)
    return manifest, sha256_file(path)


def protocol_file(protocol_dir, manifest, name):
    """Verify only the requested arm; tune never parses audit labels."""
    semantic_name = name.removesuffix(".json").replace("-", "_")
    entry = manifest["files"].get(semantic_name, manifest["files"].get(name))
    if not isinstance(entry, dict) or not entry.get("sha256"):
        raise ValueError(f"Protocol inventory lacks {name}")
    root = Path(protocol_dir).resolve()
    path = (root / entry.get("path", name)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Protocol file is missing or outside its directory: {name}")
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(f"Protocol file SHA256 differs: {name}")
    return path


def read_phase_labels(protocol_dir, manifest, phase):
    path = protocol_file(protocol_dir, manifest, f"labels-{phase}.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("partition") != phase or not isinstance(payload.get("groups"), list):
        raise ValueError("Protocol label partition differs from requested phase")
    groups, by_text = {}, defaultdict(list)
    for group in payload["groups"]:
        key = group.get("query_key")
        if not isinstance(key, list) or len(key) != 5 or any(not isinstance(x, str) for x in key):
            raise ValueError("Protocol query key must contain five strings")
        key_json = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
        identity = hashlib.sha256(key_json.encode("utf-8")).hexdigest()
        text = normalize_query_text(key[0])
        if (group.get("group_id") != identity or group.get("query_key_json") != key_json
                or group.get("query_text") != text or identity in groups):
            raise ValueError("Invalid or repeated protocol group identity")
        full, available = group.get("relevant_item_ids"), group.get("available_item_ids")
        if not isinstance(full, list) or not isinstance(available, list):
            raise ValueError("Protocol group lacks complete target ID sets")
        if (not full or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{16}", value)
                            for value in full + available)
                or len(full) != len(set(full)) or len(available) != len(set(available))
                or not set(available) <= set(full)
                or group.get("full_relevant_count") != len(full)
                or group.get("available_relevant_count") != len(available)):
            raise ValueError("Protocol target sets or denominators are invalid")
        if not isinstance(group.get("representative"), bool):
            raise ValueError("Protocol representative flag must be explicit")
        groups[identity] = group
        by_text[text].append(group)
    if not groups:
        raise ValueError("Fresh validation arm has no groups")
    for text, variants in by_text.items():
        if sum(group["representative"] for group in variants) != 1:
            raise ValueError(f"Expected one label-blind representative for {text!r}")
        salt = manifest.get("split", {}).get("representative_salt")
        if not isinstance(salt, str) or not salt:
            raise ValueError("Protocol must seal the label-blind representative salt")
        expected = min(variants, key=lambda group: hashlib.sha256(json.dumps(
            [salt, group["query_key_json"]], ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).digest())
        if not expected["representative"]:
            raise ValueError("Representative selection differs from label-blind hash rule")
    texts_path = protocol_file(protocol_dir, manifest, f"{phase}-texts.json")
    texts = json.loads(texts_path.read_text(encoding="utf-8")).get("texts")
    if not isinstance(texts, list) or len(texts) != len(set(texts)) or set(texts) != set(by_text):
        raise ValueError("Protocol text inventory differs from group labels")
    return groups


def phase_cache_paths(cache_dir, phase):
    root = Path(cache_dir)
    if (root / f"{phase}.parquet").is_file():
        return root / f"{phase}.parquet", root / f"{phase}.json"
    return root / "validation.parquet", root / "validation.json"


def check_phase_cache_inputs(path, metadata, phase, protocol_hash, model_hash):
    """Validate cache identity, bytes and shape without opening protocol labels."""
    import pyarrow.parquet as pq
    if (metadata.get("cache_version") != 2 or metadata.get("role") != "validation"
            or metadata.get("fresh_partition") != phase
            or metadata.get("fresh_validation_protocol_sha256") != protocol_hash
            or metadata.get("model_sha256") != model_hash
            or metadata.get("historical_holdout_reused") is not False
            or metadata.get("full_truth_preserved") is not True):
        raise ValueError("Fresh cache identity or partition differs before opening labels")
    actual_hash = sha256_file(path)
    if metadata.get("cache_parquet_sha256") != actual_hash:
        raise ValueError("Fresh cache Parquet bytes changed or are unsealed")
    features = metadata.get("feature_names")
    if (not isinstance(features, list) or not features
            or any(not isinstance(name, str) for name in features)
            or len(features) != len(set(features))
            or not {"rrf_score", "baseline_top50"} <= set(features)):
        raise ValueError("Fresh cache feature schema is invalid")
    groups = metadata.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("Fresh cache must preserve its complete group inventory")
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError("Fresh cache group metadata is invalid")
        size, group_id = group.get("pool_size"), group.get("group_id")
        if (isinstance(group_id, bool) or not isinstance(group_id, int)
                or group_id != index or isinstance(size, bool)
                or not isinstance(size, int) or not 0 <= size <= 998):
            raise ValueError("Fresh cache group order or candidate budget is invalid")
    rows = metadata.get("rows")
    if (isinstance(rows, bool) or not isinstance(rows, int) or rows < 0
            or rows != sum(group["pool_size"] for group in groups)):
        raise ValueError("Fresh cache rows differ from its complete pools")
    parquet = pq.ParquetFile(path)
    required = set(features) | {"group_id", "item_id", "label", "positive_local", "positive_seen"}
    if (parquet.metadata.num_rows != rows
            or not required <= set(parquet.schema_arrow.names)
            or any(str(parquet.schema_arrow.field(name).type) != "float" for name in features)):
        raise ValueError("Fresh cache Parquet rows or schema differ from sealed metadata")
    return actual_hash


def bind_cache_groups(metadata, protocol_groups, phase, protocol_hash, model_hash):
    if (metadata.get("role") != "validation"
            or metadata.get("fresh_partition") != phase
            or metadata.get("fresh_validation_protocol_sha256") != protocol_hash
            or metadata.get("model_sha256") != model_hash):
        raise ValueError("Cache protocol, partition, role, or model identity differs")
    groups = metadata.get("groups")
    if not isinstance(groups, list) or len(groups) != len(protocol_groups):
        raise ValueError("Cache must include every protocol group, even unavailable targets")
    bound, identities = [], set()
    for index, cached in enumerate(groups):
        if cached.get("group_id") != index:
            raise ValueError("Cache group IDs must be ordered and contiguous")
        key_json = json.dumps(cached.get("query_key"), ensure_ascii=False, separators=(",", ":"))
        identity = hashlib.sha256(key_json.encode("utf-8")).hexdigest()
        if identity not in protocol_groups or identity in identities:
            raise ValueError("Cache query identity differs from protocol")
        source = protocol_groups[identity]
        if normalize_query_text(cached.get("query_text", "")) != source["query_text"]:
            raise ValueError("Cache query text differs from protocol")
        for field in ("full_relevant_count", "available_relevant_count", "representative"):
            if cached.get(field) != source[field]:
                raise ValueError(f"Cache {field} differs from sealed protocol")
        if int(cached.get("relevant_count", -1)) != source["available_relevant_count"]:
            raise ValueError("Cache Recall denominator differs from available targets")
        if int(cached.get("pool_size", -1)) < 0:
            raise ValueError("Cache pool size cannot be negative")
        for left, right in (("local_relevant_count", "nonlocal_relevant_count"),
                            ("seen_relevant_count", "unseen_relevant_count")):
            if left in cached or right in cached:
                if (left not in cached or right not in cached
                        or int(cached[left]) < 0 or int(cached[right]) < 0
                        or int(cached[left]) + int(cached[right]) != source["available_relevant_count"]):
                    raise ValueError("Cache slice denominators do not sum to available targets")
        bound.append({**cached, "protocol_group_id": identity,
                      "query_text": source["query_text"], "partition": phase,
                      "_available_ids": set(source["available_item_ids"])})
        identities.add(identity)
    if identities != set(protocol_groups):
        raise ValueError("Cache omitted protocol groups")
    return bound


def score_cached_group(group, items, labels, local, seen, baseline, rrf, predictions):
    count = len(items)
    if (count != group["pool_size"] or len(set(items)) != count
            or not all(len(values) == count for values in (labels, local, baseline, rrf))
            or (seen is not None and len(seen) != count)):
        raise ValueError("Incomplete or duplicate candidate pool")
    expected = [int(item in group["_available_ids"]) for item in items]
    if list(labels) != expected:
        raise ValueError("Cache labels differ from sealed protocol targets")
    if sum(labels) != int(group.get("positives_in_pool", -1)):
        raise ValueError("Positive pool count differs from cache metadata")
    if sum(baseline) > 50:
        raise ValueError("RRF baseline contains more than 50 candidates")
    baseline_hits = sum(label for label, selected in zip(labels, baseline) if selected)
    if baseline_hits != int(group.get("baseline_hits", -1)):
        raise ValueError("RRF baseline hits differ from cache metadata")
    row = {key: value for key, value in group.items() if not key.startswith("_")}
    row["baseline_hits"] = baseline_hits
    row["pool_hits"] = sum(labels)
    masks = {"local_positive": list(map(bool, local)),
             "nonlocal_positive": [not bool(x) for x in local]}
    if seen is not None:
        masks["item_seen_in_allowed_fit"] = list(map(bool, seen))
        masks["item_unseen_in_allowed_fit"] = [not bool(x) for x in seen]
    for kind, mask in masks.items():
        row[f"baseline_{kind}_hits"] = sum(label for label, selected, valid
            in zip(labels, baseline, mask) if selected and valid)
        row[f"pool_{kind}_hits"] = sum(label for label, valid in zip(labels, mask) if valid)
        denominator = int(group.get(SLICE_DENOMINATORS[kind], 0))
        if row[f"pool_{kind}_hits"] > denominator:
            raise ValueError("Candidate slice positives exceed their full available denominator")
    for stage, scores in predictions.items():
        if len(scores) != count:
            raise ValueError("Model scores do not cover the complete pool")
        selected = top_indices(scores, rrf, 50)
        row[f"s{stage}_hits"] = sum(labels[index] for index in selected)
        for kind, mask in masks.items():
            row[f"s{stage}_{kind}_hits"] = sum(labels[index] for index in selected if mask[index])
    return row


def score_cache(path, metadata, groups, artifact, stages, batch_size):
    """Carry a partial query pool across batches; retain metadata-only empty pools."""
    import numpy as np
    import pyarrow.parquet as pq
    features = metadata["feature_names"]
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    required = set(features) | {"group_id", "item_id", "label", "positive_local"}
    if not required <= names or not {"baseline_top50", "rrf_score"} <= set(features):
        raise ValueError("Candidate cache lacks required scoring columns")
    has_seen = "positive_seen" in names
    if any("seen_relevant_count" in group for group in groups) and not has_seen:
        raise ValueError("Item-seen denominators require positive_seen candidates")
    columns = list(features) + ["group_id", "item_id", "label", "positive_local"]
    if has_seen:
        columns.append("positive_seen")
    positions = {name: index for index, name in enumerate(features)}
    rows, pending, pending_id, next_id = [], None, None, 0

    def zero_until(stop):
        nonlocal next_id
        while next_id < stop:
            group = groups[next_id]
            if group["pool_size"] != 0:
                raise ValueError("Cache omitted a nonempty protocol pool")
            rows.append(score_cached_group(group, [], [], [], [] if has_seen else None,
                                           [], [], {stage: [] for stage in stages}))
            next_id += 1

    def flush():
        nonlocal next_id
        if pending_id != next_id:
            raise ValueError("Cache group order is inconsistent")
        rows.append(score_cached_group(groups[pending_id], **pending))
        next_id += 1

    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        matrix = np.column_stack([batch.column(name).to_numpy(zero_copy_only=False)
                                  for name in features]).astype(np.float32, copy=False)
        if not np.isfinite(matrix).all():
            raise ValueError("Cache model features are not finite")
        if artifact.get("disable_microcategory_prior"):
            for name in ("microcat_token_probability", "microcat_token_rank"):
                if name not in positions:
                    raise ValueError("Disabled microcategory prior is absent from schema")
                matrix[:, positions[name]] = 0
        predicted = dict(prefix_scores(artifact["model"], matrix, stages))
        ids = batch.column("group_id").to_pylist()
        items = batch.column("item_id").to_pylist()
        labels = batch.column("label").to_pylist()
        local = batch.column("positive_local").to_pylist()
        seen = batch.column("positive_seen").to_pylist() if has_seen else None
        for index, group_id in enumerate(ids):
            if not isinstance(group_id, int) or not 0 <= group_id < len(groups):
                raise ValueError("Candidate group ID has no metadata")
            if pending_id != group_id:
                if pending_id is not None:
                    flush()
                zero_until(group_id)
                if group_id != next_id:
                    raise ValueError("Candidate groups are not sorted or repeat")
                pending_id = group_id
                pending = {"items": [], "labels": [], "local": [],
                           "seen": [] if has_seen else None, "baseline": [], "rrf": [],
                           "predictions": {stage: [] for stage in stages}}
            pending["items"].append(items[index])
            pending["labels"].append(labels[index])
            pending["local"].append(local[index])
            if has_seen:
                pending["seen"].append(seen[index])
            pending["baseline"].append(bool(matrix[index, positions["baseline_top50"]] > 0))
            pending["rrf"].append(float(matrix[index, positions["rrf_score"]]))
            for stage in stages:
                pending["predictions"][stage].append(float(predicted[stage][index]))
    if pending_id is not None:
        flush()
    zero_until(len(groups))
    return rows


def aggregate(rows, hit_name, denominator_name):
    usable = [row for row in rows if int(row[denominator_name]) > 0]
    by_text = defaultdict(list)
    for row in usable:
        hits, denominator = int(row[hit_name]), int(row[denominator_name])
        if not 0 <= hits <= denominator:
            raise ValueError("Recall numerator exceeds denominator")
        by_text[normalize_query_text(row["query_text"])].append(hits / denominator)
    representatives = [row for row in usable if row["representative"]]
    return {
        "groups": len(usable), "texts": len(by_text),
        "representative_groups": len(representatives),
        "text_macro_recall_at_50": (sum(sum(values) / len(values) for values in by_text.values())
                                    / len(by_text) if by_text else None),
        "group_macro_recall_at_50": (sum(sum(values) for values in by_text.values())
                                     / len(usable) if usable else None),
        "representative_macro_recall_at_50": (sum(row[hit_name] / row[denominator_name]
                                                  for row in representatives)
                                               / len(representatives) if representatives else None),
        "micro_recall_at_50": (sum(row[hit_name] for row in usable)
                                / sum(row[denominator_name] for row in usable) if usable else None),
    }


def summarize(rows, stage):
    sources = {"reranker": f"s{stage}_hits", "baseline_rrf": "baseline_hits", "pool": "pool_hits"}
    result = {
        "conditional_available": {name: aggregate(rows, hits, "available_relevant_count")
                                  for name, hits in sources.items()},
        "full_truth": {name: aggregate(rows, hits, "full_relevant_count")
                       for name, hits in sources.items()},
        "coverage": {
            "groups": len(rows), "texts": len({row["query_text"] for row in rows}),
            "groups_without_available_target": sum(row["available_relevant_count"] == 0 for row in rows),
            "full_relevant_pairs": sum(row["full_relevant_count"] for row in rows),
            "available_relevant_pairs": sum(row["available_relevant_count"] for row in rows),
            "group_macro_available_fraction": sum(row["available_relevant_count"] / row["full_relevant_count"]
                                                   for row in rows) / len(rows),
            "maximum_pool_size": max(row["pool_size"] for row in rows),
            "average_pool_size": sum(row["pool_size"] for row in rows) / len(rows),
        }, "slices": {},
    }
    for kind, denominator in SLICE_DENOMINATORS.items():
        if all(denominator in row and f"s{stage}_{kind}_hits" in row for row in rows):
            result["slices"][kind] = {
                "reranker": aggregate(rows, f"s{stage}_{kind}_hits", denominator),
                "baseline_rrf": aggregate(rows, f"baseline_{kind}_hits", denominator),
                "pool": aggregate(rows, f"pool_{kind}_hits", denominator),
            }
    for key in ("has_filters", "no_exact_location", "one_word"):
        if all(key in row for row in rows):
            for value in (False, True):
                subset = [row for row in rows if bool(row[key]) == value]
                if subset:
                    result["slices"][f"{key}={str(value).lower()}"] = {
                        name: aggregate(subset, hits, "available_relevant_count")
                        for name, hits in sources.items()}
    return result


def choose_stage(reports):
    scores = {int(stage): report["conditional_available"]["reranker"]["text_macro_recall_at_50"]
              for stage, report in reports.items()}
    if not scores or any(value is None or not math.isfinite(value) for value in scores.values()):
        raise ValueError("Stage selection needs nonempty tune text Recall")
    best = max(scores.values())
    return min(stage for stage, value in scores.items() if abs(value - best) <= 1e-12)


def paired_bootstrap(rows, stage, *, replicates=2000, seed=20261013):
    by_text = defaultdict(list)
    for row in rows:
        denominator = row["available_relevant_count"]
        if denominator:
            by_text[normalize_query_text(row["query_text"])].append(
                (row[f"s{stage}_hits"] - row["baseline_hits"]) / denominator)
    deltas = [sum(values) / len(values) for values in by_text.values()]
    if not deltas or replicates < 1:
        raise ValueError("Bootstrap needs conditional-available texts and positive replicates")
    rng = random.Random(seed)
    sampled = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(replicates))
    low, high = int(0.025 * (replicates - 1)), int(0.975 * (replicates - 1))
    return {"metric": POLICY["selection_metric"], "comparison": "selected minus baseline RRF",
            "query_text_clusters": len(deltas), "replicates": replicates, "seed": seed,
            "delta": sum(deltas) / len(deltas), "percentile_95_interval": [sampled[low], sampled[high]]}


def make_selection(protocol_hash, model_hash, cache_hashes, report_hash, stage, reports, train_hash,
                   cache_paths):
    if (set(reports) != {str(value) for value in TUNE_STAGES}
            or stage not in TUNE_STAGES or stage != choose_stage(reports)):
        raise ValueError("Selection does not match preregistered tune winner")
    payload = {"selection_version": 1, "policy": POLICY,
               "protocol_sha256": protocol_hash, "model_sha256": model_hash,
               "tune_cache_sha256": cache_hashes, "tune_report_sha256": report_hash,
               "tune_cache_paths": {name: str(Path(path).resolve()) for name, path in cache_paths.items()},
               "train_data_sha256": train_hash, "selected_stage": stage,
               "selected_tune_text_recall": reports[str(stage)]["conditional_available"]["reranker"]["text_macro_recall_at_50"]}
    return {**payload, "seal_sha256": canonical_sha256(payload)}


def verify_selection(selection, protocol_hash, model_hash, output_dir, train_hash):
    payload = {key: value for key, value in selection.items() if key != "seal_sha256"}
    if (selection.get("seal_sha256") != canonical_sha256(payload)
            or selection.get("selection_version") != 1 or selection.get("policy") != POLICY
            or selection.get("protocol_sha256") != protocol_hash
            or selection.get("model_sha256") != model_hash
            or selection.get("train_data_sha256") != train_hash
            or selection.get("selected_stage") not in TUNE_STAGES):
        raise ValueError("Tune selection seal, protocol, model, training file, or policy differs")
    paths, hashes = selection.get("tune_cache_paths"), selection.get("tune_cache_sha256")
    if (not isinstance(paths, dict) or not isinstance(hashes, dict)
            or set(paths) != {"parquet", "metadata"} or set(hashes) != set(paths)):
        raise ValueError("Tune selection lacks its complete cache seal")
    for name, path in paths.items():
        if not Path(path).is_file() or sha256_file(Path(path)) != hashes[name]:
            raise ValueError(f"Sealed tune cache is missing or changed: {name}")
    report_path = Path(output_dir) / "tune-report.json"
    if not report_path.is_file() or sha256_file(report_path) != selection.get("tune_report_sha256"):
        raise ValueError("Sealed tune report is missing or changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (set(report["stages"]) != {str(value) for value in TUNE_STAGES}
            or choose_stage(report["stages"]) != selection["selected_stage"]):
        raise ValueError("Sealed selection differs from tune report winner")
    return int(selection["selected_stage"])


def check_training_data(path, artifact, protocol_dir, manifest):
    """The trained rows must use this model's own cache and only permitted texts."""
    if path is None:
        raise ValueError("Fresh evaluation requires the model's own proven training cache")
    import pyarrow.parquet as pq
    path = Path(path)
    digest = sha256_file(path)
    provenance = artifact.get("provenance", {})
    if provenance.get("training_data_sha256") != digest:
        raise ValueError("Supplied train data is not the model's proven training cache")
    metadata_path = path.with_suffix(path.suffix + ".json")
    if (not metadata_path.is_file()
            or sha256_file(metadata_path) != provenance.get("training_metadata_sha256")):
        raise ValueError("Training metadata is not the model's proven metadata cache")
    training_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sampled = training_metadata.get("stats", {}).get("sampled_queries")
    if (not isinstance(sampled, int) or isinstance(sampled, bool) or not 1 <= sampled <= 30000
            or artifact.get("metadata", {}).get("stats", {}).get("sampled_queries") != sampled):
        raise ValueError("Fresh training must record at most 30000 sampled query groups")
    if training_metadata.get("feature_names") != artifact.get("metadata", {}).get("feature_names"):
        raise ValueError("Training feature metadata differs from the fitted model")
    fit_path = protocol_file(protocol_dir, manifest, "fit-texts.json")
    allowed = set(json.loads(fit_path.read_text(encoding="utf-8"))["texts"])
    parquet = pq.ParquetFile(path)
    if not {"query_text", "group_id", "query_key_json"} <= set(parquet.schema_arrow.names):
        raise ValueError("Fresh training cache must preserve normalized query texts and group identities")
    groups = set()
    for batch in parquet.iter_batches(batch_size=65536, columns=["query_text", "group_id", "query_key_json"]):
        for text, group_id, key_json in zip(*(batch.column(index).to_pylist() for index in range(3))):
            if text != normalize_query_text(text) or text not in allowed:
                raise ValueError("Reranker training cache contains a non-fit or unnormalized query text")
            key = json.loads(key_json)
            if (not isinstance(key, list) or len(key) != 5
                    or any(not isinstance(value, str) or value != normalize_query_text(value) for value in key)
                    or key[0] != text
                    or json.dumps(key, ensure_ascii=False, separators=(",", ":")) != key_json
                    or hashlib.sha256(key_json.encode("utf-8")).hexdigest() != group_id):
                raise ValueError("Training query group identity or normalized feature metadata is invalid")
            groups.add(group_id)
            if len(groups) > 30000 or len(groups) > sampled:
                raise ValueError("Training cache exceeds its sealed sampled-query group budget")
    if not groups:
        raise ValueError("Fresh training cache has no query groups")
    return digest


def check_encoder_identity(artifact, manifest, *, fingerprint=None):
    """A new reranker cannot silently use a different or newly trained encoder."""
    if fingerprint is None:
        from avito_candidates.dense import local_checkpoint_content_fingerprint
        fingerprint = local_checkpoint_content_fingerprint
    encoder = manifest.get("encoder", {})
    expected = encoder.get("content_sha256")
    if (not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)
            or artifact.get("metadata", {}).get("embedding_checkpoint_sha256") != expected):
        raise ValueError("Reranker encoder identity differs from the sealed cold-text protocol")
    source = Path(encoder.get("path", ""))
    configured = artifact.get("metadata", {}).get("retrieval_config", {}).get("embedding_model")
    if not source.is_dir() or not isinstance(configured, str) or not Path(configured).is_dir():
        raise ValueError("Fresh protocol requires the live frozen local encoder checkpoint")
    paths = {source.resolve(), Path(configured).resolve()}
    if any(fingerprint(path) != expected for path in paths):
        raise ValueError("Live encoder contents differ from the sealed cold-text protocol")
    stats_path = source.parent / "data_stats.json"
    if (not stats_path.is_file() or sha256_file(stats_path) != encoder.get("data_stats_sha256")
            or json.loads(stats_path.read_text(encoding="utf-8")) != encoder.get("data_stats")):
        raise ValueError("Frozen encoder training statistics changed or are missing")
    stats = encoder["data_stats"]
    if stats.get("seed") != manifest.get("split", {}).get("query_seed"):
        raise ValueError("Frozen encoder training split differs from the fresh protocol")
    if artifact["metadata"]["retrieval_config"].get("embedding_max_length") != 128:
        raise ValueError("Fresh cold-text protocol retains the frozen 128-token encoder settings")
    return expected


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("tune", "audit"), required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True,
                        help="Model's proven grouped training cache, with at most 30000 sampled groups")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Same directory for tune selection and subsequent audit")
    parser.add_argument("--batch-size", type=int, default=16384)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / f"{args.phase}-report.json").exists():
        raise ValueError("This phase already has a report; do not overwrite a completed evaluation")
    manifest, protocol_hash = checked_protocol(args.protocol_dir)
    model_hash = sha256_file(args.model)
    import joblib
    artifact = joblib.load(args.model)
    if artifact.get("estimator", "histgb") != "histgb" or maximum_iterations(artifact["model"]) < 150:
        raise ValueError("Fresh protocol requires an HGB model fitted to at least 150 iterations")
    if artifact.get("selected_iteration") is not None:
        raise ValueError("Use the unselected fresh fitted model for sealed prefix evaluation")
    encoder_hash = check_encoder_identity(artifact, manifest)
    train_hash = check_training_data(args.train_data, artifact, args.protocol_dir, manifest)
    selection_path = args.output_dir / "selection.json"
    if args.phase == "audit":
        if not selection_path.is_file():
            raise ValueError("Audit cannot open before tune seals selection.json")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        stages = (verify_selection(selection, protocol_hash, model_hash, args.output_dir, train_hash),)
    else:
        if selection_path.exists():
            raise ValueError("Tune selection already exists and cannot be replaced")
        stages = TUNE_STAGES
    cache_path, cache_metadata_path = phase_cache_paths(args.cache_dir, args.phase)
    metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
    for key in ("feature_names", "retrieval_config", "corpus_fingerprint"):
        if key not in metadata or key not in artifact.get("metadata", {}):
            raise ValueError(f"Fresh model and cache must record {key}")
    ensure_compatible(artifact["metadata"], metadata, "fresh HGB")
    prior_policy = ensure_prior_policy(artifact, metadata, "fresh HGB")
    parquet_hash = check_phase_cache_inputs(cache_path, metadata, args.phase, protocol_hash, model_hash)
    if sha256_file(args.model) != model_hash:
        raise ValueError("Fresh model changed before opening protocol labels")
    if sha256_file(args.protocol_dir / "manifest.json") != protocol_hash:
        raise ValueError("Fresh protocol changed before opening its label partition")
    # Parsing labels comes after the seal, model/encoder/training provenance,
    # saved priors, cache compatibility and independent Parquet integrity checks.
    labels = read_phase_labels(args.protocol_dir, manifest, args.phase)
    groups = bind_cache_groups(metadata, labels, args.phase, protocol_hash, model_hash)
    cache_hashes = {"parquet": parquet_hash, "metadata": sha256_file(cache_metadata_path)}
    rows = score_cache(cache_path, metadata, groups, artifact, stages, args.batch_size)
    reports = {str(stage): summarize(rows, stage) for stage in stages}
    report = {"report_version": 1, "phase": args.phase, "policy": POLICY,
              "protocol_sha256": protocol_hash, "model_sha256": model_hash,
              "encoder_content_sha256": encoder_hash,
              "cache_sha256": cache_hashes, "train_data_sha256": train_hash,
              "inference_prior_policy": prior_policy, "stages": reports,
              "scope": "Fresh cold-query tail cohort; sampled label-blind inventory, not the original official benchmark. Scores must not be compared directly with historical Recall 0.85.",
              "overfitting_note": "Training log loss measures clicked versus sampled unobserved rows; it is not percentage error. Only tune chooses capacity; audit is scored at that selected capacity.",
              "sampled_training_row_loss": (sampled_train_losses(args.train_data, artifact, stages, args.batch_size)
                                            if args.train_data else {"applicable": False, "reason": "No own training cache supplied"})}
    selected = choose_stage(reports) if args.phase == "tune" else stages[0]
    report["selected_stage"] = selected
    report["paired_selected_vs_rrf"] = paired_bootstrap(rows, selected)
    if args.phase == "tune":
        best = reports[str(selected)]["conditional_available"]["reranker"]["text_macro_recall_at_50"]
        final = reports["150"]["conditional_available"]["reranker"]["text_macro_recall_at_50"]
        losses = report["sampled_training_row_loss"].get("stages", {})
        loss_change = (losses["150"]["weighted_log_loss"] - losses[str(selected)]["weighted_log_loss"]
                       if "150" in losses and str(selected) in losses else None)
        report["tune_capacity_diagnostic"] = {
            "final_150_minus_selected_text_recall": final - best,
            "weighted_training_loss_150_minus_selected": loss_change,
            "possible_capacity_overfit": (final < best - 1e-12 and loss_change is not None and loss_change < 0),
            "note": "Tune selected the stage, so this decline is a diagnostic, not an unbiased estimate of its advantage. Audit does not inspect the final-150 alternative.",
        }
    scores_path = args.output_dir / f"{args.phase}-group-scores.jsonl"
    with scores_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    report["group_scores_sha256"] = sha256_file(scores_path)
    report_path = args.output_dir / f"{args.phase}-report.json"
    write_new_json(report_path, report)
    if args.phase == "tune":
        selection = make_selection(protocol_hash, model_hash, cache_hashes, sha256_file(report_path),
                                   selected, reports, train_hash,
                                   {"parquet": cache_path, "metadata": cache_metadata_path})
        write_new_json(selection_path, selection)
    print(json.dumps({"phase": args.phase, "selected_stage": selected,
                      "report": str(report_path),
                      "conditional_text_recall": reports[str(selected)]["conditional_available"]["reranker"]["text_macro_recall_at_50"],
                      "full_truth_text_recall": reports[str(selected)]["full_truth"]["reranker"]["text_macro_recall_at_50"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
