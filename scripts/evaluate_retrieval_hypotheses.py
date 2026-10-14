"""Compare retrieval hypotheses on the fresh Tune arm, reusing saved pools.

The producer supplies only ranked item identifiers and train-only provenance.
This evaluator never retrieves, fits a model, inserts a relevant item, opens
Audit labels, or modifies the completed fresh-validation reports. It reports
candidate Recall separately from the unchanged original RRF Recall@50.

Supplement files are JSONL in the selected cache-group order. Each row is
``{group_id: canonical_sha256(query_key_json), query_key: [five strings],
channels: {name: [ordered unique item_id, ...]}}``. A companion JSON binds the
file, protocol, corpus, producer and train-only model provenance. See
``validate_supplement_metadata`` for the explicit contract.

Fill variants preserve every original candidate. A matched family fills the
same number of *novel* candidates per group across its members, allowing a
microcategory route to be compared with unrestricted retrieval at equal size.
Replace variants use explicit route quotas whose sum may not exceed 998.
The metrics remain Tune results and are unsuitable as independent evidence
after trying a hypothesis grid.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time

import evaluate_fresh_validation as EVAL


MAXIMUM_POOL_SIZE = 998
TEXT_SELECTION_SALT = "retrieval-hypotheses-tune-texts-v1"
ITEM_PATTERN = re.compile(r"[0-9a-f]{16}")
CHANNEL_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,79}")
BASE_CHANNELS = ("title", "params", "description", "title_stem", "params_stem",
                 "title_geo", "params_geo", "description_geo", "title_stem_geo",
                 "params_stem_geo", "dense")


def choose_texts(groups, limit=0, *, salt=TEXT_SELECTION_SALT):
    """Pick texts from identities only; preserve every selected text's context."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("Text limit must be a nonnegative integer")
    texts = {EVAL.normalize_query_text(group["query_text"]) for group in groups}
    ordered = sorted(texts, key=lambda text: (hashlib.sha256(json.dumps(
        [salt, text], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).digest(), text))
    return set(ordered[:limit] if limit else ordered)


def identity_for_key(key):
    if (not isinstance(key, list) or len(key) != 5
            or any(not isinstance(value, str) for value in key)):
        raise ValueError("Query key must contain exactly five strings")
    raw = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_variants(payload):
    if payload.get("experiment_version") != 1 or payload.get("phase") != "tune":
        raise ValueError("Only a version-1 Tune hypothesis declaration is accepted")
    variants = payload.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("At least one explicit hypothesis variant is required")
    names, families = set(), defaultdict(list)
    for variant in variants:
        name, mode = variant.get("name"), variant.get("mode")
        if (not isinstance(name, str) or not CHANNEL_PATTERN.fullmatch(name)
                or name in names or mode not in {"fill", "replace"}):
            raise ValueError("Variants require unique safe names and fill/replace mode")
        names.add(name)
        channels = variant.get("channels")
        if mode == "fill":
            if (not isinstance(channels, dict) or not channels
                    or any(not CHANNEL_PATTERN.fullmatch(key) for key in channels)
                    or any(isinstance(weight, bool) or not isinstance(weight, (int, float))
                           or not math.isfinite(weight) or weight <= 0
                           for weight in channels.values())):
                raise ValueError("Fill channels must map route names to positive finite weights")
            quota = variant.get("addition_quota")
            if (quota is not None and (isinstance(quota, bool) or not isinstance(quota, int)
                                      or not 0 <= quota <= MAXIMUM_POOL_SIZE)):
                raise ValueError("Distinct addition quota must be null or an integer within the budget")
            family = variant.get("matched_addition_family")
            if family is not None:
                if not isinstance(family, str) or not CHANNEL_PATTERN.fullmatch(family):
                    raise ValueError("Matched family must have a safe explicit name")
                families[family].append(variant)
        else:
            if "matched_addition_family" in variant or "addition_quota" in variant:
                raise ValueError("Replacement quotas cannot use a fill-family policy")
            if not isinstance(channels, dict) or not channels:
                raise ValueError("Replacement requires explicit route quotas")
            total = 0
            for name, declaration in channels.items():
                if (not CHANNEL_PATTERN.fullmatch(name) or not isinstance(declaration, dict)
                        or declaration.get("source") not in {"base", "new"}):
                    raise ValueError("Replacement routes must declare base/new source")
                quota = declaration.get("quota")
                if (isinstance(quota, bool) or not isinstance(quota, int) or quota < 0
                        or declaration["source"] == "base" and name not in BASE_CHANNELS):
                    raise ValueError("Replacement route quota or base channel name is invalid")
                total += quota
            if not 1 <= total <= MAXIMUM_POOL_SIZE:
                raise ValueError("Sum of replacement route quotas must be between 1 and 998")
    for family, members in families.items():
        if len(members) < 2 or len({member.get("addition_quota") for member in members}) != 1:
            raise ValueError(f"Matched family {family!r} needs at least two equal-quota controls")
    by_name = {variant["name"]: variant for variant in variants}
    for variant in variants:
        reference = variant.get("comparison_reference")
        if reference is not None and (reference not in by_name or reference == variant["name"]
                or by_name[reference]["mode"] != variant["mode"]):
            raise ValueError("Comparison reference must name a different variant with the same mode")
    return variants


def file_identity(entry, *, label):
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise ValueError(f"Missing {label} path identity")
    path = Path(entry["path"])
    digest = entry.get("sha256")
    if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not path.is_file() or EVAL.sha256_file(path) != digest):
        raise ValueError(f"Missing or changed {label} bytes")
    return {"path": str(path.resolve()), "sha256": digest}


def candidate_metadata(candidate_path, protocol_dir, cache_dir, producer_path, *,
                       model_path=None, provenance_kind="fitted", text_limit=0,
                       extra=None):
    """Shared producer-side metadata contract; reads inventories, not Tune GT.

    Call only after the candidate stream is successfully closed. ``extra`` is
    recorded under a separate producer-details key and cannot override guards.
    Dense-only/lexical-only variants may use frozen_encoder provenance; a new
    fitted microcategory predictor must supply its actual hashed model file.
    """
    manifest, protocol_hash = EVAL.checked_protocol(protocol_dir)
    cache_path, metadata_path = EVAL.phase_cache_paths(cache_dir, "tune")
    cache = json.loads(metadata_path.read_text(encoding="utf-8"))
    cache_hash = EVAL.check_phase_cache_inputs(cache_path, cache, "tune", protocol_hash,
                                              cache.get("model_sha256"))
    texts = choose_texts(cache["groups"], text_limit)
    selected = [group for group in cache["groups"] if group["query_text"] in texts]
    provenance = {"kind": provenance_kind,
        "fit_texts_sha256": manifest["files"]["fit_texts"]["sha256"],
        "labels_fit_sha256": manifest["files"]["labels_fit"]["sha256"]}
    if provenance_kind == "fitted":
        if model_path is None:
            raise ValueError("Fitted producer must provide the actual train-only model file")
        provenance["model"] = {"path": str(Path(model_path).resolve()),
                               "sha256": EVAL.sha256_file(model_path)}
    elif provenance_kind == "frozen_encoder":
        provenance["encoder"] = {"path": manifest["encoder"]["path"],
                                  "content_sha256": manifest["encoder"]["content_sha256"]}
    else:
        raise ValueError("Producer provenance must be fitted or frozen_encoder")
    return {"supplement_version": 1, "phase": "tune", "complete": True,
        "protocol_sha256": protocol_hash, "base_cache_sha256": cache_hash,
        "base_metadata_sha256": EVAL.sha256_file(metadata_path),
        "corpus_fingerprint": cache["corpus_fingerprint"],
        "candidate_file_sha256": EVAL.sha256_file(candidate_path),
        "groups": len(selected),
        "group_ids_sha256": EVAL.canonical_sha256([group["group_identity"] for group in selected]),
        "text_limit": text_limit, "text_selection_salt": TEXT_SELECTION_SALT,
        "producer": {"path": str(Path(producer_path).resolve()),
                     "sha256": EVAL.sha256_file(producer_path)},
        "train_only_provenance": provenance,
        "tune_targets_used_for_fit_or_retrieval": False,
        "audit_targets_used": False, "relevant_items_inserted": False,
        "producer_details": extra or {}}


def validate_supplement_metadata(metadata, path, manifest, protocol_hash,
                                 cache_hash, cache_metadata, selected_groups, text_limit,
                                 *, cache_metadata_hash):
    """Check provenance before the Tune GT is used in any candidate metric.

    ``train_only_provenance`` contains sealed fit-label/text hashes and either
    ``kind=fitted`` plus a hashed model file, or ``kind=frozen_encoder`` plus
    the unchanged encoder content hash sealed by the protocol. The producer's
    declarations are recorded; hashes cannot prove a producer's behavior.
    """
    expected_ids = [group["group_identity"] for group in selected_groups]
    if (metadata.get("supplement_version") != 1 or metadata.get("phase") != "tune"
            or metadata.get("complete") is not True
            or metadata.get("protocol_sha256") != protocol_hash
            or metadata.get("base_cache_sha256") != cache_hash
            or metadata.get("base_metadata_sha256") != cache_metadata_hash
            or metadata.get("corpus_fingerprint") != cache_metadata.get("corpus_fingerprint")
            or metadata.get("candidate_file_sha256") != EVAL.sha256_file(path)
            or metadata.get("groups") != len(expected_ids)
            or metadata.get("group_ids_sha256") != EVAL.canonical_sha256(expected_ids)
            or metadata.get("text_limit") != text_limit
            or metadata.get("text_selection_salt") != TEXT_SELECTION_SALT):
        raise ValueError("Supplement file, protocol, group scope, corpus or base-cache identity differs")
    if (metadata.get("tune_targets_used_for_fit_or_retrieval") is not False
            or metadata.get("audit_targets_used") is not False
            or metadata.get("relevant_items_inserted") is not False):
        raise ValueError("Supplement producer must explicitly declare train-only, label-blind retrieval")
    provenance = metadata.get("train_only_provenance", {})
    if (provenance.get("fit_texts_sha256") != manifest["files"]["fit_texts"]["sha256"]
            or provenance.get("labels_fit_sha256") != manifest["files"]["labels_fit"]["sha256"]):
        raise ValueError("Supplement model provenance differs from the allowed training partition")
    kind = provenance.get("kind")
    if kind == "fitted":
        file_identity(provenance.get("model"), label="train-only fitted model")
    elif kind == "frozen_encoder":
        encoder = provenance.get("encoder", {})
        if (encoder.get("path") != manifest["encoder"]["path"]
                or encoder.get("content_sha256") != manifest["encoder"]["content_sha256"]):
            raise ValueError("Query variant uses a different frozen encoder")
        # Use the same directory digest as the protocol builder, not a new rule.
        from avito_candidates.dense import local_checkpoint_content_fingerprint
        if local_checkpoint_content_fingerprint(Path(encoder["path"])) != encoder["content_sha256"]:
            raise ValueError("Frozen encoder bytes changed")
    else:
        raise ValueError("Supplement provenance must be fitted or frozen_encoder")
    file_identity(metadata.get("producer"), label="supplement producer")
    return provenance


def open_candidates(path):
    """Both ordinary and compressed JSONL stay streaming."""
    path = Path(path)
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def iter_supplements(path, scope_groups, selected_ids, corpus_ids):
    """A full-scope producer can be reused by a label-blind smaller evaluation."""
    with open_candidates(path) as stream:
        for group in scope_groups:
            line = next(stream, None)
            if line is None:
                raise ValueError("Supplement omitted a declared Tune group")
            row = json.loads(line)
            if (row.get("group_id") != group["group_identity"]
                    or row.get("query_key") != group["query_key"]
                    or identity_for_key(row.get("query_key")) != row.get("group_id")):
                raise ValueError("Supplement rows differ from declared Tune cache order")
            if group["group_id"] in selected_ids:
                yield group["group_id"], validate_supplement_row(row, group, corpus_ids)
        if next(stream, None) is not None:
            raise ValueError("Supplement contains extra or repeated Tune groups")


def validate_supplement_row(row, group, corpus_ids):
    if (row.get("group_id") != group["group_identity"]
            or row.get("query_key") != group["query_key"]
            or identity_for_key(row.get("query_key")) != row.get("group_id")):
        raise ValueError("Supplement rows must match selected cache groups in order")
    channels = row.get("channels")
    if (not isinstance(channels, dict) or not channels
            or any(not isinstance(name, str) or not CHANNEL_PATTERN.fullmatch(name)
                   for name in channels)):
        raise ValueError("Supplement must contain named ranked channels")
    for name, items in channels.items():
        if (not isinstance(items, list) or len(items) != len(set(items))
                or any(not isinstance(item, str) or not ITEM_PATTERN.fullmatch(item)
                       or item not in corpus_ids for item in items)):
            raise ValueError(f"Supplement channel {name!r} has duplicate, invalid or non-catalog IDs")
    return channels


def recover_ranked_channel(items, inverse_ranks, presence, rank_constant):
    ordered, ranks = [], set()
    for item, value, exists in zip(items, inverse_ranks, presence):
        if exists not in (0, 0.0, 1, 1.0) or not math.isfinite(value):
            raise ValueError("Cached route presence or inverse rank is invalid")
        if not exists:
            if value != 0:
                raise ValueError("Absent cached route has a nonzero inverse rank")
            continue
        if value <= 0:
            raise ValueError("Present cached route needs a positive inverse rank")
        rank = round(1 / value - rank_constant)
        if rank < 1 or rank in ranks or abs(value - 1 / (rank_constant + rank)) > 1e-8:
            raise ValueError("Cached route ranks cannot be recovered consistently")
        ranks.add(rank)
        ordered.append((rank, item))
    if ranks != set(range(1, len(ranks) + 1)):
        raise ValueError("Cached ranked channel has gaps")
    return [item for _, item in sorted(ordered)]


def validate_base_group(raw, group, *, rank_constant=60):
    items = raw["item_id"]
    if (len(items) != group["pool_size"] or len(items) != len(set(items))
            or any(not ITEM_PATTERN.fullmatch(item) for item in items)):
        raise ValueError("Saved base candidate pool is incomplete or has duplicate IDs")
    expected = [int(item in group["_available_ids"]) for item in items]
    if raw["label"] != expected or sum(expected) != group["positives_in_pool"]:
        raise ValueError("Saved base positive hits differ from sealed Tune labels")
    baseline = raw["baseline_top50"]
    if (any(value not in (0, 0.0, 1, 1.0) for value in baseline)
            or sum(baseline) > 50
            or sum(label for label, chosen in zip(expected, baseline) if chosen) != group["baseline_hits"]):
        raise ValueError("Saved original RRF top50 differs from cache metadata")
    channels = {}
    for name in BASE_CHANNELS:
        channels[name] = recover_ranked_channel(items, raw[f"rank_{name}"],
                                                raw[f"present_{name}"], rank_constant)
    return {"items": items, "channels": channels, "pool_hits": sum(expected),
            "baseline_hits": group["baseline_hits"]}


def iter_base_groups(path, metadata, groups, selected_ids, *, batch_size=65536):
    """Carry selected complete pools across batches; never hold the full cache."""
    import numpy as np
    import pyarrow.parquet as pq
    columns = ["group_id", "item_id", "label", "baseline_top50"]
    columns += [f"{prefix}_{name}" for name in BASE_CHANNELS for prefix in ("rank", "present")]
    parquet = pq.ParquetFile(path)
    if not set(columns) <= set(parquet.schema_arrow.names) or batch_size < 1:
        raise ValueError("Base cache lacks the exact ranked-route columns or batch size is invalid")
    pending, pending_id, next_id, observed = None, None, 0, 0

    def empty_before(stop):
        nonlocal next_id
        while next_id < stop:
            group = groups[next_id]
            if group["pool_size"]:
                raise ValueError("Saved cache omitted a nonempty base pool")
            if group["group_id"] in selected_ids:
                yield group, {name: [] for name in columns if name != "group_id"}
            next_id += 1

    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        ids = batch.column("group_id").to_numpy(zero_copy_only=False)
        if (not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0)
                or np.any(ids >= len(groups)) or np.any(ids[1:] < ids[:-1])):
            raise ValueError("Base cache group IDs must be valid and sorted")
        arrays = {name: batch.column(name).to_pylist() for name in columns if name != "group_id"}
        cuts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
        for start, end in zip(cuts[:-1], cuts[1:]):
            group_id = int(ids[start])
            if group_id != pending_id:
                if pending_id is not None:
                    if pending_id != next_id:
                        raise ValueError("Saved base pools repeat or are out of order")
                    if pending is not None:
                        yield groups[pending_id], pending
                    next_id += 1
                yield from empty_before(group_id)
                if group_id != next_id:
                    raise ValueError("Saved base pools repeat or are out of order")
                pending_id = group_id
                pending = ({name: [] for name in arrays} if group_id in selected_ids else None)
            if pending is not None:
                for name, values in arrays.items():
                    pending[name].extend(values[int(start):int(end)])
        observed += len(ids)
    if pending_id is not None:
        if pending_id != next_id:
            raise ValueError("Saved base pools are out of order")
        if pending is not None:
            yield groups[pending_id], pending
        next_id += 1
    yield from empty_before(len(groups))
    if observed != metadata["rows"]:
        raise ValueError("Saved cache row count differs from the sealed metadata")


def ordered_novel(channels, weights, base_ids, *, rank_constant=60):
    scores = defaultdict(float)
    for name, weight in weights.items():
        if name not in channels:
            raise ValueError(f"Declared supplemental channel {name!r} is missing")
        for rank, item in enumerate(channels[name], 1):
            if item not in base_ids:
                scores[item] += weight / (rank_constant + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))


def form_variant_pools(base, new_channels, variants):
    """Apply identity-only pool policies, before looking at a positive set."""
    base_ids = set(base["items"])
    if len(base_ids) > MAXIMUM_POOL_SIZE:
        raise ValueError("Base pool already exceeds the research candidate budget")
    novel, requested, families = {}, {}, defaultdict(list)
    for variant in variants:
        if variant["mode"] != "fill":
            continue
        name = variant["name"]
        novel[name] = ordered_novel(new_channels, variant["channels"], base_ids)
        quota = variant.get("addition_quota")
        requested[name] = min(MAXIMUM_POOL_SIZE - len(base_ids), len(novel[name]),
                              quota if quota is not None else MAXIMUM_POOL_SIZE)
        if variant.get("matched_addition_family"):
            families[variant["matched_addition_family"]].append(name)
    for names in families.values():
        common = min(requested[name] for name in names)
        for name in names:
            requested[name] = common
    result = {}
    for variant in variants:
        name = variant["name"]
        if variant["mode"] == "fill":
            result[name] = [*base["items"], *novel[name][:requested[name]]]
        else:
            pool, seen = [], set()
            for channel, policy in variant["channels"].items():
                source = base["channels"] if policy["source"] == "base" else new_channels
                if channel not in source:
                    raise ValueError(f"Declared replacement channel {channel!r} is missing")
                for item in source[channel][:policy["quota"]]:
                    if item not in seen:
                        pool.append(item)
                        seen.add(item)
            result[name] = pool
        if len(result[name]) > MAXIMUM_POOL_SIZE or len(result[name]) != len(set(result[name])):
            raise ValueError("Hypothesis pool is repeated or exceeds the research budget")
    return result


def score_group(group, base, variant_pools):
    available, base_ids = group["_available_ids"], set(base["items"])
    row = {key: value for key, value in group.items() if not key.startswith("_")}
    row.update(base_pool_hits=base["pool_hits"], base_pool_size=len(base_ids),
               baseline_hits=base["baseline_hits"])
    for name, items in variant_pools.items():
        pool = set(items)
        row[f"{name}_pool_hits"] = len(pool & available)
        row[f"{name}_pool_size"] = len(pool)
        row[f"{name}_new_candidates"] = len(pool - base_ids)
        row[f"{name}_removed_candidates"] = len(base_ids - pool)
        row[f"{name}_positive_additions"] = len((pool - base_ids) & available)
        row[f"{name}_positive_losses"] = len((base_ids - pool) & available)
    return row


def recall_summary(rows, hits, denominator, *, candidate=False):
    values = EVAL.aggregate(rows, hits, denominator)
    if candidate:
        values = {key.replace("recall_at_50", "candidate_recall"): value for key, value in values.items()}
    return values


def paired_pool_bootstrap(rows, name, *, replicates=2000, seed=20261013):
    by_text = defaultdict(list)
    for row in rows:
        denominator = row["available_relevant_count"]
        if denominator:
            by_text[EVAL.normalize_query_text(row["query_text"])].append(
                (row[f"{name}_pool_hits"] - row["base_pool_hits"]) / denominator)
    if not by_text or replicates < 1:
        raise ValueError("Candidate bootstrap requires conditional-available texts and replicates")
    deltas = [sum(values) / len(values) for values in by_text.values()]
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(replicates))
    return {"metric": "conditional_available_text_macro_candidate_recall",
            "comparison": f"{name} minus unchanged original pool", "query_text_clusters": len(deltas),
            "replicates": replicates, "seed": seed, "delta": sum(deltas) / len(deltas),
            "percentile_95_interval": [draws[int(.025 * (replicates - 1))],
                                       draws[int(.975 * (replicates - 1))]],
            "scope": "Exploratory Tune comparison; not independent evidence after hypothesis selection"}


def matched_pair_bootstrap(rows, left, right, *, replicates=2000, seed=20261013):
    projected = [{**row, "base_pool_hits": row[f"{right}_pool_hits"]} for row in rows]
    result = paired_pool_bootstrap(projected, left, replicates=replicates, seed=seed)
    result["comparison"] = f"{left} minus {right}, exactly matched addition count per group"
    if any(row[f"{left}_pool_size"] != row[f"{right}_pool_size"] for row in rows):
        raise ValueError("Matched controls differ in their realized per-group pool sizes")
    return result


def paired_variant_bootstrap(rows, left, right, *, replicates=2000, seed=20261013):
    """A query-input control shares retrieval depths; union overlap may differ."""
    projected = [{**row, "base_pool_hits": row[f"{right}_pool_hits"]} for row in rows]
    result = paired_pool_bootstrap(projected, left, replicates=replicates, seed=seed)
    result["comparison"] = f"{left} minus declared reference variant {right}"
    result["scope"] += "; route quotas are declared, realized unique-pool sizes may differ"
    return result


def summarize(rows, variants, *, bootstrap_replicates=2000):
    if not rows:
        raise ValueError("Hypothesis evaluation has no selected Tune groups")
    result = {"reference": {}, "variants": {}, "matched_controls": {}, "coverage": {
        "groups": len(rows), "texts": len({row["query_text"] for row in rows}),
        "groups_without_available_target": sum(row["available_relevant_count"] == 0 for row in rows),
        "available_relevant_pairs": sum(row["available_relevant_count"] for row in rows),
        "full_relevant_pairs": sum(row["full_relevant_count"] for row in rows),
        "positive_pair_availability": sum(row["available_relevant_count"] for row in rows)
                                      / sum(row["full_relevant_count"] for row in rows)}}
    for scope, denominator in (("conditional_available", "available_relevant_count"),
                               ("full_truth", "full_relevant_count")):
        result["reference"][scope] = {
            "pool": recall_summary(rows, "base_pool_hits", denominator, candidate=True),
            "original_rrf_top50": recall_summary(rows, "baseline_hits", denominator)}
    result["reference"]["pool_statistics"] = {
        "average_size": sum(row["base_pool_size"] for row in rows) / len(rows),
        "maximum_size": max(row["base_pool_size"] for row in rows)}
    families = defaultdict(list)
    for variant in variants:
        name = variant["name"]
        report = {"configuration": variant, "pool_statistics": {
            "average_size": sum(row[f"{name}_pool_size"] for row in rows) / len(rows),
            "maximum_size": max(row[f"{name}_pool_size"] for row in rows),
            "average_new_candidates": sum(row[f"{name}_new_candidates"] for row in rows) / len(rows),
            "average_removed_candidates": sum(row[f"{name}_removed_candidates"] for row in rows) / len(rows),
            "positive_additions": sum(row[f"{name}_positive_additions"] for row in rows),
            "positive_losses": sum(row[f"{name}_positive_losses"] for row in rows)},
            "paired_vs_original_pool": paired_pool_bootstrap(
                rows, name, replicates=bootstrap_replicates), "slices": {}}
        for scope, denominator in (("conditional_available", "available_relevant_count"),
                                   ("full_truth", "full_relevant_count")):
            report[scope] = recall_summary(rows, f"{name}_pool_hits", denominator, candidate=True)
        for field in ("one_word", "has_filters", "no_exact_location"):
            for value in (False, True):
                subset = [row for row in rows if bool(row.get(field)) == value]
                if subset:
                    report["slices"][f"{field}={str(value).lower()}"] = {
                        "variant": recall_summary(subset, f"{name}_pool_hits", "available_relevant_count", candidate=True),
                        "reference": recall_summary(subset, "base_pool_hits", "available_relevant_count", candidate=True)}
        if variant.get("comparison_reference"):
            report["paired_vs_declared_reference"] = paired_variant_bootstrap(
                rows, name, variant["comparison_reference"], replicates=bootstrap_replicates)
        result["variants"][name] = report
        if variant.get("matched_addition_family"):
            families[variant["matched_addition_family"]].append(name)
    for family, names in families.items():
        reference = names[0]
        result["matched_controls"][family] = {
            "reference_variant": reference,
            "exact_per_group_size_matching": True,
            "comparisons": {name: matched_pair_bootstrap(rows, name, reference,
                replicates=bootstrap_replicates) for name in names[1:]}}
    return result


def evaluate(args):
    """Read the allowed Tune inventory and scan only selected full base pools."""
    if args.output.exists():
        raise ValueError("Experiment output already exists; choose a distinct path")
    configuration = json.loads(args.configuration.read_text(encoding="utf-8"))
    variants = validate_variants(configuration)
    manifest, protocol_hash = EVAL.checked_protocol(args.protocol_dir)
    cache_path, metadata_path = EVAL.phase_cache_paths(args.cache_dir, "tune")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cache_hash = EVAL.check_phase_cache_inputs(cache_path, metadata, "tune", protocol_hash,
                                              metadata.get("model_sha256"))
    selected_texts = choose_texts(metadata["groups"], args.text_limit)
    cache_metadata_hash = EVAL.sha256_file(metadata_path)
    if len(args.supplement) != len(args.supplement_metadata):
        raise ValueError("Each supplementary file needs its own metadata argument")
    supplement_entries = []
    for candidate_path, declaration_path in zip(args.supplement, args.supplement_metadata):
        declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
        producer_limit = declaration.get("text_limit")
        if producer_limit not in {0, args.text_limit}:
            raise ValueError("Producer scope must be the full Tune arm or exactly the selected text limit")
        scope_texts = choose_texts(metadata["groups"], producer_limit)
        scope_groups = [group for group in metadata["groups"] if group["query_text"] in scope_texts]
        validate_supplement_metadata(declaration, candidate_path, manifest, protocol_hash,
                                     cache_hash, metadata, scope_groups, producer_limit,
                                     cache_metadata_hash=cache_metadata_hash)
        supplement_entries.append({"path": candidate_path, "metadata_path": declaration_path,
            "metadata": declaration, "scope_groups": scope_groups,
            "metadata_sha256": EVAL.sha256_file(declaration_path)})
    inventory_path = EVAL.protocol_file(args.protocol_dir, manifest, "catalog-item-ids.json")
    raw_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory = raw_inventory.get("item_ids", raw_inventory.get("ids"))
    if (not isinstance(inventory, list) or len(inventory) != manifest["corpus"]["size"]
            or len(set(inventory)) != len(inventory)
            or any(not isinstance(item, str) or not ITEM_PATTERN.fullmatch(item) for item in inventory)):
        raise ValueError("Sealed research catalog item inventory is invalid")
    corpus_ids = set(inventory)
    # No labels are read before the input, producer and partition guards pass.
    labels = EVAL.read_phase_labels(args.protocol_dir, manifest, "tune")
    groups = EVAL.bind_cache_groups(metadata, labels, "tune", protocol_hash, metadata["model_sha256"])
    selected_ids = {group["group_id"] for group in groups if group["query_text"] in selected_texts}
    rows = []
    started = time.monotonic()
    streams = [iter_supplements(entry["path"], entry["scope_groups"], selected_ids, corpus_ids)
               for entry in supplement_entries]
    for group, raw in iter_base_groups(cache_path, metadata, groups, selected_ids,
                                       batch_size=args.batch_size):
        channels = {}
        for stream in streams:
            value = next(stream, None)
            if value is None or value[0] != group["group_id"]:
                raise ValueError("Supplement omitted or reordered a selected Tune group")
            for name, items in value[1].items():
                if name in channels:
                    raise ValueError("Repeated channel names across supplementary files")
                channels[name] = items
        base = validate_base_group(raw, group, rank_constant=metadata["retrieval_config"]["rrf_k"])
        if not set(base["items"]) <= corpus_ids:
            raise ValueError("Saved base candidate pool contains non-catalog items")
        pools = form_variant_pools(base, channels, variants)
        rows.append(score_group(group, base, pools))
        if len(rows) % 512 == 0:
            print(f"Evaluated Tune retrieval groups: {len(rows)}/{len(selected_ids)}; "
                  f"elapsed={time.monotonic() - started:.1f}s", file=sys.stderr, flush=True)
    if any(next(stream, None) is not None for stream in streams):
        raise ValueError("Supplement contains extra selected Tune groups")
    if len(rows) != len(selected_ids):
        raise ValueError("Selected Tune groups are incomplete")
    result = summarize(rows, variants, bootstrap_replicates=args.bootstrap_replicates)
    report = {"report_version": 1, "phase": "tune", "complete": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_hash, "base_cache_sha256": cache_hash,
        "base_metadata_sha256": cache_metadata_hash,
        "configuration": configuration, "configuration_sha256": EVAL.sha256_file(args.configuration),
        "supplement_metadata": [entry["metadata"] for entry in supplement_entries],
        "supplement_metadata_sha256": [entry["metadata_sha256"] for entry in supplement_entries],
        "text_selection": {"limit": args.text_limit, "salt": TEXT_SELECTION_SALT,
                           "texts_sha256": EVAL.canonical_sha256(sorted(selected_texts))},
        "results": result,
        "limitations": [
            "Only Tune is scored; no independent Audit labels or scores are opened.",
            "Full-truth denominators include unavailable targets and zero-availability queries.",
            "Candidate Recall is a pool upper bound, not the final reranker's Recall@50.",
            "The original RRF top50 reference is unchanged; no new-pool reranker is fitted or scored.",
            "Tune hypothesis-grid confidence intervals do not remove selection bias.",
            "Producer/model hashes seal identity, but declarations cannot prove label-blind behavior.",
            "This sampled cold catalog is not the official benchmark or its expected score.",
        ]}
    # Recheck files at closure: a changing producer/candidate/config must fail.
    if (any(EVAL.sha256_file(entry["path"]) != entry["metadata"]["candidate_file_sha256"]
            or EVAL.sha256_file(entry["metadata_path"]) != entry["metadata_sha256"]
            for entry in supplement_entries)
            or EVAL.sha256_file(args.configuration) != report["configuration_sha256"]
            or EVAL.sha256_file(metadata_path) != report["base_metadata_sha256"]):
        raise ValueError("Experiment input changed while evaluating")
    for entry in supplement_entries:
        file_identity(entry["metadata"]["producer"], label="supplement producer at closure")
        provenance = entry["metadata"]["train_only_provenance"]
        if provenance["kind"] == "fitted":
            file_identity(provenance["model"], label="train-only fitted model at closure")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents accidentally replacing a completed result.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, action="append", required=True,
                        help="Repeat for separately produced ranked-channel JSONL or JSONL.gz")
    parser.add_argument("--supplement-metadata", type=Path, action="append", required=True,
                        help="Repeat once per supplementary file, in the same order")
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-limit", type=int, default=0,
                        help="Label-blind number of Tune texts; zero keeps every Tune text")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args(argv)
    try:
        report = evaluate(args)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report["results"], ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
