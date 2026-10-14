"""Verify four-route production pool selection on real saved candidate lists.

Read a label-blind metadata-order prefix (default32 Tune groups). Only
group_id/item_id columns of the old pool are parsed. No GPU, vector encoding,
BM25 indexing, classifier fitting/inference, Audit files or GT are needed.
Cached route files contain ranked identifiers but no measured raw scores;
zero raw-score placeholders satisfy the production helper's interface and
are explicitly NEVER a claim of score/feature parity. Selection ignores them.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from avito_candidates import supplemental as PRODUCTION
import evaluate_retrieval_hypotheses as REFERENCE


ROUTE_MAPPING = (
    ("supplemental_title_values", "lexical_title_values"),
    ("supplemental_params_values", "lexical_params_values"),
    ("micro_combined", "micro_combined_top3"),
    ("supplemental_dense_current", "dense_current"),
)


def base_prefix(path, groups, limit, *, batch_size=65536):
    """Read only two columns and stop after the requested complete pool prefix."""
    import pyarrow.parquet as pq
    if (isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= len(groups)
            or batch_size < 1):
        raise ValueError("Parity requires a valid positive metadata-order group limit")
    pools = [[] for _ in range(limit)]
    previous, done = -1, False
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["group_id", "item_id"]):
        ids, items = (batch.column(name).to_pylist() for name in ("group_id", "item_id"))
        for group_id, item_id in zip(ids, items):
            if (isinstance(group_id, bool) or not isinstance(group_id, int)
                    or group_id < previous or not 0 <= group_id < len(groups)):
                raise ValueError("Saved base pool group IDs are invalid or out of order")
            previous = group_id
            if group_id >= limit:
                done = True
                break
            if not isinstance(item_id, str) or not REFERENCE.ITEM_PATTERN.fullmatch(item_id):
                raise ValueError("Saved base pool contains an invalid item ID")
            pools[group_id].append(item_id)
        if done:
            break
    for index, pool in enumerate(pools):
        group = groups[index]
        if (group.get("group_id") != index or len(pool) != group.get("pool_size")
                or len(pool) != len(set(pool)) or len(pool) > 998
                or REFERENCE.identity_for_key(group.get("query_key")) != group.get("group_identity")):
            raise ValueError("Saved base prefix is incomplete, repeated or differs from metadata identity")
    return pools


def frozen_inputs(cache_dir, reference_report, candidates, sidecars):
    """Bind inputs to the completed full-Tune report, without protocol GT reads."""
    if len(candidates) != len(sidecars) or not candidates:
        raise ValueError("Each cached candidate stream requires its own sidecar")
    report = json.loads(Path(reference_report).read_text(encoding="utf-8"))
    if (report.get("report_version") != 1 or report.get("phase") != "tune"
            or report.get("complete") is not True or report.get("text_selection", {}).get("limit") != 0):
        raise ValueError("Parity reference must be the completed full-Tune retrieval report")
    root = Path(cache_dir)
    cache, metadata_path = root / "tune.parquet", root / "tune.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cache_hash, metadata_hash = REFERENCE.EVAL.sha256_file(cache), REFERENCE.EVAL.sha256_file(metadata_path)
    if (metadata.get("cache_version") != 2 or metadata.get("fresh_partition") != "tune"
            or metadata.get("role") != "validation" or metadata.get("historical_holdout_reused") is not False
            or metadata.get("cache_parquet_sha256") != cache_hash
            or report.get("base_cache_sha256") != cache_hash
            or report.get("base_metadata_sha256") != metadata_hash
            or report.get("protocol_sha256") != metadata.get("fresh_validation_protocol_sha256")):
        raise ValueError("Base metadata/cache/protocol differs from the completed retrieval reference")
    reference_declarations, declaration_hashes = (report.get(name) for name in
        ("supplement_metadata", "supplement_metadata_sha256"))
    if (not isinstance(reference_declarations, list) or not isinstance(declaration_hashes, list)
            or len(reference_declarations) != len(declaration_hashes)):
        raise ValueError("Completed reference lacks its supplementary input seals")
    used, declarations, input_hashes = set(), [], []
    group_ids = [group["group_identity"] for group in metadata["groups"]]
    for candidate_path, sidecar_path in zip(candidates, sidecars):
        declaration = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
        candidate_hash, sidecar_hash = (REFERENCE.EVAL.sha256_file(path) for path in (candidate_path, sidecar_path))
        matches = [index for index, value in enumerate(reference_declarations) if value == declaration
                   and declaration_hashes[index] == sidecar_hash]
        if (len(matches) != 1 or matches[0] in used
                or declaration.get("supplement_version") != 1 or declaration.get("phase") != "tune"
                or declaration.get("complete") is not True
                or declaration.get("candidate_file_sha256") != candidate_hash
                or declaration.get("base_cache_sha256") != cache_hash
                or declaration.get("base_metadata_sha256") != metadata_hash
                or declaration.get("protocol_sha256") != report["protocol_sha256"]
                or declaration.get("corpus_fingerprint") != metadata["corpus_fingerprint"]
                or declaration.get("text_limit") != 0
                or declaration.get("groups") != len(metadata["groups"])
                or declaration.get("group_ids_sha256") != REFERENCE.EVAL.canonical_sha256(group_ids)):
            raise ValueError("Cached candidate/sidecar differs from the completed full-Tune input seal")
        used.add(matches[0]); declarations.append(declaration)
        input_hashes.append({"candidate": str(Path(candidate_path).resolve()), "candidate_sha256": candidate_hash,
                             "sidecar": str(Path(sidecar_path).resolve()), "sidecar_sha256": sidecar_hash})
    if used != set(range(len(reference_declarations))):
        raise ValueError("Parity must receive every supplementary stream used in the completed reference")
    variants = report.get("configuration", {}).get("variants", [])
    matching = [value for value in variants if value.get("name") == "F_hybrid_routes"]
    expected_channels = {source: 1 for _, source in ROUTE_MAPPING}
    if (len(matching) != 1 or matching[0].get("mode") != "fill"
            or matching[0].get("addition_quota") is not None
            or matching[0].get("matched_addition_family") is not None
            or matching[0].get("channels") != expected_channels
            or tuple(matching[0]["channels"]) != tuple(expected_channels)
            or tuple(PRODUCTION.ROUTES) != tuple(name for name, _ in ROUTE_MAPPING)
            or PRODUCTION.FILL_RANK_CONSTANT != 60 or PRODUCTION.DEPTH != 998):
        raise ValueError("Production route policy differs from the frozen four-route prototype")
    return cache, metadata_path, metadata, matching[0], input_hashes


def verify_group(base, prototype_channels, variant):
    """Compare ordered membership against the frozen evaluator, not new GT."""
    required = {source for _, source in ROUTE_MAPPING}
    if set(prototype_channels) != required:
        raise ValueError("Real cached parity input must have the four exact prototype routes")
    for name, ids in prototype_channels.items():
        if (len(ids) > 998 or len(ids) != len(set(ids))
                or any(not isinstance(item, str) or not REFERENCE.ITEM_PATTERN.fullmatch(item) for item in ids)):
            raise ValueError(f"Invalid cached raw route:{name}")
    production_routes = {name: prototype_channels[source] for name, source in ROUTE_MAPPING}
    placeholders = {name: {item: 0. for item in ids} for name, ids in production_routes.items()}
    selected, retained, retained_scores, selection_scores = PRODUCTION.select_supplemental(
        production_routes, placeholders, base, maximum_pool_size=998)
    expected = REFERENCE.form_variant_pools({"items": base}, prototype_channels, [variant])["F_hybrid_routes"]
    actual = [*base, *selected]
    if (actual != expected or actual[:len(base)] != base or len(actual) > 998
            or len(actual) != len(set(actual)) or set(selection_scores) != set(selected)):
        raise ValueError("Production ordered pool/novel prefix differs from the frozen evaluator")
    for name, ids in retained.items():
        if (ids != [item for item in production_routes[name] if item in set(selected)]
                or set(retained_scores[name]) != set(ids)
                or any(value != 0. for value in retained_scores[name].values())):
            raise ValueError("Production retained-route membership/order differs from its selected IDs")
    return {"base_size": len(base), "novel_size": len(selected), "pool_size": len(actual),
            "ordered_pool_sha256": REFERENCE.EVAL.canonical_sha256(actual),
            "ordered_novel_sha256": REFERENCE.EVAL.canonical_sha256(selected),
            "ordered_pool_matches": True, "base_order_preserved": True,
            "raw_route_sizes": {source: len(ids) for source, ids in prototype_channels.items()}}


def check(args):
    if args.output.exists():
        raise ValueError("Cached parity report already exists; choose a new path")
    cache_path, metadata_path, metadata, variant, input_hashes = frozen_inputs(
        args.cache_dir, args.reference_report, args.candidates, args.candidate_metadata)
    limit = min(args.groups, len(metadata["groups"]))
    bases = base_prefix(cache_path, metadata["groups"], limit)
    helper_path = Path(inspect.getfile(PRODUCTION.select_supplemental))
    prototype_path = Path(inspect.getfile(REFERENCE.form_variant_pools))
    before = {"helper": REFERENCE.EVAL.sha256_file(helper_path),
              "prototype": REFERENCE.EVAL.sha256_file(prototype_path),
              "metadata": REFERENCE.EVAL.sha256_file(metadata_path),
              "reference": REFERENCE.EVAL.sha256_file(args.reference_report),
              "checker": REFERENCE.EVAL.sha256_file(Path(__file__))}
    streams = [REFERENCE.open_candidates(path) for path in args.candidates]
    checks = []
    try:
        for index, base in enumerate(bases):
            group, combined = metadata["groups"][index], {}
            for stream in streams:
                line = next(stream, None)
                if line is None:
                    raise ValueError("Cached supplementary stream omitted a parity-prefix group")
                row = json.loads(line)
                if (row.get("group_id") != group["group_identity"] or row.get("query_key") != group["query_key"]
                        or REFERENCE.identity_for_key(row.get("query_key")) != row.get("group_id")):
                    raise ValueError("Cached route stream is misaligned with the original metadata order")
                for name, ids in row.get("channels", {}).items():
                    if name in combined:
                        raise ValueError("Repeated route name across cached candidate streams")
                    combined[name] = ids
            needed = {source: combined[source] for _, source in ROUTE_MAPPING}
            checks.append({"group_id": index, "group_identity": group["group_identity"],
                           **verify_group(base, needed, variant)})
    finally:
        for stream in streams:
            stream.close()
    after = {"helper": REFERENCE.EVAL.sha256_file(helper_path),
             "prototype": REFERENCE.EVAL.sha256_file(prototype_path),
             "metadata": REFERENCE.EVAL.sha256_file(metadata_path),
             "reference": REFERENCE.EVAL.sha256_file(args.reference_report),
             "checker": REFERENCE.EVAL.sha256_file(Path(__file__))}
    if before != after:
        raise ValueError("Cached parity source/metadata/reference changed while checking")
    report = {"parity_version": 1, "complete": True, "phase": "tune", "groups_checked": len(checks),
        "selection": "first metadata-order groups, no target-based selection",
        "variant": variant, "reference_report_sha256": before["reference"], "source_sha256": before,
        "base_cache": {"path": str(cache_path.resolve()), "parquet_sha256": metadata["cache_parquet_sha256"],
                       "metadata_sha256": before["metadata"], "columns_parsed": ["group_id", "item_id"]},
        "candidate_inputs": input_hashes, "route_mapping": dict(ROUTE_MAPPING),
        "all_ordered_pools_match": all(row["ordered_pool_matches"] for row in checks),
        "all_base_orders_preserved": all(row["base_order_preserved"] for row in checks),
        "maximum_pool_size": max(row["pool_size"] for row in checks),
        "raw_scores": {"present_in_cached_streams": False,
            "helper_interface_placeholders": "finite zero per item, used only by this parity check",
            "measured_raw_score_or_feature_parity_claim": False,
            "selection_depends_on_raw_scores": False},
        "labels_or_audit_used": False, "checks": checks,
        "limitations": ["This verifies ranked-list pool selection/membership/order only on the saved prefix.",
            "No BM25 score recomputation, encoder/vector generation, classifier inference or GPU is tested.",
            "Dummy raw scores are not suitable for training or inference features.",
            "Catalog membership and complete-stream seals rely on the completed full-Tune evaluator; this check scans only its prefix."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, action="append", required=True)
    parser.add_argument("--candidate-metadata", type=Path, action="append", required=True)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = check(args)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: report[key] for key in ("groups_checked", "all_ordered_pools_match",
          "all_base_orders_preserved", "maximum_pool_size", "raw_scores")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
