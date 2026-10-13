"""Build a frozen cold-query protocol without reusing the historical holdout.

Catalog sampling uses item IDs only. All known positives are retained in the
label files, including positives unavailable in the sampled catalog. Those
files are evaluation metadata and must never become model features.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq


REPRESENTATIVE_SALT = "fresh-validation-representative-v1"
CATALOG_SALT = "fresh-validation-catalog-v1"


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def canonical_query_key_json(key):
    if not isinstance(key, (tuple, list)) or len(key) != 5 or any(
            not isinstance(value, str) for value in key):
        raise ValueError("Query key must contain five normalized strings")
    return canonical_json(list(key))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def salted_digest(salt, value):
    return hashlib.sha256(canonical_json([salt, value]).encode("utf-8")).hexdigest()


def values_digest(values):
    return hashlib.sha256(canonical_json(sorted(values)).encode("utf-8")).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def load_manifest(protocol_dir, verify=True):
    root = Path(protocol_dir).resolve()
    if (root / "INCOMPLETE").exists():
        raise ValueError("Fresh validation protocol is incomplete")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("protocol_version") != 1 or manifest.get("complete") is not True
            or manifest.get("sealed_audit") is not True):
        raise ValueError("Expected a completed sealed fresh validation protocol")
    required = {"labels_fit", "labels_tune", "labels_audit", "fit_texts", "tune_texts",
                "audit_texts", "historical_excluded_texts", "train", "benchmark_items",
                "benchmark_queries", "catalog_item_ids"}
    files = manifest.get("files", {})
    if not required.issubset(files):
        raise ValueError("Fresh validation manifest lacks required files")
    for entry in files.values():
        relative = Path(entry["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError("Protocol file path escapes its directory")
        if verify and sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Protocol file SHA256 mismatch: {entry['path']}")
    return manifest


def load_contract():
    from avito_candidates.core import SEARCH_FIELDS, held_out, normalize, query_key
    from avito_candidates.dense import local_checkpoint_content_fingerprint
    return {"search_fields": SEARCH_FIELDS, "held_out": held_out,
            "normalize": normalize, "query_key": query_key,
            "encoder_fingerprint": local_checkpoint_content_fingerprint}


def select_catalog_ids(union_ids, size, salt):
    """No query, partition, label or positive-availability argument is accepted."""
    if size <= 0 or size > len(union_ids):
        raise ValueError("Corpus size must be positive and no larger than the catalog union")
    return set(sorted(union_ids, key=lambda item: (salted_digest(salt, item), item))[:size])


def read_groups(train_path, benchmark_ids, contract, seed, fraction, batch_size):
    """Use the same batch conversion and query-key API as the project CLI."""
    fields = tuple(contract["search_fields"])
    groups, text_rows, cold_by_text, historical = {}, Counter(), {}, set()
    for batch in pq.ParquetFile(train_path).iter_batches(
            batch_size=batch_size, columns=[*fields, "item_id"]):
        for row in batch.to_pandas().fillna("").astype(str).to_dict("records"):
            item_id = row["item_id"]
            if not re.fullmatch(r"[0-9a-f]{16}", item_id):
                raise ValueError("Train contains an invalid item_id")
            key = tuple(contract["query_key"](row))
            text = key[0]
            text_rows[text] += 1
            if text not in cold_by_text:
                cold_by_text[text] = contract["held_out"](text, fraction, seed)
            if cold_by_text[text] and item_id in benchmark_ids:
                historical.add(text)
            serialized = canonical_query_key_json(key)
            record = groups.setdefault(serialized, {"query_key": key, "text": text,
                                                     "relevant": set(), "rows": 0})
            record["relevant"].add(item_id)
            record["rows"] += 1
    return groups, text_rows, cold_by_text, historical


def partition_groups(groups, cold_by_text, historical, contract, arm_seed,
                     arm_fraction, selected_ids):
    partitions = {name: {} for name in ("fit", "tune", "audit")}
    for serialized, record in groups.items():
        text = record["text"]
        if text in historical:
            continue
        if not cold_by_text[text]:
            role = "fit"
        else:
            role = ("audit" if contract["held_out"](text, arm_fraction, arm_seed)
                    else "tune")
        partitions[role][serialized] = record
    representatives = {}
    for role, records in partitions.items():
        chosen = {}
        for serialized, record in records.items():
            score = (salted_digest(REPRESENTATIVE_SALT, serialized), serialized)
            previous = chosen.get(record["text"])
            if previous is None or score < previous:
                chosen[record["text"]] = score
        representatives[role] = {value[1] for value in chosen.values()}
    if not all(partitions.values()):
        raise ValueError("Fit, tuning and audit must each contain query groups")
    stats = {}
    for role, records in partitions.items():
        available = {key: record["relevant"] & selected_ids for key, record in records.items()}
        texts = {record["text"] for record in records.values()}
        full_pairs = sum(len(record["relevant"]) for record in records.values())
        available_pairs = sum(len(items) for items in available.values())
        stats[role] = {
            "rows": sum(record["rows"] for record in records.values()),
            "texts": len(texts), "groups": len(records),
            "unique_relevant_pairs": full_pairs,
            "available_relevant_pairs": available_pairs,
            "groups_with_available_positive": sum(bool(items) for items in available.values()),
            "texts_with_available_positive": len({records[key]["text"] for key, items
                                                  in available.items() if items}),
            "representative_groups": len(representatives[role]),
            "representative_groups_with_available_positive": sum(
                bool(available[key]) for key in representatives[role]),
            "positive_pair_availability": available_pairs / full_pairs,
            "texts_sha256": values_digest(texts),
        }
    return partitions, representatives, stats


def write_labels(path, role, records, representatives, selected_ids):
    """Stream large label manifests instead of keeping another copy in memory."""
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write('{"partition":' + canonical_json(role) + ',"groups":[\n')
        first = True
        for serialized in sorted(records, key=lambda key: hashlib.sha256(key.encode()).hexdigest()):
            record = records[serialized]
            truth = sorted(record["relevant"])
            available = sorted(record["relevant"] & selected_ids)
            row = {"group_id": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
                   "query_key": list(record["query_key"]), "query_key_json": serialized,
                   "query_text": record["text"], "representative": serialized in representatives,
                   "relevant_item_ids": truth, "available_item_ids": available,
                   "full_relevant_count": len(truth), "available_relevant_count": len(available)}
            stream.write(("" if first else ",\n") + canonical_json(row))
            first = False
        stream.write("\n]}\n")


def write_catalog(train_path, benchmark_path, output, selected_ids, batch_size):
    """Benchmark metadata wins; otherwise keep the first source train row.

    Source hashes freeze that deterministic metadata rule. Selection itself is
    independent of row order and never consults which items were clicked.
    """
    schema = pq.ParquetFile(benchmark_path).schema_arrow
    columns = schema.names
    seen = set()
    with pq.ParquetWriter(output, schema) as writer:
        for source in (benchmark_path, train_path):
            for batch in pq.ParquetFile(source).iter_batches(batch_size=batch_size, columns=columns):
                mask = []
                for item_id in batch.column(columns.index("item_id")).to_pylist():
                    keep = item_id in selected_ids and item_id not in seen
                    mask.append(keep)
                    if keep:
                        seen.add(item_id)
                if any(mask):
                    table = pa.Table.from_batches([batch]).filter(pa.array(mask))
                    writer.write_table(table.cast(schema))
    if seen != selected_ids:
        raise ValueError("Not all sampled item IDs received metadata")


def write_filtered_train(source, output, allowed_texts, normalize, batch_size):
    parquet = pq.ParquetFile(source)
    total = 0
    with pq.ParquetWriter(output, parquet.schema_arrow) as writer:
        for batch in parquet.iter_batches(batch_size=batch_size):
            raw_texts = batch.column(parquet.schema_arrow.get_field_index("search_query")).to_pylist()
            mask = [normalize(text or "") in allowed_texts for text in raw_texts]
            if any(mask):
                table = pa.Table.from_batches([batch]).filter(pa.array(mask))
                writer.write_table(table)
                total += table.num_rows
    return total


def write_placeholder_queries(path, partitions, fields):
    """Synthetic CLI identifiers are evaluation metadata, never model features."""
    seen, rows = set(), []
    for role in ("tune", "audit"):
        for serialized, record in sorted(partitions[role].items()):
            query_id = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
            if query_id in seen:
                raise ValueError("Synthetic query_id collision")
            seen.add(query_id)
            rows.append({"query_id": query_id, **dict(zip(fields, record["query_key"]))})
    pq.write_table(pa.Table.from_pylist(rows), path)


def build_protocol(data_dir, output_dir, encoder, expected_encoder_sha256, *,
                   corpus_size=200000, corpus_salt=CATALOG_SALT, query_seed=42,
                   query_fraction=0.2, arm_seed=20261013, arm_fraction=0.5,
                   batch_size=32768, contract=None):
    data_dir, output_dir, encoder = map(Path, (data_dir, output_dir, encoder))
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")
    if batch_size <= 0 or not 0 < query_fraction < 1 or not 0 < arm_fraction < 1:
        raise ValueError("Require positive batch size and fractions between zero and one")
    if not corpus_salt:
        raise ValueError("Corpus salt must not be empty")
    contract = contract or load_contract()
    encoder_sha = contract["encoder_fingerprint"](encoder)
    if encoder_sha != expected_encoder_sha256:
        raise ValueError("Frozen encoder contents differ from the expected SHA256")
    encoder_stats_path = encoder.parent / "data_stats.json"
    encoder_stats = json.loads(encoder_stats_path.read_text(encoding="utf-8"))
    if encoder_stats.get("seed") != query_seed:
        raise ValueError("Query split must match the frozen encoder training seed")
    train_path, benchmark_path = data_dir / "train.parquet", data_dir / "benchmark_items.parquet"
    source_sha = {path: sha256_file(path) for path in (train_path, benchmark_path)}
    train_schema = pq.ParquetFile(train_path).schema_arrow
    benchmark_schema = pq.ParquetFile(benchmark_path).schema_arrow
    if any(not name.startswith("item_") for name in benchmark_schema.names):
        raise ValueError("Catalog may contain only item metadata columns")
    required = {*contract["search_fields"], *benchmark_schema.names}
    if not required.issubset(train_schema.names):
        raise ValueError("Train metadata cannot reconstruct the benchmark item schema")
    benchmark_list = pq.read_table(benchmark_path, columns=["item_id"]).column(0).to_pylist()
    benchmark_ids = set(benchmark_list)
    if len(benchmark_ids) != len(benchmark_list) or any(
            not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{16}", item)
            for item in benchmark_ids):
        raise ValueError("Benchmark catalog has duplicate or invalid item IDs")
    # This pass deliberately reads no query fields or relevance structures.
    # Finish catalog selection before looking at any query-label partitions.
    train_ids = set()
    for batch in pq.ParquetFile(train_path).iter_batches(
            batch_size=batch_size, columns=["item_id"]):
        train_ids.update(batch.column(0).to_pylist())
    if any(not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{16}", item)
           for item in train_ids):
        raise ValueError("Train catalog contains invalid item IDs")
    selected_ids = select_catalog_ids(train_ids | benchmark_ids, corpus_size, corpus_salt)
    groups, text_rows, cold_by_text, historical = read_groups(
        train_path, benchmark_ids, contract, query_seed, query_fraction, batch_size)
    fit_texts = {text for text, cold in cold_by_text.items() if not cold}
    cold_texts = {text for text, cold in cold_by_text.items() if cold}
    if (encoder_stats.get("fit_query_texts") != len(fit_texts)
            or encoder_stats.get("training_pairs") != len(fit_texts)
            or encoder_stats.get("fit_rows") != sum(text_rows[text] for text in fit_texts)
            or encoder_stats.get("validation_query_texts_excluded") != len(cold_texts)):
        raise ValueError("Source train and split do not match the frozen encoder data statistics")
    partitions, representatives, stats = partition_groups(
        groups, cold_by_text, historical, contract, arm_seed, arm_fraction, selected_ids)
    # No output is created before preflight succeeds. An incomplete marker
    # remains after an interrupted write, and reruns refuse to overwrite it.
    output_dir.mkdir(parents=True)
    (output_dir / "INCOMPLETE").write_text("Protocol is not ready for evaluation.\n", encoding="utf-8")
    dataset = output_dir / "dataset"
    dataset.mkdir()
    write_catalog(train_path, benchmark_path, dataset / "benchmark_items.parquet",
                  selected_ids, batch_size)
    allowed_texts = {record["text"] for records in partitions.values() for record in records.values()}
    written_rows = write_filtered_train(train_path, dataset / "train.parquet", allowed_texts,
                                        contract["normalize"], batch_size)
    if written_rows != sum(part["rows"] for part in stats.values()):
        raise ValueError("Filtered dataset row counts disagree with query partitions")
    write_placeholder_queries(dataset / "benchmark_queries.parquet", partitions,
                              contract["search_fields"])
    for role, records in partitions.items():
        texts = sorted({record["text"] for record in records.values()})
        write_json(output_dir / f"{role}-texts.json",
                   {"texts": texts, "texts_sha256": values_digest(texts)})
        write_labels(output_dir / f"labels-{role}.json", role, records,
                     representatives[role], selected_ids)
    write_json(output_dir / "historical-excluded-texts.json",
               {"texts": sorted(historical), "texts_sha256": values_digest(historical)})
    write_json(output_dir / "catalog-item-ids.json",
               {"item_ids": sorted(selected_ids), "item_ids_sha256": values_digest(selected_ids)})
    file_paths = {"train": dataset / "train.parquet",
                  "benchmark_items": dataset / "benchmark_items.parquet",
                  "benchmark_queries": dataset / "benchmark_queries.parquet",
                  "historical_excluded_texts": output_dir / "historical-excluded-texts.json",
                  "catalog_item_ids": output_dir / "catalog-item-ids.json"}
    for role in partitions:
        file_paths[f"{role}_texts"] = output_dir / f"{role}-texts.json"
        file_paths[f"labels_{role}"] = output_dir / f"labels-{role}.json"
    files = {name: {"path": path.relative_to(output_dir).as_posix(), "sha256": sha256_file(path)}
             for name, path in file_paths.items()}
    if any(sha256_file(path) != original for path, original in source_sha.items()):
        raise ValueError("Source Parquet changed during protocol construction")
    manifest = {
        "protocol_version": 1, "complete": True, "sealed_audit": True,
        "audit_access_policy": "Open labels-audit.json only after the tuning selection is sealed",
        "sources": {name: {"path": str(path.resolve()), "sha256": source_sha[path],
                            "rows": pq.ParquetFile(path).metadata.num_rows,
                            "schema": str(pq.ParquetFile(path).schema_arrow.remove_metadata())}
                    for name, path in (("train", train_path), ("benchmark_items", benchmark_path))},
        "encoder": {"path": str(encoder.resolve()), "content_sha256": encoder_sha,
                    "data_stats": encoder_stats, "data_stats_sha256": sha256_file(encoder_stats_path)},
        "split": {"query_seed": query_seed, "query_fraction": query_fraction,
                  "arm_seed": arm_seed, "arm_fraction": arm_fraction,
                  "normalization": "avito_candidates.core.normalize",
                  "grouping": "normalized search_query",
                  "representative_rule": "min SHA256([representative_salt,query_key_json]) per text",
                  "representative_salt": REPRESENTATIVE_SALT,
                  "historical_excluded_texts": len(historical)},
        "corpus": {"size": len(selected_ids), "union_size": len(train_ids | benchmark_ids),
                   "selection_rule": "lowest SHA256([salt,item_id]), item_id tie break",
                   "salt": corpus_salt, "selected_ids_sha256": values_digest(selected_ids),
                   "selection_sha256": salted_digest(corpus_salt, sorted(selected_ids)),
                   "metadata_only": True, "forced_relevant_items": False,
                   "metadata_rule": "benchmark authoritative; otherwise first original train row"},
        "stats": stats, "files": files,
        "metrics_policy": {"selection": "tune text-macro conditional on available positive",
                           "additional": ["group-macro conditional", "representative conditional",
                                          "full-truth recall including unavailable positives", "availability"],
                           "identifier_features": False},
        "limitations": ["Cold texts are selected by absence from the historical benchmark-positive holdout",
                        "Sampled catalog differs from the submission catalog; scores are not directly comparable",
                        "Query text normalization does not guarantee unseen semantic intent",
                        "The audit seal is an access protocol, not label encryption"],
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "INCOMPLETE").unlink()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--expected-encoder-sha256", required=True)
    parser.add_argument("--corpus-size", type=int, default=200000)
    parser.add_argument("--corpus-salt", default=CATALOG_SALT)
    parser.add_argument("--query-seed", type=int, default=42)
    parser.add_argument("--query-fraction", type=float, default=0.2)
    parser.add_argument("--arm-seed", type=int, default=20261013)
    parser.add_argument("--arm-fraction", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32768)
    args = parser.parse_args()
    manifest = build_protocol(**vars(args))
    print(json.dumps({"manifest": str(args.output_dir / "manifest.json"),
                      "corpus": manifest["corpus"], "stats": manifest["stats"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
