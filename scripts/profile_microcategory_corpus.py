"""Inventory-domain diagnostic; no relevance partitions, candidate targets or inference.

Only column batches from the original inventories and frozen metadata catalog
are read. Full-train fit/heldout clicked category counts are descriptive; no
query-to-item evaluation labels or per-query targets are opened or exported.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import pyarrow.parquet as pq

from avito_candidates.core import held_out, normalize


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def columns(path, names):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16384, columns=names):
        yield from zip(*(batch.column(index).to_pylist() for index in range(len(names))))


_canonical_ids = {}
def category(value):
    if value is None:
        return ""
    return _canonical_ids.setdefault(value, str(value))


def item_id(value):
    if not isinstance(value, str) or len(value) != 16:
        raise ValueError("Expected a canonical 16-character hexadecimal item ID")
    return int(value, 16)


def counter_report(counter):
    return [{"category_id": key, "rows": value} for key, value in
            sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))]


started = time.monotonic()
protocol = Path("artifacts/fresh-validation-2026-10-13/protocol")
output = Path("artifacts/microcategory-research-2026-10-13")
manifest_path = protocol / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
full_train = Path(manifest["sources"]["train"]["path"])
original_benchmark = Path(manifest["sources"]["benchmark_items"]["path"])
catalog = protocol / manifest["files"]["benchmark_items"]["path"]
catalog_ids_path = protocol / manifest["files"]["catalog_item_ids"]["path"]
model = output / "soft-microcategory-oof3.joblib"
model_sidecar = model.with_suffix(model.suffix + ".json")
envelope = json.loads(model_sidecar.read_text(encoding="utf-8"))
if digest(model) != envelope["model_sha256"]:
    raise ValueError("Classifier weights changed; this diagnostic does not load them")
classes = set(envelope["metadata"]["classes"])

expected_sha = {
    full_train: manifest["sources"]["train"]["sha256"],
    original_benchmark: manifest["sources"]["benchmark_items"]["sha256"],
    catalog: manifest["files"]["benchmark_items"]["sha256"],
    catalog_ids_path: manifest["files"]["catalog_item_ids"]["sha256"],
}
for path, expected in expected_sha.items():
    if digest(path) != expected:
        raise ValueError(f"Frozen source changed: {path}")

bench_metadata = {}
benchmark_micro, benchmark_macro = Counter(), Counter()
for iid, raw_cid, raw_macro in columns(original_benchmark,
        ("item_id", "item_microcat_id", "item_category_id")):
    iid, cid, macro = item_id(iid), category(raw_cid), category(raw_macro)
    if iid in bench_metadata:
        raise ValueError("Duplicate original benchmark ID")
    bench_metadata[iid] = (cid, macro)
    benchmark_micro[cid] += 1
    benchmark_macro[macro] += 1

train_ids = set()
split_micro = {split: Counter() for split in ("fit", "heldout")}
split_macro = {split: Counter() for split in ("fit", "heldout")}
split_rows, missing_micro, empty_text_rows = Counter(), Counter(), Counter()
overlap_mismatch_first, overlap_first = 0, 0
for raw_query, iid, raw_cid, raw_macro in columns(full_train,
        ("search_query", "item_id", "item_microcat_id", "item_category_id")):
    text = normalize(raw_query or "")
    split = "heldout" if held_out(text, .2, 42) else "fit"
    cid, macro, iid = category(raw_cid), category(raw_macro), item_id(iid)
    split_rows[split] += 1
    split_macro[split][macro] += 1
    if not text:
        empty_text_rows[split] += 1
    if not cid or cid == "0":
        missing_micro[split] += 1
    else:
        split_micro[split][cid] += 1
    if iid not in train_ids and iid in bench_metadata:
        overlap_first += 1
        overlap_mismatch_first += cid != bench_metadata[iid][0]
    train_ids.add(iid)

fit_classes, heldout_classes = set(split_micro["fit"]), set(split_micro["heldout"])
catalog_micro, catalog_macro = Counter(), Counter()
source_counts = defaultdict(Counter)
source_micro = defaultdict(Counter)
source_macro = defaultdict(Counter)
unknown_macro = defaultdict(Counter)
catalog_ids = set()
for iid, raw_cid, raw_macro in columns(catalog, ("item_id", "item_microcat_id", "item_category_id")):
    iid, cid, macro = item_id(iid), category(raw_cid), category(raw_macro)
    if iid in catalog_ids:
        raise ValueError("Duplicate frozen catalog ID")
    catalog_ids.add(iid)
    in_train, in_benchmark = iid in train_ids, iid in bench_metadata
    source = "overlap" if in_train and in_benchmark else "train_only" if in_train else "benchmark_only" if in_benchmark else "neither"
    if source == "neither":
        raise ValueError("Frozen catalog contains an item absent from both source inventories")
    status = "known" if cid in classes else "unknown"
    catalog_micro[cid] += 1
    catalog_macro[macro] += 1
    source_counts[source][status] += 1
    source_micro[source][cid] += 1
    source_macro[source][macro] += 1
    if status == "unknown":
        unknown_macro[source][macro] += 1

# This file is an inventory list, not a query's relevance or candidate list.
declared_ids = json.loads(catalog_ids_path.read_text(encoding="utf-8"))["item_ids"]
if catalog_ids != {item_id(value) for value in declared_ids}:
    raise ValueError("Catalog IDs differ from the sealed metadata-only inventory")

catalog_classes = set(catalog_micro)
unknown_classes = catalog_classes - classes
heldout_only_classes = heldout_classes - fit_classes
unknown_click_rows = {split: sum(value for cid, value in counts.items() if cid not in classes)
                      for split, counts in split_micro.items()}
source_summary = {}
for source, counts in sorted(source_counts.items()):
    total = sum(counts.values())
    source_summary[source] = {"items": total, "known_class_items": counts["known"],
        "unknown_class_items": counts["unknown"], "known_class_fraction": counts["known"] / total,
        "unique_microcategories": len(source_micro[source]),
        "macrocategory_counts": counter_report(source_macro[source]),
        "unknown_class_macrocategory_counts": counter_report(unknown_macro[source])}

report = {
    "diagnostic_schema": 1, "metadata_only_diagnostic": True,
    "no_query_to_item_relevance_files_read": True, "no_candidate_targets_read": True,
    "no_model_inference_or_model_object_load": True,
    "train_split_category_counts_include_descriptive_heldout_rows": True,
    "files_opened": [str(path) for path in (manifest_path, full_train, original_benchmark,
                                            catalog, catalog_ids_path, model, model_sidecar)],
    "source_files": {str(path): {"sha256": expected, "rows": pq.ParquetFile(path).metadata.num_rows,
                                "schema": str(pq.ParquetFile(path).schema_arrow)}
                     for path, expected in expected_sha.items() if path.suffix == ".parquet"},
    "manifest_sha256": digest(manifest_path), "model_sidecar_sha256": digest(model_sidecar),
    "classifier_model_sha256": envelope["model_sha256"],
    "classifier_training_source_sha256": envelope["metadata"]["source_sha256"],
    "inventory_policy": manifest["corpus"],
    "classifier": {"classes": len(classes), "class_labels": sorted(classes),
        "classes_in_catalog": len(classes & catalog_classes),
        "classifier_classes_absent_catalog": sorted(classes - catalog_classes),
        "fit_classes_absent_classifier": sorted(fit_classes - classes),
        "crossfit_folds": envelope["metadata"]["crossfit_folds"]},
    "original_train": {"rows": sum(split_rows.values()), "unique_items": len(train_ids),
        "fit_rows": split_rows["fit"], "heldout_rows": split_rows["heldout"],
        "fit_unique_valid_microcategories": len(fit_classes),
        "heldout_unique_valid_microcategories": len(heldout_classes),
        "full_unique_valid_microcategories": len(fit_classes | heldout_classes),
        "heldout_only_microcategories": sorted(heldout_only_classes),
        "heldout_only_click_rows": sum(split_micro["heldout"][cid] for cid in heldout_only_classes),
        "unknown_classifier_click_rows": unknown_click_rows,
        "missing_microcategory_rows": dict(missing_micro), "empty_query_rows": dict(empty_text_rows),
        "fit_microcategory_counts": counter_report(split_micro["fit"]),
        "heldout_microcategory_counts": counter_report(split_micro["heldout"]),
        "fit_macrocategory_counts": counter_report(split_macro["fit"]),
        "heldout_macrocategory_counts": counter_report(split_macro["heldout"]),
        "original_inventory_overlap_items": overlap_first,
        "first_train_microcategory_differs_from_authoritative_benchmark": overlap_mismatch_first},
    "original_benchmark": {"items": len(bench_metadata), "microcategories": len(benchmark_micro),
                           "macrocategory_counts": counter_report(benchmark_macro)},
    "frozen_catalog": {"items": len(catalog_ids), "microcategories": len(catalog_classes),
        "unknown_classifier_microcategories": len(unknown_classes),
        "known_class_items": sum(counts["known"] for counts in source_counts.values()),
        "unknown_class_items": sum(counts["unknown"] for counts in source_counts.values()),
        "unknown_classes_also_in_fit": sorted(unknown_classes & fit_classes),
        "unknown_classes_heldout_only": sorted(unknown_classes & heldout_only_classes),
        "unknown_classes_absent_all_train_clicks": sorted(unknown_classes - fit_classes - heldout_classes),
        "macrocategory_counts": counter_report(catalog_macro), "source_breakdown": source_summary},
    "limits": ["Microcategory IDs are categorical codes; macrocategory concentration gives domain context, not a verified taxonomy name.",
        "Known-class status may reflect both service-domain compatibility and inventory source; this diagnostic does not distinguish their causal effects.",
        "No heldout Recall or tuning decision is computed. Unseen classes are not assumed irrelevant.",
        "This is the sampled metadata-only research catalog, not the original submission catalog."],
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "producer_sha256": digest(Path(__file__)), "runtime_seconds": time.monotonic() - started,
}
json_path = output / "microcategory-corpus-coverage.json"
if json_path.exists():
    raise FileExistsError("Refusing to replace an existing coverage diagnostic")
json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
lines = ["Диагностика покрытия микрокатегорий — без inference и relevance-файлов.",
    f"Классификатор: {len(classes)} классов; каталог: {len(catalog_classes)}; пересечение: {len(classes & catalog_classes)}.",
    f"Train fit/heldout: {split_rows['fit']}/{split_rows['heldout']} строк; классов {len(fit_classes)}/{len(heldout_classes)}.",
    f"Уникальных классов полного train: {len(fit_classes | heldout_classes)}; heldout-only: {len(heldout_only_classes)} ({report['original_train']['heldout_only_click_rows']} строк).",
    f"Класс неизвестен классификатору: fit {unknown_click_rows['fit']} строк; heldout {unknown_click_rows['heldout']} строк.",
    f"Каталог: {report['frozen_catalog']['known_class_items']} объявлений с известным классом; {report['frozen_catalog']['unknown_class_items']} с неизвестным.",
    f"Неизвестных классов вне всех кликов train: {len(report['frozen_catalog']['unknown_classes_absent_all_train_clicks'])}."]
for source, values in source_summary.items():
    lines.append(f"{source}: {values['items']} объявлений, known={values['known_class_items']} ({values['known_class_fraction']:.4%}), unknown={values['unknown_class_items']}; unknown macro={values['unknown_class_macrocategory_counts']}.")
lines.extend(["Важно: наличие класса может отражать и домен запроса, и источник корпуса. Причинный вклад здесь не измеряется.",
              "Читались только категории/идентификаторы исходных inventories, train split counts и model sidecar. Tune/Audit relevance и списки найденных кандидатов не читались.",
              "Числа heldout описывают категории строк train; они не использованы для оценки или подбора параметров.",
              f"Производитель SHA256: {report['producer_sha256']}; git: {report['git_head']}; JSON SHA256: {digest(json_path)}."])
output.joinpath("microcategory-corpus-coverage.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
