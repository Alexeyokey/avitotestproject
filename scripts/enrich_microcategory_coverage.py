"""Separate unspecified CID0 from rare valid classes; no relevance inputs."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


root = Path("artifacts/microcategory-research-2026-10-13")
path = root / "microcategory-corpus-coverage.json"
report = json.loads(path.read_text(encoding="utf-8"))
catalog = Path("artifacts/fresh-validation-2026-10-13/protocol/dataset/benchmark_items.parquet")
counts = Counter()
for batch in pq.ParquetFile(catalog).iter_batches(batch_size=16384, columns=["item_microcat_id"]):
    counts.update("" if value is None else str(value) for value in batch.column(0).to_pylist())
missing = {"", "0"}
valid_absent = set(report["frozen_catalog"]["unknown_classes_absent_all_train_clicks"]) - missing
report["frozen_catalog"]["unspecified_microcategory_items"] = sum(counts.get(cid, 0) for cid in missing)
report["frozen_catalog"]["unique_valid_microcategories"] = len(set(counts) - missing)
report["frozen_catalog"]["unknown_valid_classes_absent_all_train_clicks"] = sorted(valid_absent)
report["frozen_catalog"]["unknown_valid_classes_absent_all_train_clicks_count"] = len(valid_absent)
report["frozen_catalog"]["unknown_classes_absent_all_train_clicks_key_excludes_missing_in_interpretation"] = True
source = report["frozen_catalog"]["source_breakdown"]
positive_total = source["train_only"]["items"] + source["overlap"]["items"]
positive_known = source["train_only"]["known_class_items"] + source["overlap"]["known_class_items"]
negative_total, negative_known = source["benchmark_only"]["items"], source["benchmark_only"]["known_class_items"]
report["known_class_source_association"] = {
    "target": "item appears in original full train inventory (overlap counts as true)",
    "binary_score": "microcategory is a known global classifier class",
    "positive_known_fraction": positive_known / positive_total,
    "negative_known_fraction": negative_known / negative_total,
    "analytic_binary_auc": .5 + .5 * (positive_known / positive_total - negative_known / negative_total),
    "no_model_fit_or_inference": True,
    "scope": "catalog source membership, not query relevance",
}
report["enrichment_producer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
report["interpretation"] = [
    "528 raw catalog labels include the unspecified CID0. There are 527 valid catalog microcategories.",
    "The classifier models 209 classes covering 206 catalog classes but 199304/200000 actual items. Label diversity counts alone overstate the inventory gap.",
    "Of 322 unknown raw labels: CID0 is unspecified, 1 valid class is heldout-only, 320 valid classes never occur among known train clicks.",
    "Class-known is almost constant for both inventory sources; analytic binary source AUC is about 0.5052. Unknown status identifies a small unusual subset, not most benchmark-only items.",
    "This does not establish the contribution of classifier probabilities or disprove other source confounding. No relevance metric or hyperparameter decision is computed.",
]
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
train, catalog_report = report["original_train"], report["frozen_catalog"]
lines = [
    "Диагностика покрытия микрокатегорий. Корпус и model sidecar; без inference и relevance-файлов.",
    "209 обученных классов против 528 меток каталога; CID0 означает отсутствие категории, валидных классов 527.",
    "206 обученных классов присутствуют в каталоге и покрывают 199304/200000 объявлений (99,652%). Неизвестных 696 (0,348%).",
    f"Неизвестные метки: 322 = CID0 + 1 валидный класс только отложенной части + {len(valid_absent)} валидных классов вне всех кликов train.",
    f"CID0: {catalog_report['unspecified_microcategory_items']} объявлений в каталоге; fit 17 / heldout 1 строка train.",
    "Fit 404057 / heldout 93616 строк; валидных классов 209/204; полный train 211. Два класса только отложенной части составляют лишь 3 строки.",
    "Неизвестные модели валидные CID в fit: 0 строк; heldout: 3 строки. Классы модели вне каталога: 1178078, 2171444, 2303720.",
    "Макрокатегория 114: 404048/404057 fit-строк (99,9978%); 199256/200000 объявлений каталога (99,628%). Название макрокатегории здесь не восстанавливается.",
    "Первый CID train и авторитетный CID benchmark совпадают на всех 18142 пересекающихся ID.",
]
for name, values in source.items():
    lines.append(f"{name}: items={values['items']}; known={values['known_class_items']} ({values['known_class_fraction']:.4%}); unknown={values['unknown_class_items']}.")
lines.extend([
    f"Бинарный known-class как признак источника train/benchmark-only: AUC={report['known_class_source_association']['analytic_binary_auc']:.6f}. Это аналитическая ассоциация источников, не Recall и не обученная модель.",
    "Разница в количестве классов в основном относится к редким категориям каталога. Простое знание класса почти постоянно для обоих источников; называть его сильным общим source proxy оснований нет.",
    "Вероятности классификатора всё ещё могут зависеть от состава корпуса. Причинный вклад этим подсчётом не установлен, неизвестные классы не объявляются нерелевантными.",
    "Tune/Audit relevance-файлы, найденные кандидаты и их положительные targetID не читались. Heldout использован лишь для описательного подсчёта категорий train.",
    f"Основнойproducer SHA={report['producer_sha256']}; enrichment SHA={report['enrichment_producer_sha256']}; git={report['git_head']}.",
    f"Итоговый JSON SHA256={hashlib.sha256(path.read_bytes()).hexdigest()}",
])
root.joinpath("microcategory-corpus-coverage.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
