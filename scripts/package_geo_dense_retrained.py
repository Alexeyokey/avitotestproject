#!/usr/bin/env python3
"""Package a completed, independently checked experiment without large caches.

Run from the WSL research checkout after summarize_geo_dense_retraining.py.
The package preserves the fitted model and its selected runtime prefix; it
does not claim compatibility with the ordinary ANN-based avito predict route.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import shutil
import subprocess


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.experiment_dir, args.output_dir
    summary = read(root / "summary/summary.json")
    report = read(root / "full-tune/report.json")
    if not (summary["verified"] and report["full_tune_complete"]
            and summary["groups"] == 12103 and summary["coverage"]["maximum_pool_size"] <= 998):
        raise ValueError("A complete independently verified full Tune report is required")
    if output.exists():
        raise FileExistsError(output)
    model = Path(summary["selected_model"]["path"])
    if sha(model) != summary["selected_model"]["sha256"]:
        raise ValueError("Selected model has changed since verification")
    subprocess.run(["git", "diff", "--cached", "--check"], check=True)
    unstaged = git("diff", "--name-only")
    if unstaged:
        raise ValueError("Stage reviewed changes before archiving: " + unstaged)
    tree = git("write-tree")
    output.mkdir(parents=True)
    files = ["experiment-plan.json", "deployment-handoff.txt", "fit.log", "pipeline.log", "prepare-fit-vectors.log",
             "prepare-training.log", "train-geo005-v8.parquet.json", "full-tune.log",
             "full-tune/report.json", "full-tune/declaration.json",
             "full-tune/tune-features.parquet.json", "summary/summary.json", "summary/SUMMARY.txt"]
    for relative in files:
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    handoff = output / "deployment-handoff.txt"
    handoff.write_text("Эта техническая записка составлена до окончания Tune. Итоговый статус и метрики находятся в FINAL_REPORT.txt.\n\n"
                       + handoff.read_text(encoding="utf-8"), encoding="utf-8")
    # fit-reranker prints its JSON result to stdout; --output is not used there.
    fit = read(root / "fit.log")
    (output / "fit-result.json").write_text(json.dumps(fit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(model, output / model.name)
    for sidecar in model.parent.glob(model.name + "*"):
        if sidecar != model and sidecar.is_file():
            shutil.copy2(sidecar, output / sidecar.name)
    with (root / "full-tune/group-scores.jsonl").open("rb") as source:
        with gzip.open(output / "full-tune/group-scores.jsonl.gz", "wb") as destination:
            shutil.copyfileobj(source, destination)
    archive = output / ("source-tree-" + tree[:12] + ".zip")
    subprocess.run(["git", "archive", "--format=zip", "--output=" + str(archive.resolve()), tree], check=True)
    state = {"base_commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
             "staged_tree": tree, "archive": archive.name, "committed": False,
             "note": "Archive contains the staged reviewed source tree, including earlier geo work. No new commit or push."}
    (output / "CODE_STATE.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    table = {row["name"]: row for row in summary["table"]}
    stage = summary["selected_stage"]
    selected = table[f"retrained_hgb{stage}"]
    old = table["old_hgb100_on_geo005"]
    delta = 100 * (selected["recall_at_50"] - old["recall_at_50"])
    header = ("Переобучение реранкера на географическом dense-пуле — 14 октября 2026\n\n"
              f"Выбран HGB с runtime-префиксом {stage} деревьев из 150 обученных.\n"
              f"Recall@50 на том же геопуле: {100*old['recall_at_50']:.5f}% → "
              f"{100*selected['recall_at_50']:.5f}% ({delta:+.5f} п.п.).\n"
              "Это результат на исследовательском Tune, не внешняя оценка соревнования.\n"
              "Пул, эмбеддинги объявлений и состав признаков не менялись при сравнении старого и нового HGB.\n\n")
    (output / "FINAL_REPORT.txt").write_text(header + (root / "summary/SUMMARY.txt").read_text(encoding="utf-8"), encoding="utf-8")
    reproduce = """ВОСПРОИЗВЕДЕНИЕ И ОГРАНИЧЕНИЯ ПАКЕТА

Исходный checkout: /home/alex/avitotestproject-reranker-new-octen
Python: /home/alex/avitotestproject/.venv/bin/python
PYTHONPATH=src:scripts HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4

Порядок эксперимента сохранён в scripts/run_geo_dense_retraining_after_prepare.sh.
Не запускайте этот оркестратор повторно в существующие каталоги: он защищает результаты от перезаписи.
Точный запуск оценки: bash scripts/run_geo_dense_retrained_eval.sh <НОВЫЙ_КАТАЛОГ> --bootstrap-replicates 2000 --with-training-loss
Независимая проверка: python scripts/summarize_geo_dense_retraining.py --report <НОВЫЙ_КАТАЛОГ>/report.json --output-dir <НОВЫЙ_КАТАЛОГ_СВОДКИ>

В архиве source-tree — полное дерево исходников, но не датасеты, эмбеддинги или зависимости.
Для повторного расчёта нужны исходные fit/Tune данные, protocol, frozen encoder и локальные кеши по путям из report.json.
Большой кеш Tune оставлен в WSL: full-tune/tune-features.parquet. Он запрещён для обучения; его назначение — быстрая оценка на том же Tune.

Модель хранит все 150 деревьев и selected_iteration. Потребитель обязан учитывать selected_iteration.
Нужен retrieval exact-cosine-plus-location-log-prior-v1: geo_dense_weight=0.05, smoothing_mass=3, distance_scale_km=40, probability_floor=1e-7.
Финальный location_bonus=0.02, максимум пула=998, dense_k=449. Geo-поправка применяется ДО отбора dense-кандидатов.
Эта модель привязана к исследовательскому корпусу 200000 объявлений. Обычный avito predict с прежним ANN-пулом не воспроизводит эксперимент.
В этом шаге новый production-путь predict/answer.csv не реализован и не запускался.

Split выполнен по нормализованным текстам. Fit-группы не смешивались с Tune; Audit не использовался.
Однако Tune уже применялся для подбора поиска и числа деревьев: независимая проверка качества ещё нужна.
"""
    (output / "REPRODUCE.txt").write_text(reproduce, encoding="utf-8")
    manifest = {str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": sha(path)}
                for path in sorted(output.rglob("*")) if path.is_file()}
    (output / "PACKAGE_SHA256.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "files": len(manifest), "selected_stage": stage,
                      "selected_model_sha256": sha(output / model.name)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
