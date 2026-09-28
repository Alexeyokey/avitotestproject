"""Повторяем оценку новой Octen и обучаем реранкер на расширенном пуле."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def retrieval_arguments(report, *, data_dir, checkpoint, device, batch_size,
                        max_length, ef_search, candidate_k=None,
                        geo_candidate_k=None, location_bonus=None):
    """Восстанавливаем конфигурацию поиска из исходного отчёта оценки."""
    if report.get("method") != "hybrid" or report.get("split") != "queries":
        raise ValueError("Нужен отчёт hybrid с разбиением split=queries")
    bm25 = report["bm25_fields"]
    arguments = [
        "--data-dir", str(data_dir), "--split", "queries",
        "--seed", str(report["seed"]),
        "--validation-fraction", str(report.get("validation_fraction", 0.2)),
        "--method", "hybrid", "--device", device,
        "--embedding-model", str(checkpoint),
        "--embedding-batch-size", str(batch_size),
        "--embedding-max-length", str(max_length),
        "--embedding-params-words", str(report["embedding_params_words"]),
        "--embedding-description-words", str(report["embedding_description_words"]),
        "--ef-search", str(ef_search),
        "--candidate-k", str(candidate_k or report["candidate_k"]),
        "--rrf-k", str(report["rrf_k"]),
        "--bm25-weight", str(report["bm25_weight"]),
        "--dense-weight", str(report["dense_weight"]),
        "--channel-quota", str(report["channel_quota"]),
        "--stem-title-weight", str(bm25["title_stem"]["weight"]),
        "--stem-params-weight", str(bm25["params_stem"]["weight"]),
        "--location-bonus", str(
            bm25["location_bonus"] if location_bonus is None else location_bonus
        ),
        "--bm25-rrf-k", str(bm25["rrf_k"]),
        "--bm25-channel-quota", str(bm25["channel_quota"]),
        "--geo-candidate-k", str(
            bm25.get("geo_candidate_k", 0) if geo_candidate_k is None
            else geo_candidate_k
        ),
        "--geo-top-locations", str(bm25.get("geo_top_locations", 3)),
        "--geo-min-history", str(bm25.get("geo_min_history", 20)),
        "--geo-weight", str(bm25.get("geo_weight", 0.25)),
    ]
    for field in ("title", "params", "description"):
        for parameter in ("k1", "b", "weight"):
            arguments.extend((f"--{field}-{parameter}",
                              str(bm25[field][parameter])))
    return arguments


def run(command, *, dry_run=False):
    """Показываем точную команду и останавливаем цепочку при ошибке."""
    import shlex

    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True,
                        help="JSON полной проверки дообученной Octen")
    parser.add_argument("--embedding-model", type=Path, required=True,
                        help="Путь к сохранённой модели Sentence Transformers")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/reranker-new-octen"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-max-length", type=int, default=128)
    parser.add_argument("--ef-search", type=int, default=300)
    parser.add_argument("--target-candidate-k", type=int, default=500)
    parser.add_argument("--target-geo-candidate-k", type=int, default=100)
    parser.add_argument("--reranker-negatives-per-query", type=int, default=64)
    parser.add_argument("--reranker-max-iter", type=int, default=150)
    parser.add_argument("--reproduction-tolerance", type=float, default=0.005)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = json.loads(args.source_report.read_text(encoding="utf-8"))
    if not args.embedding_model.is_dir():
        parser.error("Путь --embedding-model должен вести к локальному чекпойнту")
    if not all((args.data_dir / name).exists() for name in
               ("train.parquet", "benchmark_items.parquet")):
        parser.error("В --data-dir нужны train.parquet и benchmark_items.parquet")
    if args.target_candidate_k <= 0 or args.target_geo_candidate_k < 0:
        parser.error("Глубина поиска должна быть положительной")
    if args.embedding_batch_size <= 0 or args.ef_search <= 0:
        parser.error("Размер пакета и ef-search должны быть положительными")
    output = args.output_dir
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    python = [sys.executable, "-m", "avito_candidates.cli"]
    source = retrieval_arguments(
        report, data_dir=args.data_dir, checkpoint=args.embedding_model.resolve(),
        device=args.device, batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length, ef_search=args.ef_search,
    )
    source_result = output / "reference-reproduced.json"
    run([*python, "evaluate", *source, "--output", str(source_result)],
        dry_run=args.dry_run)
    if not args.dry_run:
        reproduced = json.loads(source_result.read_text(encoding="utf-8"))
        delta = reproduced["recall_at_50"] - report["recall_at_50"]
        print(f"Воспроизведение: {reproduced['recall_at_50']:.6f}; "
              f"исходный отчёт: {report['recall_at_50']:.6f}; "
              f"разница: {delta:+.6f}", flush=True)
        if abs(delta) > args.reproduction_tolerance:
            raise RuntimeError(
                "Базовое качество не воспроизвелось. Проверьте промпты, "
                "чекпойнт, ef-search и версию кода до обучения реранкера."
            )

    # Расширяем пул географическими кандидатами и глубиной 500, затем
    # сравниваем обученную модель с тем же поиском на отложенных запросах.
    target = retrieval_arguments(
        report, data_dir=args.data_dir, checkpoint=args.embedding_model.resolve(),
        device=args.device, batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length, ef_search=args.ef_search,
        candidate_k=args.target_candidate_k,
        geo_candidate_k=args.target_geo_candidate_k,
    )
    training_data = output / "train.parquet"
    model = output / "model.joblib"
    final_result = output / "heldout-eval.json"
    run([*python, "prepare-reranker", *target,
         "--reranker-negatives-per-query", str(args.reranker_negatives_per_query),
         "--reranker-data", str(training_data)], dry_run=args.dry_run)
    run([*python, "fit-reranker", "--reranker-data", str(training_data),
         "--reranker-max-iter", str(args.reranker_max_iter),
         "--reranker-model", str(model)], dry_run=args.dry_run)
    run([*python, "evaluate", *target,
         "--reranker-model", str(model), "--output", str(final_result)],
        dry_run=args.dry_run)
    if not args.dry_run:
        final = json.loads(final_result.read_text(encoding="utf-8"))
        diagnostics = final["retrieval_diagnostics"]
        print(json.dumps({
            "candidate_pool_recall": diagnostics["candidate_pool_recall"],
            "baseline_recall_at_50": diagnostics["baseline_recall_at_50"],
            "reranker_recall_at_50": final["recall_at_50"],
            "reranker_delta_at_50": diagnostics["reranker_delta_at_50"],
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
