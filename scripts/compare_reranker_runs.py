"""Сравнение двух сохранённых выдач на одинаковых запросах и разметке."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from avito_candidates.cli import read, read_validation
from avito_candidates.core import query_key


def read_scores(path):
    """Восстанавливаем упорядоченные списки объявлений из отчёта RRF."""
    predictions = defaultdict(list)
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = query_key(row)
            if int(row["rank"]) != len(predictions[key]) + 1:
                raise ValueError(f"Нарушен порядок рангов для запроса {key}")
            predictions[key].append(row["item_id"])
    return dict(predictions)


def compare(truth, baseline, reranked, *, seed=42, bootstrap_samples=2000,
            corpus_locations=None, benchmark_queries=None):
    """Считаем парный Recall и интервал с пересэмплированием текстов."""
    if not baseline or set(baseline) != set(reranked) or not set(baseline) <= set(truth):
        raise ValueError("Оба отчёта должны содержать одинаковые размеченные запросы")
    missing = set(truth) - set(baseline)
    if len(missing) > max(5, len(truth) // 100):
        raise ValueError("Отчёты неполные: слишком много запросов без строк выдачи")
    # Запросы без найденных кандидатов отсутствуют в CSV, но дают нулевой Recall.
    keys = sorted(truth)
    base_scores = np.array([
        len(set(baseline.get(key, ())[:50]) & truth[key]) / len(truth[key]) for key in keys
    ])
    model_scores = np.array([
        len(set(reranked.get(key, ())[:50]) & truth[key]) / len(truth[key]) for key in keys
    ])
    differences = model_scores - base_scores

    # Один текст может встречаться во многих группах с разными локациями.
    # Для интервала пересэмплируем тексты целиком, сохраняя их группы вместе.
    text_indices = defaultdict(list)
    for index, key in enumerate(keys):
        text_indices[key[0]].append(index)
    groups = list(text_indices.values())
    cluster_sums = np.array([differences[indices].sum() for indices in groups])
    cluster_sizes = np.array([len(indices) for indices in groups])
    cluster_means = cluster_sums / cluster_sizes
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_samples)
    for index in range(bootstrap_samples):
        sample = rng.integers(0, len(groups), size=len(groups))
        bootstrap[index] = cluster_sums[sample].sum() / cluster_sizes[sample].sum()

    filter_slices = {}
    for name, has_filter in (("без_фильтров", False), ("с_фильтрами", True)):
        selected = [index for index, key in enumerate(keys) if bool(key[3]) == has_filter]
        filter_slices[name] = {
            "запросов": len(selected),
            "baseline_recall_at_50": float(base_scores[selected].mean()) if selected else None,
            "reranker_recall_at_50": float(model_scores[selected].mean()) if selected else None,
            "прирост": float(differences[selected].mean()) if selected else None,
        }
    result = {
        "запросов": len(keys),
        "групп_без_кандидатов": len(missing),
        "уникальных_текстов": len(groups),
        "baseline_recall_at_50": float(base_scores.mean()),
        "reranker_recall_at_50": float(model_scores.mean()),
        "прирост": float(differences.mean()),
        "95pct_интервал_прироста_по_текстам": [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
        ],
        "средний_прирост_при_равном_весе_текстов": float(cluster_means.mean()),
        "групп_лучше": int(np.count_nonzero(differences > 0)),
        "групп_хуже": int(np.count_nonzero(differences < 0)),
        "групп_без_изменений": int(np.count_nonzero(differences == 0)),
        "фильтры": filter_slices,
    }
    if corpus_locations is not None and benchmark_queries is not None:
        # Перевзвешивание учитывает только наблюдаемые фильтры и локации;
        # различия разметки внутри каждой группы оно устранить не может.
        stratum_indices = defaultdict(list)
        for index, key in enumerate(keys):
            stratum_indices[(bool(key[3]), key[1] in corpus_locations)].append(index)
        benchmark_counts = Counter(
            (bool(row["search_infm_params_text"].strip()),
             row["search_location_id"] in corpus_locations)
            for row in benchmark_queries
        )
        strata = {}
        weighted_base = weighted_model = 0.0
        for has_filter in (False, True):
            for has_exact_location in (False, True):
                stratum = (has_filter, has_exact_location)
                selected = stratum_indices[stratum]
                weight = benchmark_counts[stratum] / len(benchmark_queries)
                if not selected and weight:
                    raise ValueError("Для структуры бенчмарка нет оценочных запросов")
                base_mean = float(base_scores[selected].mean()) if selected else 0.0
                model_mean = float(model_scores[selected].mean()) if selected else 0.0
                name = (f"фильтр_{int(has_filter)}_"
                        f"точная_локация_{int(has_exact_location)}")
                strata[name] = {
                    "оценочных_запросов": len(selected),
                    "запросов_бенчмарка": benchmark_counts[stratum],
                    "baseline_recall_at_50": base_mean,
                    "reranker_recall_at_50": model_mean,
                    "прирост": model_mean - base_mean,
                }
                weighted_base += weight * base_mean
                weighted_model += weight * model_mean
        result["срезы_по_фильтрам_и_локации"] = strata
        result["приближение_к_распределению_бенчмарка"] = {
            "baseline_recall_at_50": weighted_base,
            "reranker_recall_at_50": weighted_model,
            "прирост": weighted_model - weighted_base,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--baseline-scores", type=Path, required=True)
    parser.add_argument("--reranked-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()

    items = pq.read_table(args.data_dir / "benchmark_items.parquet",
                          columns=["item_id", "item_location_id"])
    item_ids = set(items["item_id"].to_pylist())
    corpus_locations = set(str(value or "") for value in
                           items["item_location_id"].to_pylist())
    corpus_locations.discard("")
    benchmark_queries = read(args.data_dir / "benchmark_queries.parquet",
                             ("search_infm_params_text", "search_location_id"))
    truth, *_ = read_validation(
        args.data_dir / "train.parquet", item_ids, mode="queries",
        fraction=args.validation_fraction, seed=args.seed, batch_size=32768,
    )
    result = compare(
        truth, read_scores(args.baseline_scores), read_scores(args.reranked_scores),
        seed=args.seed, corpus_locations=corpus_locations,
        benchmark_queries=benchmark_queries,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
