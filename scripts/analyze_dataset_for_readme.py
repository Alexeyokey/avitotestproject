"""Считает проверяемые характеристики train и тестовых файлов для README."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from avito_candidates.core import SEARCH_FIELDS, held_out, normalize


def _search_rows(path: Path, *, with_item: bool) -> pd.DataFrame:
    columns = [*SEARCH_FIELDS, "item_id"] if with_item else list(SEARCH_FIELDS)
    rows = pq.read_table(path, columns=columns).to_pandas()
    for name in SEARCH_FIELDS:
        rows[name] = rows[name].fillna("").astype(str)
    if with_item:
        rows["item_id"] = rows["item_id"].fillna("").astype(str)
    rows["normalized_text"] = rows["search_query"].map(normalize)
    return rows


def _population(rows: pd.DataFrame, locations: set[str]) -> dict:
    filters = rows["search_infm_params_text"].str.strip().ne("")
    no_exact = ~rows["search_location_id"].isin(locations)
    return {
        "count": int(len(rows)),
        "with_filter_count": int(filters.sum()),
        "with_filter_fraction": float(filters.mean()),
        "no_exact_location_count": int(no_exact.sum()),
        "no_exact_location_fraction": float(no_exact.mean()),
        "unique_normalized_texts": int(rows["normalized_text"].nunique()),
    }


def _overlap(count: int, total: int) -> dict:
    return {"count": int(count), "total": int(total), "fraction": count / total}


def analyze(data_dir: Path) -> dict:
    train = _search_rows(data_dir / "train.parquet", with_item=True)
    test = _search_rows(data_dir / "benchmark_queries.parquet", with_item=False)
    items = pq.read_table(data_dir / "benchmark_items.parquet",
                          columns=["item_id", "item_location_id"]).to_pandas()
    corpus_ids = set(items["item_id"].dropna().astype(str))
    locations = set(items["item_location_id"].dropna().astype(str))

    # Разбиение совпадает с evaluate: по нормализованному тексту до группировки.
    held_texts = {text for text in train["normalized_text"].unique()
                  if held_out(text, fraction=0.2, seed=42)}
    held = train[train["normalized_text"].isin(held_texts)]
    valid = held[held["item_id"].isin(corpus_ids)].copy()
    for name in SEARCH_FIELDS:
        valid[name] = valid[name].map(normalize)
    groups = valid.drop_duplicates(list(SEARCH_FIELDS)).copy()

    # Для пересечения полных запросов сравниваем все поисковые признаки,
    # а не только search_query; два написания с пробелами считаются одним.
    normalized_train = train[list(SEARCH_FIELDS)].drop_duplicates().copy()
    normalized_test = test[list(SEARCH_FIELDS)].copy()
    for name in SEARCH_FIELDS:
        normalized_train[name] = normalized_train[name].map(normalize)
        normalized_test[name] = normalized_test[name].map(normalize)
    train_full_queries = set(map(tuple, normalized_train.itertuples(index=False,
                                                                     name=None)))
    test_full_queries = list(normalized_test.itertuples(index=False, name=None))
    train_texts = set(train["normalized_text"])
    train_items = set(train["item_id"])

    return {
        "source": "train.parquet, benchmark_queries.parquet, benchmark_items.parquet",
        "validation": {"split": "queries", "seed": 42, "fraction": 0.2,
                       "heldout_rows": int(len(held)),
                       "heldout_rows_in_corpus": int(len(valid))},
        "populations": {
            "train_events": _population(train, locations),
            "local_validation_groups": _population(groups, locations),
            "benchmark_queries": _population(test, locations),
        },
        "overlap": {
            "train_rows_with_item_in_corpus": _overlap(
                int(train["item_id"].isin(corpus_ids).sum()), len(train)),
            "corpus_items_seen_in_train": _overlap(
                len(corpus_ids & train_items), len(corpus_ids)),
            "benchmark_texts_seen_in_train": _overlap(
                int(test["normalized_text"].isin(train_texts).sum()), len(test)),
            "benchmark_full_queries_seen_in_train": _overlap(
                sum(key in train_full_queries for key in test_full_queries), len(test)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path,
                        default=Path("docs/dataset-analysis.json"))
    args = parser.parse_args()
    results = analyze(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
