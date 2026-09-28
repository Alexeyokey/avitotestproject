"""Команды локального поиска, обучения реранкера, оценки и отправки ответа."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .core import SEARCH_FIELDS, held_out, normalize, query_key, recall_at_k, validate_answers
from .dense import DEFAULT_EMBEDDING_MODEL


def read(path, columns):
    # Строковое представление сохраняет ID и не превращает признаки в числа.
    frame = pd.read_parquet(path, columns=list(columns))
    return frame.fillna("").astype(str).to_dict("records")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def batches(rows, size):
    """Ограничиваем память запросов, сохраняя пакетное кодирование dense."""
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


class RrfScoreReport:
    """Потоково сохраняем баллы итоговых 50 без хранения всех пулов."""

    def __init__(self, report_path, location_bonus):
        self.path = report_path.with_suffix(".rrf.csv")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("w", encoding="utf-8", newline="")
        self.writer = csv.writer(self.stream)
        self.writer.writerow([*SEARCH_FIELDS, "rank", "item_id", "rrf_score"])
        self.count = 0
        self.total = 0.0
        self.minimum = None
        self.maximum = None
        self.location_bonus = location_bonus

    def add(self, query, prediction, details):
        # RRF остаётся баллом поиска, даже если реранкер меняет порядок.
        # При включённой добавке он уже учитывает совпадение точной локации.
        scores = details["rrf_scores"]
        key = query_key(query)
        for rank, item_id in enumerate(prediction, start=1):
            score = scores[item_id]
            self.writer.writerow([*key, rank, item_id, repr(score)])
            self.count += 1
            self.total += score
            self.minimum = score if self.minimum is None else min(self.minimum, score)
            self.maximum = score if self.maximum is None else max(self.maximum, score)

    def finish(self):
        self.stream.close()
        return {
            "file": str(self.path),
            "rows": self.count,
            "mean": self.total / self.count if self.count else None,
            "min": self.minimum,
            "max": self.maximum,
            "location_bonus": self.location_bonus,
        }


class LocationBonusSweep:
    """Проверяем географические добавки на одних и тех же кандидатах."""

    def __init__(self, items, keys, *, rank_constant, bm25_weight, dense_weight,
                 reference_bonus):
        from .hybrid import rank_location_bonus_grid

        self.rerank = rank_location_bonus_grid
        self.item_locations = {item["item_id"]: item["item_location_id"] for item in items}
        # Порог жёсткого приоритета не выше (BM25 + dense) / (RRF k + 1).
        # Проверяем мелкий шаг ниже порога и добавляем текущую настройку.
        self.hard_priority_threshold = (bm25_weight + dense_weight) / (rank_constant + 1)
        self.bonuses = sorted(set([*(index / 1000 for index in range(41)),
                                   0.05, 0.1, reference_bonus]))
        self.rank_constant = rank_constant
        self.tune_keys = set(keys[:max(1, len(keys) // 2)])
        self.tune_count = len(self.tune_keys)
        self.confirm_count = len(keys) - self.tune_count
        self.count = len(keys)
        self.totals = {bonus: {name: 0.0 for name in (
            "all", "tune", "confirm", "local", "other", "local_share")}
            for bonus in self.bonuses}
        self.local_query_count = 0
        self.other_query_count = 0

    def add(self, key, query, details, relevant):
        location = query.get("search_location_id", "")
        local_relevant = ({item_id for item_id in relevant
                           if self.item_locations.get(item_id) == location}
                          if location else set())
        other_relevant = set(relevant) - local_relevant
        if local_relevant:
            self.local_query_count += 1
        if other_relevant and location:
            self.other_query_count += 1
        ranked = self.rerank(
            details["channels"], details["channel_weights"], self.item_locations,
            location, self.bonuses, rank_constant=self.rank_constant,
        )
        for bonus, prediction in ranked.items():
            selected = set(prediction)
            totals = self.totals[bonus]
            score = len(selected & relevant) / len(relevant)
            totals["all"] += score
            totals["tune" if key in self.tune_keys else "confirm"] += score
            if local_relevant:
                totals["local"] += len(selected & local_relevant) / len(local_relevant)
            if other_relevant and location:
                totals["other"] += len(selected & other_relevant) / len(other_relevant)
            if location:
                totals["local_share"] += sum(
                    self.item_locations.get(item_id) == location for item_id in prediction
                ) / max(1, len(prediction))

    def report(self):
        rows = []
        for bonus in self.bonuses:
            totals = self.totals[bonus]
            rows.append({
                "bonus": bonus,
                "recall_at_50": totals["all"] / self.count,
                "tuning_recall_at_50": totals["tune"] / self.tune_count,
                "confirmation_recall_at_50": (
                    totals["confirm"] / self.confirm_count if self.confirm_count else None
                ),
                "local_positive_recall_at_50": (
                    totals["local"] / self.local_query_count
                    if self.local_query_count else None
                ),
                "nonlocal_positive_recall_at_50": (
                    totals["other"] / self.other_query_count
                    if self.other_query_count else None
                ),
                "average_local_share_of_top_50": totals["local_share"] / self.count,
            })
        best_tune = max(rows, key=lambda row: (row["tuning_recall_at_50"], -row["bonus"]))
        best_full = max(rows, key=lambda row: (row["recall_at_50"], -row["bonus"]))
        return {
            "hard_priority_threshold_upper_bound": self.hard_priority_threshold,
            "tuning_queries": self.tune_count,
            "confirmation_queries": self.confirm_count,
            "best_bonus_on_tuning_half": best_tune["bonus"],
            "best_bonus_on_full_validation": best_full["bonus"],
            "results": rows,
        }


def geography_recall_slices(keys, truth, representatives, predictions, pools, items):
    """Показываем, где геопоиск находит или теряет известные ответы."""
    item_locations = {item["item_id"]: item.get("item_location_id", "") for item in items}
    corpus_locations = set(item_locations.values())
    slices = {name: {"queries": 0, "pool_recall_sum": 0.0, "top50_recall_sum": 0.0}
              for name in ("exact_location_available", "no_exact_location_available",
                           "local_positive", "nonlocal_positive")}
    for key in keys:
        query_location = representatives[key].get("search_location_id", "")
        relevant = truth[key]
        pool, top50 = set(pools[key]), set(predictions[key][:50])
        name = ("exact_location_available" if query_location in corpus_locations
                else "no_exact_location_available")
        groups = [(name, relevant)]
        local = {item_id for item_id in relevant
                 if query_location and item_locations[item_id] == query_location}
        if local:
            groups.append(("local_positive", local))
        nonlocal_items = relevant - local
        if nonlocal_items:
            groups.append(("nonlocal_positive", nonlocal_items))
        for name, positives in groups:
            summary = slices[name]
            summary["queries"] += 1
            summary["pool_recall_sum"] += len(pool & positives) / len(positives)
            summary["top50_recall_sum"] += len(top50 & positives) / len(positives)
    return {name: {
        "queries": value["queries"],
        "pool_recall": value["pool_recall_sum"] / value["queries"] if value["queries"] else None,
        "recall_at_50": value["top50_recall_sum"] / value["queries"]
        if value["queries"] else None,
    } for name, value in slices.items()}


def read_validation(path, corpus_ids, *, mode, fraction, seed, batch_size):
    """Читаем train потоково и оставляем подходящие для оценки ответы."""
    if mode not in {"pairs", "queries", "all"} or (mode != "all" and not 0 < fraction < 1):
        raise ValueError("Invalid split mode or fraction")
    if batch_size <= 0:
        raise ValueError("Train batch size must be positive")
    truth, representatives = defaultdict(set), {}
    fit_rows = heldout_rows = heldout_in_corpus = 0
    query_selection = {}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=[*SEARCH_FIELDS, "item_id"]
    ):
        # Как и в read(), пропуски становятся пустыми строками, а ID остаются строками.
        rows = batch.to_pandas().fillna("").astype(str).to_dict("records")
        for row in rows:
            if mode == "all":
                selected = True
            elif mode == "queries":
                text = normalize(row["search_query"])
                if text not in query_selection:
                    query_selection[text] = held_out(text, fraction, seed)
                selected = query_selection[text]
            else:
                selected = held_out((query_key(row), row["item_id"]), fraction, seed)
            if not selected:
                fit_rows += 1
                continue
            heldout_rows += 1
            if row["item_id"] in corpus_ids:
                heldout_in_corpus += 1
                key = query_key(row)
                truth[key].add(row["item_id"])
                representatives[key] = row
    return truth, representatives, fit_rows, heldout_rows, heldout_in_corpus


def build_retriever(args, items):
    from .baseline import Baseline

    associations = None
    if args.geo_candidate_k > 0:
        from .geography import LocationAssociations, fit_location_associations

        corpus_locations = {item["item_location_id"] for item in items
                            if item.get("item_location_id")}
        # При --split all нет независимой обучающей части для оценки.
        if args.command == "evaluate" and args.split == "all":
            associations = LocationAssociations(corpus_locations)
        else:
            fit_split = "all" if args.command == "predict" else args.split
            associations = fit_location_associations(
                args.data_dir / "train.parquet", corpus_locations,
                split=fit_split, fraction=args.validation_fraction,
                seed=args.seed, batch_size=args.train_batch_size,
            )

    bm25_kwargs = {
        "title_k1": args.k1 if args.k1 is not None else args.title_k1,
        "title_b": args.b if args.b is not None else args.title_b,
        "params_k1": args.k1 if args.k1 is not None else args.params_k1,
        "params_b": args.b if args.b is not None else args.params_b,
        "description_k1": args.k1 if args.k1 is not None else args.description_k1,
        "description_b": args.b if args.b is not None else args.description_b,
        "title_weight": args.title_weight,
        "params_weight": args.params_weight,
        "description_weight": args.description_weight,
        "stem_title_weight": args.stem_title_weight,
        "stem_params_weight": args.stem_params_weight,
        "location_bonus": args.location_bonus,
        "candidate_k": args.candidate_k,
        "rank_constant": args.bm25_rrf_k,
        "channel_quota": args.bm25_channel_quota,
        "geo_candidate_k": args.geo_candidate_k,
        "geo_top_locations": args.geo_top_locations,
        "geo_include_related": args.geo_include_related,
        "geo_min_history": args.geo_min_history,
        "geo_weight": args.geo_weight,
        "geo_associations": associations,
    }
    if args.method == "bm25":
        return Baseline(items, **bm25_kwargs)

    from .dense import DenseRetriever

    dense = DenseRetriever(
        items,
        model_name=args.embedding_model,
        cache_dir=args.dense_cache,
        batch_size=args.embedding_batch_size,
        max_seq_length=args.embedding_max_length,
        params_words=args.embedding_params_words,
        description_words=args.embedding_description_words,
        device=args.device,
        ef_search=args.ef_search,
    )
    if args.method == "dense":
        return dense

    from .hybrid import HybridRetriever

    return HybridRetriever(
        Baseline(items, **bm25_kwargs),
        dense,
        candidate_k=args.candidate_k,
        rank_constant=args.rrf_k,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        channel_quota=args.channel_quota,
        location_bonus=args.location_bonus,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["profile", "evaluate", "predict", "validate",
                                             "prepare-reranker", "fit-reranker"])
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/result.json"))
    parser.add_argument("--answer", type=Path, default=Path("artifacts/answer.csv"))
    parser.add_argument("--split", choices=["pairs", "queries", "all"], default="pairs",
                        help="Validation selection: held-out pairs, held-out query texts, or all train rows")
    parser.add_argument("--validation-fraction", type=float, default=0.2,
                        help="Fraction held out by pairs/queries; ignored for --split all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--k1", type=float, help="Override k1 for all BM25 fields")
    parser.add_argument("--b", type=float, help="Override b for all BM25 fields")
    parser.add_argument("--title-k1", type=float, default=1.0)
    parser.add_argument("--title-b", type=float, default=0.2)
    parser.add_argument("--params-k1", type=float, default=1.2)
    parser.add_argument("--params-b", type=float, default=0.7)
    parser.add_argument("--description-k1", type=float, default=1.0)
    parser.add_argument("--description-b", type=float, default=0.8)
    parser.add_argument("--title-weight", type=float, default=2.0)
    parser.add_argument("--params-weight", type=float, default=1.0)
    parser.add_argument("--description-weight", type=float, default=0.5)
    parser.add_argument("--stem-title-weight", type=float, default=0.0,
                        help="Weight of a separate Russian-stemmed title channel; 0 disables it")
    parser.add_argument("--stem-params-weight", type=float, default=0.0,
                        help="Weight of a separate Russian-stemmed item-parameters channel; 0 disables it")
    parser.add_argument("--location-bonus", type=float, default=0.0,
                        help="Additive RRF bonus for exact query/item location match; 0 disables it")
    parser.add_argument("--geo-candidate-k", type=int, default=0,
                        help="Extra BM25 candidates per field from query-related item locations; 0 disables")
    parser.add_argument("--geo-top-locations", type=int, default=3,
                        help="Maximum locations in the geography-aware BM25 channel")
    parser.add_argument("--geo-include-related", action="store_true",
                        help="Also use related cities when the exact city exists in the corpus")
    parser.add_argument("--geo-min-history", type=int, default=20,
                        help="Minimum fit clicks before using historical location associations")
    parser.add_argument("--geo-weight", type=float, default=0.25,
                        help="Relative RRF weight of each geography-aware BM25 field")
    parser.add_argument("--bm25-rrf-k", type=int, default=30)
    parser.add_argument("--bm25-channel-quota", type=int, default=10)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=16,
                        help="Batch size for item and query embeddings")
    parser.add_argument("--embedding-max-length", type=int, default=128)
    parser.add_argument("--embedding-params-words", type=int, default=40)
    parser.add_argument("--embedding-description-words", type=int, default=48)
    parser.add_argument("--device", help="Sentence Transformers device: cpu, mps or cuda")
    parser.add_argument("--dense-cache", type=Path, default=Path("artifacts/dense"))
    parser.add_argument("--ef-search", type=int, default=300)
    parser.add_argument("--candidate-k", type=int, default=300)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--channel-quota", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Evaluation smoke-test limit; 0 evaluates all queries")
    parser.add_argument("--sweep-location-bonus", action="store_true",
                        help="Evaluate a grid of hybrid location bonuses in one retrieval pass")
    parser.add_argument("--train-batch-size", type=int, default=32768,
                        help="Parquet rows read at a time during evaluation")
    parser.add_argument("--reranker-data", type=Path,
                        default=Path("artifacts/reranker-train.parquet"),
                        help="Sampled candidate features for supervised training")
    parser.add_argument("--reranker-model", type=Path,
                        help="Trained model to use in evaluate/predict; output path in fit-reranker")
    parser.add_argument("--reranker-negatives-per-query", type=int, default=64)
    parser.add_argument("--reranker-max-iter", type=int, default=150)
    parser.add_argument("--reranker-estimator", choices=["histgb", "lightgbm"],
                        default="histgb",
                        help="Classifier for fit-reranker (LightGBM requires optional dependency)")
    parser.add_argument("--max-train-queries", type=int, default=0,
                        help="Limit training-query preparation for a smoke test")
    args = parser.parse_args()
    if args.command == "fit-reranker":
        from .reranker import fit_model
        if args.reranker_max_iter <= 0:
            parser.error("--reranker-max-iter must be positive")
        result = fit_model(args.reranker_data,
                           args.reranker_model or Path("artifacts/reranker.joblib"),
                           max_iter=args.reranker_max_iter,
                           estimator=args.reranker_estimator)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.data_dir is None:
        parser.error("--data-dir is required for this command")
    if args.sweep_location_bonus and (args.command != "evaluate" or args.method != "hybrid"
                                      or args.channel_quota != 0 or args.reranker_model):
        parser.error("Location sweep requires evaluate --method hybrid --channel-quota 0 "
                     "without --reranker-model")
    if args.command == "prepare-reranker" and args.split == "pairs":
        parser.error("Reranker preparation requires --split queries or --split all")
    if args.reranker_negatives_per_query <= 0 or args.max_train_queries < 0:
        parser.error("Reranker negatives must be positive; max train queries nonnegative")
    if args.method == "dense" and args.location_bonus > 0:
        parser.error("--location-bonus is supported by bm25 and hybrid, not dense")
    if args.method == "dense" and args.geo_candidate_k > 0:
        parser.error("--geo-candidate-k is supported by bm25 and hybrid, not dense")
    if (args.geo_candidate_k < 0 or args.geo_top_locations <= 0
            or args.geo_min_history <= 0 or args.geo_weight <= 0):
        parser.error("Geography retrieval parameters must be positive (K may be zero)")
    root = args.data_dir
    queries = (read(root / "benchmark_queries.parquet", ("query_id", *SEARCH_FIELDS))
               if args.command in {"predict", "profile", "validate"} else [])
    item_columns = (
        "item_id", "item_title_raw", "item_infm_params_text", "item_description_raw",
        "item_category_id", "item_location_id",
    )
    if args.command == "prepare-reranker" or args.reranker_model is not None:
        item_columns += ("item_microcat_id", "item_price", "item_rating",
                         "item_rating_reviews_count",
                         "item_is_phone_hidden", "item_is_message_forbidden")
    items = read(root / "benchmark_items.parquet", item_columns)
    ids = {x["item_id"] for x in items}
    if args.command == "validate":
        with args.answer.open(encoding="utf-8", newline="") as stream:
            validate_answers(list(csv.DictReader(stream)), [q["query_id"] for q in queries], ids)
        print("Submission valid")
        return
    if args.command == "profile":
        train = read(root / "train.parquet", (*SEARCH_FIELDS, "item_id"))
    if args.command == "profile":
        texts = {normalize(x["search_query"]) for x in train}
        keys = {query_key(x) for x in train}
        pairs = {(query_key(x), x["item_id"]) for x in train}
        save_json(args.output, {
            "files": {p.name: {"rows": pq.ParquetFile(p).metadata.num_rows,
                               "columns": pq.ParquetFile(p).schema_arrow.names}
                      for p in sorted(root.glob("*.parquet"))},
            "unique_train_pairs": len(pairs),
            "duplicate_train_pairs": len(train) - len(pairs),
            "benchmark_text_seen_fraction": sum(normalize(q["search_query"]) in texts for q in queries) / len(queries),
            "benchmark_full_query_seen_fraction": sum(query_key(q) in keys for q in queries) / len(queries),
            "corpus_items_seen_fraction": len(ids & {x["item_id"] for x in train}) / len(ids),
            "train_positive_in_corpus_fraction": sum(x["item_id"] in ids for x in train) / len(train),
        })
        return
    if args.command == "prepare-reranker":
        from .reranker import (FeatureBuilder, prepare_training_data, read_fit_labels,
                               retrieval_config)
        truth, representatives, label_stats = read_fit_labels(
            root / "train.parquet", ids, split=args.split,
            fraction=args.validation_fraction, seed=args.seed,
            batch_size=args.train_batch_size,
        )
        if not truth:
            raise ValueError("No fit positives overlap the benchmark item corpus")
        model = build_retriever(args, items)
        from .microcategory import fit_microcategory_prior
        microcategory_prior = fit_microcategory_prior(
            root / "train.parquet", split=args.split,
            fraction=args.validation_fraction, seed=args.seed,
            batch_size=args.train_batch_size)
        bm25 = getattr(model, "bm25", model)
        builder = FeatureBuilder(
            items, retrieval_config(args), microcategory_prior=microcategory_prior,
            geo_associations=getattr(bm25, "geo_associations", None),
            leave_query_out=True)
        metadata = prepare_training_data(
            args.reranker_data, model, builder,
            truth, representatives, split=args.split, fraction=args.validation_fraction,
            seed=args.seed, negatives=args.reranker_negatives_per_query,
            batch_size=args.embedding_batch_size if args.method != "bm25" else 64,
            max_queries=args.max_train_queries,
        )
        print(json.dumps({"data": str(args.reranker_data), **label_stats, **metadata},
                         ensure_ascii=False, indent=2))
        return
    if args.command == "evaluate":
        from .reranker import Reranker, retrieval_config
        reranker = (Reranker(args.reranker_model, items, retrieval_config(args),
                             evaluation_split=(args.split, args.validation_fraction, args.seed))
                    if args.reranker_model else None)
        truth, representatives, fit_rows, heldout_rows, heldout_in_corpus = read_validation(
            root / "train.parquet", ids, mode=args.split,
            fraction=args.validation_fraction, seed=args.seed,
            batch_size=args.train_batch_size,
        )
        # Порядок по хешу не ограничивает оценку первыми строками датасета.
        import hashlib
        keys = sorted(truth, key=lambda x: hashlib.sha256(repr(x).encode()).digest())
        if args.max_queries > 0:
            keys = keys[:args.max_queries]
        model = build_retriever(args, items)
        retrieval_diagnostics = {}
        geo_extra_count = geo_extra_recall = geo_new_positive_queries = 0

        def record_geo_contribution(details, relevant):
            nonlocal geo_extra_count, geo_extra_recall, geo_new_positive_queries
            if args.geo_candidate_k <= 0:
                return
            global_ids = {item_id for name, ids in details["channels"].items()
                          if not name.endswith("_geo") for item_id in ids}
            geo_ids = {item_id for name, ids in details["channels"].items()
                       if name.endswith("_geo") for item_id in ids}
            extra = geo_ids - global_ids
            found = len(extra & relevant)
            geo_extra_count += len(extra)
            geo_extra_recall += found / len(relevant)
            geo_new_positive_queries += bool(found)
        rrf_report = (RrfScoreReport(args.output, args.location_bonus)
                      if args.method in {"bm25", "hybrid"} else None)
        location_sweep = (LocationBonusSweep(
            items, keys, rank_constant=args.rrf_k, bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight, reference_bonus=args.location_bonus,
        ) if args.sweep_location_bonus else None)
        baseline_predictions = {} if reranker else None
        if args.method == "hybrid":
            predictions = {}
            candidate_pools = {}
            predictions_without_quota = {}
            predictions_with_quota_10 = {}
            channel_results = defaultdict(dict)
            for batch_keys in batches(keys, args.embedding_batch_size):
                queries_batch = [representatives[key] for key in batch_keys]
                results = model.retrieve_batch_with_diagnostics(queries_batch)
                if reranker:
                    reranked = reranker.rank_batch(queries_batch, results)
                for index, (key, (prediction, details)) in enumerate(zip(batch_keys, results)):
                    record_geo_contribution(details, truth[key])
                    if location_sweep:
                        location_sweep.add(key, queries_batch[index], details, truth[key])
                    if reranker:
                        baseline_predictions[key] = prediction
                    predictions[key] = (reranked[index] if reranker
                                        else prediction)
                    rrf_report.add(queries_batch[index], predictions[key], details)
                    candidate_pools[key] = details["candidate_pool"]
                    predictions_without_quota[key] = details["prediction_without_quota"]
                    predictions_with_quota_10[key] = details["prediction_with_quota_10"]
                    for channel, item_ids in details["channels"].items():
                        channel_results[channel][key] = item_ids
            relevant = {key: truth[key] for key in keys}
            recall_without_quota = recall_at_k(predictions_without_quota, relevant)
            recall_with_quota_10 = recall_at_k(predictions_with_quota_10, relevant)
            retrieval_diagnostics.update({
                "quota_comparison": {
                    "recall_at_50_without_quota": recall_without_quota,
                    "recall_at_50_with_quota_10": recall_with_quota_10,
                    "without_quota_minus_quota_10": (
                        recall_without_quota - recall_with_quota_10
                    ),
                },
                "channel_recall_at_candidate_k": {
                    channel: recall_at_k(
                        {key: results.get(key, []) for key in keys}, relevant, args.candidate_k)
                    for channel, results in channel_results.items()
                },
            })
        elif args.method == "bm25":
            predictions, candidate_pools = {}, {}
            for batch_keys in batches(keys, 64):
                queries_batch = [representatives[key] for key in batch_keys]
                results = [model.retrieve_with_diagnostics(query) for query in queries_batch]
                reranked = reranker.rank_batch(queries_batch, results) if reranker else None
                for index, (key, (prediction, details)) in enumerate(zip(batch_keys, results)):
                    record_geo_contribution(details, truth[key])
                    if reranker:
                        baseline_predictions[key] = prediction
                    predictions[key] = reranked[index] if reranker else prediction
                    rrf_report.add(queries_batch[index], predictions[key], details)
                    candidate_pools[key] = details["candidate_pool"]
        else:
            # У dense один канал, поэтому весь его список образует пул.
            candidate_pools = {}
            predictions = {}
            for batch_keys in batches(keys, args.embedding_batch_size):
                queries_batch = [representatives[key] for key in batch_keys]
                dense_results = (model.retrieve_batch_with_scores(
                    queries_batch, max(50, args.candidate_k)) if reranker else
                    [(pool, {}) for pool in model.retrieve_batch(
                        queries_batch, max(50, args.candidate_k))])
                pools = [pool for pool, _scores in dense_results]
                candidate_pools.update(zip(batch_keys, pools))
                if reranker:
                    results = [(pool[:50], {"candidate_pool": pool,
                                             "channels": {"dense": pool},
                                             "channel_scores": {"dense": scores}})
                               for pool, scores in dense_results]
                    reranked = reranker.rank_batch(queries_batch, results)
                    for key, pool, prediction in zip(batch_keys, pools, reranked):
                        baseline_predictions[key] = pool[:50]
                        predictions[key] = prediction
            if not reranker:
                predictions = {key: pool[:50] for key, pool in candidate_pools.items()}
        relevant = {key: truth[key] for key in keys}
        pool_recall = recall_at_k(candidate_pools, relevant, len(ids))
        top50_recall = recall_at_k(predictions, relevant)
        retrieval_diagnostics.update({
            "candidate_pool_recall": pool_recall,
            "candidate_pool_complete_query_fraction": sum(
                relevant[key].issubset(candidate_pools[key]) for key in keys
            ) / len(keys),
            "candidate_pool_average_size": sum(map(len, candidate_pools.values())) / len(keys),
            "complete_query_fraction_at_50": sum(
                relevant[key].issubset(predictions[key]) for key in keys
            ) / len(keys),
            "recall_lost_when_cutting_pool_to_50": pool_recall - top50_recall,
            "geography_slices": geography_recall_slices(
                keys, relevant, representatives, predictions, candidate_pools, items),
        })
        if args.geo_candidate_k > 0:
            retrieval_diagnostics["geo_channel_contribution"] = {
                "average_unique_extra_candidates": geo_extra_count / len(keys),
                "incremental_pool_recall": geo_extra_recall / len(keys),
                "queries_with_new_positive": geo_new_positive_queries,
            }
        retrieval_diagnostics["rrf_scores"] = (
            rrf_report.finish() if rrf_report else None
        )
        if reranker:
            baseline_recall = recall_at_k(baseline_predictions, relevant)
            retrieval_diagnostics["baseline_recall_at_50"] = baseline_recall
            retrieval_diagnostics["reranker_delta_at_50"] = top50_recall - baseline_recall
            retrieval_diagnostics["baseline_geography_slices"] = geography_recall_slices(
                keys, relevant, representatives, baseline_predictions, candidate_pools, items)
        if location_sweep:
            sweep_report = location_sweep.report()
            current = next(row for row in sweep_report["results"]
                           if row["bonus"] == args.location_bonus)
            if abs(current["recall_at_50"] - top50_recall) > 1e-10:
                raise RuntimeError("Location sweep does not reproduce the current hybrid ranking")
            retrieval_diagnostics["location_bonus_sweep"] = sweep_report
        save_json(args.output, {"method": args.method,
            "reranker_model": str(args.reranker_model) if reranker else None,
            "fusion": (
                "flat_bm25_fields_dense" if args.method == "hybrid"
                else "bm25_fields" if args.method == "bm25" else None
            ),
            "bm25_fields": {
                "title": {"k1": args.k1 if args.k1 is not None else args.title_k1,
                          "b": args.b if args.b is not None else args.title_b,
                          "weight": args.title_weight},
                "params": {"k1": args.k1 if args.k1 is not None else args.params_k1,
                           "b": args.b if args.b is not None else args.params_b,
                           "weight": args.params_weight},
                "description": {
                    "k1": args.k1 if args.k1 is not None else args.description_k1,
                    "b": args.b if args.b is not None else args.description_b,
                    "weight": args.description_weight},
                "title_stem": {"weight": args.stem_title_weight},
                "params_stem": {"weight": args.stem_params_weight},
                "location_bonus": args.location_bonus,
                "rrf_k": args.bm25_rrf_k,
                "channel_quota": args.bm25_channel_quota,
                "geo_candidate_k": args.geo_candidate_k,
                "geo_top_locations": args.geo_top_locations,
                "geo_include_related": args.geo_include_related,
                "geo_min_history": args.geo_min_history,
                "geo_weight": args.geo_weight,
            } if args.method != "dense" else None,
            "embedding_model": args.embedding_model if args.method != "bm25" else None,
            "embedding_params_words": (
                args.embedding_params_words if args.method != "bm25" else None
            ),
            "embedding_description_words": (
                args.embedding_description_words if args.method != "bm25" else None
            ),
            "embedding_query_fields": (
                ["search_query", "search_infm_params_text", "search_category"]
                if args.method != "bm25" else None
            ),
            "candidate_k": args.candidate_k,
            "bm25_weight": args.bm25_weight if args.method == "hybrid" else None,
            "dense_weight": args.dense_weight if args.method == "hybrid" else None,
            "rrf_k": args.rrf_k if args.method == "hybrid" else None,
            "channel_quota": args.channel_quota if args.method == "hybrid" else None,
            "retrieval_diagnostics": retrieval_diagnostics,
            "split": args.split, "seed": args.seed,
            "validation_fraction": 1.0 if args.split == "all" else args.validation_fraction,
            "fit_rows": fit_rows, "heldout_rows": heldout_rows, "evaluated_queries": len(keys),
            "heldout_in_corpus_fraction": heldout_in_corpus / heldout_rows,
            "recall_at_50": top50_recall,
            "max_queries": args.max_queries})
    else:
        from .reranker import Reranker, retrieval_config
        reranker = (Reranker(args.reranker_model, items, retrieval_config(args))
                    if args.reranker_model else None)
        model = build_retriever(args, items)
        rows = []
        if args.method == "bm25":
            for query_batch in batches(queries, 64):
                results = [model.retrieve_with_diagnostics(query) for query in query_batch]
                predictions = (reranker.rank_batch(query_batch, results) if reranker else
                               [prediction for prediction, _details in results])
                rows.extend({"query_id": q["query_id"], "answer": " ".join(prediction)}
                            for q, prediction in zip(query_batch, predictions))
        else:
            for query_batch in batches(queries, args.embedding_batch_size):
                if args.method == "hybrid":
                    results = model.retrieve_batch_with_diagnostics(query_batch)
                    predictions = (reranker.rank_batch(query_batch, results) if reranker else
                                   [prediction for prediction, _details in results])
                else:
                    if reranker:
                        dense_results = model.retrieve_batch_with_scores(
                            query_batch, max(50, args.candidate_k))
                        results = [(pool[:50], {"candidate_pool": pool,
                                                 "channels": {"dense": pool},
                                                 "channel_scores": {"dense": scores}})
                                   for pool, scores in dense_results]
                        predictions = reranker.rank_batch(query_batch, results)
                    else:
                        pools = model.retrieve_batch(query_batch, max(50, args.candidate_k))
                        predictions = [pool[:50] for pool in pools]
                rows.extend({"query_id": q["query_id"], "answer": " ".join(prediction)}
                            for q, prediction in zip(query_batch, predictions))
        validate_answers(rows, [q["query_id"] for q in queries], ids)
        args.answer.parent.mkdir(parents=True, exist_ok=True)
        with args.answer.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["query_id", "answer"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Validated submission written: {args.answer} ({len(rows)} queries)")


if __name__ == "__main__":
    main()
