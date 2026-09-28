"""Обучаемый отбор 50 объявлений из готового пула кандидатов.

Положительные пары берутся только из обучающей части разбиения по тексту.
Остальные найденные объявления остаются неразмеченными: модель учится на
положительных и неразмеченных примерах, а не на полной разметке релевантности.
"""

from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier

from .core import SEARCH_FIELDS, held_out, normalize, query_key


BM25_CHANNELS = ("title", "params", "description", "title_stem", "params_stem")
GEO_CHANNELS = tuple(f"{name}_geo" for name in BM25_CHANNELS)
CHANNELS = (*BM25_CHANNELS, *GEO_CHANNELS, "dense")
FEATURE_VERSION = 3
FEATURES = (
    "rrf_score", "pool_rank", "baseline_top50", "channel_count",
    *(f"rank_{name}" for name in CHANNELS),
    *(f"present_{name}" for name in CHANNELS),
    *(f"score_{name}" for name in CHANNELS),
    "location_match", "location_known", "geo_destination_match",
    "geo_destination_known", "geo_destination_is_fallback", "geo_channel_count",
    "category_match", "category_known",
    "title_overlap", "params_overlap", "description_overlap",
    "title_query_coverage", "params_query_coverage", "description_query_coverage",
    "title_exact_phrase", "params_exact_phrase", "title_length", "description_length",
    "filter_title_overlap", "filter_params_overlap",
    "filter_title_coverage", "filter_params_coverage",
    "filter_params_bigram_coverage", "filter_params_exact_phrase",
    "rating_filter_known", "rating_filter_match",
    "log_price", "rating", "log_reviews", "phone_hidden", "message_forbidden",
    "has_query_filters", "delivery_search",
)
_WORDS = re.compile(r"(?u)\b\w+\b")
_RATING_FLOOR = re.compile(r"рейтинг пользователя\s*(\d+(?:[.,]\d+)?)")


def _tokens(value):
    return set(_WORDS.findall(normalize(value)))


def _bigrams(value):
    words = _WORDS.findall(normalize(value))
    return set(zip(words, words[1:]))


def _number(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


def corpus_fingerprint(items):
    """Проверяем, что модель обучалась на том же корпусе и его признаках."""
    digest = hashlib.sha256()
    for item in sorted(items, key=lambda row: row["item_id"]):
        for field in ("item_id", "item_title_raw", "item_infm_params_text",
                      "item_description_raw", "item_category_id", "item_location_id",
                      "item_price", "item_rating", "item_rating_reviews_count",
                      "item_is_phone_hidden", "item_is_message_forbidden"):
            digest.update(str(item.get(field, "")).encode("utf-8"))
            digest.update(b"\0")
        digest.update(b"\n")
    return digest.hexdigest()


def retrieval_config(args):
    """Настройки, определяющие список кандидатов и ранговые признаки."""
    config = {"method": args.method, "candidate_k": args.candidate_k}
    if args.method in {"bm25", "hybrid"}:
        for field in ("title", "params", "description"):
            config[f"{field}_k1"] = args.k1 if args.k1 is not None else getattr(args, f"{field}_k1")
            config[f"{field}_b"] = args.b if args.b is not None else getattr(args, f"{field}_b")
            config[f"{field}_weight"] = getattr(args, f"{field}_weight")
        for name in ("stem_title_weight", "stem_params_weight", "location_bonus",
                     "bm25_rrf_k", "bm25_channel_quota"):
            config[name] = getattr(args, name)
        if args.geo_candidate_k > 0:
            for name in ("geo_candidate_k", "geo_top_locations", "geo_min_history",
                         "geo_weight"):
                config[name] = getattr(args, name)
    if args.method in {"dense", "hybrid"}:
        for name in ("embedding_model", "embedding_max_length", "embedding_params_words",
                     "embedding_description_words", "ef_search"):
            config[name] = getattr(args, name)
    if args.method == "hybrid":
        for name in ("rrf_k", "bm25_weight", "dense_weight", "channel_quota"):
            config[name] = getattr(args, name)
    return config


class FeatureBuilder:
    """Числовые признаки пары; ID объявления не подаётся модели."""

    def __init__(self, items, config):
        self.items = {item["item_id"]: item for item in items}
        self.config = config

    @lru_cache(maxsize=50_000)
    def _item_text(self, item_id):
        item = self.items[item_id]
        title = normalize(item.get("item_title_raw", ""))
        params = normalize(item.get("item_infm_params_text", ""))
        # Для пересечений достаточно начала описания; хранение полного текста
        # в кеше дублировало бы большую часть уже загруженного корпуса.
        description_prefix = str(item.get("item_description_raw", ""))[:1500]
        # Пересечение слов дополняет ранги поисковых каналов.
        return (title, params, _tokens(title), _tokens(params),
                _tokens(description_prefix), _bigrams(params))

    def transform(self, query, details, baseline, candidate_ids=None):
        pool = details["candidate_pool"]
        candidate_ids = pool if candidate_ids is None else candidate_ids
        # Ранги и оценки берём из полного пула поиска даже тогда, когда для
        # обучения отобрана лишь часть кандидатов. Иначе признаки зависели бы
        # от случайной выборки неразмеченных объявлений.
        channels = details["channels"]
        ranks = {name: {item_id: rank for rank, item_id in enumerate(ids, 1)}
                 for name, ids in channels.items()}
        pool_ranks = {item_id: rank for rank, item_id in enumerate(pool, 1)}
        baseline_set = set(baseline)
        query_text = normalize(query.get("search_query", ""))
        query_tokens = _tokens(query_text)
        filter_text = normalize(query.get("search_infm_params_text", ""))
        filter_tokens = _tokens(filter_text)
        filter_bigrams = _bigrams(filter_text)
        rating_match = _RATING_FLOOR.search(filter_text)
        rating_floor = (float(rating_match.group(1).replace(",", "."))
                        if rating_match else None)
        location = str(query.get("search_location_id", ""))
        category = str(query.get("search_category", ""))
        geo_locations = set(details.get("geo_locations", ()))
        channel_scores = details.get("channel_scores", {})
        channel_rank_maps = [ranks.get(name, {}) for name in CHANNELS]
        channel_score_maps = [channel_scores.get(name, {}) for name in CHANNELS]
        geo_rank_maps = [ranks.get(name, {}) for name in GEO_CHANNELS]
        config = self.config
        if config["method"] == "hybrid":
            weights = {name: config["bm25_weight"] * config[f"{name}_weight"]
                       for name in ("title", "params", "description")}
            weights["title_stem"] = config["stem_title_weight"] * config["bm25_weight"]
            weights["params_stem"] = config["stem_params_weight"] * config["bm25_weight"]
            total = sum(weights.values())
            weights = {name: weight / total if total else 0 for name, weight in weights.items()}
            weights["dense"] = config["dense_weight"]
            rank_constant = config["rrf_k"]
        elif config["method"] == "bm25":
            weights = {name: config[f"{name}_weight"] for name in ("title", "params", "description")}
            weights.update(title_stem=config["stem_title_weight"],
                           params_stem=config["stem_params_weight"], dense=0)
            rank_constant = config["bm25_rrf_k"]
        else:
            weights = {name: (1 if name == "dense" else 0) for name in CHANNELS}
            rank_constant = 60
        output = np.empty((len(candidate_ids), len(FEATURES)), dtype=np.float32)
        for row_number, item_id in enumerate(candidate_ids):
            item = self.items[item_id]
            title, params, title_tokens, params_tokens, desc_tokens, params_bigrams = (
                self._item_text(item_id))
            item_ranks = [rank_map.get(item_id, 0) for rank_map in channel_rank_maps]
            item_location = str(item.get("item_location_id", ""))
            item_category = str(item.get("item_category_id", ""))
            overlap = [len(query_tokens & tokens) for tokens in (title_tokens, params_tokens, desc_tokens)]
            # Геоканалы участвуют в объединении; берём готовый итоговый балл,
            # чтобы не восстанавливать его по неполному набору рангов.
            if "rrf_scores" in details:
                rrf_score = details["rrf_scores"][item_id]
            else:
                rrf_score = sum(weights[name] / (rank_constant + rank)
                                for name, rank in zip(CHANNELS, item_ranks)
                                if rank and name in weights)
                rrf_score += config.get("location_bonus", 0) * bool(
                    location and location == item_location)
            output[row_number] = (
                rrf_score, 1 / (1 + pool_ranks[item_id]), float(item_id in baseline_set),
                sum(item_id in channel_ranks for channel_ranks in ranks.values()),
                *(1 / (rank_constant + rank) if rank else 0 for rank in item_ranks),
                *(float(bool(rank)) for rank in item_ranks),
                *(score_map.get(item_id, 0.0) for score_map in channel_score_maps),
                float(bool(location and location == item_location)),
                float(bool(location and item_location)),
                float(bool(item_location and item_location in geo_locations)),
                float(bool(geo_locations)),
                float(bool(geo_locations and location not in geo_locations)),
                sum(bool(rank_map.get(item_id)) for rank_map in geo_rank_maps),
                float(bool(category and category == item_category)),
                float(bool(category and item_category)),
                *overlap,
                *(count / max(1, len(query_tokens)) for count in overlap),
                float(bool(query_text and query_text in title)),
                float(bool(query_text and query_text in params)),
                np.log1p(len(title_tokens)), np.log1p(len(desc_tokens)),
                len(filter_tokens & title_tokens), len(filter_tokens & params_tokens),
                len(filter_tokens & title_tokens) / max(1, len(filter_tokens)),
                len(filter_tokens & params_tokens) / max(1, len(filter_tokens)),
                len(filter_bigrams & params_bigrams) / max(1, len(filter_bigrams)),
                float(bool(filter_text and filter_text in params)),
                float(bool(rating_floor is not None and item.get("item_rating", "") != "")),
                float(bool(rating_floor is not None and
                           _number(item.get("item_rating")) >= rating_floor)),
                np.log1p(max(0, _number(item.get("item_price")))),
                _number(item.get("item_rating")),
                np.log1p(max(0, _number(item.get("item_rating_reviews_count")))),
                _number(item.get("item_is_phone_hidden")),
                _number(item.get("item_is_message_forbidden")),
                float(bool(query.get("search_infm_params_text", ""))),
                _number(query.get("search_is_delivery_search")),
            )
        return output


def read_fit_labels(path, corpus_ids, *, split, fraction, seed, batch_size):
    """Берём обучающие строки и исключаем все варианты отложенного текста."""
    if split not in {"queries", "all"}:
        raise ValueError("Reranker training requires --split queries or --split all")
    if split == "queries" and not 0 < fraction < 1:
        raise ValueError("Validation fraction must lie strictly between 0 and 1")
    truth, representatives = defaultdict(set), {}
    selected_rows = in_corpus = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size,
                                                   columns=[*SEARCH_FIELDS, "item_id"]):
        for row in batch.to_pandas().fillna("").astype(str).to_dict("records"):
            if split == "queries" and held_out(normalize(row["search_query"]), fraction, seed):
                continue
            selected_rows += 1
            if row["item_id"] in corpus_ids:
                in_corpus += 1
                key = query_key(row)
                truth[key].add(row["item_id"])
                representatives[key] = row
    return truth, representatives, {"fit_rows": selected_rows,
                                    "fit_rows_in_corpus": in_corpus,
                                    "fit_queries_in_corpus": len(truth)}


def retrieve_details(retriever, queries, method, candidate_k):
    """Единый интерфейс для трёх методов поиска."""
    if method == "hybrid":
        return retriever.retrieve_batch_with_diagnostics(queries)
    if method == "bm25":
        return [retriever.retrieve_with_diagnostics(query) for query in queries]
    pools = retriever.retrieve_batch_with_scores(queries, max(50, candidate_k))
    return [(pool[:50], {"candidate_pool": pool, "channels": {"dense": pool},
                         "channel_scores": {"dense": scores}})
            for pool, scores in pools]


def sample_training_ids(pool, baseline, channels, positives, limit, seed):
    """Сохраняем положительные, сложные и случайные неразмеченные примеры."""
    found = [item_id for item_id in pool if item_id in positives]
    if not found:
        return []
    # Остальные объявления в пуле не имеют метки выбора. Берём среди них и
    # высоко стоящие сложные примеры, и случайные для разнообразия обучения.
    hard_budget = limit // 2
    baseline_budget = hard_budget // 2
    chosen = [item_id for item_id in baseline
              if item_id not in positives][:baseline_budget]
    chosen_set = set(chosen) | positives
    # Чередование каналов не даёт первому списку занять все места для сложных
    # примеров до того, как дойдём до dense и географического поиска.
    channel_heads = [ids[:20] for ids in channels.values()]
    for rank in range(20):
        for ids in channel_heads:
            if len(chosen) >= hard_budget:
                break
            if rank < len(ids) and ids[rank] not in chosen_set:
                chosen.append(ids[rank])
                chosen_set.add(ids[rank])
        if len(chosen) >= hard_budget:
            break
    if len(chosen) < hard_budget:
        for item_id in baseline:
            if item_id not in chosen_set:
                chosen.append(item_id)
                chosen_set.add(item_id)
                if len(chosen) >= hard_budget:
                    break
    chosen_set = set(chosen) | positives
    rest = [item_id for item_id in pool if item_id not in chosen_set]
    rng = np.random.default_rng(seed)
    extra = rng.choice(len(rest), size=min(max(0, limit - len(chosen)), len(rest)),
                       replace=False) if rest else []
    chosen.extend(rest[int(index)] for index in extra)
    return found + chosen


def prepare_training_data(path, retriever, builder, truth, representatives, *,
                          split, fraction, seed, negatives, batch_size, max_queries=0):
    """Потоково сохраняем признаки кандидатов в Parquet без обучения модели."""
    import hashlib as _hashlib

    keys = sorted(truth, key=lambda x: _hashlib.sha256(repr(x).encode()).digest())
    if max_queries:
        keys = keys[:max_queries]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([*(pa.field(name, pa.float32()) for name in FEATURES),
                        pa.field("label", pa.int8()), pa.field("weight", pa.float32())])
    writer = pq.ParquetWriter(path, schema, compression="zstd")
    stats = {"sampled_queries": 0, "queries_with_positive_in_pool": 0,
             "positives_in_pool": 0, "positives_in_top50": 0,
             "positive_rows": 0, "unobserved_rows": 0}
    try:
        for offset in range(0, len(keys), batch_size):
            batch_keys = keys[offset:offset + batch_size]
            queries = [representatives[key] for key in batch_keys]
            results = retrieve_details(retriever, queries, builder.config["method"],
                                       builder.config["candidate_k"])
            columns = [[] for _ in FEATURES]
            labels, weights = [], []
            for key, query, (baseline, details) in zip(batch_keys, queries, results):
                positives = truth[key]
                pool = details["candidate_pool"]
                stats["sampled_queries"] += 1
                stats["positives_in_pool"] += len(positives.intersection(pool))
                stats["positives_in_top50"] += len(positives.intersection(baseline))
                key_seed = int.from_bytes(_hashlib.sha256(repr((seed, key)).encode()).digest()[:8], "big")
                chosen = sample_training_ids(pool, baseline, details["channels"],
                                             positives, negatives, key_seed)
                if not chosen or len(chosen) == len(positives.intersection(pool)):
                    continue
                matrix = builder.transform(query, details, baseline, chosen)
                n_positive = sum(item_id in positives for item_id in chosen)
                n_negative = len(chosen) - n_positive
                stats["queries_with_positive_in_pool"] += 1
                stats["positive_rows"] += n_positive
                stats["unobserved_rows"] += n_negative
                # Каждый запрос получает суммарный вес 1 для выбранных и 1
                # для неразмеченных объявлений независимо от размера пула.
                for index, item_id in enumerate(chosen):
                    positive = item_id in positives
                    labels.append(int(positive))
                    weights.append(1 / (n_positive if positive else n_negative))
                    for column, value in zip(columns, matrix[index]):
                        column.append(float(value))
            if labels:
                arrays = [pa.array(values, type=pa.float32()) for values in columns]
                arrays.extend((pa.array(labels, type=pa.int8()),
                               pa.array(weights, type=pa.float32())))
                writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
    finally:
        writer.close()
    if not stats["positive_rows"]:
        path.unlink(missing_ok=True)
        raise ValueError("No retrieved training positives; check corpus and retrieval settings")
    metadata = {"feature_version": FEATURE_VERSION, "feature_names": list(FEATURES),
                "retrieval_config": builder.config,
                "corpus_fingerprint": corpus_fingerprint(builder.items.values()),
                "split": split, "validation_fraction": fraction if split == "queries" else 1.0,
                "seed": seed, "stats": stats}
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def fit_model(data_path, model_path, *, max_iter=150):
    """Обучаем локальную модель после освобождения памяти поискового индекса."""
    data_path, model_path = Path(data_path), Path(model_path)
    metadata = json.loads(data_path.with_suffix(data_path.suffix + ".json").read_text(encoding="utf-8"))
    if metadata.get("feature_version") != FEATURE_VERSION or metadata["feature_names"] != list(FEATURES):
        raise ValueError("Training feature schema does not match code")
    table = pq.read_table(data_path)
    matrix = np.column_stack([table[name].to_numpy() for name in FEATURES]).astype(np.float32)
    labels = table["label"].to_numpy()
    weights = table["weight"].to_numpy()
    if len(np.unique(labels)) != 2:
        raise ValueError("Training requires both clicked and unobserved examples")
    model = HistGradientBoostingClassifier(max_iter=max_iter, max_leaf_nodes=31,
                                           l2_regularization=1.0, early_stopping=False,
                                           random_state=metadata["seed"])
    model.fit(matrix, labels, sample_weight=weights)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, model_path)
    return {"model": str(model_path), "training_rows": len(labels),
            "positive_rows": int(labels.sum()), "iterations": max_iter,
            "training_stats": metadata["stats"]}


class Reranker:
    def __init__(self, path, items, config, *, evaluation_split=None):
        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.metadata = artifact["metadata"]
        if (self.metadata.get("feature_version") != FEATURE_VERSION
                or self.metadata["feature_names"] != list(FEATURES)):
            raise ValueError("Reranker feature schema differs from code")
        if self.metadata["retrieval_config"] != config:
            raise ValueError("Reranker retrieval settings differ from training")
        if self.metadata["corpus_fingerprint"] != corpus_fingerprint(items):
            raise ValueError("Reranker was trained for a different item corpus")
        if evaluation_split is not None:
            mode, fraction, seed = evaluation_split
            if (mode != "queries" or self.metadata["split"] != "queries"
                    or self.metadata["validation_fraction"] != fraction
                    or self.metadata["seed"] != seed):
                raise ValueError("Evaluate reranker only on its held-out query-text split")
        self.builder = FeatureBuilder(items, config)

    def rank_batch(self, queries, results, limit=50):
        """Считаем оценки пачкой и сортируем каждый пул отдельно."""
        matrices = [self.builder.transform(query, details, baseline)
                    for query, (baseline, details) in zip(queries, results)]
        nonempty = [matrix for matrix in matrices if len(matrix)]
        if not nonempty:
            return [[] for _ in queries]
        probabilities = self.model.predict_proba(np.concatenate(nonempty))[:, 1]
        predictions, start = [], 0
        for matrix, (_baseline, details) in zip(matrices, results):
            count = len(matrix)
            if count:
                # География служит признаком, а не жёстким разделением пула.
                # При равенстве оценок модели используем исходный балл поиска.
                order = np.lexsort((np.arange(count), -matrix[:, 0],
                                    -probabilities[start:start + count]))[:limit]
                predictions.append([details["candidate_pool"][int(index)] for index in order])
                start += count
            else:
                predictions.append([])
        return predictions
