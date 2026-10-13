"""Обучаемый отбор 50 объявлений из готового пула кандидатов.

Положительные пары берутся только из обучающей части разбиения по тексту.
Остальные найденные объявления остаются неразмеченными: модель учится на
положительных и неразмеченных примерах, а не на полной разметке релевантности.
"""

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
from importlib import metadata as package_metadata
import json
from numbers import Integral
from pathlib import Path
import re
import sys
import time

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier

from .core import SEARCH_FIELDS, held_out, normalize, query_key


BM25_CHANNELS = ("title", "params", "description", "title_stem", "params_stem")
GEO_CHANNELS = tuple(f"{name}_geo" for name in BM25_CHANNELS)
CHANNELS = (*BM25_CHANNELS, *GEO_CHANNELS, "dense")
FEATURE_VERSION = 5
LEGACY_FEATURES = (
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
V4_FEATURES = (*LEGACY_FEATURES,
    "microcat_token_probability", "microcat_token_rank",
    "geo_click_probability",
    "geo_click_lift", "geo_click_rank",
)
FEATURES = (*V4_FEATURES, "microcat_top50_share")
# V5 remains loadable; the full-pool cosine experiment starts from verified V4.
V6_FEATURES = (*V4_FEATURES, "full_dense_score", "full_dense_known")
V7_FEATURES = (*V4_FEATURES,
    "microcat_classifier_probability", "microcat_classifier_rank",
    "microcat_classifier_entropy", "microcat_classifier_top3_mass",
    "microcat_classifier_known", "rank_microcategory_channel",
    "present_microcategory_channel", "score_microcategory_channel")
V8_FEATURES = (*V7_FEATURES, "full_dense_score", "full_dense_known")
FEATURE_SCHEMAS = {3: LEGACY_FEATURES, 4: V4_FEATURES, 5: FEATURES,
                   6: V6_FEATURES, 7: V7_FEATURES, 8: V8_FEATURES}
GROUP_COLUMNS = (
    ("group_id", pa.string()), ("query_text", pa.string()),
    ("query_key_json", pa.string()), ("item_id", pa.string()),
    ("relevant_count", pa.int32()),
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


def corpus_fingerprint(items, *, feature_version=FEATURE_VERSION):
    """Bind rerankers to the same corpus without invalidating v3 models."""
    fields = ("item_id", "item_title_raw", "item_infm_params_text",
              "item_description_raw", "item_category_id", "item_location_id",
              "item_price", "item_rating", "item_rating_reviews_count",
              "item_is_phone_hidden", "item_is_message_forbidden")
    if feature_version >= 4:
        fields += ("item_microcat_id",)
    digest = hashlib.sha256()
    for item in sorted(items, key=lambda row: row["item_id"]):
        for field in fields:
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
            if args.geo_include_related:
                config["geo_include_related"] = True
    if args.method in {"dense", "hybrid"}:
        for name in ("embedding_model", "embedding_max_length", "embedding_params_words",
                     "embedding_description_words", "ef_search"):
            config[name] = getattr(args, name)
        if getattr(args, "embedding_document_format", "current") != "current":
            config["embedding_document_format"] = args.embedding_document_format
    if args.method == "hybrid":
        for name in ("rrf_k", "bm25_weight", "dense_weight", "channel_quota"):
            config[name] = getattr(args, name)
        if getattr(args, "rescore_dense_union", False):
            config["rescore_dense_union"] = True
        if getattr(args, "microcategory_candidate_k", 0) > 0:
            from .microcategory import _sha_file
            config.update({
                "microcategory_model": str(args.microcategory_model),
                "microcategory_model_sha256": _sha_file(args.microcategory_model),
                "microcategory_candidate_k": args.microcategory_candidate_k,
                "microcategory_field": args.microcategory_field,
                "microcategory_weight": args.microcategory_weight,
                "microcategory_top_categories": args.microcategory_top_categories,
                "microcategory_selection_policy": "top998-then-novel-prefix",
            })
        if getattr(args, "supplemental_fill", "off") != "off":
            config.update(supplemental_fill=args.supplemental_fill,
                          supplemental_raw_depth=998, supplemental_rank_constant=60,
                          supplemental_selection_policy="raw-ranks-equal-route-novel-rrf-id-ties")
    return config


def inference_microcategory_classifier(classifier):
    """Share fitted estimators while excluding disposable prediction-cache data."""
    from .microcategory import SoftMicrocategoryClassifier
    return SoftMicrocategoryClassifier(classifier.model, classifier.crossfit_models,
                                        dict(classifier.metadata))


class FeatureBuilder:
    """Числовые признаки пары; ID объявления не подаётся модели."""

    def __init__(self, items, config, *, microcategory_prior=None,
                 geo_associations=None, feature_names=None,
                 leave_query_out=False, leave_geography_query_out=False,
                 microcategory_classifier=None):
        self.items = {item["item_id"]: item for item in items}
        self.config = config
        self.microcategory_prior = microcategory_prior
        self.microcategory_classifier = microcategory_classifier
        self.geo_associations = geo_associations
        self.leave_query_out = leave_query_out
        self.leave_geography_query_out = leave_geography_query_out
        self.feature_names = tuple(feature_names or FEATURES)
        if self.feature_names not in FEATURE_SCHEMAS.values():
            raise ValueError("Unsupported reranker feature schema")
        if self.feature_names == V6_FEATURES and (
                config.get("method") != "hybrid" or not config.get("rescore_dense_union")):
            raise ValueError("V6 requires hybrid retrieval with full-pool dense rescoring")
        if self.feature_names in {V7_FEATURES, V8_FEATURES}:
            if (config.get("method") != "hybrid" or config.get("microcategory_candidate_k", 0) <= 0
                    or microcategory_classifier is None):
                raise ValueError("V7 requires hybrid microcategory retrieval and its saved classifier")
            if microcategory_classifier.metadata.get("crossfit_folds", 0) < 2:
                raise ValueError("V7 requires a grouped OOF microcategory classifier")
            if microcategory_classifier.metadata.get("corpus_fingerprint") != corpus_fingerprint(
                    self.items.values(), feature_version=4):
                raise ValueError("V7 microcategory classifier was trained for a different item corpus")
        if self.feature_names == V8_FEATURES and not config.get("rescore_dense_union"):
            raise ValueError("V8 requires full-pool measured dense rescoring")
        from collections import Counter
        self.location_counts = Counter(str(item.get("item_location_id", ""))
                                       for item in items)

    # Token sets dominate memory on the 8 GB WSL host. Capacity affects only
    # recomputation: the text/feature function and its output stay identical.
    @lru_cache(maxsize=10_000)
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
        classifier_probabilities, classifier_ranks = {}, {}
        classifier_entropy = classifier_top3_mass = 0.0
        micro_name, micro_rank_map, micro_score_map = None, {}, {}
        if self.feature_names in {V7_FEATURES, V8_FEATURES}:
            classifier = self.microcategory_classifier
            try:
                # Seen query texts always use their held-out-fold model, also
                # at prediction time. All classes are needed for item scores
                # and entropy, rather than truncating the distribution at K.
                predictions = classifier.predict_categories(query,
                    top_k=len(classifier.model.classes), exclude_self=True)
            finally:
                # No 30k-query cache of 200-class Python tuples is serialized
                # or retained between feature frames. Retrieval's top-K cache
                # is disposable and the actual model weights remain untouched.
                classifier._prediction_cache.clear()
            probabilities = np.asarray([prob for _category, prob in predictions], dtype=np.float64)
            classes = [str(category) for category, _prob in predictions]
            if (not len(probabilities) or len(set(classes)) != len(classes)
                    or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                    or np.any(probabilities > 1) or not np.isclose(probabilities.sum(), 1, atol=1e-5)):
                raise ValueError("V7 classifier requires a complete finite probability distribution")
            classifier_probabilities = dict(zip(classes, probabilities))
            classifier_ranks = {category: 1 / rank for rank, category in enumerate(classes, 1)}
            positive = probabilities[probabilities > 0]
            classifier_entropy = (float(-(positive * np.log(positive)).sum() / np.log(len(classes)))
                                  if len(classes) > 1 else 0.0)
            classifier_entropy = float(np.clip(classifier_entropy, 0, 1))
            classifier_top3_mass = float(probabilities[:3].sum())
            micro_name = "micro_" + self.config.get("microcategory_field", "combined")
            active = [name for name in details["channels"] if name.startswith("micro_")]
            if any(name != micro_name for name in active):
                raise ValueError("V7 expects only the configured microcategory candidate channel")
            micro_ids = details["channels"].get(micro_name, [])
            micro_score_map = details.get("channel_scores", {}).get(micro_name, {})
            try:
                valid_micro_scores = all(np.isfinite(float(score)) and float(score) >= 0
                                         for score in micro_score_map.values())
            except (TypeError, ValueError):
                valid_micro_scores = False
            if (len(micro_ids) != len(set(micro_ids)) or not set(micro_ids).issubset(pool)
                    or set(micro_score_map) != set(micro_ids)
                    or not valid_micro_scores):
                raise ValueError("V7 microcategory scores must cover its channel exactly and be finite")
            micro_rank_map = {item_id: 1 / (self.config["rrf_k"] + rank)
                              for rank, item_id in enumerate(micro_ids, 1)}
        full_dense_scores = {}
        if self.feature_names in {V6_FEATURES, V8_FEATURES}:
            full_dense_scores = details.get("full_dense_scores", {})
            if set(full_dense_scores) != set(pool):
                raise ValueError("V6 full dense scores must cover every pool candidate")
            try:
                valid_full_scores = all(np.isfinite(float(value))
                                        and abs(float(value)) <= 1.0001
                                        for value in full_dense_scores.values())
            except (TypeError, ValueError):
                valid_full_scores = False
            if not valid_full_scores:
                raise ValueError("V6 full dense scores must be finite cosine values")
        # Ранги и оценки берём из полного пула поиска даже тогда, когда для
        # обучения отобрана лишь часть кандидатов. Иначе признаки зависели бы
        # от случайной выборки неразмеченных объявлений.
        channels = details["channels"]
        ranks = {name: {item_id: rank for rank, item_id in enumerate(ids, 1)}
                 for name, ids in channels.items()}
        pool_ranks = {item_id: rank for rank, item_id in enumerate(pool, 1)}
        baseline_set = set(baseline)
        # Share of the original search top 50 in the candidate's microcategory.
        # This uses only retrieved items, never their held-out labels.
        top50_microcats = Counter(str(self.items[item_id].get("item_microcat_id", ""))
                                  for item_id in baseline)
        top50_microcats.pop("", None)
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
        # Zero denotes an unspecified search category, as in dense.query_text.
        # It must not look like an observed category or match an item ID of 0.
        if category == "0":
            category = ""
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
        output = np.empty((len(candidate_ids), len(self.feature_names)), dtype=np.float32)
        associations = self.geo_associations
        if associations:
            click_counts, click_total = associations.history(
                location, exclude_query_text=(query_text
                                              if self.leave_geography_query_out else None))
        else:
            click_counts, click_total = {}, 0
        minimum_history = config.get("geo_min_history", 20)
        if click_total < minimum_history:
            click_counts, click_total = {}, 0
        geo_ranks = {destination: 1 / rank for rank, (_count, destination)
                     in enumerate(sorted(((count, destination)
                                          for destination, count in click_counts.items()),
                                         key=lambda pair: (-pair[0], pair[1])), start=1)}
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
            values = (
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
            if self.feature_names == LEGACY_FEATURES:
                output[row_number] = values
                continue
            microcat = str(item.get("item_microcat_id", ""))
            mc = (self.microcategory_prior.features(
                      query_text, microcat, exclude_self=self.leave_query_out)[1:]
                  if self.microcategory_prior else (0.0, 0.0))
            click_count = click_counts.get(item_location, 0)
            click_probability = click_count / click_total if click_total else 0.0
            corpus_probability = self.location_counts[item_location] / max(1, len(self.items))
            click_lift = (np.log1p(click_probability / corpus_probability)
                          if corpus_probability else 0.0)
            v4_values = (*values, *mc, click_probability,
                         click_lift, geo_ranks.get(item_location, 0.0))
            if self.feature_names == V4_FEATURES:
                output[row_number] = v4_values
            elif self.feature_names == V6_FEATURES:
                # A score of zero is a real measured cosine, not missing data.
                # Nonempty V6 pools must be fully scored; empty pools have no rows.
                output[row_number] = (*v4_values, full_dense_scores[item_id], 1.0)
            elif self.feature_names in {V7_FEATURES, V8_FEATURES}:
                v7_values = (*v4_values,
                    classifier_probabilities.get(microcat, 0.0),
                    classifier_ranks.get(microcat, 0.0), classifier_entropy,
                    classifier_top3_mass, float(microcat in classifier_probabilities),
                    micro_rank_map.get(item_id, 0.0), float(item_id in micro_rank_map),
                    micro_score_map.get(item_id, 0.0))
                output[row_number] = (v7_values if self.feature_names == V7_FEATURES else
                                      (*v7_values, full_dense_scores[item_id], 1.0))
            else:
                share = top50_microcats[microcat] / max(1, len(baseline)) if microcat else 0.0
                output[row_number] = (*v4_values, share)
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
    schema = pa.schema([*(pa.field(name, pa.float32()) for name in builder.feature_names),
                        pa.field("label", pa.int8()), pa.field("weight", pa.float32()),
                        *(pa.field(name, dtype) for name, dtype in GROUP_COLUMNS)])
    writer = pq.ParquetWriter(path, schema, compression="zstd")
    stats = {"sampled_queries": 0, "queries_with_positive_in_pool": 0,
             "positives_in_pool": 0, "positives_in_top50": 0,
             "positive_rows": 0, "unobserved_rows": 0}
    started = last_progress = time.monotonic()
    try:
        for offset in range(0, len(keys), batch_size):
            batch_keys = keys[offset:offset + batch_size]
            queries = [representatives[key] for key in batch_keys]
            results = retrieve_details(retriever, queries, builder.config["method"],
                                       builder.config["candidate_k"])
            columns = [[] for _ in builder.feature_names]
            labels, weights = [], []
            group_columns = {name: [] for name, _dtype in GROUP_COLUMNS}
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
                # The full key identifies a ranking group; query_text lets CV
                # keep all its location/filter variants on the same side.
                key_json = json.dumps(list(key), ensure_ascii=False, separators=(",", ":"))
                group_id = _hashlib.sha256(key_json.encode("utf-8")).hexdigest()
                query_text = normalize(query.get("search_query", ""))
                for index, item_id in enumerate(chosen):
                    positive = item_id in positives
                    labels.append(int(positive))
                    weights.append(1 / (n_positive if positive else n_negative))
                    for column, value in zip(columns, matrix[index]):
                        column.append(float(value))
                    for name, value in zip(group_columns,
                                           (group_id, query_text, key_json, item_id, len(positives))):
                        group_columns[name].append(value)
            if labels:
                arrays = [pa.array(values, type=pa.float32()) for values in columns]
                arrays.extend((pa.array(labels, type=pa.int8()),
                               pa.array(weights, type=pa.float32())))
                arrays.extend(pa.array(group_columns[name], type=dtype)
                              for name, dtype in GROUP_COLUMNS)
                writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
            if time.monotonic() - last_progress >= 30 or stats["sampled_queries"] == len(keys):
                print(json.dumps({"prepare_reranker": {
                    "queries": stats["sampled_queries"], "total": len(keys),
                    "positive_rows": stats["positive_rows"],
                    "unobserved_rows": stats["unobserved_rows"],
                    "elapsed_seconds": round(time.monotonic() - started),
                }}), flush=True)
                last_progress = time.monotonic()
    finally:
        writer.close()
    if not stats["positive_rows"]:
        path.unlink(missing_ok=True)
        raise ValueError("No retrieved training positives; check corpus and retrieval settings")
    feature_version = next(version for version, names in FEATURE_SCHEMAS.items()
                           if names == builder.feature_names)
    metadata = {"feature_version": feature_version,
                "training_data_schema_version": 2,
                "group_columns": [name for name, _dtype in GROUP_COLUMNS],
                "training_prior_policy": {
                    "microcategory": ("leave-query-text-out" if builder.leave_query_out
                                      else "full-fit-history"),
                    "geography": ("leave-query-text-out"
                                  if builder.leave_geography_query_out
                                  else "full-fit-history"),
                },
                "feature_names": list(builder.feature_names),
                "retrieval_config": builder.config,
                "corpus_fingerprint": corpus_fingerprint(
                    builder.items.values(), feature_version=feature_version),
                "split": split, "validation_fraction": fraction if split == "queries" else 1.0,
                "seed": seed, "stats": stats}
    if feature_version in {7, 8}:
        metadata["training_prior_policy"]["microcategory_classifier"] = "grouped-out-of-fold"
        metadata["microcategory_classifier_metadata"] = dict(builder.microcategory_classifier.metadata)
    if builder.config["method"] in {"dense", "hybrid"}:
        from .dense import local_checkpoint_content_fingerprint, offline_adapter_base_reference
        checkpoint = builder.config.get("embedding_model")
        base = (offline_adapter_base_reference(checkpoint) if checkpoint else
                {"is_adapter": False, "base_repo": None, "base_revision": None})
        metadata.update(
            embedding_checkpoint_sha256=(local_checkpoint_content_fingerprint(checkpoint)
                                          if checkpoint else None),
            embedding_is_adapter=base["is_adapter"],
            embedding_base_repo=base["base_repo"],
            embedding_base_revision=base["base_revision"],
        )
    priors = {"microcategory_prior": builder.microcategory_prior,
              "geo_associations": builder.geo_associations}
    if feature_version in {7, 8}:
        classifier = builder.microcategory_classifier
        # Share immutable TF-IDF/SGD objects; do not deepcopy their coefficients.
        # A fresh wrapper drops the disposable prediction cache from the file.
        priors["microcategory_classifier"] = inference_microcategory_classifier(classifier)
    joblib.dump(priors, path.with_suffix(path.suffix + ".priors.joblib"))
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def validate_training_groups(table, metadata):
    """Validate explicit contiguous query groups without adding IDs to features."""
    required = [name for name, _dtype in GROUP_COLUMNS]
    if (metadata.get("training_data_schema_version") != 2
            or set(metadata.get("group_columns", ())) != set(required)
            or not set(required).issubset(table.column_names)):
        raise ValueError("LambdaRank requires newly prepared grouped training data")
    group_ids = table["group_id"].to_pylist()
    if not group_ids:
        raise ValueError("Grouped training data is empty")
    labels = table["label"].to_numpy()
    weights = table["weight"].to_numpy()
    if (not np.isin(labels, (0, 1)).all() or not np.isfinite(weights).all()
            or np.any(weights <= 0)):
        raise ValueError("Grouped labels must be binary and weights finite and positive")
    query_texts = table["query_text"].to_pylist()
    query_keys = table["query_key_json"].to_pylist()
    item_ids = table["item_id"].to_pylist()
    relevant_counts = table["relevant_count"].to_pylist()
    starts, seen = [], set()
    previous = None
    for row, group_id in enumerate(group_ids):
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("Every ranking row needs an explicit group ID")
        if group_id != previous:
            if group_id in seen:
                raise ValueError("Ranking group IDs must form contiguous query runs")
            seen.add(group_id)
            starts.append(row)
            previous = group_id
    starts.append(len(group_ids))
    sizes = []
    for start, stop in zip(starts[:-1], starts[1:]):
        text, key_json, relevant = query_texts[start], query_keys[start], relevant_counts[start]
        if (not isinstance(text, str) or not isinstance(key_json, str)
                or not isinstance(relevant, int) or relevant <= 0
                or any(value != text for value in query_texts[start:stop])
                or any(value != key_json for value in query_keys[start:stop])
                or any(value != relevant for value in relevant_counts[start:stop])):
            raise ValueError("Ranking query metadata must be consistent within each group")
        try:
            key = json.loads(key_json)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid query-key JSON in ranking data") from error
        if (not isinstance(key, list) or len(key) != len(SEARCH_FIELDS)
                or any(not isinstance(value, str) for value in key)
                or normalize(key[0]) != text
                or hashlib.sha256(key_json.encode("utf-8")).hexdigest() != group_ids[start]):
            raise ValueError("Ranking group ID, query key and normalized text differ")
        ids = item_ids[start:stop]
        if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("Ranking groups must contain distinct item IDs")
        group_labels, group_weights = labels[start:stop], weights[start:stop]
        n_positive = int(group_labels.sum())
        if not 0 < n_positive < stop - start or n_positive > relevant:
            raise ValueError("Each ranking group requires clicked and unobserved rows")
        for label in (0, 1):
            class_weights = group_weights[group_labels == label]
            if (not np.isclose(class_weights.sum(dtype=np.float64), 1.0, atol=1e-5, rtol=0)
                    or not np.allclose(class_weights, 1 / len(class_weights), atol=1e-7, rtol=1e-5)):
                raise ValueError("Ranking source weights do not match per-query class totals")
        sizes.append(stop - start)
    stats = metadata.get("stats", {})
    if (stats.get("queries_with_positive_in_pool") != len(sizes)
            or stats.get("positive_rows") != int(labels.sum())
            or stats.get("unobserved_rows") != len(labels) - int(labels.sum())):
        raise ValueError("Ranking group or label counts differ from preparation metadata")
    return sizes


def _file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def check_retrieval_identity(metadata, config):
    """Allow path relocation only when local model content and base are verified."""
    saved_config = metadata["retrieval_config"]
    checkpoint_sha = metadata.get("embedding_checkpoint_sha256")
    if not checkpoint_sha:
        # Legacy artifacts have no evidence for treating paths as equivalent.
        if saved_config != config:
            raise ValueError("Reranker retrieval settings differ from training")
        return
    from .dense import local_checkpoint_content_fingerprint, offline_adapter_base_reference

    checkpoint = config.get("embedding_model", "")
    if not checkpoint:
        raise ValueError("Reranker embedding checkpoint path is required for content verification")
    if local_checkpoint_content_fingerprint(checkpoint) != checkpoint_sha:
        raise ValueError("Reranker embedding checkpoint content differs from training")
    current_base = offline_adapter_base_reference(checkpoint)
    saved_repo = metadata.get("embedding_base_repo")
    saved_revision = metadata.get("embedding_base_revision")
    saved_is_adapter = metadata.get("embedding_is_adapter", bool(saved_repo))
    if (bool(saved_is_adapter) != current_base["is_adapter"]
            or saved_repo != current_base["base_repo"]):
        raise ValueError("Reranker embedding base model differs from training")
    if saved_revision is not None and current_base["base_revision"] != saved_revision:
        raise ValueError("Reranker embedding base revision differs from training or is unavailable offline")
    path_changed = saved_config.get("embedding_model") != checkpoint
    if path_changed and saved_is_adapter and (saved_revision is None
                                              or current_base["base_revision"] != saved_revision):
        raise ValueError("Checkpoint path alias requires a verified cached adapter base revision")
    comparable = dict(config)
    if path_changed:
        comparable["embedding_model"] = saved_config.get("embedding_model")
    if comparable != saved_config:
        raise ValueError("Reranker retrieval settings differ from training")


def fitted_iterations(model):
    """Return the available iteration count, not a requested estimator setting."""
    booster = getattr(model, "booster_", None)
    if booster is not None:
        return int(booster.current_iteration())
    count = getattr(model, "n_iter_", None)
    if count is None:
        raise ValueError("Cannot verify the fitted model iteration count")
    return int(count)


def validate_selected_iteration(model, selected_iteration):
    if selected_iteration is None:
        return None
    if (isinstance(selected_iteration, bool) or not isinstance(selected_iteration, Integral)
            or selected_iteration < 1):
        raise ValueError("Selected iteration must be a positive integer")
    available = fitted_iterations(model)
    if selected_iteration > available:
        raise ValueError(f"Selected iteration {selected_iteration} exceeds fitted iterations {available}")
    return int(selected_iteration)


def prediction_scores(model, matrix, *, estimator="histgb", selected_iteration=None):
    """Use raw ranker scores or the selected classifier probability prefix."""
    selected_iteration = validate_selected_iteration(model, selected_iteration)
    if estimator == "lambdarank":
        kwargs = {} if selected_iteration is None else {"num_iteration": selected_iteration}
        values = np.asarray(model.predict(matrix, **kwargs))
    else:
        if selected_iteration is None:
            probabilities = model.predict_proba(matrix)
        elif hasattr(model, "booster_"):
            probabilities = model.predict_proba(matrix, num_iteration=selected_iteration)
        elif callable(getattr(model, "staged_predict_proba", None)):
            probabilities = None
            for stage, output in enumerate(model.staged_predict_proba(matrix), 1):
                if stage == selected_iteration:
                    probabilities = output
                    break
            if probabilities is None:
                raise ValueError("Model did not produce the selected probability prefix")
        else:
            raise ValueError("Model does not support selected iteration inference")
        probabilities = np.asarray(probabilities)
        if probabilities.shape != (len(matrix), 2):
            raise ValueError("Classifier must return two probabilities per candidate")
        values = probabilities[:, 1]
    if values.shape != (len(matrix),) or not np.isfinite(values).all():
        raise ValueError("Reranker must return one finite score per candidate")
    return values


def fit_model(data_path, model_path, *, max_iter=150, estimator="histgb",
              disable_microcategory_prior=False):
    """Обучаем локальную модель после освобождения памяти поискового индекса."""
    data_path, model_path = Path(data_path), Path(model_path)
    metadata_path = data_path.with_suffix(data_path.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    version = metadata.get("feature_version")
    expected = FEATURE_SCHEMAS.get(version, ())
    if not expected or tuple(metadata["feature_names"]) != expected:
        raise ValueError("Training feature schema does not match code")
    if version == 6 and (metadata.get("retrieval_config", {}).get("method") != "hybrid"
                         or not metadata["retrieval_config"].get("rescore_dense_union")):
        raise ValueError("V6 fitting requires hybrid full-pool dense rescoring metadata")
    if version in {7, 8}:
        config = metadata.get("retrieval_config", {})
        classifier_metadata = metadata.get("microcategory_classifier_metadata", {})
        if (config.get("method") != "hybrid" or config.get("microcategory_candidate_k", 0) <= 0
                or classifier_metadata.get("crossfit_folds", 0) < 2
                or classifier_metadata.get("corpus_fingerprint") != metadata.get("corpus_fingerprint")
                or metadata.get("training_prior_policy", {}).get("microcategory_classifier") != "grouped-out-of-fold"):
            raise ValueError("V7 fitting requires verified grouped OOF classifier provenance")
        if version == 8 and not config.get("rescore_dense_union"):
            raise ValueError("V8 fitting requires full-pool measured dense rescoring metadata")
    if not isinstance(max_iter, Integral) or isinstance(max_iter, bool) or max_iter <= 0:
        raise ValueError("Reranker iterations must be positive")
    if estimator not in {"histgb", "lightgbm", "lambdarank"}:
        raise ValueError(f"Unknown reranker estimator: {estimator}")
    if estimator in {"lightgbm", "lambdarank"}:
        try:
            from lightgbm import LGBMClassifier, LGBMRanker
        except ImportError as exc:
            raise ImportError("Install the optional LightGBM dependency: pip install -e .[lightgbm]") from exc
    # Group IDs and item IDs are audit metadata, never classifier features.
    # Selecting columns also preserves support for older ungrouped caches.
    columns = [*expected, "label", "weight"]
    if estimator == "lambdarank":
        if metadata.get("training_data_schema_version") != 2:
            raise ValueError("LambdaRank requires newly prepared grouped training data")
        columns.extend(name for name, _dtype in GROUP_COLUMNS)
    table = pq.read_table(data_path, columns=columns)
    matrix = np.column_stack([table[name].to_numpy() for name in expected]).astype(np.float32)
    if disable_microcategory_prior:
        for name in ("microcat_token_probability", "microcat_token_rank"):
            if name not in expected:
                raise ValueError("Microcategory prior is absent from this feature schema")
            matrix[:, expected.index(name)] = 0.0
    labels = table["label"].to_numpy()
    weights = table["weight"].to_numpy()
    if len(np.unique(labels)) != 2 or not np.isin(labels, (0, 1)).all():
        raise ValueError("Training requires both clicked and unobserved examples")
    if not np.isfinite(matrix).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Training features and positive weights must be finite")
    if version in {6, 8} and (np.any(matrix[:, expected.index("full_dense_known")] != 1.0)
                         or np.any(np.abs(matrix[:, expected.index("full_dense_score")]) > 1.0001)):
        raise ValueError("Every V6 training row requires a measured valid full dense cosine")
    group_sizes = validate_training_groups(table, metadata) if estimator == "lambdarank" else None
    if estimator == "lightgbm":
        model = LGBMClassifier(
            n_estimators=max_iter, learning_rate=0.05, num_leaves=31,
            min_child_samples=50, reg_lambda=1.0,
            random_state=metadata["seed"], n_jobs=4, verbosity=-1)
    elif estimator == "lambdarank":
        model = LGBMRanker(
            objective="lambdarank", n_estimators=max_iter, learning_rate=0.05,
            num_leaves=31, min_child_samples=50, reg_lambda=1.0,
            random_state=metadata["seed"], n_jobs=4, verbosity=-1,
            label_gain=[0, 1], lambdarank_norm=True, lambdarank_truncation_level=60)
    else:
        model = HistGradientBoostingClassifier(
            max_iter=max_iter, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=False, random_state=metadata["seed"])
    if estimator == "lambdarank":
        # Ranking groups define comparisons. Pointwise class weights belong to
        # the classifier objective and are not reused as ranking row weights.
        model.fit(matrix, labels, group=group_sizes, feature_name=list(expected))
    else:
        model.fit(matrix, labels, sample_weight=weights)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    prior_path = data_path.with_suffix(data_path.suffix + ".priors.joblib")
    priors = joblib.load(prior_path) if prior_path.exists() and version >= 4 else {}
    if version in {7, 8}:
        classifier = priors.get("microcategory_classifier")
        if classifier is None or classifier.metadata != metadata["microcategory_classifier_metadata"]:
            raise ValueError("V7 fitting requires the matching saved microcategory classifier")
        classifier._prediction_cache.clear()
    versions = {}
    for name in ("numpy", "scikit-learn", "lightgbm", "pyarrow", "joblib"):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    provenance = {
        "version": 1, "training_data": str(data_path),
        "training_data_sha256": _file_sha256(data_path),
        "training_metadata_sha256": _file_sha256(metadata_path),
        "training_priors_sha256": _file_sha256(prior_path) if prior_path.exists() else None,
        "python_version": sys.version.split()[0], "package_versions": versions,
        "training_prior_policy": metadata.get("training_prior_policy"),
    }
    training_configuration = {
        "estimator": estimator, "parameters": model.get_params(),
        "fitted_iterations": fitted_iterations(model), "early_stopping": False,
        "selection_status": "unselected", "selected_iteration": None,
        "group_count": len(group_sizes) if group_sizes is not None else None,
        "identifier_features": False, "random_row_split": False,
        "row_weighting": ("group-normalised LambdaRank, no pointwise class weights"
                          if estimator == "lambdarank" else "original per-query class weights"),
        "disable_microcategory_prior": disable_microcategory_prior,
    }
    joblib.dump({"model": model, "metadata": metadata, "priors": priors,
                 "estimator": estimator,
                 "provenance": provenance, "training_configuration": training_configuration,
                 "selection_status": "unselected", "selected_iteration": None,
                 "disable_microcategory_prior": disable_microcategory_prior}, model_path)
    return {"model": str(model_path), "estimator": estimator,
            "disable_microcategory_prior": disable_microcategory_prior,
            "training_rows": len(labels),
            "positive_rows": int(labels.sum()), "iterations": max_iter,
            "training_configuration": training_configuration, "provenance": provenance,
            "training_stats": metadata["stats"]}


class Reranker:
    def __init__(self, path, items, config, *, evaluation_split=None,
                 microcategory_exact_bonus=0.0, selected_iteration=None):
        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.metadata = artifact["metadata"]
        version = self.metadata.get("feature_version")
        expected = FEATURE_SCHEMAS.get(version, ())
        if not expected or tuple(self.metadata["feature_names"]) != expected:
            raise ValueError("Reranker feature schema differs from code")
        self.estimator = artifact.get("estimator", "histgb")
        if self.estimator not in {"histgb", "lightgbm", "lambdarank"}:
            raise ValueError("Unsupported reranker estimator in artifact")
        selected = artifact.get("selected_iteration") if selected_iteration is None else selected_iteration
        self.selected_iteration = validate_selected_iteration(self.model, selected)
        check_retrieval_identity(self.metadata, config)
        if self.metadata["corpus_fingerprint"] != corpus_fingerprint(
                items, feature_version=version):
            raise ValueError("Reranker was trained for a different item corpus")
        if evaluation_split is not None:
            mode, fraction, seed = evaluation_split
            if (mode != "queries" or self.metadata["split"] != "queries"
                    or self.metadata["validation_fraction"] != fraction
                    or self.metadata["seed"] != seed):
                raise ValueError("Evaluate reranker only on its held-out query-text split")
        priors = artifact.get("priors", {})
        if version in {7, 8}:
            classifier = priors.get("microcategory_classifier")
            if (classifier is None or classifier.metadata != self.metadata.get("microcategory_classifier_metadata")
                    or self.metadata.get("training_prior_policy", {}).get("microcategory_classifier") != "grouped-out-of-fold"):
                raise ValueError("V7 reranker requires the matching saved OOF microcategory classifier")
            classifier._prediction_cache.clear()
        self.exact_microcategory_prior = priors.get("microcategory_prior")
        self.microcategory_exact_bonus = microcategory_exact_bonus
        self.builder = FeatureBuilder(
            items, config,
            microcategory_prior=(None if artifact.get("disable_microcategory_prior")
                                 else priors.get("microcategory_prior")),
            geo_associations=priors.get("geo_associations"),
            feature_names=expected,
            microcategory_classifier=priors.get("microcategory_classifier"))

    def rank_batch(self, queries, results, limit=50):
        """Считаем оценки пачкой и сортируем каждый пул отдельно."""
        queries, results = list(queries), list(results)
        if len(queries) != len(results):
            raise ValueError("Reranker query and result batch sizes differ")
        if limit < 0:
            raise ValueError("Reranker limit must be nonnegative")
        matrices = [self.builder.transform(query, details, baseline)
                    for query, (baseline, details) in zip(queries, results)]
        nonempty = [matrix for matrix in matrices if len(matrix)]
        if not nonempty:
            return [[] for _ in queries]
        probabilities = prediction_scores(
            self.model, np.concatenate(nonempty),
            estimator=getattr(self, "estimator", "histgb"),
            selected_iteration=getattr(self, "selected_iteration", None))
        predictions, start = [], 0
        for query, matrix, (_baseline, details) in zip(queries, matrices, results):
            count = len(matrix)
            if count:
                scores = probabilities[start:start + count].copy()
                if (getattr(self, "microcategory_exact_bonus", 0.0)
                        and getattr(self, "exact_microcategory_prior", None)):
                    # Aggregate training history by normalized query text,
                    # then shrink short histories toward zero. No test labels
                    # or query IDs enter this prior.
                    history = self.exact_microcategory_prior.exact_counts.get(
                        normalize(query.get("search_query", "")), {})
                    total = sum(history.values())
                    if total:
                        scale = self.microcategory_exact_bonus / (total + 3)
                        scores += np.fromiter(
                            (scale * history.get(
                                str(self.builder.items[item_id].get("item_microcat_id", "")), 0)
                             for item_id in details["candidate_pool"]),
                            dtype=np.float64, count=count)
                # При равенстве оценок модели используем исходный балл поиска.
                order = np.lexsort((np.arange(count), -matrix[:, 0],
                                    -scores))[:limit]
                predictions.append([details["candidate_pool"][int(index)] for index in order])
                start += count
            else:
                predictions.append([])
        return predictions
