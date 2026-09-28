"""Полевой BM25 по заголовку, параметрам и описанию объявления."""

from __future__ import annotations

from functools import lru_cache
from collections import defaultdict
import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import snowballstemmer

from .core import normalize
from .hybrid import reciprocal_rank_fusion


_TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")
_RUSSIAN_WORD = re.compile(r"[а-я]+")
_RUSSIAN_STEMMER = snowballstemmer.stemmer("russian")


@lru_cache(maxsize=500_000)
def _stem_word(word):
    # Числа и латинские названия сохраняем; стемминг применяем лишь к русским словам.
    return _RUSSIAN_STEMMER.stemWord(word) if _RUSSIAN_WORD.fullmatch(word) else word


def stem_text(text):
    """Используем границы токенов BM25 и приводим русские слова к основе."""
    return " ".join(_stem_word(word) for word in _TOKEN_PATTERN.findall(normalize(text)))


class BM25Index:
    def __init__(self, item_ids, texts, *, k1, b, transform=normalize):
        if not np.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be positive and finite")
        if not np.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        if len(item_ids) != len(texts):
            raise ValueError("item_ids and texts must have the same length")
        self.ids = list(item_ids)
        self.transform = transform
        self.vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
        self.index = None

        normalized = [self.transform(text) for text in texts]
        if not normalized or not any(self.vectorizer.build_analyzer()(text) for text in normalized):
            return
        counts = self.vectorizer.fit_transform(normalized).tocsr()
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        average_length = lengths.mean()
        # В разреженной матрице каждое ненулевое значение соответствует одному
        # документу с данным словом. Частоту по корпусу считаем до замены
        # обычных счётчиков слов на слагаемые формулы BM25.
        document_frequency = np.bincount(counts.indices, minlength=counts.shape[1])
        idf = np.log1p(
            (len(normalized) - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        length_penalty = k1 * (1 - b + b * lengths / average_length)
        row_penalty = np.repeat(length_penalty, np.diff(counts.indptr))
        counts.data = (
            idf[counts.indices] * counts.data * (k1 + 1) / (counts.data + row_penalty)
        ).astype(np.float32)
        self.index = counts.tocsc()

    def _score(self, text):
        """Считаем совпадения один раз для общего и географического списков."""
        if self.index is None:
            return None, None
        vector = self.vectorizer.transform([self.transform(text)])
        scores = np.zeros(len(self.ids), dtype=np.float32)
        matched = []
        for term in sorted(vector.indices):
            start, end = self.index.indptr[term:term + 2]
            rows = self.index.indices[start:end]
            scores[rows] += self.index.data[start:end]
            matched.append(rows)
        if not matched:
            return None, None
        return scores, np.unique(np.concatenate(matched))

    def _top(self, scores, candidates, limit, allowed=None, return_scores=False):
        if scores is None or limit == 0:
            return ([], {}) if return_scores else []
        # Маску географии применяем до выбора K лучших: местное объявление может
        # отсутствовать в глобальном списке, но попасть в отдельный геоканал.
        if allowed is not None:
            if len(allowed) != len(self.ids):
                raise ValueError("Allowed-item mask must match the index size")
            candidates = candidates[allowed[candidates]]
        order = np.lexsort((candidates, -scores[candidates]))[:limit]
        selected = candidates[order]
        ids = [self.ids[row] for row in selected]
        if return_scores:
            return ids, {self.ids[row]: float(scores[row]) for row in selected}
        return ids

    def retrieve(self, text, limit, *, allowed=None, return_scores=False):
        if limit < 0:
            raise ValueError("limit must be non-negative")
        scores, candidates = self._score(text) if limit else (None, None)
        return self._top(scores, candidates, limit, allowed, return_scores)

    def retrieve_global_and_allowed(self, text, global_limit, allowed_limit, allowed,
                                    *, return_scores=False):
        """Получаем два списка top-K без повторного расчёта оценок запроса."""
        if global_limit < 0 or allowed_limit < 0:
            raise ValueError("Limits must be non-negative")
        scores, candidates = self._score(text)
        global_result = self._top(scores, candidates, global_limit,
                                  return_scores=return_scores)
        allowed_result = self._top(scores, candidates, allowed_limit, allowed,
                                   return_scores=return_scores)
        return global_result, allowed_result


class Baseline:
    """Объединяет BM25-поля и стемминг, затем учитывает локацию."""

    def __init__(
        self,
        items,
        *,
        title_k1=1.0,
        title_b=0.2,
        params_k1=1.2,
        params_b=0.7,
        description_k1=1.0,
        description_b=0.8,
        title_weight=2.0,
        params_weight=1.0,
        description_weight=0.5,
        stem_title_weight=0.0,
        stem_params_weight=0.0,
        location_bonus=0.0,
        candidate_k=300,
        rank_constant=30,
        channel_quota=10,
        geo_candidate_k=0,
        geo_top_locations=3,
        geo_min_history=20,
        geo_weight=0.25,
        geo_associations=None,
    ):
        self.ids = [item["item_id"] for item in items]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Duplicate corpus item_id")
        weights = (title_weight, params_weight, description_weight,
                   stem_title_weight, stem_params_weight)
        if any(not np.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("BM25 field weights must be non-negative and finite")
        if sum(weights) <= 0:
            raise ValueError("At least one BM25 field weight must be positive")
        if not np.isfinite(location_bonus) or location_bonus < 0:
            raise ValueError("Location bonus must be non-negative and finite")
        if candidate_k <= 0 or rank_constant < 0 or channel_quota < 0:
            raise ValueError("BM25 fusion parameters are invalid")
        if (geo_candidate_k < 0 or geo_top_locations <= 0 or geo_min_history <= 0
                or not np.isfinite(geo_weight) or geo_weight <= 0):
            raise ValueError("BM25 geography parameters are invalid")

        self.title_weight = title_weight
        self.params_weight = params_weight
        self.description_weight = description_weight
        self.stem_title_weight = stem_title_weight
        self.stem_params_weight = stem_params_weight
        self.location_bonus = location_bonus
        self._item_locations = (
            {item["item_id"]: str(item.get("item_location_id", "")) for item in items}
            if location_bonus > 0 or geo_candidate_k > 0 else None
        )
        self.geo_candidate_k = geo_candidate_k
        self.geo_top_locations = geo_top_locations
        self.geo_min_history = geo_min_history
        self.geo_weight = geo_weight
        self.geo_associations = geo_associations
        self._geo_rows = defaultdict(list)
        if geo_candidate_k:
            for row, item in enumerate(items):
                location = str(item.get("item_location_id", ""))
                if location:
                    self._geo_rows[location].append(row)
        self.candidate_k = candidate_k
        self.rank_constant = rank_constant
        self.channel_quota = channel_quota
        self.title_index = BM25Index(
            self.ids,
            [item.get("item_title_raw", "") for item in items],
            k1=title_k1,
            b=title_b,
        )
        self.params_index = BM25Index(
            self.ids,
            [item.get("item_infm_params_text", "") for item in items],
            k1=params_k1,
            b=params_b,
        )
        self.description_index = BM25Index(
            self.ids,
            [item.get("item_description_raw", "") for item in items],
            k1=description_k1,
            b=description_b,
        )
        # Дополнительные индексы строим только для включённых каналов.
        self.stem_title_index = (
            BM25Index(
                self.ids,
                [item.get("item_title_raw", "") for item in items],
                k1=title_k1,
                b=title_b,
                transform=stem_text,
            ) if stem_title_weight > 0 else None
        )
        self.stem_params_index = (
            BM25Index(
                self.ids,
                [item.get("item_infm_params_text", "") for item in items],
                k1=params_k1,
                b=params_b,
                transform=stem_text,
            ) if stem_params_weight > 0 else None
        )

    def ranked_lists(self, query, limit):
        """Возвращаем взвешенные списки кандидатов через прежний интерфейс."""
        channels, _scores = self.ranked_lists_with_scores(query, limit)
        return channels

    def ranked_lists_with_scores(self, query, limit):
        """Сохраняем исходные оценки BM25 выбранных объявлений по каждому полю."""
        if limit < 0:
            raise ValueError("limit must be non-negative")
        text = query.get("search_query", "")
        fields = (
            ("title", self.title_index, self.title_weight),
            ("params", self.params_index, self.params_weight),
            ("description", self.description_index, self.description_weight),
            ("title_stem", self.stem_title_index, self.stem_title_weight),
            ("params_stem", self.stem_params_index, self.stem_params_weight),
        )
        channels = []
        channel_scores = {}
        geo_mask = self._geo_mask(query) if self.geo_candidate_k else None
        for name, index, weight in fields:
            if weight > 0:
                if geo_mask is None:
                    ids, scores = index.retrieve(text, limit, return_scores=True)
                    channels.append((name, ids, weight))
                    channel_scores[name] = scores
                else:
                    (global_ids, global_scores), (geo_ids, geo_scores) = (
                        index.retrieve_global_and_allowed(
                            text, limit, self.geo_candidate_k, geo_mask,
                            return_scores=True))
                    channels.append((name, global_ids, weight))
                    channels.append((f"{name}_geo", geo_ids, weight * self.geo_weight))
                    channel_scores[name] = global_scores
                    channel_scores[f"{name}_geo"] = geo_scores
        return channels, channel_scores

    def geo_locations(self, query):
        """Локации, выбранные географическими каналами для запроса."""
        location = str(query.get("search_location_id", ""))
        if not location:
            return ()
        if location in self._geo_rows:
            return (location,)
        if self.geo_associations is not None:
            return self.geo_associations.destinations(
                location, max_locations=self.geo_top_locations,
                min_history=self.geo_min_history)
        return ()

    def _geo_mask(self, query):
        """Точная или найденная по обучению локация; общий поиск сохраняется."""
        destinations = self.geo_locations(query)
        if not destinations:
            return None
        mask = np.zeros(len(self.ids), dtype=bool)
        for destination in destinations:
            mask[self._geo_rows[destination]] = True
        return mask

    def retrieve_with_diagnostics(self, query, limit=50):
        """Возвращаем итоговый порядок и все уникальные объявления каналов BM25."""
        depth = max(limit, self.candidate_k)
        channels, channel_scores = self.ranked_lists_with_scores(query, depth)
        candidate_pool = list(dict.fromkeys(
            item_id for _name, item_ids, _weight in channels for item_id in item_ids
        ))
        score_boosts = self.location_score_boosts(query, candidate_pool)
        prediction, rrf_scores = reciprocal_rank_fusion(
            [(item_ids, weight) for _name, item_ids, weight in channels],
            limit=limit,
            rank_constant=self.rank_constant,
            channel_quota=self.channel_quota,
            score_boosts=score_boosts,
            return_scores=True,
        )
        # Пул включает результаты всех включённых полей до отбора 50 по RRF.
        return prediction, {
            "candidate_pool": candidate_pool,
            "rrf_scores": rrf_scores,
            "channels": {name: item_ids for name, item_ids, _weight in channels},
            "channel_scores": channel_scores,
            "geo_locations": self.geo_locations(query) if self.geo_candidate_k else (),
        }

    def location_score_boosts(self, query, item_ids):
        """Повышаем точные совпадения локации, сохраняя неместных кандидатов."""
        location = str(query.get("search_location_id", ""))
        if self.location_bonus == 0 or not location:
            return {}
        return {item_id: self.location_bonus for item_id in item_ids
                if self._item_locations[item_id] == location}

    def retrieve(self, query, limit=50):
        prediction, _diagnostics = self.retrieve_with_diagnostics(query, limit)
        return prediction
