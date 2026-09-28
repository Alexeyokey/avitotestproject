"""Полевой BM25 по заголовку, параметрам и описанию объявления."""

from __future__ import annotations

from functools import lru_cache
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
    # Keep numbers and Latin brand names intact; stem only Russian words.
    return _RUSSIAN_STEMMER.stemWord(word) if _RUSSIAN_WORD.fullmatch(word) else word


def stem_text(text):
    """Use the same token boundaries as BM25, then stem Russian tokens."""
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

    def retrieve(self, text, limit):
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0 or self.index is None:
            return []
        vector = self.vectorizer.transform([self.transform(text)])
        scores = np.zeros(len(self.ids), dtype=np.float32)
        matched = []
        for term in sorted(vector.indices):
            start, end = self.index.indptr[term:term + 2]
            rows = self.index.indices[start:end]
            scores[rows] += self.index.data[start:end]
            matched.append(rows)
        if not matched:
            return []
        candidates = np.unique(np.concatenate(matched))
        order = np.lexsort((candidates, -scores[candidates]))[:limit]
        return [self.ids[candidates[index]] for index in order]


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

        self.title_weight = title_weight
        self.params_weight = params_weight
        self.description_weight = description_weight
        self.stem_title_weight = stem_title_weight
        self.stem_params_weight = stem_params_weight
        self.location_bonus = location_bonus
        self._item_locations = (
            {item["item_id"]: str(item.get("item_location_id", "")) for item in items}
            if location_bonus > 0 else None
        )
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
        # Build extra indices only when their channels are enabled.
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
        for name, index, weight in fields:
            if weight > 0:
                channels.append((name, index.retrieve(text, limit), weight))
        return channels

    def retrieve_with_diagnostics(self, query, limit=50):
        """Return the final ranking and every unique item found by BM25 fields."""
        depth = max(limit, self.candidate_k)
        channels = self.ranked_lists(query, depth)
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
        # The pool includes results from all enabled fields before RRF keeps 50.
        return prediction, {
            "candidate_pool": candidate_pool,
            "rrf_scores": rrf_scores,
            "channels": {name: item_ids for name, item_ids, _weight in channels},
        }

    def location_score_boosts(self, query, item_ids):
        """Promote exact-location matches while retaining nonlocal candidates."""
        location = str(query.get("search_location_id", ""))
        if self.location_bonus == 0 or not location:
            return {}
        return {item_id: self.location_bonus for item_id in item_ids
                if self._item_locations[item_id] == location}

    def retrieve(self, query, limit=50):
        prediction, _diagnostics = self.retrieve_with_diagnostics(query, limit)
        return prediction
