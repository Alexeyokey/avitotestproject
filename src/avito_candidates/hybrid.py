"""Объединение лексического и семантического поиска."""

from __future__ import annotations

import math


def reciprocal_rank_fusion(
    ranked_lists,
    *,
    limit=50,
    rank_constant=60,
    channel_quota=10,
    score_boosts=None,
    return_scores=False,
):
    """Объединяем ранжированные списки и при необходимости возвращаем баллы.

    Баллы учитывают добавки запроса. Квота канала может изменить состав
    итогового списка, но не баллы RRF отдельных объявлений.
    """
    if limit < 0 or rank_constant < 0 or channel_quota < 0:
        raise ValueError("RRF parameters must be non-negative")
    ranked_lists = list(ranked_lists)
    scores = {}
    guaranteed = set()
    for item_ids, weight in ranked_lists:
        seen = set()
        unique = []
        for item_id in item_ids:
            if item_id not in seen:
                seen.add(item_id)
                unique.append(item_id)
        guaranteed.update(unique[:channel_quota])
        for rank, item_id in enumerate(unique, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (rank_constant + rank)

    # Добавки запроса меняют порядок только среди найденных кандидатов.
    # Объявления вне всех поисковых каналов сюда не добавляются.
    if score_boosts:
        for item_id, boost in score_boosts.items():
            if item_id in scores:
                scores[item_id] += boost

    fused = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))
    selected = fused[:limit]
    selected_set = set(selected)
    missing = [item_id for item_id in fused if item_id in guaranteed and item_id not in selected_set]
    if not missing:
        return (selected, scores) if return_scores else selected

    droppable = [
        index for index in range(len(selected) - 1, -1, -1)
        if selected[index] not in guaranteed
    ]
    for item_id in missing[:len(droppable)]:
        selected[droppable.pop(0)] = item_id
    selected = sorted(selected, key=lambda item_id: (-scores[item_id], item_id))
    return (selected, scores) if return_scores else selected


def rank_location_bonus_grid(
    channels, weights, item_locations, location, bonuses, *, rank_constant=60, limit=50
):
    """Пересортировываем один пул для нескольких географических добавок.

    Добавка одинакова для всех совпадений точной локации, поэтому их взаимный
    порядок не меняется. Один раз сортируем местные и остальные объявления,
    затем объединяем верхние ``limit`` для каждой добавки. Результат совпадает
    с RRF без квот и не требует повторного поиска BM25, dense или HNSW.
    """
    if limit < 0 or rank_constant < 0:
        raise ValueError("RRF parameters must be non-negative")
    bonuses = tuple(bonuses)
    if any(not math.isfinite(bonus) or bonus < 0 for bonus in bonuses):
        raise ValueError("Location bonuses must be finite and non-negative")
    scores = {}
    for name, item_ids in channels.items():
        weight = weights[name]
        seen = set()
        rank = 0
        for item_id in item_ids:
            if item_id in seen:
                continue
            seen.add(item_id)
            rank += 1
            scores[item_id] = scores.get(item_id, 0.0) + weight / (rank_constant + rank)

    local = sorted(
        (item_id for item_id in scores if location and item_locations.get(item_id) == location),
        key=lambda item_id: (-scores[item_id], item_id),
    )
    other = sorted(
        (item_id for item_id in scores if not location or item_locations.get(item_id) != location),
        key=lambda item_id: (-scores[item_id], item_id),
    )
    output = {}
    for bonus in bonuses:
        selected, local_index, other_index = [], 0, 0
        while len(selected) < limit and (local_index < len(local) or other_index < len(other)):
            if local_index == len(local):
                take_local = False
            elif other_index == len(other):
                take_local = True
            else:
                local_id, other_id = local[local_index], other[other_index]
                take_local = (-scores[local_id] - bonus, local_id) <= (
                    -scores[other_id], other_id
                )
            if take_local:
                selected.append(local[local_index])
                local_index += 1
            else:
                selected.append(other[other_index])
                other_index += 1
        output[bonus] = selected
    return output


class HybridRetriever:
    def __init__(
        self,
        bm25,
        dense,
        *,
        candidate_k=300,
        rank_constant=60,
        bm25_weight=1.0,
        dense_weight=1.0,
        channel_quota=0,
        location_bonus=0.0,
    ):
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        weights = (bm25_weight, dense_weight)
        if (any(not math.isfinite(weight) or weight < 0 for weight in weights)
                or sum(weights) <= 0):
            raise ValueError("At least one retrieval weight must be positive")
        if not math.isfinite(location_bonus) or location_bonus < 0:
            raise ValueError("Location bonus must be non-negative and finite")
        if location_bonus > 0 and getattr(bm25, "_item_locations", None) is None:
            raise ValueError("Location bonus requires BM25 item locations")
        self.bm25 = bm25
        self.dense = dense
        self.candidate_k = candidate_k
        self.rank_constant = rank_constant
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.channel_quota = channel_quota
        self.location_bonus = location_bonus

    def _ranked_channels(self, query, dense_result=None):
        channels = []
        channel_scores = {}
        if self.bm25_weight > 0:
            if hasattr(self.bm25, "ranked_lists_with_scores"):
                bm25_lists, bm25_scores = self.bm25.ranked_lists_with_scores(
                    query, self.candidate_k)
                channel_scores.update(bm25_scores)
            else:
                bm25_lists = self.bm25.ranked_lists(query, self.candidate_k)
            total_field_weight = sum(weight for _name, _items, weight in bm25_lists)
            channels.extend(
                (
                    name,
                    item_ids,
                    self.bm25_weight * field_weight / total_field_weight,
                )
                for name, item_ids, field_weight in bm25_lists
            )
        if self.dense_weight > 0:
            if dense_result is None:
                if hasattr(self.dense, "retrieve_with_scores"):
                    dense_result = self.dense.retrieve_with_scores(query, self.candidate_k)
                else:
                    dense_result = (self.dense.retrieve(query, self.candidate_k), {})
            dense_ids, dense_scores = dense_result
            channels.append(("dense", dense_ids, self.dense_weight))
            channel_scores["dense"] = dense_scores
        return channels, channel_scores

    def _fuse_with_diagnostics(self, query, ranked, limit):
        channels, channel_scores = ranked
        ranked_lists = [(item_ids, weight) for _name, item_ids, weight in channels]
        candidate_pool = []
        seen = set()
        for _name, item_ids, _weight in channels:
            for item_id in item_ids:
                if item_id not in seen:
                    seen.add(item_id)
                    candidate_pool.append(item_id)
        score_boosts = (self.bm25.location_score_boosts(query, candidate_pool)
                        if self.location_bonus > 0 else {})
        prediction, rrf_scores = reciprocal_rank_fusion(
            ranked_lists,
            limit=limit,
            rank_constant=self.rank_constant,
            channel_quota=self.channel_quota,
            score_boosts=score_boosts,
            return_scores=True,
        )
        return prediction, {
            "channels": {name: item_ids for name, item_ids, _weight in channels},
            "channel_scores": channel_scores,
            "channel_weights": {name: weight for name, _item_ids, weight in channels},
            "geo_locations": (self.bm25.geo_locations(query)
                              if self.bm25_weight > 0 and hasattr(self.bm25, "geo_locations")
                              and self.bm25.geo_candidate_k else ()),
            "candidate_pool": candidate_pool,
            "rrf_scores": rrf_scores,
            "prediction_without_quota": reciprocal_rank_fusion(
                ranked_lists,
                limit=limit,
                rank_constant=self.rank_constant,
                channel_quota=0,
                score_boosts=score_boosts,
            ),
            "prediction_with_quota_10": reciprocal_rank_fusion(
                ranked_lists,
                limit=limit,
                rank_constant=self.rank_constant,
                channel_quota=10,
                score_boosts=score_boosts,
            ),
        }

    def retrieve_with_diagnostics(self, query, limit=50):
        return self._fuse_with_diagnostics(
            query, self._ranked_channels(query), limit
        )

    def retrieve_batch_with_diagnostics(self, queries, limit=50):
        """Ищем dense-кандидатов пачкой, сохраняя BM25 и RRF для каждого запроса."""
        queries = list(queries)
        if self.dense_weight > 0:
            if hasattr(self.dense, "retrieve_batch_with_scores"):
                dense_results = self.dense.retrieve_batch_with_scores(
                    queries, self.candidate_k)
            else:
                dense_results = [(ids, {}) for ids in
                                 self.dense.retrieve_batch(queries, self.candidate_k)]
        else:
            dense_results = [None] * len(queries)
        return [
            self._fuse_with_diagnostics(
                query, self._ranked_channels(query, dense_ids), limit
            )
            for query, dense_ids in zip(queries, dense_results)
        ]

    def retrieve(self, query, limit=50):
        prediction, _diagnostics = self.retrieve_with_diagnostics(query, limit)
        return prediction
