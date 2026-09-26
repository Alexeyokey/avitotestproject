"""Объединение лексического и семантического поиска."""

from __future__ import annotations

import math


def reciprocal_rank_fusion(
    ranked_lists,
    *,
    limit=50,
    rank_constant=60,
    channel_quota=10,
):
    if limit < 0 or rank_constant < 0 or channel_quota < 0:
        raise ValueError("RRF parameters must be non-negative")
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

    fused = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))
    selected = fused[:limit]
    selected_set = set(selected)
    missing = [item_id for item_id in fused if item_id in guaranteed and item_id not in selected_set]
    if not missing:
        return selected

    droppable = [
        index for index in range(len(selected) - 1, -1, -1)
        if selected[index] not in guaranteed
    ]
    for item_id in missing[:len(droppable)]:
        selected[droppable.pop(0)] = item_id
    return sorted(selected, key=lambda item_id: (-scores[item_id], item_id))


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
    ):
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        weights = (bm25_weight, dense_weight)
        if (any(not math.isfinite(weight) or weight < 0 for weight in weights)
                or sum(weights) <= 0):
            raise ValueError("At least one retrieval weight must be positive")
        self.bm25 = bm25
        self.dense = dense
        self.candidate_k = candidate_k
        self.rank_constant = rank_constant
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.channel_quota = channel_quota

    def _ranked_channels(self, query):
        channels = []
        if self.bm25_weight > 0:
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
            channels.append(
                ("dense", self.dense.retrieve(query, self.candidate_k), self.dense_weight)
            )
        return channels

    def retrieve_with_diagnostics(self, query, limit=50):
        channels = self._ranked_channels(query)
        ranked_lists = [(item_ids, weight) for _name, item_ids, weight in channels]
        prediction = reciprocal_rank_fusion(
            ranked_lists,
            limit=limit,
            rank_constant=self.rank_constant,
            channel_quota=self.channel_quota,
        )
        candidate_pool = []
        seen = set()
        for _name, item_ids, _weight in channels:
            for item_id in item_ids:
                if item_id not in seen:
                    seen.add(item_id)
                    candidate_pool.append(item_id)
        return prediction, {
            "channels": {name: item_ids for name, item_ids, _weight in channels},
            "candidate_pool": candidate_pool,
            "prediction_without_quota": reciprocal_rank_fusion(
                ranked_lists,
                limit=limit,
                rank_constant=self.rank_constant,
                channel_quota=0,
            ),
            "prediction_with_quota_10": reciprocal_rank_fusion(
                ranked_lists,
                limit=limit,
                rank_constant=self.rank_constant,
                channel_quota=10,
            ),
        }

    def retrieve(self, query, limit=50):
        prediction, _diagnostics = self.retrieve_with_diagnostics(query, limit)
        return prediction
