"""Объединение лексического и семантического поиска."""

from __future__ import annotations


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
        channel_quota=10,
    ):
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if bm25_weight < 0 or dense_weight < 0 or bm25_weight + dense_weight <= 0:
            raise ValueError("At least one retrieval weight must be positive")
        self.bm25 = bm25
        self.dense = dense
        self.candidate_k = candidate_k
        self.rank_constant = rank_constant
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.channel_quota = channel_quota

    def retrieve_with_candidates(self, query, limit=50):
        ranked_lists = []
        if self.bm25_weight > 0:
            ranked_lists.append((self.bm25.retrieve(query, self.candidate_k), self.bm25_weight))
        if self.dense_weight > 0:
            ranked_lists.append((self.dense.retrieve(query, self.candidate_k), self.dense_weight))
        prediction = reciprocal_rank_fusion(
            ranked_lists,
            limit=limit,
            rank_constant=self.rank_constant,
            channel_quota=self.channel_quota,
        )
        candidate_pool = list(dict.fromkeys(
            item_id
            for item_ids, _weight in ranked_lists
            for item_id in item_ids
        ))
        return prediction, candidate_pool

    def retrieve(self, query, limit=50):
        prediction, _candidate_pool = self.retrieve_with_candidates(query, limit)
        return prediction
