"""Optional protected-pool fill matching the four-route research hypothesis.

Selection uses the original top998 ranks. Removing protected IDs must not
compress those ranks: the Tune evaluator skips their contributions in place.
"""
from __future__ import annotations

import math

import numpy as np


SUPPLEMENTAL_MODE = "micro-typed-dense-v1"
DEPTH = 998
FILL_RANK_CONSTANT = 60
ROUTES = ("supplemental_title_values", "supplemental_params_values",
          "micro_combined", "supplemental_dense_current")


def select_supplemental(routes, scores, protected_ids, *, maximum_pool_size=998):
    """Equal-route novel RRF; exact score ties use the string item ID."""
    protected = set(protected_ids)
    if len(protected) > maximum_pool_size or not 1 <= maximum_pool_size <= 998:
        raise ValueError("Protected pool exceeds the supplemental budget")
    if set(routes) != set(ROUTES) or set(scores) != set(ROUTES):
        raise ValueError("Supplemental fill requires all four declared routes")
    novel_scores = {}
    for name in ROUTES:
        ids = routes[name]
        if (len(ids) > DEPTH or len(ids) != len(set(ids))
                or set(scores[name]) != set(ids)):
            raise ValueError("Supplemental raw routes must be unique, bounded and fully scored")
        if any(not math.isfinite(float(score)) for score in scores[name].values()):
            raise ValueError("Supplemental raw scores must be finite")
        for rank, item_id in enumerate(ids, 1):
            if item_id not in protected:
                novel_scores[item_id] = novel_scores.get(item_id, 0.) + 1 / (FILL_RANK_CONSTANT + rank)
    selected = sorted(novel_scores, key=lambda item_id: (-novel_scores[item_id], item_id))[
        :maximum_pool_size - len(protected)]
    selected_set = set(selected)
    retained = {name: [item_id for item_id in routes[name] if item_id in selected_set]
                for name in ROUTES}
    retained_scores = {name: {item_id: float(scores[name][item_id]) for item_id in ids}
                       for name, ids in retained.items()}
    return selected, retained, retained_scores, {item_id: novel_scores[item_id] for item_id in selected}


class SupplementalFill:
    """Reuse one lexical score pass; keep the original dense request separate."""
    def __init__(self, microcategory_channel, dense):
        if (microcategory_channel.top_categories != 3
                or not {"title", "params", "description"}.issubset(microcategory_channel.indexes)
                or getattr(dense, "document_format", "current") != "current"):
            raise ValueError("Supplemental v1 requires top3 microcategories, all lexical fields and current dense format")
        if not callable(getattr(dense, "retrieve_batch_with_scores", None)):
            raise ValueError("Supplemental v1 requires scored batch dense retrieval")
        self.micro = microcategory_channel
        self.dense = dense

    def retrieve(self, query, protected_ids, *, dense_result=None, exclude_self=True):
        if dense_result is None:
            dense_result = self.dense.retrieve_batch_with_scores([query], DEPTH)[0]
        dense_ids, dense_scores = dense_result
        if set(dense_scores) != set(dense_ids):
            raise ValueError("Deep dense scores must cover every supplemental neighbor")
        lexical = self.micro._scores(query)
        routes, scores = {}, {}
        for field in ("title", "params"):
            index = self.micro.indexes[field]
            rows = np.flatnonzero(lexical[field] > 0)
            ids, measured = index._top(lexical[field], rows, DEPTH, return_scores=True)
            name = "supplemental_" + field + "_values"
            routes[name], scores[name] = ids, measured
        ids, measured, _diagnostics = self.micro.retrieve(query, DEPTH,
            existing_ids=(), field="combined", lexical_scores=lexical,
            exclude_self=exclude_self)
        routes["micro_combined"], scores["micro_combined"] = ids, measured
        routes["supplemental_dense_current"] = dense_ids
        scores["supplemental_dense_current"] = dense_scores
        return select_supplemental(routes, scores, protected_ids)
