"""Transparent baseline: lexical retrieval + observed choices, fused using RRF.

No benchmark query IDs enter scoring. Defaults are starting points, not tuned weights.
The corpus itself may be indexed in validation; held-out interactions may not.
"""
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .core import normalize, query_key


class Baseline:
    def __init__(self, items, train, per_channel_k=300):
        self.items = items
        self.ids = [x["item_id"] for x in items]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Duplicate corpus item_id")
        self.allowed = set(self.ids)
        self.per_channel_k = per_channel_k
        self.history = defaultdict(Counter)
        self.local = defaultdict(Counter)
        self.exact = defaultdict(Counter)
        self.popularity = Counter()
        for row in train:
            iid = row["item_id"]
            if iid not in self.allowed:
                continue
            key = query_key(row)
            self.history[key[0]][iid] += 1
            self.local[key[:2]][iid] += 1
            self.exact[key][iid] += 1
            self.popularity[iid] += 1
        # Start with compact fields; description BM25 and dense retrieval are
        # separate future channels, to measure their marginal contribution.
        texts = [normalize(x.get("item_title_raw", "") + " " +
                           x.get("item_infm_params_text", "")) for x in items]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                        dtype=np.float32, max_features=400_000)
        self.matrix = self.vectorizer.fit_transform(texts)
        self.fallback = sorted(self.ids, key=lambda x: (-self.popularity[x], x))

    def retrieve(self, query, limit=50):
        key = query_key(query)
        vector = self.vectorizer.transform([key[0]])
        similarities = (self.matrix @ vector.T).tocoo()
        # Explicit tie-breaking makes results reproducible across runs.
        order = np.lexsort((similarities.row, -similarities.data))[:self.per_channel_k]
        lexical = [self.ids[similarities.row[i]] for i in order]

        def ranked(counter):
            return sorted(counter, key=lambda x: (-counter[x], x))[:self.per_channel_k]

        channels = [(lexical, 1.0), (ranked(self.history.get(key[0], {})), 1.0),
                    (ranked(self.local.get(key[:2], {})), 1.0),
                    (ranked(self.exact.get(key, {})), 1.0)]
        scores = defaultdict(float)
        for ids, weight in channels:
            for rank, iid in enumerate(ids, 1):
                scores[iid] += weight / (60 + rank)
        result = sorted(scores, key=lambda x: (-scores[x], x))[:limit]
        # Fallback is intentionally weak but guarantees a valid full submission.
        if len(result) < limit:
            selected = set(result)
            for iid in self.fallback:
                if iid not in selected:
                    result.append(iid)
                    selected.add(iid)
                if len(result) >= limit:
                    break
        return result
