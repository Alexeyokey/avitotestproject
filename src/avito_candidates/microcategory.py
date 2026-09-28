"""Prior over service microcategories learned only from permitted training rows.

Each distinct query text contributes one unit of evidence, so frequent generic
queries do not drown out rare service names. This is a ranking feature, never a
hard filter: other microcategories remain available in the candidate pool.
"""

from collections import Counter, defaultdict
from functools import lru_cache
import re

import pyarrow.parquet as pq

from .core import held_out, normalize


_WORDS = re.compile(r"(?u)\b\w{2,}\b")


class MicrocategoryPrior:
    def __init__(self, exact_counts, token_counts, token_totals):
        self.exact_counts = exact_counts
        self.token_counts = token_counts
        self.token_totals = token_totals

    @lru_cache(maxsize=100_000)
    def scores(self, text, exclude_self=False):
        text = normalize(text)
        exact_counts = self.exact_counts.get(text, {})
        exact_total = sum(exact_counts.values())
        own = ({category: count / exact_total
                for category, count in exact_counts.items()}
               if exact_total else {})
        # Training rows must not score themselves through their clicked category.
        exact = {} if exclude_self else own
        tokens = [token for token in set(_WORDS.findall(text))
                  if self.token_totals.get(token, 0) - bool(exclude_self and own) >= 5]
        token_scores = Counter()
        for token in tokens:
            total = self.token_totals[token] - bool(exclude_self and own)
            for category, count in self.token_counts[token].items():
                adjusted = count - own.get(category, 0.0) if exclude_self else count
                if adjusted > 0:
                    token_scores[category] += adjusted / total
        if tokens:
            for category in token_scores:
                token_scores[category] /= len(tokens)
        top = {category: 1 / rank for rank, (category, _score)
               in enumerate(token_scores.most_common(10), start=1)}
        return exact, token_scores, top

    def features(self, text, category, *, exclude_self=False):
        if not category:
            return 0.0, 0.0, 0.0
        exact, token, top = self.scores(text, exclude_self)
        return (exact.get(category, 0.0), token.get(category, 0.0),
                top.get(category, 0.0))


def fit_microcategory_prior(path, *, split="queries", fraction=0.2, seed=42,
                            batch_size=32768):
    """Fit from the same query-text split used to train the reranker."""
    if split not in {"queries", "all"}:
        raise ValueError("Microcategory prior requires queries or all split")
    if batch_size <= 0:
        raise ValueError("Microcategory batch size must be positive")
    by_text = defaultdict(Counter)
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=["search_query", "item_microcat_id"]
    ):
        texts, categories = batch.column(0).to_pylist(), batch.column(1).to_pylist()
        for raw_text, raw_category in zip(texts, categories):
            text = normalize(raw_text or "")
            category = str(raw_category or "")
            if not text or not category:
                continue
            if split == "queries" and held_out(text, fraction, seed):
                continue
            by_text[text][category] += 1
    token_counts = defaultdict(Counter)
    token_totals = Counter()
    for text, category_counts in by_text.items():
        total = sum(category_counts.values())
        for token in set(_WORDS.findall(text)):
            for category, count in category_counts.items():
                token_counts[token][category] += count / total
            token_totals[token] += 1
    return MicrocategoryPrior(dict(by_text), dict(token_counts), token_totals)
