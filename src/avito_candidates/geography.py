"""Связи локаций из выбранных объявлений без внешнего геокодирования."""

from collections import Counter, defaultdict

import pyarrow.parquet as pq

from .core import SEARCH_FIELDS, held_out, normalize, query_key


class LocationAssociations:
    """Связываем локацию поиска с локациями объявлений из обучения."""

    def __init__(self, corpus_locations, counts=None, totals=None):
        self.corpus_locations = set(corpus_locations)
        self.counts = counts if counts is not None else defaultdict(Counter)
        self.totals = totals if totals is not None else Counter()

    def destinations(self, search_location, *, max_locations=3, min_history=20,
                     include_related=False):
        """Keep the exact city, optionally adding historically related cities.

        The extra cities only widen the geographic retrieval channel; they
        never remove global candidates or impose a hard location filter.
        """
        location = str(search_location or "")
        if not location:
            return ()
        exact = (location,) if location in self.corpus_locations else ()
        if exact and not include_related:
            return exact
        if self.totals[location] < min_history:
            return exact
        candidates = (
            (count, item_location)
            for item_location, count in self.counts[location].items()
            if item_location in self.corpus_locations and item_location != location
        )
        related = tuple(item_location for _count, item_location in
                        sorted(candidates, key=lambda pair: (-pair[0], pair[1]))
                        [:max(0, max_locations - len(exact))])
        return exact + related


def fit_location_associations(path, corpus_locations, *, split="all", fraction=0.2,
                              seed=42, batch_size=32768):
    """Считаем связи поиска и объявления без ответов отложенной части.

    Режим ``all`` предназначен для итогового прогноза или обучения на всём
    наборе. При оценке с ``--split all`` связи не обучаются, потому что
    независимой обучающей части нет.
    """
    if split not in {"pairs", "queries", "all"}:
        raise ValueError("Invalid geography fit split")
    if batch_size <= 0:
        raise ValueError("Geography fit batch size must be positive")
    associations = LocationAssociations(corpus_locations)
    columns = ["search_query", "search_location_id", "item_location_id"]
    if split == "pairs":
        columns.extend([field for field in SEARCH_FIELDS if field not in columns])
        columns.append("item_id")
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pandas().fillna("").astype(str).to_dict("records"):
            if split == "queries" and held_out(normalize(row["search_query"]), fraction, seed):
                continue
            if split == "pairs" and held_out((query_key(row), row["item_id"]), fraction, seed):
                continue
            search_location = row["search_location_id"]
            item_location = row["item_location_id"]
            if search_location and item_location:
                associations.counts[search_location][item_location] += 1
                associations.totals[search_location] += 1
    return associations
