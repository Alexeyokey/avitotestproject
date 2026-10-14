"""Связи локаций из выбранных объявлений без внешнего геокодирования."""

from collections import Counter, defaultdict

import pyarrow.parquet as pq

from .core import SEARCH_FIELDS, held_out, normalize, query_key


class LocationAssociations:
    """Связываем локацию поиска с локациями объявлений из обучения."""

    def __init__(self, corpus_locations, counts=None, totals=None, query_counts=None):
        self.corpus_locations = set(corpus_locations)
        self.counts = counts if counts is not None else defaultdict(Counter)
        self.totals = totals if totals is not None else Counter()
        # Optional training-only attribution allows all variants of one query
        # text to be removed without rebuilding the whole history per query.
        self.query_counts = query_counts

    def history(self, search_location, *, exclude_query_text=None):
        """Return counts after optionally removing this text at every location.

        Older serialized priors have no query_counts and remain valid when no
        exclusion is requested. Training exclusion requires tracked history.
        """
        location = str(search_location or "")
        counts = self.counts.get(location, {})
        total = self.totals.get(location, 0)
        if exclude_query_text is None:
            return counts, total
        by_text = getattr(self, "query_counts", None)
        if by_text is None:
            raise ValueError("Query-text geography exclusion requires tracked history")
        own = by_text.get(normalize(exclude_query_text), {}).get(location, {})
        adjusted = Counter({destination: count - own.get(destination, 0)
                            for destination, count in counts.items()
                            if count > own.get(destination, 0)})
        return adjusted, sum(adjusted.values())

    def destinations(self, search_location, *, max_locations=3, min_history=20,
                     include_related=False, exclude_query_text=None):
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
        # По умолчанию связи из train применяются при отсутствии точного ID;
        # расширение доступной точной локации включается явно.
        # Минимальная история защищает от выбора по единичному клику.
        counts, total = self.history(location, exclude_query_text=exclude_query_text)
        if total < min_history:
            return exact
        candidates = (
            (count, item_location)
            for item_location, count in counts.items()
            if item_location in self.corpus_locations and item_location != location
        )
        related = tuple(item_location for _count, item_location in
                        sorted(candidates, key=lambda pair: (-pair[0], pair[1]))
                        [:max(0, max_locations - len(exact))])
        return exact + related


def fit_location_associations(path, corpus_locations, *, split="all", fraction=0.2,
                              seed=42, batch_size=32768, track_query_texts=False):
    """Считаем связи поиска и объявления без ответов отложенной части.

    Режим ``all`` предназначен для итогового прогноза или обучения на всём
    наборе. При оценке с ``--split all`` связи не обучаются, потому что
    независимой обучающей части нет.
    """
    if split not in {"pairs", "queries", "all"}:
        raise ValueError("Invalid geography fit split")
    if batch_size <= 0:
        raise ValueError("Geography fit batch size must be positive")
    associations = LocationAssociations(
        corpus_locations, query_counts={} if track_query_texts else None)
    columns = ["search_query", "search_location_id", "item_location_id"]
    if split == "pairs":
        columns.extend([field for field in SEARCH_FIELDS if field not in columns])
        columns.append("item_id")
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pandas().fillna("").astype(str).to_dict("records"):
            # При локальной оценке отложенные клики не участвуют даже в
            # построении географических связей: иначе подсмотрели бы ответ.
            if split == "queries" and held_out(normalize(row["search_query"]), fraction, seed):
                continue
            if split == "pairs" and held_out((query_key(row), row["item_id"]), fraction, seed):
                continue
            search_location = row["search_location_id"]
            item_location = row["item_location_id"]
            if search_location and item_location:
                associations.counts[search_location][item_location] += 1
                associations.totals[search_location] += 1
                if track_query_texts:
                    text = normalize(row["search_query"])
                    locations = associations.query_counts.setdefault(text, {})
                    locations.setdefault(search_location, Counter())[item_location] += 1
    return associations
