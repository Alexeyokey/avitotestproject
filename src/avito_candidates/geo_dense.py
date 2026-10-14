"""Exact dense retrieval with geography applied before candidate truncation.

The encoder and its stored vectors are unchanged. The only fitted information
accepted here is a LocationAssociations object prepared from the fit partition.
Coordinates describe the catalog, and never use validation relevance labels.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import numpy as np


def catalog_location_centers(items):
    """Median coordinates per location, using available catalog metadata only."""
    coordinates = defaultdict(list)
    for item in items:
        location = str(item.get("item_location_id") or "")
        if not location:
            continue
        try:
            lat = float(item.get("item_latitude"))
            lon = float(item.get("item_longitude"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(lat) and np.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180:
            coordinates[location].append((lat, lon))
    return {location: tuple(np.median(points, axis=0))
            for location, points in coordinates.items()}


def _distance_km(center, coordinates):
    lat1, lon1 = np.radians(center)
    lat2, lon2 = np.radians(coordinates).T
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class GeoDensePrior:
    """Smoothed fit-only location probabilities in an exact corpus order.

    Score = cosine + weight * log(max(P(destination | search location), floor)).
    Smoothing mass is spread over catalog locations with an exponential
    distance kernel; unknown coordinates receive a small nonzero mass.
    Noncatalog search-region centers use fit destination counts and catalog
    centers. No history means no adjustment, even for a known catalog city.
    Location probability is not divided by its number of catalog documents.
    """

    def __init__(self, items, associations, *, location_centers=None,
                 smoothing_mass=3.0, distance_scale_km=40.0,
                 probability_floor=1e-7):
        for name, value in (("smoothing_mass", smoothing_mass),
                            ("distance_scale_km", distance_scale_km)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(probability_floor) or not 0 < probability_floor < 1:
            raise ValueError("probability_floor must be between zero and one")
        items = list(items)
        self.ids = [str(item["item_id"]) for item in items]
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Duplicate corpus item_id")
        self.associations = associations
        self.smoothing_mass = float(smoothing_mass)
        self.distance_scale_km = float(distance_scale_km)
        self.probability_floor = float(probability_floor)
        locations = [str(item.get("item_location_id") or "") for item in items]
        self.locations = sorted(set(locations))
        by_location = {value: index for index, value in enumerate(self.locations)}
        self.location_labels = np.asarray([by_location[value] for value in locations], dtype=np.int64)
        self.centers = (catalog_location_centers(items) if location_centers is None
                        else dict(location_centers))
        for location, center in self.centers.items():
            array = np.asarray(center, dtype=np.float64)
            if (array.shape != (2,) or not np.isfinite(array).all()
                    or not -90 <= array[0] <= 90 or not -180 <= array[1] <= 180):
                raise ValueError(f"Invalid coordinate center for {location}")
        self._tie_order = np.argsort(np.asarray(self.ids), kind="stable")
        self._tie_rank = np.empty(len(self.ids), dtype=np.int64)
        self._tie_rank[self._tie_order] = np.arange(len(self.ids))
        # Bounded cache: avoid a locations x corpus allocation for every city.
        self._cached_prior = lru_cache(maxsize=8)(self._compute_prior)

    def _compute_prior(self, search_location, exclude_query_text):
        counts, total = self.associations.history(
            search_location, exclude_query_text=exclude_query_text)
        values = np.asarray(list(counts.values()), dtype=np.float64)
        if not np.isfinite(total) or total < 0 or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid fitted geography counts")
        if not np.isclose(values.sum(), total, rtol=1e-10, atol=1e-8):
            raise ValueError("Geography counts disagree with fitted total")
        if not search_location or total == 0 or not self.ids:
            result = np.zeros(len(self.ids), dtype=np.float32)
            result.setflags(write=False)
            return result

        center = self.centers.get(search_location)
        if center is None:
            weighted = [(self.centers[location], count) for location, count in counts.items()
                        if count > 0 and location in self.centers]
            if weighted:
                # Only fit history determines a noncatalog region's center.
                center = np.average([value for value, _ in weighted], axis=0,
                                    weights=[count for _, count in weighted])
        smoothing = np.ones(len(self.locations), dtype=np.float64)
        if center is not None:
            known = [i for i, location in enumerate(self.locations) if location in self.centers]
            smoothing.fill(self.probability_floor)
            if known:
                distances = _distance_km(center, np.asarray([self.centers[self.locations[i]] for i in known]))
                smoothing[known] = np.maximum(np.exp(-distances / self.distance_scale_km), self.probability_floor)
        # Missing item location should never receive a geographic advantage.
        if "" in self.locations:
            smoothing[self.locations.index("")] = self.probability_floor
        smoothing /= smoothing.sum()
        destination_counts = np.asarray([counts.get(location, 0) for location in self.locations], dtype=np.float64)
        probabilities = (destination_counts + self.smoothing_mass * smoothing) / (total + self.smoothing_mass)
        logs = np.log(np.maximum(probabilities, self.probability_floor))
        result = np.asarray(logs[self.location_labels], dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Invalid geographic log probabilities")
        result.setflags(write=False)
        return result

    def log_prior(self, search_location, *, exclude_query_text=None):
        """Readonly float32 vector; optionally subtract this text's fit history."""
        location = str(search_location or "")
        return self._cached_prior(location, exclude_query_text)

    def rank(self, cosine_scores, search_location, limit=449, weight=0.05,
             *, exclude_query_text=None):
        """Select globally, breaking exact score ties by the original item ID."""
        if not isinstance(limit, (int, np.integer)) or limit < 0:
            raise ValueError("limit must be a nonnegative integer")
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("weight must be finite and nonnegative")
        cosines = np.asarray(cosine_scores, dtype=np.float32)
        if cosines.shape != (len(self.ids),) or not np.isfinite(cosines).all():
            raise ValueError("Expected finite cosine scores for the entire corpus")
        if np.any(np.abs(cosines) > 1.0001):
            raise ValueError("Cosine scores must lie between -1 and 1")
        count = min(limit, len(self.ids))
        if count == 0:
            return np.empty(0, dtype=np.int64)
        scores = cosines if weight == 0 else cosines + weight * self.log_prior(
            search_location, exclude_query_text=exclude_query_text)
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite geography-adjusted scores")
        # Argpartition alone has arbitrary ties at the cutoff. Explicitly
        # select tied IDs before sorting the small retained candidate set.
        threshold = np.partition(scores, len(scores) - count)[len(scores) - count]
        above = np.flatnonzero(scores > threshold)
        ties = np.flatnonzero(scores == threshold)
        ties = ties[np.argsort(self._tie_rank[ties], kind="stable")[:count - len(above)]]
        labels = np.concatenate((above, ties))
        return labels[np.lexsort((self._tie_rank[labels], -scores[labels]))]

    def retrieve(self, cosine_scores, search_location, limit=449, weight=0.05,
                 *, exclude_query_text=None):
        """Return existing dense API output, retaining RAW cosine as a feature."""
        labels = self.rank(cosine_scores, search_location, limit, weight,
                           exclude_query_text=exclude_query_text)
        ids = [self.ids[int(label)] for label in labels]
        return ids, {item_id: float(cosine_scores[int(label)])
                     for item_id, label in zip(ids, labels)}


class ExactDenseScorer:
    """Batched full-corpus cosine from saved vectors, without any encoder.

    The saved float32 corpus is retained without a CPU copy, and validated in
    blocks. CUDA uploads it once; query batch size bounds the score matrix.
    Already unit-normalized HNSW vectors are kept exactly as stored. Float32
    math avoids FP16 perturbations; CPU selection stays deterministic.
    """

    def __init__(self, corpus_vectors, *, device="cpu", item_block_size=32768):
        if not isinstance(item_block_size, int) or item_block_size <= 0:
            raise ValueError("item_block_size must be positive")
        corpus = np.asarray(corpus_vectors, dtype=np.float32)
        if corpus.ndim != 2 or corpus.shape[1] == 0:
            raise ValueError("Invalid corpus vectors")
        self._norms = np.empty(len(corpus), dtype=np.float32)
        for start in range(0, len(corpus), item_block_size):
            stop = start + item_block_size
            block = corpus[start:stop]
            if not np.isfinite(block).all():
                raise ValueError("Invalid corpus vectors")
            # Einsum avoids an array as large as the block for square(x).
            norms = np.sqrt(np.einsum("ij,ij->i", block, block, dtype=np.float64))
            if (norms <= 0).any() or not np.isfinite(norms).all():
                raise ValueError("Corpus vectors must have positive finite norms")
            self._norms[start:stop] = np.where(np.abs(norms - 1) <= 1e-4, 1, norms)
        self.corpus = corpus
        self.item_block_size = item_block_size
        self.device = device
        if device != "cpu":
            import torch
            self._torch = torch
            self._torch_device = torch.device(device)
            if self._torch_device.type != "cuda":
                raise ValueError("Only cpu and cuda scoring are supported")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            with torch.inference_mode():
                # One upload for all query batches, not one transfer of the
                # entire corpus per call. The 200k x 1024 float32 corpus is
                # about 0.8 GB and fits alongside batch scores on a 6 GB GPU.
                self._corpus_gpu = torch.as_tensor(corpus, device=self._torch_device)
                if np.any(self._norms != 1):
                    self._corpus_gpu.div_(torch.as_tensor(self._norms, device=self._torch_device)[:, None])

    def score(self, query_vectors):
        queries = np.asarray(query_vectors, dtype=np.float32)
        if (queries.ndim != 2 or queries.shape[1] != self.corpus.shape[1]
                or not np.isfinite(queries).all()):
            raise ValueError("Invalid query vectors")
        norms = np.linalg.norm(queries, axis=1)
        if (norms <= 0).any() or not np.isfinite(norms).all():
            raise ValueError("Query vectors must have positive finite norms")
        queries = queries / norms[:, None]
        output = np.empty((len(queries), len(self.corpus)), dtype=np.float32)
        if not len(queries):
            return output
        if self.device == "cpu":
            for start in range(0, len(self.corpus), self.item_block_size):
                stop = start + self.item_block_size
                block = self.corpus[start:stop]
                norms = self._norms[start:stop]
                if np.any(norms != 1):
                    block = block / norms[:, None]
                output[:, start:stop] = queries @ block.T
        else:
            torch = self._torch
            # Disable optional reduced-precision matmul for this computation.
            old_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                with torch.inference_mode():
                    query_gpu = torch.as_tensor(queries, device=self._torch_device)
                    for start in range(0, len(self.corpus), self.item_block_size):
                        stop = start + self.item_block_size
                        block = self._corpus_gpu[start:stop]
                        output[:, start:stop] = (query_gpu @ block.T).cpu().numpy()
            finally:
                torch.backends.cuda.matmul.allow_tf32 = old_tf32
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite cosine scores")
        return np.clip(output, -1, 1, out=output)
