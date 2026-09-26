"""Семантический поиск по эмбеддингам с локальным HNSW-индексом."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .core import normalize


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
_CACHE_VERSION = 1


def item_text(item) -> str:
    return normalize(
        f"{item.get('item_title_raw', '')} {item.get('item_infm_params_text', '')}"
    )


def _fingerprint(items, model_name: str) -> str:
    digest = hashlib.sha256(f"{_CACHE_VERSION}\0{model_name}\0".encode())
    for item in items:
        digest.update(str(item["item_id"]).encode())
        digest.update(b"\0")
        digest.update(item_text(item).encode())
        digest.update(b"\0")
    return digest.hexdigest()[:20]


def _load_dependencies():
    try:
        import hnswlib
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Для dense и hybrid установите зависимости: pip install -e '.[dense]'"
        ) from error
    return hnswlib, SentenceTransformer


class DenseRetriever:
    def __init__(
        self,
        items,
        *,
        model_name=DEFAULT_EMBEDDING_MODEL,
        cache_dir=Path("artifacts/dense"),
        batch_size=64,
        encode_chunk_size=4096,
        device=None,
        ef_construction=200,
        ef_search=300,
        m=32,
    ):
        if batch_size <= 0 or encode_chunk_size <= 0:
            raise ValueError("Embedding batch and chunk sizes must be positive")
        if ef_construction <= 0 or ef_search <= 0 or m <= 0:
            raise ValueError("HNSW parameters must be positive")
        self.ids = [item["item_id"] for item in items]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Duplicate corpus item_id")
        self.items = items
        self.model_name = model_name
        self.batch_size = batch_size
        self.encode_chunk_size = encode_chunk_size
        self.ef_search = ef_search
        self.index = None

        hnswlib, sentence_transformer = _load_dependencies()
        model_kwargs = {"device": device} if device else {}
        self.model = sentence_transformer(model_name, **model_kwargs)
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError(f"Модель {model_name} не сообщила размер эмбеддинга")
        self.dimension = int(dimension)

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = _fingerprint(items, model_name)
        index_path = cache_dir / f"{fingerprint}.bin"
        metadata_path = cache_dir / f"{fingerprint}.json"

        self.index = hnswlib.Index(space="cosine", dim=self.dimension)
        if index_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata == self._metadata():
                self.index.load_index(str(index_path), max_elements=len(items))
                self.index.set_ef(max(self.ef_search, 50))
                return

        self.index.init_index(
            max_elements=max(1, len(items)),
            ef_construction=ef_construction,
            M=m,
            random_seed=42,
        )
        for start in range(0, len(items), encode_chunk_size):
            stop = min(start + encode_chunk_size, len(items))
            texts = [f"passage: {item_text(item)}" for item in items[start:stop]]
            vectors = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32, copy=False)
            self.index.add_items(vectors, np.arange(start, stop))
            print(f"Эмбеддинги объявлений: {stop}/{len(items)}", flush=True)
        self.index.set_ef(max(self.ef_search, 50))

        temporary_index = index_path.with_name(index_path.name + ".tmp")
        self.index.save_index(str(temporary_index))
        os.replace(temporary_index, index_path)
        temporary_metadata = metadata_path.with_name(metadata_path.name + ".tmp")
        temporary_metadata.write_text(
            json.dumps(self._metadata(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary_metadata, metadata_path)

    def _metadata(self):
        return {
            "version": _CACHE_VERSION,
            "model": self.model_name,
            "dimension": self.dimension,
            "items": len(self.ids),
        }

    def retrieve(self, query, limit=50):
        if limit < 0:
            raise ValueError("limit must be non-negative")
        text = normalize(query.get("search_query", ""))
        if not text or limit == 0 or not self.ids:
            return []
        vector = self.model.encode(
            [f"query: {text}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
        count = min(limit, len(self.ids))
        self.index.set_ef(max(self.ef_search, count))
        labels, _distances = self.index.knn_query(vector, k=count)
        return [self.ids[int(label)] for label in labels[0]]
