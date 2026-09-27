"""Семантический поиск по эмбеддингам с локальным HNSW-индексом."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .core import normalize


DEFAULT_EMBEDDING_MODEL = "Octen/Octen-Embedding-0.6B"
_CACHE_VERSION = 5


def encoding_input(texts, model_name, role):
    """Apply the model's retrieval prompts without mixing Octen and E5 formats.

    Octen publishes `query` and `document` prompts in its Sentence Transformers
    configuration. E5 instead expects literal `query:`/`passage:` prefixes.
    Other models receive plain text unless their encoding is configured here.
    """
    if role not in {"query", "document"}:
        raise ValueError("Embedding role must be query or document")
    if model_name == DEFAULT_EMBEDDING_MODEL:
        return list(texts), {"prompt_name": role}
    if model_name.startswith("intfloat/multilingual-e5"):
        prefix = "query: " if role == "query" else "passage: "
        return [prefix + text for text in texts], {}
    return list(texts), {}


def _first_words(text, limit):
    if limit < 0:
        raise ValueError("Text field word limits must be non-negative")
    return " ".join(normalize(text).split()[:limit])


def item_text(item, *, params_words=40, description_words=48) -> str:
    title = normalize(item.get("item_title_raw", ""))
    category = normalize(item.get("item_category_id", ""))
    params = _first_words(item.get("item_infm_params_text", ""), params_words)
    description = _first_words(item.get("item_description_raw", ""), description_words)
    return normalize(
        f"заголовок: {title} категория: {category} "
        f"параметры: {params} описание: {description}"
    )


def query_text(query) -> str:
    parts = [f"запрос: {normalize(query.get('search_query', ''))}"]
    filters = normalize(query.get("search_infm_params_text", ""))
    if filters:
        parts.append(f"фильтры: {filters}")
    category = normalize(query.get("search_category", ""))
    if category and category != "0":
        parts.append(f"категория: {category}")
    return " ".join(parts)


def _fingerprint(
    items,
    model_name: str,
    max_seq_length=128,
    params_words=40,
    description_words=48,
) -> str:
    digest = hashlib.sha256(
        f"{_CACHE_VERSION}\0{model_name}\0{max_seq_length}\0"
        f"{params_words}\0{description_words}\0".encode()
    )
    for item in items:
        digest.update(str(item["item_id"]).encode())
        digest.update(b"\0")
        digest.update(item_text(
            item,
            params_words=params_words,
            description_words=description_words,
        ).encode())
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
        batch_size=16,
        encode_chunk_size=4096,
        max_seq_length=128,
        params_words=40,
        description_words=48,
        checkpoint_items=32768,
        device=None,
        ef_construction=200,
        ef_search=300,
        m=32,
    ):
        if (batch_size <= 0 or encode_chunk_size <= 0 or max_seq_length <= 0
                or checkpoint_items <= 0):
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
        self.max_seq_length = max_seq_length
        self.params_words = params_words
        self.description_words = description_words
        self.checkpoint_items = checkpoint_items
        self.ef_search = ef_search
        self.index = None

        hnswlib, sentence_transformer = _load_dependencies()
        model_kwargs = {"device": device} if device else {}
        self.model = sentence_transformer(model_name, **model_kwargs)
        self.model.max_seq_length = max_seq_length
        get_dimension = getattr(self.model, "get_embedding_dimension", None)
        if get_dimension is None:
            get_dimension = self.model.get_sentence_embedding_dimension
        dimension = get_dimension()
        if dimension is None:
            raise ValueError(f"Модель {model_name} не сообщила размер эмбеддинга")
        self.dimension = int(dimension)

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = _fingerprint(
            items,
            model_name,
            max_seq_length,
            params_words,
            description_words,
        )
        index_path = cache_dir / f"{fingerprint}.bin"
        metadata_path = cache_dir / f"{fingerprint}.json"
        partial_index_path = cache_dir / f"{fingerprint}.partial.bin"
        partial_metadata_path = cache_dir / f"{fingerprint}.partial.json"

        self.index = hnswlib.Index(space="cosine", dim=self.dimension)
        if index_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata == self._metadata():
                self.index.load_index(str(index_path), max_elements=len(items))
                self.index.set_ef(max(self.ef_search, 50))
                return

        start_item = 0
        if partial_index_path.exists() and partial_metadata_path.exists():
            partial_metadata = json.loads(partial_metadata_path.read_text(encoding="utf-8"))
            start_item = int(partial_metadata.pop("indexed_items", 0))
            if partial_metadata == self._metadata() and 0 < start_item <= len(items):
                self.index.load_index(str(partial_index_path), max_elements=max(1, len(items)))
                if self.index.get_current_count() != start_item:
                    start_item = 0
                    self.index = hnswlib.Index(space="cosine", dim=self.dimension)
            else:
                start_item = 0
        if start_item == 0:
            self.index.init_index(
                max_elements=max(1, len(items)),
                ef_construction=ef_construction,
                M=m,
                random_seed=42,
            )
        for start in range(start_item, len(items), encode_chunk_size):
            stop = min(start + encode_chunk_size, len(items))
            texts = [
                item_text(
                    item,
                    params_words=self.params_words,
                    description_words=self.description_words,
                )
                for item in items[start:stop]
            ]
            texts, prompt_kwargs = encoding_input(texts, model_name, "document")
            vectors = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                **prompt_kwargs,
            ).astype(np.float32, copy=False)
            self.index.add_items(vectors, np.arange(start, stop))
            print(f"Эмбеддинги объявлений: {stop}/{len(items)}", flush=True)
            if stop < len(items) and (
                stop % checkpoint_items < encode_chunk_size
            ):
                self._save_index(self.index, partial_index_path)
                self._write_metadata(
                    partial_metadata_path,
                    {**self._metadata(), "indexed_items": stop},
                )
        self.index.set_ef(max(self.ef_search, 50))

        self._save_index(self.index, index_path)
        self._write_metadata(metadata_path, self._metadata())
        partial_index_path.unlink(missing_ok=True)
        partial_metadata_path.unlink(missing_ok=True)

    @staticmethod
    def _save_index(index, path):
        temporary = path.with_name(path.name + ".tmp")
        index.save_index(str(temporary))
        os.replace(temporary, path)

    @staticmethod
    def _write_metadata(path, metadata):
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)

    def _metadata(self):
        return {
            "version": _CACHE_VERSION,
            "model": self.model_name,
            "dimension": self.dimension,
            "items": len(self.ids),
            "max_seq_length": self.max_seq_length,
            "params_words": self.params_words,
            "description_words": self.description_words,
        }

    def retrieve(self, query, limit=50):
        if limit < 0:
            raise ValueError("limit must be non-negative")
        search_query = normalize(query.get("search_query", ""))
        if not search_query or limit == 0 or not self.ids:
            return []
        text = query_text(query)
        texts, prompt_kwargs = encoding_input([text], self.model_name, "query")
        vector = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            **prompt_kwargs,
        ).astype(np.float32, copy=False)
        count = min(limit, len(self.ids))
        self.index.set_ef(max(self.ef_search, count))
        labels, _distances = self.index.knn_query(vector, k=count)
        return [self.ids[int(label)] for label in labels[0]]
