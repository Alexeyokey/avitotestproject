"""BM25 по заголовкам и параметрам услуг с разреженным индексом слов."""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from .core import normalize


class Baseline:
    def __init__(self, items, k1=1.5, b=0.75):
        if not np.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be positive and finite")
        if not np.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.ids = [x["item_id"] for x in items]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Duplicate corpus item_id")
        # Заголовок и параметры образуют один документ. Длина считается в словах.
        texts = [normalize(x.get("item_title_raw", "") + " " +
                           x.get("item_infm_params_text", "")) for x in items]
        self.vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
        self.index = None
        if not texts or not any(self.vectorizer.build_analyzer()(text) for text in texts):
            return
        counts = self.vectorizer.fit_transform(texts).tocsr()
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        average_length = lengths.mean()

        # BM25 ограничивает вклад повторов слова и учитывает длину объявления.
        # IDF положительный даже у слов, встречающихся в большинстве документов.
        document_frequency = np.bincount(counts.indices, minlength=counts.shape[1])
        idf = np.log1p((len(items) - document_frequency + 0.5) / (document_frequency + 0.5))
        length_penalty = k1 * (1 - b + b * lengths / average_length)
        row_penalty = np.repeat(length_penalty, np.diff(counts.indptr))
        counts.data = (idf[counts.indices] * counts.data * (k1 + 1)
                       / (counts.data + row_penalty)).astype(np.float32)
        # В CSC каждый столбец — список объявлений с конкретным словом.
        # При поиске читаем только списки слов запроса, а не всю матрицу.
        self.index = counts.tocsc()

    def retrieve(self, query, limit=50):
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0 or self.index is None:
            return []
        vector = self.vectorizer.transform([normalize(query.get("search_query", ""))])
        scores = np.zeros(len(self.ids), dtype=np.float32)
        matched = []
        # Каждое слово запроса учитываем один раз, независимо от его повторов.
        for term in sorted(vector.indices):
            start, end = self.index.indptr[term:term + 2]
            rows = self.index.indices[start:end]
            scores[rows] += self.index.data[start:end]
            matched.append(rows)
        if not matched:
            return []
        candidates = np.unique(np.concatenate(matched))
        order = np.lexsort((candidates, -scores[candidates]))[:limit]
        # Если совпадений меньше 50, возвращаем только найденные объявления.
        return [self.ids[candidates[i]] for i in order]
