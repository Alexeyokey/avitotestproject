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


# The token-count prior above remains unchanged for existing feature schemas.
# This independent channel tests whether a supervised word/character model can
# recover service types that the global description and dense channels miss.
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier


SOFT_MICROCATEGORY_SCHEMA = "soft-microcategory-word-char-values-v1"


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_digest(texts):
    return hashlib.sha256(_canonical(sorted(texts)).encode("utf-8")).hexdigest()


def _filter_atoms(value, prefix=""):
    """Use parsed values and their paths; JSON braces are not lexical features."""
    if isinstance(value, dict):
        output = []
        for key in sorted(value, key=str):
            path = normalize(key).replace(" ", "_")
            output.extend(_filter_atoms(value[key], f"{prefix}_{path}" if prefix else path))
        return output
    if isinstance(value, (list, tuple)):
        # Filter alternatives are sets: their serialization order must not
        # create a different classifier context or training weight.
        return sorted({atom for element in value for atom in _filter_atoms(element, prefix)})
    if value is None:
        return []
    atom = "true" if value is True else "false" if value is False else normalize(value)
    return [f"filter_{prefix} {atom}"] if atom else []


def microcategory_query_text(query):
    """Query + typed filter values + known macrocategory; never item labels/IDs."""
    text = normalize(query.get("search_query", "") or "")
    raw = query.get("search_infm_params_text", "")
    if isinstance(raw, (dict, list)):
        filters = raw
    else:
        try:
            filters = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            filters = str(raw or "")
    atoms = _filter_atoms(filters)
    category = normalize(query.get("search_category", "") or "")
    if category == "0":
        category = ""
    return " | ".join([text, *atoms, *([f"category_{category}"] if category else [])])


def _word_stems(value):
    # Deferred import preserves the legacy prior's import graph. Character
    # n-grams use the original spelling, so typo and suffix signals remain.
    from .baseline import stem_text
    return stem_text(value)


class _TextModel:
    def __init__(self, contexts, *, config, seed):
        self.fitted_texts = frozenset(contexts)
        documents, labels, weights = [], [], []
        for text, feature_contexts in sorted(contexts.items()):
            for document, categories in sorted(feature_contexts.items()):
                for category in sorted(categories):
                    documents.append(document)
                    labels.append(category)
                    # A normalized text contributes exactly one unit, divided
                    # between its contexts and distinct observed categories.
                    weights.append(1 / len(feature_contexts) / len(categories))
        self.training_examples = len(labels)
        if not documents:
            raise ValueError("No allowed fit examples for microcategory classifier")
        self.word = TfidfVectorizer(
            ngram_range=(1, 2), min_df=min(config["word_min_df"], len(documents)),
            max_features=config["word_max_features"], sublinear_tf=True,
            dtype=np.float32, preprocessor=_word_stems, lowercase=False,
            token_pattern=r"(?u)\b\w+\b")
        self.char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            min_df=min(config["char_min_df"], len(documents)), max_features=config["char_max_features"],
            sublinear_tf=True, dtype=np.float32, lowercase=False)
        parts = []
        for vectorizer in (self.word, self.char):
            try:
                parts.append(vectorizer.fit_transform(documents))
            except ValueError as error:
                if "empty vocabulary" not in str(error) and "no terms remain" not in str(error).lower():
                    raise
                # Small cross-fit fixtures or rare contexts still need a
                # defined vocabulary. This fallback is recorded in the model.
                vectorizer.set_params(min_df=1)
                parts.append(vectorizer.fit_transform(documents))
        matrix = sparse.hstack(parts, format="csr", dtype=np.float32)
        classes = sorted(set(labels))
        coefficient_estimate = len(classes) * matrix.shape[1] * 4
        if coefficient_estimate > config["max_model_bytes"]:
            raise ValueError("Classifier coefficient budget exceeded; reduce max_features")
        self.classes = np.asarray(classes)
        self.classifier = None
        if len(classes) > 1:
            self.classifier = SGDClassifier(
                loss="log_loss", alpha=config["alpha"], max_iter=config["max_iter"],
                tol=None, n_jobs=1, random_state=seed)
            self.classifier.fit(matrix, np.asarray(labels),
                                sample_weight=np.asarray(weights, dtype=np.float32))
            self.classes = self.classifier.classes_
        self.feature_count = matrix.shape[1]
        self.coefficient_bytes = (self.classifier.coef_.nbytes
                                  if self.classifier is not None else 0)
        self.total_weight = float(sum(weights))

    def probabilities(self, documents):
        if self.classifier is None:
            return np.ones((len(documents), 1), dtype=np.float32)
        matrix = sparse.hstack((self.word.transform(documents), self.char.transform(documents)),
                               format="csr", dtype=np.float32)
        return self.classifier.predict_proba(matrix)


class SoftMicrocategoryClassifier:
    """Fit-only classifier with optional text-grouped cross-fitting.

    Probabilities are soft model scores, not calibrated relevance probabilities.
    Predicting a fitted text is forbidden unless its held-out-fold model exists.
    """
    def __init__(self, model, crossfit_models, metadata):
        self.model = model
        self.crossfit_models = crossfit_models
        self.metadata = metadata
        self._prediction_cache = {}

    @staticmethod
    def fold(text, folds, seed):
        value = _canonical([seed, normalize(text)])
        return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big") % folds

    @classmethod
    def fit_rows(cls, rows, *, allowed_fit_texts, corpus_fingerprint,
                 source_sha256, crossfit_folds=0, seed=42,
                 word_max_features=80000, char_max_features=120000,
                 word_min_df=2, char_min_df=3, alpha=2e-6, max_iter=15,
                 max_model_bytes=256 * 1024**2, max_total_model_bytes=768 * 1024**2):
        allowed = frozenset(normalize(text) for text in allowed_fit_texts if normalize(text))
        if not allowed:
            raise ValueError("Explicit allowed fit texts must be nonempty")
        if not re.fullmatch(r"[0-9a-f]{64}", corpus_fingerprint):
            raise ValueError("Corpus fingerprint must be a SHA256 digest")
        if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
            raise ValueError("Training source fingerprint must be a SHA256 digest")
        if crossfit_folds != 0 and crossfit_folds < 2:
            raise ValueError("Cross-fitting requires at least two folds")
        if (word_max_features < 1 or char_max_features < 1 or word_min_df < 1
                or char_min_df < 1 or max_iter < 1 or not np.isfinite(alpha) or alpha <= 0
                or max_model_bytes < 1 or max_total_model_bytes < 1):
            raise ValueError("Invalid microcategory classifier settings")
        contexts = defaultdict(lambda: defaultdict(set))
        accepted_rows = excluded_rows = 0
        for row in rows:
            text = normalize(row.get("search_query", "") or "")
            if text not in allowed:
                excluded_rows += 1
                continue
            category = str(row.get("item_microcat_id", "") or "")
            if not category or category == "0":
                continue
            contexts[text][microcategory_query_text(row)].add(category)
            accepted_rows += 1
        config = dict(word_max_features=word_max_features, char_max_features=char_max_features,
                      word_min_df=word_min_df, char_min_df=char_min_df, alpha=alpha,
                      max_iter=max_iter, max_model_bytes=max_model_bytes)
        model = _TextModel(contexts, config=config, seed=seed)
        crossfit = {}
        memory = model.coefficient_bytes
        if crossfit_folds:
            assignments = {text: cls.fold(text, crossfit_folds, seed) for text in contexts}
            for fold in range(crossfit_folds):
                fit_contexts = {text: values for text, values in contexts.items()
                                if assignments[text] != fold}
                if not fit_contexts:
                    raise ValueError("Cross-fit fold has no training texts")
                crossfit[fold] = _TextModel(fit_contexts, config=config, seed=seed + fold + 1)
                memory += crossfit[fold].coefficient_bytes
                if memory > max_total_model_bytes:
                    raise ValueError("Cross-fit classifier memory budget exceeded")
        metadata = {
            "schema": SOFT_MICROCATEGORY_SCHEMA, "source_sha256": source_sha256,
            "corpus_fingerprint": corpus_fingerprint,
            "allowed_fit_texts_sha256": _text_digest(allowed),
            "observed_fit_texts_sha256": _text_digest(contexts),
            "allowed_fit_texts_count": len(allowed), "fit_texts_count": len(contexts),
            "accepted_fit_rows": accepted_rows, "excluded_nonfit_rows": excluded_rows,
            "training_examples": model.training_examples,
            "training_contexts": sum(len(value) for value in contexts.values()),
            "text_balanced_total_weight": model.total_weight,
            "classes": model.classes.tolist(), "feature_count": model.feature_count,
            "coefficient_bytes_total": memory, "crossfit_folds": crossfit_folds,
            "seed": seed, "config": config,
            "training_policy": "explicit-fit-texts; one-weight-per-text; grouped-crossfit-required",
        }
        return cls(model, crossfit, metadata)

    def predict_categories_batch(self, queries, *, top_k=3, exclude_self=False):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        queries = list(queries)
        output = [None] * len(queries)
        grouped = defaultdict(list)
        for index, query in enumerate(queries):
            text = normalize(query.get("search_query", "") or "")
            fitted = text in self.model.fitted_texts
            if fitted:
                if not exclude_self or not self.crossfit_models:
                    raise ValueError("Fitted query text requires a held-out cross-fit model")
                fold = self.fold(text, self.metadata["crossfit_folds"], self.metadata["seed"])
                if text in self.crossfit_models[fold].fitted_texts:
                    raise ValueError("Cross-fit classifier contains the query's own text")
            else:
                fold = None
            document = microcategory_query_text(query)
            key = (fold, document)
            model = self.model if fold is None else self.crossfit_models[fold]
            needed = min(top_k, len(model.classes))
            if key in self._prediction_cache and len(self._prediction_cache[key]) >= needed:
                output[index] = self._prediction_cache[key][:needed]
            else:
                grouped[fold].append((index, document, key))
        for fold, entries in grouped.items():
            model = self.model if fold is None else self.crossfit_models[fold]
            probabilities = model.probabilities([entry[1] for entry in entries])
            for (index, _document, key), scores in zip(entries, probabilities):
                order = sorted(range(len(scores)),
                               key=lambda i: (-float(scores[i]), str(model.classes[i])))[:top_k]
                result = tuple((str(model.classes[i]), float(scores[i])) for i in order)
                output[index] = result
                # Bound per-process cache independently of the number of runs.
                if len(self._prediction_cache) < 100000 or key in self._prediction_cache:
                    self._prediction_cache[key] = result
        return output

    def predict_categories(self, query, *, top_k=3, exclude_self=False):
        return self.predict_categories_batch([query], top_k=top_k,
                                             exclude_self=exclude_self)[0]

    def save(self, path):
        path = Path(path)
        manifest = path.with_suffix(path.suffix + ".json")
        if path.exists() or manifest.exists():
            raise ValueError("Refusing to overwrite microcategory classifier")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Prediction caches are disposable and must not inflate saved models.
        payload = {"model": self.model, "crossfit_models": self.crossfit_models,
                   "metadata": self.metadata}
        joblib.dump(payload, path, compress=3)
        manifest.write_text(_canonical({"model_sha256": _sha_file(path),
                                        "metadata": self.metadata}) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path, *, allowed_fit_texts, corpus_fingerprint, source_sha256):
        path = Path(path)
        manifest = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
        if _sha_file(path) != manifest["model_sha256"]:
            raise ValueError("Microcategory classifier bytes changed")
        metadata = manifest["metadata"]
        allowed = frozenset(normalize(text) for text in allowed_fit_texts if normalize(text))
        expected = {"schema": SOFT_MICROCATEGORY_SCHEMA,
                    "allowed_fit_texts_sha256": _text_digest(allowed),
                    "corpus_fingerprint": corpus_fingerprint,
                    "source_sha256": source_sha256}
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Microcategory classifier provenance mismatch")
        payload = joblib.load(path)
        if payload["metadata"] != metadata:
            raise ValueError("Microcategory classifier metadata changed")
        if _text_digest(payload["model"].fitted_texts) != metadata["observed_fit_texts_sha256"]:
            raise ValueError("Microcategory classifier training text inventory changed")
        return cls(payload["model"], payload["crossfit_models"], metadata)


def fit_soft_microcategory_classifier(path, *, allowed_fit_texts,
                                     corpus_fingerprint, batch_size=32768, **settings):
    """Fit from an explicit normalized-text allowlist, never from tune/audit labels."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    parquet = pq.ParquetFile(path)
    fields = ["search_query", "search_infm_params_text", "search_category", "item_microcat_id"]
    def rows():
        for batch in parquet.iter_batches(batch_size=batch_size, columns=fields):
            yield from batch.to_pylist()
    source_sha256 = _sha_file(path)
    result = SoftMicrocategoryClassifier.fit_rows(rows(), allowed_fit_texts=allowed_fit_texts,
        corpus_fingerprint=corpus_fingerprint, source_sha256=source_sha256, **settings)
    if _sha_file(path) != source_sha256:
        raise ValueError("Training source changed during microcategory fit")
    return result


class MicrocategoryChannel:
    """Add candidates from several predicted service types without deleting any base ID.

    BM25 fields are scored once per query; microcategory masks are applied before
    top-K. This is an independent candidate route, not a filter on the base pool.
    """
    def __init__(self, classifier, items, *, indexes, field_weights=None,
                 top_categories=3, min_per_category=5, probability_power=.5,
                 location_bonus=0.0, max_pool_size=998):
        from .reranker import corpus_fingerprint
        fingerprint = corpus_fingerprint(items, feature_version=4)
        if classifier.metadata["corpus_fingerprint"] != fingerprint:
            raise ValueError("Microcategory channel corpus differs from the fitted artifact")
        self.classifier = classifier
        self.ids = [item["item_id"] for item in items]
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Duplicate corpus item_id")
        self.categories = np.asarray([str(item.get("item_microcat_id", "") or "") for item in items])
        self.locations = np.asarray([str(item.get("item_location_id", "") or "") for item in items])
        self._id_rows = {item_id: row for row, item_id in enumerate(self.ids)}
        category_rows = defaultdict(list)
        for row, category in enumerate(self.categories):
            category_rows[category].append(row)
        self._category_rows = {category: np.asarray(rows, dtype=np.int64)
                               for category, rows in category_rows.items()}
        self.indexes = dict(indexes)
        self.field_weights = dict(field_weights or {name: 1.0 for name in indexes})
        if set(self.indexes) != set(self.field_weights) or not self.indexes:
            raise ValueError("Every microcategory lexical field requires a weight")
        if any(index.ids != self.ids for index in self.indexes.values()):
            raise ValueError("Microcategory index order differs from corpus order")
        if any(not np.isfinite(weight) or weight < 0 for weight in self.field_weights.values()):
            raise ValueError("Lexical weights must be finite and nonnegative")
        if (sum(self.field_weights.values()) <= 0 or top_categories < 1 or min_per_category < 0
                or probability_power < 0 or not np.isfinite(probability_power)
                or location_bonus < 0 or not np.isfinite(location_bonus)
                or not 1 <= max_pool_size < 1000):
            raise ValueError("Invalid microcategory candidate settings")
        self.top_categories = top_categories
        self.min_per_category = min_per_category
        self.probability_power = probability_power
        self.location_bonus = location_bonus
        self.max_pool_size = max_pool_size

    def _scores(self, query):
        from .embedding_formats import extract_filter_values
        # Share the lexical query with the frozen candidate experiment. The
        # classifier's fitted document formatter remains separate and unchanged.
        values = extract_filter_values(query.get("search_infm_params_text", ""))
        lexical_text = (str(query.get("search_query", "") or "") + " " + values).strip()
        output = {}
        for name, index in self.indexes.items():
            scores, _matched = index._score(lexical_text)
            output[name] = (np.zeros(len(self.ids), dtype=np.float32) if scores is None else scores)
        return output

    def ranked_lists_with_scores(self, query, limit, *, existing_ids=(), exclude_self=False,
                                 predicted_categories=None, lexical_scores=None):
        if limit < 0:
            raise ValueError("limit must be nonnegative")
        existing = set(existing_ids)
        if len(existing) > self.max_pool_size:
            raise ValueError("Base pool already exceeds the microcategory budget")
        if not existing.issubset(self._id_rows):
            raise ValueError("Base pool contains an unknown corpus ID")
        limit = min(limit, self.max_pool_size - len(existing))
        if predicted_categories is not None:
            # Root batch integration may reuse the classifier's result, but
            # cannot supply arbitrary categories or bypass self-exclusion.
            verified = self.classifier.predict_categories(query,
                top_k=self.top_categories, exclude_self=exclude_self)
            if tuple(predicted_categories) != verified:
                raise ValueError("Supplied microcategories differ from classifier predictions")
            predicted = verified
        else:
            predicted = self.classifier.predict_categories(query,
                top_k=self.top_categories, exclude_self=exclude_self)
        diagnostics = {"predicted_categories": predicted, "base_pool_size": len(existing),
                       "free_slots": self.max_pool_size - len(existing)}
        if not limit:
            return [], {}, diagnostics
        raw_fields = self._scores(query) if lexical_scores is None else dict(lexical_scores)
        if (set(raw_fields) != set(self.indexes) or any(
                np.asarray(scores).shape != (len(self.ids),)
                or not np.all(np.isfinite(scores)) or np.any(scores < 0)
                for scores in raw_fields.values())):
            raise ValueError("Precomputed lexical scores must match all indexed fields")
        combined = np.zeros(len(self.ids), dtype=np.float32)
        for name, scores in raw_fields.items():
            peak = float(scores.max(initial=0))
            if peak > 0:
                combined += self.field_weights[name] * scores / peak
        raw_fields["combined"] = combined
        probabilities = {category: probability for category, probability in predicted}
        rows_by_category = {category: self._category_rows.get(category, np.asarray([], dtype=np.int64))
                            for category in probabilities}
        available = np.ones(len(self.ids), dtype=bool)
        available[[self._id_rows[item_id] for item_id in existing]] = False
        channels, channel_scores = [], {}
        location = str(query.get("search_location_id", "") or "")
        for name, lexical_scores in raw_fields.items():
            score = np.zeros(len(self.ids), dtype=np.float32)
            ranked_categories = []
            for category, probability in predicted:
                rows = rows_by_category[category]
                rows = rows[available[rows] & (lexical_scores[rows] > 0)]
                score[rows] = lexical_scores[rows] * probability ** self.probability_power
                if location and self.location_bonus:
                    score[rows] += self.location_bonus * (self.locations[rows] == location)
                ranked_categories.append(rows[np.lexsort((rows, -score[rows]))])
            # Guarantee a small share from each predicted class, then fill by
            # soft scores. A wrong highest class cannot monopolize the route.
            guaranteed = []
            for rank in range(self.min_per_category):
                for rows in ranked_categories:
                    if rank < len(rows) and len(guaranteed) < limit:
                        guaranteed.append(int(rows[rank]))
            all_rows = np.concatenate(ranked_categories) if ranked_categories else np.asarray([], dtype=int)
            ordered = all_rows[np.lexsort((all_rows, -score[all_rows]))]
            selected = list(dict.fromkeys([*guaranteed, *ordered.tolist()]))[:limit]
            selected.sort(key=lambda row: (-float(score[row]), row))
            channel_name = f"micro_{name}"
            ids = [self.ids[row] for row in selected]
            channels.append((channel_name, ids, 0.0))
            channel_scores[channel_name] = {self.ids[row]: float(score[row]) for row in selected}
        # Each route is an alternative experiment. Combining their unions
        # requires another explicit allocation; returning them is not merging.
        diagnostics["channel_sizes"] = {name: len(ids) for name, ids, _weight in channels}
        return channels, channel_scores, diagnostics

    def retrieve(self, query, limit, *, existing_ids=(), exclude_self=False,
                 predicted_categories=None, field="combined", lexical_scores=None):
        channels, scores, diagnostics = self.ranked_lists_with_scores(query, limit,
            existing_ids=existing_ids, exclude_self=exclude_self,
            predicted_categories=predicted_categories, lexical_scores=lexical_scores)
        name = f"micro_{field}"
        if not channels:
            return [], {}, diagnostics
        for channel_name, ids, _weight in channels:
            if channel_name == name:
                return ids, scores[name], diagnostics
        raise ValueError("Unknown microcategory lexical field")
