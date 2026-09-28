"""Структура данных и воспроизводимые разбиения без модели и сети."""
import hashlib
import json
import re

SEARCH_FIELDS = ("search_query", "search_location_id", "search_is_delivery_search",
                 "search_infm_params_text", "search_category")


def normalize(text):
    return " ".join(str(text).lower().replace("ё", "е").split())


def query_key(row):
    # Идентификаторы остаются строками: близость их числовых значений не имеет смысла.
    return tuple(normalize(row.get(field, "")) for field in SEARCH_FIELDS)


def held_out(key, fraction=0.2, seed=42):
    """Сохраняем одинаковое разбиение между процессами и версиями Python."""
    value = json.dumps([seed, key], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def split_rows(rows, mode="pairs", fraction=0.2, seed=42):
    if mode not in {"pairs", "queries", "all"} or (mode != "all" and not 0 < fraction < 1):
        raise ValueError("Invalid split mode or fraction")
    if mode == "all":
        # Поиск без обучения можно проверить на каждой известной положительной паре.
        return [], list(rows)
    train, valid = [], []
    for row in rows:
        # Повторные положительные пары остаются вместе. Разбиение по тексту
        # также скрывает все локации и фильтры одного нормализованного запроса.
        key = (query_key(row), row["item_id"]) if mode == "pairs" else normalize(row["search_query"])
        (valid if held_out(key, fraction, seed) else train).append(row)
    return train, valid


def recall_at_k(predictions, relevant, k=50):
    if set(predictions) != set(relevant) or not relevant:
        raise ValueError("Prediction and relevance query sets must match and be nonempty")
    scores = []
    for key, truth in relevant.items():
        truth = set(truth)
        if not truth:
            raise ValueError("Empty relevance set")
        scores.append(len(set(predictions[key][:k]) & truth) / len(truth))
    return sum(scores) / len(scores)


def validate_answers(rows, expected_queries, corpus_ids):
    expected_queries = list(expected_queries)
    expected = set(expected_queries)
    corpus = set(corpus_ids)
    if len(expected) != len(expected_queries):
        raise ValueError("Duplicate benchmark query_id")
    seen = set()
    for row in rows:
        if set(row) != {"query_id", "answer"}:
            raise ValueError("Expected exactly query_id,answer columns")
        qid, answer = row["query_id"], row["answer"]
        if not isinstance(qid, str) or len(qid) != 16 or qid not in expected or qid in seen:
            raise ValueError(f"Invalid or duplicate query_id: {qid}")
        if not isinstance(answer, str):
            raise ValueError("answer must be a string")
        ids = answer.split(" ") if answer else []
        if len(ids) > 50 or len(ids) != len(set(ids)):
            raise ValueError(f"Too many or duplicate candidates: {qid}")
        if any(not re.fullmatch(r"[0-9a-f]{16}", x) or x not in corpus for x in ids):
            raise ValueError(f"Invalid item_id: {qid}")
        seen.add(qid)
    if seen != expected:
        raise ValueError("Missing benchmark queries")
