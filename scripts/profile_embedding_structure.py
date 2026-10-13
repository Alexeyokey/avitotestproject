"""Profile token allocation without loading encoder weights or using relevance.

Rows are sampled solely by salted hashes of item IDs / normalized query keys.
The tokenizer and prompts come from the saved checkpoint, and offsets show
which *content* fields survive the existing 128-token window. Alternative text
layouts here are diagnostics, not a claim that their retrieval is better.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path

QUERY_FIELDS = ("search_query", "search_location_id", "search_is_delivery_search",
                "search_infm_params_text", "search_category")
ITEM_FIELDS = ("item_id", "item_title_raw", "item_category_id",
               "item_infm_params_text", "item_description_raw")


def normalize(value):
    return " ".join(str(value if value is not None else "").lower().replace("ё", "е").split())


def add_fields(fields, prompt=""):
    """Return text and exact content spans; labels have their own token budget."""
    text, spans = prompt, {}
    for field, label, value in fields:
        value = normalize(value)
        if not value and label is None:
            continue
        if text and not text.endswith(" "):
            text += " "
        if label:
            text += label + " "
        start = len(text)
        text += value
        spans[field] = (start, len(text))
    return text, spans


def item_layout(row, variant, prompt=" ", params_words=40, description_words=48):
    title = ("title", "заголовок:", row.get("item_title_raw", ""))
    category = ("category", "категория:", row.get("item_category_id", ""))
    params = ("params", "параметры:", " ".join(normalize(row.get("item_infm_params_text", "")).split()[:params_words]))
    description = ("description", "описание:", " ".join(normalize(row.get("item_description_raw", "")).split()[:description_words]))
    if variant == "current":
        return add_fields([title, category, params, description], prompt)
    if variant == "description_before_params":
        return add_fields([title, category, description, params], prompt)
    if variant == "no_params":
        return add_fields([title, category, description], prompt)
    if variant == "title_description_400chars":
        return add_fields([("title", None, str(row.get("item_title_raw", "") or "") + "."),
                           ("description", None, str(row.get("item_description_raw", "") or "")[:400])], prompt)
    if variant == "compact":
        from avito_candidates.embedding_formats import extract_filter_values
        title = normalize(row.get("item_title_raw", ""))
        values = " ".join(extract_filter_values(row.get("item_infm_params_text", "")).split()[:params_words])
        desc = " ".join(normalize(row.get("item_description_raw", "")).split()[:description_words])
        nonempty = [("title", title), ("params", values), ("description", desc)]
        nonempty = [(field, value) for field, value in nonempty if value]
        text, spans = prompt, {}
        for i, (field, value) in enumerate(nonempty):
            if i:
                text += ". "
            start = len(text)
            text += value
            spans[field] = (start, len(text))
        return text, spans
    raise ValueError(f"Unknown item variant: {variant}")


def query_layout(row, variant, prompt):
    text = normalize(row.get("search_query", ""))
    if variant == "plain":
        return prompt + text, {"query": (len(prompt), len(prompt) + len(text))}
    if variant == "values":
        from avito_candidates.embedding_formats import extract_filter_values, query_format
        values = extract_filter_values(row.get("search_infm_params_text", ""))
        rendered = query_format(row, "values")
        spans = {"query": (len(prompt), len(prompt) + len(text))}
        if values:
            start = len(prompt) + len(text) + (3 if text else 0)
            spans["filters"] = (start, len(prompt) + len(rendered))
        return prompt + rendered, spans
    if variant not in {"current", "no_category", "no_filters"}:
        raise ValueError(f"Unknown query variant: {variant}")
    fields = [("query", "запрос:", text)]
    filters = normalize(row.get("search_infm_params_text", ""))
    if filters and variant != "no_filters":
        fields.append(("filters", "фильтры:", filters))
    category = normalize(row.get("search_category", ""))
    if category and category != "0" and variant != "no_category":
        fields.append(("category", "категория:", category))
    # Octen's Query: prompt deliberately has no trailing space. Training and
    # inference concatenate it directly with the first content string.
    rendered, spans = add_fields(fields)
    return prompt + rendered, {k: (a + len(prompt), b + len(prompt)) for k, (a, b) in spans.items()}


class HashSample:
    """Bounded, input-order-independent sample; duplicate keys count once."""
    def __init__(self, limit, salt):
        if limit <= 0:
            raise ValueError("Sample size must be positive")
        self.limit, self.salt = limit, salt
        self.heap, self.rows = [], {}

    def offer(self, key, row):
        if key in self.rows:
            return
        score = int.from_bytes(hashlib.sha256((self.salt + "\0" + key).encode()).digest(), "big")
        if len(self.rows) >= self.limit and score >= -self.heap[0][0]:
            return
        if len(self.rows) >= self.limit:
            _, old_key = heapq.heappop(self.heap)
            del self.rows[old_key]
        heapq.heappush(self.heap, (-score, key))
        self.rows[key] = dict(row)

    def result(self):
        return [self.rows[key] for key in sorted(self.rows)]


def token_profile(encoding, spans, max_length, retained_encoding=None):
    """Tokens crossing a boundary count as present in either content field."""
    tokens = [offset for offset in encoding.offsets if offset[1] > offset[0]]
    # Special tokens, if the tokenizer inserts any, occupy the window too.
    # Tokenizers truncate the sequence *before* post-processing. A final EOS
    # therefore survives, and merely slicing a full encoding overcounts one
    # content token when the full text exceeds the window.
    retained_offsets = (retained_encoding.offsets if retained_encoding is not None
                        else encoding.offsets[:max_length])
    result = {"tokens": len(encoding.ids), "truncated": len(encoding.ids) > max_length}
    for field, (start, end) in spans.items():
        full = sum(a < end and b > start for a, b in tokens) if end > start else 0
        kept = sum(a < end and b > start for a, b in retained_offsets) if end > start else 0
        result[field] = {"full": full, "kept": kept,
                         "fraction": kept / full if full else None}
    return result


def aggregate(profiles):
    import numpy as np
    if not profiles:
        return {"rows": 0}
    fields = sorted(set().union(*(set(p) - {"tokens", "truncated"} for p in profiles)))
    result = {"rows": len(profiles),
              "truncated_fraction": sum(p["truncated"] for p in profiles) / len(profiles),
              "token_length_quantiles": dict(zip(["p10", "p50", "p90", "p99"],
                                                 map(float, np.quantile([p["tokens"] for p in profiles], [.1, .5, .9, .99])))),
              "fields": {}}
    for field in fields:
        present = [p[field] for p in profiles if field in p and p[field]["full"]]
        result["fields"][field] = {
            "content_nonempty_rows": len(present),
            "mean_full_content_tokens": float(np.mean([p["full"] for p in present])) if present else 0,
            "mean_retained_content_tokens": float(np.mean([p["kept"] for p in present])) if present else 0,
            "mean_retained_fraction": float(np.mean([p["fraction"] for p in present])) if present else None,
            "completely_lost_fraction": sum(p["kept"] == 0 for p in present) / len(present) if present else None,
        }
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_texts(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    texts = document.get("texts") if isinstance(document, dict) else document
    if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
        raise ValueError("Text manifest must contain a list of strings")
    return set(texts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--fit-texts", type=Path, required=True)
    parser.add_argument("--tune-texts", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_length <= 0:
        parser.error("max-length must be positive")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing profile")
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    tokenizer_path = args.model_dir / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    window_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    window_tokenizer.enable_truncation(max_length=args.max_length, direction="right")
    window_tokenizer.no_padding()
    config = json.loads((args.model_dir / "config_sentence_transformers.json").read_text())
    prompts = config["prompts"]
    item_sample = HashSample(args.sample_size, "embedding-structure-items-v1")
    for batch in pq.ParquetFile(args.catalog).iter_batches(batch_size=1024, columns=list(ITEM_FIELDS)):
        for row in batch.to_pylist():
            item_sample.offer(str(row["item_id"]), row)
    text_sets = {"fit": load_texts(args.fit_texts), "tune": load_texts(args.tune_texts)}
    if text_sets["fit"] & text_sets["tune"]:
        raise ValueError("Fit and tune text sets overlap")
    samples = {phase: HashSample(args.sample_size, f"embedding-structure-{phase}-queries-v1") for phase in text_sets}
    for batch in pq.ParquetFile(args.train).iter_batches(batch_size=1024, columns=list(QUERY_FIELDS)):
        for row in batch.to_pylist():
            normalized_text = normalize(row["search_query"])
            for phase, allowed in text_sets.items():
                if normalized_text in allowed:
                    key = json.dumps([normalize(row[f]) for f in QUERY_FIELDS], ensure_ascii=False, separators=(",", ":"))
                    samples[phase].offer(key, row)
    results = {"profile_version": 2,
               "policy": "Metadata-only salted-hash sampling; no relevance IDs or evaluation scores used",
               "truncation": "Actual tokenizer truncation preserves post-processor special tokens",
               "max_length": args.max_length, "model_dir": str(args.model_dir), "prompts": prompts,
               "tokenizer_sha256": sha256(tokenizer_path), "inputs_sha256": {
                   name: sha256(getattr(args, name)) for name in ("catalog", "train", "fit_texts", "tune_texts")},
               "prompt_tokens": {key: len(tokenizer.encode(value).ids) for key, value in prompts.items()},
               "items": {}, "queries": {}}
    items = item_sample.result()
    for variant in ("current", "compact", "description_before_params", "no_params", "title_description_400chars"):
        profiles = []
        for row in items:
            text, spans = item_layout(row, variant, prompts["document"])
            profiles.append(token_profile(tokenizer.encode(text), spans, args.max_length, window_tokenizer.encode(text)))
        results["items"][variant] = aggregate(profiles)
    for phase, sample in samples.items():
        queries = sample.result()
        results["queries"][phase] = {}
        for variant in ("current", "plain", "values", "no_category", "no_filters"):
            profiles = []
            for row in queries:
                text, spans = query_layout(row, variant, prompts["query"])
                profiles.append(token_profile(tokenizer.encode(text), spans, args.max_length, window_tokenizer.encode(text)))
            results["queries"][phase][variant] = aggregate(profiles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "item_rows": len(items),
                      "query_rows": {phase: len(sample.rows) for phase, sample in samples.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
