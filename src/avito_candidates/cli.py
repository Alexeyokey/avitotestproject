"""Local commands: profile, evaluate, predict, validate. Never call external APIs."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .core import SEARCH_FIELDS, normalize, query_key, recall_at_k, split_rows, validate_answers


def read(path, columns):
    # String conversion avoids accidental numeric feature interpretation.
    frame = pd.read_parquet(path, columns=list(columns))
    return frame.fillna("").astype(str).to_dict("records")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_retriever(args, items):
    from .baseline import Baseline

    bm25_kwargs = {
        "title_k1": args.k1 if args.k1 is not None else args.title_k1,
        "title_b": args.b if args.b is not None else args.title_b,
        "params_k1": args.k1 if args.k1 is not None else args.params_k1,
        "params_b": args.b if args.b is not None else args.params_b,
        "description_k1": args.k1 if args.k1 is not None else args.description_k1,
        "description_b": args.b if args.b is not None else args.description_b,
        "title_weight": args.title_weight,
        "params_weight": args.params_weight,
        "description_weight": args.description_weight,
        "candidate_k": args.candidate_k,
        "rank_constant": args.bm25_rrf_k,
        "channel_quota": args.bm25_channel_quota,
    }
    if args.method == "bm25":
        return Baseline(items, **bm25_kwargs)

    from .dense import DenseRetriever

    dense = DenseRetriever(
        items,
        model_name=args.embedding_model,
        cache_dir=args.dense_cache,
        batch_size=args.embedding_batch_size,
        max_seq_length=args.embedding_max_length,
        device=args.device,
        ef_search=args.ef_search,
    )
    if args.method == "dense":
        return dense

    from .hybrid import HybridRetriever

    return HybridRetriever(
        Baseline(items, **bm25_kwargs),
        dense,
        candidate_k=args.candidate_k,
        rank_constant=args.rrf_k,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        channel_quota=args.channel_quota,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["profile", "evaluate", "predict", "validate"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/result.json"))
    parser.add_argument("--answer", type=Path, default=Path("artifacts/answer.csv"))
    parser.add_argument("--split", choices=["pairs", "queries"], default="pairs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--k1", type=float, help="Override k1 for all BM25 fields")
    parser.add_argument("--b", type=float, help="Override b for all BM25 fields")
    parser.add_argument("--title-k1", type=float, default=1.0)
    parser.add_argument("--title-b", type=float, default=0.2)
    parser.add_argument("--params-k1", type=float, default=1.2)
    parser.add_argument("--params-b", type=float, default=0.7)
    parser.add_argument("--description-k1", type=float, default=1.0)
    parser.add_argument("--description-b", type=float, default=0.8)
    parser.add_argument("--title-weight", type=float, default=2.0)
    parser.add_argument("--params-weight", type=float, default=1.0)
    parser.add_argument("--description-weight", type=float, default=0.5)
    parser.add_argument("--bm25-rrf-k", type=int, default=30)
    parser.add_argument("--bm25-channel-quota", type=int, default=10)
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--embedding-max-length", type=int, default=128)
    parser.add_argument("--device", help="Sentence Transformers device: cpu, mps or cuda")
    parser.add_argument("--dense-cache", type=Path, default=Path("artifacts/dense"))
    parser.add_argument("--ef-search", type=int, default=300)
    parser.add_argument("--candidate-k", type=int, default=300)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--channel-quota", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Evaluation smoke-test limit; 0 evaluates all queries")
    args = parser.parse_args()
    root = args.data_dir
    queries = read(root / "benchmark_queries.parquet", ("query_id", *SEARCH_FIELDS))
    item_columns = (
        "item_id", "item_title_raw", "item_infm_params_text", "item_description_raw"
    )
    items = read(root / "benchmark_items.parquet", item_columns)
    ids = {x["item_id"] for x in items}
    if args.command == "validate":
        with args.answer.open(encoding="utf-8", newline="") as stream:
            validate_answers(list(csv.DictReader(stream)), [q["query_id"] for q in queries], ids)
        print("Submission valid")
        return
    if args.command in {"profile", "evaluate"}:
        train = read(root / "train.parquet", (*SEARCH_FIELDS, "item_id"))
    if args.command == "profile":
        texts = {normalize(x["search_query"]) for x in train}
        keys = {query_key(x) for x in train}
        pairs = {(query_key(x), x["item_id"]) for x in train}
        save_json(args.output, {
            "files": {p.name: {"rows": pq.ParquetFile(p).metadata.num_rows,
                               "columns": pq.ParquetFile(p).schema_arrow.names}
                      for p in sorted(root.glob("*.parquet"))},
            "unique_train_pairs": len(pairs),
            "duplicate_train_pairs": len(train) - len(pairs),
            "benchmark_text_seen_fraction": sum(normalize(q["search_query"]) in texts for q in queries) / len(queries),
            "benchmark_full_query_seen_fraction": sum(query_key(q) in keys for q in queries) / len(queries),
            "corpus_items_seen_fraction": len(ids & {x["item_id"] for x in train}) / len(ids),
            "train_positive_in_corpus_fraction": sum(x["item_id"] in ids for x in train) / len(train),
        })
        return
    if args.command == "evaluate":
        fit, valid = split_rows(train, mode=args.split, seed=args.seed)
        truth, representatives = defaultdict(set), {}
        for row in valid:
            if row["item_id"] in ids:
                key = query_key(row)
                truth[key].add(row["item_id"])
                representatives[key] = row
        # Stable hash order avoids evaluating only the first rows of the dataset.
        import hashlib
        keys = sorted(truth, key=lambda x: hashlib.sha256(repr(x).encode()).digest())
        if args.max_queries > 0:
            keys = keys[:args.max_queries]
        model = build_retriever(args, items)
        predictions = {key: model.retrieve(representatives[key]) for key in keys}
        save_json(args.output, {"method": args.method,
            "bm25_fields": {
                "title": {"k1": args.k1 if args.k1 is not None else args.title_k1,
                          "b": args.b if args.b is not None else args.title_b,
                          "weight": args.title_weight},
                "params": {"k1": args.k1 if args.k1 is not None else args.params_k1,
                           "b": args.b if args.b is not None else args.params_b,
                           "weight": args.params_weight},
                "description": {
                    "k1": args.k1 if args.k1 is not None else args.description_k1,
                    "b": args.b if args.b is not None else args.description_b,
                    "weight": args.description_weight},
                "rrf_k": args.bm25_rrf_k,
                "channel_quota": args.bm25_channel_quota,
            } if args.method != "dense" else None,
            "embedding_model": args.embedding_model if args.method != "bm25" else None,
            "candidate_k": args.candidate_k if args.method == "hybrid" else None,
            "bm25_weight": args.bm25_weight if args.method == "hybrid" else None,
            "dense_weight": args.dense_weight if args.method == "hybrid" else None,
            "rrf_k": args.rrf_k if args.method == "hybrid" else None,
            "channel_quota": args.channel_quota if args.method == "hybrid" else None,
            "split": args.split, "seed": args.seed,
            "fit_rows": len(fit), "heldout_rows": len(valid), "evaluated_queries": len(keys),
            "heldout_in_corpus_fraction": sum(x["item_id"] in ids for x in valid) / len(valid),
            "recall_at_50": recall_at_k(predictions, {k: truth[k] for k in keys}),
            "max_queries": args.max_queries})
    else:
        model = build_retriever(args, items)
        rows = [{"query_id": q["query_id"], "answer": " ".join(model.retrieve(q))} for q in queries]
        validate_answers(rows, [q["query_id"] for q in queries], ids)
        args.answer.parent.mkdir(parents=True, exist_ok=True)
        with args.answer.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["query_id", "answer"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Validated submission written: {args.answer} ({len(rows)} queries)")


if __name__ == "__main__":
    main()
