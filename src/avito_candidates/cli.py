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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["profile", "evaluate", "predict", "validate"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/result.json"))
    parser.add_argument("--answer", type=Path, default=Path("artifacts/answer.csv"))
    parser.add_argument("--split", choices=["pairs", "queries"], default="pairs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Evaluation smoke-test limit; 0 evaluates all queries")
    args = parser.parse_args()
    root = args.data_dir
    queries = read(root / "benchmark_queries.parquet", ("query_id", *SEARCH_FIELDS))
    item_columns = ("item_id", "item_title_raw", "item_infm_params_text")
    items = read(root / "benchmark_items.parquet", item_columns)
    ids = {x["item_id"] for x in items}
    if args.command == "validate":
        with args.answer.open(encoding="utf-8", newline="") as stream:
            validate_answers(list(csv.DictReader(stream)), [q["query_id"] for q in queries], ids)
        print("Submission valid")
        return
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
    from .baseline import Baseline
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
        model = Baseline(items, fit)
        predictions = {key: model.retrieve(representatives[key]) for key in keys}
        save_json(args.output, {"split": args.split, "seed": args.seed,
            "fit_rows": len(fit), "heldout_rows": len(valid), "evaluated_queries": len(keys),
            "heldout_in_corpus_fraction": sum(x["item_id"] in ids for x in valid) / len(valid),
            "recall_at_50": recall_at_k(predictions, {k: truth[k] for k in keys}),
            "max_queries": args.max_queries})
    else:
        model = Baseline(items, train)
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
