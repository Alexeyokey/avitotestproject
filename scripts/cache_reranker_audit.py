"""Cache full candidate pools for a reranker audit without changing retrieval.

The denominator includes every known positive in the corpus, including those
missing from the pool. The original query-text split is preserved because the
LoRA encoder was trained with that split. This cache is retrospective: the
same holdout was inspected during the September experiments.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from avito_candidates.cli import build_retriever, read, read_validation
from avito_candidates.core import SEARCH_FIELDS, held_out, normalize
from avito_candidates.reranker import Reranker, read_fit_labels, retrieve_details


ITEM_COLUMNS = (
    "item_id", "item_title_raw", "item_infm_params_text", "item_description_raw",
    "item_category_id", "item_location_id", "item_microcat_id", "item_price",
    "item_rating", "item_rating_reviews_count", "item_is_phone_hidden",
    "item_is_message_forbidden",
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def query_order(key):
    return hashlib.sha256(repr(key).encode()).digest()


def partition(text):
    # All variants of the same text stay together. These are retrospective
    # tuning/confirmation slices, not a new untouched evaluation set.
    return "tune" if held_out(normalize(text), 0.5, 20261012) else "confirmation"


def dense_scores_for_full_pools(dense, queries, results, id_to_label):
    """Read stored item vectors; do not recalculate corpus embeddings.

    A second inexpensive query-only encode keeps the existing retrieval path
    untouched. These extra scores do not replace the baseline's top-K scores.
    """
    from avito_candidates.dense import encoding_input, query_text
    output = [np.full(len(details["candidate_pool"]), np.nan, dtype=np.float32)
              for _baseline, details in results]
    if all("full_dense_scores" in details for _baseline, details in results):
        for index, (_baseline, details) in enumerate(results):
            scores = details["full_dense_scores"]
            output[index] = np.array([scores[item] for item in details["candidate_pool"]],
                                     dtype=np.float32)
        return output, 0.0
    nonempty = [(i, query_text(query)) for i, query in enumerate(queries)
                if normalize(query.get("search_query", ""))]
    if not nonempty:
        return output, 0.0
    texts, prompt_kwargs = encoding_input(
        [text for _i, text in nonempty], dense.model_name, "query", dense.model)
    vectors = dense.model.encode(texts, batch_size=dense.batch_size,
        convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False, **prompt_kwargs).astype(np.float32, copy=False)
    max_error = 0.0
    for (i, _text), vector in zip(nonempty, vectors):
        details = results[i][1]
        pool = details["candidate_pool"]
        if not pool:
            continue
        stored = np.asarray(dense.index.get_items([id_to_label[item] for item in pool]),
                            dtype=np.float32)
        # Match DenseRetriever.score_candidates_batch: normalize in float32
        # after encode(), since bf16 normalization can leave a small norm error.
        norm = float(np.linalg.norm(vector))
        if norm <= 0 or not np.isfinite(norm):
            raise ValueError("Full-pool query vector must have a positive finite norm")
        scores = stored @ (vector / norm)
        if not np.isfinite(scores).all() or np.max(np.abs(scores)) > 1.0001:
            raise ValueError("Invalid full-pool cosine scores")
        output[i] = scores
        existing = details.get("channel_scores", {}).get("dense", {})
        max_error = max(max_error, max((abs(float(score) - existing[item])
                         for item, score in zip(pool, scores) if item in existing), default=0.0))
    if max_error > 0.005:
        raise ValueError(f"Full-pool dense scores differ from existing search: {max_error}")
    return output, max_error


def build_cache(path, *, truth, representatives, keys, builder, retriever,
                items_by_id, fit_items, benchmark_strata, model_metadata,
                model_reference, model_hash, batch_size, role, max_pool_size):
    meta_path = path.with_suffix(".json")
    if path.exists() or meta_path.exists():
        if not (path.exists() and meta_path.exists()):
            raise ValueError(f"Incomplete cache already exists: {path}")
        cached = json.loads(meta_path.read_text(encoding="utf-8"))
        if (cached.get("cache_version") != 2 or cached["model_sha256"] != model_hash
                or cached["retrieval_config"] != builder.config
                or len(cached["groups"]) != len(keys)
                or cached["role"] != role):
            raise ValueError(f"Existing cache differs from this audit: {path}")
        print(f"Reusing completed cache {path}", flush=True)
        return cached

    has_full_dense_feature = "full_dense_score" in builder.feature_names
    fields = [*(pa.field(name, pa.float32()) for name in builder.feature_names),
              pa.field("label", pa.int8()), pa.field("group_id", pa.int32()),
              pa.field("item_id", pa.string()), pa.field("positive_local", pa.bool_()),
              pa.field("positive_seen", pa.bool_()),
              *([] if has_full_dense_feature else [pa.field("full_dense_score", pa.float32())]),
              pa.field("item_microcat_id", pa.string())]
    schema = pa.schema(fields)
    temporary = path.with_suffix(".partial.parquet")
    groups, rows = [], 0
    pool_recall_sum = baseline_recall_sum = 0.0
    max_dense_error = 0.0
    dense = getattr(retriever, "dense", retriever)
    id_to_label = {item: i for i, item in enumerate(dense.ids)}
    corpus_locations = {item["item_location_id"] for item in items_by_id.values()}
    start = last_progress = time.monotonic()
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for offset in range(0, len(keys), batch_size):
            batch_keys = keys[offset:offset + batch_size]
            queries = [representatives[key] for key in batch_keys]
            results = retrieve_details(retriever, queries, builder.config["method"],
                                       builder.config["candidate_k"])
            full_scores, dense_error = dense_scores_for_full_pools(dense, queries, results, id_to_label)
            max_dense_error = max(max_dense_error, dense_error)
            tables = []
            for key, query, (baseline, details), dense_scores in zip(batch_keys, queries, results, full_scores):
                pool = details["candidate_pool"]
                if len(pool) != len(set(pool)) or len(pool) > max_pool_size:
                    raise ValueError(f"Duplicate or oversized candidate pool: {key}")
                relevant = truth[key]
                query_location = query.get("search_location_id", "")
                local = {item_id for item_id in relevant if query_location
                         and items_by_id[item_id]["item_location_id"] == query_location}
                group_id = len(groups)
                text = normalize(query.get("search_query", ""))
                found = len(set(pool) & relevant)
                baseline_hits = len(set(baseline[:50]) & relevant)
                group = {
                    "group_id": group_id, "query_key": list(key), "query_text": text,
                    "relevant_count": len(relevant), "local_relevant_count": len(local),
                    "nonlocal_relevant_count": len(relevant - local),
                    "seen_relevant_count": len(relevant & fit_items),
                    "unseen_relevant_count": len(relevant - fit_items),
                    "pool_size": len(pool), "positives_in_pool": found,
                    "baseline_hits": baseline_hits,
                    "partition": partition(text) if role == "validation" else "train_sample",
                    "no_exact_location": query_location not in corpus_locations,
                    "has_filters": bool(query.get("search_infm_params_text", "")),
                    "one_word": len(text.split()) <= 1,
                }
                groups.append(group)
                matrix = builder.transform(query, details, baseline)
                if matrix.shape != (len(pool), len(builder.feature_names)):
                    raise ValueError("Feature matrix and pool differ")
                arrays = [pa.array(matrix[:, i], type=pa.float32())
                          for i in range(matrix.shape[1])]
                arrays.extend([
                    pa.array([int(item_id in relevant) for item_id in pool], type=pa.int8()),
                    pa.array(np.full(len(pool), group_id, dtype=np.int32)),
                    pa.array(pool, type=pa.string()),
                    pa.array([item_id in local for item_id in pool], type=pa.bool_()),
                    pa.array([item_id in relevant and item_id in fit_items for item_id in pool],
                             type=pa.bool_()),
                    *([] if has_full_dense_feature else [pa.array(dense_scores, type=pa.float32())]),
                    pa.array([items_by_id[item_id]["item_microcat_id"] for item_id in pool],
                             type=pa.string()),
                ])
                tables.append(pa.Table.from_arrays(arrays, schema=schema))
                rows += len(pool)
                pool_recall_sum += found / len(relevant)
                baseline_recall_sum += baseline_hits / len(relevant)
            if tables:
                writer.write_table(pa.concat_tables(tables))
            if time.monotonic() - last_progress >= 30 or len(groups) == len(keys):
                print(json.dumps({"cache": role, "groups": len(groups), "total": len(keys),
                                  "rows": rows, "elapsed_seconds": round(time.monotonic() - start)},
                                 ensure_ascii=False), flush=True)
                last_progress = time.monotonic()

    metadata = {
        "cache_version": 2, "role": role,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(builder.feature_names),
        "feature_schema_version": model_metadata["feature_version"],
        "retrieval_config": builder.config,
        "corpus_fingerprint": model_metadata["corpus_fingerprint"],
        "split": model_metadata["split"], "seed": model_metadata["seed"],
        "validation_fraction": model_metadata["validation_fraction"],
        "historical_holdout_reused": True,
        "model_reference": str(model_reference), "model_sha256": model_hash,
        "training_prior_policy": model_metadata.get("training_prior_policy", {
            "microcategory": "leave-query-text-out", "geography": "full-fit-history"}),
        "leave_query_out": builder.leave_query_out,
        "leave_geography_query_out": getattr(builder, "leave_geography_query_out", False),
        "full_dense_score_max_abs_difference_from_topk": max_dense_error,
        "rows": rows, "groups": groups, "benchmark_strata": benchmark_strata,
        "pool_recall": pool_recall_sum / len(keys),
        "baseline_recall_at_50": baseline_recall_sum / len(keys),
        "limitations": [
            "Positives are observed choices, and other candidates are unlabelled.",
            "Only positive items present in benchmark_items enter the denominator.",
            "The historical holdout has already been used for model selection.",
            "Train-sample features exclude the query's own microcategory history, as in training.",
        ],
    }
    temporary.replace(path)
    meta_temporary = meta_path.with_suffix(".json.partial")
    meta_temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_temporary.replace(meta_path)
    print(f"Completed {path}: pool={metadata['pool_recall']:.6f}, "
          f"baseline={metadata['baseline_recall_at_50']:.6f}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-sample-queries", type=int, default=1000)
    parser.add_argument("--max-validation-queries", type=int, default=0,
                        help="Smoke-test limit; 0 keeps the complete heldout set")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dense-cache", type=Path, default=Path("artifacts/dense"))
    parser.add_argument("--max-pool-size", type=int, default=998)
    args = parser.parse_args()
    if args.train_sample_queries < 0 or args.max_validation_queries < 0 or args.batch_size <= 0:
        parser.error("Train sample must be nonnegative and batch size positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    artifact = joblib.load(args.model)
    metadata = artifact["metadata"]
    if metadata["split"] != "queries":
        parser.error("Audit requires a model trained on the held-out queries split, not split=all")
    config = metadata["retrieval_config"]
    encoder_stats = Path(config["embedding_model"]).parent / "data_stats.json"
    if encoder_stats.exists():
        stats = json.loads(encoder_stats.read_text(encoding="utf-8"))
        if stats.get("seed") != metadata["seed"] or metadata["validation_fraction"] != 0.2:
            parser.error("The saved LoRA encoder used seed42/fraction0.2; preserve its excluded texts")

    items = read(args.data_dir / "benchmark_items.parquet", ITEM_COLUMNS)
    items_by_id = {item["item_id"]: item for item in items}
    ids = set(items_by_id)
    reranker = Reranker(args.model, items, config, evaluation_split=(
        "queries", metadata["validation_fraction"], metadata["seed"]))
    truth, representatives, *_ = read_validation(
        args.data_dir / "train.parquet", ids, mode="queries",
        fraction=metadata["validation_fraction"], seed=metadata["seed"], batch_size=65536)
    fit_truth, fit_representatives, _ = read_fit_labels(
        args.data_dir / "train.parquet", ids, split="queries",
        fraction=metadata["validation_fraction"], seed=metadata["seed"], batch_size=65536)
    if ({normalize(q[0]) for q in truth} & {normalize(q[0]) for q in fit_truth}):
        raise ValueError("Training and validation query texts overlap")
    fit_items = set().union(*fit_truth.values())
    locations = {item["item_location_id"] for item in items}
    benchmark = read(args.data_dir / "benchmark_queries.parquet", SEARCH_FIELDS)
    strata = Counter((q["search_location_id"] not in locations,
                      bool(q["search_infm_params_text"]),
                      len(normalize(q["search_query"]).split()) <= 1) for q in benchmark)
    benchmark_strata = [{"no_exact_location": a, "has_filters": b, "one_word": c,
                         "queries": n} for (a, b, c), n in sorted(strata.items())]
    namespace = SimpleNamespace(**{
        **config, "command": "evaluate", "data_dir": args.data_dir, "split": "queries",
        "validation_fraction": metadata["validation_fraction"], "seed": metadata["seed"],
        "train_batch_size": 65536, "k1": None, "b": None,
        "geo_include_related": config.get("geo_include_related", False),
        "dense_cache": args.dense_cache, "embedding_batch_size": args.batch_size,
        "device": args.device,
    })
    print(f"Building retriever for {len(truth)} validation groups; using existing dense cache", flush=True)
    retriever = build_retriever(namespace, items)
    common = dict(retriever=retriever, items_by_id=items_by_id, fit_items=fit_items,
                  benchmark_strata=benchmark_strata, model_metadata=metadata,
                  model_reference=args.model, model_hash=digest(args.model),
                  batch_size=args.batch_size, max_pool_size=args.max_pool_size)
    validation_keys = sorted(truth, key=query_order)
    if args.max_validation_queries:
        validation_keys = validation_keys[:args.max_validation_queries]
    build_cache(args.output_dir / "validation.parquet", truth=truth,
                representatives=representatives, keys=validation_keys,
                builder=reranker.builder, role="validation", **common)
    if args.train_sample_queries:
        reranker.builder.leave_query_out = True
        geography_policy = metadata.get("training_prior_policy", {}).get("geography")
        if geography_policy == "leave-query-text-out":
            associations = reranker.builder.geo_associations
            if getattr(associations, "query_counts", None) is None:
                raise ValueError("Geography training audit requires stored query-text attribution")
            # Match both candidate generation and features used during fitting.
            bm25 = getattr(retriever, "bm25", retriever)
            bm25.geo_associations = associations
            bm25.geo_leave_query_out = True
            reranker.builder.leave_geography_query_out = True
        build_cache(args.output_dir / "train_sample.parquet", truth=fit_truth,
                    representatives=fit_representatives,
                    keys=sorted(fit_truth, key=query_order)[:args.train_sample_queries],
                    builder=reranker.builder, role="train_sample", **common)


if __name__ == "__main__":
    main()
