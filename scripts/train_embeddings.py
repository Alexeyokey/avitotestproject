"""Train a query-to-listing LoRA adapter on the query-held-out split.

The first experiment uses one selected listing per normalized query text.
This prevents another known positive for the same query from becoming an
in-batch negative. The evaluation query texts never enter this dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.sentence_transformer import losses

from avito_candidates.core import SEARCH_FIELDS, held_out, normalize
from avito_candidates.dense import (
    DEFAULT_EMBEDDING_MODEL,
    item_text,
    query_text,
)


ITEM_FIELDS = (
    "item_id", "item_title_raw", "item_category_id",
    "item_infm_params_text", "item_description_raw",
)


def make_dataset(data_dir: Path, seed: int, limit: int, prompts: dict):
    columns = list(dict.fromkeys((*SEARCH_FIELDS, *ITEM_FIELDS)))
    rows = pd.read_parquet(data_dir / "train.parquet", columns=columns)
    rows = rows.fillna("").astype(str)
    rows["_query_text_key"] = rows["search_query"].map(normalize)
    keys = rows["_query_text_key"].drop_duplicates()
    validation_keys = set(keys[keys.map(lambda key: held_out(key, seed=seed))])
    fit = rows.loc[~rows["_query_text_key"].isin(validation_keys)]
    fit_rows = len(fit)
    # Choose one positive per normalized query text, independent of input order.
    # The same seed gives the same sample on repeated runs.
    selected = (fit.sample(frac=1, random_state=seed)
                  .drop_duplicates("_query_text_key"))
    if limit:
        selected = selected.sample(n=min(limit, len(selected)), random_state=seed)
    selected = selected.sort_values("_query_text_key")

    query_prompt = prompts["query"]
    document_prompt = prompts["document"]
    anchors = [query_prompt + query_text(row) for row in selected.to_dict("records")]
    positives = [document_prompt + item_text(row) for row in selected.to_dict("records")]
    dataset = Dataset.from_dict({"anchor": anchors, "positive": positives})
    stats = {
        "fit_rows": fit_rows,
        "fit_query_texts": int(fit["_query_text_key"].nunique()),
        "training_pairs": len(dataset),
        "validation_query_texts_excluded": len(validation_keys),
        "seed": seed,
    }
    return dataset, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--limit-query-texts", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()
    if args.max_length != 128:
        parser.error("This experiment fixes the embedding length at 128 tokens")
    if args.batch_size < 2 or not 0 < args.mini_batch_size <= args.batch_size:
        parser.error("Require batch-size >= 2 and 0 < mini-batch-size <= batch-size")
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    if not torch.cuda.is_available():
        parser.error("CUDA GPU is required for this experiment")
    args.output_dir.mkdir(parents=True)

    model = SentenceTransformer(
        args.model,
        device="cuda",
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    model.max_seq_length = args.max_length
    if "query" not in model.prompts or "document" not in model.prompts:
        raise ValueError("The selected model must provide query and document prompts")
    dataset, stats = make_dataset(
        args.data_dir, args.seed, args.limit_query_texts, model.prompts,
    )
    (args.output_dir / "data_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False), flush=True)

    model.add_adapter(LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    ))
    model[0].auto_model.config.use_cache = False
    model[0].auto_model.gradient_checkpointing_enable()
    loss = losses.CachedMultipleNegativesRankingLoss(
        model, mini_batch_size=args.mini_batch_size,
    )
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=0.05,
        bf16=True,
        gradient_checkpointing=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=10,
        logging_first_step=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss,
    )
    trainer.train()
    model_path = args.output_dir / "model"
    model.save(str(model_path))
    print(f"Saved model: {model_path}", flush=True)


if __name__ == "__main__":
    main()
