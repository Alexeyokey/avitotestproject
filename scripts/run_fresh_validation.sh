#!/usr/bin/env bash
# Run only after run_fresh_prepare.sh finishes successfully. Every output has
# its own name; the submitted model, corpus and answer.csv are left untouched.
set -euo pipefail
cd /home/alex/avitotestproject-reranker-new-octen
export PYTHONPATH=src:scripts HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
PYTHON=/home/alex/avitotestproject/.venv/bin/python
ROOT=artifacts/fresh-validation-2026-10-13

[[ -f "$ROOT/hgb150.joblib" && -f "$ROOT/train.parquet.json" ]]
if [[ -e "$ROOT/hgb150-corrected.joblib" || -e "$ROOT/train-corrected.parquet" ]]; then
  echo 'Refusing to replace an existing fresh corrected experiment' >&2
  exit 1
fi

# The original preparation process imported the old category sentinel rule.
# Correct these two flags without repeating retrieval or changing any labels.
"$PYTHON" scripts/fix_reranker_category_sentinel.py \
  --source "$ROOT/train.parquet" --target "$ROOT/train-corrected.parquet" \
  > "$ROOT/category-correction.log" 2>&1
"$PYTHON" -m avito_candidates.cli fit-reranker \
  --reranker-data "$ROOT/train-corrected.parquet" \
  --reranker-model "$ROOT/hgb150-corrected.joblib" \
  --reranker-estimator histgb --reranker-max-iter 150 \
  > "$ROOT/fit-corrected.log" 2>&1

CACHE=(
  --protocol-dir "$ROOT/protocol"
  --model "$ROOT/hgb150-corrected.joblib"
  --train-data "$ROOT/train-corrected.parquet"
  --output-dir "$ROOT/cache"
  --batch-size 64 --device cuda --dense-cache "$ROOT/dense"
)
EVAL=(
  --protocol-dir "$ROOT/protocol" --cache-dir "$ROOT/cache"
  --model "$ROOT/hgb150-corrected.joblib"
  --train-data "$ROOT/train-corrected.parquet"
  --output-dir "$ROOT/evaluation"
)

# Tune is the only place where the five tree-count alternatives are compared.
"$PYTHON" scripts/cache_fresh_validation.py "${CACHE[@]}" --phase tune \
  > "$ROOT/cache-tune.log" 2>&1
"$PYTHON" scripts/evaluate_fresh_validation.py "${EVAL[@]}" --phase tune \
  > "$ROOT/evaluate-tune.log" 2>&1

# This command refuses to parse audit labels until the tune selection and all
# associated model, training, cache and report hashes have passed their guards.
"$PYTHON" scripts/cache_fresh_validation.py "${CACHE[@]}" --phase audit \
  --selection "$ROOT/evaluation/selection.json" \
  > "$ROOT/cache-audit.log" 2>&1
"$PYTHON" scripts/evaluate_fresh_validation.py "${EVAL[@]}" --phase audit \
  > "$ROOT/evaluate-audit.log" 2>&1

"$PYTHON" scripts/publish_fresh_reranker.py \
  --protocol-dir "$ROOT/protocol" --evaluation-dir "$ROOT/evaluation" \
  --model "$ROOT/hgb150-corrected.joblib" \
  --train-data "$ROOT/train-corrected.parquet" \
  --output "$ROOT/hgb-selected.joblib" \
  > "$ROOT/publication.log" 2>&1

echo 'Fresh tune selection and single sealed audit completed.'
