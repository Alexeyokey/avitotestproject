#!/usr/bin/env bash
# The finished model and feature preparation are reused unchanged.
set -euo pipefail
cd /home/alex/avitotestproject-reranker-new-octen
export PYTHONPATH=src:scripts HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
PYTHON=/home/alex/avitotestproject/.venv/bin/python
BASE=artifacts/fresh-validation-2026-10-13
NEW=artifacts/microcategory-research-2026-10-13
TRAIN="$NEW/train-v8-composite.parquet"
MODEL="$NEW/hgb-v8-composite150.joblib"
if [[ ! -s "$MODEL" || ! -s "$TRAIN" || -e "$NEW/cache-v8-composite" || -e "$NEW/eval-v8-composite" ]]; then
  echo 'Require finished inputs and fresh cache/evaluation outputs' >&2
  exit 1
fi
"$PYTHON" scripts/cache_research_reranker.py \
  --protocol-dir "$BASE/protocol" --model "$MODEL" --train-data "$TRAIN" \
  --output-dir "$NEW/cache-v8-composite" --dense-cache "$BASE/dense" \
  --encoder-base-reference "$NEW/frozen-encoder-base.json" --batch-size 64 \
  > "$NEW/cache-v8-composite.log" 2>&1
"$PYTHON" scripts/evaluate_research_reranker.py \
  --protocol-dir "$BASE/protocol" --new-model "$MODEL" --new-train-data "$TRAIN" \
  --new-cache-dir "$NEW/cache-v8-composite" \
  --old-model "$BASE/hgb150-corrected.joblib" --old-train-data "$BASE/train-corrected.parquet" \
  --old-cache-dir "$BASE/cache" --new-stages 75 100 150 --primary-stage 150 \
  --output-dir "$NEW/eval-v8-composite" --with-training-loss \
  > "$NEW/eval-v8-composite.log" 2>&1
printf '%s\n' 'COMPOSITE_V8_COMPLETE'
