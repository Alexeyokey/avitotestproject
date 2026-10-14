#!/usr/bin/env bash
set -euo pipefail
cd /home/alex/avitotestproject-reranker-new-octen
export PYTHONPATH=src HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
PYTHON=/home/alex/avitotestproject/.venv/bin/python
ROOT=artifacts/fresh-validation-2026-10-13
CHECKPOINT=/home/alex/avitotestproject/artifacts/octen-lora-128-full-mb16-s42/model
COMMON=(
  --data-dir "$ROOT/protocol/dataset" --split queries --seed 42
  --method hybrid --device cuda --embedding-model "$CHECKPOINT"
  --embedding-batch-size 64 --embedding-max-length 128
  --embedding-params-words 40 --embedding-description-words 48
  --candidate-k 449 --ef-search 1000 --rrf-k 60
  --title-weight 0 --params-weight 0 --description-weight 0.5
  --stem-title-weight 0 --stem-params-weight 0
  --location-bonus 0.02 --channel-quota 0
  --geo-candidate-k 100 --geo-top-locations 3 --geo-min-history 20 --geo-weight 0.25
  --dense-cache "$ROOT/dense"
)
if [[ -e "$ROOT/train.parquet" || -e "$ROOT/hgb150.joblib" ]]; then
  echo 'Refusing to overwrite a previous fresh preparation/model' >&2
  exit 1
fi
"$PYTHON" -m avito_candidates.cli prepare-reranker "${COMMON[@]}" \
  --reranker-feature-version 4 --reranker-crossfit-geography \
  --max-train-queries 30000 --reranker-negatives-per-query 64 \
  --reranker-data "$ROOT/train.parquet"
# Release the search indices before fitting the trees.
"$PYTHON" -m avito_candidates.cli fit-reranker \
  --reranker-data "$ROOT/train.parquet" --reranker-model "$ROOT/hgb150.joblib" \
  --reranker-estimator histgb --reranker-max-iter 150
