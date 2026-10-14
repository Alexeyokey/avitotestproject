#!/usr/bin/env bash
# Tune-only geographical dense retrieval on existing, immutable Octen vectors.
set -euo pipefail
cd /home/alex/avitotestproject-reranker-new-octen
export PYTHONPATH=src:scripts HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
PYTHON=/home/alex/avitotestproject/.venv/bin/python
BASE=artifacts/fresh-validation-2026-10-13
CURRENT=artifacts/microcategory-research-2026-10-13
EXPERIMENT=artifacts/geo-dense-research-2026-10-14
"$PYTHON" scripts/evaluate_geo_dense_tune.py \
  --protocol-dir "$BASE/protocol" \
  --model "$CURRENT/hgb-v8-selected100.joblib" \
  --train-data "$CURRENT/train-v8-composite.parquet" \
  --cache-dir "$CURRENT/cache-v8-composite" \
  --baseline-report "$CURRENT/eval-v8-composite/research-tune-report.json" \
  --baseline-scores "$CURRENT/eval-v8-composite/research-tune-group-scores.jsonl" \
  --vector-dir "$EXPERIMENT/vectors" \
  --encoder-base-reference "$CURRENT/frozen-encoder-base.json" \
  --output-dir "$EXPERIMENT/full-tune" \
  --batch-size 16 --bootstrap-replicates 2000 \
  > "$EXPERIMENT/full-tune.log" 2>&1
echo GEO_DENSE_FULL_TUNE_COMPLETE
