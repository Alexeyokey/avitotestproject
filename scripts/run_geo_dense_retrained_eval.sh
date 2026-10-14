#!/usr/bin/env bash
# Evaluate the separately trained HGB on the frozen geo0.05 Tune intervention.
set -euo pipefail
cd /home/alex/avitotestproject-reranker-new-octen
export PYTHONPATH=src:scripts HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
PYTHON=/home/alex/avitotestproject/.venv/bin/python
BASE=artifacts/fresh-validation-2026-10-13
OLD=artifacts/microcategory-research-2026-10-13
GEO=artifacts/geo-dense-research-2026-10-14
NEW=artifacts/geo-dense-reranker-2026-10-14
OUTPUT_DIR=${1:?Supply a fresh output directory, followed by optional evaluator arguments}
shift
"$PYTHON" -u scripts/evaluate_geo_dense_retrained.py \
  --protocol-dir "$BASE/protocol" \
  --source-model "$OLD/hgb-v8-selected100.joblib" \
  --new-model "$NEW/hgb-geo005-v8-150.joblib" \
  --source-train-data "$OLD/train-v8-composite.parquet" \
  --train-data "$NEW/train-geo005-v8.parquet" \
  --cache-dir "$OLD/cache-v8-composite" \
  --baseline-report "$OLD/eval-v8-composite/research-tune-report.json" \
  --baseline-scores "$OLD/eval-v8-composite/research-tune-group-scores.jsonl" \
  --previous-geo-report "$GEO/full-tune/report.json" \
  --previous-geo-scores "$GEO/full-tune/group-scores.jsonl" \
  --vector-dir "$GEO/vectors" \
  --encoder-base-reference "$OLD/frozen-encoder-base.json" \
  --output-dir "$OUTPUT_DIR" --batch-size 16 --save-feature-cache "$@"
