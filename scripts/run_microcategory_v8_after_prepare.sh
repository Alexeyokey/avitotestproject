#!/usr/bin/env bash
# Resume only AFTER a fully closed V8 preparation; never edit this running file.
set -euo pipefail
cd /home/alex/avitotestproject-reranker-new-octen
export PYTHONPATH=src:scripts HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
PYTHON=/home/alex/avitotestproject/.venv/bin/python
BASE=artifacts/fresh-validation-2026-10-13
NEW=artifacts/microcategory-research-2026-10-13
TRAIN="$NEW/train-v8-composite.parquet"
MODEL="$NEW/hgb-v8-composite150.joblib"
if [[ -e "$MODEL" || -e "$NEW/cache-v8-composite" || -e "$NEW/eval-v8-composite" ]]; then
  echo 'Refusing to overwrite later-stage artifacts' >&2
  exit 1
fi
"$PYTHON" -c 'import json,pyarrow.parquet as pq; p="artifacts/microcategory-research-2026-10-13/train-v8-composite.parquet"; m=json.load(open(p+".json")); s=m["stats"]; assert m["feature_version"]==8 and s["sampled_queries"]==30000; assert pq.ParquetFile(p).metadata.num_rows==s["positive_rows"]+s["unobserved_rows"]; assert s["positive_rows"]==31860 and s["queries_with_positive_in_pool"]==27362'
"$PYTHON" -m avito_candidates.cli fit-reranker \
  --reranker-data "$TRAIN" --reranker-model "$MODEL" \
  --reranker-estimator histgb --reranker-max-iter 150 \
  --output "$NEW/fit-v8-result.json" > "$NEW/fit-v8-composite.log" 2>&1
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
