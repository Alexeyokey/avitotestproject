#!/usr/bin/env bash
# A new, matching reranker for the measured <=998-item composite pool.
# The existing encoder/index and old selected reranker remain separate inputs.
set -euo pipefail
cd /home/alex/avitotestproject-reranker-new-octen
export PYTHONPATH=src:scripts HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
PYTHON=/home/alex/avitotestproject/.venv/bin/python
BASE=artifacts/fresh-validation-2026-10-13
NEW=artifacts/microcategory-research-2026-10-13
TRAIN="$NEW/train-v8-composite.parquet"
MODEL="$NEW/hgb-v8-composite150.joblib"
if [[ -e "$TRAIN" || -e "$MODEL" || -e "$NEW/cache-v8-composite" || -e "$NEW/eval-v8-composite" ]]; then
  echo 'Use fresh output paths; refusing to overwrite a completed or partial experiment' >&2
  exit 1
fi
# Cold/global equivalence was checked before substituting the OOF artifact.
"$PYTHON" -c 'import json; p=json.load(open("artifacts/microcategory-research-2026-10-13/global-cold-parity.json")); assert p["global_cold_model_equivalent"] and p["queries"] == 100'
"$PYTHON" -c 'import json; p=json.load(open("artifacts/microcategory-research-2026-10-13/cached-supplemental-production-parity-32.json")); assert p["complete"] and p["groups_checked"] == 32 and p["all_ordered_pools_match"] and p["all_base_orders_preserved"] and p["maximum_pool_size"] <= 998'
COMMON=(
  --data-dir "$BASE/protocol/dataset" --split queries --seed 42
  --method hybrid --device cuda
  --embedding-model /home/alex/avitotestproject/artifacts/octen-lora-128-full-mb16-s42/model
  --embedding-batch-size 64 --embedding-max-length 128
  --embedding-params-words 40 --embedding-description-words 48
  --candidate-k 449 --ef-search 1000 --rrf-k 60
  --title-weight 0 --params-weight 0 --description-weight 0.5
  --stem-title-weight 0 --stem-params-weight 0
  --location-bonus 0.02 --channel-quota 0
  --geo-candidate-k 100 --geo-top-locations 3 --geo-min-history 20 --geo-weight 0.25
  --dense-cache "$BASE/dense"
  --microcategory-model "$NEW/soft-microcategory-oof3.joblib"
  --microcategory-candidate-k 998 --microcategory-field combined
  --microcategory-top-categories 3 --microcategory-weight 0.25
  --supplemental-fill micro-typed-dense-v1 --rescore-dense-union
)
# Hash-order sampling selects up to 30,000 eligible fit groups. The previous
# pool yielded training rows for 26,753 of its 30,000 sampled groups.
# Both geography and the learned microcategory signal exclude the current text.
"$PYTHON" -m avito_candidates.cli prepare-reranker "${COMMON[@]}" \
  --reranker-feature-version 8 --reranker-crossfit-geography \
  --max-train-queries 30000 --reranker-negatives-per-query 64 \
  --reranker-data "$TRAIN" > "$NEW/prepare-v8-composite.log" 2>&1
# Separate process releases retrieval indices before histogram tree fitting.
"$PYTHON" -m avito_candidates.cli fit-reranker \
  --reranker-data "$TRAIN" --reranker-model "$MODEL" \
  --reranker-estimator histgb --reranker-max-iter 150 \
  > "$NEW/fit-v8-composite.log" 2>&1
"$PYTHON" scripts/cache_research_reranker.py \
  --protocol-dir "$BASE/protocol" --model "$MODEL" --train-data "$TRAIN" \
  --output-dir "$NEW/cache-v8-composite" --dense-cache "$BASE/dense" \
  --encoder-base-reference "$NEW/frozen-encoder-base.json" --batch-size 64 \
  > "$NEW/cache-v8-composite.log" 2>&1
# Tune is exploratory after the retrieval grid. Audit is not consulted.
"$PYTHON" scripts/evaluate_research_reranker.py \
  --protocol-dir "$BASE/protocol" --new-model "$MODEL" --new-train-data "$TRAIN" \
  --new-cache-dir "$NEW/cache-v8-composite" \
  --old-model "$BASE/hgb150-corrected.joblib" --old-train-data "$BASE/train-corrected.parquet" \
  --old-cache-dir "$BASE/cache" --new-stages 75 100 150 --primary-stage 150 \
  --output-dir "$NEW/eval-v8-composite" --with-training-loss \
  > "$NEW/eval-v8-composite.log" 2>&1
printf '%s\n' 'COMPOSITE_V8_COMPLETE'
