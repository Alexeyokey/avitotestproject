#!/usr/bin/env bash
# The Tune-sealed pipeline is evaluated once without changing its capacity.
set -euo pipefail
cd /home/alex/avitotestproject-reranker-new-octen
export PYTHONPATH=src:scripts HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
PYTHON=/home/alex/avitotestproject/.venv/bin/python
BASE=artifacts/fresh-validation-2026-10-13
NEW=artifacts/microcategory-research-2026-10-13
if [[ ! -s "$NEW/frozen-regression-selection.json" || -e "$NEW/frozen-regression-cache" || -e "$NEW/frozen-regression-evaluation" ]]; then
  echo 'Require the completed immutable selection and new regression outputs' >&2
  exit 1
fi
"$PYTHON" scripts/cache_frozen_research_audit.py \
  --protocol-dir "$BASE/protocol" --selection "$NEW/frozen-regression-selection.json" \
  --output-dir "$NEW/frozen-regression-cache" --dense-cache "$BASE/dense" \
  --batch-size 64 > "$NEW/frozen-regression-cache.log" 2>&1
"$PYTHON" scripts/evaluate_frozen_research_audit.py \
  --protocol-dir "$BASE/protocol" --selection "$NEW/frozen-regression-selection.json" \
  --cache-dir "$NEW/frozen-regression-cache" \
  --output-dir "$NEW/frozen-regression-evaluation" > "$NEW/frozen-regression-eval.log" 2>&1
printf '%s\n' 'FROZEN_REGRESSION_COMPLETE'
