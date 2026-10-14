#!/usr/bin/env bash
# Continue only when the running fit-only preparation closes its final metadata.
set -euo pipefail
cd /home/alex/avitotestproject-reranker-new-octen
export PYTHONPATH=src:scripts HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
PYTHON=/home/alex/avitotestproject/.venv/bin/python
NEW=artifacts/geo-dense-reranker-2026-10-14
TRAIN="$NEW/train-geo005-v8.parquet"
MODEL="$NEW/hgb-geo005-v8-150.joblib"
PREPARATION_PID=${1:?Supply the PID of the already running preparation}
printf '%s\n' "$$" > "$NEW/pipeline.pid"
for output in "$MODEL" "$NEW/fit-result.json" "$NEW/preflight64" "$NEW/full-tune"; do
  if [[ -e "$output" ]]; then
    printf '%s\n' "Refusing to overwrite $output" >&2
    exit 1
  fi
done
while kill -0 "$PREPARATION_PID" 2>/dev/null; do
  sleep 10
done
"$PYTHON" - "$TRAIN" <<'PY'
import json, sys
from pathlib import Path
import pyarrow.parquet as pq
path = Path(sys.argv[1])
assert not Path(str(path) + '.INCOMPLETE').exists(), 'Preparation did not finish'
meta = json.loads(Path(str(path) + '.json').read_text())
assert meta['geo_dense_training']['complete'] is True
assert meta['stats']['sampled_queries'] == 30000
assert meta['geo_dense_training']['actual_maximum_pool_size'] <= 998
assert meta['geo_dense_training']['training_geo_dense_history'] == 'leave-query-text-out'
assert pq.ParquetFile(path).metadata.num_rows == meta['stats']['positive_rows'] + meta['stats']['unobserved_rows']
print(json.dumps({'phase': 'fit_ready', 'stats': meta['stats']}), flush=True)
PY
date -u '+FIT_START %Y-%m-%dT%H:%M:%SZ'
"$PYTHON" -u -m avito_candidates.cli fit-reranker \
  --reranker-data "$TRAIN" --reranker-model "$MODEL" \
  --reranker-estimator histgb --reranker-max-iter 150 \
  --output "$NEW/fit-result.json" > "$NEW/fit.log" 2>&1
date -u '+PREFLIGHT_START %Y-%m-%dT%H:%M:%SZ'
bash scripts/run_geo_dense_retrained_eval.sh "$NEW/preflight64" \
  --limit-groups 64 --bootstrap-replicates 100 > "$NEW/preflight64.log" 2>&1
"$PYTHON" - "$NEW/preflight64/report.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report['complete'] and report['groups'] == 64 and not report['full_tune_complete']
assert report['all_per_context_pool_metrics_reproduced']
print('PREFLIGHT_VERIFIED', flush=True)
PY
date -u '+FULL_TUNE_START %Y-%m-%dT%H:%M:%SZ'
bash scripts/run_geo_dense_retrained_eval.sh "$NEW/full-tune" \
  --bootstrap-replicates 2000 --with-training-loss > "$NEW/full-tune.log" 2>&1
date -u '+GEO_RERANKER_COMPLETE %Y-%m-%dT%H:%M:%SZ'
