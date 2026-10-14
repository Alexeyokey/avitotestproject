"""Declare the two requested one-factor tests before fitting or scoring."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json


def identity(path):
    path = path.resolve()
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


base = Path('artifacts/geo-dense-reranker-2026-10-14')
out = Path('artifacts/geo-dense-regularization-2026-10-14')
report = json.loads((base / 'full-tune/report.json').read_text())
cache = json.loads((base / 'full-tune/tune-features.parquet.json').read_text())
assert report['full_tune_complete'] and report['selected_stage'] == 50
assert cache['complete'] and not cache['training_allowed']
assert cache['rows'] == 4740500 and cache['groups'] == 12103
plan = {
    'created_at': datetime.now(timezone.utc).isoformat(),
    'purpose': 'Two user-requested separate one-factor HGB regularization tests using the frozen Tune feature cache',
    'variants': {'maxfeatures08': {'max_features': 0.8}, 'leaves15': {'max_leaf_nodes': 15}},
    'primary_stage': 50, 'primary_baseline_stage': 50,
    'primary_metric': 'text-macro Recall@50 conditional on available positives',
    'secondary_stages': [50, 75, 100, 150],
    'secondary_selection_rule': 'smallest of 50/75/100/150 within 0.002 absolute Recall of the best for each variant',
    'fitted_trees': 150, 'early_stopping': False, 'seed': 42,
    'all_other_parameters': 'exactly clone source estimator parameters; no combined variant',
    'training_rows': 1935640, 'training_groups_with_positive': 29696,
    'training_features': 85, 'training_weights': 'unchanged original per-query class weights',
    'tune_groups': 12103, 'available_tune_groups': 4750, 'available_query_texts': 3069,
    'maximum_pool_size': 998,
    'scoring_order': 'model probability descending, RRF score descending, original pool position ascending',
    'baseline_parity': 'reproduce all original current HGB stage hits per context for 50/75/100/150',
    'uncertainty': 'paired bootstrap by normalized query text, 10000 replicates, exploratory reused Tune',
    'training_loss': 'optional diagnostic only; never selection metric',
    'retrieval_rebuilt': False, 'inventory_reencoded': False, 'audit_opened': False,
    'source_model': identity(base / 'hgb-geo005-v8-150.joblib'),
    'current_selected_model': identity(base / 'full-tune/hgb-geo005-selected50.joblib'),
    'previous_report': identity(base / 'full-tune/report.json'),
    'previous_group_scores': identity(base / 'full-tune/group-scores.jsonl'),
    'training_data': identity(base / 'train-geo005-v8.parquet'),
    'training_metadata': identity(base / 'train-geo005-v8.parquet.json'),
    'cached_tune_data': cache['data'],
    'cached_tune_metadata': identity(base / 'full-tune/tune-features.parquet.json'),
    'interpretation': 'Primary isolates each requested parameter at the same 50 trees. Secondary capacity choice is explicitly separate. This is not independent Audit or an official submission score.'
}
out.mkdir(parents=True, exist_ok=True)
with (out / 'experiment-plan.json').open('x', encoding='utf-8') as destination:
    json.dump(plan, destination, ensure_ascii=False, indent=2)
    destination.write('\n')
print(json.dumps({'plan': str(out / 'experiment-plan.json'), 'declared': True}))
