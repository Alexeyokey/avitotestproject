"""Verify that adding grouped OOF models preserves cold-query predictions.

Only query identities from cache metadata are used, never relevance labels.
The check binds both saved classifiers and the inspected query metadata.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from avito_candidates.core import SEARCH_FIELDS, normalize
from avito_candidates.microcategory import SoftMicrocategoryClassifier


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cold-model', type=Path, required=True)
    p.add_argument('--oof-model', type=Path, required=True)
    p.add_argument('--tune-metadata', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    hashes = {name: digest(path) for name, path in (
        ('cold_model', args.cold_model), ('oof_model', args.oof_model),
        ('tune_metadata', args.tune_metadata))}
    a, b = (joblib.load(path) for path in (args.cold_model, args.oof_model))
    ma, mb = a['model'], b['model']
    checks = {
        'same_fit_texts': ma.fitted_texts == mb.fitted_texts,
        'same_classes': np.array_equal(ma.classes, mb.classes),
        'same_coefficients': np.array_equal(ma.classifier.coef_, mb.classifier.coef_),
        'same_intercepts': np.array_equal(ma.classifier.intercept_, mb.classifier.intercept_),
        'same_word_vocabulary': ma.word.vocabulary_ == mb.word.vocabulary_,
        'same_word_idf': np.array_equal(ma.word.idf_, mb.word.idf_),
        'same_char_vocabulary': ma.char.vocabulary_ == mb.char.vocabulary_,
        'same_char_idf': np.array_equal(ma.char.idf_, mb.char.idf_),
        'has_three_oof_models': b['metadata']['crossfit_folds'] == 3 and len(b['crossfit_models']) == 3,
    }
    metadata = json.loads(args.tune_metadata.read_text(encoding='utf-8'))
    groups = sorted(metadata['groups'], key=lambda g: g['group_identity'])[:100]
    queries = [dict(zip(SEARCH_FIELDS, g['query_key'])) for g in groups]
    if any(normalize(q['search_query']) in ma.fitted_texts for q in queries):
        raise ValueError('Inspected query identities are not cold')
    ca = SoftMicrocategoryClassifier(a['model'], a['crossfit_models'], a['metadata'])
    cb = SoftMicrocategoryClassifier(b['model'], b['crossfit_models'], b['metadata'])
    pa = ca.predict_categories_batch(queries, top_k=209, exclude_self=True)
    pb = cb.predict_categories_batch(queries, top_k=209, exclude_self=True)
    checks['same_cold_probabilities_and_order'] = pa == pb
    actual = {name: digest(path) for name, path in (
        ('cold_model', args.cold_model), ('oof_model', args.oof_model),
        ('tune_metadata', args.tune_metadata))}
    if actual != hashes:
        raise ValueError('Input changed while checking global prediction parity')
    report = {'input_sha256': hashes, 'queries': len(queries), 'checks': checks,
              'global_cold_model_equivalent': all(checks.values()),
              'scope': 'Identical global weights and cold inference only; OOF predictions intentionally differ on fit texts'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    if not all(checks.values()):
        raise ValueError('Global cold model parity failed')


if __name__ == '__main__':
    main()
