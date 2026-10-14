"""Encode the original 30k fit contexts, reusing immutable inventory vectors.

Fit labels determine eligible contexts exactly as in the previous preparation.
No Tune or Audit labels are opened. Corpus embeddings are never recomputed.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from avito_candidates.core import SEARCH_FIELDS, normalize, held_out
from avito_candidates.dense import DenseRetriever, encoding_input, query_text
from avito_candidates.reranker import corpus_fingerprint, read_fit_labels
from build_fresh_validation_protocol import load_manifest
from produce_document_hypotheses import (iter_catalog_records, check_encoder_base,
                                        require_offline_environment)
import evaluate_fresh_validation as EVAL


def selected_keys(truth, fit_texts, limit):
    keys = sorted(truth, key=lambda key: hashlib.sha256(repr(key).encode()).digest())
    if limit:
        keys = keys[:limit]
    for key in keys:
        text = normalize(key[0])
        if text not in fit_texts or held_out(text, .2, 42):
            raise ValueError('Fit query overlaps the held-out side of the protocol')
    return keys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol-dir', 'source-train-data', 'dense-cache',
                 'corpus-vector-dir', 'encoder-base-reference', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--max-train-queries', type=int, default=30000)
    a = p.parse_args()
    require_offline_environment()
    if a.output_dir.exists():
        raise FileExistsError('Use a new output directory')
    if a.batch_size <= 0 or a.max_train_queries <= 0:
        raise ValueError('Positive batch and sampling limits required')
    manifest = load_manifest(a.protocol_dir, verify=False)
    protocol_hash = EVAL.sha256_file(a.protocol_dir / 'manifest.json')
    metadata_path = a.source_train_data.with_suffix(a.source_train_data.suffix + '.json')
    metadata_hash = EVAL.sha256_file(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    config = metadata['retrieval_config']
    if metadata['feature_version'] != 8 or metadata['seed'] != 42 or metadata['split'] != 'queries':
        raise ValueError('Expected original V8 seed42 metadata')
    old_vector_manifest = json.loads((a.corpus_vector_dir / 'manifest.json').read_text())
    if not old_vector_manifest.get('complete') or old_vector_manifest['protocol_sha256'] != protocol_hash:
        raise ValueError('Corpus extraction belongs to another protocol')
    for name in ('corpus.npy', 'order.json'):
        if EVAL.sha256_file(a.corpus_vector_dir / name) != old_vector_manifest['files'][name]['sha256']:
            raise ValueError('Original corpus extraction changed')
    corpus_order = json.loads((a.corpus_vector_dir / 'order.json').read_text())['corpus_item_ids']
    fit_path = EVAL.protocol_file(a.protocol_dir, manifest, 'fit_texts')
    fit_texts = set(json.loads(fit_path.read_text())['texts'])
    item_path = EVAL.protocol_file(a.protocol_dir, manifest, 'benchmark_items')
    train_path = EVAL.protocol_file(a.protocol_dir, manifest, 'train')
    items = list(iter_catalog_records(item_path))
    ids = [item['item_id'] for item in items]
    if ids != corpus_order or corpus_fingerprint(items, feature_version=8) != metadata['corpus_fingerprint']:
        raise ValueError('Original inventory identity or vector order differs')
    truth, representatives, stats = read_fit_labels(train_path, set(ids), split='queries',
        fraction=.2, seed=42, batch_size=4096)
    keys = selected_keys(truth, fit_texts, a.max_train_queries)
    query_groups = [dict(zip(SEARCH_FIELDS, key)) for key in keys]
    del truth, representatives
    gc.collect()
    base = check_encoder_base(config['embedding_model'], manifest, a.encoder_base_reference)
    index_inputs = {str(path.resolve()): EVAL.sha256_file(path)
        for path in sorted(a.dense_cache.iterdir()) if path.suffix in ('.bin', '.json')}
    if index_inputs != old_vector_manifest['inventory_index_inputs']:
        raise ValueError('Existing HNSW inventory differs from previous experiment')
    print(json.dumps({'phase': 'load_existing_index', 'items': len(items),
        'queries': len(keys), 'fit_stats': stats}), flush=True)
    dense = DenseRetriever(items, model_name=config['embedding_model'], cache_dir=a.dense_cache,
        batch_size=a.batch_size, device='cuda', max_seq_length=config['embedding_max_length'],
        params_words=config['embedding_params_words'], description_words=config['embedding_description_words'],
        ef_search=config['ef_search'], require_cached_index=True)
    dense.items = None
    del items
    gc.collect()
    a.output_dir.mkdir(parents=True)
    qvectors = np.lib.format.open_memmap(a.output_dir / 'queries.npy', mode='w+',
        dtype=np.float32, shape=(len(keys), dense.dimension))
    ann = np.lib.format.open_memmap(a.output_dir / 'ann998.npy', mode='w+',
        dtype=np.int32, shape=(len(keys), 998))
    cosines = np.lib.format.open_memmap(a.output_dir / 'ann998-cosines.npy', mode='w+',
        dtype=np.float32, shape=(len(keys), 998))
    dense.index.set_ef(config['ef_search'])
    start = last = time.monotonic()
    for offset in range(0, len(keys), a.batch_size):
        stop = min(offset + a.batch_size, len(keys))
        texts, kwargs = encoding_input([query_text(q) for q in query_groups[offset:stop]],
            dense.model_name, 'query', dense.model)
        raw = np.asarray(dense.model.encode(texts, batch_size=a.batch_size,
            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False, **kwargs), dtype=np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        if raw.shape != (stop - offset, dense.dimension) or not np.isfinite(raw).all() or (norms <= 0).any():
            raise ValueError('Invalid encoded fit queries')
        qvectors[offset:stop] = raw / norms
        labels, distances = dense.index.knn_query(raw, k=998, num_threads=4)
        ann[offset:stop], cosines[offset:stop] = labels, 1 - distances
        now = time.monotonic()
        if offset == 0 or stop == len(keys) or now - last >= 30:
            print(json.dumps({'phase': 'fit_queries_encode_and_ann', 'done': stop,
                'total': len(keys), 'seconds': round(now - start, 1)}), flush=True)
            last = now
    for array in (qvectors, ann, cosines):
        array.flush()
    order = {'corpus_item_ids': ids, 'query_keys': [list(key) for key in keys],
        'query_group_ids': [hashlib.sha256(json.dumps(list(key), ensure_ascii=False,
            separators=(',', ':')).encode()).hexdigest() for key in keys]}
    (a.output_dir / 'order.json').write_text(json.dumps(order, ensure_ascii=False), encoding='utf-8')
    if any(EVAL.sha256_file(Path(path)) != digest for path, digest in index_inputs.items()):
        raise ValueError('Existing HNSW inventory changed')
    if EVAL.sha256_file(metadata_path) != metadata_hash:
        raise ValueError('Original training metadata changed')
    report = {'complete': True, 'phase': 'fit', 'fit_only': True,
        'tune_labels_opened': False, 'audit_labels_opened': False,
        'inventory_reencoded': False, 'queries_encoded': len(keys), 'items': len(ids),
        'dimension': dense.dimension, 'protocol_sha256': protocol_hash,
        'metadata_sha256': metadata_hash, 'metadata_path': str(metadata_path.resolve()),
        'fit_texts_sha256': EVAL.sha256_file(fit_path), 'fit_data_sha256': EVAL.sha256_file(train_path),
        'selection': 'sha256(repr(query_key)); eligible available positives; first30000',
        'max_train_queries': a.max_train_queries, 'fit_stats': stats,
        'corpus_vectors_sha256': old_vector_manifest['files']['corpus.npy']['sha256'],
        'encoder_base': base, 'inventory_index_inputs': index_inputs,
        'source_sha256': EVAL.sha256_file(Path(__file__)),
        'files': {path.name: {'sha256': EVAL.sha256_file(path), 'bytes': path.stat().st_size}
                  for path in sorted(a.output_dir.iterdir())}}
    (a.output_dir / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'complete': True, 'seconds': time.monotonic() - start,
        'output_dir': str(a.output_dir)}), flush=True)


if __name__ == '__main__':
    main()
