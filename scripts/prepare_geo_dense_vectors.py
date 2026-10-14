"""Extract existing inventory vectors and encode only the frozen Tune queries.

No relevance labels are read. Existing HNSW bytes are immutable inputs; a
cache mismatch fails instead of silently rebuilding inventory embeddings.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np

import evaluate_fresh_validation as EVAL
from produce_retrieval_hypotheses import selected_groups, sha256
from produce_document_hypotheses import (
    iter_catalog_records, check_encoder_base, require_offline_environment,
)
from avito_candidates.core import SEARCH_FIELDS
from avito_candidates.dense import DenseRetriever, encoding_input, query_text
from avito_candidates.reranker import corpus_fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol-dir', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--dense-cache', type=Path, required=True)
    p.add_argument('--encoder-base-reference', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=64)
    a = p.parse_args()
    require_offline_environment()
    if a.output_dir.exists():
        raise FileExistsError('Use a new immutable output directory')
    manifest, protocol_hash = EVAL.checked_protocol(a.protocol_dir)
    metadata_path = a.cache_dir / 'tune.json'
    metadata_hash = sha256(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    if metadata['fresh_validation_protocol_sha256'] != protocol_hash:
        raise ValueError('Tune cache belongs to another protocol')
    fit_path = EVAL.protocol_file(a.protocol_dir, manifest, 'fit_texts')
    fit_texts = set(json.loads(fit_path.read_text())['texts'])
    groups = selected_groups(metadata, fit_texts)
    config = metadata['retrieval_config']
    encoder_base = check_encoder_base(config['embedding_model'], manifest, a.encoder_base_reference)
    index_inputs = {str(p.resolve()): sha256(p) for p in sorted(a.dense_cache.iterdir())
                    if p.suffix in {'.bin', '.json'}}
    if len(index_inputs) != 2:
        raise ValueError('Expect exactly one completed original index and metadata')
    item_path = EVAL.protocol_file(a.protocol_dir, manifest, 'benchmark_items')
    items = list(iter_catalog_records(item_path))
    if corpus_fingerprint(items, feature_version=8) != metadata['corpus_fingerprint']:
        raise ValueError('Inventory identity differs from the current V8 cache')
    ids = [item['item_id'] for item in items]
    print(json.dumps({'stage': 'load_existing_index', 'items': len(items), 'queries': len(groups)}), flush=True)
    dense = DenseRetriever(items, model_name=config['embedding_model'],
        cache_dir=a.dense_cache, batch_size=a.batch_size, device='cuda',
        max_seq_length=config['embedding_max_length'],
        params_words=config['embedding_params_words'],
        description_words=config['embedding_description_words'],
        ef_search=config['ef_search'], require_cached_index=True)
    a.output_dir.mkdir(parents=True)
    output = a.output_dir
    vectors = np.lib.format.open_memmap(output / 'corpus.npy', mode='w+',
        dtype=np.float32, shape=(len(ids), dense.dimension))
    for start in range(0, len(ids), 2048):
        end = min(start + 2048, len(ids))
        block = np.asarray(dense.index.get_items(np.arange(start, end)), dtype=np.float32)
        if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block, axis=1), 1, atol=1e-4):
            raise ValueError('Stored corpus vectors are not finite unit vectors')
        vectors[start:end] = block
    vectors.flush()
    del vectors, block
    dense.items = None
    del items
    gc.collect()
    qvectors = np.lib.format.open_memmap(output / 'queries.npy', mode='w+',
        dtype=np.float32, shape=(len(groups), dense.dimension))
    ann = {k: np.lib.format.open_memmap(output / f'ann{k}.npy', mode='w+',
        dtype=np.int32, shape=(len(groups), k)) for k in (449, 998)}
    distances = {k: np.lib.format.open_memmap(output / f'ann{k}-cosines.npy', mode='w+',
        dtype=np.float32, shape=(len(groups), k)) for k in (449, 998)}
    dense.index.set_ef(config['ef_search'])
    started = time.monotonic()
    for start in range(0, len(groups), a.batch_size):
        stop = min(start + a.batch_size, len(groups))
        queries = [dict(zip(SEARCH_FIELDS, row['query_key'])) for row in groups[start:stop]]
        texts, kwargs = encoding_input([query_text(q) for q in queries], dense.model_name, 'query', dense.model)
        raw = np.asarray(dense.model.encode(texts, batch_size=a.batch_size,
            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False, **kwargs), dtype=np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        if not np.isfinite(raw).all() or np.any(norms <= 0):
            raise ValueError('Invalid query vectors')
        qvectors[start:stop] = raw / norms
        for k in (449, 998):
            labels, d = dense.index.knn_query(raw, k=k, num_threads=4)
            ann[k][start:stop], distances[k][start:stop] = labels, 1 - d
        if start == 0 or stop == len(groups) or stop % 640 == 0:
            print(json.dumps({'stage': 'query_only_encode_and_ann', 'groups': stop,
                'total': len(groups), 'seconds': round(time.monotonic() - started, 1)}), flush=True)
    qvectors.flush()
    for arr in list(ann.values()) + list(distances.values()):
        arr.flush()
    order = {'corpus_item_ids': ids,
             'query_group_ids': [row['group_identity'] for row in groups]}
    (output / 'order.json').write_text(json.dumps(order, ensure_ascii=False), encoding='utf-8')
    if any(sha256(Path(path)) != digest for path, digest in index_inputs.items()):
        raise ValueError('Original inventory index changed')
    if sha256(metadata_path) != metadata_hash:
        raise ValueError('Tune query metadata changed')
    report = {'complete': True, 'phase': 'tune', 'target_blind': True,
        'inventory_reencoded': False, 'queries_encoded': len(groups), 'items': len(ids),
        'dimension': dense.dimension, 'query_vectors_float32_renormalized': True,
        'ann_depths': [449, 998], 'ef_search': config['ef_search'],
        'protocol_sha256': protocol_hash,
        'metadata_path': str(metadata_path.resolve()), 'metadata_sha256': metadata_hash,
        'encoder_base': encoder_base, 'inventory_index_inputs': index_inputs,
        'source_sha256': sha256(Path(__file__)),
        'files': {path.name: {'sha256': sha256(path), 'bytes': path.stat().st_size}
                  for path in sorted(output.iterdir())}}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'complete': True, 'output_dir': str(output), 'seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__':
    main()
