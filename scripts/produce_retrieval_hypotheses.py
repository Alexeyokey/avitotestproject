"""Produce target-blind ranked channels for the fixed fresh Tune cohort.

The existing inventory, encoder and baseline cache are immutable. Query-only
layouts reuse the completed dense index. This process reads query contexts
from cache metadata, never Tune or Audit relevance labels. Outputs are
streamed in cache group order, with SHA-bound provenance for later evaluation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from avito_candidates.core import SEARCH_FIELDS, normalize
from avito_candidates.dense import DenseRetriever, encoding_input, local_checkpoint_content_fingerprint
from avito_candidates.embedding_formats import query_format, extract_filter_values, FORMAT_VERSION
from avito_candidates.reranker import corpus_fingerprint


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def selected_groups(metadata, fit_texts):
    if metadata.get('fresh_partition') != 'tune' or metadata.get('historical_holdout_reused') is not False:
        raise ValueError('Only the fresh Tune cache is permitted for hypothesis selection')
    groups = metadata['groups']
    identities = set()
    for group in groups:
        key = group['query_key']
        if len(key) != len(SEARCH_FIELDS) or not all(isinstance(x, str) for x in key):
            raise ValueError('Invalid query context')
        expected = json_digest(key)
        if group['group_identity'] != expected or expected in identities:
            raise ValueError('Duplicate or inconsistent group identity')
        if normalize(key[0]) in fit_texts:
            raise ValueError('Tune query overlaps the permitted training texts')
        identities.add(expected)
    return groups


def dense_query_channels(dense, queries, variants, depth):
    """Change query input only; keep corpus vectors, HNSW and depth fixed."""
    output = [{} for _ in queries]
    for variant in variants:
        layout = 'current' if variant == 'no_prompt' else variant
        texts = [query_format(query, layout) for query in queries]
        prompt_kwargs = {}
        if variant != 'no_prompt':
            texts, prompt_kwargs = encoding_input(texts, dense.model_name, 'query', dense.model)
        elif getattr(dense.model, 'default_prompt_name', None) is not None:
            raise ValueError('No-prompt diagnostic requires no implicit default prompt')
        vectors = np.asarray(dense.model.encode(
            texts, batch_size=dense.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False, **prompt_kwargs,
        ), dtype=np.float32)
        if vectors.shape != (len(queries), dense.dimension) or not np.isfinite(vectors).all():
            raise ValueError('Invalid encoded query vectors')
        count = min(depth, len(dense.ids))
        dense.index.set_ef(max(dense.ef_search, count))
        labels, _distances = dense.index.knn_query(vectors, k=count, num_threads=4)
        for row, neighbors in zip(output, labels):
            ids = [dense.ids[int(label)] for label in neighbors]
            if len(ids) != len(set(ids)):
                raise ValueError('Dense search returned duplicate items')
            row['dense_' + variant] = ids
    return output


def build_indexes(items):
    from avito_candidates.baseline import BM25Index
    ids = [item['item_id'] for item in items]
    return {
        'title': BM25Index(ids, [i.get('item_title_raw', '') for i in items], k1=1, b=.2),
        'params': BM25Index(ids, [i.get('item_infm_params_text', '') for i in items], k1=1.2, b=.7),
        'description': BM25Index(ids, [i.get('item_description_raw', '') for i in items], k1=1, b=.8),
    }


def produce(args):
    from avito_candidates.cli import read
    root = args.protocol_dir.resolve()
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    metadata_path = args.cache_dir / 'tune.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata['fresh_validation_protocol_sha256'] != sha256(manifest_path):
        raise ValueError('Protocol differs from the completed Tune cache')
    fit_path = root / manifest['files']['fit_texts']['path']
    fit_texts = set(json.loads(fit_path.read_text(encoding='utf-8'))['texts'])
    if sha256(fit_path) != manifest['files']['fit_texts']['sha256']:
        raise ValueError('Permitted training text list changed')
    groups = selected_groups(metadata, fit_texts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = Path(str(args.output) + '.json')
    if args.output.exists() or sidecar.exists():
        raise FileExistsError('Use a new candidate output path')
    item_path = root / manifest['files']['benchmark_items']['path']
    if sha256(item_path) != manifest['files']['benchmark_items']['sha256']:
        raise ValueError('Inventory changed')
    import pyarrow.parquet as pq
    items = read(item_path, pq.ParquetFile(item_path).schema_arrow.names)
    if corpus_fingerprint(items, feature_version=4) != metadata['corpus_fingerprint']:
        raise ValueError('Inventory feature fingerprint differs')
    config = metadata['retrieval_config']
    dense, indexes, classifier, micro = None, None, None, None
    model_path = Path(config['embedding_model']) / 'adapter_model.safetensors'
    if args.kind == 'dense':
        dense = DenseRetriever(items, model_name=config['embedding_model'],
            cache_dir=args.dense_cache, batch_size=args.batch_size,
            max_seq_length=config['embedding_max_length'],
            params_words=config['embedding_params_words'],
            description_words=config['embedding_description_words'],
            ef_search=config['ef_search'], device=args.device,
            require_cached_index=True)
    else:
        indexes = build_indexes(items)
        if args.kind == 'micro':
            from avito_candidates.microcategory import SoftMicrocategoryClassifier, MicrocategoryChannel
            train_path = root / manifest['files']['train']['path']
            classifier = SoftMicrocategoryClassifier.load(args.microcategory_model,
                allowed_fit_texts=fit_texts, corpus_fingerprint=metadata['corpus_fingerprint'],
                source_sha256=manifest['files']['train']['sha256'])
            if sha256(train_path) != manifest['files']['train']['sha256']:
                raise ValueError('Classifier training source changed')
            micro = {top_n: MicrocategoryChannel(classifier, items, indexes=indexes,
                top_categories=top_n, max_pool_size=998) for top_n in args.top_categories}
            model_path = args.microcategory_model
    started = time.monotonic()
    open_fn = gzip.open if args.output.suffix == '.gz' else open
    with open_fn(args.output, 'wt', encoding='utf-8') as stream:
        for start in range(0, len(groups), args.batch_size):
            batch = groups[start:start + args.batch_size]
            queries = [dict(zip(SEARCH_FIELDS, group['query_key'])) for group in batch]
            if args.kind == 'dense':
                channels = dense_query_channels(dense, queries, args.query_variants, args.depth)
            else:
                channels = []
                predictions = (classifier.predict_categories_batch(queries, top_k=5)
                               if classifier is not None else [None] * len(queries))
                for query, categories in zip(queries, predictions):
                    text = query['search_query']
                    values = extract_filter_values(query.get('search_infm_params_text', ''))
                    typed_text = (text + ' ' + values).strip()
                    lexical = {name: index._score(text) for name, index in indexes.items()}
                    ranked = {}
                    if args.kind == 'lexical':
                        for name, (scores, candidates) in lexical.items():
                            ranked['lexical_' + name] = indexes[name]._top(scores, candidates, args.depth)
                            ranked['lexical_' + name + '_values'] = indexes[name].retrieve(typed_text, args.depth)
                    else:
                        lexical = {name: index._score(typed_text)[0] for name, index in indexes.items()}
                        lexical = {name: (scores if scores is not None else np.zeros(len(items), dtype=np.float32))
                                   for name, scores in lexical.items()}
                        for top_n in args.top_categories:
                            lists, _scores, _details = micro[top_n].ranked_lists_with_scores(
                                query, args.depth, predicted_categories=categories[:top_n],
                                lexical_scores=lexical)
                            for name, ids, _weight in lists:
                                ranked[f'{name}_top{top_n}'] = ids
                    channels.append(ranked)
            for group, ranked in zip(batch, channels):
                stream.write(json.dumps({'group_id': group['group_identity'],
                    'query_key': group['query_key'], 'channels': ranked},
                    ensure_ascii=False, separators=(',', ':')) + '\n')
            if start == 0 or (start // args.batch_size + 1) % 10 == 0 or start + len(batch) == len(groups):
                print(json.dumps({'kind': args.kind, 'groups': start + len(batch),
                    'total': len(groups), 'seconds': round(time.monotonic() - started, 1)}), flush=True)
    from evaluate_retrieval_hypotheses import candidate_metadata
    result = candidate_metadata(args.output, root, args.cache_dir, Path(__file__),
        model_path=model_path if args.kind == 'micro' else None,
        provenance_kind='fitted' if args.kind == 'micro' else 'frozen_encoder',
        extra={'kind': args.kind, 'depth': args.depth, 'query_variants': args.query_variants,
            'top_categories': args.top_categories, 'format_version': FORMAT_VERSION,
            'groups': len(groups), 'seconds': time.monotonic() - started,
            'lexical_micro_input': 'query plus parsed semantic filter values',
            'document_format': 'current', 'encoder_content_sha256':
                local_checkpoint_content_fingerprint(config['embedding_model'])})
    sidecar.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'complete': True, 'output': str(args.output), 'metadata': str(sidecar)}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--dense-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=['dense', 'lexical', 'micro'], required=True)
    parser.add_argument('--microcategory-model', type=Path)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--depth', type=int, default=998)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--query-variants', nargs='+', default=['current', 'plain', 'values', 'no_category', 'no_prompt'])
    parser.add_argument('--top-categories', nargs='+', type=int, default=[1, 3, 5])
    args = parser.parse_args(argv)
    if args.batch_size < 1 or not 0 < args.depth <= 998:
        parser.error('Require positive batch and depth <=998')
    if args.kind == 'micro' and args.microcategory_model is None:
        parser.error('Microcategory candidate production requires its fitted model')
    produce(args)


if __name__ == '__main__':
    main()
