"""Cache fresh full pools; open audit labels only after a tuning selection seal.

The protocol owns query membership and all known positive IDs. Empty available
truth sets stay in the cohort, so inventory coverage and full-truth Recall can
be reported instead of hiding unsearchable targets. Retrieval uses fit-only
saved priors. This script never changes the submitted model or its index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from cache_reranker_audit import ITEM_COLUMNS, build_cache, digest, query_order
from build_fresh_validation_protocol import load_manifest
from reranker_audit_provenance import artifact_prior_policy
from avito_candidates.cli import build_retriever, read
from avito_candidates.core import SEARCH_FIELDS, held_out, normalize
from avito_candidates.reranker import Reranker, read_fit_labels


def read_protocol(root, phase, *, expected_protocol_sha=None):
    root = Path(root)
    manifest_path = root / 'manifest.json'
    if expected_protocol_sha is not None and digest(manifest_path) != expected_protocol_sha:
        raise ValueError('Protocol changed before opening its label partition')
    manifest = load_manifest(root, verify=True)
    files = manifest['files']
    for name in ('train', 'benchmark_items', f'labels_{phase}', f'{phase}_texts'):
        entry = files[name]
        path = (root / entry['path']).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != entry['sha256']:
            raise ValueError(f'Protocol input changed: {name}')
    if expected_protocol_sha is not None and digest(manifest_path) != expected_protocol_sha:
        raise ValueError('Protocol changed before opening its label partition')
    from evaluate_fresh_validation import read_phase_labels
    groups = list(read_phase_labels(root, manifest, phase).values())
    if not groups or len({tuple(group['query_key']) for group in groups}) != len(groups):
        raise ValueError('Protocol groups must be nonempty and unique')
    return manifest, groups, digest(manifest_path)


def verify_audit_seal(path, protocol_sha, model_sha, train_sha):
    seal = json.loads(Path(path).read_text(encoding='utf-8'))
    from evaluate_fresh_validation import verify_selection
    verify_selection(seal, protocol_sha, model_sha, Path(path).parent, train_sha)
    return seal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--train-data', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=('tune', 'audit', 'fit'), required=True)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--train-sample-queries', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dense-cache', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.train_sample_queries < 1:
        parser.error('Batch and fit sample sizes must be positive')
    manifest_path = args.protocol_dir / 'manifest.json'
    protocol_sha, model_sha = digest(manifest_path), digest(args.model)
    if args.phase == 'audit':
        if args.selection is None:
            parser.error('Audit labels remain locked until --selection exists')
        verify_audit_seal(args.selection, protocol_sha, model_sha, digest(args.train_data))
    # Checking the manifest and catalog bytes does not parse any audit labels.
    # Complete the model, encoder and training-provenance checks before opening
    # that partition, including schema/corpus/base checks in Reranker below.
    manifest = load_manifest(args.protocol_dir, verify=True)
    if protocol_sha != digest(manifest_path):
        raise ValueError('Protocol changed while validating its inputs')
    dataset = (args.protocol_dir / manifest['files']['train']['path']).parent
    import joblib
    artifact = joblib.load(args.model)
    metadata = artifact['metadata']
    if (metadata['split'] != 'queries' or metadata['seed'] != 42
            or metadata['validation_fraction'] != .2
            or metadata.get('training_prior_policy') != {
                'microcategory': 'leave-query-text-out', 'geography': 'leave-query-text-out'}):
        raise ValueError('Fresh model must use seed42 fit-only, self-excluding priors')
    from evaluate_fresh_validation import check_encoder_identity, check_training_data
    check_encoder_identity(artifact, manifest)
    check_training_data(args.train_data, artifact, args.protocol_dir, manifest)
    config = metadata['retrieval_config']
    items = read(dataset / 'benchmark_items.parquet', ITEM_COLUMNS)
    items_by_id = {item['item_id']: item for item in items}
    corpus_ids = set(items_by_id)
    reranker = Reranker(args.model, items, config, evaluation_split=('queries', .2, 42))
    if model_sha != digest(args.model):
        raise ValueError('Model changed while validating its provenance')
    manifest, groups, actual_protocol_sha = read_protocol(
        args.protocol_dir, args.phase, expected_protocol_sha=protocol_sha)
    if protocol_sha != actual_protocol_sha:
        raise ValueError('Protocol changed while validating its inputs')
    for group in groups:
        if held_out(normalize(group['query_text']), .2, 42) != (args.phase != 'fit'):
            raise ValueError('Protocol crosses the frozen encoder query split')
    # "seen" means selected by any allowed fit query, not proven LoRA exposure.
    import pyarrow.parquet as pq
    fit_items = set()
    for batch in pq.ParquetFile(dataset / 'train.parquet').iter_batches(
            batch_size=65536, columns=['search_query', 'item_id']):
        for row in batch.to_pylist():
            if not held_out(normalize(row['search_query'] or ''), .2, 42):
                fit_items.add(str(row['item_id']))
    namespace = SimpleNamespace(**{
        **config, 'command': 'evaluate', 'data_dir': dataset, 'split': 'queries',
        'validation_fraction': .2, 'seed': 42, 'train_batch_size': 65536,
        'k1': None, 'b': None, 'geo_include_related': config.get('geo_include_related', False),
        'dense_cache': args.dense_cache, 'embedding_batch_size': args.batch_size,
        'device': args.device,
    })
    retriever = build_retriever(namespace, items,
        geo_associations_override=reranker.builder.geo_associations)
    if args.phase == 'fit':
        fit_truth, _, _ = read_fit_labels(dataset / 'train.parquet', corpus_ids,
            split='queries', fraction=.2, seed=42, batch_size=65536)
        selected_keys = set(sorted(fit_truth, key=query_order)[:args.train_sample_queries])
        groups = [group for group in groups if tuple(group['query_key']) in selected_keys]
        reranker.builder.leave_query_out = True
        reranker.builder.leave_geography_query_out = True
        retriever.bm25.geo_leave_query_out = True
    truth, representatives, additions = {}, {}, {}
    for group in groups:
        key = tuple(group['query_key'])
        relevant = set(group['relevant_item_ids'])
        available = set(group['available_item_ids'])
        if available != relevant & corpus_ids:
            raise ValueError('Protocol item availability differs from the frozen inventory')
        truth[key] = available
        representatives[key] = dict(zip(SEARCH_FIELDS, key))
        additions[key] = {
            'group_identity': group['group_id'], 'full_relevant_count': len(relevant),
            'available_relevant_count': len(available), 'representative': group['representative'],
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase = 'train_sample' if args.phase == 'fit' else args.phase
    extra = {
        'fresh_validation_protocol_sha256': protocol_sha,
        'fresh_partition': args.phase,
        **artifact_prior_policy(artifact),
        'producer_sha256': digest(Path(__file__)),
        'full_truth_preserved': True,
    }
    build_cache(args.output_dir / f'{phase}.parquet', truth=truth,
        representatives=representatives, keys=sorted(truth, key=query_order),
        builder=reranker.builder, retriever=retriever, items_by_id=items_by_id,
        fit_items=fit_items, benchmark_strata=[], model_metadata=metadata,
        model_reference=args.model, model_hash=model_sha, batch_size=args.batch_size,
        role='train_sample' if args.phase == 'fit' else 'validation', max_pool_size=998,
        partitioner=lambda _text: args.phase, historical_holdout_reused=False,
        extra_group_metadata=additions, extra_metadata=extra, allow_empty_relevance=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
