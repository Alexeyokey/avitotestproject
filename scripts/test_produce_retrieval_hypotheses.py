import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from produce_retrieval_hypotheses import selected_groups, json_digest, dense_query_channels
from avito_candidates.dense import DenseRetriever


class ProducerTest(unittest.TestCase):
    def group(self, text='ремонт'):
        key = [text, '1', '0', '', '114']
        return {'group_identity': json_digest(key), 'query_key': key}

    def metadata(self, groups):
        return {'fresh_partition': 'tune', 'historical_holdout_reused': False, 'groups': groups}

    def test_context_identity_and_training_exclusion(self):
        group = self.group()
        self.assertEqual(selected_groups(self.metadata([group]), {'другой'}), [group])
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            selected_groups(self.metadata([group]), {'ремонт'})

    def test_refuses_audit_and_duplicate_or_modified_group(self):
        group = self.group()
        for meta in ({**self.metadata([group]), 'fresh_partition': 'audit'},
                     self.metadata([group, group]),
                     self.metadata([{**group, 'group_identity': '0' * 64}])):
            with self.assertRaises(ValueError):
                selected_groups(meta, set())

    def dense(self, calls, default=None):
        class Model:
            prompts = {'query': 'instruction', 'document': ' '}
            default_prompt_name = default
            def encode(self, texts, **kwargs):
                calls.append((list(texts), kwargs))
                return np.asarray([[1, 0] for _ in texts], dtype=np.float32)
        class Index:
            def set_ef(self, ef):
                self.ef = ef
            def knn_query(self, vectors, k, num_threads):
                return np.tile(np.arange(k), (len(vectors), 1)), np.zeros((len(vectors), k))
        return SimpleNamespace(model=Model(), model_name='/local/model',
            ids=['a', 'b'], dimension=2, batch_size=2, ef_search=1000, index=Index())

    def test_query_only_variants_preserve_prompts_and_neighbors(self):
        calls = []
        dense = self.dense(calls)
        rows = dense_query_channels(dense, [{'search_query': 'ремонт', 'search_category': '114'}],
            ['current', 'plain', 'no_prompt'], 2)
        self.assertEqual(rows, [{'dense_current': ['a', 'b'], 'dense_plain': ['a', 'b'],
                                 'dense_no_prompt': ['a', 'b']}])
        self.assertEqual(calls[1][0], ['ремонт'])
        self.assertEqual(calls[0][1]['prompt_name'], 'query')
        self.assertNotIn('prompt_name', calls[2][1])
        self.assertEqual(dense.index.ef, 1000)

    def test_no_prompt_refuses_implicit_prompt(self):
        with self.assertRaisesRegex(ValueError, 'implicit'):
            dense_query_channels(self.dense([], 'query'), [{'search_query': 'x'}], ['no_prompt'], 2)

    def test_cache_only_dense_refuses_new_inventory_encoding(self):
        class Model:
            prompts = {}
            def get_embedding_dimension(self):
                return 2
            def encode(self, *_args, **_kwargs):
                raise AssertionError('Must not encode inventory')
        dependencies = (SimpleNamespace(Index=lambda **_kw: object()), lambda *_a, **_kw: Model())
        with tempfile.TemporaryDirectory() as temp, patch('avito_candidates.dense._load_dependencies',
                                                          return_value=dependencies):
            with self.assertRaisesRegex(FileNotFoundError, 'Matching completed'):
                DenseRetriever([{'item_id': 'a'}], model_name='model', cache_dir=Path(temp),
                               require_cached_index=True)


if __name__ == '__main__':
    unittest.main()
