import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from avito_candidates.core import SEARCH_FIELDS
from avito_candidates.dense import query_text
from augment_secondary_dense_v9 import (checked_key, recover_training_representatives,
    iter_complete_groups, append_cosine, score_both_document_indexes, main)


def canonical(text):
    key = [text, "1", "0", "", "114"]
    raw = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    return key, raw, hashlib.sha256(raw.encode()).hexdigest()


class SecondaryDenseTest(unittest.TestCase):
    def test_query_context_identity_is_exact_not_just_query_text(self):
        key, _raw, identity = canonical("ремонт")
        self.assertEqual(checked_key(key, identity)["search_query"], "ремонт")
        with self.assertRaises(ValueError):
            checked_key(["ремонт", "2", "0", "", "114"], identity)
        with self.assertRaises(ValueError):
            checked_key(["РЕМОНТ", "1", "0", "", "114"])

    def test_raw_representative_recovery_matches_last_in_corpus_primary_rule(self):
        key, raw, _identity = canonical("монтаж елок")
        rows = []
        for text, item in ((" МоНТаЖ  Ёлок ", "a"), ("Монтаж\nёлок", "b"),
                           ("Монтаж ёлок ", "outside")):
            rows.append({**dict(zip(SEARCH_FIELDS, key)), "search_query": text, "item_id": item})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fit.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path)
            recovered = recover_training_representatives(path, {raw}, {"монтаж елок"}, {"a", "b"}, batch_size=1)
            self.assertEqual(recovered[raw]["search_query"], "Монтаж\nёлок")
            self.assertEqual(query_text(recovered[raw]), query_text(checked_key(key)))
            with self.assertRaisesRegex(ValueError, "could not be recovered"):
                recover_training_representatives(path, {canonical("missing")[1]}, {"missing"}, {"a"})

    def test_groups_carry_across_batches_without_changing_candidate_order(self):
        first, raw1, identity1 = canonical("first")
        second, raw2, identity2 = canonical("second")
        table = pa.table({"group_id": [identity1, identity1, identity2], "query_text": ["first", "first", "second"],
            "query_key_json": [raw1, raw1, raw2], "item_id": ["b", "a", "c"], "label": [1, 0, 1]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.parquet"
            pq.write_table(table, path)
            groups = list(iter_complete_groups(path, {}, "train", batch_size=1))
            self.assertEqual([group[1] for group in groups], [["b", "a"], ["c"]])
            self.assertEqual(groups[0][0], dict(zip(SEARCH_FIELDS, first)))
            self.assertEqual(pa.concat_tables([group[2] for group in groups]), table)

    def test_tune_empty_groups_remain_in_original_metadata_not_fabricated_rows(self):
        key, _raw, identity = canonical("query")
        metadata = {"groups": [{"group_id": 0, "pool_size": 0}, {"group_id": 1,
            "pool_size": 2, "query_key": key, "group_identity": identity}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tune.parquet"
            pq.write_table(pa.table({"group_id": [1, 1], "item_id": ["a", "b"]}), path)
            groups = list(iter_complete_groups(path, metadata, "tune", batch_size=1))
            self.assertEqual(len(groups), 1)
            self.assertEqual(metadata["groups"][0]["pool_size"], 0)

    def test_source_duplicate_or_noncontiguous_groups_are_rejected(self):
        _key1, raw1, identity1 = canonical("first")
        _key2, raw2, identity2 = canonical("second")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.parquet"
            pq.write_table(pa.table({"group_id": [identity1, identity2, identity1],
                "query_text": ["first", "second", "first"], "query_key_json": [raw1, raw2, raw1],
                "item_id": ["a", "b", "c"]}), path)
            with self.assertRaisesRegex(ValueError, "noncontiguous"):
                list(iter_complete_groups(path, {}, "train", batch_size=1))

    def table(self):
        return pa.table({"item_id": ["a", "b"], "label": [1, 0], "weight": pa.array([1., .5], pa.float32()),
            "full_dense_score": pa.array([1., 0.], pa.float32()),
            "full_dense_known": pa.array([1., 1.], pa.float32())})

    def test_two_added_columns_leave_every_original_feature_label_weight_id_unchanged(self):
        table = self.table()
        output = append_cosine(table, ["a", "b"], {"a": .25, "b": .75}, {"a": 1., "b": 0.})
        for name in table.column_names:
            self.assertTrue(output[name].equals(table[name]))
        self.assertEqual(str(output.schema.field("compact_dense_score").type), "float")
        self.assertEqual(output["compact_dense_known"].to_pylist(), [1., 1.])

    def test_wrong_primary_query_encoding_and_invented_secondary_ids_fail_closed(self):
        table = self.table()
        for scores, control in (({"a": .2}, None), ({"a": .2, "b": .1, "c": .3}, None),
                                ({"a": float("nan"), "b": 0}, None),
                                ({"a": .2, "b": 0}, {"a": .9, "b": 0})):
            with self.assertRaises(ValueError):
                append_cosine(table, ["a", "b"], scores, control)

    def test_one_query_vector_is_shared_by_both_views_without_neighbor_search(self):
        calls = []
        class Model:
            prompts = {"query": "instruction"}
            def encode(self, texts, **kwargs):
                calls.append((texts, kwargs))
                return np.asarray([[2., 0.], [0., 2.]], dtype=np.float32)
        class Index:
            def __init__(self, rows):
                self.rows = np.asarray(rows, dtype=np.float32)
            def get_items(self, positions):
                return self.rows[positions]
            def knn_query(self, *_args, **_kwargs):
                raise AssertionError("No neighbor retrieval allowed")
            def add_items(self, *_args, **_kwargs):
                raise AssertionError("No corpus encoding/insertion allowed")
        dense = SimpleNamespace(ids=["a", "b"], dimension=2, batch_size=64,
            model_name="local", model=Model(), index=Index([[0, 1], [1, 0]]))
        primary = Index([[1, 0], [0, 1]])
        scores = score_both_document_indexes(dense, primary,
            [{"search_query": "First"}, {"search_query": "Second"}], [["a", "b"], ["b", "a"]])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], ["запрос: first", "запрос: second"])
        self.assertEqual(calls[0][1]["prompt_name"], "query")
        self.assertEqual(scores[0], ({"a": 1., "b": 0.}, {"a": 0., "b": 1.}))
        self.assertEqual(scores[1], ({"b": 1., "a": 0.}, {"b": 0., "a": 1.}))

    def test_audit_and_nonpositive_batches_have_no_cli_path(self):
        base = ["--mode", "train", "--protocol-dir", "protocol", "--reference-model", "model",
            "--source", "train", "--output", "new", "--compact-cache", "compact",
            "--current-cache", "current", "--encoder-base-reference", "base.json"]
        with patch("augment_secondary_dense_v9.derive") as derive:
            for extra in (["--mode", "audit"], ["--query-batch-size", "0"]):
                with self.assertRaises(SystemExit):
                    main(base + extra)
            derive.assert_not_called()


if __name__ == "__main__":
    unittest.main()
