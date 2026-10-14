"""Focused guards for fit-only geography and saved-vector alignment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from avito_candidates.core import SEARCH_FIELDS
from prepare_geo_dense_reranker import (GEO_CONFIG, PreencodedFitRetriever,
    group_id, selected_fit_keys, validate_geo_config, verify_fit_vectors)
import evaluate_fresh_validation as EVAL


def query(text="ремонт", location="10"):
    return dict(zip(SEARCH_FIELDS, (text, location, "", "", "")))


class SelectionTest(unittest.TestCase):
    def test_exact_original_hash_order_and_nonfit_rejection(self):
        keys = [(name, "10", "", "", "") for name in ("а", "б", "в")]
        truth = {key: {"item"} for key in keys}
        reps = {key: dict(zip(SEARCH_FIELDS, key)) for key in keys}
        expected = sorted(keys, key=lambda key: hashlib.sha256(repr(key).encode()).digest())[:2]
        self.assertEqual(selected_fit_keys(truth, reps, {"а", "б", "в"}, 2), expected)
        with self.assertRaisesRegex(ValueError, "non-fit"):
            selected_fit_keys(truth, reps, {"а", "б"}, 2)

    def test_identity_and_policy_cannot_silently_reuse_old_retriever(self):
        key = ("ремонт", "10", "", "", "")
        self.assertEqual(group_id(key), hashlib.sha256(
            json.dumps(list(key), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest())
        self.assertEqual(validate_geo_config(GEO_CONFIG), GEO_CONFIG)
        with self.assertRaises(ValueError):
            validate_geo_config({})
        with self.assertRaises(ValueError):
            validate_geo_config({**GEO_CONFIG, "geo_dense_weight": .02})


class AdapterTest(unittest.TestCase):
    def make_adapter(self, pool=None):
        ids = ["a", "b"]
        keys = [("ремонт", "10", "", "", "")]
        arrays = {"queries": np.asarray([[1., 0.]], dtype=np.float32),
                  "ann998": np.asarray([[0, 1]]),
                  "ann998-cosines": np.asarray([[.8, .2]], dtype=np.float32)}
        calls = []
        def retrieve(cosine, location, **kwargs):
            calls.append((location, kwargs))
            return ["b", "a"], {"a": float(cosine[0]), "b": float(cosine[1])}
        def compose(q, h, shared, dense, cosine, mapping, weight):
            self.assertEqual(shared, "shared")
            self.assertEqual(dense[0], ["b", "a"])
            # Geography changes ranks; numerical dense features stay cosine.
            self.assertAlmostEqual(dense[1]["a"], .8)
            self.assertEqual(mapping, {"a": 0, "b": 1})
            return ["b", "a"], {"candidate_pool": ids if pool is None else pool}
        hybrid = SimpleNamespace(bm25=SimpleNamespace(geo_leave_query_out=False))
        adapter = PreencodedFitRetriever(hybrid, None, SimpleNamespace(retrieve=retrieve),
            SimpleNamespace(score=lambda q: np.tile(np.asarray([[.8, .2]], dtype=np.float32), (len(q), 1))),
            arrays, keys, ids, micro_weight=.25,
            shared_fn=lambda q, h, m, supplemental: "shared", compose_fn=compose,
            ann_fn=lambda labels, values, item_ids: ([item_ids[int(x)] for x in labels], values))
        return adapter, calls

    def test_all_training_geography_excludes_own_normalized_text(self):
        adapter, calls = self.make_adapter()
        self.assertTrue(adapter.hybrid.bm25.geo_leave_query_out)
        adapter.retrieve_batch_with_diagnostics([query(" РЕМОНТ ")])
        self.assertEqual(calls, [("10", {"limit": 449, "weight": .05,
                                        "exclude_query_text": "ремонт"})])
        self.assertEqual(adapter.cursor, 1)
        self.assertEqual(adapter.maximum_pool, 2)

    def test_mismatched_preencoded_query_order_rejected(self):
        adapter, calls = self.make_adapter()
        with self.assertRaisesRegex(ValueError, "order"):
            adapter.retrieve_batch_with_diagnostics([query("переезд")])
        self.assertEqual(calls, [])

    def test_duplicate_and_overbudget_pools_rejected(self):
        for pool in (["a", "a"], [str(i) for i in range(999)]):
            adapter, _ = self.make_adapter(pool)
            with self.assertRaisesRegex(ValueError, "998|duplicates"):
                adapter.retrieve_batch_with_diagnostics([query()])


class VectorManifestTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fit, self.corpus = self.root / "fit", self.root / "corpus"
        self.fit.mkdir()
        self.corpus.mkdir()
        self.keys = [("ремонт", "10", "", "", "")]
        self.ids = ["a", "b"]
        order = {"query_keys": [list(key) for key in self.keys],
                 "query_group_ids": [group_id(key) for key in self.keys],
                 "corpus_item_ids": self.ids}
        (self.fit / "order.json").write_text(json.dumps(order), encoding="utf-8")
        (self.corpus / "order.json").write_text(json.dumps({"corpus_item_ids": self.ids}), encoding="utf-8")
        np.save(self.fit / "queries.npy", np.asarray([[1., 0.]], dtype=np.float32))
        np.save(self.fit / "ann998.npy", np.asarray([[0, 1]], dtype=np.int64))
        np.save(self.fit / "ann998-cosines.npy", np.asarray([[1., 0.]], dtype=np.float32))
        np.save(self.corpus / "corpus.npy", np.eye(2, dtype=np.float32))
        self.manifest = {"complete": True, "phase": "fit", "fit_only": True,
                         "tune_labels_opened": False, "audit_labels_opened": False,
                         "inventory_reencoded": False, "protocol_sha256": "p", "metadata_sha256": "m",
                         "corpus_vectors_sha256": EVAL.sha256_file(self.corpus / "corpus.npy"),
                         "inventory_index_inputs": {"original.bin": "indexsha"},
                         "encoder_base": {"base_revision": "frozenrevision"}}
        for root, payload in ((self.fit, self.manifest), (self.corpus, dict(self.manifest))):
            payload["files"] = {path.name: {"sha256": EVAL.sha256_file(path)} for path in root.iterdir()}
            (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def verify(self):
        return verify_fit_vectors(self.fit, self.corpus, self.keys, self.ids,
                                  protocol_hash="p", metadata_hash="m")

    def test_valid_fit_vectors_have_original_shapes_and_ids(self):
        _, _, arrays = self.verify()
        self.assertEqual(arrays["corpus"].shape, (2, 2))
        self.assertEqual(arrays["queries"].dtype, np.float32)

    def test_changed_file_and_query_order_rejected(self):
        self.keys = [("переезд", "10", "", "", "")]
        with self.assertRaisesRegex(ValueError, "ordered query"):
            self.verify()
        self.keys = [("ремонт", "10", "", "", "")]
        np.save(self.fit / "queries.npy", np.asarray([[0., 1.]], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "content differs"):
            self.verify()

    def test_no_inference_or_unsealed_manifest_accepted_for_fit(self):
        for key, value in (("phase", "tune"), ("complete", False),
                           ("audit_labels_opened", True), ("inventory_reencoded", True)):
            manifest = {**self.manifest, key: value}
            (self.fit / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest"):
                self.verify()

    def test_changed_embedding_index_binding_rejected(self):
        for key, value in (("corpus_vectors_sha256", "wrong"),
                           ("inventory_index_inputs", {"original.bin": "wrong"}),
                           ("encoder_base", {"base_revision": "wrong"})):
            manifest = {**self.manifest, key: value}
            (self.fit / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "encoder and inventory"):
                self.verify()

    def test_float_ann_labels_rejected_without_silent_rounding(self):
        np.save(self.fit / "ann998.npy", np.asarray([[0., 1.]], dtype=np.float32))
        self.manifest["files"]["ann998.npy"]["sha256"] = EVAL.sha256_file(self.fit / "ann998.npy")
        (self.fit / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
