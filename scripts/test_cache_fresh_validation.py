"""Tiny producer fixtures: no GPU, live model, index or project labels."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pyarrow.parquet as pq

import cache_fresh_validation as FRESH
import cache_reranker_audit as CACHE
import evaluate_fresh_validation as EVAL
from build_fresh_validation_protocol import (
    REPRESENTATIVE_SALT, canonical_query_key_json, sha256_file,
)


A, B, C = "a" * 16, "b" * 16, "c" * 16
FIRST = ("available", "city", "0", "", "category")
ZERO = ("unavailable", "other", "0", "", "category")


class FakeBuilder:
    def __init__(self, full_dense=False):
        self.feature_names = ("signal", "full_dense_score") if full_dense else ("signal",)
        self.config = {"method": "hybrid", "candidate_k": 2, "location_bonus": 0.02}
        self.leave_query_out = False
        self.leave_geography_query_out = False

    def transform(self, query, details, baseline):
        return np.tile(np.array([0.25, 0.75][:len(self.feature_names)], dtype=np.float32),
                       (len(details["candidate_pool"]), 1))


class CacheProducerTests(unittest.TestCase):
    def fixture(self, directory, full_dense=False):
        root = Path(directory)
        builder = FakeBuilder(full_dense=full_dense)
        args = {
            "truth": {FIRST: {A}, ZERO: set()},
            "representatives": {FIRST: dict(zip(FRESH.SEARCH_FIELDS, FIRST)),
                                ZERO: dict(zip(FRESH.SEARCH_FIELDS, ZERO))},
            "keys": [FIRST, ZERO], "builder": builder,
            "retriever": SimpleNamespace(dense=SimpleNamespace(ids=[A, B])),
            "items_by_id": {A: {"item_location_id": "city", "item_microcat_id": "m1"},
                            B: {"item_location_id": "other", "item_microcat_id": "m2"}},
            "fit_items": {A}, "benchmark_strata": [],
            "model_metadata": {"feature_version": 4, "corpus_fingerprint": "corpus",
                               "split": "queries", "seed": 42, "validation_fraction": 0.2,
                               "training_prior_policy": {"microcategory": "leave-query-text-out",
                                                         "geography": "leave-query-text-out"}},
            "model_reference": root / "model.joblib", "model_hash": "model-sha",
            "batch_size": 2, "role": "validation", "max_pool_size": 998,
            "partitioner": lambda _text: "tune", "historical_holdout_reused": False,
            "extra_group_metadata": {
                FIRST: {"group_identity": "1" * 64, "full_relevant_count": 2,
                        "available_relevant_count": 1, "representative": True},
                ZERO: {"group_identity": "2" * 64, "full_relevant_count": 1,
                       "available_relevant_count": 0, "representative": True}},
            "extra_metadata": {"fresh_validation_protocol_sha256": "protocol-sha",
                               "fresh_partition": "tune", "full_truth_preserved": True},
            "allow_empty_relevance": True,
        }
        return root / "tune.parquet", args

    def build(self, path, args):
        def retrieve(_retriever, queries, _method, _k):
            return [([A, B], {"candidate_pool": [A, B]}) for _query in queries]

        def full_scores(_dense, _queries, results, _ids):
            return [np.array([0.9, 0.1], dtype=np.float32) for _ in results], 0.0

        with mock.patch.object(CACHE, "retrieve_details", side_effect=retrieve), \
                mock.patch.object(CACHE, "dense_scores_for_full_pools", side_effect=full_scores):
            return CACHE.build_cache(path, **args)

    def test_zero_available_groups_and_full_denominators_remain_in_cache(self):
        with TemporaryDirectory() as directory:
            path, args = self.fixture(directory)
            result = self.build(path, args)
            self.assertEqual(len(result["groups"]), 2)
            zero = result["groups"][1]
            self.assertEqual(zero["relevant_count"], 0)
            self.assertEqual(zero["available_relevant_count"], 0)
            self.assertEqual(zero["full_relevant_count"], 1)
            self.assertTrue(zero["representative"])
            self.assertEqual(zero["group_identity"], "2" * 64)
            self.assertEqual(zero["partition"], "tune")
            self.assertFalse(result["historical_holdout_reused"])
            self.assertEqual(result["available_groups"], 1)
            self.assertEqual(result["pool_recall"], 1.0)
            self.assertEqual(result["baseline_recall_at_50"], 1.0)
            self.assertEqual(result["cache_parquet_sha256"], CACHE.digest(path))
            table = pq.read_table(path).to_pydict()
            self.assertEqual(table["group_id"], [0, 0, 1, 1])
            self.assertEqual(table["label"], [1, 0, 0, 0])
            self.assertEqual(table["positive_local"], [True, False, False, False])
            self.assertEqual(table["positive_seen"], [True, False, False, False])
            self.assertNotIn("full_relevant_count", table)  # Metadata, never predictor features.
            self.assertFalse(path.with_suffix(".partial.parquet").exists())

    def test_all_unavailable_cohort_has_explicit_null_conditional_summaries(self):
        with TemporaryDirectory() as directory:
            path, args = self.fixture(directory)
            args["truth"][FIRST] = set()
            args["extra_group_metadata"][FIRST]["available_relevant_count"] = 0
            result = self.build(path, args)
            self.assertEqual(result["available_groups"], 0)
            self.assertIsNone(result["pool_recall"])
            self.assertIsNone(result["baseline_recall_at_50"])
            self.assertEqual(pq.read_table(path).num_rows, 4)

    def test_empty_truth_requires_opt_in_and_metadata_cannot_replace_outcomes(self):
        with TemporaryDirectory() as directory:
            path, args = self.fixture(directory)
            args["allow_empty_relevance"] = False
            with self.assertRaisesRegex(ValueError, "nonempty relevance"):
                self.build(path, args)
            self.assertFalse(path.exists())
            args["allow_empty_relevance"] = True
            args["extra_group_metadata"][FIRST]["baseline_hits"] = 100
            with self.assertRaisesRegex(ValueError, "cannot overwrite retrieval outcomes"):
                self.build(path, args)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".json").exists())

    def test_full_dense_feature_is_not_duplicated_or_replaced(self):
        with TemporaryDirectory() as directory:
            path, args = self.fixture(directory, full_dense=True)
            self.build(path, args)
            table = pq.read_table(path)
            self.assertEqual(table.column_names.count("full_dense_score"), 1)
            np.testing.assert_array_equal(table["full_dense_score"].to_numpy(),
                                          np.full(4, 0.75, dtype=np.float32))

    def test_verified_resume_avoids_retrieval_and_rejects_stale_identity(self):
        with TemporaryDirectory() as directory:
            path, args = self.fixture(directory)
            original = self.build(path, args)
            with mock.patch.object(CACHE, "retrieve_details", side_effect=AssertionError("must reuse")):
                self.assertEqual(CACHE.build_cache(path, **args), original)
            for name, mutate in (
                ("model", lambda value: value.update(model_hash="another-model")),
                ("protocol", lambda value: value["extra_metadata"].update(
                    fresh_validation_protocol_sha256="another-protocol")),
                ("retrieval", lambda value: value["builder"].config.update(candidate_k=3)),
                ("corpus", lambda value: value["model_metadata"].update(corpus_fingerprint="another")),
            ):
                with self.subTest(identity=name):
                    modified = deepcopy(args)
                    mutate(modified)
                    with self.assertRaisesRegex(ValueError, "differs from this audit"):
                        self.build(path, modified)

    def test_resume_rejects_changed_parquet_unsealed_and_incomplete_pairs(self):
        with TemporaryDirectory() as directory:
            path, args = self.fixture(directory)
            self.build(path, args)
            original = path.read_bytes()
            path.write_bytes(original + b"changed bytes")
            with self.assertRaisesRegex(ValueError, "bytes changed or are unsealed"):
                self.build(path, args)
            path.write_bytes(original)
            metadata_path = path.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text())
            del metadata["cache_parquet_sha256"]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bytes changed or are unsealed"):
                self.build(path, args)
            metadata_path.unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete cache already exists"):
                self.build(path, args)


class ProtocolAndAuditOpeningTests(unittest.TestCase):
    def protocol_fixture(self, root):
        key = list(ZERO)
        key_json = canonical_query_key_json(key)
        group = {"group_id": hashlib.sha256(key_json.encode()).hexdigest(),
                 "query_key": key, "query_key_json": key_json, "query_text": key[0],
                 "representative": True, "relevant_item_ids": [C], "available_item_ids": [],
                 "full_relevant_count": 1, "available_relevant_count": 0}
        names = ("labels_fit", "labels_tune", "labels_audit", "fit_texts", "tune_texts",
                 "audit_texts", "historical_excluded_texts", "train", "benchmark_items",
                 "benchmark_queries", "catalog_item_ids")
        files = {}
        for name in names:
            path = root / (name + ".json")
            if name.startswith("labels_"):
                value = {"partition": name[7:], "groups": [group]}
            elif name.endswith("texts"):
                value = {"texts": [key[0]]}
            else:
                value = {"opaque": True}
            path.write_text(json.dumps(value), encoding="utf-8")
            files[name] = {"path": path.name, "sha256": sha256_file(path)}
        manifest = {"protocol_version": 1, "complete": True, "sealed_audit": True,
                    "split": {"representative_salt": REPRESENTATIVE_SALT},
                    "files": files}
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest, group

    def test_read_protocol_keeps_zero_available_groups_and_checks_completion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, group = self.protocol_fixture(root)
            loaded, groups, protocol_sha = FRESH.read_protocol(root, "tune")
            self.assertEqual(loaded, manifest)
            self.assertEqual(groups, [group])
            self.assertEqual(protocol_sha, CACHE.digest(root / "manifest.json"))
            for field, wrong in (("protocol_version", 99), ("sealed_audit", False), ("complete", False)):
                with self.subTest(field=field):
                    changed = {**manifest, field: wrong}
                    (root / "manifest.json").write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "completed sealed"):
                        FRESH.read_protocol(root, "tune")
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "INCOMPLETE").write_text("not finished", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                FRESH.read_protocol(root, "tune")

    def test_read_protocol_rejects_changed_inventory_bytes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.protocol_fixture(root)
            (root / "benchmark_items.json").write_text("changed inventory", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                FRESH.read_protocol(root, "tune")

    def test_minimal_fabricated_audit_json_is_rejected_before_protocol_labels(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ("manifest.json", "model.joblib", "train.parquet"):
                (root / filename).write_bytes(b"opaque bytes")
            selection = root / "selection.json"
            selection.write_text(json.dumps({"selected_stage": 50, "protocol_sha256": "claimed",
                                             "model_sha256": "claimed"}), encoding="utf-8")
            with mock.patch.object(FRESH, "read_protocol") as read_protocol:
                with self.assertRaisesRegex(ValueError, "selection seal"):
                    FRESH.main(["--phase", "audit", "--protocol-dir", str(root),
                                "--model", str(root / "model.joblib"),
                                "--train-data", str(root / "train.parquet"),
                                "--selection", str(selection), "--dense-cache", str(root / "dense"),
                                "--output-dir", str(root / "cache")])
                read_protocol.assert_not_called()
            self.assertFalse((root / "cache").exists())

    def test_real_verified_selection_seal_is_accepted_and_tamper_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            reports = {str(stage): {"conditional_available": {"reranker": {
                "text_macro_recall_at_50": 0.8 if stage == 50 else 0.7}}}
                for stage in EVAL.TUNE_STAGES}
            report_path = root / "tune-report.json"
            report_path.write_text(json.dumps({"stages": reports}), encoding="utf-8")
            paths = {"parquet": root / "tune.parquet", "metadata": root / "tune.json"}
            for path in paths.values():
                path.write_bytes(b"unchanged tune artifact")
            hashes = {name: EVAL.sha256_file(path) for name, path in paths.items()}
            sealed = EVAL.make_selection("protocol", "model", hashes, EVAL.sha256_file(report_path),
                                         50, reports, "train", paths)
            selection = root / "selection.json"
            selection.write_text(json.dumps(sealed), encoding="utf-8")
            self.assertEqual(FRESH.verify_audit_seal(selection, "protocol", "model", "train"), sealed)
            paths["metadata"].write_bytes(b"tampered tune metadata")
            with self.assertRaisesRegex(ValueError, "tune cache"):
                FRESH.verify_audit_seal(selection, "protocol", "model", "train")

    def audit_main_fixture(self, root):
        self.protocol_fixture(root)
        model, training, selection = (root / name for name in
                                      ("model.joblib", "training.parquet", "selection.json"))
        for path in (model, training, selection):
            path.write_bytes(b"opaque ordering fixture")
        artifact = {"metadata": {"split": "queries", "seed": 42, "validation_fraction": .2,
                    "training_prior_policy": {"microcategory": "leave-query-text-out",
                                              "geography": "leave-query-text-out"},
                    "retrieval_config": {"method": "hybrid", "candidate_k": 449}}}
        argv = ["--phase", "audit", "--protocol-dir", str(root), "--model", str(model),
                "--train-data", str(training), "--selection", str(selection),
                "--dense-cache", str(root / "dense"), "--output-dir", str(root / "cache")]
        return artifact, argv

    def test_audit_labels_stay_closed_until_model_encoder_training_and_corpus_checks_pass(self):
        for failure in ("metadata", "encoder", "training", "corpus"):
            with self.subTest(failure=failure), TemporaryDirectory() as directory:
                root = Path(directory)
                artifact, argv = self.audit_main_fixture(root)
                if failure == "metadata":
                    artifact["metadata"]["split"] = "all"
                error = ValueError("invalid " + failure)
                with mock.patch.object(FRESH, "verify_audit_seal", return_value={}), \
                        mock.patch("joblib.load", return_value=artifact), \
                        mock.patch.object(EVAL, "check_encoder_identity",
                                          side_effect=error if failure == "encoder" else None), \
                        mock.patch.object(EVAL, "check_training_data",
                                          side_effect=error if failure == "training" else None), \
                        mock.patch.object(FRESH, "read", return_value=[]), \
                        mock.patch.object(FRESH, "Reranker",
                                          side_effect=error if failure == "corpus" else None), \
                        mock.patch.object(FRESH, "read_protocol") as read_protocol:
                    with self.assertRaises(ValueError):
                        FRESH.main(argv)
                    read_protocol.assert_not_called()
                self.assertFalse((root / "cache").exists())

    def test_audit_opens_labels_only_after_successful_preflight_in_exact_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, argv = self.audit_main_fixture(root)
            order = []
            def record(name, value=None):
                def call(*_args, **_kwargs):
                    order.append(name)
                    return value
                return call
            def open_labels(*_args, **kwargs):
                order.append("labels")
                self.assertEqual(kwargs["expected_protocol_sha"], CACHE.digest(root / "manifest.json"))
                raise RuntimeError("stop after proven label opening")
            with mock.patch.object(FRESH, "verify_audit_seal", side_effect=record("seal")), \
                    mock.patch("joblib.load", side_effect=record("model", artifact)), \
                    mock.patch.object(EVAL, "check_encoder_identity", side_effect=record("encoder")), \
                    mock.patch.object(EVAL, "check_training_data", side_effect=record("training")), \
                    mock.patch.object(FRESH, "read", side_effect=record("catalog", [])), \
                    mock.patch.object(FRESH, "Reranker", side_effect=record("corpus")), \
                    mock.patch.object(FRESH, "read_protocol", side_effect=open_labels):
                with self.assertRaisesRegex(RuntimeError, "proven label opening"):
                    FRESH.main(argv)
            self.assertEqual(order, ["seal", "model", "encoder", "training", "catalog", "corpus", "labels"])
            self.assertFalse((root / "cache").exists())

    def test_changed_protocol_cannot_open_label_partition(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.protocol_fixture(root)
            with mock.patch.object(EVAL, "read_phase_labels") as read_labels:
                with self.assertRaisesRegex(ValueError, "changed before opening"):
                    FRESH.read_protocol(root, "audit", expected_protocol_sha="wrong sealed digest")
                read_labels.assert_not_called()


if __name__ == "__main__":
    unittest.main()
