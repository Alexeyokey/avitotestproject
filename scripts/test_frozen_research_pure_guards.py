"""Low-memory AST tests: actual guard bodies, synthetic files/GT, no ML imports.

This deliberately avoids importing the pipeline modules: their transitive
imports initialize sklearn. No real Audit file, model, cache or index is read.
The full integration fixtures remain in test_frozen_research_regression.py.
"""
import argparse
import ast
from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def bodies(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError("Requested real guard bodies are absent")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    namespace.setdefault("__file__", str(path))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def literals(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    result[target.id] = ast.literal_eval(node.value)
    if set(result) != set(names):
        raise AssertionError("Requested real constants are absent")
    return result


def load_guards():
    common = dict(argparse=argparse, defaultdict=defaultdict, hashlib=hashlib,
                  json=json, math=math, Path=Path, random=random, re=re)
    learning = ROOT / "scripts/audit_reranker_learning_curve.py"
    simple = bodies(learning, ("normalize_query_text", "top_indices"), dict(common))
    evaluator = ROOT / "scripts/evaluate_fresh_validation.py"
    ev_ns = {**common, **literals(evaluator, ("SLICE_DENOMINATORS",)),
             "normalize_query_text": simple["normalize_query_text"], "top_indices": simple["top_indices"]}
    bodies(evaluator, ("aggregate", "score_cached_group"), ev_ns)
    ev = SimpleNamespace(sha256_file=sha256_file, canonical_sha256=canonical_sha256,
                         normalize_query_text=simple["normalize_query_text"],
                         aggregate=ev_ns["aggregate"], score_cached_group=ev_ns["score_cached_group"])
    comparison = bodies(ROOT / "scripts/evaluate_research_reranker.py",
        ("aligned_rows", "paired_pipeline_bootstrap"), {**common, "EVAL": ev})
    freeze_path = ROOT / "scripts/freeze_research_selection.py"
    freeze = {**common, "EVAL": ev, **literals(freeze_path, ("REGRESSION_POLICY", "REQUIRED_FILES"))}
    def check_sources(snapshot):
        if any(not Path(path).is_file() or sha256_file(path) != value for path, value in snapshot.items()):
            raise ValueError("Research feature/retrieval producer source changed")
    freeze["check_sources"] = check_sources
    bodies(freeze_path, ("identity", "verify_files", "regression_source_paths", "check_regression_sources",
        "select_dense_index", "verify_dense_index", "validate_tune_report", "main"), freeze)
    cache_path = ROOT / "scripts/cache_frozen_research_audit.py"
    cache = {**common, "EVAL": ev, "digest": sha256_file,
             "REGRESSION_POLICY": freeze["REGRESSION_POLICY"], "__file__": str(cache_path)}
    bodies(cache_path, ("check_regression_cache", "check_loaded_dense_index", "produce"), cache)
    audit = {**common, "EVAL": ev, "aligned_rows": comparison["aligned_rows"],
             "paired_pipeline_bootstrap": comparison["paired_pipeline_bootstrap"]}
    bodies(ROOT / "scripts/evaluate_frozen_research_audit.py",
        ("assert_targets_in_public_train", "source_summary", "compare_rows", "evaluate"), audit)
    dense_path = ROOT / "src/avito_candidates/dense.py"
    dense = {**common, "normalize": simple["normalize_query_text"],
             **literals(dense_path, ("_CACHE_VERSION",))}
    bodies(dense_path, ("_first_words", "item_text", "document_text", "_fingerprint"), dense)
    return freeze, cache, audit, comparison, ev, dense


class PureRegressionGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.freeze, cls.cache, cls.audit, cls.comparison, cls.ev, cls.dense = load_guards()

    def index_fixture(self, root):
        directory = root / "dense"
        directory.mkdir()
        documents = [{"item_id": "0000000000000001", "item_title_raw": "Стол"},
                     {"item_id": "0000000000000002", "item_title_raw": "Кресло"}]
        config = {"embedding_model": "local-encoder", "embedding_max_length": 128,
                  "embedding_params_words": 40, "embedding_description_words": 48}
        signature = "frozen-signature"
        fingerprint = self.dense["_fingerprint"](documents, config["embedding_model"],
                                                  128, 40, 48, signature)
        binary, meta_path = directory / (fingerprint + ".bin"), directory / (fingerprint + ".json")
        binary.write_bytes(b"synthetic completed index")
        metadata = {"version": self.dense["_CACHE_VERSION"], "model": "local-encoder",
            "max_seq_length": 128, "params_words": 40, "description_words": 48,
            "dimension": 1024, "items": 2, "model_signature": signature}
        meta_path.write_text(json.dumps(metadata))
        return directory, documents, config, signature, fingerprint, metadata

    def select(self, directory, documents, config):
        return self.freeze["select_dense_index"](directory, config, "synthetic-catalog",
            fingerprint=self.dense["_fingerprint"], records=lambda _path: iter(documents),
            cache_version=self.dense["_CACHE_VERSION"])

    def test_exact_source_closure_contains_scoring_denominators_and_supplemental(self):
        paths = self.freeze["regression_source_paths"]()
        names = {path.name for path in paths}
        self.assertTrue({"evaluate_fresh_validation.py", "audit_reranker_learning_curve.py",
                         "build_fresh_validation_protocol.py", "supplemental.py", "core.py"} <= names)
        snapshot = {str(path.resolve()): sha256_file(path) for path in paths}
        self.freeze["check_regression_sources"](snapshot)
        omitted = dict(snapshot)
        del omitted[str(next(path for path in paths if path.name == "evaluate_fresh_validation.py").resolve())]
        with self.assertRaisesRegex(ValueError, "source closure"):
            self.freeze["check_regression_sources"](omitted)

    def test_source_mutation_is_detected_without_touching_real_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.py"
            path.write_text("old")
            snapshot = {str(path): sha256_file(path)}
            path.write_text("new")
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.freeze["check_sources"](snapshot)

    def test_dense_selection_ignores_unrelated_and_partial_indexes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, documents, config, _signature, fingerprint, metadata = self.index_fixture(Path(temporary))
            (directory / ("a" * 20 + ".json")).write_text(json.dumps({**metadata, "model": "unrelated"}))
            (directory / ("a" * 20 + ".bin")).write_bytes(b"unrelated")
            (directory / (fingerprint + ".partial.json")).write_text(json.dumps(metadata))
            result = self.select(directory, documents, config)
            self.assertEqual(result["fingerprint"], fingerprint)
            self.assertEqual(result["binary"]["sha256"], sha256_file(directory / (fingerprint + ".bin")))

    def test_dense_selection_binds_actual_ordered_document_texts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, documents, config, *_rest = self.index_fixture(Path(temporary))
            for altered in (list(reversed(documents)), [{**documents[0], "item_title_raw": "Диван"}, documents[1]]):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    self.select(directory, altered, config)

    def test_multiple_matching_saved_signatures_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, documents, config, _signature, _fingerprint, metadata = self.index_fixture(Path(temporary))
            other = self.dense["_fingerprint"](documents, "local-encoder", 128, 40, 48, "other-signature")
            (directory / (other + ".bin")).write_bytes(b"other index")
            (directory / (other + ".json")).write_text(json.dumps({**metadata, "model_signature": "other-signature"}))
            with self.assertRaisesRegex(ValueError, "exactly one"):
                self.select(directory, documents, config)

    def test_selected_index_path_and_binary_bytes_are_sealed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, documents, config, *_rest = self.index_fixture(root)
            index = self.select(directory, documents, config)
            selection = {"dense_index": index, "files": {"dense_index_binary": index["binary"],
                                                          "dense_index_metadata": index["metadata"]}}
            self.freeze["verify_dense_index"](selection, directory)
            with self.assertRaisesRegex(ValueError, "path differs"):
                self.freeze["verify_dense_index"](selection, root / "other")
            Path(index["binary"]["path"]).write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "Frozen input"):
                self.freeze["verify_dense_index"](selection, directory)

    def test_runtime_model_signature_must_match_selected_index_before_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, documents, config, signature, _fingerprint, _metadata = self.index_fixture(Path(temporary))
            selection = {"dense_index": self.select(directory, documents, config)}
            dense = SimpleNamespace(model_name="local-encoder", max_seq_length=128,
                params_words=40, description_words=48, model_signature=signature, document_format="current")
            retriever = SimpleNamespace(dense=dense)
            self.cache["check_loaded_dense_index"](retriever, documents, selection, fingerprint=self.dense["_fingerprint"])
            dense.model_signature = "different runtime signature"
            with self.assertRaisesRegex(ValueError, "Restored encoder/index"):
                self.cache["check_loaded_dense_index"](retriever, documents, selection, fingerprint=self.dense["_fingerprint"])

    def test_cache_requires_frozen_producer_index_and_integer_group_ids(self):
        # Fake the tiny Parquet header; this test opens no actual cache/labels.
        pa, pq = ModuleType("pyarrow"), ModuleType("pyarrow.parquet")
        pa.__path__ = []
        pa.parquet = pq
        names = ["rrf_score", "baseline_top50", "group_id", "item_id", "label", "positive_local", "positive_seen"]
        pq.ParquetFile = lambda _path: SimpleNamespace(metadata=SimpleNamespace(num_rows=1),
            schema_arrow=SimpleNamespace(names=names, field=lambda _name: SimpleNamespace(type="float")))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.parquet"
            path.write_bytes(b"synthetic header; not Audit data")
            producer = self.cache["__file__"]
            selection = {"files": {"protocol_manifest": {"sha256": "protocol"}, "new_model": {"sha256": "model"}},
                "selected_stage": 100, "selection_sha256": "selection", "dense_index": {"fingerprint": "index"},
                "source_sha256": {str(Path(producer).resolve()): "producer"}}
            metadata = {"cache_version": 2, "role": "validation", "fresh_partition": "audit",
                "fresh_validation_protocol_sha256": "protocol", "model_sha256": "model",
                "historical_holdout_reused": True, "full_truth_preserved": True,
                "frozen_regression_cache_version": 1, "previous_baseline_audit_observed": True,
                "independent_fresh_test": False, "purpose": "frozen regression", "frozen_selected_stage": 100,
                "regression_policy": self.freeze["REGRESSION_POLICY"], "regression_selection_sha256": "selection",
                "regression_selection_file_sha256": "selection-file", "regression_dense_index": selection["dense_index"],
                "producer_sha256": "producer", "feature_names": ["rrf_score", "baseline_top50"],
                "groups": [{"group_id": 0, "pool_size": 1}], "rows": 1,
                "cache_parquet_sha256": sha256_file(path)}
            with patch.dict(sys.modules, {"pyarrow": pa, "pyarrow.parquet": pq}):
                self.cache["check_regression_cache"](path, metadata, selection, "selection-file")
                for changed in ({**metadata, "producer_sha256": "different"},
                                {**metadata, "regression_dense_index": {"fingerprint": "other"}},
                                {**metadata, "groups": [{"group_id": 0.0, "pool_size": 1}]},
                                {**metadata, "groups": [None]},
                                {**metadata, "independent_fresh_test": True}):
                    with self.assertRaises(ValueError):
                        self.cache["check_regression_cache"](path, changed, selection, "selection-file")

    def row(self, identity, text, available, full, new_hits, old_hits):
        return ({"group_identity": identity, "query_text": text, "query_key": [text, "1", "0", "", "114"],
            "representative": True, "available_relevant_count": available, "full_relevant_count": full,
            "s75_hits": new_hits, "pool_hits": new_hits, "s75_train_source_only_hits": new_hits},
            {"group_identity": identity, "query_text": text, "query_key": [text, "1", "0", "", "114"],
            "representative": True, "available_relevant_count": available, "full_relevant_count": full,
            "s150_hits": old_hits, "pool_hits": old_hits, "s150_train_source_only_hits": old_hits})

    def test_paired_bootstrap_averages_text_clusters_and_preserves_zero_fulltruth(self):
        pairs = [self.row("a1", "a", 1, 1, 1, 0), self.row("a2", "a", 1, 1, 0, 0),
                 self.row("b", "b", 1, 1, 0, 0), self.row("c", "c", 0, 1, 0, 0)]
        new, old = [p[0] for p in pairs], [p[1] for p in pairs]
        result = self.audit["compare_rows"](new, old, 75, 100)
        self.assertEqual(result["conditional_available"]["recall_at_50"]["query_text_clusters"], 2)
        self.assertEqual(result["conditional_available"]["recall_at_50"]["delta"], .25)
        self.assertEqual(result["full_truth"]["recall_at_50"]["query_text_clusters"], 3)
        self.assertAlmostEqual(result["full_truth"]["recall_at_50"]["delta"], 1 / 6)
        self.assertEqual(result, self.audit["compare_rows"](list(reversed(new)), old, 75, 100))

    def test_paired_denominator_mutation_and_impossible_numerator_fail(self):
        left, right = self.row("a", "q", 2, 3, 1, 0)
        right["available_relevant_count"] = 1
        with self.assertRaises(ValueError):
            self.audit["compare_rows"]([left], [right], 75, 10)
        left, right = self.row("a", "q", 2, 3, 3, 0)
        with self.assertRaises(ValueError):
            self.audit["compare_rows"]([left], [right], 75, 10)

    def test_source_control_keeps_denominators_and_zeroavailability_contexts(self):
        rows = []
        for text, available, hits in (("a", 1, 1), ("b", 0, 0)):
            rows.append({"query_text": text, "representative": True, "available_relevant_count": available,
                "full_relevant_count": 2, "pool_hits": hits, "train_source_only_pool_hits": hits,
                "train_source_only_pool_size": 10, "s75_train_source_only_hits": hits,
                "train_source_only_baseline_hits": 0})
        result = self.audit["source_summary"](rows, 75)
        self.assertEqual(result["conditional_available"]["reranker"]["text_macro_recall_at_50"], 1.)
        self.assertEqual(result["full_truth"]["reranker"]["text_macro_recall_at_50"], .25)
        self.assertEqual(result["full_truth"]["reranker"]["groups"], 2)
        rows[0]["train_source_only_pool_hits"] = 0
        with self.assertRaisesRegex(ValueError, "removed a known positive"):
            self.audit["source_summary"](rows, 75)

    def test_source_targets_outside_global_public_train_are_rejected_not_dropped(self):
        labels = {"q": {"relevant_item_ids": ["a", "b"]}}
        self.audit["assert_targets_in_public_train"](labels, {"a", "b"})
        with self.assertRaises(ValueError):
            self.audit["assert_targets_in_public_train"](labels, {"a"})
        self.assertEqual(labels["q"]["relevant_item_ids"], ["a", "b"])

    def test_drivers_fail_selection_before_any_ml_or_target_read(self):
        def unsealed(_path):
            raise ValueError("unsealed")
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(selection=Path(temporary) / "missing", output_dir=Path(temporary))
            for namespace, function in ((self.cache, "produce"), (self.audit, "evaluate")):
                with patch.dict(namespace, verify_frozen_selection=unsealed, require_offline_environment=lambda: None):
                    with self.assertRaisesRegex(ValueError, "unsealed"):
                        namespace[function](args)

    def test_freeze_cli_requires_explicit_dense_cache_and_only_reported_stages(self):
        args = ["--protocol-dir", "protocol", "--tune-report", "report", "--old-audit-cache-dir", "old",
                "--stage", "100", "--reason", "Tune only", "--output", "selection"]
        calls = []
        with patch.dict(self.freeze, freeze=lambda value: calls.append(value)):
            with self.assertRaises(SystemExit):
                self.freeze["main"](args)
            self.assertFalse(calls)
            self.freeze["main"](args + ["--dense-cache", "index"])
            self.assertEqual(calls[0].dense_cache, Path("index"))

    def test_no_ml_package_was_imported_by_pure_guards(self):
        self.assertFalse({"torch", "transformers", "sentence_transformers", "sklearn", "lightgbm", "peft"} & set(sys.modules))


if __name__ == "__main__":
    unittest.main()
