"""Fresh validation invariants use tiny synthetic fixtures, never project labels."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "fresh_validation_evaluation", Path(__file__).with_name("evaluate_fresh_validation.py"))
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)

A, B, C = "a" * 16, "b" * 16, "c" * 16


def protocol_group(text="query", location="city", available=(A,), full=(A, B), representative=True):
    key = [text, location, "0", "", "0"]
    key_json = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    return {"group_id": EVAL.hashlib.sha256(key_json.encode()).hexdigest(),
            "query_key": key, "query_key_json": key_json, "query_text": text,
            "representative": representative, "relevant_item_ids": list(full),
            "available_item_ids": list(available), "full_relevant_count": len(full),
            "available_relevant_count": len(available)}


def cached_group(source, index=0, pool_size=2, positives=1, baseline_hits=1):
    return {"group_id": index, "query_key": source["query_key"],
            "query_text": source["query_text"], "pool_size": pool_size,
            "positives_in_pool": positives, "baseline_hits": baseline_hits,
            "relevant_count": source["available_relevant_count"],
            "available_relevant_count": source["available_relevant_count"],
            "full_relevant_count": source["full_relevant_count"],
            "representative": source["representative"],
            "local_relevant_count": source["available_relevant_count"],
            "nonlocal_relevant_count": 0}


def scoring_group(source, **kwargs):
    return {**cached_group(source, **kwargs), "_available_ids": set(source["available_item_ids"]),
            "partition": "tune", "protocol_group_id": source["group_id"]}


def row(text, available, full, hits, baseline, *, representative=True, stage=25):
    return {"query_text": text, "available_relevant_count": available,
            "full_relevant_count": full, "representative": representative,
            "pool_size": 2, "pool_hits": hits, "baseline_hits": baseline,
            f"s{stage}_hits": hits}


class DenominatorTests(unittest.TestCase):
    def test_full_truth_retains_absent_targets_and_zero_available_groups(self):
        rows = [row("first", 1, 2, 1, 0), row("second", 0, 1, 0, 0)]
        report = EVAL.summarize(rows, 25)
        self.assertEqual(report["conditional_available"]["reranker"]["text_macro_recall_at_50"], 1.0)
        self.assertEqual(report["full_truth"]["reranker"]["text_macro_recall_at_50"], 0.25)
        self.assertEqual(report["coverage"]["groups_without_available_target"], 1)
        self.assertEqual(report["coverage"]["full_relevant_pairs"], 3)

    def test_text_macro_does_not_overweight_multiple_location_groups(self):
        rows = [row("frequent", 1, 1, 1, 0), row("frequent", 1, 1, 1, 0, representative=False),
                row("rare", 1, 1, 0, 0)]
        metrics = EVAL.aggregate(rows, "s25_hits", "available_relevant_count")
        self.assertEqual(metrics["text_macro_recall_at_50"], 0.5)
        self.assertAlmostEqual(metrics["group_macro_recall_at_50"], 2 / 3)
        self.assertEqual(metrics["representative_macro_recall_at_50"], 0.5)

    def test_unavailable_representative_is_not_replaced_by_an_available_variant(self):
        rows = [row("one", 0, 1, 0, 0), row("one", 1, 1, 1, 0, representative=False)]
        conditional = EVAL.aggregate(rows, "s25_hits", "available_relevant_count")
        self.assertIsNone(conditional["representative_macro_recall_at_50"])
        self.assertEqual(conditional["text_macro_recall_at_50"], 1.0)
        self.assertEqual(EVAL.aggregate(rows, "s25_hits", "full_relevant_count")["text_macro_recall_at_50"], 0.5)

    def test_targets_outside_pool_remain_in_available_denominator(self):
        source = protocol_group(available=(A, B), full=(A, B, C))
        scored = EVAL.score_cached_group(scoring_group(source), [A, C], [1, 0],
                                         [True, True], None, [True, False], [0.2, 0.1],
                                         {25: [0.9, 0.1]})
        metrics = EVAL.summarize([scored], 25)
        self.assertEqual(metrics["conditional_available"]["reranker"]["text_macro_recall_at_50"], 0.5)
        self.assertAlmostEqual(metrics["full_truth"]["reranker"]["text_macro_recall_at_50"], 1 / 3)

    def test_label_tampering_and_incomplete_pools_fail(self):
        group = scoring_group(protocol_group())
        with self.assertRaisesRegex(ValueError, "sealed protocol targets"):
            EVAL.score_cached_group(group, [A, C], [0, 1], [True, True], None,
                                    [True, False], [0.2, 0.1], {25: [0.9, 0.1]})
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            EVAL.score_cached_group(group, [A], [1], [True], None, [True], [0.2], {25: [0.9]})

    def test_local_and_filter_slices_use_metadata_denominators(self):
        source = protocol_group(available=(A, B))
        group = scoring_group(source)
        group.update({"local_relevant_count": 1, "nonlocal_relevant_count": 1,
                      "seen_relevant_count": 1, "unseen_relevant_count": 1, "has_filters": True})
        group["positives_in_pool"] = 2
        scored = EVAL.score_cached_group(group, [A, B], [1, 1], [True, False], [True, False],
                                         [True, False], [0.2, 0.1], {25: [0.9, 0.1]})
        slices = EVAL.summarize([scored], 25)["slices"]
        self.assertEqual(slices["nonlocal_positive"]["baseline_rrf"]["text_macro_recall_at_50"], 0.0)
        self.assertEqual(slices["item_unseen_in_allowed_fit"]["reranker"]["text_macro_recall_at_50"], 1.0)
        self.assertIn("has_filters=true", slices)


class CacheBindingTests(unittest.TestCase):
    def metadata(self, groups):
        return {"role": "validation", "fresh_partition": "tune", "fresh_validation_protocol_sha256": "protocol",
                "model_sha256": "model", "groups": groups}

    def test_all_unavailable_groups_must_be_present(self):
        a = protocol_group()
        b = protocol_group(text="other", available=(), full=(C,))
        groups = {x["group_id"]: x for x in (a, b)}
        cached = [cached_group(a), cached_group(b, index=1, positives=0, baseline_hits=0)]
        bound = EVAL.bind_cache_groups(self.metadata(cached), groups, "tune", "protocol", "model")
        self.assertEqual(bound[1]["available_relevant_count"], 0)
        with self.assertRaisesRegex(ValueError, "every protocol group"):
            EVAL.bind_cache_groups(self.metadata(cached[:1]), groups, "tune", "protocol", "model")

    def test_wrong_phase_and_denominator_are_rejected(self):
        source = protocol_group()
        groups = {source["group_id"]: source}
        with self.assertRaisesRegex(ValueError, "identity differs"):
            EVAL.bind_cache_groups(self.metadata([cached_group(source)]), groups, "audit", "protocol", "model")
        cached = cached_group(source)
        cached["full_relevant_count"] = 1
        with self.assertRaisesRegex(ValueError, "full_relevant_count"):
            EVAL.bind_cache_groups(self.metadata([cached]), groups, "tune", "protocol", "model")


class SelectionTests(unittest.TestCase):
    def reports(self):
        return {str(stage): EVAL.summarize([row("one", 1, 1, int(stage <= 50), 0, stage=stage)], stage)
                for stage in EVAL.TUNE_STAGES}

    def test_tie_selects_smallest_stage_by_text_macro(self):
        self.assertEqual(EVAL.choose_stage(self.reports()), 25)
        with self.assertRaisesRegex(ValueError, "nonempty tune"):
            EVAL.choose_stage({"25": EVAL.summarize([row("one", 0, 1, 0, 0)], 25)})

    def sealed_fixture(self, root):
        report = root / "tune-report.json"
        report.write_text(json.dumps({"stages": self.reports()}), encoding="utf-8")
        paths = {"parquet": root / "tune.parquet", "metadata": root / "tune.json"}
        for path in paths.values():
            path.write_bytes(b"unchanged")
        hashes = {name: EVAL.sha256_file(path) for name, path in paths.items()}
        selection = EVAL.make_selection("protocol", "model", hashes, EVAL.sha256_file(report),
                                         25, self.reports(), "train", paths)
        return selection, paths, report

    def test_selection_seals_model_protocol_train_and_tune_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection, paths, report = self.sealed_fixture(root)
            self.assertEqual(EVAL.verify_selection(selection, "protocol", "model", root, "train"), 25)
            for protocol, model, train in (("wrong", "model", "train"), ("protocol", "wrong", "train"),
                                            ("protocol", "model", "wrong")):
                with self.assertRaisesRegex(ValueError, "selection seal"):
                    EVAL.verify_selection(selection, protocol, model, root, train)
            paths["parquet"].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "tune cache"):
                EVAL.verify_selection(selection, "protocol", "model", root, "train")

    def test_modified_selection_and_report_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection, paths, report = self.sealed_fixture(root)
            with self.assertRaisesRegex(ValueError, "selection seal"):
                EVAL.verify_selection({**selection, "selected_stage": 50}, "protocol", "model", root, "train")
            report.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tune report"):
                EVAL.verify_selection(selection, "protocol", "model", root, "train")

    def test_bootstrap_pairs_normalized_text_clusters(self):
        rows = [row("one", 1, 1, 1, 0), row("  ONE ", 1, 1, 1, 0, representative=False),
                row("two", 1, 1, 0, 0), row("absent", 0, 1, 0, 0)]
        result = EVAL.paired_bootstrap(rows, 25, replicates=100)
        self.assertEqual(result["query_text_clusters"], 2)
        self.assertEqual(result["delta"], 0.5)
        self.assertEqual(result, EVAL.paired_bootstrap(rows, 25, replicates=100))


class ProtocolFilesTests(unittest.TestCase):
    def phase_fixture(self, root, variants):
        salt = "fixture-label-blind"
        chosen = min(variants, key=lambda group: EVAL.hashlib.sha256(json.dumps(
            [salt, group["query_key_json"]], ensure_ascii=False, separators=(",", ":")
        ).encode()).digest())
        for group in variants:
            group["representative"] = group is chosen
        payloads = {"labels_tune": ("labels-tune.json", {"partition": "tune", "groups": variants}),
                    "tune_texts": ("tune-texts.json", {"texts": sorted({x["query_text"] for x in variants})})}
        files = {}
        for name, (filename, payload) in payloads.items():
            path = root / filename
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            files[name] = {"path": filename, "sha256": EVAL.sha256_file(path)}
        return {"split": {"representative_salt": salt}, "files": files}

    def test_label_blind_representative_is_verified_without_audit_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            variants = [protocol_group(), protocol_group(location="other", available=())]
            manifest = self.phase_fixture(root, variants)
            loaded = EVAL.read_phase_labels(root, manifest, "tune")
            self.assertEqual(len(loaded), 2)
            self.assertEqual(sum(x["representative"] for x in loaded.values()), 1)
            for group in variants:
                group["representative"] = not group["representative"]
            path = root / "labels-tune.json"
            path.write_text(json.dumps({"partition": "tune", "groups": variants}), encoding="utf-8")
            manifest["files"]["labels_tune"]["sha256"] = EVAL.sha256_file(path)
            with self.assertRaisesRegex(ValueError, "label-blind hash rule"):
                EVAL.read_phase_labels(root, manifest, "tune")

    def test_phase_inventory_must_match_exact_text_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.phase_fixture(root, [protocol_group()])
            path = root / "tune-texts.json"
            path.write_text(json.dumps({"texts": ["wrong"]}), encoding="utf-8")
            manifest["files"]["tune_texts"]["sha256"] = EVAL.sha256_file(path)
            with self.assertRaisesRegex(ValueError, "text inventory differs"):
                EVAL.read_phase_labels(root, manifest, "tune")

    def test_semantic_names_and_file_sha_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "labels-tune.json"
            path.write_text("{}", encoding="utf-8")
            manifest = {"files": {"labels_tune": {"path": path.name, "sha256": EVAL.sha256_file(path)}}}
            self.assertEqual(EVAL.protocol_file(root, manifest, "labels-tune.json"), path)
            path.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                EVAL.protocol_file(root, manifest, "labels-tune.json")

    def test_protocol_path_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "child"
            root.mkdir()
            outside = root.parent / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            manifest = {"files": {"labels_tune": {"path": "../outside.json", "sha256": EVAL.sha256_file(outside)}}}
            with self.assertRaisesRegex(ValueError, "outside"):
                EVAL.protocol_file(root, manifest, "labels-tune.json")

    def test_audit_without_selection_never_reads_audit_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.joblib"
            model.write_bytes(b"fixture")
            artifact = {"model": types.SimpleNamespace(n_iter_=150, staged_predict_proba=lambda x: []),
                        "metadata": {}, "estimator": "histgb"}
            with mock.patch.object(EVAL, "checked_protocol", return_value=({}, "protocol")), \
                    mock.patch.dict(sys.modules, {"joblib": types.SimpleNamespace(load=lambda path: artifact)}), \
                    mock.patch.object(EVAL, "check_encoder_identity", return_value="encoder"), \
                    mock.patch.object(EVAL, "check_training_data", return_value="train"), \
                    mock.patch.object(EVAL, "read_phase_labels") as read_labels:
                with self.assertRaisesRegex(ValueError, "cannot open before tune"):
                    EVAL.main(["--phase", "audit", "--protocol-dir", str(root), "--cache-dir", str(root),
                               "--model", str(model), "--train-data", str(root / "train.parquet"),
                               "--output-dir", str(root / "out")])
                read_labels.assert_not_called()

    def test_cli_does_not_accept_audit_stage_choices(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            EVAL.main(["--phase", "audit", "--protocol-dir", "fixture", "--cache-dir", "fixture",
                       "--model", "fixture", "--train-data", "fixture", "--output-dir", "fixture", "--stages", "150"])

    def test_cli_requires_proven_training_cache(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            EVAL.main(["--phase", "tune", "--protocol-dir", "fixture", "--cache-dir", "fixture",
                       "--model", "fixture", "--output-dir", "fixture"])
        with self.assertRaisesRegex(ValueError, "requires the model's own"):
            EVAL.check_training_data(None, {}, Path("fixture"), {})


class EncoderIdentityTests(unittest.TestCase):
    def fixture(self, root):
        checkpoint = root / "run" / "model"
        checkpoint.mkdir(parents=True)
        stats = {"seed": 42, "training_pairs": 10}
        path = checkpoint.parent / "data_stats.json"
        path.write_text(json.dumps(stats), encoding="utf-8")
        expected = "a" * 64
        manifest = {"encoder": {"path": str(checkpoint), "content_sha256": expected,
                                "data_stats": stats, "data_stats_sha256": EVAL.sha256_file(path)},
                    "split": {"query_seed": 42}}
        artifact = {"metadata": {"embedding_checkpoint_sha256": expected,
                    "retrieval_config": {"embedding_model": str(checkpoint), "embedding_max_length": 128}}}
        return artifact, manifest, path

    def test_different_encoder_with_same_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, manifest, path = self.fixture(Path(tmp))
            self.assertEqual(EVAL.check_encoder_identity(artifact, manifest, fingerprint=lambda p: "a" * 64), "a" * 64)
            artifact["metadata"]["embedding_checkpoint_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "encoder identity differs"):
                EVAL.check_encoder_identity(artifact, manifest, fingerprint=lambda p: "b" * 64)

    def test_changed_live_weights_and_changed_training_stats_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, manifest, path = self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, "Live encoder contents"):
                EVAL.check_encoder_identity(artifact, manifest, fingerprint=lambda p: "b" * 64)
            path.write_text(json.dumps({"seed": 42, "training_pairs": 11}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training statistics changed"):
                EVAL.check_encoder_identity(artifact, manifest, fingerprint=lambda p: "a" * 64)


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Cache gate fixture requires project pyarrow")
class AuditCacheGateTests(unittest.TestCase):
    def fixture(self, root):
        import pyarrow as pa
        import pyarrow.parquet as pq
        model = root / "model.joblib"
        model.write_bytes(b"opaque gate model fixture")
        protocol = root / "protocol"
        protocol.mkdir()
        (protocol / "manifest.json").write_text("{}", encoding="utf-8")
        output = root / "reports"
        output.mkdir()
        (output / "selection.json").write_text("{}", encoding="utf-8")
        cache = root / "cache"
        cache.mkdir()
        path = cache / "audit.parquet"
        features = ["rrf_score", "baseline_top50"]
        table = pa.table({"rrf_score": pa.array([0.2], type=pa.float32()),
                          "baseline_top50": pa.array([1.0], type=pa.float32()),
                          "group_id": [0], "item_id": [A], "label": [1],
                          "positive_local": [True], "positive_seen": [False]})
        pq.write_table(table, path)
        metadata = {"cache_version": 2, "role": "validation", "fresh_partition": "audit",
                    "fresh_validation_protocol_sha256": EVAL.sha256_file(protocol / "manifest.json"),
                    "model_sha256": EVAL.sha256_file(model), "historical_holdout_reused": False,
                    "full_truth_preserved": True, "cache_parquet_sha256": EVAL.sha256_file(path),
                    "feature_names": features, "retrieval_config": {"method": "hybrid"},
                    "corpus_fingerprint": "corpus", "rows": 1,
                    "groups": [{"group_id": 0, "pool_size": 1}]}
        (cache / "audit.json").write_text(json.dumps(metadata), encoding="utf-8")
        artifact = {"model": types.SimpleNamespace(n_iter_=150, staged_predict_proba=lambda x: []),
                    "metadata": {key: metadata[key] for key in
                                 ("feature_names", "retrieval_config", "corpus_fingerprint")},
                    "estimator": "histgb"}
        argv = ["--phase", "audit", "--protocol-dir", str(protocol), "--cache-dir", str(cache),
                "--model", str(model), "--train-data", str(root / "training.parquet"),
                "--output-dir", str(output)]
        return argv, artifact, metadata, path, protocol

    def test_invalid_compatibility_priors_and_parquet_metadata_never_open_audit_labels(self):
        for failure in ("compatibility", "priors", "parquet_hash", "rows", "phase", "budget"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                argv, artifact, metadata, path, protocol = self.fixture(root)
                if failure == "compatibility":
                    metadata["corpus_fingerprint"] = "other corpus"
                elif failure == "parquet_hash":
                    metadata["cache_parquet_sha256"] = "wrong"
                elif failure == "rows":
                    metadata["rows"] = 2
                    metadata["groups"][0]["pool_size"] = 2
                elif failure == "phase":
                    metadata["fresh_partition"] = "tune"
                elif failure == "budget":
                    metadata["rows"] = metadata["groups"][0]["pool_size"] = 999
                (path.parent / "audit.json").write_text(json.dumps(metadata), encoding="utf-8")
                with mock.patch.object(EVAL, "checked_protocol", return_value=({}, EVAL.sha256_file(protocol / "manifest.json"))), \
                        mock.patch.dict(sys.modules, {"joblib": types.SimpleNamespace(load=lambda _path: artifact)}), \
                        mock.patch.object(EVAL, "check_encoder_identity", return_value="encoder"), \
                        mock.patch.object(EVAL, "check_training_data", return_value="train"), \
                        mock.patch.object(EVAL, "verify_selection", return_value=50), \
                        mock.patch.object(EVAL, "ensure_prior_policy",
                                          side_effect=ValueError("invalid cache priors") if failure == "priors" else None), \
                        mock.patch.object(EVAL, "read_phase_labels") as read_labels:
                    with self.assertRaises(ValueError):
                        EVAL.main(argv)
                    read_labels.assert_not_called()
                self.assertFalse((root / "reports" / "audit-report.json").exists())

    def test_successful_cache_preflight_is_required_before_label_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv, artifact, metadata, path, protocol = self.fixture(root)
            order = []
            original = EVAL.check_phase_cache_inputs
            def check_cache(*args):
                order.append("cache")
                return original(*args)
            def read_labels(*_args):
                order.append("labels")
                raise RuntimeError("stop at proven label opening")
            with mock.patch.object(EVAL, "checked_protocol", return_value=({}, EVAL.sha256_file(protocol / "manifest.json"))), \
                    mock.patch.dict(sys.modules, {"joblib": types.SimpleNamespace(load=lambda _path: artifact)}), \
                    mock.patch.object(EVAL, "check_encoder_identity", return_value="encoder"), \
                    mock.patch.object(EVAL, "check_training_data", return_value="train"), \
                    mock.patch.object(EVAL, "verify_selection", return_value=50), \
                    mock.patch.object(EVAL, "ensure_prior_policy", return_value={}), \
                    mock.patch.object(EVAL, "check_phase_cache_inputs", side_effect=check_cache), \
                    mock.patch.object(EVAL, "read_phase_labels", side_effect=read_labels):
                with self.assertRaisesRegex(RuntimeError, "proven label opening"):
                    EVAL.main(argv)
            self.assertEqual(order, ["cache", "labels"])


class StreamingFixtureTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Tiny Parquet fixture requires project pyarrow")
    def test_partial_pools_and_missing_zero_pools_are_retained(self):
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        class Model:
            max_consumed = []

            def staged_predict_proba(self, matrix):
                consumed = 0
                try:
                    for stage in range(1, 151):
                        consumed = stage
                        scores = matrix[:, 2]
                        yield np.column_stack((1 - scores, scores))
                finally:
                    self.max_consumed.append(consumed)

        zero_a = protocol_group(text="empty first", available=())
        real_a = protocol_group(text="real first", available=(A,), full=(A, B))
        zero_b = protocol_group(text="empty middle", available=())
        real_b = protocol_group(text="real last", available=(B,), full=(B,))
        groups = [scoring_group(zero_a, index=0, pool_size=0, positives=0, baseline_hits=0),
                  scoring_group(real_a, index=1, pool_size=3),
                  scoring_group(zero_b, index=2, pool_size=0, positives=0, baseline_hits=0),
                  scoring_group(real_b, index=3, pool_size=1)]
        features = ["rrf_score", "baseline_top50", "signal"]
        columns = {"rrf_score": [0.2, 0.1, 0.0, 0.1], "baseline_top50": [1., 0., 0., 1.],
                   "signal": [0.9, 0.1, 0., 0.9], "group_id": [1, 1, 1, 3],
                   "item_id": [A, C, "d" * 16, B], "label": [1, 0, 0, 1],
                   "positive_local": [True] * 4}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "validation.parquet"
            pq.write_table(pa.table(columns), path)
            model = Model()
            rows = EVAL.score_cache(path, {"feature_names": features}, groups,
                                    {"model": model}, (50,), batch_size=2)
            self.assertEqual([x["group_id"] for x in rows], [0, 1, 2, 3])
            self.assertEqual([x["s50_hits"] for x in rows], [0, 1, 0, 1])
            self.assertTrue(all("s150_hits" not in x for x in rows))
            self.assertEqual(model.max_consumed, [50, 50])
            self.assertEqual(EVAL.summarize(rows, 50)["full_truth"]["reranker"]["text_macro_recall_at_50"], 0.375)


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Tiny training provenance fixture requires project pyarrow")
class TrainingProvenanceFixtureTests(unittest.TestCase):
    def fixture(self, root, sources=None, sampled=1, allowed=("query",)):
        import pyarrow as pa
        import pyarrow.parquet as pq
        sources = sources or [protocol_group()]
        path = root / "train.parquet"
        pq.write_table(pa.table({"query_text": [x["query_text"] for x in sources],
            "group_id": [x["group_id"] for x in sources],
            "query_key_json": [x["query_key_json"] for x in sources],
            "feature": [1.] * len(sources)}), path)
        metadata = {"feature_names": ["feature"], "stats": {"sampled_queries": sampled}}
        meta_path = path.with_suffix(".parquet.json")
        meta_path.write_text(json.dumps(metadata), encoding="utf-8")
        fit_path = root / "fit-texts.json"
        fit_path.write_text(json.dumps({"texts": list(allowed)}), encoding="utf-8")
        manifest = {"files": {"fit_texts": {"path": fit_path.name, "sha256": EVAL.sha256_file(fit_path)}}}
        artifact = {"metadata": metadata, "provenance": {"training_data_sha256": EVAL.sha256_file(path),
                     "training_metadata_sha256": EVAL.sha256_file(meta_path)}}
        return path, artifact, manifest

    def test_own_train_rows_and_metadata_are_bound_to_allowed_fit_texts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, artifact, manifest = self.fixture(root)
            self.assertEqual(EVAL.check_training_data(path, artifact, root, manifest), EVAL.sha256_file(path))
            artifact["provenance"]["training_metadata_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "proven metadata"):
                EVAL.check_training_data(path, artifact, root, manifest)

    def test_non_fit_training_text_fails_even_with_matching_file_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, artifact, manifest = self.fixture(root, allowed=("other",))
            with self.assertRaisesRegex(ValueError, "non-fit"):
                EVAL.check_training_data(path, artifact, root, manifest)

    def test_sampled_group_budget_cannot_exceed_30000(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, artifact, manifest = self.fixture(root, sampled=30001)
            with self.assertRaisesRegex(ValueError, "at most 30000"):
                EVAL.check_training_data(path, artifact, root, manifest)

    def test_actual_unique_groups_cannot_exceed_declared_sampled_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, artifact, manifest = self.fixture(root, sources=[protocol_group(), protocol_group(location="other")])
            with self.assertRaisesRegex(ValueError, "sampled-query group budget"):
                EVAL.check_training_data(path, artifact, root, manifest)


if __name__ == "__main__":
    unittest.main()
