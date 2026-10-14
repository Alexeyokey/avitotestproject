"""Synthetic publication seals/scores only; no real Audit labels or fitted models."""
import ast
from collections import defaultdict
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import joblib

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("publish_research", ROOT / "publish_research_reranker.py")
PUBLISH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISH)


def source(name):
    path = ROOT / name
    if not path.is_file(): path = ROOT / (name + ".contract.txt")
    return ast.parse(path.read_text(encoding="utf-8"))


def functions(name, names, namespace):
    nodes = [node for node in source(name).body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names): raise AssertionError("Missing real production helper")
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), name, "exec"), namespace)


class TinyHistModel:
    n_iter_ = 150
    def __init__(self): self._predictors = list(range(150))
    def staged_predict_proba(self, *_args): raise AssertionError("Publication must not predict")
    def predict_proba(self, *_args): raise AssertionError("Publication must not predict")


class ResearchPublicationTests(unittest.TestCase):
    def fixture(self, root):
        namespace = dict(hashlib=hashlib, Path=Path, json=json, re=re, defaultdict=defaultdict,
                         normalize_query_text=lambda value: " ".join(value.lower().split()), Any=object)
        functions("audit_reranker_learning_curve.py", {"sha256_file", "maximum_iterations"}, namespace)
        functions("evaluate_fresh_validation.py", {"canonical_sha256", "aggregate", "summarize"}, namespace)
        namespace["SLICE_DENOMINATORS"] = {}
        evaluation = SimpleNamespace(**{name: namespace[name] for name in (
            "sha256_file", "canonical_sha256", "normalize_query_text", "maximum_iterations", "summarize")})
        summary_ns = {"EVAL": evaluation}
        functions("evaluate_research_reranker.py", {"named_summary"}, summary_ns)
        sha, canonical = evaluation.sha256_file, evaluation.canonical_sha256
        policy = next(ast.literal_eval(node.value) for node in source("freeze_research_selection.py").body
                      if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
                         target.id == "REGRESSION_POLICY" for target in node.targets))
        required = next(ast.literal_eval(node.value) for node in source("freeze_research_selection.py").body
                        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
                           target.id == "REQUIRED_FILES" for target in node.targets))
        protocol = root / "protocol"; protocol.mkdir()
        context_keys = [["q alpha", "10", "{}", "114", ""],
                        ["q alpha", "20", "{}", "114", ""], ["empty", "", "", "114", ""]]
        contexts = []
        salt = "fixture-representatives"
        first = min(context_keys[:2], key=lambda key: hashlib.sha256(json.dumps([salt,
            json.dumps(key, ensure_ascii=False, separators=(",", ":"))], ensure_ascii=False,
            separators=(",", ":")).encode()).digest())
        for index, key in enumerate(context_keys):
            identity = hashlib.sha256(json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            empty = key[0] == "empty"
            contexts.append(dict(group_id=index, group_identity=identity, query_key=key,
                query_text=key[0], full_relevant_count=2 if not empty else 1,
                available_relevant_count=0 if empty else 1, representative=empty or key == first,
                pool_size=0 if empty else 2, positives_in_pool=0 if empty else 1,
                baseline_hits=0 if empty else 1, relevant_count=0 if empty else 1,
                partition="audit", no_exact_location=False, has_filters=False, one_word=empty))
        manifest = dict(stats={"audit": {"groups": 3, "texts": 2}}, corpus={"size": 200000},
                        split={"representative_salt": salt})
        manifest_path = protocol / "manifest.json"; manifest_path.write_text(json.dumps(manifest))
        audit_texts = protocol / "audit-texts.json"; audit_texts.write_text(json.dumps({"texts": ["empty", "q alpha"]}))
        config = dict(method="hybrid", embedding_model="saved-encoder", candidate_k=449,
                      supplemental_fill="micro-typed-dense-v1", rescore_dense_union=True)
        prior_policy = dict(research_classifier_fingerprint_version=2, research_classifier_semantic_sha256="classifier",
                            research_priors_sha256="priors", inference_priors_sha256="legacy-priors")
        artifact = dict(model=TinyHistModel(), metadata=dict(feature_version=8,
            feature_names=["rrf_score", "baseline_top50"], retrieval_config=config,
            corpus_fingerprint="corpus", seed=42, split="queries", validation_fraction=.2),
            priors={"geo": {"places": {"a", "b"}}, "classifier": {"coeff": [.1, .2], "text_set": frozenset({"train"})}},
            provenance={"keep": True}, selected_iteration=None, selection_status="unselected",
            training_configuration={"parameters": {"keep": 3}, "fitted_iterations": 150})
        model = root / "new-model.joblib"; joblib.dump(artifact, model)
        old_model = root / "old-model.joblib"; joblib.dump({"model": TinyHistModel(), "old": True}, old_model)
        train = root / "new-train.parquet"; train.write_bytes(b"opaque completed train cache")
        old_train = root / "old-train.parquet"; old_train.write_bytes(b"opaque completed old train cache")
        files = {}
        for name in required:
            path = root / name; path.write_bytes(name.encode())
            files[name] = dict(path=str(path), sha256=sha(path))
        def identity(path): return dict(path=str(path.resolve()), sha256=sha(path))
        files.update(protocol_manifest=identity(manifest_path), new_model=identity(model), old_model=identity(old_model),
                     new_train_data=identity(train), old_train_data=identity(old_train))
        old_cache_dir = root / "old-cache"; old_cache_dir.mkdir()
        old_cache = old_cache_dir / "audit.parquet"; old_cache.write_bytes(b"completed old audit features")
        old_meta_path = old_cache_dir / "audit.json"
        old_contexts = copy.deepcopy(list(reversed(contexts)))
        for index, group in enumerate(old_contexts): group["group_id"] = index
        old_meta = dict(groups=old_contexts, rows=4, cache_parquet_sha256=sha(old_cache),
            feature_names=["rrf_score", "baseline_top50"], model_sha256=sha(old_model),
            fresh_validation_protocol_sha256=sha(manifest_path))
        old_meta_path.write_text(json.dumps(old_meta))
        files.update(old_audit_cache=identity(old_cache), old_audit_metadata=identity(old_meta_path))
        dense = root / "dense"; dense.mkdir(); fingerprint = "1" * 20
        binary, index_meta = dense / (fingerprint + ".bin"), dense / (fingerprint + ".json")
        binary.write_bytes(b"sealed index"); index_meta.write_text("{}")
        files.update(dense_index_binary=identity(binary), dense_index_metadata=identity(index_meta))
        index = dict(directory=str(dense), fingerprint=fingerprint, binary=identity(binary), metadata=identity(index_meta))
        report = dict(report_version=1, phase="tune", exploratory=True, audit_labels_read=False,
            selection_performed=False, old_stage=150, protocol_sha256=sha(manifest_path),
            group_scores_sha256=files["tune_group_scores"]["sha256"],
            new_capacity_curve={str(stage): {"conditional_available": {"reranker": {
                "text_macro_recall_at_50": value}}} for stage, value in ((75, .9), (100, .8), (150, .7))})
        for prefix, key in (("new", "new_pipeline_identity"), ("old", "old_pipeline_identity")):
            report[key] = {field: files[f"{prefix}_{suffix}"] for field, suffix in (
                ("model", "model"), ("training_data", "train_data"), ("cache", "tune_cache"), ("metadata", "tune_metadata"))}
        Path(files["tune_report"]["path"]).write_text(json.dumps(report))
        files["tune_report"] = identity(Path(files["tune_report"]["path"]))
        frozen_source = root / "frozen-source.py"; frozen_source.write_text("producer source")
        selection = dict(selection_version=1, policy=policy, selected_stage=100, selected_tune_text_macro_recall=.8,
            files=files, source_sha256={str(frozen_source): sha(frozen_source)}, dense_index=index,
            retrieval_config=config, research_prior_policy=prior_policy, encoder_base_reference={"base": "pinned"})
        selection_path = root / "selection.json"
        selection["selection_sha256"] = canonical(selection); selection_path.write_text(json.dumps(selection))
        freeze_ns = dict(Path=Path, json=json, re=re, EVAL=evaluation, REGRESSION_POLICY=policy, REQUIRED_FILES=required,
            FROZEN_REFERENCE_MODEL_SHA256=sha(old_model), regression_source_paths=lambda: [frozen_source],
            check_sources=lambda snapshot: self.assert_source_hashes(snapshot, sha))
        functions("freeze_research_selection.py", {"verify_files", "verify_dense_index", "validate_tune_report",
                    "check_regression_sources", "verify_frozen_selection"}, freeze_ns)
        freeze_ns["verify_frozen_selection"](selection_path)
        regression = root / "frozen-regression-evaluation"; regression.mkdir()
        new_cache_dir = root / "new-cache"; new_cache_dir.mkdir()
        new_cache = new_cache_dir / "audit.parquet"; new_cache.write_bytes(b"completed new audit features")
        new_meta_path = new_cache_dir / "audit.json"
        new_meta = dict(groups=copy.deepcopy(contexts), rows=4, feature_names=artifact["metadata"]["feature_names"],
            retrieval_config=config, cache_parquet_sha256=sha(new_cache), **prior_policy)
        new_meta_path.write_text(json.dumps(new_meta))
        old_by_id = {group["group_identity"]: group for group in old_contexts}
        pairs = []
        for group in contexts:
            old = old_by_id[group["group_identity"]]
            pairs.append(dict(group_identity=group["group_identity"], new={**group, "s100_hits": 0,
                "pool_hits": group["positives_in_pool"]}, old={**old, "s150_hits": old["positives_in_pool"],
                "pool_hits": old["positives_in_pool"]}))
        scores = regression / "frozen-regression-group-scores.jsonl"
        scores.write_text("".join(json.dumps(pair) + "\n" for pair in pairs))
        regression_report = dict(report_version=1, **policy, phase="audit", selected_stage=100,
            selection_sha256=selection["selection_sha256"], selection_file_sha256=sha(selection_path),
            frozen_selection=identity(selection_path), dense_index=index, protocol_sha256=sha(manifest_path),
            new_audit_metadata=identity(new_meta_path), new_audit_cache_sha256=sha(new_cache),
            old_audit_cache=files["old_audit_cache"], group_scores_sha256=sha(scores),
            new_pipeline=summary_ns["named_summary"]([pair["new"] for pair in pairs], 100),
            old_hgb150_pipeline=summary_ns["named_summary"]([pair["old"] for pair in pairs], 150))
        report_path = regression / "frozen-regression-report.json"; report_path.write_text(json.dumps(regression_report))
        evaluation.checked_protocol = lambda path: (manifest, sha(Path(path) / "manifest.json"))
        evaluation.protocol_file = lambda *_args: audit_texts
        evaluation.phase_cache_paths = lambda path, phase: (Path(path) / f"{phase}.parquet", Path(path) / f"{phase}.json")
        evaluation.check_phase_cache_inputs = lambda path, meta, *_args: self.check_cache_bytes(path, meta, sha)
        evaluation.ensure_compatible = lambda meta, cached, _name: self.assertEqual(meta["feature_names"], cached["feature_names"])
        evaluation.ensure_prior_policy = lambda *_args: None
        evaluation.check_encoder_identity = lambda *_args: "encoder-proof"
        evaluation.check_training_data = lambda path, *_args: sha(Path(path))
        atomic_ns = dict(Path=Path, os=os, tempfile=tempfile, joblib=joblib)
        functions("publish_fresh_reranker.py", {"publish_new"}, atomic_ns)
        contract = dict(evaluation=evaluation, verify_selection=freeze_ns["verify_frozen_selection"], policy=policy,
            check_regression_cache=lambda path, meta, *_args: self.check_cache_bytes(path, meta, sha),
            named_summary=summary_ns["named_summary"], research_prior_policy=lambda _artifact: prior_policy,
            check_training_priors=lambda *_args: None, check_reference=lambda *_args: None,
            check_base=lambda *_args: selection["encoder_base_reference"], publish_new=atomic_ns["publish_new"])
        return dict(protocol=protocol, selection_path=selection_path, selection=selection, regression=regression,
            output=root / "selected-v8.joblib", artifact=artifact, model=model, old_model=old_model,
            contract=contract, report_path=report_path, report=regression_report, scores=scores, pairs=pairs,
            new_meta=new_meta, new_meta_path=new_meta_path, frozen_source=frozen_source)

    @staticmethod
    def assert_source_hashes(snapshot, sha):
        if any(sha(Path(path)) != value for path, value in snapshot.items()): raise ValueError("Frozen source changed")

    @staticmethod
    def check_cache_bytes(path, metadata, sha):
        value = sha(Path(path))
        if metadata["cache_parquet_sha256"] != value: raise ValueError("Completed cache changed")
        return value

    def publish(self, fixture):
        return PUBLISH.publish(fixture["protocol"], fixture["selection_path"], fixture["regression"],
                               fixture["output"], contract=fixture["contract"])

    def rewrite_scores(self, fixture, pairs):
        fixture["scores"].write_text("".join(json.dumps(pair) + "\n" for pair in pairs))
        fixture["report"]["group_scores_sha256"] = fixture["contract"]["evaluation"].sha256_file(fixture["scores"])
        fixture["report_path"].write_text(json.dumps(fixture["report"]))

    def test_publishes_tune_sealed_prefix_even_when_regression_worse_preserves_full_source(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory)); original = fixture["model"].read_bytes()
            old = fixture["old_model"].read_bytes()
            result = self.publish(fixture); selected = joblib.load(fixture["output"])
            self.assertEqual(selected["selected_iteration"], 100)
            self.assertEqual(selected["model"]._predictors, list(range(150)))
            for key in ("metadata", "priors", "provenance"):
                self.assertEqual(selected[key], fixture["artifact"][key])
            self.assertEqual(selected["training_configuration"]["parameters"], {"keep": 3})
            self.assertEqual(fixture["model"].read_bytes(), original); self.assertEqual(fixture["old_model"].read_bytes(), old)
            proof = selected["research_selection"]
            self.assertEqual(proof["publication_role"], "candidate-only")
            self.assertFalse(proof["official_benchmark_validated"]); self.assertFalse(proof["audit_independent"])
            self.assertFalse(proof["automatic_deployment_selection"]); self.assertFalse(proof["audit_used_for_capacity_selection"])
            self.assertEqual(proof["frozen_regression"]["context_count"], 3)
            self.assertEqual(proof["new_pipeline_summary"]["coverage"]["groups_without_available_target"], 1)
            self.assertEqual(result["sha256"], fixture["contract"]["evaluation"].sha256_file(fixture["output"]))

    def test_complete_seal_required_and_changed_frozen_inputs_rejected(self):
        for mode in ("selection_checksum", "source", "old_model", "model", "train", "tune", "index"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(Path(directory))
                if mode == "selection_checksum":
                    selection = json.loads(fixture["selection_path"].read_text()); selection["selected_stage"] = 75
                    fixture["selection_path"].write_text(json.dumps(selection))
                else:
                    path = {"source": fixture["frozen_source"], "old_model": fixture["old_model"], "model": fixture["model"],
                            "train": Path(fixture["selection"]["files"]["new_train_data"]["path"]),
                            "tune": Path(fixture["selection"]["files"]["tune_report"]["path"]),
                            "index": Path(fixture["selection"]["dense_index"]["binary"]["path"])}[mode]
                    path.write_bytes(b"changed")
                with self.assertRaises(ValueError): self.publish(fixture)
                self.assertFalse(fixture["output"].exists())

    def test_missing_wrong_or_independent_regression_report_rejected_before_model_load(self):
        for key, value in (("selected_stage", 75), ("protocol_sha256", "bad"), ("selection_file_sha256", "bad"),
            ("selection_sha256", "bad"), ("new_audit_cache_sha256", "bad"), ("old_audit_cache", {}),
            ("independent_fresh_test", True), ("previous_baseline_audit_observed", False), ("phase", "tune"),
            ("group_scores_sha256", "bad"), ("report_version", 2), ("new_pipeline", {}), ("old_hgb150_pipeline", {})):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(Path(directory)); fixture["report"][key] = value
                fixture["report_path"].write_text(json.dumps(fixture["report"]))
                with mock.patch.object(joblib, "load", side_effect=AssertionError("No model load before full proof")):
                    with self.assertRaises(ValueError): self.publish(fixture)
                self.assertFalse(fixture["output"].exists())
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory)); fixture["report_path"].unlink()
            with self.assertRaises(ValueError): self.publish(fixture)

    def test_context_duplicate_omission_unknown_identity_and_extra_prefix_rejected(self):
        for mode in ("duplicate", "missing_zero_available", "unknown", "wrapper_mismatch", "extra_new_stage", "wrong_old_stage", "denominator"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(Path(directory)); pairs = copy.deepcopy(fixture["pairs"])
                if mode == "duplicate": pairs[2] = copy.deepcopy(pairs[0])
                elif mode == "missing_zero_available": pairs = pairs[:2]
                elif mode == "unknown": pairs[0]["group_identity"] = "f" * 64
                elif mode == "wrapper_mismatch": pairs[0]["new"]["group_identity"] = "f" * 64
                elif mode == "extra_new_stage": pairs[0]["new"]["s75_hits"] = 1
                elif mode == "wrong_old_stage": pairs[0]["old"]["s100_hits"] = pairs[0]["old"].pop("s150_hits")
                else: pairs[0]["new"]["available_relevant_count"] = 2
                self.rewrite_scores(fixture, pairs)
                with self.assertRaises(ValueError): self.publish(fixture)
                self.assertFalse(fixture["output"].exists())

    def test_completed_cache_metadata_changed_or_does_not_cover_protocol_rejected(self):
        for mode in ("metadata_bytes", "metadata_identity", "metadata_duplicate", "metadata_omitted", "settings", "priors"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(Path(directory))
                if mode == "metadata_bytes": fixture["new_meta_path"].write_text("{}")
                elif mode == "metadata_identity": fixture["report"]["new_audit_metadata"]["path"] = str(Path(directory) / "wrong.json")
                else:
                    metadata = fixture["new_meta"]
                    if mode == "metadata_duplicate": metadata["groups"][2] = copy.deepcopy(metadata["groups"][0]); metadata["groups"][2]["group_id"] = 2
                    elif mode == "metadata_omitted": metadata["groups"].pop()
                    elif mode == "settings": metadata["retrieval_config"] = {}
                    else: metadata["research_classifier_semantic_sha256"] = "wrong"
                    fixture["new_meta_path"].write_text(json.dumps(metadata))
                    fixture["report"]["new_audit_metadata"]["sha256"] = fixture["contract"]["evaluation"].sha256_file(fixture["new_meta_path"])
                fixture["report_path"].write_text(json.dumps(fixture["report"]))
                with self.assertRaises((ValueError, FileNotFoundError)): self.publish(fixture)
                self.assertFalse(fixture["output"].exists())

    def test_existing_output_and_atomic_race_preserve_earlier_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory)); fixture["output"].write_bytes(b"keep previous")
            with self.assertRaises(FileExistsError): self.publish(fixture)
            self.assertEqual(fixture["output"].read_bytes(), b"keep previous")
            fixture["output"].unlink()
            def race(_source, target): Path(target).write_bytes(b"raced existing artifact"); raise FileExistsError("race")
            with mock.patch.object(os, "link", side_effect=race):
                with self.assertRaises(FileExistsError): self.publish(fixture)
            self.assertEqual(fixture["output"].read_bytes(), b"raced existing artifact")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_summary_recomputation_restores_original_cache_order_for_float_accumulation(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            evaluation = fixture["contract"]["evaluation"]
            inventory = PUBLISH._context_inventory(fixture["new_meta"], 3, evaluation)
            records = [{"group_identity": pair["group_identity"], "row": pair["new"]}
                       for pair in reversed(fixture["pairs"])]
            rows = PUBLISH._checked_rows(records, fixture["new_meta"], inventory, 100)
            self.assertEqual([row["group_id"] for row in rows], [0, 1, 2])
            # Use the actual recorded-summary helper and fractions whose sum
            # depends on order; original group order must remain authoritative.
            fractions = [.1, .2, .3]
            def accumulate(values):
                result = 0.
                for value in values: result += value
                return result
            self.assertNotEqual(accumulate(fractions), accumulate(reversed(fractions)))
            self.assertEqual(accumulate(fractions[row["group_id"]] for row in rows), accumulate(fractions))

    def test_mid_check_report_or_seal_change_rejected(self):
        for mode in ("report", "seal"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(Path(directory))
                def changed(*_args):
                    path = fixture["report_path"] if mode == "report" else fixture["selection_path"]
                    path.write_text(path.read_text() + " ")
                    return fixture["selection"]["encoder_base_reference"]
                fixture["contract"]["check_base"] = changed
                with self.assertRaises(ValueError): self.publish(fixture)
                self.assertFalse(fixture["output"].exists())

    def test_cli_has_no_stage_best_deploy_or_audit_label_access(self):
        arguments = ["--protocol-dir", "protocol", "--selection", "selection", "--regression-dir", "regression", "--output", "new.joblib"]
        with mock.patch.object(PUBLISH, "publish", return_value={}) as publish:
            for option in ("--stage", "--best", "--deploy"):
                with self.assertRaises(SystemExit): PUBLISH.main(arguments + [option, "100"])
            publish.assert_not_called()
            PUBLISH.main(arguments)
            self.assertNotIn("stage", publish.call_args.kwargs)
        tree = ast.parse((ROOT / "publish_research_reranker.py").read_text())
        calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse({"read_phase_labels", "score_cache", "predict", "predict_proba", "staged_predict_proba"} & calls)


if __name__ == "__main__": unittest.main()
