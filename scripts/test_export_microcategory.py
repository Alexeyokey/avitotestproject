"""Tiny V7/V8 exporter fixtures; no Torch, sklearn, fitted weights or downloads."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import joblib

SPEC = importlib.util.spec_from_file_location("export_fixture_base", Path(__file__).with_name("test_export_offline_solution.py"))
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)
EXPORT = BASE.EXPORT
FINGERPRINT_PATH = Path(__file__).resolve().parents[1] / "src/avito_candidates/fitted_fingerprint.py"
if not FINGERPRINT_PATH.is_file():
    FINGERPRINT_PATH = Path(__file__).with_name("fitted_fingerprint.py")
FP_SPEC = importlib.util.spec_from_file_location("export_fitted_fingerprint", FINGERPRINT_PATH)
FINGERPRINT = importlib.util.module_from_spec(FP_SPEC)
sys.modules[FP_SPEC.name] = FINGERPRINT
FP_SPEC.loader.exec_module(FINGERPRINT)


def fingerprint_contract():
    return {"model": FINGERPRINT.fitted_estimator_fingerprint,
            "classifier": FINGERPRINT.classifier_semantic_fingerprint,
            "version": FINGERPRINT.FITTED_FINGERPRINT_VERSION}


def fold_for(text, folds=3, seed=42):
    value = json.dumps([seed, text], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big") % folds


class FixtureClassifier:
    def __init__(self, metadata):
        texts = frozenset("query " + str(index) for index in range(12))
        self.model = SimpleNamespace(fitted_texts=texts, classes=("101", "202"),
                                     vocabulary={"typed": 0}, idf=(1.5, 2.), coefficients=(.1, -.2))
        self.crossfit_models = {fold: SimpleNamespace(
            fitted_texts=frozenset(text for text in texts if fold_for(text) != fold),
            classes=("101", "202"), vocabulary={"fold": fold}, coefficients=(.3 + fold,))
            for fold in range(3)}
        self.metadata = metadata
        self._prediction_cache = {("temporary",): (("101", .9),)}


class MicrocategoryExportTests(unittest.TestCase):
    def fixture(self, directory, *, version=8, supplemental=True, relative_source=False):
        fixture = BASE.OfflineExportTests().fixture(directory)
        project, model = fixture[:2]
        source = project / "local-artifacts" / "microcategory.joblib"
        source.parent.mkdir()
        snapshot = dict(schema=EXPORT.SOFT_MICROCATEGORY_SCHEMA,
            source_sha256=hashlib.sha256(b"original train source fixture").hexdigest(),
            corpus_fingerprint="a" * 64, seed=42, fit_split="queries", validation_fraction=.2,
            crossfit_folds=3, classes=["101", "202"], fit_texts_count=12,
            allowed_fit_texts_count=13, allowed_fit_texts_sha256="b" * 64)
        classifier = FixtureClassifier(snapshot)
        snapshot["observed_fit_texts_sha256"] = EXPORT._text_inventory_digest(classifier.model.fitted_texts)
        payload = {"model": classifier.model, "crossfit_models": classifier.crossfit_models,
                   "metadata": snapshot}
        joblib.dump(payload, source)
        source.with_suffix(".joblib.json").write_text(json.dumps(
            {"model_sha256": EXPORT.sha256(source), "metadata": snapshot}), encoding="utf-8")
        artifact = joblib.load(model)
        artifact["priors"].update(microcategory_classifier=classifier, geo_associations={"saved": "history"})
        artifact["metadata"].update(feature_version=version, corpus_fingerprint=snapshot["corpus_fingerprint"],
            seed=42, split="queries", validation_fraction=.2, microcategory_classifier_metadata=snapshot,
            training_prior_policy={"microcategory_classifier": "grouped-out-of-fold"})
        config = artifact["metadata"]["retrieval_config"]
        config.update(microcategory_model=("local-artifacts/microcategory.joblib" if relative_source else str(source)),
            microcategory_model_sha256=EXPORT.sha256(source), microcategory_candidate_k=998,
            microcategory_field="combined", microcategory_weight=.25, microcategory_top_categories=3,
            microcategory_selection_policy="top998-then-novel-prefix", bm25_weight=1., dense_weight=1.)
        for field in ("title", "params", "description"):
            config.update({field + "_k1": 1., field + "_b": .5,
                           field + "_weight": .5 if field == "description" else 0.})
        config.update(stem_title_weight=0., stem_params_weight=0., bm25_rrf_k=30,
            bm25_channel_quota=0, geo_top_locations=3, geo_min_history=20, geo_weight=.25,
            embedding_params_words=40, embedding_description_words=48, rrf_k=60)
        if supplemental:
            config.update(supplemental_fill="micro-typed-dense-v1", supplemental_raw_depth=998,
                supplemental_rank_constant=60, supplemental_selection_policy="raw-ranks-equal-route-novel-rrf-id-ties")
        joblib.dump(artifact, model)
        cli = project / "src/avito_candidates/cli.py"
        cli.write_text(cli.read_text(encoding="utf-8").replace(
            '(root / "command.json").write_text(json.dumps(args))', '''micro = args[args.index("--microcategory-model") + 1]
        assert micro == "../models/microcategory.joblib"
        assert Path.cwd() == root / "project"
        import hashlib
        runtime = json.loads((root / "runtime.json").read_text())
        assert hashlib.sha256(Path(micro).read_bytes()).hexdigest() == runtime["microcategory_provenance"]["model_sha256"]
        assert not any(x in args for x in ("--microcategory-model-sha256", "--microcategory-selection-policy", "--supplemental-raw-depth", "--supplemental-rank-constant", "--supplemental-selection-policy"))
        (root / "command.json").write_text(json.dumps(args))'''), encoding="utf-8")
        return fixture, source

    def export(self, fixture):
        return EXPORT.export_bundle(*fixture[:5], requirements=fixture[5], answer=fixture[6],
            contract={**BASE.fixture_contract(), "fitted_fingerprint_contract": fingerprint_contract})

    def test_v7_and_v8_copy_bound_classifier_keep_original_config_and_priors(self):
        for version in (7, 8):
            with self.subTest(version=version), TemporaryDirectory() as directory:
                fixture, source = self.fixture(directory, version=version, supplemental=version == 8,
                                               relative_source=True)
                original_bytes = fixture[1].read_bytes()
                original = joblib.load(fixture[1])
                manifest = self.export(fixture)
                output = fixture[4]
                exported = joblib.load(output / "models/reranker.joblib")
                self.assertEqual(fixture[1].read_bytes(), original_bytes)
                self.assertEqual((output / "models/microcategory.joblib").read_bytes(), source.read_bytes())
                self.assertEqual((output / "models/microcategory.joblib.json").read_bytes(),
                                 source.with_suffix(".joblib.json").read_bytes())
                expected = dict(original["metadata"]["retrieval_config"], microcategory_model=EXPORT.PORTABLE_MICROCATEGORY)
                self.assertEqual(exported["metadata"]["retrieval_config"], expected)
                self.assertEqual(exported["priors"]["geo_associations"], {"saved": "history"})
                self.assertFalse(exported["priors"]["microcategory_classifier"]._prediction_cache)
                attest = manifest["microcategory_export_attestation"]
                self.assertEqual(attest["original_retrieval_config"], original["metadata"]["retrieval_config"])
                self.assertEqual(set(attest["immutable_model_hashes"]), {"global", "fold-0", "fold-1", "fold-2"})
                self.assertTrue(attest["integrity_only_not_a_signature"])
                self.assertFalse(attest["training_identity_proven_by_export"])
                self.assertEqual(attest["fitted_fingerprint_version"], 2)
                self.assertEqual(attest["immutable_hash_algorithm"], "canonical-fitted-state-v2-sha256")
                self.assertEqual(attest["classifier_semantic_sha256"],
                    FINGERPRINT.classifier_semantic_fingerprint(original["priors"]["microcategory_classifier"]))
                self.assertEqual(attest["classifier_semantic_sha256"],
                    FINGERPRINT.classifier_semantic_fingerprint(exported["priors"]["microcategory_classifier"]))
                self.assertEqual(attest["inference_priors_semantic_sha256"],
                    EXPORT.inference_priors_fingerprint(original, fingerprint_contract()))
                self.assertEqual(attest["inference_priors_semantic_sha256"],
                    EXPORT.inference_priors_fingerprint(exported, fingerprint_contract()))
                self.assertEqual(exported["metadata"]["corpus_fingerprint"], "a" * 64)
                self.assertEqual(exported["metadata"]["microcategory_classifier_metadata"]["source_sha256"],
                                 original["metadata"]["microcategory_classifier_metadata"]["source_sha256"])
                runtime = json.loads((output / "runtime.json").read_text())
                flags = dict(runtime["retrieval_arguments"])
                self.assertEqual(flags["--microcategory-model"], EXPORT.PORTABLE_MICROCATEGORY)
                for key in EXPORT.DERIVED_KEYS:
                    self.assertNotIn("--" + key.replace("_", "-"), flags)
                self.assertIn("fresh 200k-catalog", (output / "OFFLINE_README.txt").read_text())

    def test_relocated_v8_wrapper_uses_exact_relative_path_and_all_cli_flags(self):
        with TemporaryDirectory() as directory:
            fixture, _ = self.fixture(directory)
            self.export(fixture)
            moved = Path(directory) / "different" / "bundle"
            moved.parent.mkdir()
            fixture[4].rename(moved)
            result = subprocess.run([sys.executable, str(moved / "run_predict.py"), "--data-dir", directory,
                "--output", str(Path(directory) / "answer.csv")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = json.loads((moved / "command.json").read_text())
            self.assertEqual(args[args.index("--supplemental-fill") + 1], "micro-typed-dense-v1")
            self.assertIn("--rescore-dense-union", args)

    def test_changed_external_bytes_or_sidecar_rejected_before_output(self):
        for mode in ("bytes", "sidecar", "missing"):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                fixture, source = self.fixture(directory)
                if mode == "bytes":
                    source.write_bytes(source.read_bytes() + b"changed")
                elif mode == "sidecar":
                    envelope = json.loads(source.with_suffix(".joblib.json").read_text())
                    envelope["metadata"]["source_sha256"] = "0" * 64
                    source.with_suffix(".joblib.json").write_text(json.dumps(envelope))
                else:
                    source.with_suffix(".joblib.json").unlink()
                with self.assertRaises(ValueError):
                    self.export(fixture)
                self.assertFalse(fixture[4].exists())

    def test_changed_embedded_fitted_state_or_fold_inventory_rejected(self):
        for mode in ("coefficients", "vocabulary", "idf", "fold_coefficients", "fold_inventory", "classes", "texts", "missing"):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                fixture, _ = self.fixture(directory)
                artifact = joblib.load(fixture[1])
                classifier = artifact["priors"]["microcategory_classifier"]
                if mode == "coefficients": classifier.model.coefficients = (.7,)
                elif mode == "vocabulary": classifier.model.vocabulary = {"changed": 99}
                elif mode == "idf": classifier.model.idf = (99.,)
                elif mode == "fold_coefficients": classifier.crossfit_models[1].coefficients = (.99,)
                elif mode == "fold_inventory": classifier.crossfit_models[1].fitted_texts = classifier.model.fitted_texts
                elif mode == "classes": classifier.model.classes = ("999",)
                elif mode == "texts": classifier.model.fitted_texts = frozenset({"changed query"})
                else: classifier.crossfit_models.pop(1)
                joblib.dump(artifact, fixture[1])
                with self.assertRaises(ValueError): self.export(fixture)
                self.assertFalse(fixture[4].exists())

    def test_frozenset_roundtrip_and_runtime_vectorizer_pointer_preserve_fitted_semantics(self):
        with TemporaryDirectory() as directory:
            fixture, source = self.fixture(directory)
            artifact = joblib.load(fixture[1])
            classifier = artifact["priors"]["microcategory_classifier"]
            # A larger membership inventory reproduces serialization ordering
            # changes without any fitted model or ML dependency.
            texts = frozenset("normalized training query " + str(index) for index in range(1024))
            classifier.model.fitted_texts = texts
            for fold, model in classifier.crossfit_models.items():
                model.fitted_texts = frozenset(text for text in texts if fold_for(text) != fold)
            snapshot = classifier.metadata
            snapshot.update(fit_texts_count=len(texts), allowed_fit_texts_count=len(texts) + 1,
                            observed_fit_texts_sha256=EXPORT._text_inventory_digest(texts))
            external_payload = dict(model=classifier.model, crossfit_models=classifier.crossfit_models,
                                    metadata=snapshot)
            joblib.dump(external_payload, source)
            # The embedded and standalone classifier take independent pickle
            # paths, exactly as prepare -> fit and MCC.save do in production.
            embedded_path = Path(directory) / "embedded-roundtrip.joblib"
            joblib.dump(classifier, embedded_path)
            artifact["priors"]["microcategory_classifier"] = joblib.load(embedded_path)
            artifact["metadata"]["microcategory_classifier_metadata"] = copy.deepcopy(snapshot)
            artifact["metadata"]["retrieval_config"]["microcategory_model_sha256"] = EXPORT.sha256(source)
            source.with_suffix(".joblib.json").write_text(json.dumps({
                "metadata": snapshot, "model_sha256": EXPORT.sha256(source)}))
            joblib.dump(artifact, fixture[1])
            self.assertEqual(artifact["priors"]["microcategory_classifier"].model.fitted_texts, texts)
            manifest = self.export(fixture)
            loaded = joblib.load(fixture[4] / "models/reranker.joblib")
            expected = FINGERPRINT.classifier_semantic_fingerprint(classifier)
            self.assertEqual(manifest["microcategory_export_attestation"]["classifier_semantic_sha256"], expected)
            self.assertEqual(FINGERPRINT.classifier_semantic_fingerprint(
                loaded["priors"]["microcategory_classifier"]), expected)
            prior_sha = EXPORT.inference_priors_fingerprint(loaded, fingerprint_contract())
            self.assertEqual(prior_sha, manifest["microcategory_export_attestation"]["inference_priors_semantic_sha256"])
            loaded["priors"]["geo_associations"]["saved"] = "different history"
            self.assertNotEqual(EXPORT.inference_priors_fingerprint(loaded, fingerprint_contract()), prior_sha)
            # Mimic the exact sklearn runtime cache field without importing
            # sklearn. Only the helper's exact vectorizer type exemption applies.
            Vectorizer = type("TfidfVectorizer", (), {})
            Vectorizer.__module__ = "sklearn.feature_extraction.text"
            vectorizer = Vectorizer()
            vectorizer.vocabulary_ = {"typed": 0}
            vectorizer.idf_ = (1.5,)
            vectorizer._stop_words_id = 123
            before = FINGERPRINT.fitted_estimator_fingerprint(vectorizer)
            vectorizer._stop_words_id = 987654
            self.assertEqual(FINGERPRINT.fitted_estimator_fingerprint(vectorizer), before)
            vectorizer.idf_ = (9.,)
            self.assertNotEqual(FINGERPRINT.fitted_estimator_fingerprint(vectorizer), before)

    def test_export_uses_shared_fingerprint_under_different_python_hash_seeds(self):
        program = '''import hashlib, importlib.util, json, sys
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location("fingerprint", sys.argv[1])
helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
texts = frozenset("query " + str(x) for x in range(256))
model = SimpleNamespace(fitted_texts=texts, vocabulary={"alpha":0,"beta":1}, idf=(1.2,2.3), coefficients=(.1,-.2))
classifier = SimpleNamespace(model=model, crossfit_models={0:model}, metadata={"seed":42,"classes":["a","b"]})
print(helper.classifier_semantic_fingerprint(classifier))
'''
        outputs = []
        import os
        for seed in ("1", "719"):
            result = subprocess.run([sys.executable, "-c", program, str(FINGERPRINT_PATH)],
                capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": seed})
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs.append(result.stdout.strip())
        self.assertEqual(outputs[0], outputs[1])

    def test_corpus_source_split_schema_snapshot_and_v8_measurement_fail_closed(self):
        changes = [("corpus_fingerprint", "c" * 64), ("source_sha256", "not-a-sha"),
                   ("schema", "unexpected-schema"), ("seed", 43), ("fit_split", "all"),
                   ("validation_fraction", .3), ("crossfit_folds", 0)]
        for key, value in changes:
            with self.subTest(key=key), TemporaryDirectory() as directory:
                fixture, source = self.fixture(directory)
                artifact = joblib.load(fixture[1])
                classifier = artifact["priors"]["microcategory_classifier"]
                classifier.metadata[key] = value
                artifact["metadata"]["microcategory_classifier_metadata"] = copy.deepcopy(classifier.metadata)
                payload = joblib.load(source)
                payload["metadata"] = classifier.metadata
                joblib.dump(payload, source)
                source.with_suffix(".joblib.json").write_text(json.dumps({
                    "model_sha256": EXPORT.sha256(source), "metadata": classifier.metadata}))
                artifact["metadata"]["retrieval_config"]["microcategory_model_sha256"] = EXPORT.sha256(source)
                joblib.dump(artifact, fixture[1])
                with self.assertRaises(ValueError): self.export(fixture)
                self.assertFalse(fixture[4].exists())
        with TemporaryDirectory() as directory:
            fixture, _ = self.fixture(directory)
            artifact = joblib.load(fixture[1])
            artifact["metadata"]["retrieval_config"].pop("rescore_dense_union")
            joblib.dump(artifact, fixture[1])
            with self.assertRaisesRegex(ValueError, "V8.*rescoring"): self.export(fixture)

    def test_unknown_derived_value_missing_key_and_mixed_policy_rejected(self):
        for key, value in [("supplemental_raw_depth", 999), ("supplemental_rank_constant", 60.),
            ("supplemental_selection_policy", "novel-ranks"), ("microcategory_selection_policy", "other"),
            ("microcategory_candidate_k", 500), ("microcategory_field", "title"),
            ("microcategory_top_categories", 5), ("microcategory_weight", float("nan")),
            ("embedding_document_format", "compact"), ("future_unsafe_identity", 1)]:
            with self.subTest(key=key), TemporaryDirectory() as directory:
                fixture, _ = self.fixture(directory)
                artifact = joblib.load(fixture[1])
                artifact["metadata"]["retrieval_config"][key] = value
                joblib.dump(artifact, fixture[1])
                with self.assertRaises(ValueError): self.export(fixture)
                self.assertFalse(fixture[4].exists())
        with TemporaryDirectory() as directory:
            fixture, _ = self.fixture(directory)
            artifact = joblib.load(fixture[1])
            artifact["metadata"]["retrieval_config"].pop("supplemental_raw_depth")
            joblib.dump(artifact, fixture[1])
            with self.assertRaises(ValueError): self.export(fixture)

    def test_actual_retrieval_config_reconstructs_derived_identity_from_only_cli_flags(self):
        source = Path(__file__).resolve().parents[1] / "src/avito_candidates/reranker.py"
        if not source.is_file():
            source = Path(__file__).with_name("reranker-contract.txt")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "retrieval_config")
        # Only replace the lazy SHA import with a standard-library file digest;
        # the actual project's config construction body remains unchanged.
        class ShaImport(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return None if node.module == "microcategory" else node
        module = ast.fix_missing_locations(ast.Module(body=[ShaImport().visit(node)], type_ignores=[]))
        namespace = {"_sha_file": EXPORT.sha256}
        exec(compile(module, str(source), "exec"), namespace)
        with TemporaryDirectory() as directory:
            fixture, _ = self.fixture(directory)
            config = joblib.load(fixture[1])["metadata"]["retrieval_config"]
            serialized = EXPORT.portable_config_args(config, checkpoint=fixture[2])
            restored = SimpleNamespace(k1=None, b=None, geo_include_related=False,
                embedding_document_format="current", supplemental_fill="off", **{
                    key: value for key, value in config.items()
                    if key not in EXPORT.DERIVED_KEYS and key != "supplemental_fill"})
            for key in config:
                if key in EXPORT.DERIVED_KEYS: continue
                option = "--" + key.replace("_", "-")
                if key in EXPORT.BOOL_KEYS:
                    setattr(restored, key, option in serialized)
                else:
                    raw = serialized[serialized.index(option) + 1]
                    setattr(restored, key, type(config[key])(raw))
            self.assertEqual(namespace["retrieval_config"](restored), config)

    def test_microcategory_manifest_detects_tampered_portable_artifact(self):
        with TemporaryDirectory() as directory:
            fixture, _ = self.fixture(directory)
            self.export(fixture)
            (fixture[4] / "models/microcategory.joblib").write_bytes(b"changed")
            result = subprocess.run([sys.executable, str(fixture[4] / "run_predict.py"), "--data-dir", directory],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Bundled file content changed", result.stderr)
            self.assertFalse((fixture[4] / "command.json").exists())


if __name__ == "__main__":
    unittest.main()
