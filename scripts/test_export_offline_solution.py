"""Small offline bundle fixtures: no GPU, real model download or large copy."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import joblib

SPEC = importlib.util.spec_from_file_location("offline_export", Path(__file__).with_name("export_offline_solution.py"))
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class FixtureModel:
    n_iter_ = 150


def fixture_contract():
    def base_reference(path, *, hub_cache=None):
        base = EXPORT.resolve_adapter_base(path, hub_cache or os.environ["HF_HUB_CACHE"])
        return {"is_adapter": True, "base_repo": base["repo"], "base_revision": base["revision"]}

    def check_identity(metadata, config):
        if EXPORT.checkpoint_fingerprint(config["embedding_model"]) != metadata["embedding_checkpoint_sha256"]:
            raise ValueError("Checkpoint differs")
        base = base_reference(config["embedding_model"])
        if (base["base_repo"] != metadata["embedding_base_repo"]
                or base["base_revision"] != metadata["embedding_base_revision"]):
            raise ValueError("Base differs")
        comparable = dict(config)
        comparable["embedding_model"] = metadata["retrieval_config"]["embedding_model"]
        if comparable != metadata["retrieval_config"]:
            raise ValueError("Config differs")

    def validate_iteration(model, selected):
        if selected is not None and (isinstance(selected, bool) or not isinstance(selected, int)
                                     or not 1 <= selected <= model.n_iter_):
            raise ValueError("Invalid selected iteration")
        return selected

    return {"checkpoint_fingerprint": EXPORT.checkpoint_fingerprint, "base_reference": base_reference,
            "check_identity": check_identity, "validate_iteration": validate_iteration}


class OfflineExportTests(unittest.TestCase):
    def fixture(self, directory, *, legacy=False):
        root = Path(directory)
        project, checkpoint, cache, output = (root / name for name in ("project", "adapter", "hub", "bundle"))
        package = project / "src" / "avito_candidates"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "cli.py").write_text('''import csv, json, os, sys
from pathlib import Path
def main():
    args = sys.argv[1:]
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_DATASETS_OFFLINE"] == "1"
    assert not os.environ.get("HF_TOKEN")
    root = Path(os.environ["HF_HUB_CACHE"]).parent
    assert "--ef-search" in args if args[0] == "predict" else True
    if args[0] == "predict":
        assert args[args.index("--embedding-model") + 1] == str(root / "models" / "embedding")
        (root / "command.json").write_text(json.dumps(args))
        output = Path(args[args.index("--answer") + 1])
        with output.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows([["query_id", "answer"], ["0123456789ABCDEF", "0123456789abcdef"]])
    else:
        answer = Path(args[args.index("--answer") + 1])
        assert answer.read_text(encoding="utf-8").startswith("query_id,answer")
if __name__ == "__main__": main()
''', encoding="utf-8")
        (project / "pyproject.toml").write_text('[project]\nname="fixture"\nversion="0.1"\n', encoding="utf-8")
        (project / "README.md").write_text("Fixture README", encoding="utf-8")
        (project / ".env").write_text("NEVER_COPY=secret", encoding="utf-8")
        (project / "dataset").mkdir()
        (project / "dataset" / "train.parquet").write_bytes(b"must not copy")
        (project / "scripts").mkdir()
        (project / "scripts" / "safe.py").write_text("print('fixture')", encoding="utf-8")
        documentation = project / "docs" / "figures"
        documentation.mkdir(parents=True)
        (project / "docs" / "research.md").write_text("![curve](figures/curve.png)", encoding="utf-8")
        (project / "docs" / "research.json").write_text('{"fixture": true}', encoding="utf-8")
        (documentation / "curve.png").write_bytes(b"tiny fixture image")
        (documentation / "curve.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
        (documentation / "excluded.parquet").write_bytes(b"dataset must not copy")
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": "Octen/Octen-Embedding-0.6B", "revision": None}), encoding="utf-8")
        (checkpoint / "adapter_model.safetensors").write_bytes(b"unchanged adapter fixture")
        (checkpoint / "modules.json").write_text("[]", encoding="utf-8")
        repo = cache / "models--Octen--Octen-Embedding-0.6B"
        snapshot = repo / "snapshots" / EXPORT.OCTEN_REVISION
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text('{"fixture": true}', encoding="utf-8")
        # HF new-format blob links can point outside this repository's cache.
        shared_blob = cache / "blobs" / "67" / "global-model-blob"
        shared_blob.parent.mkdir(parents=True)
        shared_blob.write_bytes(b"small base weights fixture")
        try:
            (snapshot / "model.safetensors").symlink_to(shared_blob)
        except OSError:
            (snapshot / "model.safetensors").write_bytes(shared_blob.read_bytes())
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text(EXPORT.OCTEN_REVISION, encoding="utf-8")
        (cache / "token").write_text("credential must not copy", encoding="utf-8")
        artifact = {"model": FixtureModel(), "estimator": "histgb", "selected_iteration": None,
                    "priors": {"opaque": "unchanged"}, "metadata": {
                        "feature_version": 4, "feature_names": ["x"], "corpus_fingerprint": "fixture",
                        "retrieval_config": {"method": "hybrid", "embedding_model": str(checkpoint),
                                             "candidate_k": 449, "ef_search": 1000, "location_bonus": 0.02,
                                             "geo_candidate_k": 100, "channel_quota": 0,
                                             "embedding_max_length": 128, "rescore_dense_union": True}}}
        if not legacy:
            artifact["metadata"].update(embedding_checkpoint_sha256=EXPORT.checkpoint_fingerprint(checkpoint),
                                        embedding_base_repo="Octen/Octen-Embedding-0.6B",
                                        embedding_base_revision=EXPORT.OCTEN_REVISION, embedding_is_adapter=True)
        model = root / "reranker.joblib"
        joblib.dump(artifact, model)
        requirements = root / "requirements.txt"
        requirements.write_text("joblib==1.6.0\n", encoding="utf-8")
        answer = root / "submission.csv"
        answer.write_text("query_id,answer\n0123456789ABCDEF,0123456789abcdef\n", encoding="utf-8")
        return project, model, checkpoint, cache, output, requirements, answer

    def export(self, fixture, **kwargs):
        return EXPORT.export_bundle(*fixture[:5], requirements=fixture[5], answer=fixture[6],
                                    contract=fixture_contract(), **kwargs)

    def test_adapter_preserved_global_blob_dereferenced_and_credentials_excluded(self):
        with TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            project, model, checkpoint, cache, output, _, _ = fixture
            original_model, adapter_config = model.read_bytes(), (checkpoint / "adapter_config.json").read_bytes()
            manifest = self.export(fixture, selected_iteration=50)
            self.assertEqual(model.read_bytes(), original_model)
            self.assertEqual((output / "models/embedding/adapter_config.json").read_bytes(), adapter_config)
            self.assertEqual(EXPORT.checkpoint_fingerprint(checkpoint),
                             EXPORT.checkpoint_fingerprint(output / "models/embedding"))
            weights = output / f"hf-cache/models--Octen--Octen-Embedding-0.6B/snapshots/{EXPORT.OCTEN_REVISION}/model.safetensors"
            self.assertEqual(weights.read_bytes(), b"small base weights fixture")
            self.assertFalse(weights.is_symlink())
            self.assertFalse((output / "hf-cache/token").exists())
            self.assertFalse((output / "project/.env").exists())
            self.assertFalse((output / "project/dataset").exists())
            self.assertEqual((output / "project/docs/research.md").read_text(encoding="utf-8"),
                             "![curve](figures/curve.png)")
            self.assertEqual((output / "project/docs/figures/curve.png").read_bytes(), b"tiny fixture image")
            self.assertTrue((output / "project/docs/research.json").is_file())
            self.assertTrue((output / "project/docs/figures/curve.svg").is_file())
            self.assertFalse((output / "project/docs/figures/excluded.parquet").exists())
            self.assertFalse((output / "INCOMPLETE").exists())
            self.assertEqual(manifest["base_revision"], EXPORT.OCTEN_REVISION)
            self.assertEqual(manifest["selected_iteration"], 50)
            self.assertFalse(manifest["adapter_merged"])
            bundled = joblib.load(output / "models/reranker.joblib")
            self.assertEqual(bundled["selected_iteration"], 50)
            self.assertEqual(bundled["priors"], {"opaque": "unchanged"})
            self.assertEqual(bundled["metadata"]["retrieval_config"]["embedding_model"], str(checkpoint))
            for relative, digest in manifest["files"].items():
                self.assertEqual(EXPORT.sha256(output / relative), digest)

    def test_legacy_requires_explicit_independent_identity_attestation(self):
        with TemporaryDirectory() as directory:
            fixture = self.fixture(directory, legacy=True)
            with self.assertRaisesRegex(ValueError, "Legacy.*attestation"):
                self.export(fixture)
            self.assertFalse(fixture[4].exists())
            original = fixture[1].read_bytes()
            result = self.export(fixture, attest_legacy_checkpoint=True,
                                 expected_checkpoint_sha=EXPORT.checkpoint_fingerprint(fixture[2]),
                                 expected_base_revision=EXPORT.OCTEN_REVISION)
            self.assertEqual(fixture[1].read_bytes(), original)
            self.assertFalse(result["legacy_attestation"]["training_identity_proven_by_export"])
            self.assertEqual(result["legacy_attestation"]["source_artifact_sha256"], EXPORT.sha256(fixture[1]))
            self.assertIn("embedding_checkpoint_sha256", joblib.load(fixture[4] / "models/reranker.joblib")["metadata"])

    def test_wrong_checkpoint_sha_and_base_commit_rejected_before_output(self):
        for option in ("checkpoint", "base"):
            with self.subTest(option=option), TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                args = {"expected_checkpoint_sha": "0" * 64} if option == "checkpoint" else {"expected_base_revision": "0" * 40}
                with self.assertRaisesRegex(ValueError, "differs"):
                    self.export(fixture, **args)
                self.assertFalse(fixture[4].exists())

    def test_metadata_mismatch_rejected_before_output(self):
        for key, value in (("embedding_checkpoint_sha256", "0" * 64),
                           ("embedding_base_revision", "0" * 40), ("embedding_base_revision", None)):
            with self.subTest(key=key, value=value), TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                artifact = joblib.load(fixture[1])
                artifact["metadata"][key] = value
                joblib.dump(artifact, fixture[1])
                with self.assertRaisesRegex(ValueError, "identity is not verified"):
                    self.export(fixture)
                self.assertFalse(fixture[4].exists())

    def test_existing_output_invalid_prefix_and_unknown_config_refused(self):
        for mode in ("existing", "prefix", "config"):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                if mode == "existing":
                    fixture[4].mkdir()
                if mode == "config":
                    artifact = joblib.load(fixture[1])
                    artifact["metadata"]["retrieval_config"]["unknown_unsafe"] = 1
                    joblib.dump(artifact, fixture[1])
                with self.assertRaises(ValueError):
                    self.export(fixture, selected_iteration=151 if mode == "prefix" else 50)

    def test_sensitive_file_in_model_refused_as_incomplete(self):
        with TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            (fixture[2] / "token").write_text("secret", encoding="utf-8")
            artifact = joblib.load(fixture[1])
            artifact["metadata"]["embedding_checkpoint_sha256"] = EXPORT.checkpoint_fingerprint(fixture[2])
            joblib.dump(artifact, fixture[1])
            with self.assertRaisesRegex(ValueError, "Credential-like"):
                self.export(fixture)
            self.assertTrue((fixture[4] / "INCOMPLETE").is_file())
            self.assertFalse((fixture[4] / "manifest.json").exists())

    def test_wrapper_offline_relocated_args_prediction_and_validation(self):
        with TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            self.export(fixture, selected_iteration=50, microcategory_exact_bonus=0.05)
            output = fixture[4]
            dataset = Path(directory) / "external-dataset"
            dataset.mkdir()
            answer = Path(directory) / "new-answer.csv"
            env = {**os.environ, "HF_TOKEN": "do not forward this token"}
            result = subprocess.run([sys.executable, str(output / "run_predict.py"), "--data-dir", str(dataset),
                                     "--output", str(answer), "--device", "cuda", "--embedding-batch-size", "32"],
                                    capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(answer.is_file())
            args = json.loads((output / "command.json").read_text())
            self.assertEqual(args[args.index("--ef-search") + 1], "1000")
            self.assertEqual(args[args.index("--microcategory-exact-bonus") + 1], "0.05")
            self.assertIn("--rescore-dense-union", args)

    def test_wrapper_detects_tampered_adapter(self):
        with TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            self.export(fixture)
            output = fixture[4]
            (output / "models/embedding/adapter_model.safetensors").write_bytes(b"changed")
            result = subprocess.run([sys.executable, str(output / "run_predict.py"), "--data-dir", directory],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Bundled file content changed", result.stderr)
            self.assertFalse((output / "command.json").exists())

    def test_bundle_can_move_without_changing_adapter_config_or_saved_paths(self):
        with TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            self.export(fixture, selected_iteration=50)
            moved = Path(directory) / "other-location" / "portable-bundle"
            moved.parent.mkdir()
            fixture[4].rename(moved)
            result = subprocess.run([sys.executable, str(moved / "run_predict.py"), "--data-dir", directory,
                                     "--output", str(Path(directory) / "moved-answer.csv")],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(directory) / "moved-answer.csv").is_file())
            self.assertEqual(EXPORT.checkpoint_fingerprint(fixture[2]),
                             EXPORT.checkpoint_fingerprint(moved / "models/embedding"))

    def test_nested_output_and_path_traversal_revision_refused(self):
        with TemporaryDirectory() as directory:
            fixture = list(self.fixture(directory))
            fixture[4] = fixture[0] / "exports" / ".." / "bundle"
            with self.assertRaisesRegex(ValueError, "must not be nested"):
                self.export(fixture)
            config_path = fixture[2] / "adapter_config.json"
            config = json.loads(config_path.read_text())
            config["revision"] = "../../token"
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "Unsafe adapter revision"):
                EXPORT.resolve_adapter_base(fixture[2], fixture[3])


if __name__ == "__main__":
    unittest.main()
