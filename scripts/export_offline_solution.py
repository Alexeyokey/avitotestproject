"""Export a relocatable, offline inference bundle without merging LoRA weights.

The input joblib must be a trusted local artifact (joblib loading executes Python
objects). Only the selected model snapshot is copied, never HF_HOME, credentials,
the training data or the whole Hugging Face cache. Existing model files stay
unchanged. A legacy artifact is refused unless an explicit provenance attestation
is supplied; that attestation is recorded, not presented as proof of training.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys


OCTEN_REVISION = "d715b32ee68f057b54dff09fc93c23485bc403d3"
SENSITIVE_NAMES = {"token", ".token", "stored_tokens", "credentials", "credentials.json",
                   ".env", "id_rsa", "id_ed25519", "authorized_keys", "known_hosts"}
CONFIG_KEYS = {
    "method", "candidate_k", "title_k1", "title_b", "title_weight", "params_k1",
    "params_b", "params_weight", "description_k1", "description_b", "description_weight",
    "stem_title_weight", "stem_params_weight", "location_bonus", "bm25_rrf_k",
    "bm25_channel_quota", "geo_candidate_k", "geo_top_locations", "geo_min_history",
    "geo_weight", "geo_include_related", "embedding_model", "embedding_max_length",
    "embedding_params_words", "embedding_description_words", "ef_search", "rrf_k",
    "bm25_weight", "dense_weight", "channel_quota", "rescore_dense_union",
}
BOOL_KEYS = {"geo_include_related", "rescore_dense_union"}
MICROCATEGORY_CLI_KEYS = {"microcategory_model", "microcategory_candidate_k",
                         "microcategory_field", "microcategory_weight", "microcategory_top_categories"}
DERIVED_FIXED_KEYS = {
    "microcategory_selection_policy": "top998-then-novel-prefix",
    "supplemental_raw_depth": 998,
    "supplemental_rank_constant": 60,
    "supplemental_selection_policy": "raw-ranks-equal-route-novel-rrf-id-ties",
}
DERIVED_KEYS = {*DERIVED_FIXED_KEYS, "microcategory_model_sha256"}
PORTABLE_MICROCATEGORY = "../models/microcategory.joblib"
SOFT_MICROCATEGORY_SCHEMA = "soft-microcategory-word-char-values-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_fingerprint(path):
    """Match dense.local_checkpoint_content_fingerprint, including all filenames."""
    path = Path(path)
    files = sorted(file for file in path.rglob("*") if file.is_file())
    if not files:
        return None
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file.stat().st_size.to_bytes(8, "big"))
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


@contextmanager
def temporary_environment(**values):
    old = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update({key: str(value) for key, value in values.items()})
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def project_contract(project_root):
    """Use the actual project's identity and model-prefix guards, without a GPU."""
    source = str(Path(project_root).resolve() / "src")
    sys.path.insert(0, source)
    dense = importlib.import_module("avito_candidates.dense")
    reranker = importlib.import_module("avito_candidates.reranker")
    for module in (dense, reranker):
        if not Path(module.__file__).resolve().is_relative_to(Path(source)):
            raise ValueError("Loaded project package differs from the supplied export source")
    return {"checkpoint_fingerprint": dense.local_checkpoint_content_fingerprint,
            "base_reference": dense.offline_adapter_base_reference,
            "check_identity": reranker.check_retrieval_identity,
            "validate_iteration": reranker.validate_selected_iteration}


def _safe_file_name(path):
    if any(part.lower() in SENSITIVE_NAMES or part.lower().startswith(".env.")
           for part in Path(path).parts):
        raise ValueError(f"Credential-like filename is not exported: {path}")


def copy_files_tree(source, target):
    """Dereference file symlinks, including HF's cross-repository blob links."""
    source, target = Path(source), Path(target)
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Model directory contains no files: {source}")
    for path in source.rglob("*"):
        if path.is_symlink() and path.is_dir():
            raise ValueError(f"Directory symlink is unsupported in model snapshot: {path}")
        if path.is_symlink() and not path.exists():
            raise ValueError(f"Broken model snapshot symlink: {path}")
    for path in files:
        relative = path.relative_to(source)
        _safe_file_name(relative)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination, follow_symlinks=True)


def resolve_adapter_base(checkpoint, hub_cache, *, expected_revision=None):
    """Resolve the adapter's actual cached ref, and return the one snapshot only."""
    configs = sorted(Path(checkpoint).rglob("adapter_config.json"))
    if not configs:
        raise ValueError("Exporter requires an unchanged local LoRA adapter checkpoint")
    references = set()
    for path in configs:
        config = json.loads(path.read_text(encoding="utf-8"))
        references.add((config.get("base_model_name_or_path"), config.get("revision") or "main"))
    if len(references) != 1:
        raise ValueError("Conflicting adapter base references")
    repo, revision = references.pop()
    if not isinstance(repo, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("Adapter must reference a Hugging Face model repository")
    if not isinstance(revision, str) or not re.fullmatch(r"[\w./-]+", revision):
        raise ValueError("Unsafe adapter revision")
    if ".." in Path(revision).parts or revision.startswith("/"):
        raise ValueError("Unsafe adapter revision")
    repo_cache = Path(hub_cache) / ("models--" + repo.replace("/", "--"))
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        commit = revision.lower()
    else:
        ref = repo_cache / "refs" / revision
        if not ref.is_file():
            raise ValueError(f"Cached base reference is missing: {ref}")
        commit = ref.read_text(encoding="utf-8").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Cached base revision is not a full commit SHA")
    expected = expected_revision
    if expected is None and repo == "Octen/Octen-Embedding-0.6B":
        expected = OCTEN_REVISION
    if expected is not None and commit != expected.lower():
        raise ValueError("Cached base revision differs from the expected verified commit")
    snapshot = repo_cache / "snapshots" / commit
    if not snapshot.is_dir():
        raise ValueError(f"Cached base snapshot is missing: {snapshot}")
    return {"repo": repo, "revision": commit, "requested_revision": revision,
            "snapshot": snapshot, "repo_cache_name": repo_cache.name}


def validate_extended_config(config):
    """Derived identity fields are checked, never emitted as nonexistent CLI flags."""
    micro = bool(set(config) & (MICROCATEGORY_CLI_KEYS | {"microcategory_model_sha256",
                                                       "microcategory_selection_policy"}))
    supplemental = bool(set(config) & ({"supplemental_fill"} | {
        key for key in DERIVED_FIXED_KEYS if key.startswith("supplemental_")}))
    if micro:
        required = MICROCATEGORY_CLI_KEYS | {"microcategory_model_sha256", "microcategory_selection_policy"}
        if not required <= set(config) or config.get("method") != "hybrid":
            raise ValueError("Incomplete or non-hybrid microcategory retrieval settings")
        if (not isinstance(config["microcategory_model"], str) or not config["microcategory_model"]
                or not re.fullmatch(r"[0-9a-f]{64}", str(config["microcategory_model_sha256"]))
                or isinstance(config["microcategory_candidate_k"], bool)
                or not isinstance(config["microcategory_candidate_k"], int)
                or not 1 <= config["microcategory_candidate_k"] <= 998
                or config["microcategory_field"] not in {"combined", "title", "params", "description"}
                or isinstance(config["microcategory_top_categories"], bool)
                or not isinstance(config["microcategory_top_categories"], int)
                or config["microcategory_top_categories"] < 1
                or isinstance(config["microcategory_weight"], bool)
                or not isinstance(config["microcategory_weight"], (int, float))
                or not math.isfinite(config["microcategory_weight"]) or config["microcategory_weight"] < 0):
            raise ValueError("Invalid microcategory retrieval settings")
    if supplemental:
        required = {"supplemental_fill", *(key for key in DERIVED_FIXED_KEYS
                                          if key.startswith("supplemental_"))}
        if (not required <= set(config) or not micro
                or config["supplemental_fill"] != "micro-typed-dense-v1"
                or config["microcategory_candidate_k"] != 998
                or config["microcategory_field"] != "combined"
                or config["microcategory_top_categories"] != 3
                or config.get("embedding_document_format", "current") != "current"
                or config.get("bm25_weight", 0) <= 0 or config.get("dense_weight", 0) <= 0):
            raise ValueError("Incomplete or unsupported supplemental-fill retrieval settings")
    for key, expected in DERIVED_FIXED_KEYS.items():
        if key in config and (type(config[key]) is not type(expected) or config[key] != expected):
            raise ValueError(f"Unsupported derived retrieval identity: {key}")
    if "embedding_document_format" in config and config["embedding_document_format"] not in {"compact", "title_description"}:
        raise ValueError("Unsupported non-default embedding document format")


def portable_config_args(config, *, checkpoint, microcategory=None):
    """Serialize every saved retrieval setting; fail on unknown settings."""
    unknown = set(config) - (CONFIG_KEYS | MICROCATEGORY_CLI_KEYS | DERIVED_KEYS
                              | {"supplemental_fill", "embedding_document_format"})
    if unknown:
        raise ValueError(f"Unrecognized retrieval settings: {sorted(unknown)}")
    if config.get("method") not in {"dense", "hybrid"} or not config.get("embedding_model"):
        raise ValueError("Offline LoRA bundle requires dense or hybrid retrieval")
    validate_extended_config(config)
    arguments = []
    for key, value in sorted(config.items()):
        if key in DERIVED_KEYS:
            continue
        option = "--" + key.replace("_", "-")
        if key in BOOL_KEYS:
            if not isinstance(value, bool):
                raise ValueError(f"Boolean retrieval setting is invalid: {key}")
            if value:
                arguments.append(option)
        elif value is None or not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise ValueError(f"Retrieval setting cannot be serialized: {key}")
        else:
            if key == "embedding_model":
                value = checkpoint
            elif key == "microcategory_model" and microcategory is not None:
                value = microcategory
            arguments.extend((option, str(value)))
    return arguments


def _text_inventory_digest(texts):
    serialized = json.dumps(sorted(texts), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def verify_microcategory_artifact(artifact, project_root, *, joblib_module):
    """Bind the external file to the embedded immutable TF-IDF/SGD/OOF models.

    Only trusted local joblib files may be loaded. Estimator content hashes are
    integrity attestations, not signatures or evidence of unbiased training.
    """
    metadata = artifact["metadata"]
    config = metadata["retrieval_config"]
    version = metadata.get("feature_version")
    has_micro = "microcategory_model" in config
    if not has_micro:
        if version in {7, 8}:
            raise ValueError("V7/V8 export requires the recorded microcategory model")
        return None
    if version not in {7, 8}:
        raise ValueError("Microcategory bundle export currently requires saved V7/V8 inference priors")
    if version == 8 and config.get("rescore_dense_union") is not True:
        raise ValueError("V8 export requires measured full-pool dense rescoring")
    classifier = artifact.get("priors", {}).get("microcategory_classifier")
    snapshot = metadata.get("microcategory_classifier_metadata")
    if (classifier is None or not isinstance(snapshot, dict) or classifier.metadata != snapshot
            or metadata.get("training_prior_policy", {}).get("microcategory_classifier") != "grouped-out-of-fold"):
        raise ValueError("V7/V8 export requires matching embedded OOF microcategory metadata")
    source = Path(config["microcategory_model"])
    if not source.is_absolute():
        source = Path(project_root) / source
    source = source.resolve()
    sidecar = source.with_suffix(source.suffix + ".json")
    if not source.is_file() or not sidecar.is_file():
        raise ValueError("Recorded microcategory artifact or sidecar is missing")
    model_sha, sidecar_sha = sha256(source), sha256(sidecar)
    envelope = json.loads(sidecar.read_text(encoding="utf-8"))
    if model_sha != config["microcategory_model_sha256"] or envelope.get("model_sha256") != model_sha:
        raise ValueError("Microcategory artifact bytes differ from recorded SHA")
    if envelope.get("metadata") != snapshot:
        raise ValueError("Microcategory sidecar differs from the embedded metadata snapshot")
    folds = snapshot.get("crossfit_folds")
    corpus = metadata.get("corpus_fingerprint")
    if (snapshot.get("schema") != SOFT_MICROCATEGORY_SCHEMA
            or isinstance(folds, bool) or folds not in {3, 5}
            or snapshot.get("corpus_fingerprint") != corpus
            or not isinstance(corpus, str) or not re.fullmatch(r"[0-9a-f]{64}", corpus)
            or not re.fullmatch(r"[0-9a-f]{64}", str(snapshot.get("source_sha256", "")))
            or snapshot.get("seed") != metadata.get("seed")
            or metadata.get("split") not in {"queries", "all"}
            or snapshot.get("fit_split", metadata["split"]) != metadata["split"]
            or (metadata["split"] == "queries" and snapshot.get("validation_fraction", .2)
                != metadata.get("validation_fraction"))):
        raise ValueError("Microcategory corpus, source, query split or OOF provenance differs")
    payload = joblib_module.load(source)
    if not isinstance(payload, dict) or payload.get("metadata") != snapshot:
        raise ValueError("Microcategory payload metadata differs")
    observed = frozenset(classifier.model.fitted_texts)
    classes = list(classifier.model.classes)
    if (not observed or _text_inventory_digest(observed) != snapshot.get("observed_fit_texts_sha256")
            or len(observed) != snapshot.get("fit_texts_count")
            or snapshot.get("allowed_fit_texts_count", 0) < len(observed)
            or not re.fullmatch(r"[0-9a-f]{64}", str(snapshot.get("allowed_fit_texts_sha256", "")))
            or classes != snapshot.get("classes")
            or set(classifier.crossfit_models) != set(range(folds))
            or set(payload.get("crossfit_models", {})) != set(range(folds))):
        raise ValueError("Microcategory fitted inventory, classes or OOF fold set differs")
    seed = snapshot["seed"]
    for fold, model in classifier.crossfit_models.items():
        expected = frozenset(text for text in observed if int.from_bytes(hashlib.sha256(
            json.dumps([seed, text], ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")).encode("utf-8")).digest()[:8], "big") % folds != fold)
        if frozenset(model.fitted_texts) != expected:
            raise ValueError("Microcategory cross-fit model contains an incorrect training-text inventory")
    content = {}
    for name, embedded, external in [("global", classifier.model, payload.get("model")),
            *((f"fold-{fold}", classifier.crossfit_models[fold], payload["crossfit_models"][fold])
              for fold in range(folds))]:
        digest = joblib_module.hash(embedded, hash_name="sha1")
        if external is None or joblib_module.hash(external, hash_name="sha1") != digest:
            raise ValueError(f"Embedded and external immutable microcategory model differs: {name}")
        content[name] = digest
    # Prediction caches are transient; clearing one never changes fitted state.
    classifier._prediction_cache.clear()
    return {"source": source, "sidecar": sidecar, "model_sha256": model_sha,
            "sidecar_sha256": sidecar_sha, "metadata": dict(snapshot),
            "immutable_model_hashes": content, "immutable_hash_algorithm": "joblib-sha1"}


def copy_project(project_root, target):
    """Copy inference code and established docs, excluding data, cache and Git."""
    root, target = Path(project_root), Path(target)
    if not (root / "src" / "avito_candidates" / "cli.py").is_file():
        raise ValueError("Project source package is missing")
    if not (root / "pyproject.toml").is_file():
        raise ValueError("Project pyproject.toml is missing")
    code_suffixes = {".py", ".md", ".txt", ".toml", ".sh", ".json"}
    documentation_suffixes = {".md", ".txt", ".json", ".svg", ".png"}
    for directory in ("src", "scripts", "tests", "docs"):
        allowed_suffixes = documentation_suffixes if directory == "docs" else code_suffixes
        for path in sorted((root / directory).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
                continue
            if any(part in {"__pycache__", ".git", ".venv"} or part.endswith(".egg-info")
                   for part in path.relative_to(root).parts):
                continue
            _safe_file_name(path.relative_to(root))
            destination = target / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination, follow_symlinks=True)
    for name in ("pyproject.toml", "README.md", "SOLUTION.md", "requirements-lock.txt"):
        if (root / name).is_file():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, target / name)


WRAPPER = '''"""Run this bundle locally: python run_predict.py --data-dir /path/to/dataset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("answer.csv"))
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.embedding_batch_size < 1:
        parser.error("Embedding batch size must be positive")
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    # This detects accidental changes; the manifest is not a signed authenticity proof.
    for relative, expected in manifest["files"].items():
        path = root / relative
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError("Missing or unsafe bundled file: " + relative)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("Bundled file content changed: " + relative)
    runtime = json.loads((root / "runtime.json").read_text(encoding="utf-8"))
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_CACHE=str(root / "hf-cache"),
               HUGGINGFACE_HUB_CACHE=str(root / "hf-cache"),
               HF_HOME=str(root / "hf-home"), PYTHONPATH=str(root / "project" / "src"))
    for credential in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        env.pop(credential, None)
    config_args = []
    for option, value in runtime["retrieval_arguments"]:
        config_args.append(option)
        if value is not None:
            config_args.append(str(root / "models" / "embedding") if option == "--embedding-model" else value)
    command = [sys.executable, "-m", "avito_candidates.cli", "predict", *config_args,
               "--data-dir", str(args.data_dir.resolve()), "--answer", str(args.output.resolve()),
               "--reranker-model", str(root / "models" / "reranker.joblib"),
               "--device", args.device, "--embedding-batch-size", str(args.embedding_batch_size),
               "--dense-cache", str(root / "generated" / "dense"),
               "--microcategory-exact-bonus", str(runtime["microcategory_exact_bonus"])]
    subprocess.run(command, cwd=root / "project", env=env, check=True)
    subprocess.run([sys.executable, "-m", "avito_candidates.cli", "validate", "--data-dir",
                    str(args.data_dir.resolve()), "--answer", str(args.output.resolve())],
                   cwd=root / "project", env=env, check=True)

if __name__ == "__main__":
    main()
'''


def export_bundle(project_root, reranker_path, checkpoint, hub_cache, output_dir, *,
                  requirements=None, answer=None, selected_iteration=None,
                  microcategory_exact_bonus=0.0, expected_checkpoint_sha=None,
                  expected_base_revision=None, attest_legacy_checkpoint=False,
                  contract=None):
    """Export to a new directory; a failed copy is left explicit and incomplete."""
    import joblib

    root, reranker_path = Path(project_root).resolve(), Path(reranker_path).resolve()
    checkpoint, hub_cache = Path(checkpoint).resolve(), Path(hub_cache).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("Output directory already exists; choose a new bundle path")
    if not checkpoint.is_dir() or not reranker_path.is_file():
        raise ValueError("Checkpoint or reranker input is missing")
    if output.is_relative_to(root) or output.is_relative_to(checkpoint) or output.is_relative_to(hub_cache):
        raise ValueError("Bundle must not be nested inside the copied project or model/cache trees")
    if requirements is None or not Path(requirements).is_file():
        raise ValueError("Runtime requirements file is missing")
    if answer is not None and not Path(answer).is_file():
        raise ValueError("Answer CSV is missing")
    if not isinstance(microcategory_exact_bonus, (int, float)) or not 0 <= microcategory_exact_bonus <= 1:
        raise ValueError("Microcategory exact bonus must be finite and between zero and one")
    base = resolve_adapter_base(checkpoint, hub_cache, expected_revision=expected_base_revision)
    artifact = joblib.load(reranker_path)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("metadata"), dict):
        raise ValueError("Reranker artifact lacks metadata")
    contract = contract or project_contract(root)
    actual_sha = contract["checkpoint_fingerprint"](checkpoint)
    if actual_sha != checkpoint_fingerprint(checkpoint):
        raise ValueError("Exporter checkpoint signature differs from the project contract")
    if expected_checkpoint_sha is not None and actual_sha != expected_checkpoint_sha:
        raise ValueError("Checkpoint differs from expected independently verified content SHA")
    metadata = artifact["metadata"]
    legacy = not metadata.get("embedding_checkpoint_sha256")
    attestation = None
    if legacy:
        if not attest_legacy_checkpoint or not expected_checkpoint_sha or not expected_base_revision:
            raise ValueError("Legacy reranker relocation requires explicit checkpoint/base attestation")
        if metadata.get("retrieval_config", {}).get("embedding_model") != str(checkpoint):
            raise ValueError("Legacy attestation must use the original recorded checkpoint path")
        metadata.update(embedding_checkpoint_sha256=actual_sha, embedding_is_adapter=True,
                        embedding_base_repo=base["repo"], embedding_base_revision=base["revision"])
        attestation = {"kind": "explicit-legacy-checkpoint-attestation",
                       "source_artifact_sha256": sha256(reranker_path),
                       "expected_checkpoint_sha256": expected_checkpoint_sha,
                       "expected_base_revision": expected_base_revision,
                       "training_identity_proven_by_export": False}
    saved_revision = metadata.get("embedding_base_revision")
    if (metadata.get("embedding_checkpoint_sha256") != actual_sha
            or metadata.get("embedding_base_repo") != base["repo"]
            or saved_revision != base["revision"]
            or not metadata.get("embedding_is_adapter", False)):
        raise ValueError("Reranker checkpoint or cached base identity is not verified")
    selected = artifact.get("selected_iteration") if selected_iteration is None else selected_iteration
    selected = contract["validate_iteration"](artifact["model"], selected)
    artifact["selected_iteration"] = selected
    if selected_iteration is not None:
        artifact["selection_status"] = "explicit-prefix-export"
        if isinstance(artifact.get("training_configuration"), dict):
            artifact["training_configuration"].update(
                selection_status="explicit-prefix-export", selected_iteration=selected)
    source_sha = sha256(reranker_path)
    if attestation is not None:
        artifact["offline_export_attestation"] = attestation
    config = dict(metadata["retrieval_config"])
    portable_config_args(config, checkpoint=checkpoint)  # validate before writing
    microcategory = verify_microcategory_artifact(artifact, root, joblib_module=joblib)
    with temporary_environment(HF_HUB_CACHE=hub_cache, HF_HUB_OFFLINE="1"):
        contract["check_identity"](metadata, {**config, "embedding_model": str(checkpoint)})
    output.mkdir(parents=True, exist_ok=False)
    (output / "INCOMPLETE").write_text("Export has not completed. Do not use this directory.\n", encoding="utf-8")
    copy_project(root, output / "project")
    copy_files_tree(checkpoint, output / "models" / "embedding")
    portable_config = dict(config)
    if microcategory is not None:
        target = output / "models" / "microcategory.joblib"
        shutil.copy2(microcategory["source"], target, follow_symlinks=True)
        shutil.copy2(microcategory["sidecar"], target.with_suffix(".joblib.json"), follow_symlinks=True)
        if (sha256(target) != microcategory["model_sha256"]
                or sha256(target.with_suffix(".joblib.json")) != microcategory["sidecar_sha256"]):
            raise ValueError("Copied microcategory artifact or sidecar bytes changed")
        # This exact relative string is used both by saved metadata and by CLI
        # from bundle/project. Core path equality is deliberately not weakened.
        portable_config["microcategory_model"] = PORTABLE_MICROCATEGORY
        metadata["retrieval_config"] = dict(portable_config)
        artifact["offline_export_microcategory_attestation"] = {
            "kind": "verified-microcategory-artifact-relocation",
            "original_retrieval_config": config,
            "aliases": {"microcategory_model": {"original": config["microcategory_model"],
                                                   "portable": PORTABLE_MICROCATEGORY,
                                                   "resolution_cwd": "bundle/project"}},
            "source_artifact_sha256": source_sha,
            "microcategory_model_sha256": microcategory["model_sha256"],
            "microcategory_sidecar_sha256": microcategory["sidecar_sha256"],
            "immutable_model_hashes": microcategory["immutable_model_hashes"],
            "immutable_hash_algorithm": microcategory["immutable_hash_algorithm"],
            "microcategory_metadata": microcategory["metadata"],
            "integrity_only_not_a_signature": True,
            "training_identity_proven_by_export": False,
        }
    repo_out = output / "hf-cache" / base["repo_cache_name"]
    copy_files_tree(base["snapshot"], repo_out / "snapshots" / base["revision"])
    # Pin main and any named adapter ref to the exact cached commit; no Hub lookup.
    for reference in {"main", base["requested_revision"]}:
        if not re.fullmatch(r"[0-9a-fA-F]{40}", reference):
            ref = repo_out / "refs" / reference
            ref.parent.mkdir(parents=True, exist_ok=True)
            # HF cache readers use f.read() verbatim as the snapshot directory
            # name. A trailing newline passes our stripped identity check but
            # breaks actual offline AutoConfig/model loading.
            ref.write_text(base["revision"], encoding="utf-8")
    relocated = {**portable_config, "embedding_model": str(output / "models" / "embedding")}
    with temporary_environment(HF_HUB_CACHE=output / "hf-cache", HF_HUB_OFFLINE="1"):
        if contract["checkpoint_fingerprint"](output / "models" / "embedding") != actual_sha:
            raise ValueError("Copied adapter bytes changed")
        exported_base = contract["base_reference"](output / "models" / "embedding", hub_cache=output / "hf-cache")
        if exported_base != {"is_adapter": True, "base_repo": base["repo"], "base_revision": base["revision"]}:
            raise ValueError("Exported adapter base cannot resolve offline")
        contract["check_identity"](metadata, relocated)
    joblib.dump(artifact, output / "models" / "reranker.joblib")
    if requirements is not None:
        shutil.copy2(requirements, output / "requirements-runtime.txt")
    if answer is not None:
        shutil.copy2(answer, output / "answer.csv")
    arguments = []
    for key, value in sorted(portable_config.items()):
        if key in DERIVED_KEYS:
            continue
        option = "--" + key.replace("_", "-")
        if key in BOOL_KEYS:
            if value:
                arguments.append([option, None])
        else:
            arguments.append([option, "models/embedding" if key == "embedding_model" else str(value)])
    runtime = {"retrieval_arguments": arguments, "microcategory_exact_bonus": microcategory_exact_bonus,
               "selected_iteration": selected, "dataset_included": False,
               "dense_index_included": False}
    if microcategory is not None:
        runtime["microcategory_provenance"] = {
            "source_sha256": microcategory["metadata"]["source_sha256"],
            "corpus_fingerprint": microcategory["metadata"]["corpus_fingerprint"],
            "model_sha256": microcategory["model_sha256"],
            "relative_path": PORTABLE_MICROCATEGORY,
            "query_split": metadata["split"], "seed": metadata["seed"],
            "validation_fraction": metadata["validation_fraction"],
        }
    (output / "runtime.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "run_predict.py").write_text(WRAPPER, encoding="utf-8")
    bundle_notice = ("\nV7/V8 research artifacts remain bound to the recorded training-source SHA and\n"
                     "item-corpus fingerprint. Relocation does not make a fresh 200k-catalog\n"
                     "model compatible with the original benchmark catalog. Microcategory\n"
                     "classifier and geography/history priors are saved; prediction does not refit them.\n"
                     if microcategory is not None else "")
    (output / "OFFLINE_README.txt").write_text(
        "Use Python 3.12 and the recorded runtime dependencies. The dataset is supplied separately.\n"
        "Run: python run_predict.py --data-dir /absolute/path/to/dataset --device cuda\n"
        "The wrapper verifies bundled bytes, runs prediction offline and validates answer.csv.\n"
        "No model download or external inference API is used. A first local HNSW index build\n"
        "is required on a new computer; the training-time index is not copied.\n"
        "Package installation is a separate setup step; for disconnected installation prepare\n"
        "a compatible wheelhouse and use pip --no-index --find-links.\n"
        "The manifest provides integrity checks, not a signed authenticity guarantee.\n" + bundle_notice,
        encoding="utf-8")
    manifest = {"bundle_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "source_artifact_sha256": source_sha, "embedding_checkpoint_sha256": actual_sha,
                "base_repo": base["repo"], "base_revision": base["revision"],
                "selected_iteration": selected, "legacy_attestation": attestation,
                "adapter_merged": False, "credential_cache_copied": False,
                "files": {path.relative_to(output).as_posix(): sha256(path)
                          for path in sorted(output.rglob("*"))
                          if path.is_file() and path.name != "INCOMPLETE"}}
    if microcategory is not None:
        manifest["microcategory_export_attestation"] = artifact["offline_export_microcategory_attestation"]
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "INCOMPLETE").unlink()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--reranker-model", type=Path, required=True)
    parser.add_argument("--embedding-checkpoint", type=Path, required=True)
    parser.add_argument("--hub-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--answer", type=Path)
    parser.add_argument("--selected-iteration", type=int)
    parser.add_argument("--microcategory-exact-bonus", type=float, default=0.0)
    parser.add_argument("--expected-checkpoint-sha")
    parser.add_argument("--expected-base-revision")
    parser.add_argument("--attest-legacy-checkpoint", action="store_true")
    args = parser.parse_args()
    result = export_bundle(args.project_root, args.reranker_model, args.embedding_checkpoint,
                           args.hub_cache, args.output_dir, requirements=args.requirements,
                           answer=args.answer, selected_iteration=args.selected_iteration,
                           microcategory_exact_bonus=args.microcategory_exact_bonus,
                           expected_checkpoint_sha=args.expected_checkpoint_sha,
                           expected_base_revision=args.expected_base_revision,
                           attest_legacy_checkpoint=args.attest_legacy_checkpoint)
    print(json.dumps({"output_dir": str(args.output_dir), "file_count": len(result["files"]),
                      "embedding_checkpoint_sha256": result["embedding_checkpoint_sha256"],
                      "base_revision": result["base_revision"],
                      "selected_iteration": result["selected_iteration"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
