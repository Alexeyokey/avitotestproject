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


def portable_config_args(config, *, checkpoint):
    """Serialize every saved retrieval setting; fail on unknown settings."""
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unrecognized retrieval settings: {sorted(unknown)}")
    if config.get("method") not in {"dense", "hybrid"} or not config.get("embedding_model"):
        raise ValueError("Offline LoRA bundle requires dense or hybrid retrieval")
    arguments = []
    for key, value in sorted(config.items()):
        option = "--" + key.replace("_", "-")
        if key in BOOL_KEYS:
            if not isinstance(value, bool):
                raise ValueError(f"Boolean retrieval setting is invalid: {key}")
            if value:
                arguments.append(option)
        elif value is None or not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise ValueError(f"Retrieval setting cannot be serialized: {key}")
        else:
            arguments.extend((option, str(checkpoint) if key == "embedding_model" else str(value)))
    return arguments


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
    with temporary_environment(HF_HUB_CACHE=hub_cache, HF_HUB_OFFLINE="1"):
        contract["check_identity"](metadata, {**config, "embedding_model": str(checkpoint)})
    output.mkdir(parents=True, exist_ok=False)
    (output / "INCOMPLETE").write_text("Export has not completed. Do not use this directory.\n", encoding="utf-8")
    copy_project(root, output / "project")
    copy_files_tree(checkpoint, output / "models" / "embedding")
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
    relocated = {**config, "embedding_model": str(output / "models" / "embedding")}
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
    for key, value in sorted(config.items()):
        option = "--" + key.replace("_", "-")
        if key in BOOL_KEYS:
            if value:
                arguments.append([option, None])
        else:
            arguments.append([option, "models/embedding" if key == "embedding_model" else str(value)])
    runtime = {"retrieval_arguments": arguments, "microcategory_exact_bonus": microcategory_exact_bonus,
               "selected_iteration": selected, "dataset_included": False,
               "dense_index_included": False}
    (output / "runtime.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "run_predict.py").write_text(WRAPPER, encoding="utf-8")
    (output / "OFFLINE_README.txt").write_text(
        "Use Python 3.12 and the recorded runtime dependencies. The dataset is supplied separately.\n"
        "Run: python run_predict.py --data-dir /absolute/path/to/dataset --device cuda\n"
        "The wrapper verifies bundled bytes, runs prediction offline and validates answer.csv.\n"
        "No model download or external inference API is used. A first local HNSW index build\n"
        "is required on a new computer; the training-time index is not copied.\n"
        "Package installation is a separate setup step; for disconnected installation prepare\n"
        "a compatible wheelhouse and use pip --no-index --find-links.\n"
        "The manifest provides integrity checks, not a signed authenticity guarantee.\n",
        encoding="utf-8")
    manifest = {"bundle_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "source_artifact_sha256": source_sha, "embedding_checkpoint_sha256": actual_sha,
                "base_repo": base["repo"], "base_revision": base["revision"],
                "selected_iteration": selected, "legacy_attestation": attestation,
                "adapter_merged": False, "credential_cache_copied": False,
                "files": {path.relative_to(output).as_posix(): sha256(path)
                          for path in sorted(output.rglob("*"))
                          if path.is_file() and path.name != "INCOMPLETE"}}
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
