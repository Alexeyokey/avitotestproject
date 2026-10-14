"""Freeze inference-prior provenance after both audit caches have completed.

Run under the frozen package PYTHONPATH used by the cache producer so joblib
can deserialize its prior classes. This updates only the two JSON manifests;
candidate rows and model artifacts are never changed.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import tempfile


def _number(value):
    """Normalize equivalent int/float counts without losing integral precision."""
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Inference prior counts must be finite")
        return int(value) if value.is_integer() else value
    raise TypeError(f"Unsupported inference prior count: {type(value).__name__}")


def _counts(value):
    if not isinstance(value, Mapping):
        return _number(value)
    normalized = {}
    for key, count in value.items():
        name = str(key)
        if name in normalized:
            raise ValueError(f"Prior keys collide after string conversion: {name!r}")
        normalized[name] = _counts(count)
    return {key: normalized[key] for key in sorted(normalized)}


def _attribute(prior, name):
    if isinstance(prior, Mapping):
        if name not in prior:
            raise ValueError(f"Inference prior lacks {name}")
        return prior[name]
    if not hasattr(prior, name):
        raise ValueError(f"Inference prior lacks {name}")
    return getattr(prior, name)


def inference_prior_fingerprint(priors):
    """Hash only the aggregate fields used by inference, with canonical ordering.

    Geography query_counts is training-only attribution and is deliberately
    ignored. Containers may be real prior instances or plain mock mappings.
    Empty/None priors both represent the absence of inference histories.
    """
    if priors is None:
        priors = {}
    if not isinstance(priors, Mapping):
        raise TypeError("Priors must be a mapping or None")
    microcategory = priors.get("microcategory_prior")
    geography = priors.get("geo_associations")
    payload = {"microcategory_prior": None, "geo_associations": None}
    if microcategory is not None:
        payload["microcategory_prior"] = {
            name: _counts(_attribute(microcategory, name))
            for name in ("exact_counts", "token_counts", "token_totals")}
    if geography is not None:
        locations = _attribute(geography, "corpus_locations")
        if isinstance(locations, (str, bytes)):
            raise TypeError("corpus_locations must be a collection of location IDs")
        payload["geo_associations"] = {
            "corpus_locations": sorted({str(location) for location in locations}),
            "counts": _counts(_attribute(geography, "counts")),
            "totals": _counts(_attribute(geography, "totals")),
        }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def artifact_prior_policy(artifact):
    """Return the two mandatory policy fields that audit consumers must match."""
    if not isinstance(artifact, Mapping):
        raise TypeError("Expected a reranker artifact mapping")
    return {
        "inference_priors_sha256": inference_prior_fingerprint(artifact.get("priors")),
        "disable_microcategory_prior": bool(artifact.get("disable_microcategory_prior", False)),
    }


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completed_manifest(cache_dir, role):
    parquet = cache_dir / f"{role}.parquet"
    manifest = cache_dir / f"{role}.json"
    if not parquet.is_file() or not manifest.is_file():
        raise ValueError(f"Audit cache is not complete: {parquet} and {manifest} are required")
    # The producer writes final names and its JSON completion marker last.
    # Verify the Parquet end marker as an additional truncation guard; the
    # actual audit reader checks row counts and semantic contents separately.
    size = parquet.stat().st_size
    if size < 12:
        raise ValueError(f"Incomplete Parquet file: {parquet}")
    with parquet.open("rb") as stream:
        beginning = stream.read(4)
        stream.seek(-8, os.SEEK_END)
        footer = stream.read(8)
    footer_size = int.from_bytes(footer[:4], "little")
    if beginning != b"PAR1" or footer[4:] != b"PAR1" or footer_size > size - 12:
        raise ValueError(f"Incomplete Parquet file: {parquet}")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if metadata.get("role") != role or not isinstance(metadata.get("groups"), list):
        raise ValueError(f"Invalid completed cache manifest: {manifest}")
    return manifest, metadata


def _reference_path(cache_dir, value):
    path = Path(value)
    if path.is_file():
        return path.resolve()
    if not path.is_absolute():
        # model_reference historically used a project-relative artifact path.
        # Locate it from the cache's ancestors when invoked outside that cwd.
        for ancestor in (cache_dir, *cache_dir.parents):
            candidate = ancestor / path
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(f"Reference reranker artifact not found: {path}")


def _atomic_json(path, metadata):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def finalize_cache_provenance(cache_dir, reference_model=None, *, validation_only=False):
    """Annotate completed JSONs, rejecting changed model bytes or prior policy.

    A missing reference path is inferred from validation.json:model_reference.
    Both manifests are checked before either is replaced by default. The
    explicit validation_only mode checks and annotates only validation; it
    never silently substitutes or skips a required fit-side cache. Repeated
    calls with the same policy are idempotent.
    """
    cache_dir = Path(cache_dir).resolve()
    roles = ("validation",) if validation_only else ("validation", "train_sample")
    manifests = [_completed_manifest(cache_dir, role) for role in roles]
    reference_model = reference_model or manifests[0][1].get("model_reference")
    if not reference_model:
        raise ValueError("validation.json has no model_reference; pass --reference-model")
    reference = _reference_path(cache_dir, reference_model)
    reference_hash = _file_sha256(reference)
    for path, metadata in manifests:
        if metadata.get("model_sha256") != reference_hash:
            raise ValueError(f"Reference model SHA256 differs from cache producer: {path}")
    import joblib
    artifact = joblib.load(reference)
    if not isinstance(artifact, Mapping) or "model" not in artifact or "metadata" not in artifact:
        raise ValueError("Expected a saved reranker artifact with model and metadata")
    policy = {**artifact_prior_policy(artifact), "reference_artifact_sha256": reference_hash}
    for path, metadata in manifests:
        for key, value in policy.items():
            if key in metadata and metadata[key] != value:
                raise ValueError(f"Existing {key} differs from reference inference policy: {path}")
    updated = []
    for path, metadata in manifests:
        if any(key not in metadata for key in policy):
            _atomic_json(path, {**metadata, **policy})
            updated.append(str(path))
    return {"cache_dir": str(cache_dir), "reference_model": str(reference),
            "validation_only": bool(validation_only), "roles": list(roles),
            **policy, "updated_metadata_files": updated}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path)
    parser.add_argument("--validation-only", action="store_true",
                        help="Explicitly finalize validation only; default requires both completed caches")
    args = parser.parse_args(argv)
    result = finalize_cache_provenance(args.cache_dir, args.reference_model,
                                       validation_only=args.validation_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
