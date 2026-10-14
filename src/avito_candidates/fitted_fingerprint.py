"""Content fingerprints of fitted state, independent of pickle aliasing.

Every trained attribute is bound. Unordered membership containers are
canonicalized recursively; mutable query-prediction caches are outside the
explicit classifier payload. Count/TfidfVectorizer's _stop_words_id is the
sole estimator-state exclusion: sklearn stores id(stop_words) there as a
runtime cache check, and transform rewrites it after deserialization.
"""
from collections.abc import Mapping
import hashlib
import json
from types import FunctionType, MethodType, ModuleType


FITTED_FINGERPRINT_VERSION = 2


def _canonical_hash(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _type_name(value):
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _transient_field(value, name):
    return (name == "_stop_words_id"
            and type(value).__module__ == "sklearn.feature_extraction.text"
            and type(value).__qualname__ in {"CountVectorizer", "TfidfVectorizer"})


def _state_fingerprint(value, active):
    import joblib
    tag = _type_name(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return _canonical_hash({"type": tag, "value": value})
    container = isinstance(value, (Mapping, set, frozenset, list, tuple))
    stateful = (hasattr(value, "__dict__")
                and not isinstance(value, (type, FunctionType, MethodType, ModuleType)))
    if not container and not stateful:
        # Numeric arrays are hashed directly by joblib, including dtype,
        # shape/order and the full data buffer, without another serialized copy.
        return _canonical_hash({"type": tag, "joblib_sha1": joblib.hash(value, hash_name="sha1")})
    if id(value) in active:
        raise ValueError("Cyclic fitted state cannot be fingerprinted safely")
    active.add(id(value))
    try:
        if isinstance(value, Mapping):
            contents = sorted([_state_fingerprint(key, active), _state_fingerprint(item, active)]
                              for key, item in value.items())
        elif isinstance(value, (set, frozenset)):
            contents = sorted(_state_fingerprint(item, active) for item in value)
        elif isinstance(value, (list, tuple)):
            contents = [_state_fingerprint(item, active) for item in value]
        else:
            contents = {name: _state_fingerprint(item, active)
                        for name, item in sorted(vars(value).items()) if not _transient_field(value, name)}
        return _canonical_hash({"type": tag, "contents": contents})
    finally:
        active.remove(id(value))


def fitted_estimator_fingerprint(model):
    """Bind an estimator's complete typed fitted state with canonical containers."""
    return _state_fingerprint(model, set())


def classifier_semantic_fingerprint(classifier):
    """Bind global and OOF estimators, membership sets and training metadata."""
    models = {"global": classifier.model,
              **{f"fold:{fold}": model for fold, model in sorted(classifier.crossfit_models.items())}}
    for model in models.values():
        texts = getattr(model, "fitted_texts", None)
        if (not isinstance(texts, (set, frozenset))
                or any(not isinstance(text, str) for text in texts)):
            raise ValueError("Fitted classifier must preserve explicit string query-text membership")
    return _canonical_hash({"fingerprint_version": FITTED_FINGERPRINT_VERSION,
        "models": {name: fitted_estimator_fingerprint(model) for name, model in models.items()},
        "metadata_sha256": _canonical_hash(classifier.metadata)})
