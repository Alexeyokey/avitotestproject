"""Small fixtures for inference-policy hashes and completed-cache annotation."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reranker_audit_provenance import (
    artifact_prior_policy, finalize_cache_provenance, inference_prior_fingerprint,
)


def priors(reverse=False, float_counts=False):
    count = float if float_counts else int
    exact = {"ремонт": Counter({"one": count(2), "two": count(1)}),
             "монтаж": Counter({"two": count(3)})}
    tokens = {"ремонт": Counter({"two": 0.25, "one": 0.75}),
              "монтаж": Counter({"two": count(1)})}
    totals = {"ремонт": count(1), "монтаж": count(1)}
    locations = {"region": Counter({"city": count(3), "neighbor": count(2)}),
                 "second": Counter({"city": count(1)})}
    geo_totals = {"region": count(5), "second": count(1)}
    if reverse:
        exact, tokens, totals, locations, geo_totals = [dict(reversed(list(value.items())))
                                                       for value in (exact, tokens, totals,
                                                                     locations, geo_totals)]
        exact = {key: dict(reversed(list(value.items()))) for key, value in exact.items()}
        tokens = {key: dict(reversed(list(value.items()))) for key, value in tokens.items()}
        locations = {key: dict(reversed(list(value.items()))) for key, value in locations.items()}
    return {
        "microcategory_prior": SimpleNamespace(exact_counts=exact, token_counts=tokens,
                                               token_totals=totals),
        "geo_associations": SimpleNamespace(corpus_locations={"city", "neighbor"},
                                             counts=locations, totals=geo_totals),
    }


class PriorProvenanceTest(unittest.TestCase):
    def test_equivalent_mapping_orders_and_numeric_counts_hash_identically(self):
        self.assertEqual(inference_prior_fingerprint(priors()),
                         inference_prior_fingerprint(priors(reverse=True, float_counts=True)))
        self.assertEqual(inference_prior_fingerprint(None), inference_prior_fingerprint({}))

    def test_changed_history_and_disabled_feature_policy_are_detected(self):
        original = priors()
        changed = priors()
        changed["geo_associations"].counts["region"]["city"] += 1
        self.assertNotEqual(inference_prior_fingerprint(original),
                            inference_prior_fingerprint(changed))
        enabled = artifact_prior_policy({"priors": original})
        disabled = artifact_prior_policy({"priors": original, "disable_microcategory_prior": True})
        self.assertEqual(enabled["inference_priors_sha256"], disabled["inference_priors_sha256"])
        self.assertNotEqual(enabled, disabled)

    def test_training_only_geo_attribution_does_not_change_inference_hash(self):
        original = priors()
        tracked = priors()
        tracked["geo_associations"].query_counts = {"ремонт": {"region": Counter({"city": 3})}}
        self.assertEqual(inference_prior_fingerprint(original), inference_prior_fingerprint(tracked))

    def test_nonfinite_and_ambiguous_mapping_counts_are_rejected(self):
        changed = priors()
        changed["microcategory_prior"].token_totals["ремонт"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            inference_prior_fingerprint(changed)
        changed = priors()
        changed["geo_associations"].totals = {1: 2, "1": 2}
        with self.assertRaisesRegex(ValueError, "collide"):
            inference_prior_fingerprint(changed)

    def make_completed_cache(self, directory):
        cache = Path(directory) / "cache"
        cache.mkdir()
        reference = Path(directory) / "reference.joblib"
        artifact = {"model": None, "metadata": {}, "priors": priors()}
        reference.write_bytes(pickle.dumps(artifact))
        reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
        manifests = {}
        for role in ("validation", "train_sample"):
            # Only the footer completion guard is tested here. Actual Parquet
            # row/schema validation belongs to the audit reader's tests.
            (cache / f"{role}.parquet").write_bytes(b"PAR1" + b"x" + (1).to_bytes(4, "little") + b"PAR1")
            value = {"role": role, "groups": [{"group_id": 0}], "rows": 1,
                     "model_reference": str(reference), "model_sha256": reference_hash}
            manifest = cache / f"{role}.json"
            manifest.write_text(json.dumps(value), encoding="utf-8")
            manifests[role] = manifest
        return cache, reference, artifact, manifests

    def test_completed_metadata_updates_are_idempotent_and_leave_rows_unchanged(self):
        with TemporaryDirectory() as directory:
            cache, _reference, artifact, manifests = self.make_completed_cache(directory)
            old_rows = {(cache / f"{role}.parquet"): (cache / f"{role}.parquet").read_bytes()
                        for role in manifests}
            result = finalize_cache_provenance(cache)
            self.assertEqual(len(result["updated_metadata_files"]), 2)
            for role, manifest in manifests.items():
                metadata = json.loads(manifest.read_text())
                for key, value in artifact_prior_policy(artifact).items():
                    self.assertEqual(metadata[key], value)
                self.assertEqual(metadata["rows"], 1)
                self.assertEqual((cache / f"{role}.parquet").read_bytes(), old_rows[cache / f"{role}.parquet"])
            self.assertEqual(finalize_cache_provenance(cache)["updated_metadata_files"], [])

    def test_conflicting_existing_policy_is_rejected_before_either_manifest_changes(self):
        with TemporaryDirectory() as directory:
            cache, _reference, _artifact, manifests = self.make_completed_cache(directory)
            value = json.loads(manifests["train_sample"].read_text())
            value["inference_priors_sha256"] = "different-history"
            manifests["train_sample"].write_text(json.dumps(value), encoding="utf-8")
            before = {role: path.read_bytes() for role, path in manifests.items()}
            with self.assertRaisesRegex(ValueError, "inference_priors_sha256"):
                finalize_cache_provenance(cache)
            self.assertEqual(before, {role: path.read_bytes() for role, path in manifests.items()})

    def test_incomplete_cache_and_changed_source_model_are_rejected(self):
        with TemporaryDirectory() as directory:
            cache, reference, _artifact, manifests = self.make_completed_cache(directory)
            reference.write_bytes(reference.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                finalize_cache_provenance(cache)
            (cache / "train_sample.parquet").unlink()
            with self.assertRaisesRegex(ValueError, "not complete"):
                finalize_cache_provenance(cache)
            self.assertNotIn("inference_priors_sha256", json.loads(manifests["validation"].read_text()))

    def test_validation_only_is_explicit_and_preserves_completion_and_source_checks(self):
        with TemporaryDirectory() as directory:
            cache, reference, artifact, manifests = self.make_completed_cache(directory)
            (cache / "train_sample.parquet").unlink()
            manifests["train_sample"].unlink()
            before = (cache / "validation.parquet").read_bytes()
            with self.assertRaisesRegex(ValueError, "not complete"):
                finalize_cache_provenance(cache)
            # Trusted tiny fixture uses the stdlib pickle loader, not a real model.
            loader = SimpleNamespace(load=lambda path: pickle.loads(Path(path).read_bytes()))
            with patch.dict(sys.modules, {"joblib": loader}):
                result = finalize_cache_provenance(cache, validation_only=True)
            self.assertTrue(result["validation_only"])
            self.assertEqual(result["roles"], ["validation"])
            self.assertEqual(len(result["updated_metadata_files"]), 1)
            saved = json.loads(manifests["validation"].read_text())
            self.assertEqual(saved["inference_priors_sha256"], artifact_prior_policy(artifact)["inference_priors_sha256"])
            self.assertEqual((cache / "validation.parquet").read_bytes(), before)
            reference.write_bytes(reference.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                finalize_cache_provenance(cache, validation_only=True)


if __name__ == "__main__":
    unittest.main()
