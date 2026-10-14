"""Single-factor and fit-only guards; these tests do not fit a model."""
from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest

from train_geo_dense_regularization import (matched_parameters, new_envelope,
                                          validate_fit_metadata)
from avito_candidates.reranker import V8_FEATURES
from cache_research_reranker import RESEARCH_POLICY
from prepare_geo_dense_reranker import GEO_CONFIG


class RegularizationTest(unittest.TestCase):
    def parameters(self):
        return {"max_iter": 150, "early_stopping": False, "random_state": 42,
                "max_features": 1., "max_leaf_nodes": 31, "warm_start": False,
                "learning_rate": .1, "l2_regularization": 1., "min_samples_leaf": 20}

    def metadata(self):
        return {"feature_version": 8, "feature_names": list(V8_FEATURES),
                "training_data_schema_version": 2, "split": "queries", "seed": 42,
                "validation_fraction": .2, "training_prior_policy": dict(RESEARCH_POLICY),
                "retrieval_config": dict(GEO_CONFIG), "stats": {"sampled_queries": 30000},
                "geo_dense_training": {"complete": True, "phase": "fit",
                    "tune_labels_read": False, "audit_labels_read": False,
                    "inventory_reencoded": False, "selected_groups": 30000,
                    "maximum_pool_size": 998}}

    def test_each_variant_changes_only_one_parameter_and_never_source(self):
        original = self.parameters()
        for variant, key, value in (("maxfeatures08", "max_features", .8),
                                    ("leaves15", "max_leaf_nodes", 15)):
            result = matched_parameters(original, variant)
            self.assertEqual({k for k in original if original[k] != result[k]}, {key})
            self.assertEqual(result[key], value)
        self.assertEqual(original, self.parameters())

    def test_source_drift_is_rejected_instead_of_silently_overridden(self):
        for key, value in (("max_iter", 50), ("early_stopping", True), ("random_state", 43),
                           ("max_features", .8), ("max_leaf_nodes", 15), ("warm_start", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                matched_parameters({**self.parameters(), key: value}, "maxfeatures08")
        with self.assertRaises(ValueError):
            matched_parameters(self.parameters(), "combined")

    def test_only_completed_fit_cache_is_accepted(self):
        metadata = self.metadata()
        self.assertEqual(validate_fit_metadata(metadata), V8_FEATURES)
        for key in ("role", "phase", "fresh_partition"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Tune/Audit"):
                validate_fit_metadata({**metadata, key: "tune"})
        for key, value in (("phase", "tune"), ("complete", False),
                           ("audit_labels_read", True), ("tune_labels_read", True)):
            altered = copy.deepcopy(metadata)
            altered["geo_dense_training"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_fit_metadata(altered)

    def test_envelope_resets_selection_without_copying_or_mutating_frozen_priors(self):
        source = {"model": "old", "metadata": self.metadata(), "priors": {"immutable": object()},
                  "selection_status": "selected", "selected_iteration": 50,
                  "selection": {"old": 50}, "provenance": {"training_data_sha256": "hash"},
                  "training_configuration": {"parameters": self.parameters(),
                                              "selected_iteration": 50, "selection_status": "selected"}}
        newparams = matched_parameters(self.parameters(), "leaves15")
        model = SimpleNamespace(get_params=lambda deep=False: newparams, n_iter_=150)
        result = new_envelope(source, model, {"variant": "leaves15"})
        self.assertIs(result["priors"], source["priors"])
        self.assertIs(result["metadata"], source["metadata"])
        self.assertEqual(result["training_configuration"]["parameters"], newparams)
        self.assertEqual(result["training_configuration"]["regularization_variant"], "leaves15")
        self.assertEqual(result["selection_status"], "unselected")
        self.assertIsNone(result["selected_iteration"])
        self.assertNotIn("selection", result)
        self.assertEqual(source["selected_iteration"], 50)
        self.assertEqual(source["training_configuration"]["parameters"]["max_leaf_nodes"], 31)
        self.assertNotIn("regularization_experiment", source["provenance"])


if __name__ == "__main__":
    unittest.main()
