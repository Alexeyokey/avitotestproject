"""Static-source policy evidence tests with tiny stubs, no real-data reads."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import joblib

from annotate_legacy_audit_policy import annotate_legacy_policy, verify_frozen_producer
from select_reranker_checkpoint import LEGACY_V4_TRAINING_PRIOR_POLICY, select_checkpoint
from reranker_audit_provenance import artifact_prior_policy
import test_select_reranker_checkpoint as checkpoint_fixtures

sha = checkpoint_fixtures.sha


CLI = '''
def main():
    if args.command == "prepare-reranker":
        builder = FeatureBuilder(items, config, leave_query_out=True)
        metadata = prepare_training_data(args.reranker_data, model, builder)
'''
RERANKER = '''
class FeatureBuilder:
    def __init__(self, items, config, leave_query_out=False):
        self.leave_query_out = leave_query_out
    def transform(self, query, details, baseline):
        associations = self.geo_associations
        click_total = associations.totals[location] if associations else 0
        click_counts = associations.counts[location] if associations and click_total >= 20 else {}
        mc = self.microcategory_prior.features(query_text, microcat, exclude_self=self.leave_query_out)
'''
GEOGRAPHY = '''
class LocationAssociations:
    def destinations(self, search_location, min_history=20):
        location = search_location
        if self.totals[location] < min_history:
            return ()
        return self.counts[location]
'''
MICROCATEGORY = '''
class MicrocategoryPrior:
    def scores(self, text, exclude_self=False):
        total = self.token_totals[token] - bool(exclude_self and own)
        adjusted = count - own.get(category, 0.0) if exclude_self else count
        return total, adjusted
'''


class LegacyPolicyAnnotationTest(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        model, source, _data, report, original = checkpoint_fixtures.CheckpointSelectionTest().make_fixture(root, legacy=True)
        package = root / "frozen" / "avito_candidates"
        package.mkdir(parents=True)
        for name, code in (("cli", CLI), ("reranker", RERANKER),
                           ("geography", GEOGRAPHY), ("microcategory", MICROCATEGORY)):
            (package / f"{name}.py").write_text(code, encoding="utf-8")
        fit = {**source["metadata"], "role": "train_sample", "cache_version": 2,
               "feature_schema_version": 4, "model_sha256": sha(model),
               "leave_query_out": True, **artifact_prior_policy(source)}
        fit_path = root / "train_sample.json"
        fit_path.write_text(json.dumps(fit), encoding="utf-8")
        output = root / "hgb-report-with-policy.json"
        return model, source, report, original, package, fit_path, fit, output

    def test_annotation_preserves_original_metrics_and_binds_all_evidence(self):
        with TemporaryDirectory() as directory:
            model, _source, report, original, package, fit_path, _fit, output = self.fixture(directory)
            before = report.read_bytes()
            result = annotate_legacy_policy(report, package.parent, fit_path, model, output,
                                            expected_model_sha256=sha(model))
            annotated = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual({key: value for key, value in annotated.items()
                              if key not in {"training_prior_policy", "training_prior_policy_evidence"}}, original)
            self.assertEqual(annotated["training_prior_policy"], LEGACY_V4_TRAINING_PRIOR_POLICY)
            evidence = annotated["training_prior_policy_evidence"]
            self.assertFalse(evidence["historical_training_policy_verified"])
            self.assertEqual(evidence["audited_model_sha256"], sha(model))
            self.assertEqual(evidence["original_audit_report"]["sha256"], sha(report))
            self.assertEqual(evidence["fit_cache_metadata"]["sha256"], sha(fit_path))
            for name, metadata in evidence["frozen_producer_modules"].items():
                self.assertEqual(metadata["sha256"], sha(package / f"{name}.py"))
            self.assertEqual(result["sha256"], sha(output))
            self.assertEqual(report.read_bytes(), before)
            with self.assertRaises(FileExistsError):
                annotate_legacy_policy(report, package, fit_path, model, output,
                                       expected_model_sha256=sha(model))

    def test_hash_and_cache_flags_cannot_be_bypassed(self):
        with TemporaryDirectory() as directory:
            model, _source, report, _original, package, fit_path, fit, output = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                annotate_legacy_policy(report, package, fit_path, model, output,
                                       expected_model_sha256="incorrect")
            for changes in ({"leave_query_out": False}, {"leave_geography_query_out": True},
                            {"feature_schema_version": 6}, {"model_sha256": "different"}):
                changed = {**fit, **changes}
                fit_path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    annotate_legacy_policy(report, package, fit_path, model, output,
                                           expected_model_sha256=sha(model))
                self.assertFalse(output.exists())

    def test_static_false_builder_and_geography_exclusion_are_rejected(self):
        with TemporaryDirectory() as directory:
            _model, _source, _report, _original, package, _fit_path, _fit, _output = self.fixture(directory)
            (package / "cli.py").write_text(CLI.replace("leave_query_out=True", "leave_query_out=False"))
            with self.assertRaisesRegex(ValueError, "leave_query_out=True"):
                verify_frozen_producer(package)
            (package / "cli.py").write_text(CLI)
            (package / "geography.py").write_text(GEOGRAPHY + "\nquery_counts = {}\n")
            with self.assertRaisesRegex(ValueError, "exclusion mechanism"):
                verify_frozen_producer(package)
            (package / "geography.py").write_text(GEOGRAPHY)
            (package / "reranker.py").write_text(RERANKER.replace("associations.counts[location]", "adjusted_counts"))
            with self.assertRaisesRegex(ValueError, "click_counts calculation"):
                verify_frozen_producer(package)

    def test_recovered_report_evidence_is_carried_into_capacity_transfer(self):
        with TemporaryDirectory() as directory:
            model, source, report, _original, package, fit_path, _fit, annotated = self.fixture(directory)
            annotate_legacy_policy(report, package, fit_path, model, annotated,
                                   expected_model_sha256=sha(model))
            modern_source = deepcopy(source)
            modern_source["metadata"]["training_prior_policy"] = dict(LEGACY_V4_TRAINING_PRIOR_POLICY)
            final, _target = checkpoint_fixtures.CheckpointSelectionTest().make_transfer_target(directory, modern_source)
            output = Path(directory) / "selected-final.joblib"
            select_checkpoint(annotated, output, transfer_to_model=final, allow_legacy_provenance=True)
            selected = joblib.load(output)
            evidence = selected["selection_provenance"]["capacity_selection"]["compatibility"]["source_training_prior_policy_evidence"]
            self.assertEqual(evidence["frozenproducer_evidence"]["original_audit_report"]["sha256"], sha(report))
            self.assertFalse(evidence["historical_training_policy_verified"])


if __name__ == "__main__":
    unittest.main()
