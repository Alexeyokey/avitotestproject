"""Tests of cache streaming, fixed-pool integrity and exact source-stage parity."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from evaluate_geo_dense_regularization import (STAGES, VARIANTS, rank_hits, score_one_group,
    stream_scores, validate_variant, check_source_identity, save_secondary_model)
import evaluate_fresh_validation as EVAL


class FixedEstimator:
    n_iter_ = 150

    def __init__(self, **parameters):
        self.parameters = parameters

    def get_params(self):
        return self.parameters

    def staged_predict_proba(self, matrix):
        scores = np.asarray(matrix[:, 2], dtype=float)
        for _ in range(150):
            yield np.column_stack((1 - scores, scores))


class CachedRegularizationTest(unittest.TestCase):
    def test_post_scoring_source_check_accepts_json_string_paths_and_rejects_change(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "producer.py"
            path.write_text("original code", encoding="utf-8")
            snapshot = {str(path): EVAL.sha256_file(path)}
            check_source_identity(snapshot)
            path.write_text("changed code", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "implementation changed"):
                check_source_identity(snapshot)
            path.unlink()
            with self.assertRaisesRegex(ValueError, "implementation changed"):
                check_source_identity(snapshot)

    def test_secondary_artifact_finalization_roundtrip_preserves_full_model_and_prefix(self):
        import joblib
        from sklearn.ensemble import HistGradientBoostingClassifier
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data = np.random.RandomState(15).randn(120, 3).astype(np.float32)
            model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=3,
                early_stopping=False, random_state=42).fit(data, data[:, 0] > 0)
            artifact = {"model": model, "metadata": {"retrieval_config": {"geo_dense_weight": .05}},
                "training_configuration": {"parameters": model.get_params()}, "provenance": {"fit_only": True}}
            source = root / "source.joblib"
            joblib.dump(artifact, source)
            artifact = joblib.load(source)
            report = root / "report.json"
            report.write_text("{}", encoding="utf-8")
            result = save_secondary_model(artifact, source, "maxfeatures08", 50, report, root)
            restored = joblib.load(result["path"])
            self.assertEqual(restored["selected_iteration"], 50)
            self.assertEqual(restored["model"].n_iter_, 150)
            self.assertEqual(restored["metadata"], artifact["metadata"])
            self.assertEqual(restored["regularization_selection"]["report"]["sha256"], EVAL.sha256_file(report))
            self.assertTrue(Path(result["path"]).with_suffix(".joblib.json").is_file())
            with self.assertRaises(FileExistsError):
                save_secondary_model(artifact, source, "maxfeatures08", 50, report, root)

    def fixture(self, key="q", size=52):
        items = [f"{key}-{index}" for index in range(size)]
        labels = np.zeros(size, dtype=bool)
        labels[-1] = True
        rrf = np.zeros(size, dtype=np.float32)
        rrf[-1] = 2
        baseline = np.zeros(size, dtype=np.float32)
        baseline[:50] = 1
        scores = np.zeros(size)
        reference = {"group_identity": key, "pool_size": size, "available_relevant_count": 1,
            "full_relevant_count": 2, "pool_hits": 1, "baseline_hits": 0,
            "ordered_pool_sha256": EVAL.canonical_sha256(items), **{f"s{s}_hits": 1 for s in STAGES}}
        predictions = {name: {stage: scores.copy() for stage in STAGES} for name in ("source", *VARIANTS)}
        return reference, items, labels, rrf, baseline, predictions, {items[-1]}

    def test_fixed50_uses_exact_probability_then_rrf_then_pool_position(self):
        row, items, labels, rrf, baseline, predictions, relevant = self.fixture()
        # Under score ties last item enters through RRF; changing its model score
        # below the others excludes it. Item strings do not participate in ties.
        self.assertEqual(rank_hits(np.zeros(len(items)), rrf, labels), 1)
        for stage in STAGES:
            predictions["maxfeatures08"][stage][-1] = -1
        result = score_one_group(row, items, labels, rrf, baseline, predictions, relevant)
        self.assertEqual(result["maxfeatures08"]["s50_hits"], 0)
        self.assertEqual(result["leaves15"]["s50_hits"], 1)
        reversed_names = list(reversed(items))
        tied_labels = np.zeros(52, dtype=bool)
        tied_labels[49] = True
        self.assertEqual(rank_hits(np.zeros(52), np.zeros(52), tied_labels), 1)

    def test_source_must_reproduce_every_stage(self):
        values = list(self.fixture())
        values[5]["source"][150][-1] = -1
        with self.assertRaisesRegex(ValueError, "stage150"):
            score_one_group(*values)

    def test_label_membership_detects_same_hit_count_with_wrong_positive(self):
        values = list(self.fixture())
        values[2] = np.roll(values[2], -1)
        with self.assertRaisesRegex(ValueError, "membership"):
            score_one_group(*values)

    def test_candidate_order_hash_prevents_same_size_pool_substitution(self):
        values = list(self.fixture())
        values[1] = list(reversed(values[1]))
        with self.assertRaisesRegex(ValueError, "identity, order"):
            score_one_group(*values)

    def test_all_models_and_all_stages_required(self):
        values = list(self.fixture())
        del values[5]["leaves15"][75]
        with self.assertRaisesRegex(ValueError, "predeclared HGB stage"):
            score_one_group(*values)

    def test_variants_change_exactly_one_parameter(self):
        metadata = {"features": ["a"]}
        provenance = {"training_data_sha256": "fit", "training_metadata_sha256": "meta",
                      "training_priors_sha256": "priors", "training_prior_policy": "oof"}
        parameters = {"max_iter": 150, "max_features": 1., "max_leaf_nodes": 31, "random_state": 42}
        source = {"model": FixedEstimator(**parameters), "metadata": metadata, "provenance": provenance}
        with patch("evaluate_geo_dense_regularization.research_prior_policy", return_value={"same": True}):
            for name, change in VARIANTS.items():
                actual = {**parameters, **change}
                variant = {"estimator": "histgb", "selected_iteration": None,
                    "model": FixedEstimator(**actual), "metadata": metadata, "provenance": provenance,
                    "training_configuration": {"parameters": actual}}
                validate_variant(variant, source, name)
                combined = {**parameters, "max_features": .8, "max_leaf_nodes": 15}
                variant["model"] = FixedEstimator(**combined)
                variant["training_configuration"]["parameters"] = combined
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    validate_variant(variant, source, name)

    def test_stream_carries_partial_queries_across_batches_and_preserves_zero_groups(self):
        with tempfile.TemporaryDirectory() as root:
            rows, records, targets = [], [], {}
            for key, size in (("first", 52), ("second", 55)):
                row, items, labels, rrf, baseline, _, relevant = self.fixture(key, size)
                rows.append(row)
                targets[key] = {"available_item_ids": list(relevant)}
                for item, label, r, b in zip(items, labels, rrf, baseline):
                    records.append({"rrf_score": r, "baseline_top50": b, "signal": np.float32(0),
                        "group_identity": key, "item_id": item, "available_positive": bool(label)})
            zero = {"group_identity": "zero", "pool_size": 998, "available_relevant_count": 0,
                    "full_relevant_count": 3, "pool_hits": 0, "baseline_hits": 0,
                    **{f"s{s}_hits": 0 for s in STAGES}}
            rows.insert(1, zero)
            targets["zero"] = {"available_item_ids": []}
            schema = pa.schema([(name, pa.float32()) for name in ("rrf_score", "baseline_top50", "signal")] +
                [("group_identity", pa.string()), ("item_id", pa.string()), ("available_positive", pa.bool_())])
            path = Path(root) / "features.parquet"
            pq.write_table(pa.Table.from_pylist(records, schema=schema), path, row_group_size=19)
            models = {name: FixedEstimator() for name in ("source", *VARIANTS)}
            result, parity = stream_scores(path, ["rrf_score", "baseline_top50", "signal"], rows, models,
                                           targets, batch_size=17)
            self.assertEqual((parity["rows"], parity["available_groups"], parity["all_groups"]), (107, 2, 3))
            for name in VARIANTS:
                self.assertEqual([r["group_identity"] for r in result[name]], ["first", "zero", "second"])
                self.assertEqual(result[name][1]["full_relevant_count"], 3)
                self.assertEqual([r["s50_hits"] for r in result[name]], [1, 0, 1])
            bad_rows = [rows[2], rows[1], rows[0]]
            with self.assertRaisesRegex(ValueError, "missing, repeated, unknown or reordered"):
                stream_scores(path, ["rrf_score", "baseline_top50", "signal"], bad_rows, models,
                              targets, batch_size=17)


if __name__ == "__main__":
    unittest.main()
