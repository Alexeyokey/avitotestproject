"""Tiny post-completion route fixtures; no actual fresh labels or results read."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

import analyze_fresh_retrieval as ANALYSIS


FEATURES = (*ANALYSIS.ROUTE_COLUMNS, "rrf_score", "baseline_top50")


class FreshRouteAnalysisTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        groups = []
        for index, (text, available, full, size, hits, representative) in enumerate((
                ("same", 2, 3, 3, 1, True), ("same", 1, 1, 1, 1, False),
                ("none", 0, 2, 2, 0, True), ("empty", 1, 1, 0, 0, True))):
            groups.append({"group_id": index, "protocol_group_id": str(index) * 64,
                           "query_text": text, "query_key": [text, str(index), "0", "", "category"],
                           "pool_size": size, "positives_in_pool": hits,
                           "available_relevant_count": available, "full_relevant_count": full,
                           "representative": representative})
        path = root / "audit.parquet"
        columns = {"group_id": pa.array([0, 0, 0, 1, 2, 2], type=pa.int32()),
                   "label": pa.array([1, 0, 0, 1, 0, 0], type=pa.int8()),
                   "present_description": pa.array([1, 1, 0, 0, 1, 0], type=pa.float32()),
                   "present_dense": pa.array([1, 0, 1, 0, 0, 1], type=pa.float32()),
                   "geo_channel_count": pa.array([0, 1, 0, 1, 0, 0], type=pa.float32()),
                   "rrf_score": pa.array([0.3, 0.2, 0.1, 0.3, 0.2, 0.1], type=pa.float32()),
                   "baseline_top50": pa.array([1] * 6, type=pa.float32()),
                   "item_id": [f"{index:016x}" for index in range(6)],
                   "positive_local": [True, False, False, True, False, False],
                   "positive_seen": [False] * 6}
        pq.write_table(pa.table(columns), path)
        metadata = {"feature_schema_version": 4, "feature_names": list(FEATURES), "rows": 6,
                    "groups": groups, "model_reference": str(root / "model.joblib"),
                    "cache_version": 2, "role": "validation", "fresh_partition": "audit",
                    "fresh_validation_protocol_sha256": "1" * 64, "model_sha256": "2" * 64,
                    "full_truth_preserved": True, "historical_holdout_reused": False,
                    "cache_parquet_sha256": ANALYSIS.EVAL.sha256_file(path)}
        completed = [{"pool_size": group["pool_size"], "pool_hits": group["positives_in_pool"]} for group in groups]
        return path, metadata, groups, completed

    def stream(self, path, metadata, groups, completed, batch_size=2):
        return ANALYSIS.stream_route_stats(path, metadata, groups, completed,
                                           batch_size=batch_size, expected_features=FEATURES)

    def test_recall_uses_full_gt_not_positive_hits_found_in_pool(self):
        with TemporaryDirectory() as directory:
            rows = self.stream(*self.fixture(directory))
            report = ANALYSIS.summarize_routes(rows)
            self.assertEqual(rows[0]["full_pool_hits"], 1)
            self.assertEqual(rows[0]["available_relevant_count"], 2)
            full = report["candidate_recall"]["full_pool"]
            self.assertAlmostEqual(full["conditional_available"]["group_macro_candidate_recall"], 0.5)
            self.assertAlmostEqual(full["conditional_available"]["text_macro_candidate_recall"], 0.375)
            self.assertAlmostEqual(full["conditional_available"]["representative_macro_candidate_recall"], 0.25)
            self.assertAlmostEqual(full["full_truth"]["text_macro_candidate_recall"], 2 / 9)
            self.assertEqual(report["cohort"]["groups_without_available_gt"], 1)
            self.assertEqual(report["cohort"]["empty_pools"], 1)
            self.assertEqual(report["cohort"]["full_relevant_pairs"], 7)

    def test_union_overlap_and_geo_unique_additions_are_distinct(self):
        with TemporaryDirectory() as directory:
            rows = self.stream(*self.fixture(directory))
            self.assertEqual(rows[0]["description_count"], 2)
            self.assertEqual(rows[0]["dense_count"], 2)
            self.assertEqual(rows[0]["description_and_dense_count"], 1)
            self.assertEqual(rows[0]["description_or_dense_count"], 3)
            self.assertEqual(rows[0]["geo_count"], 1)
            self.assertEqual(rows[0]["geo_unique_additions_count"], 0)
            self.assertEqual(rows[1]["geo_unique_additions_hits"], 1)
            report = ANALYSIS.summarize_routes(rows)
            overlap = report["description_dense_overlap"]
            self.assertAlmostEqual(overlap["intersection_count"]["group_mean"], .25)
            self.assertAlmostEqual(overlap["intersection_count"]["text_macro_mean"], 1 / 6)
            self.assertAlmostEqual(overlap["jaccard"]["group_mean"], 1 / 6)
            self.assertEqual(overlap["jaccard"]["undefined_groups"], 2)
            self.assertAlmostEqual(overlap["description_overlap_fraction"]["group_mean"], .25)
            routes = report["candidate_recall"]
            self.assertAlmostEqual(routes["description_or_dense"]["conditional_available"]["text_macro_candidate_recall"], .125)
            self.assertAlmostEqual(routes["geo_unique_additions"]["conditional_available"]["text_macro_candidate_recall"], .25)

    def test_batch_boundaries_do_not_change_counts_and_empty_groups_remain(self):
        with TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            self.assertEqual(self.stream(*fixture, batch_size=1), self.stream(*fixture, batch_size=4))
            rows = self.stream(*fixture, batch_size=2)
            self.assertEqual([row["full_pool_count"] for row in rows], [3, 1, 2, 0])
            self.assertEqual([row["full_pool_hits"] for row in rows], [1, 1, 0, 0])

    def test_completed_size_hits_and_exact_v4_columns_are_required(self):
        with TemporaryDirectory() as directory:
            path, metadata, groups, completed = self.fixture(directory)
            changed = [dict(value) for value in completed]
            changed[0]["pool_hits"] = 0
            with self.assertRaisesRegex(ValueError, "differ from completed group scores"):
                self.stream(path, metadata, groups, changed)
            with self.assertRaisesRegex(ValueError, "exact V4"):
                self.stream(path, {**metadata, "feature_schema_version": 6}, groups, completed)
            table = pq.read_table(path)
            table = table.rename_columns(["incorrect_desc" if name == "present_description" else name
                                          for name in table.column_names])
            pq.write_table(table, path)
            with self.assertRaisesRegex(ValueError, "exact route columns"):
                self.stream(path, metadata, groups, completed)

    def test_invalid_route_flags_fail_without_inventing_membership(self):
        with TemporaryDirectory() as directory:
            path, metadata, groups, completed = self.fixture(directory)
            table = pq.read_table(path)
            index = table.schema.get_field_index("present_description")
            table = table.set_column(index, "present_description", pa.array([.5, 1, 0, 0, 1, 0], type=pa.float32()))
            pq.write_table(table, path)
            with self.assertRaisesRegex(ValueError, "route-presence flags"):
                self.stream(path, metadata, groups, completed)

    def closed_fixture(self, directory):
        root = Path(directory)
        path, metadata, groups, completed = self.fixture(directory)
        model = root / "model.joblib"
        model.write_bytes(b"opaque completed model")
        metadata["model_sha256"] = ANALYSIS.EVAL.sha256_file(model)
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(metadata), encoding="utf-8")
        scores = root / "audit-group-scores.jsonl"
        with scores.open("w", encoding="utf-8") as stream:
            for group in groups:
                stream.write(json.dumps({**group, "pool_hits": group["positives_in_pool"]}) + "\n")
        report = {"report_version": 1, "phase": "audit", "selected_stage": 50,
                  "policy": ANALYSIS.EVAL.POLICY, "stages": {"50": {}},
                  "group_scores_sha256": ANALYSIS.EVAL.sha256_file(scores),
                  "protocol_sha256": "1" * 64, "model_sha256": metadata["model_sha256"],
                  "train_data_sha256": "3" * 64,
                  "cache_sha256": {"parquet": ANALYSIS.EVAL.sha256_file(path),
                                   "metadata": ANALYSIS.EVAL.sha256_file(meta_path)}}
        (root / "audit-report.json").write_text(json.dumps(report), encoding="utf-8")
        (root / "selection.json").write_text("{}", encoding="utf-8")
        return root, metadata, groups, report

    def test_no_protocol_or_labels_are_read_before_phase_report_completion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(ANALYSIS.EVAL, "checked_protocol") as protocol:
                with self.assertRaisesRegex(ValueError, "completed phase report first"):
                    ANALYSIS.analyze_completed("audit", root, root, root, root / "diagnostic.json")
                protocol.assert_not_called()

    def test_audit_grid_or_changed_completed_scores_are_rejected(self):
        with TemporaryDirectory() as directory:
            root, metadata, groups, report = self.closed_fixture(directory)
            report["stages"]["150"] = {}
            (root / "audit-report.json").write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selected prefix only"):
                ANALYSIS.read_completed_phase(root, "audit")
            del report["stages"]["150"]
            (root / "audit-report.json").write_text(json.dumps(report), encoding="utf-8")
            (root / "audit-group-scores.jsonl").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                ANALYSIS.read_completed_phase(root, "audit")

    def test_completed_analysis_is_separate_and_changed_cache_is_rejected(self):
        with TemporaryDirectory() as directory:
            root, metadata, groups, report = self.closed_fixture(directory)
            output = root / "routes-diagnostic.json"
            protected = {name: (root / name).read_bytes() for name in
                         ("audit.parquet", "audit.json", "audit-report.json", "selection.json")}
            with mock.patch.object(ANALYSIS.EVAL, "checked_protocol", return_value=({}, "1" * 64)), \
                    mock.patch.object(ANALYSIS.EVAL, "verify_selection", return_value=50), \
                    mock.patch.object(ANALYSIS, "v4_feature_schema", return_value=FEATURES), \
                    mock.patch.object(ANALYSIS.EVAL, "read_phase_labels", return_value={}), \
                    mock.patch.object(ANALYSIS.EVAL, "bind_cache_groups", return_value=groups):
                result = ANALYSIS.analyze_completed("audit", root, root, root, output, batch_size=2)
                self.assertEqual(result["type"], "posthoc_descriptive_retrieval_routes")
                self.assertEqual(result["cohort"]["groups"], 4)
                self.assertTrue(output.is_file())
                for name, original in protected.items():
                    self.assertEqual((root / name).read_bytes(), original)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    ANALYSIS.analyze_completed("audit", root, root, root, output)
                path = root / "audit.parquet"
                path.write_bytes(path.read_bytes() + b"changed cache")
                with mock.patch.object(ANALYSIS.EVAL, "read_phase_labels") as labels:
                    with self.assertRaisesRegex(ValueError, "changed or are unsealed"):
                        ANALYSIS.analyze_completed("audit", root, root, root, root / "other.json")
                    labels.assert_not_called()


if __name__ == "__main__":
    unittest.main()
