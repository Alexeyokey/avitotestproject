"""Tiny Tune-only retrieval fixtures; no project datasets/models are opened."""

import argparse
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

import evaluate_retrieval_hypotheses as E


def item(value):
    return f"{value:016x}"


def group(index, text, available, full=None, *, size=0, hits=0, baseline=0):
    key = [text, str(index), "0", "", "114"]
    full = available if full is None else full
    return {"group_id": index, "query_text": text, "query_key": key,
            "query_key_json": json.dumps(key, ensure_ascii=False, separators=(",", ":")),
            "group_identity": E.identity_for_key(key), "pool_size": size,
            "positives_in_pool": hits, "baseline_hits": baseline,
            "available_relevant_count": len(available), "full_relevant_count": len(full),
            "relevant_count": len(available), "representative": True,
            "_available_ids": set(available), "available_item_ids": available,
            "relevant_item_ids": full}


def raw_pool(items, labels, *, description=None, dense=None, geo=None):
    result = {"item_id": list(items), "label": list(labels),
              "baseline_top50": [1.] * min(50, len(items)) + [0.] * max(0, len(items) - 50)}
    for name in E.BASE_CHANNELS:
        ranked = {"description": description or [], "dense": dense or [],
                  "description_geo": geo or []}.get(name, [])
        ranks = {value: rank for rank, value in enumerate(ranked, 1)}
        result[f"present_{name}"] = [float(value in ranks) for value in items]
        result[f"rank_{name}"] = [1 / (60 + ranks[value]) if value in ranks else 0. for value in items]
    return result


class RetrievalHypothesisTests(unittest.TestCase):
    def test_text_sampling_ignores_targets_context_count_and_order(self):
        groups = [group(0, "same", []), group(1, "same", [item(1)]),
                  group(2, "other", [item(2)]), group(3, "third", [])]
        expected = E.choose_texts(groups, 2)
        changed = [{**value, "available_item_ids": [item(99)] * 10,
                    "full_relevant_count": 999} for value in reversed(groups)]
        self.assertEqual(E.choose_texts(changed, 2), expected)
        self.assertEqual(E.choose_texts(groups, 0), {"same", "other", "third"})
        self.assertEqual(E.choose_texts(groups, 20), {"same", "other", "third"})

    def test_family_matches_novel_additions_and_preserves_base(self):
        variants = E.validate_variants({"experiment_version": 1, "phase": "tune", "variants": [
            {"name": "micro", "mode": "fill", "channels": {"micro_title": 1},
             "addition_quota": 3, "matched_addition_family": "matched"},
            {"name": "title", "mode": "fill", "channels": {"title_control": 1},
             "addition_quota": 3, "matched_addition_family": "matched"}]})
        base = {"items": [item(1), item(2)]}
        pools = E.form_variant_pools(base, {"micro_title": [item(1), item(3), item(4), item(5)],
                                          "title_control": [item(2), item(6)]}, variants)
        self.assertEqual(pools["micro"], [item(1), item(2), item(3)])
        self.assertEqual(pools["title"], [item(1), item(2), item(6)])

    def test_capacity_is_per_group_and_does_not_truncate_existing_candidates(self):
        base = {"items": [item(value) for value in range(995)]}
        variant = {"name": "micro", "mode": "fill", "channels": {"micro": 1}, "addition_quota": 177}
        result = E.form_variant_pools(base, {"micro": [item(value) for value in range(990, 1010)]}, [variant])
        self.assertEqual(len(result["micro"]), 998)
        self.assertEqual(result["micro"][:995], base["items"])

    def test_replacement_swaps_dense_and_keeps_explicit_skeleton(self):
        variant = {"name": "query_only", "mode": "replace", "channels": {
            "description": {"source": "base", "quota": 2},
            "query_only_dense": {"source": "new", "quota": 2},
            "description_geo": {"source": "base", "quota": 1}}}
        base = {"items": [item(1), item(2), item(3), item(4)], "channels": {
            "description": [item(1), item(2)], "dense": [item(3), item(4)],
            "description_geo": [item(2)]}}
        result = E.form_variant_pools(base, {"query_only_dense": [item(5), item(1)]}, [variant])
        self.assertEqual(result["query_only"], [item(1), item(2), item(5)])
        self.assertNotIn(item(3), result["query_only"])

    def test_invalid_modes_families_and_budget_fail(self):
        cases = [
            {"name": "x", "mode": "replace", "channels": {"dense": {"source": "base", "quota": 999}}},
            {"name": "x", "mode": "fill", "channels": {"route": -1}},
            {"name": "x", "mode": "fill", "channels": {"route": 1}, "matched_addition_family": "lonely"},
            {"name": "x", "mode": "replace", "channels": {"arbitrary": {"source": "base", "quota": 1}}}]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                E.validate_variants({"experiment_version": 1, "phase": "tune", "variants": [value]})
        with self.assertRaises(ValueError):
            E.validate_variants({"experiment_version": 1, "phase": "audit", "variants": cases[:1]})

    def test_recovered_float32_ranks_are_ordered_and_gaps_rejected(self):
        inverses = pa.array([1 / 62, 1 / 61, 0.], type=pa.float32()).to_pylist()
        self.assertEqual(E.recover_ranked_channel([item(2), item(1), item(3)], inverses, [1, 1, 0], 60),
                         [item(1), item(2)])
        with self.assertRaises(ValueError):
            E.recover_ranked_channel([item(1)], [1 / 62], [1], 60)

    def test_base_gt_uses_full_available_set_not_found_hits(self):
        current = group(0, "query", [item(1), item(5)], size=2, hits=1, baseline=1)
        raw = raw_pool([item(1), item(2)], [1, 0], description=[item(1), item(2)])
        base = E.validate_base_group(raw, current)
        row = E.score_group(current, base, {"micro": [item(1), item(2), item(5)]})
        self.assertEqual(row["available_relevant_count"], 2)
        self.assertEqual(row["base_pool_hits"], 1)
        self.assertEqual(row["micro_pool_hits"], 2)
        self.assertEqual(row["micro_positive_additions"], 1)
        with self.assertRaises(ValueError):
            E.validate_base_group({**raw, "label": [0, 0]}, current)

    def test_text_macro_not_group_macro_and_unavailable_targets_retained(self):
        rows = [
            {"query_text": "same", "representative": True, "available_relevant_count": 2,
             "full_relevant_count": 3, "base_pool_hits": 1, "new_pool_hits": 2},
            {"query_text": "same", "representative": False, "available_relevant_count": 1,
             "full_relevant_count": 1, "base_pool_hits": 1, "new_pool_hits": 1},
            {"query_text": "other", "representative": True, "available_relevant_count": 1,
             "full_relevant_count": 2, "base_pool_hits": 0, "new_pool_hits": 0},
            {"query_text": "unavailable", "representative": True, "available_relevant_count": 0,
             "full_relevant_count": 1, "base_pool_hits": 0, "new_pool_hits": 0}]
        result = E.recall_summary(rows, "base_pool_hits", "available_relevant_count", candidate=True)
        self.assertEqual(result["text_macro_candidate_recall"], .375)
        self.assertEqual(result["group_macro_candidate_recall"], .5)
        full = E.recall_summary(rows, "base_pool_hits", "full_relevant_count", candidate=True)
        self.assertAlmostEqual(full["text_macro_candidate_recall"], (2 / 3) / 3)
        self.assertEqual(full["groups"], 4)
        bootstrap = E.paired_pool_bootstrap(rows, "new", replicates=30)
        self.assertEqual(bootstrap["query_text_clusters"], 2)
        self.assertEqual(bootstrap["delta"], .125)

    def test_matched_pair_reports_direct_gain_not_only_added_capacity(self):
        rows = [{"query_text": "a", "available_relevant_count": 2, "base_pool_hits": 0,
                 "micro_pool_hits": 2, "control_pool_hits": 1, "micro_pool_size": 4, "control_pool_size": 4}]
        result = E.matched_pair_bootstrap(rows, "micro", "control", replicates=10)
        self.assertEqual(result["delta"], .5)
        rows[0]["control_pool_size"] = 3
        with self.assertRaises(ValueError):
            E.matched_pair_bootstrap(rows, "micro", "control", replicates=10)

    def test_dense_variant_reference_uses_same_new_encoding_control(self):
        rows = [{"query_text": "a", "available_relevant_count": 2, "base_pool_hits": 0,
                 "plain_pool_hits": 2, "current_pool_hits": 1,
                 "plain_pool_size": 5, "current_pool_size": 4}]
        result = E.paired_variant_bootstrap(rows, "plain", "current", replicates=10)
        self.assertEqual(result["delta"], .5)
        self.assertIn("declared reference variant current", result["comparison"])
        with self.assertRaises(ValueError):
            E.validate_variants({"experiment_version": 1, "phase": "tune", "variants": [
                {"name": "only", "mode": "fill", "channels": {"route": 1},
                 "comparison_reference": "missing"}]})

    def test_stream_preserves_batch_split_pools_and_empty_groups(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tune.parquet"
            groups = [group(0, "a", [item(1)], size=3, hits=1, baseline=1),
                      group(1, "empty", [item(4)]), group(2, "b", [], size=1)]
            raw = raw_pool([item(1), item(2), item(3)], [1, 0, 0], description=[item(1), item(2), item(3)])
            other = raw_pool([item(9)], [0], dense=[item(9)])
            columns = {name: values + other[name] for name, values in raw.items()}
            columns["group_id"] = [0, 0, 0, 2]
            pq.write_table(pa.table(columns), path)
            result = list(E.iter_base_groups(path, {"rows": 4}, groups, {0, 1, 2}, batch_size=2))
            self.assertEqual([current[0]["group_id"] for current in result], [0, 1, 2])
            self.assertEqual(result[0][1]["item_id"], [item(1), item(2), item(3)])
            self.assertEqual(result[1][1]["item_id"], [])
            filtered = list(E.iter_base_groups(path, {"rows": 4}, groups, {2}, batch_size=1))
            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0][0]["group_id"], 2)

    def test_stream_missing_nonempty_group_fails(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tune.parquet"
            raw = raw_pool([item(1)], [0])
            pq.write_table(pa.table({**raw, "group_id": [1]}), path)
            groups = [group(0, "missing", [], size=1), group(1, "next", [], size=1)]
            with self.assertRaises(ValueError):
                list(E.iter_base_groups(path, {"rows": 1}, groups, {1}, batch_size=1))

    def test_supplement_rejects_non_catalog_ids_duplicates_or_wrong_query(self):
        current = group(0, "a", [])
        row = {"group_id": current["group_identity"], "query_key": current["query_key"],
               "channels": {"new": [item(1)]}}
        self.assertEqual(E.validate_supplement_row(row, current, {item(1)})["new"], [item(1)])
        for changed in ({**row, "group_id": "f" * 64},
                        {**row, "channels": {"new": [item(1), item(1)]}},
                        {**row, "channels": {"new": [item(2)]}}):
            with self.assertRaises(ValueError):
                E.validate_supplement_row(changed, current, {item(1)})

    def test_full_scope_gzip_can_stream_only_selected_group(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "supplement.jsonl.gz"
            groups = [group(0, "a", []), group(1, "b", []), group(2, "c", [])]
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                for current in groups:
                    stream.write(json.dumps({"group_id": current["group_identity"],
                        "query_key": current["query_key"], "channels": {"new": [item(1)]}}) + "\n")
            result = list(E.iter_supplements(path, groups, {1}, {item(1)}))
            self.assertEqual(result, [(1, {"new": [item(1)]})])

    def fixture_evaluation(self, root):
        protocol, cache = root / "protocol", root / "cache"
        protocol.mkdir(); cache.mkdir()
        current = group(0, "query", [item(1), item(3)], [item(1), item(3), item(99)],
                        size=2, hits=1, baseline=1)
        raw = raw_pool([item(1), item(2)], [1, 0], description=[item(1), item(2)])
        features = [name for name in raw if name not in {"item_id", "label"}] + ["rrf_score"]
        columns = {name: (pa.array(values, type=pa.float32()) if name in features else values)
                   for name, values in raw.items()}
        columns.update(group_id=pa.array([0, 0], type=pa.int32()),
                       rrf_score=pa.array([.1, .05], type=pa.float32()),
                       positive_local=[False, False], positive_seen=[False, False])
        cache_path = cache / "tune.parquet"
        pq.write_table(pa.table(columns), cache_path)
        manifest = {"protocol_version": 1, "complete": True, "sealed_audit": True,
                    "split": {"representative_salt": "test-salt"}, "corpus": {"size": 3}, "files": {}}
        for semantic in ("labels_fit", "labels_tune", "labels_audit", "fit_texts", "tune_texts",
                         "audit_texts", "historical_excluded_texts", "train", "benchmark_items",
                         "benchmark_queries", "catalog_item_ids"):
            path = protocol / f"{semantic.replace('_', '-')}.json"
            value = "AUDIT IS NOT JSON AND MUST NEVER BE PARSED" if "audit" in semantic else "{}"
            if semantic == "labels_tune":
                label = {key: value for key, value in current.items() if not key.startswith("_")}
                label["group_id"] = current["group_identity"]
                value = json.dumps({"partition": "tune", "groups": [label]})
            elif semantic == "tune_texts":
                value = json.dumps({"texts": ["query"]})
            elif semantic == "catalog_item_ids":
                value = json.dumps({"item_ids": [item(1), item(2), item(3)]})
            path.write_text(value, encoding="utf-8")
            manifest["files"][semantic] = {"path": path.name, "sha256": E.EVAL.sha256_file(path)}
        manifest_path = protocol / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        metadata = {"cache_version": 2, "role": "validation", "fresh_partition": "tune",
                    "fresh_validation_protocol_sha256": E.EVAL.sha256_file(manifest_path),
                    "model_sha256": "a" * 64, "historical_holdout_reused": False,
                    "full_truth_preserved": True, "cache_parquet_sha256": E.EVAL.sha256_file(cache_path),
                    "feature_names": features, "groups": [{key: value for key, value in current.items()
                        if not key.startswith("_")}], "rows": 2, "corpus_fingerprint": "b" * 64,
                    "retrieval_config": {"rrf_k": 60}}
        (cache / "tune.json").write_text(json.dumps(metadata), encoding="utf-8")
        candidates = root / "candidates.jsonl.gz"
        with gzip.open(candidates, "wt", encoding="utf-8") as stream:
            stream.write(json.dumps({"group_id": current["group_identity"], "query_key": current["query_key"],
                                    "channels": {"micro": [item(1), item(3)]}}) + "\n")
        model_path = root / "model.joblib"
        model_path.write_bytes(b"fixture only; no real model fitting")
        declaration = E.candidate_metadata(candidates, protocol, cache, Path(E.__file__),
                                           model_path=model_path)
        declaration_path = root / "candidates.json"
        declaration_path.write_text(json.dumps(declaration), encoding="utf-8")
        config_path = root / "configuration.json"
        config_path.write_text(json.dumps({"experiment_version": 1, "phase": "tune", "variants": [
            {"name": "micro", "mode": "fill", "channels": {"micro": 1}, "addition_quota": 1}]}), encoding="utf-8")
        return argparse.Namespace(protocol_dir=protocol, cache_dir=cache,
            supplement=[candidates], supplement_metadata=[declaration_path], configuration=config_path,
            output=root / "result.json", text_limit=0, batch_size=1, bootstrap_replicates=20)

    def test_end_to_end_tune_only_preserves_full_denominator_and_no_overwrite(self):
        with TemporaryDirectory() as directory:
            args = self.fixture_evaluation(Path(directory))
            report = E.evaluate(args)
            self.assertTrue(report["complete"])
            self.assertEqual(report["phase"], "tune")
            current = report["results"]["variants"]["micro"]
            self.assertEqual(current["conditional_available"]["text_macro_candidate_recall"], 1.)
            self.assertAlmostEqual(current["full_truth"]["text_macro_candidate_recall"], 2 / 3)
            self.assertEqual(current["pool_statistics"]["average_size"], 3.)
            self.assertEqual(report["results"]["reference"]["conditional_available"]["pool"][
                "text_macro_candidate_recall"], .5)
            with self.assertRaises(ValueError):
                E.evaluate(args)

    def test_input_tampering_rejected_before_tune_labels_open(self):
        with TemporaryDirectory() as directory:
            args = self.fixture_evaluation(Path(directory))
            with args.supplement[0].open("ab") as stream:
                stream.write(b"tampered")
            with mock.patch.object(E.EVAL, "read_phase_labels") as read_labels:
                with self.assertRaises(ValueError):
                    E.evaluate(args)
                read_labels.assert_not_called()

    def test_disallowed_producer_training_declaration_rejected(self):
        with TemporaryDirectory() as directory:
            args = self.fixture_evaluation(Path(directory))
            declaration = json.loads(args.supplement_metadata[0].read_text())
            declaration["audit_targets_used"] = True
            args.supplement_metadata[0].write_text(json.dumps(declaration), encoding="utf-8")
            with mock.patch.object(E.EVAL, "read_phase_labels") as read_labels:
                with self.assertRaises(ValueError):
                    E.evaluate(args)
                read_labels.assert_not_called()


if __name__ == "__main__":
    unittest.main()
