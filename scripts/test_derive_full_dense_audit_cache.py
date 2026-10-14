"""Tiny Parquet fixtures for a schema bridge with fixed candidate identity."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from derive_full_dense_audit_cache import (
    ALIGNMENT_PROTOCOL, CandidatePoolDigest, align_full_dense_group,
    derive_validation_batch, derive_validation_cache, iter_complete_query_groups,
    validate_bridge_metadata, validate_rescore_config, verify_corpus,
)
from reranker_audit_provenance import artifact_prior_policy


BASE_FEATURES = ["rrf_score", "baseline_top50", "score_dense", "present_dense", "rank_dense",
                 *[f"feature_{i}" for i in range(70)]]


def fixture_fingerprint(items, *, feature_version):
    if feature_version not in (4, 6):
        raise AssertionError("Both schema versions must be verified")
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


class FullDenseBridgeTest(unittest.TestCase):
    def make_fixture(self, directory):
        root = Path(directory)
        source_dir = root / "source"
        source_dir.mkdir()
        items = [{"item_id": f"item-{i}", "item_title_raw": "ремонт"} for i in range(20)]
        groups = [
            {"group_id": 0, "query_key": ["ремонт", "city", "", "", ""], "query_text": "ремонт",
             "relevant_count": 2, "local_relevant_count": 1, "nonlocal_relevant_count": 1,
             "seen_relevant_count": 1, "unseen_relevant_count": 1, "pool_size": 10,
             "positives_in_pool": 1, "baseline_hits": 1, "partition": "tune"},
            {"group_id": 1, "query_key": ["монтаж", "city", "", "", ""], "query_text": "монтаж",
             "relevant_count": 1, "local_relevant_count": 0, "nonlocal_relevant_count": 1,
             "seen_relevant_count": 0, "unseen_relevant_count": 1, "pool_size": 10,
             "positives_in_pool": 1, "baseline_hits": 0, "partition": "confirmation"},
        ]
        source = {
            "cache_version": 2, "role": "validation", "feature_schema_version": 4,
            "feature_names": BASE_FEATURES, "retrieval_config": {
                "method": "hybrid", "candidate_k": 449, "channel_quota": 0},
            "corpus_fingerprint": fixture_fingerprint(items, feature_version=4),
            "split": "queries", "seed": 42, "validation_fraction": 0.2,
            "rows": 20, "groups": groups, "pool_recall": 0.75,
            "baseline_recall_at_50": 0.25, "model_sha256": "original-reference",
            "historical_holdout_reused": True, **artifact_prior_policy({"priors": None}),
        }
        fields = {name: pa.array(np.zeros(20, dtype=np.float32)) for name in BASE_FEATURES}
        fields["rrf_score"] = pa.array(np.linspace(0.5, 0.1, 20, dtype=np.float32))
        fields["baseline_top50"] = pa.array(np.array([1] * 19 + [0], dtype=np.float32))
        cosines = np.r_[np.linspace(.9, .7, 10), np.linspace(.8, .6, 10)].astype(np.float32)
        present = np.array(([1] * 8 + [0] * 2) * 2, dtype=np.float32)
        fields["score_dense"] = pa.array(cosines * present)
        fields["present_dense"] = pa.array(present)
        fields["rank_dense"] = pa.array(np.array((list(range(1, 9)) + [0, 0]) * 2, dtype=np.float32))
        raw_scores = cosines * np.repeat(np.array([1.0035, .997], dtype=np.float32), 10)
        fields.update({"group_id": pa.array([0] * 10 + [1] * 10, type=pa.int32()),
                       "item_id": pa.array([item["item_id"] for item in items]),
                       "label": pa.array([1] + [0] * 18 + [1], type=pa.int8()),
                       "positive_local": pa.array([True] + [False] * 19),
                       "positive_seen": pa.array([True] + [False] * 19),
                       "full_dense_score": pa.array(raw_scores),
                       "item_microcat_id": pa.array(["service"] * 20)})
        table = pa.table(fields)
        pq.write_table(table, source_dir / "validation.parquet", row_group_size=2)
        (source_dir / "validation.json").write_text(json.dumps(source), encoding="utf-8")
        training = {
            "feature_version": 6, "feature_names": [*BASE_FEATURES, "full_dense_score", "full_dense_known"],
            "retrieval_config": {**source["retrieval_config"], "rescore_dense_union": True},
            "corpus_fingerprint": fixture_fingerprint(items, feature_version=6),
            "split": "queries", "seed": 42, "validation_fraction": 0.2,
            "training_prior_policy": {"microcategory": "leave-query-text-out",
                                      "geography": "leave-query-text-out"},
        }
        artifact = {"model": SimpleNamespace(n_features_in_=77), "metadata": deepcopy(training), "priors": None}
        model = root / "reference.joblib"
        joblib.dump(artifact, model)
        train_path = root / "train.parquet"
        train_fields = {name: pa.array(np.ones(2, dtype=np.float32)) for name in training["feature_names"]}
        train_fields.update(label=pa.array([1, 0], type=pa.int8()), weight=pa.array([1., 1.], type=pa.float32()))
        pq.write_table(pa.table(train_fields), train_path)
        train_path.with_suffix(".parquet.json").write_text(json.dumps(training), encoding="utf-8")
        return source_dir, source, table, train_path, training, model, artifact, items

    def test_bridge_preserves_rows_features_denominators_and_order_across_batches(self):
        with TemporaryDirectory() as directory:
            source_dir, source, table, train, _training, model, _artifact, items = self.make_fixture(directory)
            original_bytes = (source_dir / "validation.parquet").read_bytes()
            output = Path(directory) / "derived"
            versions = []
            def fingerprint(rows, *, feature_version):
                versions.append(feature_version)
                return fixture_fingerprint(rows, feature_version=feature_version)
            result = derive_validation_cache(source_dir, train, model, "unused", output,
                                             batch_size=2, items=items, fingerprint_function=fingerprint)
            self.assertEqual(versions, [4, 6])
            derived = pq.read_table(output / "validation.parquet")
            original_columns = [name for name in table.schema.names if name != "full_dense_score"]
            self.assertEqual(derived.select(original_columns).to_pylist(), table.select(original_columns).to_pylist())
            expected_cosines = np.r_[np.linspace(.9, .7, 10), np.linspace(.8, .6, 10)].astype(np.float32)
            np.testing.assert_allclose(derived["full_dense_score"].to_numpy(), expected_cosines, atol=1e-7)
            self.assertEqual(derived["full_dense_known"].to_pylist(), [1.] * 20)
            alignment = result["derivation"]["dense_score_alignment"]
            self.assertEqual(alignment["protocol"], ALIGNMENT_PROTOCOL)
            self.assertEqual(alignment["query_groups_aligned"], 2)
            self.assertAlmostEqual(alignment["query_norm_min"], .997, places=6)
            self.assertAlmostEqual(alignment["query_norm_max"], 1.0035, places=6)
            self.assertLess(alignment["max_abs_cosine_residual_vs_hnsw"], 1e-5)
            self.assertEqual(result["groups"], source["groups"])
            self.assertEqual(result["pool_recall"], 0.75)
            self.assertEqual(result["baseline_recall_at_50"], 0.25)
            self.assertEqual(result["feature_names"], [*BASE_FEATURES, "full_dense_score", "full_dense_known"])
            stamped = json.loads((source_dir / "validation.json").read_text())
            self.assertEqual(stamped["candidate_pool_digest"], result["candidate_pool_digest"])
            self.assertEqual((source_dir / "validation.parquet").read_bytes(), original_bytes)
            self.assertFalse((output / "train_sample.parquet").exists())
            self.assertFalse((output / "train_sample.json").exists())
            repeated = derive_validation_cache(source_dir, train, model, "unused", output,
                                               items=items, fingerprint_function=fixture_fingerprint)
            self.assertEqual(repeated["candidate_pool_digest"], result["candidate_pool_digest"])

    def test_digest_is_batch_invariant_but_captures_candidate_order(self):
        with TemporaryDirectory() as directory:
            _sd, source, table, *_rest = self.make_fixture(directory)
            first = CandidatePoolDigest(source)
            first.update(table)
            second = CandidatePoolDigest(source)
            for batch in table.to_batches(max_chunksize=2):
                second.update(batch)
            self.assertEqual(first.finish(), second.finish())
            changed = CandidatePoolDigest(source)
            changed.update(table.take(pa.array([1, 0, *range(2, 20)])))
            self.assertNotEqual(first.finish(), changed.finish())

    def test_nan_or_unknown_full_dense_rows_are_not_marked_known(self):
        with TemporaryDirectory() as directory:
            _sd, _source, table, *_rest = self.make_fixture(directory)
            scores = table["full_dense_score"].to_numpy().copy()
            scores[1] = np.nan
            bad = table.set_column(table.schema.get_field_index("full_dense_score"),
                                   "full_dense_score", pa.array(scores))
            with self.assertRaisesRegex(ValueError, "finite cosine"):
                derive_validation_batch(bad)
            bad = table.append_column("full_dense_known", pa.array(np.zeros(20, dtype=np.float32)))
            with self.assertRaisesRegex(ValueError, "unknown rows"):
                derive_validation_batch(bad)

    def test_quota_change_and_feature_order_change_are_rejected(self):
        with TemporaryDirectory() as directory:
            _sd, source, _table, _train, training, _model, artifact, _items = self.make_fixture(directory)
            bad_config = {**training["retrieval_config"], "channel_quota": 10}
            with self.assertRaisesRegex(ValueError, "beyond union"):
                validate_rescore_config(source["retrieval_config"], bad_config)
            bad_training = deepcopy(training)
            bad_training["feature_names"][-2:] = list(reversed(bad_training["feature_names"][-2:]))
            with self.assertRaisesRegex(ValueError, "exactly source75"):
                validate_bridge_metadata(source, bad_training, artifact)

    def test_changed_prior_and_actual_inventory_are_rejected(self):
        with TemporaryDirectory() as directory:
            _sd, source, _table, _train, training, _model, artifact, items = self.make_fixture(directory)
            changed = deepcopy(artifact)
            changed["priors"] = {"microcategory_prior": SimpleNamespace(
                exact_counts={"ремонт": {"service": 1}}, token_counts={}, token_totals={})}
            with self.assertRaisesRegex(ValueError, "prior policy"):
                validate_bridge_metadata(source, training, changed)
            changed_items = deepcopy(items)
            changed_items[0]["item_title_raw"] = "другая услуга"
            with self.assertRaisesRegex(ValueError, "actual inventory"):
                verify_corpus(source, artifact["metadata"], "unused", items=changed_items,
                              fingerprint_function=fixture_fingerprint)

    def test_failed_nan_derivation_does_not_stamp_source_or_commit_target(self):
        with TemporaryDirectory() as directory:
            source_dir, _source, table, train, _training, model, _artifact, items = self.make_fixture(directory)
            bad = table.set_column(table.schema.get_field_index("full_dense_score"), "full_dense_score",
                                   pa.array(np.full(20, np.nan, dtype=np.float32)))
            pq.write_table(bad, source_dir / "validation.parquet")
            before = (source_dir / "validation.json").read_bytes()
            output = Path(directory) / "derived"
            with self.assertRaisesRegex(ValueError, "finite cosine"):
                derive_validation_cache(source_dir, train, model, "unused", output,
                                        items=items, fingerprint_function=fixture_fingerprint)
            self.assertEqual((source_dir / "validation.json").read_bytes(), before)
            self.assertFalse((output / "validation.parquet").exists())
            self.assertFalse((output / "validation.json").exists())

    def test_alignment_matches_direct_normalized_query_dot_on_entire_union(self):
        with TemporaryDirectory() as directory:
            _sd, _source, table, *_rest = self.make_fixture(directory)
            query = np.array([.8008, .6006], dtype=np.float32)
            angles = np.linspace(.2, 1.2, 10, dtype=np.float32)
            stored = np.stack((np.cos(angles), np.sin(angles)), axis=1).astype(np.float32)
            stored /= np.linalg.norm(stored, axis=1, keepdims=True)
            expected = stored @ (query / float(np.linalg.norm(query)))
            raw = stored @ query
            group = table.slice(0, 10)
            group = group.set_column(group.schema.get_field_index("full_dense_score"), "full_dense_score", pa.array(raw))
            present = group["present_dense"].to_numpy()
            group = group.set_column(group.schema.get_field_index("score_dense"), "score_dense", pa.array(expected * present))
            aligned, stats = align_full_dense_group(group)
            np.testing.assert_allclose(aligned["full_dense_score"].to_numpy(), expected, atol=2e-7)
            self.assertEqual(stats["dense_pairs"], 8)
            self.assertAlmostEqual(stats["query_norm"], float(np.linalg.norm(query)), places=6)

    def test_nonconstant_ratios_and_insufficient_pairs_are_rejected(self):
        with TemporaryDirectory() as directory:
            _sd, _source, table, *_rest = self.make_fixture(directory)
            group = table.slice(0, 10)
            raw = group["full_dense_score"].to_numpy().copy()
            raw[0] += .002
            bad = group.set_column(group.schema.get_field_index("full_dense_score"), "full_dense_score", pa.array(raw))
            with self.assertRaisesRegex(ValueError, "not constant"):
                align_full_dense_group(bad)
            with self.assertRaisesRegex(ValueError, "Insufficient safe dense pairs"):
                align_full_dense_group(group.slice(0, 7))

    def test_stream_buffers_only_whole_groups_and_rejects_noncontiguous_groups(self):
        with TemporaryDirectory() as directory:
            _sd, _source, table, *_rest = self.make_fixture(directory)
            groups = list(iter_complete_query_groups(table.to_batches(max_chunksize=3)))
            self.assertEqual([group["group_id"].to_pylist() for group in groups], [[0] * 10, [1] * 10])
            self.assertEqual(pa.concat_tables(groups).to_pylist(), table.to_pylist())
            with self.assertRaisesRegex(ValueError, "exceeds"):
                list(iter_complete_query_groups(table.to_batches(max_chunksize=3), max_pool_size=9))
            reordered = table.take(pa.array([*range(10, 20), *range(10)]))
            with self.assertRaisesRegex(ValueError, "not contiguous"):
                list(iter_complete_query_groups(reordered.to_batches(max_chunksize=3)))


if __name__ == "__main__":
    unittest.main()
