"""Local checkpoint identity and saved-geography retrieval invariants."""

import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from avito_candidates import cli
from avito_candidates.dense import (
    local_checkpoint_content_fingerprint, offline_adapter_base_reference,
)
from avito_candidates.core import query_key
from avito_candidates.reranker import (
    V4_FEATURES, FeatureBuilder, Reranker, check_retrieval_identity,
    prepare_training_data,
)


BASE_REPO = "Octen/Octen-Embedding-0.6B"
BASE_COMMIT = "a" * 40


class CheckpointIdentityTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.original = self.root / "original"
        self.alias = self.root / "relocated"
        self.cache = self.root / "hub"
        self.original.mkdir()
        (self.original / "adapter_model.safetensors").write_bytes(b"adapter weights")
        (self.original / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": BASE_REPO, "revision": None}), encoding="utf-8")
        (self.original / "config_sentence_transformers.json").write_text(
            '{"prompts":{"query":"search: ","document":"item: "}}', encoding="utf-8")
        shutil.copytree(self.original, self.alias)
        self.reference = self.cache / "models--Octen--Octen-Embedding-0.6B" / "refs" / "main"
        self.reference.parent.mkdir(parents=True)
        self.reference.write_text(BASE_COMMIT, encoding="utf-8")
        self.environment = patch.dict(os.environ, {"HF_HUB_CACHE": str(self.cache)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.config = {"method": "hybrid", "embedding_model": str(self.original),
                       "embedding_max_length": 128, "candidate_k": 449}
        self.metadata = {"retrieval_config": self.config,
                         "embedding_checkpoint_sha256": local_checkpoint_content_fingerprint(self.original),
                         "embedding_is_adapter": True, "embedding_base_repo": BASE_REPO,
                         "embedding_base_revision": BASE_COMMIT}

    def test_identical_checkpoint_bytes_at_different_path_are_accepted(self):
        self.assertEqual(local_checkpoint_content_fingerprint(self.original),
                         local_checkpoint_content_fingerprint(self.alias))
        check_retrieval_identity(self.metadata, {**self.config, "embedding_model": str(self.alias)})
        with self.assertRaisesRegex(ValueError, "retrieval settings"):
            check_retrieval_identity(self.metadata, {**self.config,
                "embedding_model": str(self.alias), "embedding_max_length": 256})

    def test_changed_weights_are_rejected_at_original_or_relocated_path(self):
        for path in (self.original, self.alias):
            (path / "adapter_model.safetensors").write_bytes(b"changed adapter weights")
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "checkpoint content"):
                check_retrieval_identity(self.metadata, {**self.config, "embedding_model": str(path)})

    def test_changed_or_missing_cached_base_revision_is_rejected(self):
        self.reference.write_text("b" * 40, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "base revision"):
            check_retrieval_identity(self.metadata, self.config)
        self.reference.unlink()
        with self.assertRaisesRegex(ValueError, "base revision"):
            check_retrieval_identity(self.metadata, {**self.config, "embedding_model": str(self.alias)})

    def test_legacy_artifacts_keep_strict_path_comparison(self):
        legacy = {"retrieval_config": self.config}
        check_retrieval_identity(legacy, self.config)
        with self.assertRaisesRegex(ValueError, "retrieval settings"):
            check_retrieval_identity(legacy, {**self.config, "embedding_model": str(self.alias)})

    def test_unverified_adapter_base_disallows_alias(self):
        self.reference.unlink()
        unverified = {**self.metadata, "embedding_base_revision": None}
        check_retrieval_identity(unverified, self.config)
        with self.assertRaisesRegex(ValueError, "verified cached adapter base revision"):
            check_retrieval_identity(unverified, {**self.config, "embedding_model": str(self.alias)})

    def test_pinned_adapter_commit_uses_cached_snapshot_not_mutable_main(self):
        config_path = self.original / "adapter_config.json"
        config_path.write_text(json.dumps({"base_model_name_or_path": BASE_REPO,
                                          "revision": BASE_COMMIT}), encoding="utf-8")
        snapshot = self.cache / "models--Octen--Octen-Embedding-0.6B" / "snapshots" / BASE_COMMIT
        snapshot.mkdir(parents=True)
        self.reference.write_text("b" * 40, encoding="utf-8")
        result = offline_adapter_base_reference(self.original)
        self.assertEqual(result, {"is_adapter": True, "base_repo": BASE_REPO,
                                  "base_revision": BASE_COMMIT})

    def test_full_checkpoint_needs_no_separate_adapter_base_identity(self):
        for path in (self.original, self.alias):
            (path / "adapter_config.json").unlink()
        metadata = {**self.metadata, "embedding_is_adapter": False,
                    "embedding_base_repo": None, "embedding_base_revision": None,
                    "embedding_checkpoint_sha256": local_checkpoint_content_fingerprint(self.original)}
        check_retrieval_identity(metadata, {**self.config, "embedding_model": str(self.alias)})

    def test_preparation_records_identity_and_reranker_accepts_verified_relocation(self):
        config = {**self.config, "bm25_weight": 1.0, "dense_weight": 1.0,
                  "title_weight": 0.0, "params_weight": 0.0, "description_weight": 0.5,
                  "stem_title_weight": 0.0, "stem_params_weight": 0.0,
                  "rrf_k": 60, "location_bonus": 0.02}
        items = [{"item_id": "a"}, {"item_id": "b"}]
        query = {"search_query": "repair"}
        key = query_key(query)
        class Retrieval:
            def retrieve_batch_with_diagnostics(self, queries):
                return [(["a", "b"], {"candidate_pool": ["a", "b"],
                    "channels": {"description": ["a", "b"]}}) for _query in queries]
        metadata = prepare_training_data(self.root / "train.parquet", Retrieval(),
            FeatureBuilder(items, config, feature_names=V4_FEATURES), {key: {"a"}},
            {key: query}, split="queries", fraction=0.2, seed=42, negatives=1, batch_size=1)
        self.assertEqual(metadata["embedding_checkpoint_sha256"],
                         local_checkpoint_content_fingerprint(self.original))
        self.assertEqual(metadata["embedding_base_repo"], BASE_REPO)
        self.assertEqual(metadata["embedding_base_revision"], BASE_COMMIT)
        artifact = {"model": SimpleNamespace(), "metadata": metadata, "priors": {}}
        with patch("avito_candidates.reranker.joblib.load", return_value=artifact):
            ranker = Reranker("unused.joblib", items, {**config, "embedding_model": str(self.alias)})
        self.assertEqual(ranker.builder.config["embedding_model"], str(self.alias))


def geo_args():
    return SimpleNamespace(command="predict", method="bm25", data_dir=Path("unused"),
        split="queries", validation_fraction=0.2, seed=42, train_batch_size=64,
        k1=None, b=None, title_k1=1.0, title_b=0.2, params_k1=1.2, params_b=0.7,
        description_k1=1.0, description_b=0.8, title_weight=0.0, params_weight=0.0,
        description_weight=0.5, stem_title_weight=0.0, stem_params_weight=0.0,
        location_bonus=0.02, candidate_k=449, bm25_rrf_k=30, bm25_channel_quota=0,
        geo_candidate_k=100, geo_top_locations=3, geo_include_related=False,
        geo_min_history=20, geo_weight=0.25, reranker_crossfit_geography=False)


class SavedGeographyTests(unittest.TestCase):
    def test_supplied_saved_geography_prevents_refitting_train_rows(self):
        saved = object()
        with patch("avito_candidates.geography.fit_location_associations") as refit, \
                patch("avito_candidates.baseline.Baseline") as baseline:
            cli.build_retriever(geo_args(), [{"item_id": "a", "item_location_id": "city"}],
                                geo_associations_override=saved)
        refit.assert_not_called()
        self.assertIs(baseline.call_args.kwargs["geo_associations"], saved)

    def test_none_legacy_override_retains_original_all_train_prediction_fit(self):
        rebuilt = object()
        with patch("avito_candidates.geography.fit_location_associations", return_value=rebuilt) as refit, \
                patch("avito_candidates.baseline.Baseline") as baseline:
            cli.build_retriever(geo_args(), [{"item_id": "a", "item_location_id": "city"}])
        self.assertEqual(refit.call_args.kwargs["split"], "all")
        self.assertIs(baseline.call_args.kwargs["geo_associations"], rebuilt)


if __name__ == "__main__":
    unittest.main()
