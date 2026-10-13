import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from avito_candidates.core import held_out
from avito_candidates.microcategory import SoftMicrocategoryClassifier, _text_digest
from avito_candidates.reranker import FEATURE_SCHEMAS
from cache_research_reranker import (RESEARCH_POLICY, research_prior_policy,
    validate_research_artifact, check_cold_groups, check_sources, build_saved_retriever)
from cache_research_reranker import check_training_prior_file
from cache_research_reranker import check_supplemental_config
from cache_reranker_audit import digest


class CacheGuardTest(unittest.TestCase):
    def fixture(self, root):
        texts = {"a", "b", "c", "d"}
        metadata = {"source_sha256": "source", "corpus_fingerprint": "corpus",
            "allowed_fit_texts_sha256": _text_digest(texts),
            "observed_fit_texts_sha256": _text_digest(texts), "seed": 42, "crossfit_folds": 2}
        models = {fold: SimpleNamespace(fitted_texts=frozenset(
            text for text in texts if SoftMicrocategoryClassifier.fold(text, 2, 42) != fold),
            coef=np.array([float(fold)])) for fold in range(2)}
        classifier = SimpleNamespace(metadata=metadata,
            model=SimpleNamespace(fitted_texts=frozenset(texts), coef=np.array([1.])),
            crossfit_models=models, fold=SoftMicrocategoryClassifier.fold,
            _prediction_cache={})
        path = root / "classifier.joblib"
        path.write_bytes(b"classifier")
        path.with_suffix(".joblib.json").write_text(json.dumps({"metadata": metadata}))
        config = {"method": "hybrid", "microcategory_candidate_k": 100,
            "microcategory_selection_policy": "top998-then-novel-prefix",
            "microcategory_model": str(path), "microcategory_model_sha256": digest(path)}
        artifact = {"metadata": {"feature_version": 7, "feature_names": list(FEATURE_SCHEMAS[7]),
            "split": "queries", "seed": 42, "validation_fraction": .2,
            "training_prior_policy": dict(RESEARCH_POLICY), "retrieval_config": config,
            "corpus_fingerprint": "corpus", "microcategory_classifier_metadata": metadata},
            "priors": {"microcategory_classifier": classifier}}
        return artifact, texts, {"files": {"train": {"sha256": "source"}}}

    def test_saved_fit_only_oof_artifact_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact, texts, manifest = self.fixture(Path(directory))
            self.assertIs(validate_research_artifact(artifact, texts, manifest),
                          artifact["priors"]["microcategory_classifier"])

    def test_rejects_split_policy_and_classifier_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact, texts, manifest = self.fixture(Path(directory))
            for key, value in (("seed", 13), ("feature_version", 4),
                               ("training_prior_policy", {"microcategory": "full-fit-history"})):
                changed = copy.deepcopy(artifact)
                changed["metadata"][key] = value
                with self.assertRaises(ValueError):
                    validate_research_artifact(changed, texts, manifest)
            with self.assertRaisesRegex(ValueError, "fit partition"):
                validate_research_artifact(artifact, texts | {"new"}, manifest)

    def test_rejects_own_text_in_oof_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact, texts, manifest = self.fixture(Path(directory))
            classifier = artifact["priors"]["microcategory_classifier"]
            text = next(iter(texts))
            fold = classifier.fold(text, 2, 42)
            classifier.crossfit_models[fold].fitted_texts |= {text}
            with self.assertRaisesRegex(ValueError, "own held-out"):
                validate_research_artifact(artifact, texts, manifest)

    def test_v8_requires_measured_full_dense_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact, texts, manifest = self.fixture(Path(directory))
            version8 = (*FEATURE_SCHEMAS[7], "full_dense_score", "full_dense_known")
            with patch.dict(FEATURE_SCHEMAS, {8: version8}):
                artifact["metadata"]["feature_version"] = 8
                artifact["metadata"]["feature_names"] = list(version8)
                with self.assertRaisesRegex(ValueError, "full-pool"):
                    validate_research_artifact(artifact, texts, manifest)
                artifact["metadata"]["retrieval_config"]["rescore_dense_union"] = True
                validate_research_artifact(artifact, texts, manifest)

    def test_supplemental_prototype_policy_is_bound_exactly(self):
        config = {"supplemental_fill": "micro-typed-dense-v1", "supplemental_raw_depth": 998,
            "supplemental_rank_constant": 60, "supplemental_selection_policy": "raw-ranks-equal-route-novel-rrf-id-ties",
            "microcategory_candidate_k": 998, "microcategory_top_categories": 3,
            "microcategory_field": "combined"}
        self.assertEqual(check_supplemental_config(config), "micro-typed-dense-v1")
        for key, value in (("supplemental_raw_depth", 449), ("microcategory_top_categories", 1),
                           ("embedding_document_format", "compact"), ("supplemental_rank_constant", 30)):
            with self.assertRaises(ValueError):
                check_supplemental_config({**config, key: value})

    def test_external_classifier_bytes_cannot_change(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact, texts, manifest = self.fixture(Path(directory))
            Path(artifact["metadata"]["retrieval_config"]["microcategory_model"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "file provenance"):
                validate_research_artifact(artifact, texts, manifest)

    def test_extended_prior_hash_ignores_cache_but_tracks_model_coefficients(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact, _texts, _manifest = self.fixture(Path(directory))
            classifier = artifact["priors"]["microcategory_classifier"]
            first = research_prior_policy(artifact)
            classifier._prediction_cache["disposable"] = ["a"]
            self.assertEqual(first, research_prior_policy(artifact))
            classifier.model.coef[0] += .1
            self.assertNotEqual(first["research_priors_sha256"], research_prior_policy(artifact)["research_priors_sha256"])

    def test_cold_group_check_keeps_contexts_together_and_rejects_fit(self):
        cold = next("query" + str(i) for i in range(100) if held_out("query" + str(i), .2, 42))
        groups = [{"query_text": cold, "group_id": "a"}, {"query_text": cold, "group_id": "b"}]
        check_cold_groups(groups, {"another"})
        with self.assertRaises(ValueError):
            check_cold_groups(groups, {cold})
        with self.assertRaises(ValueError):
            check_cold_groups([groups[0], groups[0]], set())

    def test_source_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "producer.py"
            path.write_text("first")
            snapshot = {str(path): digest(path)}
            check_sources(snapshot)
            path.write_text("second")
            with self.assertRaises(ValueError):
                check_sources(snapshot)

    def test_training_prior_bytes_and_model_payload_must_match(self):
        import joblib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, _texts, _manifest = self.fixture(root)
            data = root / "train.parquet"
            path = data.with_suffix(".parquet.priors.joblib")
            joblib.dump(artifact["priors"], path)
            artifact["provenance"] = {"training_priors_sha256": digest(path)}
            self.assertEqual(check_training_prior_file(data, artifact), digest(path))
            artifact["priors"]["microcategory_classifier"].model.coef[0] += 1
            with self.assertRaisesRegex(ValueError, "feature-preparation"):
                check_training_prior_file(data, artifact)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed or is missing"):
                check_training_prior_file(data, artifact)

    def test_retriever_restores_saved_objects_and_requires_completed_index(self):
        config = {f"{field}_{name}": 1. for field in ("title", "params", "description")
                  for name in ("k1", "b", "weight")}
        config.update(stem_title_weight=0, stem_params_weight=0, location_bonus=.02,
            candidate_k=449, bm25_rrf_k=30, bm25_channel_quota=0, geo_candidate_k=100,
            embedding_model="frozen", embedding_max_length=128, embedding_params_words=40,
            embedding_description_words=48, ef_search=1000, rrf_k=60, bm25_weight=1,
            dense_weight=1, channel_quota=0, microcategory_top_categories=3,
            microcategory_candidate_k=100, microcategory_field="combined", microcategory_weight=.25)
        classifier, geo = object(), object()
        bm25 = SimpleNamespace(title_index="title", params_index="params", description_index="description")
        with patch("avito_candidates.baseline.Baseline", return_value=bm25) as lexical, \
             patch("avito_candidates.dense.DenseRetriever", return_value="dense") as dense, \
             patch("avito_candidates.microcategory.MicrocategoryChannel", return_value="micro") as micro, \
             patch("avito_candidates.hybrid.HybridRetriever", return_value="hybrid") as hybrid:
            result = build_saved_retriever(config, [], {"microcategory_classifier": classifier,
                "geo_associations": geo}, dense_cache="cache", batch_size=64, device="cuda")
            self.assertEqual(result, "hybrid")
            self.assertIs(lexical.call_args.kwargs["geo_associations"], geo)
            self.assertIs(micro.call_args.args[0], classifier)
            self.assertTrue(dense.call_args.kwargs["require_cached_index"])
            self.assertTrue(hybrid.call_args.kwargs["microcategory_exclude_self"])


if __name__ == "__main__":
    unittest.main()
