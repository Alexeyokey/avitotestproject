"""Execute exact CLI guard AST with tiny dependencies, without an ML process.

The optional path is an object/provenance optimisation; feature mathematics is
unchanged. Dependency spies prove which loads/scans are and are not performed.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch


def functions(path, names, namespace):
    nodes = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(nodes) != len(names):
        raise AssertionError("Production guard functions missing")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


class FakeReranker:
    pass


class SavedOverrideTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.model_path = self.path / "classifier.joblib"
        self.model_path.write_bytes(b"frozen-classifier-bytes")
        self.train_path = self.path / "train.parquet"
        self.train_path.write_bytes(b"frozen-training-source-bytes")
        source = Path(__file__).resolve().parents[1] / "src/avito_candidates"
        self.env = dict(__package__="avito_candidates", Path=Path, json=json,
                        normalize=lambda value: str(value).lower())
        functions(source / "cli.py", {"saved_microcategory_classifier", "load_microcategory_channel",
                                      "build_retriever"}, self.env)
        micro = ModuleType("avito_candidates.microcategory")
        helper = dict(hashlib=hashlib, Path=Path, json=json)
        functions(source / "microcategory.py", {"_canonical", "_sha_file", "_text_digest"}, helper)
        micro._sha_file = helper["_sha_file"]
        micro._text_digest = helper["_text_digest"]
        micro.SOFT_MICROCATEGORY_SCHEMA = "soft-microcategory-word-char-values-v1"
        self.load_calls, self.scan_calls, self.hash_calls = [], [], []
        sha = micro._sha_file
        def recorded_hash(path):
            self.hash_calls.append(Path(path))
            return sha(path)
        micro._sha_file = recorded_hash
        corpus = ModuleType("avito_candidates.reranker")
        body = dict(hashlib=hashlib, FEATURE_VERSION=5)
        functions(source / "reranker.py", {"corpus_fingerprint"}, body)
        corpus.corpus_fingerprint = body["corpus_fingerprint"]
        corpus.Reranker = FakeReranker
        self.items = [dict(item_id="a", item_title_raw="service", item_microcat_id="1")]
        observed = frozenset(("query one", "query two"))
        metadata = dict(schema=micro.SOFT_MICROCATEGORY_SCHEMA, crossfit_folds=3,
            observed_fit_texts_sha256=micro._text_digest(observed), fit_texts_count=2,
            allowed_fit_texts_count=2, source_sha256=sha(self.train_path), seed=42,
            corpus_fingerprint=corpus.corpus_fingerprint(self.items, feature_version=4), fit_split="queries")
        self.classifier = SimpleNamespace(metadata=metadata,
            model=SimpleNamespace(fitted_texts=observed, coefficients=object()),
            crossfit_models={0: None, 1: None, 2: None})
        self.reranker = FakeReranker()
        self.geography = object()
        self.reranker.builder = SimpleNamespace(microcategory_classifier=self.classifier,
                                                geo_associations=self.geography)
        self.args = SimpleNamespace(command="evaluate", data_dir=self.path, method="hybrid",
            split="queries", seed=42, validation_fraction=.2, train_batch_size=10,
            microcategory_model=self.model_path, microcategory_candidate_k=998,
            microcategory_top_categories=3, microcategory_field="combined", microcategory_weight=.25,
            geo_candidate_k=1, geo_top_locations=3, geo_min_history=20, geo_weight=.25,
            geo_include_related=False, k1=None, b=None, title_k1=1., title_b=.2,
            params_k1=1.2, params_b=.7, description_k1=1., description_b=.8,
            title_weight=0., params_weight=0., description_weight=.5,
            stem_title_weight=0., stem_params_weight=0., location_bonus=.02,
            candidate_k=449, bm25_rrf_k=30, bm25_channel_quota=0,
            embedding_model="frozen-encoder", dense_cache=self.path, embedding_batch_size=32,
            embedding_max_length=128, embedding_params_words=40, embedding_description_words=48,
            ef_search=1000, device="cpu", rrf_k=60, bm25_weight=1., dense_weight=1., channel_quota=0)
        def retrieval_config(args):
            return dict(classifier_sha=micro._sha_file(args.microcategory_model), candidate_k=args.candidate_k)
        def check_identity(metadata, config):
            if metadata["retrieval_config"] != config:
                raise ValueError("Retrieval identity changed")
        corpus.retrieval_config, corpus.check_retrieval_identity = retrieval_config, check_identity
        self.reranker.metadata = dict(feature_version=8, microcategory_classifier_metadata=copy.deepcopy(metadata),
            training_prior_policy={"microcategory_classifier": "grouped-out-of-fold"},
            corpus_fingerprint=metadata["corpus_fingerprint"], split="queries", seed=42,
            validation_fraction=.2, retrieval_config=retrieval_config(self.args))
        def load(*args, **kwargs):
            self.load_calls.append((args, kwargs))
            return self.classifier
        micro.SoftMicrocategoryClassifier = SimpleNamespace(load=load)
        micro.MicrocategoryChannel = lambda classifier, items, **kwargs: SimpleNamespace(classifier=classifier, **kwargs)
        def scan(*args, **kwargs):
            self.scan_calls.append((args, kwargs))
            return observed
        self.env["microcategory_fit_texts"] = scan
        baseline, dense, hybrid = (ModuleType("avito_candidates." + name)
                                   for name in ("baseline", "dense", "hybrid"))
        baseline.Baseline = lambda items, **kwargs: SimpleNamespace(title_index="t", params_index="p",
            description_index="d", geo_associations=kwargs["geo_associations"])
        dense.DenseRetriever = lambda items, **kwargs: SimpleNamespace(**kwargs)
        hybrid.HybridRetriever = lambda bm25, dense, **kwargs: SimpleNamespace(bm25=bm25, dense=dense, **kwargs)
        package = ModuleType("avito_candidates")
        package.__path__ = []
        self.modules = {"avito_candidates": package, micro.__name__: micro, corpus.__name__: corpus,
                        baseline.__name__: baseline, dense.__name__: dense, hybrid.__name__: hybrid}
        self.hash_calls.clear()

    def call(self):
        with patch.dict(sys.modules, self.modules):
            return self.env["saved_microcategory_classifier"](self.args, self.items, self.reranker)

    def test_reuses_exact_coefficient_object_without_load_or_parquet_scan(self):
        classifier = self.call()
        self.assertIs(classifier, self.classifier)
        self.assertIs(classifier.model.coefficients, self.classifier.model.coefficients)
        self.assertEqual(self.load_calls, [])
        self.assertEqual(self.scan_calls, [])
        self.assertEqual(self.hash_calls.count(self.train_path), 1)
        self.assertEqual(self.hash_calls.count(self.model_path), 1)

    def test_predict_keeps_saved_history_seed_instead_of_current_seed(self):
        self.args.command, self.args.seed, self.args.split = "predict", 123, "all"
        self.assertIs(self.call(), self.classifier)

    def test_changed_training_source_and_external_model_still_fail(self):
        self.train_path.write_bytes(b"changed training source")
        with self.assertRaisesRegex(ValueError, "training source differs"):
            self.call()
        self.train_path.write_bytes(b"frozen-training-source-bytes")
        self.model_path.write_bytes(b"changed coefficients")
        with self.assertRaisesRegex(ValueError, "Retrieval identity changed"):
            self.call()

    def test_snapshot_inventory_corpus_and_missing_oof_fold_fail(self):
        self.classifier.metadata["seed"] = 17
        with self.assertRaisesRegex(ValueError, "matching V7/V8"):
            self.call()
        self.classifier.metadata["seed"] = 42
        self.classifier.model.fitted_texts = frozenset(("different",))
        with self.assertRaisesRegex(ValueError, "fit-text inventory"):
            self.call()
        self.classifier.model.fitted_texts = frozenset(("query one", "query two"))
        self.classifier.crossfit_models.pop(2)
        with self.assertRaisesRegex(ValueError, "OOF provenance"):
            self.call()
        self.classifier.crossfit_models[2] = None
        self.items[0]["item_title_raw"] = "changed corpus"
        with self.assertRaisesRegex(ValueError, "corpus differs"):
            self.call()

    def test_evaluate_rejects_different_split_and_prepare_cannot_override(self):
        self.args.seed = 13
        with self.assertRaisesRegex(ValueError, "original query split"):
            self.call()
        self.args.command = "prepare-reranker"
        with self.assertRaisesRegex(ValueError, "evaluate/predict"):
            self.call()

    def test_build_reuses_exact_geography_without_refitting(self):
        with patch.dict(sys.modules, self.modules):
            result = self.env["build_retriever"](self.args, self.items, saved_reranker_override=self.reranker)
        self.assertIs(result.bm25.geo_associations, self.geography)
        self.assertIs(result.microcategory_channel.classifier, self.classifier)
        self.assertEqual(self.load_calls, [])
        self.assertEqual(self.scan_calls, [])

    def test_missing_or_conflicting_saved_geo_history_fails(self):
        with patch.dict(sys.modules, self.modules):
            with self.assertRaisesRegex(ValueError, "Geography override differs"):
                self.env["build_retriever"](self.args, self.items,
                    geo_associations_override=object(), saved_reranker_override=self.reranker)
            self.reranker.builder.geo_associations = None
            with self.assertRaisesRegex(ValueError, "lacks geography history"):
                self.env["build_retriever"](self.args, self.items, saved_reranker_override=self.reranker)

    def test_default_standalone_path_retains_allowlist_and_model_loading(self):
        self.model_path.with_suffix(".joblib.json").write_text(json.dumps({"metadata": self.classifier.metadata}))
        with patch.dict(sys.modules, self.modules):
            channel = self.env["load_microcategory_channel"](self.args, self.items,
                SimpleNamespace(title_index="t", params_index="p", description_index="d"))
        self.assertIs(channel.classifier, self.classifier)
        self.assertEqual(len(self.load_calls), 1)
        self.assertEqual(len(self.scan_calls), 1)


if __name__ == "__main__":
    unittest.main()
