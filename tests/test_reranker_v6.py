"""V6, explicit ranking groups, and selected-prefix inference invariants."""

from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from avito_candidates import cli
from avito_candidates.core import query_key
from avito_candidates.reranker import (
    FEATURES, GROUP_COLUMNS, LEGACY_FEATURES, V4_FEATURES, V6_FEATURES,
    FeatureBuilder, Reranker, fit_model, prediction_scores,
    prepare_training_data, retrieval_config, validate_training_groups,
)


def config(rescore=False):
    value = {"method": "hybrid", "candidate_k": 2, "bm25_weight": 1.0,
             "dense_weight": 1.0, "title_weight": 0.0, "params_weight": 0.0,
             "description_weight": 1.0, "stem_title_weight": 0.0,
             "stem_params_weight": 0.0, "rrf_k": 60, "location_bonus": 0.02}
    if rescore:
        value["rescore_dense_union"] = True
    return value


def grouped_table():
    rows = []
    for text in ("first", "second"):
        key_json = json.dumps(list(query_key({"search_query": text})),
                              ensure_ascii=False, separators=(",", ":"))
        group_id = hashlib.sha256(key_json.encode()).hexdigest()
        for label in (1, 0):
            rows.append({"group_id": group_id, "query_text": text,
                         "query_key_json": key_json, "item_id": f"{text}-{label}",
                         "relevant_count": 1, "label": label, "weight": 1.0})
    metadata = {"training_data_schema_version": 2,
                "group_columns": [name for name, _dtype in GROUP_COLUMNS],
                "stats": {"queries_with_positive_in_pool": 2,
                          "positive_rows": 2, "unobserved_rows": 2}}
    return pa.Table.from_pylist(rows), metadata


class FullPoolFeatureTests(unittest.TestCase):
    def setUp(self):
        self.items = [{"item_id": "lexical", "item_title_raw": "repair"},
                      {"item_id": "dense", "item_title_raw": "service"}]
        self.details = {"candidate_pool": ["lexical", "dense"],
                        "channels": {"description": ["lexical"], "dense": ["dense"]},
                        "channel_scores": {"description": {"lexical": 7.0},
                                           "dense": {"dense": 0.9}},
                        "full_dense_scores": {"lexical": 0.6, "dense": 0.9}}

    def test_lexical_only_candidate_gets_measured_cosine_and_known_indicator(self):
        query = {"search_query": "repair"}
        old = FeatureBuilder(self.items, config(), feature_names=V4_FEATURES).transform(
            query, self.details, ["lexical"])
        new = FeatureBuilder(self.items, config(True), feature_names=V6_FEATURES).transform(
            query, self.details, ["lexical"])
        np.testing.assert_array_equal(new[:, :len(V4_FEATURES)], old)
        self.assertEqual(new[0, V6_FEATURES.index("score_dense")], 0.0)
        self.assertAlmostEqual(new[0, V6_FEATURES.index("full_dense_score")], 0.6)
        self.assertEqual(new[:, V6_FEATURES.index("full_dense_known")].tolist(), [1.0, 1.0])

    def test_missing_or_nonfinite_score_is_rejected_even_outside_sampled_ids(self):
        builder = FeatureBuilder(self.items, config(True), feature_names=V6_FEATURES)
        for scores in ({"lexical": 0.6}, {"lexical": 0.6, "dense": float("nan")},
                       {"lexical": 0.6, "dense": float("inf")}):
            with self.subTest(scores=scores), self.assertRaisesRegex(ValueError, "V6 full dense scores"):
                builder.transform({}, {**self.details, "full_dense_scores": scores},
                                  ["lexical"], candidate_ids=["lexical"])

    def test_zero_cosine_is_known_and_empty_pool_requires_no_scores(self):
        builder = FeatureBuilder(self.items, config(True), feature_names=V6_FEATURES)
        matrix = builder.transform({}, {**self.details,
            "full_dense_scores": {"lexical": 0.0, "dense": 0.9}}, ["dense"])
        self.assertEqual(matrix[0, -2:].tolist(), [0.0, 1.0])
        empty = builder.transform({}, {"candidate_pool": [], "channels": {}}, [])
        self.assertEqual(empty.shape, (0, len(V6_FEATURES)))

    def test_old_schemas_ignore_added_score_map_and_v6_requires_opt_in(self):
        without = {key: value for key, value in self.details.items() if key != "full_dense_scores"}
        for features in (LEGACY_FEATURES, V4_FEATURES, FEATURES):
            builder = FeatureBuilder(self.items, config(), feature_names=features)
            np.testing.assert_array_equal(builder.transform({}, without, ["dense"]),
                                          builder.transform({}, self.details, ["dense"]))
        with self.assertRaisesRegex(ValueError, "V6 requires hybrid"):
            FeatureBuilder(self.items, config(), feature_names=V6_FEATURES)

    def test_v6_preparation_persists_grouped_77_feature_schema(self):
        details = self.details
        class Retrieval:
            def retrieve_batch_with_diagnostics(self, queries):
                return [(["lexical", "dense"], details) for _query in queries]
        query = {"search_query": "repair"}
        key = query_key(query)
        builder = FeatureBuilder(self.items, config(True), feature_names=V6_FEATURES)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.parquet"
            metadata = prepare_training_data(path, Retrieval(), builder,
                {key: {"lexical"}}, {key: query}, split="queries", fraction=0.2,
                seed=42, negatives=1, batch_size=1)
            table = pq.read_table(path)
        self.assertEqual(metadata["feature_version"], 6)
        self.assertEqual(metadata["feature_names"], list(V6_FEATURES))
        self.assertEqual(len(V6_FEATURES), 77)
        self.assertEqual(table["full_dense_known"].to_pylist(), [1.0, 1.0])
        self.assertEqual(validate_training_groups(table, metadata), [2])


class RankingGroupTests(unittest.TestCase):
    def test_explicit_groups_are_validated_without_identifier_features(self):
        table, metadata = grouped_table()
        self.assertEqual(validate_training_groups(table, metadata), [2, 2])
        self.assertTrue(set(name for name, _dtype in GROUP_COLUMNS).isdisjoint(V6_FEATURES))

    def test_noncontiguous_repeated_group_id_is_rejected(self):
        table, metadata = grouped_table()
        reordered = pa.Table.from_pylist([table.to_pylist()[i] for i in (0, 2, 1, 3)])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            validate_training_groups(reordered, metadata)

    def test_wrong_group_identity_counts_or_weights_are_rejected(self):
        table, metadata = grouped_table()
        for field, value in (("query_text", "different"), ("group_id", "wrong-hash"),
                             ("weight", 0.5), ("item_id", "first-0")):
            rows = table.to_pylist()
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_training_groups(pa.Table.from_pylist(rows), metadata)
        with self.assertRaisesRegex(ValueError, "counts differ"):
            validate_training_groups(table, {**metadata, "stats": {**metadata["stats"],
                                                                  "positive_rows": 3}})
        with self.assertRaisesRegex(ValueError, "newly prepared"):
            validate_training_groups(table, {**metadata, "training_data_schema_version": 1})

    def test_fit_ranker_uses_query_groups_and_saves_unselected_provenance(self):
        table, metadata = grouped_table()
        columns = {name: [0.0] * len(table) for name in V6_FEATURES}
        columns["full_dense_known"] = [1.0] * len(table)
        for name in table.column_names:
            columns[name] = table[name]
        table = pa.table(columns)
        metadata.update(feature_version=6, feature_names=list(V6_FEATURES),
                        retrieval_config=config(True), corpus_fingerprint="fingerprint",
                        split="queries", validation_fraction=0.2, seed=42,
                        training_prior_policy={"microcategory": "leave-query-text-out",
                                               "geography": "leave-query-text-out"})
        model = Mock()
        model.booster_.current_iteration.return_value = 3
        model.get_params.return_value = {"objective": "lambdarank"}
        with TemporaryDirectory() as directory:
            data = Path(directory) / "train.parquet"
            pq.write_table(table, data)
            data.with_suffix(".parquet.json").write_text(json.dumps(metadata), encoding="utf-8")
            joblib.dump({"microcategory_prior": None, "geo_associations": None},
                        data.with_suffix(".parquet.priors.joblib"))
            with patch("lightgbm.LGBMRanker", return_value=model) as factory, \
                    patch("avito_candidates.reranker.joblib.dump") as dump:
                report = fit_model(data, Path(directory) / "ranker.joblib",
                                   max_iter=3, estimator="lambdarank")
        self.assertEqual(model.fit.call_args.kwargs["group"], [2, 2])
        self.assertNotIn("sample_weight", model.fit.call_args.kwargs)
        self.assertEqual(model.fit.call_args.args[0].shape, (4, len(V6_FEATURES)))
        self.assertEqual(factory.call_args.kwargs["label_gain"], [0, 1])
        self.assertTrue(factory.call_args.kwargs["lambdarank_norm"])
        saved = dump.call_args.args[0]
        self.assertEqual(saved["selection_status"], "unselected")
        self.assertEqual(saved["provenance"]["training_data_sha256"],
                         report["provenance"]["training_data_sha256"])
        self.assertEqual(saved["training_configuration"]["group_count"], 2)


class ModelPrefixTests(unittest.TestCase):
    def test_ranker_raw_negative_scores_choose_top_candidates(self):
        model = Mock()
        model.predict.return_value = np.array([-5.0, -1.0])
        values = prediction_scores(model, np.zeros((2, 3)), estimator="lambdarank")
        self.assertEqual(values.tolist(), [-5.0, -1.0])
        model.predict_proba.assert_not_called()
        ranker = object.__new__(Reranker)
        ranker.estimator = "lambdarank"
        ranker.model = model
        ranker.builder = SimpleNamespace(transform=lambda *_args: np.zeros((2, 3)))
        result = (["a", "b"], {"candidate_pool": ["a", "b"]})
        self.assertEqual(ranker.rank_batch([{}], [result], limit=1), [["b"]])

    def test_lightgbm_receives_selected_available_iteration(self):
        model = Mock()
        model.booster_.current_iteration.return_value = 4
        model.predict.return_value = np.array([-1.0, 2.0])
        prediction_scores(model, np.zeros((2, 3)), estimator="lambdarank", selected_iteration=2)
        self.assertEqual(model.predict.call_args.kwargs["num_iteration"], 2)
        model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        prediction_scores(model, np.zeros((2, 3)), estimator="lightgbm", selected_iteration=3)
        self.assertEqual(model.predict_proba.call_args.kwargs["num_iteration"], 3)
        for stage in (0, 5, True, 1.5):
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "[Ss]elected iteration"):
                prediction_scores(model, np.zeros((2, 3)), selected_iteration=stage)

    def test_histgb_uses_requested_staged_probability_prefix(self):
        class Staged:
            n_iter_ = 3
            yielded = 0
            def staged_predict_proba(self, _matrix):
                for stage in range(1, 4):
                    self.yielded += 1
                    yield np.array([[1 - stage / 4, stage / 4]])
        model = Staged()
        result = prediction_scores(model, [[0]], selected_iteration=2)
        self.assertEqual(result.tolist(), [0.5])
        self.assertEqual(model.yielded, 2)


class CliOptInTests(unittest.TestCase):
    def test_preparation_defaults_to_verified_v4(self):
        query = {"search_query": "first"}
        key = query_key(query)
        args = ["avito", "prepare-reranker", "--data-dir", "unused", "--split", "queries"]
        with patch("sys.argv", args), patch.object(cli, "read", return_value=[{"item_id": "a"}]), \
                patch.object(cli, "build_retriever", return_value=SimpleNamespace()), \
                patch("avito_candidates.reranker.read_fit_labels", return_value=({key: {"a"}}, {key: query}, {})), \
                patch("avito_candidates.microcategory.fit_microcategory_prior", return_value=None), \
                patch("avito_candidates.reranker.prepare_training_data", return_value={}) as prepare:
            cli.main()
        self.assertEqual(prepare.call_args.args[2].feature_names, V4_FEATURES)

    def test_rescoring_and_v6_invalid_combinations_fail_before_data_access(self):
        for arguments in (
            ["evaluate", "--method", "bm25", "--rescore-dense-union"],
            ["prepare-reranker", "--data-dir", "unused", "--split", "queries",
             "--method", "hybrid", "--reranker-feature-version", "6"],
        ):
            with self.subTest(arguments=arguments), patch("sys.argv", ["avito", *arguments]), \
                    patch.object(cli, "read") as read, redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                cli.main()
            read.assert_not_called()

    def test_off_does_not_add_retrieval_config_key(self):
        args = SimpleNamespace(method="hybrid", candidate_k=2, k1=None, b=None,
            title_k1=1, title_b=0.2, title_weight=0, params_k1=1, params_b=0.2,
            params_weight=0, description_k1=1, description_b=0.2, description_weight=1,
            stem_title_weight=0, stem_params_weight=0, location_bonus=0.02,
            bm25_rrf_k=30, bm25_channel_quota=0, geo_candidate_k=0,
            embedding_model="model", embedding_max_length=128, embedding_params_words=40,
            embedding_description_words=48, ef_search=1000, rrf_k=60,
            bm25_weight=1, dense_weight=1, channel_quota=0, rescore_dense_union=False)
        old = retrieval_config(args)
        self.assertNotIn("rescore_dense_union", old)
        args.rescore_dense_union = True
        self.assertEqual(retrieval_config(args), {**old, "rescore_dense_union": True})


if __name__ == "__main__":
    unittest.main()
