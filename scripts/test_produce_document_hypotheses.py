"""Document producer guards and transport tests without a GPU or large catalog."""
from pathlib import Path
import json
import gzip
import os
from decimal import Decimal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from produce_document_hypotheses import (check_encoder, rename_channels, completed_index_identity,
                                         main, produce, iter_catalog_records,
                                         load_catalog_documents, DOCUMENT_FIELDS,
                                         write_encoder_base_reference, check_encoder_base,
                                         require_offline_environment)
from avito_candidates.dense import _fingerprint, local_checkpoint_content_fingerprint
from avito_candidates.reranker import corpus_fingerprint
from produce_retrieval_hypotheses import json_digest, sha256


class ProducerTest(unittest.TestCase):
    def test_frozen_encoder_and_length_bound(self):
        manifest = {"encoder": {"path": "/tmp/model", "content_sha256": "sealed"}}
        config = {"embedding_model": "/tmp/model", "embedding_max_length": 128}
        with patch("produce_document_hypotheses.local_checkpoint_content_fingerprint", return_value="sealed"):
            self.assertEqual(check_encoder(manifest, config), Path("/tmp/model").resolve())
            with self.assertRaisesRegex(ValueError, "path or bytes"):
                check_encoder(manifest, {**config, "embedding_model": "/tmp/other"})
            with self.assertRaisesRegex(ValueError, "128-token"):
                check_encoder(manifest, {**config, "embedding_max_length": 256})
        with patch("produce_document_hypotheses.local_checkpoint_content_fingerprint", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "path or bytes"):
                check_encoder(manifest, config)

    def test_query_and_document_axes_remain_separate(self):
        rows = [{"dense_current": ["a", "b"], "dense_values": ["b", "a"]}]
        self.assertEqual(rename_channels(rows, "compact", ["current", "values"]),
                         [{"dense_compact_current": ["a", "b"], "dense_compact_values": ["b", "a"]}])
        self.assertEqual(rows[0]["dense_current"], ["a", "b"])
        with self.assertRaises(ValueError):
            rename_channels(rows, "current", ["current", "values"])
        with self.assertRaises(ValueError):
            rename_channels(rows, "compact", ["current"])

    def test_index_identity_refuses_partial_or_metadata_mismatch(self):
        items = [{"item_id": "a", "item_title_raw": "title"}]
        meta = {"document_format": "compact", "document_format_version": 1}
        dense = SimpleNamespace(model_name="model", max_seq_length=128, params_words=40,
            description_words=48, model_signature="sig", document_format="compact",
            _metadata=lambda: meta)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "not complete"):
                completed_index_identity(dense, items, directory)
            fingerprint = _fingerprint(items, "model", 128, 40, 48, "sig", "compact")
            path = Path(directory)
            (path / (fingerprint + ".bin")).write_bytes(b"completed-index")
            metadata = path / (fingerprint + ".json")
            metadata.write_text(json.dumps(meta))
            identity = completed_index_identity(dense, items, directory)
            self.assertEqual(identity["fingerprint"], fingerprint)
            self.assertEqual(len(identity["index"]["sha256"]), 64)
            metadata.write_text(json.dumps({**meta, "document_format": "current"}))
            with self.assertRaises(ValueError):
                completed_index_identity(dense, items, directory)

    def test_cli_refuses_repeated_variants_or_bad_budget_before_loading(self):
        base = ["--protocol-dir", "protocol", "--cache-dir", "cache", "--dense-cache", "new-index",
                "--output", "output.jsonl", "--document-format", "compact",
                "--encoder-base-reference", "frozen-base.json"]
        with patch("produce_document_hypotheses.produce") as produce:
            for extra in (["--query-variants", "current", "current"], ["--depth", "999"],
                          ["--batch-size", "0"]):
                with self.assertRaises(SystemExit):
                    main(base + extra)
            produce.assert_not_called()
            main(base + ["--query-variants", "current", "values"])
            self.assertEqual(produce.call_args.args[0].depth, 449)
            self.assertEqual(produce.call_args.args[0].document_format, "compact")

    def test_transport_is_complete_target_blind_and_refuses_overwrite(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        key = ["ремонт", "1", "0", "", "114"]
        item_id = "0123456789abcdef"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            parquet = cache / "tune.parquet"
            parquet.write_bytes(b"sealed-cache")
            metadata_path = cache / "tune.json"
            metadata = {"fresh_partition": "tune", "historical_holdout_reused": False,
                "groups": [{"group_identity": json_digest(key), "query_key": key}],
                "corpus_fingerprint": "same", "retrieval_config": {
                    "embedding_model": "/tmp/frozen", "embedding_max_length": 128,
                    "embedding_params_words": 40, "embedding_description_words": 48,
                    "ef_search": 1000}}
            metadata_path.write_text(json.dumps(metadata))
            fit_path = root / "fit-texts.json"
            fit_path.write_text(json.dumps({"texts": ["different"]}))
            item_path = root / "items.parquet"
            pq.write_table(pa.table({"item_id": [item_id]}), item_path)
            args = SimpleNamespace(protocol_dir=root, cache_dir=cache, dense_cache=root / "separate-index",
                output=root / "candidates.jsonl.gz", document_format="compact", batch_size=64,
                query_variants=["current", "plain"], depth=449, device="cuda",
                encoder_base_reference=root / "frozen-base.json")
            manifest = {"encoder": {"content_sha256": "sealed"}}
            files_read = []

            def protocol_file(_root, _manifest, name):
                files_read.append(name)
                if name == "fit-texts.json":
                    return fit_path
                if name == "benchmark_items":
                    return item_path
                raise AssertionError("Producer must not request any relevance file")

            def candidate_metadata(*_args, **kwargs):
                return {"complete": True, "producer_details": kwargs["extra"]}

            with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}), \
                 patch("produce_document_hypotheses.EVAL.checked_protocol", return_value=(manifest, "protocol")), \
                 patch("produce_document_hypotheses.EVAL.check_phase_cache_inputs", return_value=sha256(parquet)), \
                 patch("produce_document_hypotheses.EVAL.protocol_file", side_effect=protocol_file), \
                 patch("produce_document_hypotheses.load_catalog_documents", return_value=([{"item_id": item_id}], "same")), \
                 patch("produce_document_hypotheses.check_encoder", return_value=Path("/tmp/frozen")), \
                 patch("produce_document_hypotheses.check_encoder_base", return_value={"base_revision": "frozen"}), \
                 patch("produce_document_hypotheses.DenseRetriever") as constructor, \
                 patch("produce_document_hypotheses.completed_index_identity", return_value={"fingerprint": "new-index"}), \
                 patch("produce_document_hypotheses.HELPER.dense_query_channels", return_value=[{
                     "dense_current": [item_id], "dense_plain": [item_id]}]), \
                 patch("produce_document_hypotheses.candidate_metadata", side_effect=candidate_metadata):
                result = produce(args)
                self.assertEqual(constructor.call_args.kwargs["document_format"], "compact")
                self.assertEqual(constructor.call_args.kwargs["cache_dir"], args.dense_cache)
                self.assertEqual(constructor.call_args.kwargs["max_seq_length"], 128)
                with gzip.open(args.output, "rt", encoding="utf-8") as stream:
                    rows = [json.loads(line) for line in stream]
                self.assertEqual(rows, [{"group_id": json_digest(key), "query_key": key,
                    "channels": {"dense_compact_current": [item_id], "dense_compact_plain": [item_id]}}])
                self.assertEqual(result["producer_details"]["document_format"], "compact")
                self.assertFalse(result["producer_details"]["tuning_labels_read"])
                self.assertEqual(result["producer_details"]["separate_completed_index"]["fingerprint"], "new-index")
                with self.assertRaises(FileExistsError):
                    produce(args)
                self.assertEqual(constructor.call_count, 1)
            self.assertEqual(set(files_read), {"fit-texts.json", "benchmark_items"})

    def test_cached_ref_mutation_rejected_against_recorded_training_base(self):
        commit = "d" * 40
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "adapter_config.json").write_text(json.dumps({
                "base_model_name_or_path": "Octen/Octen-Embedding-0.6B", "revision": None}))
            (model / "adapter_model.safetensors").write_bytes(b"fixture-adapter")
            cache = root / "hub"
            repo = cache / "models--Octen--Octen-Embedding-0.6B"
            (repo / "refs").mkdir(parents=True)
            (repo / "snapshots" / commit).mkdir(parents=True)
            ref = repo / "refs/main"
            ref.write_text(commit)
            fingerprint = local_checkpoint_content_fingerprint(model)
            training = root / "train.json"
            training.write_text(json.dumps({"embedding_base_repo": "Octen/Octen-Embedding-0.6B",
                "embedding_base_revision": commit, "embedding_checkpoint_sha256": fingerprint}))
            reference = root / "frozen-base.json"
            manifest = {"encoder": {"content_sha256": fingerprint}}
            with patch.dict(os.environ, {"HF_HUB_CACHE": str(cache)}):
                declared = write_encoder_base_reference(model, training, reference)
                self.assertEqual(declared["base_revision"], commit)
                self.assertEqual(check_encoder_base(model, manifest, reference)["base_revision"], commit)
                ref.write_text("a" * 40)
                with self.assertRaisesRegex(ValueError, "Cached adapter base changed"):
                    check_encoder_base(model, manifest, reference)
                with self.assertRaises(FileExistsError):
                    ref.write_text(commit)
                    write_encoder_base_reference(model, training, reference)

    def test_offline_is_explicitly_required(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            require_offline_environment()
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "0"}):
            with self.assertRaisesRegex(ValueError, "HF_HUB_OFFLINE=1"):
                require_offline_environment()

    def test_streaming_conversion_matches_cli_even_when_null_is_in_another_batch(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from avito_candidates.cli import read
        schema = pa.schema([("item_id", pa.string()), ("item_title_raw", pa.string()),
            ("item_category_id", pa.int64()), ("item_infm_params_text", pa.string()),
            ("item_description_raw", pa.string()), ("item_location_id", pa.int64()),
            ("item_price", pa.decimal128(12, 2)), ("item_rating", pa.float64()),
            ("item_rating_reviews_count", pa.int64()), ("item_is_phone_hidden", pa.bool_()),
            ("item_is_message_forbidden", pa.bool_()), ("item_microcat_id", pa.int64())])
        rows = [{"item_id": "b", "item_title_raw": "Ремонт ёлки", "item_category_id": 114,
                 "item_infm_params_text": "Вид услуги Ремонт", "item_description_raw": None,
                 "item_location_id": 1, "item_price": Decimal("1.00"), "item_rating": 4.5,
                 "item_rating_reviews_count": 1, "item_is_phone_hidden": True,
                 "item_is_message_forbidden": False, "item_microcat_id": 3},
                {"item_id": "a", "item_title_raw": None, "item_category_id": None,
                 "item_infm_params_text": None, "item_description_raw": "описание",
                 "item_location_id": 2, "item_price": None, "item_rating": float("nan"),
                 "item_rating_reviews_count": None, "item_is_phone_hidden": None,
                 "item_is_message_forbidden": True, "item_microcat_id": 4}]
        with tempfile.TemporaryDirectory() as directory:
            for statistics in (True, False):
                path = Path(directory) / (str(statistics) + ".parquet")
                pq.write_table(pa.Table.from_pylist(rows, schema=schema), path,
                               row_group_size=1, write_statistics=statistics)
                expected = read(path, schema.names)
                streamed = list(iter_catalog_records(path, batch_size=1))
                self.assertEqual(streamed, expected)
                self.assertEqual(streamed[0]["item_category_id"], "114.0")
                documents, fingerprint = load_catalog_documents(path, batch_size=1)
                self.assertEqual(fingerprint, corpus_fingerprint(expected, feature_version=4))
                self.assertEqual(documents, [{name: row[name] for name in DOCUMENT_FIELDS} for row in expected])

    def test_canonical_inventory_key_resolves_real_protocol_helper(self):
        import evaluate_fresh_validation as EVAL
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "items.parquet"
            path.write_bytes(b"sealed-inventory")
            manifest = {"files": {"benchmark_items": {"path": "items.parquet", "sha256": sha256(path)}}}
            self.assertEqual(EVAL.protocol_file(root, manifest, "benchmark_items"), path)
            with self.assertRaises(ValueError):
                EVAL.protocol_file(root, manifest, "benchmark_items.parquet")


if __name__ == "__main__":
    unittest.main()
