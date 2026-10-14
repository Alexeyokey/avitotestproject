"""Focused fixtures for label-independent sampling and cold-query separation."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from build_fresh_validation_protocol import (
    REPRESENTATIVE_SALT, build_protocol, canonical_query_key_json,
    load_manifest, salted_digest, select_catalog_ids, sha256_file,
)


FIELDS = ("search_query", "search_location_id", "search_is_delivery_search",
          "search_infm_params_text", "search_category")
ENCODER_SHA = "e" * 64


def normalize(text):
    return " ".join(str(text).lower().replace("ё", "е").split())


def held_out(text, fraction=0.2, seed=42):
    # Deliberately controlled partitions; this fixture does not test core's hash.
    return text.startswith("cold") or text == "historical" if seed == 42 else text == "cold audit"


CONTRACT = {"search_fields": FIELDS, "normalize": normalize, "held_out": held_out,
            "query_key": lambda row: tuple(normalize(row.get(field, "")) for field in FIELDS),
            "encoder_fingerprint": lambda _path: ENCODER_SHA}


class FreshProtocolTests(unittest.TestCase):
    def fixture(self, root, reverse=False):
        data, encoder = root / "original", root / "encoder" / "model"
        data.mkdir()
        encoder.mkdir(parents=True)
        (encoder / "adapter_model.safetensors").write_bytes(b"opaque frozen fixture")
        (encoder.parent / "data_stats.json").write_text(json.dumps({
            "fit_rows": 2, "fit_query_texts": 1, "training_pairs": 1,
            "validation_query_texts_excluded": 3, "seed": 42,
        }), encoding="utf-8")

        def row(text, location, item):
            return {"search_query": text, "search_location_id": location,
                    "search_is_delivery_search": "0", "search_infm_params_text": "",
                    "search_category": "service", "item_id": f"{item:016x}",
                    "item_title_raw": f"train-title-{item}", "item_location_id": location}

        rows = [row("warm", "a", 1), row("warm", "b", 2),
                row("historical", "a", 1), row("historical", "z", 8),
                row("cold tune", "a", 3), row("cold tune", "a", 4),
                row("cold tune", "b", 5), row("cold audit", "c", 6)]
        pq.write_table(pa.Table.from_pylist(list(reversed(rows)) if reverse else rows),
                       data / "train.parquet")
        benchmark = [{"item_id": f"{item:016x}", "item_title_raw": f"benchmark-title-{item}",
                      "item_location_id": "catalog"} for item in (1, 7)]
        pq.write_table(pa.Table.from_pylist(benchmark), data / "benchmark_items.parquet")
        # This original test query must not be copied into the fresh protocol.
        pq.write_table(pa.table({"query_id": ["ORIGINALTEST0001"], "search_query": ["original test"]}),
                       data / "benchmark_queries.parquet")
        return data, encoder

    def build(self, data, encoder, target, **kwargs):
        return build_protocol(data, target, encoder, ENCODER_SHA,
                              corpus_size=kwargs.pop("corpus_size", 4), batch_size=2,
                              contract=CONTRACT, **kwargs)

    def labels(self, target, role):
        return json.loads((target / f"labels-{role}.json").read_text(encoding="utf-8"))["groups"]

    def test_partitions_truth_and_identifiers_are_preserved(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data, encoder = self.fixture(root)
            before = sha256_file(data / "train.parquet")
            target = root / "fresh"
            manifest = self.build(data, encoder, target)
            self.assertEqual(load_manifest(target), manifest)
            self.assertEqual(sha256_file(data / "train.parquet"), before)
            train = pq.read_table(target / "dataset/train.parquet").to_pylist()
            self.assertEqual(len(train), 6)
            self.assertNotIn("historical", {row["search_query"] for row in train})
            self.assertEqual(manifest["stats"]["tune"]["unique_relevant_pairs"], 3)
            for role, expected in (("fit", "warm"), ("tune", "cold tune"), ("audit", "cold audit")):
                labels = self.labels(target, role)
                self.assertTrue(all(row["query_text"] == expected for row in labels))
                self.assertEqual(sum(row["representative"] for row in labels), 1)
                for row in labels:
                    self.assertEqual(row["query_key_json"], canonical_query_key_json(row["query_key"]))
                    self.assertEqual(row["group_id"], hashlib.sha256(row["query_key_json"].encode()).hexdigest())
                    self.assertEqual(row["full_relevant_count"], len(row["relevant_item_ids"]))
                    self.assertTrue(set(row["available_item_ids"]) <= set(row["relevant_item_ids"]))
            all_truth = {item for row in self.labels(target, "tune") + self.labels(target, "audit")
                         for item in row["relevant_item_ids"]}
            self.assertEqual(all_truth, {f"{item:016x}" for item in (3, 4, 5, 6)})
            queries = pq.read_table(target / "dataset/benchmark_queries.parquet").to_pylist()
            self.assertTrue(all(len(row["query_id"]) == 16 for row in queries))
            self.assertNotIn("original test", {row["search_query"] for row in queries})
            self.assertFalse((target / "INCOMPLETE").exists())

    def test_sample_and_representative_do_not_depend_on_availability(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data, encoder = self.fixture(root)
            small, full = root / "small", root / "full"
            self.build(data, encoder, small, corpus_size=1)
            self.build(data, encoder, full, corpus_size=8)
            for role in ("fit", "tune", "audit"):
                first, second = self.labels(small, role), self.labels(full, role)
                self.assertEqual([x["group_id"] for x in first if x["representative"]],
                                 [x["group_id"] for x in second if x["representative"]])
                chosen = min(first, key=lambda x: (salted_digest(REPRESENTATIVE_SALT,
                                                               x["query_key_json"]), x["query_key_json"]))
                self.assertTrue(chosen["representative"])
                self.assertEqual([x["relevant_item_ids"] for x in first],
                                 [x["relevant_item_ids"] for x in second])
            items = {row["item_id"]: row for row in pq.read_table(full / "dataset/benchmark_items.parquet").to_pylist()}
            self.assertEqual(items[f"{1:016x}"]["item_title_raw"], "benchmark-title-1")
            self.assertEqual(items[f"{1:016x}"]["item_location_id"], "catalog")
            self.assertEqual(items[f"{2:016x}"]["item_title_raw"], "train-title-2")
            self.assertEqual(select_catalog_ids(set(items), 4, "salt"),
                             select_catalog_ids(set(reversed(list(items))), 4, "salt"))

    def test_original_row_order_does_not_change_query_assignment_or_catalog_sample(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first", root / "second"
            first.mkdir(); second.mkdir()
            data1, encoder1 = self.fixture(first)
            data2, encoder2 = self.fixture(second, reverse=True)
            m1 = self.build(data1, encoder1, first / "fresh")
            m2 = self.build(data2, encoder2, second / "fresh")
            self.assertEqual(m1["corpus"]["selected_ids_sha256"], m2["corpus"]["selected_ids_sha256"])
            for role in ("fit", "tune", "audit"):
                self.assertEqual(self.labels(first / "fresh", role), self.labels(second / "fresh", role))

    def test_existing_output_and_encoder_provenance_mismatch_refused(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data, encoder = self.fixture(root)
            target = root / "fresh"
            target.mkdir()
            (target / "keep").write_text("keep")
            with self.assertRaisesRegex(ValueError, "already exists"):
                self.build(data, encoder, target)
            self.assertEqual((target / "keep").read_text(), "keep")
            with self.assertRaisesRegex(ValueError, "expected SHA256"):
                build_protocol(data, root / "bad", encoder, "wrong", contract=CONTRACT)
            self.assertFalse((root / "bad").exists())
            stats = json.loads((encoder.parent / "data_stats.json").read_text())
            stats["seed"] = 99
            (encoder.parent / "data_stats.json").write_text(json.dumps(stats))
            with self.assertRaisesRegex(ValueError, "training seed"):
                self.build(data, encoder, root / "bad-seed")
            self.assertFalse((root / "bad-seed").exists())

    def test_loader_checks_hashes_paths_and_completion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data, encoder = self.fixture(root)
            target = root / "fresh"
            original = self.build(data, encoder, target)
            (target / "tune-texts.json").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_manifest(target)
            mutated = deepcopy(original)
            mutated["files"]["train"]["path"] = "../outside.parquet"
            (target / "manifest.json").write_text(json.dumps(mutated), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_manifest(target, verify=False)
            (target / "manifest.json").write_text(json.dumps(original), encoding="utf-8")
            (target / "INCOMPLETE").write_text("interrupted")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_manifest(target, verify=False)

    def test_size_cannot_silently_be_reduced(self):
        with self.assertRaisesRegex(ValueError, "no larger"):
            select_catalog_ids({f"{1:016x}"}, 200000, "salt")
        with self.assertRaisesRegex(ValueError, "five normalized"):
            canonical_query_key_json(["only one field"])


if __name__ == "__main__":
    unittest.main()
