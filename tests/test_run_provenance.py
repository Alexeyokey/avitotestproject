"""Run provenance must identify bytes independently of file timestamps."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from avito_candidates.dense import local_checkpoint_content_fingerprint


class CheckpointProvenanceTest(unittest.TestCase):
    def test_content_hash_detects_replaced_weights_with_same_size_and_timestamp(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            weights = checkpoint / "adapter_model.safetensors"
            weights.write_bytes(b"abc")
            old_time = weights.stat().st_mtime_ns
            first = local_checkpoint_content_fingerprint(checkpoint)
            os.utime(weights, ns=(old_time + 1_000_000, old_time + 1_000_000))
            self.assertEqual(first, local_checkpoint_content_fingerprint(checkpoint))
            weights.write_bytes(b"xyz")
            os.utime(weights, ns=(old_time, old_time))
            self.assertNotEqual(first, local_checkpoint_content_fingerprint(checkpoint))
            self.assertEqual(len(first), 64)
        self.assertIsNone(local_checkpoint_content_fingerprint("Octen/Octen-Embedding-0.6B"))


if __name__ == "__main__":
    unittest.main()
