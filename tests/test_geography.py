"""Связи локаций не должны учитывать ответы валидации и выдуманные карты."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from avito_candidates.core import held_out, normalize
from avito_candidates.geography import fit_location_associations


class GeographyTest(unittest.TestCase):
    def test_query_text_holdout_is_excluded_from_associations(self):
        fit_text = next(f"fit {i}" for i in range(1000)
                        if not held_out(normalize(f"fit {i}"), 0.2, 42))
        valid_text = next(f"valid {i}" for i in range(1000)
                          if held_out(normalize(f"valid {i}"), 0.2, 42))
        rows = [
            {"search_query": fit_text, "search_location_id": "region",
             "item_location_id": "fit_city", "item_id": "a"},
            {"search_query": valid_text, "search_location_id": "region",
             "item_location_id": "heldout_city", "item_id": "b"},
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.parquet"
            pd.DataFrame(rows).to_parquet(path)
            associations = fit_location_associations(
                path, {"fit_city", "heldout_city"}, split="queries",
                fraction=0.2, seed=42, batch_size=1)
        self.assertEqual(associations.destinations("region", min_history=1), ("fit_city",))
        self.assertEqual(associations.destinations("region", min_history=2), ())


if __name__ == "__main__":
    unittest.main()
