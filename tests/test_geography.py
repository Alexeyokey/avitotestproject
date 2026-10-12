"""Связи локаций не должны учитывать ответы валидации и выдуманные карты."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from avito_candidates.core import held_out, normalize
from avito_candidates.geography import LocationAssociations, fit_location_associations
from collections import Counter


class GeographyTest(unittest.TestCase):
    def test_tracked_history_excludes_all_variants_of_query_text(self):
        rows = [
            {"search_query": " РЕМОНТ ЁЖА ", "search_location_id": "region",
             "item_location_id": "own_city"},
            {"search_query": "ремонт ежа", "search_location_id": "region",
             "item_location_id": "own_city"},
            {"search_query": "Ремонт  ежа", "search_location_id": "second_region",
             "item_location_id": "own_city"},
            {"search_query": "другая услуга", "search_location_id": "region",
             "item_location_id": "other_city"},
            {"search_query": "другая услуга", "search_location_id": "second_region",
             "item_location_id": "other_city"},
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.parquet"
            pd.DataFrame(rows).to_parquet(path)
            associations = fit_location_associations(
                path, {"own_city", "other_city"}, track_query_texts=True,
                batch_size=2)
        for location in ("region", "second_region"):
            counts, total = associations.history(
                location, exclude_query_text="ремонт ёжа")
            self.assertEqual(counts, Counter({"other_city": 1}))
            self.assertEqual(total, 1)
            self.assertEqual(associations.destinations(
                location, min_history=1, max_locations=1,
                exclude_query_text="РЕМОНТ ЕЖА"), ("other_city",))
        # Removal is a view: saved full-fit priors remain available at inference.
        self.assertEqual(associations.history("region")[1], 3)
        self.assertEqual(associations.history("second_region")[1], 2)

    def test_exclusion_changes_history_threshold_but_preserves_exact_city(self):
        associations = LocationAssociations(
            {"city", "neighbor"},
            counts={"region": Counter({"city": 19, "neighbor": 1}),
                    "city": Counter({"neighbor": 20})},
            totals=Counter({"region": 20, "city": 20}),
            query_counts={"ремонт": {"region": Counter({"city": 19}),
                                      "city": Counter({"neighbor": 20})}})
        self.assertEqual(associations.destinations("region", max_locations=1), ("city",))
        self.assertEqual(associations.destinations(
            "region", exclude_query_text="ремонт"), ())
        self.assertEqual(associations.destinations(
            "city", include_related=True, exclude_query_text="ремонт"), ("city",))
        # Priors serialized by older versions still support ordinary inference.
        del associations.query_counts
        self.assertEqual(associations.destinations("region", max_locations=1), ("city",))
        with self.assertRaisesRegex(ValueError, "tracked history"):
            associations.history("region", exclude_query_text="ремонт")

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
