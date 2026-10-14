"""Exercise the opt-in preparation policy with a tiny local dataset."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import pandas as pd
import pyarrow.parquet as pq

from avito_candidates import cli
from avito_candidates.core import SEARCH_FIELDS


class PreparationIntegrityTest(unittest.TestCase):
    def test_cli_crossfit_policy_reaches_candidates_features_and_artifact_metadata(self):
        items = []
        for item_id in ("own", "other"):
            items.append(dict.fromkeys((
                "item_title_raw", "item_infm_params_text", "item_description_raw",
                "item_category_id", "item_microcat_id", "item_price", "item_rating",
                "item_rating_reviews_count", "item_is_phone_hidden", "item_is_message_forbidden",
            ), ""))
            items[-1].update(item_id=item_id, item_location_id=item_id,
                             item_title_raw="ремонт", item_microcat_id="service")
        rows = []
        for text, item_id, repetitions in (("ремонт", "own", 3), ("иной ремонт", "other", 2)):
            for _ in range(repetitions):
                row = dict.fromkeys(SEARCH_FIELDS, "")
                row.update(search_query=text, search_location_id="region", item_id=item_id,
                           item_location_id=item_id, item_microcat_id="service")
                rows.append(row)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(rows).to_parquet(root / "train.parquet")
            pd.DataFrame(items).to_parquet(root / "benchmark_items.parquet")
            output = root / "prepared.parquet"
            arguments = ["avito", "prepare-reranker", "--data-dir", str(root),
                         "--split", "all", "--method", "bm25", "--params-weight", "0",
                         "--description-weight", "0", "--geo-candidate-k", "1",
                         "--geo-top-locations", "1", "--geo-min-history", "2",
                         "--reranker-crossfit-geography", "--reranker-data", str(output)]
            with patch("sys.argv", arguments), redirect_stdout(StringIO()):
                cli.main()
            metadata = json.loads(output.with_suffix(".parquet.json").read_text())
            self.assertEqual(metadata["training_prior_policy"]["geography"], "leave-query-text-out")
            records = pq.read_table(output).to_pylist()
            own_positive = next(row for row in records
                                if row["query_text"] == "ремонт" and row["label"] == 1)
            self.assertEqual(own_positive["geo_click_probability"], 0.0)
            self.assertEqual(own_positive["present_title_geo"], 0.0)
            other_negative = next(row for row in records
                                  if row["query_text"] == "ремонт" and row["label"] == 0)
            self.assertEqual(other_negative["geo_click_probability"], 1.0)
            self.assertEqual(other_negative["present_title_geo"], 1.0)
            saved = joblib.load(output.with_suffix(".parquet.priors.joblib"))
            self.assertEqual(saved["geo_associations"].history("region")[1], 5)


if __name__ == "__main__":
    unittest.main()
