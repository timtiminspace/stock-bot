"""Deterministic checks for V1 feature definitions."""

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "marketsignallab" / "src"
sys.path.insert(0, str(SRC_DIR))

from features import add_features


class FeatureTests(unittest.TestCase):
    def test_features_are_grouped_sorted_and_use_previous_volume(self) -> None:
        dates = pd.date_range("2024-01-01", periods=205, freq="D")
        frames = []
        for ticker, starting_close in (("AAA", 100.0), ("BBB", 200.0)):
            frames.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "ticker": ticker,
                        "open": [starting_close + index for index in range(205)],
                        "high": [starting_close + index + 1 for index in range(205)],
                        "low": [starting_close + index - 1 for index in range(205)],
                        "close": [starting_close + index for index in range(205)],
                        "volume": list(range(1, 206)),
                    }
                )
            )
        shuffled = pd.concat(frames).sample(frac=1, random_state=7)

        result = add_features(shuffled)
        aaa = result.loc[result["ticker"].eq("AAA")].reset_index(drop=True)

        self.assertEqual(result.iloc[0]["ticker"], "AAA")
        self.assertAlmostEqual(aaa.loc[1, "daily_return"], 101 / 100 - 1)
        self.assertAlmostEqual(aaa.loc[3, "return_3d"], 103 / 100 - 1)
        self.assertAlmostEqual(aaa.loc[4, "return_4d"], 104 / 100 - 1)
        self.assertAlmostEqual(aaa.loc[5, "weekly_return_5d"], 105 / 100 - 1)
        self.assertAlmostEqual(aaa.loc[30, "momentum_30d"], 130 / 100 - 1)
        self.assertAlmostEqual(aaa.loc[90, "momentum_90d"], 190 / 100 - 1)
        self.assertAlmostEqual(aaa.loc[30, "avg_volume_30d"], 15.5)
        self.assertAlmostEqual(aaa.loc[30, "relative_volume_30d"], 31 / 15.5)
        self.assertAlmostEqual(aaa.loc[49, "ma_50d"], 124.5)
        self.assertAlmostEqual(aaa.loc[199, "ma_200d"], 199.5)
        self.assertTrue(bool(aaa.loc[49, "above_ma_50d"]))
        self.assertAlmostEqual(aaa.loc[204, "down_from_start"], 304 / 100 - 1)


if __name__ == "__main__":
    unittest.main()
