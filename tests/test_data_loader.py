"""Checks for the enforced V1 price-history boundary."""

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "marketsignallab" / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import (
    DEFAULT_START_DATE,
    clean_yfinance_data,
    download_price_data,
    filter_price_dates,
)


class DataLoaderTests(unittest.TestCase):
    def test_default_window_excludes_prices_before_2023(self) -> None:
        prices = pd.DataFrame(
            {
                "date": ["2023-01-03", "2022-12-30", "2023-01-04"],
                "ticker": ["BBB", "AAA", "AAA"],
                "close": [20.0, 10.0, 11.0],
            }
        )

        result = filter_price_dates(prices)

        self.assertEqual(DEFAULT_START_DATE, "2023-01-03")
        self.assertEqual(result["date"].min(), pd.Timestamp("2023-01-03"))
        self.assertEqual(result["ticker"].tolist(), ["AAA", "BBB"])

    def test_end_date_is_exclusive(self) -> None:
        prices = pd.DataFrame(
            {
                "date": ["2023-01-03", "2023-01-04"],
                "ticker": ["AAA", "AAA"],
            }
        )

        result = filter_price_dates(prices, end="2023-01-04")

        self.assertEqual(result["date"].tolist(), [pd.Timestamp("2023-01-03")])

    def test_calendar_year_start_resolves_to_first_trading_day(self) -> None:
        prices = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "ticker": ["AAA", "AAA", "AAA"],
            }
        )

        result = filter_price_dates(prices, start="2023-01-01")

        self.assertEqual(result["date"].tolist(), [pd.Timestamp("2023-01-03")])

    def test_start_date_before_january_1_is_rejected(self) -> None:
        prices = pd.DataFrame({"date": ["2023-01-03"], "ticker": ["AAA"]})

        with self.assertRaisesRegex(ValueError, "cannot start before 2023-01-01"):
            filter_price_dates(prices, start="2022-01-01")

    def test_clean_yfinance_data_normalizes_multi_ticker_columns(self) -> None:
        dates = pd.DatetimeIndex(["2023-01-03", "2023-01-04"], name="Date")
        columns = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["BBB", "AAA"]],
            names=["Price", "Ticker"],
        )
        raw = pd.DataFrame(
            [
                [20.0, 10.0, 21.0, 11.0, 19.0, 9.0, 20.5, 10.5, 2000, 1000],
                [21.0, 11.0, 22.0, 12.0, 20.0, 10.0, 21.5, 11.5, 2100, 1100],
            ],
            index=dates,
            columns=columns,
        )

        clean = clean_yfinance_data(raw)

        self.assertEqual(
            clean.columns.tolist(),
            ["date", "ticker", "open", "high", "low", "close", "volume"],
        )
        self.assertEqual(clean["ticker"].tolist(), ["AAA", "AAA", "BBB", "BBB"])
        self.assertEqual(clean["date"].dtype, "datetime64[ns]")

    def test_download_requires_at_least_one_ticker(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one ticker"):
            download_price_data([])


if __name__ == "__main__":
    unittest.main()
