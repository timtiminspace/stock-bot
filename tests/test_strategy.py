"""Checks for market-level strategy rules."""

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "marketsignallab" / "src"
sys.path.insert(0, str(SRC_DIR))

from strategy import add_strategy_signals
from universe import Universe


class StrategyTests(unittest.TestCase):
    def test_only_runtime_approved_tickers_receive_entry_signals(self) -> None:
        date = pd.Timestamp("2025-01-10")
        base = {
            "date": date,
            "close": 120.0,
            "daily_return": 0.02,
            "return_3d": 0.05,
            "return_4d": 0.06,
            "weekly_return_5d": 0.07,
            "momentum_30d": 0.15,
            "momentum_90d": 0.25,
            "volatility_30d": 0.03,
            "avg_volume_30d": 2_000_000.0,
            "relative_volume_30d": 1.5,
            "ma_50d": 100.0,
        }
        features = pd.DataFrame(
            [
                dict(base, ticker="AAA"),
                dict(base, ticker="BBB", relative_volume_30d=10.0),
                dict(base, ticker="QQQ"),
            ]
        )
        universe = Universe(
            approved_universe=frozenset({"BBB"}),
            rejected_universe=frozenset(),
            benchmark_universe=frozenset({"QQQ"}),
            classifications={"BBB": "compounder"},
        )

        signals = add_strategy_signals(features, universe)

        aaa_signal = signals.loc[signals["ticker"].eq("AAA")].iloc[0]
        bbb_signal = signals.loc[signals["ticker"].eq("BBB")].iloc[0]
        qqq_signal = signals.loc[signals["ticker"].eq("QQQ")].iloc[0]
        self.assertFalse(bool(aaa_signal["entry_signal"]))
        self.assertTrue(bool(bbb_signal["entry_signal"]))
        self.assertFalse(bool(qqq_signal["entry_signal"]))


if __name__ == "__main__":
    unittest.main()
