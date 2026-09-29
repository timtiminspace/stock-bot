"""Checks for metrics behavior with no realized trades."""

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "marketsignallab" / "src"
sys.path.insert(0, str(SRC_DIR))

from backtester import TRADE_COLUMNS
from metrics import calculate_metric_tables, calculate_metrics


class MetricsTests(unittest.TestCase):
    def test_metrics_support_an_all_cash_backtest(self) -> None:
        dates = pd.date_range("2025-01-01", periods=3)
        equity = pd.DataFrame(
            {
                "date": dates,
                "cash": [10_000.0] * 3,
                "positions_value": [0.0] * 3,
                "total_portfolio_value": [10_000.0] * 3,
                "number_of_positions": [0] * 3,
                "drawdown": [0.0] * 3,
                "universe_snapshot_date": [pd.Timestamp("2025-01-01")] * 3,
            }
        )
        prices = pd.DataFrame(
            {"date": dates, "ticker": "QQQ", "close": [100.0, 101.0, 102.0]}
        )
        trades = pd.DataFrame(columns=TRADE_COLUMNS)

        metrics = calculate_metrics(equity, trades, prices, starting_cash=10_000.0)

        self.assertEqual(metrics["total_return"], 0.0)
        self.assertEqual(metrics["number_of_trades"], 0.0)
        self.assertAlmostEqual(metrics["benchmark_return_QQQ"], 0.02)
        self.assertEqual(metrics["cash_drag"], 1.0)
        self.assertEqual(metrics["average_positions_held"], 0.0)
        self.assertEqual(metrics["max_positions_held"], 0.0)
        self.assertEqual(metrics["days_at_max_positions"], 0.0)
        self.assertEqual(metrics["number_of_universe_snapshots_used"], 1.0)

        tables = calculate_metric_tables(equity, trades, prices, starting_cash=10_000.0)
        self.assertTrue(tables["returns_by_ticker"].empty)
        self.assertTrue(tables["drawdown_attribution_by_ticker"].empty)

    def test_drawdown_period_and_ticker_attribution(self) -> None:
        dates = pd.date_range("2025-01-01", periods=3)
        equity = pd.DataFrame(
            {
                "date": dates,
                "cash": [9_000.0] * 3,
                "positions_value": [1_000.0, 1_200.0, 600.0],
                "total_portfolio_value": [10_000.0, 10_200.0, 9_600.0],
                "number_of_positions": [1] * 3,
                "drawdown": [0.0, 0.0, 9_600 / 10_200 - 1],
            }
        )
        prices = pd.concat(
            [
                pd.DataFrame(
                    {
                        "date": dates,
                        "ticker": "AAA",
                        "close": [100.0, 120.0, 60.0],
                    }
                ),
                pd.DataFrame(
                    {
                        "date": dates,
                        "ticker": "QQQ",
                        "close": [100.0, 101.0, 102.0],
                    }
                ),
            ],
            ignore_index=True,
        )
        trade = {column: None for column in TRADE_COLUMNS}
        trade.update(
            {
                "signal_date": dates[0],
                "execution_date": dates[0],
                "ticker": "AAA",
                "action": "BUY",
                "reason": "ENTRY_SIGNAL",
                "execution_price": 100.0,
                "shares": 10.0,
                "realized_pnl": 0.0,
                "realized_pnl_pct": 0.0,
                "holding_period_days": 0,
            }
        )
        trades = pd.DataFrame([trade], columns=TRADE_COLUMNS)

        metrics = calculate_metrics(equity, trades, prices, starting_cash=10_000.0)
        tables = calculate_metric_tables(equity, trades, prices, starting_cash=10_000.0)

        self.assertEqual(metrics["max_drawdown_period_start"], "2025-01-02")
        self.assertEqual(metrics["max_drawdown_period_end"], "2025-01-03")
        attribution = tables["drawdown_attribution_by_ticker"].iloc[0]
        self.assertEqual(attribution["ticker"], "AAA")
        self.assertAlmostEqual(attribution["drawdown_period_pnl"], -600.0)
        self.assertAlmostEqual(
            attribution["contribution_to_portfolio_drawdown"], -600 / 10_200
        )

    def test_loss_recovery_metrics_count_stops_and_later_recoveries(self) -> None:
        dates = pd.date_range("2025-01-01", periods=2)
        equity = pd.DataFrame(
            {
                "date": dates,
                "cash": [10_000.0, 10_000.0],
                "positions_value": [0.0, 0.0],
                "total_portfolio_value": [10_000.0, 10_000.0],
                "number_of_positions": [0, 0],
                "drawdown": [0.0, 0.0],
            }
        )
        prices = pd.DataFrame({"date": dates, "ticker": "QQQ", "close": [100.0, 101.0]})
        episodes = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "AAA"],
                "resolution": [
                    "LOSS_RECOVERY_EXIT",
                    "CATASTROPHIC_STOP_30",
                    "OPEN_AT_END",
                ],
                "reached_breakeven_after_exit": [True, True, False],
                "reached_positive_after_exit": [True, False, False],
            }
        )
        metrics = calculate_metrics(
            equity,
            pd.DataFrame(columns=TRADE_COLUMNS),
            prices,
            starting_cash=10_000.0,
            loss_recovery_log=episodes,
        )

        self.assertEqual(metrics["number_of_loss_recovery_triggers"], 3.0)
        self.assertEqual(metrics["number_of_loss_recovery_exits"], 1.0)
        self.assertEqual(metrics["number_of_catastrophic_stops_30"], 1.0)
        self.assertEqual(metrics["unique_tickers_hitting_minus_30"], 1.0)
        self.assertEqual(metrics["number_of_open_loss_recoveries"], 1.0)
        self.assertEqual(metrics["loss_recovery_exit_rate"], 0.5)
        self.assertEqual(metrics["catastrophic_stops_later_reaching_breakeven"], 1.0)
        self.assertEqual(metrics["catastrophic_stops_later_reaching_positive"], 0.0)


if __name__ == "__main__":
    unittest.main()
