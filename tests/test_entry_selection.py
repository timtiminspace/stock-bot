"""Candidate-pool selection policy tests for rolling-universe backtests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "marketsignallab" / "src"
sys.path.insert(0, str(SRC_DIR))

from backtester import run_backtest
from config import BacktestConfig
from rolling_universe import RollingUniverse
from run_random_backtests import (
    RUN_COLUMNS,
    build_catastrophic_stop_summary,
    build_seed_rankings,
    build_seed_stock_performance,
    build_seed_trade_ledger,
    summarize_random_runs,
)
from universe import Universe


def _universe(tickers: set[str]) -> Universe:
    return Universe(
        approved_universe=frozenset(tickers),
        rejected_universe=frozenset({"BAD"}),
        benchmark_universe=frozenset({"QQQ"}),
        classifications={ticker: "compounder" for ticker in tickers},
    )


def _signals(tickers: list[str], entry_tickers: set[str] | None = None) -> pd.DataFrame:
    enabled = set(tickers) if entry_tickers is None else entry_tickers
    rows: list[dict[str, object]] = []
    for date in ("2025-01-02", "2025-01-03"):
        for index, ticker in enumerate(tickers):
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "ticker": ticker,
                    "open": 100.0 + index,
                    "close": 100.0 + index,
                    "ma_50d": 90.0,
                    "momentum_30d": 0.10,
                    "relative_volume_30d": 1.2,
                    "entry_signal": date == "2025-01-02" and ticker in enabled,
                    "spike_signal": False,
                    "weekly_underperformance_signal": False,
                }
            )
    return pd.DataFrame(rows)


def _run(seed: int, max_positions: int = 2):
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    rolling = RollingUniverse(
        {
            pd.Timestamp("2025-01-01"): pd.DataFrame(
                {"ticker": tickers, "eligible": [True] * len(tickers)}
            )
        }
    )
    return run_backtest(
        _signals(tickers),
        _universe(set(tickers)),
        backtest_config=BacktestConfig(
            max_positions=max_positions,
            random_seed=seed,
        ),
        rolling_universe=rolling,
    )


class EntrySelectionTests(unittest.TestCase):
    def test_random_mode_is_reproducible_for_the_same_seed(self) -> None:
        first = _run(42)
        second = _run(42)
        self.assertEqual(
            first.trades.loc[first.trades["action"].eq("BUY"), "ticker"].tolist(),
            second.trades.loc[second.trades["action"].eq("BUY"), "ticker"].tolist(),
        )
        pd.testing.assert_frame_equal(
            first.entry_candidate_log,
            second.entry_candidate_log,
        )

    def test_different_seeds_can_choose_different_tickers(self) -> None:
        selected_sets = {
            tuple(
                _run(seed, max_positions=1)
                .entry_candidate_log.loc[lambda data: data["selected"], "ticker"]
                .tolist()
            )
            for seed in range(10)
        }
        self.assertGreater(len(selected_sets), 1)

    def test_random_selection_only_uses_valid_entry_candidates(self) -> None:
        approved = {"AAA", "BBB", "CCC"}
        all_tickers = ["AAA", "BBB", "CCC", "BAD", "QQQ"]
        rolling = RollingUniverse(
            {
                pd.Timestamp("2025-01-01"): pd.DataFrame(
                    {
                        "ticker": ["AAA", "BBB", "BAD", "QQQ"],
                        "eligible": [True, True, True, True],
                    }
                )
            }
        )
        result = run_backtest(
            _signals(all_tickers, {"AAA", "CCC", "BAD", "QQQ"}),
            _universe(approved),
            backtest_config=BacktestConfig(
                max_positions=5,
                random_seed=7,
            ),
            rolling_universe=rolling,
        )
        self.assertEqual(result.entry_candidate_log["ticker"].tolist(), ["AAA"])
        self.assertEqual(result.trades["ticker"].tolist(), ["AAA"])

    def test_selected_positions_never_exceed_max_positions(self) -> None:
        result = _run(4, max_positions=2)
        self.assertLessEqual(int(result.equity_curve["number_of_positions"].max()), 2)
        self.assertLessEqual(
            int(result.entry_candidate_log["selected"].sum()),
            2,
        )

    def test_sidelined_half_position_does_not_consume_an_active_slot(self) -> None:
        rows: list[dict[str, object]] = []
        closes = {
            "2025-01-01": {"AAA": 100.0, "BBB": 50.0},
            "2025-01-02": {"AAA": 69.0, "BBB": 50.0},
            "2025-01-03": {"AAA": 70.0, "BBB": 51.0},
        }
        for date, ticker_closes in closes.items():
            for ticker, close in ticker_closes.items():
                rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "ticker": ticker,
                        "open": 100.0 if ticker == "AAA" else 50.0,
                        "close": close,
                        "ma_50d": 40.0,
                        "momentum_30d": 0.10,
                        "relative_volume_30d": 1.2,
                        "entry_signal": (
                            (date == "2025-01-01" and ticker == "AAA")
                            or (date == "2025-01-02" and ticker == "BBB")
                        ),
                        "spike_signal": False,
                        "weekly_underperformance_signal": False,
                    }
                )
        rolling = RollingUniverse(
            {
                pd.Timestamp("2025-01-01"): pd.DataFrame(
                    {"ticker": ["AAA", "BBB"], "eligible": [True, True]}
                )
            }
        )
        result = run_backtest(
            pd.DataFrame(rows),
            _universe({"AAA", "BBB"}),
            backtest_config=BacktestConfig(
                max_positions=1,
                target_position_fraction=0.50,
                random_seed=4,
            ),
            rolling_universe=rolling,
        )

        self.assertEqual(result.final_states["AAA"], "HELD_SIDELINED")
        self.assertEqual(result.final_states["BBB"], "HELD_FULL")
        final_equity = result.equity_curve.iloc[-1]
        self.assertEqual(int(final_equity["number_of_positions"]), 1)
        self.assertEqual(int(final_equity["number_of_sidelined_positions"]), 1)
        self.assertEqual(int(final_equity["total_number_of_positions"]), 2)

    def test_monte_carlo_summary_reports_distribution_and_seed_extremes(self) -> None:
        runs = pd.DataFrame(
            {
                "seed": [0, 1, 2],
                "total_return": [-0.10, 0.20, 0.05],
                "max_drawdown": [-0.30, -0.10, -0.20],
                "strategy_vs_QQQ": [-0.20, 0.10, 0.01],
            }
        )
        summary = summarize_random_runs(runs).iloc[0]
        self.assertAlmostEqual(float(summary["median_total_return"]), 0.05)
        self.assertAlmostEqual(float(summary["median_max_drawdown"]), -0.20)
        self.assertAlmostEqual(float(summary["pct_runs_beating_QQQ"]), 200 / 3)
        self.assertEqual(int(summary["best_seed"]), 1)
        self.assertEqual(int(summary["worst_seed"]), 0)

    def test_seed_rankings_include_best_and_worst_stock_dates_and_returns(self) -> None:
        trades = pd.DataFrame(
            [
                {
                    "signal_date": "2025-01-01",
                    "execution_date": "2025-01-02",
                    "ticker": "AAA",
                    "action": "BUY",
                    "reason": "ENTRY_SIGNAL",
                    "execution_price": 100.0,
                    "shares": 10.0,
                    "entry_price": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_pct": 0.0,
                    "holding_period_days": 0,
                    "classification": "compounder",
                },
                {
                    "signal_date": "2025-01-03",
                    "execution_date": "2025-01-04",
                    "ticker": "AAA",
                    "action": "SELL_PARTIAL",
                    "reason": "SPIKE_PARTIAL_COMPOUNDER",
                    "execution_price": 120.0,
                    "shares": 5.0,
                    "entry_price": 100.0,
                    "realized_pnl": 100.0,
                    "realized_pnl_pct": 0.20,
                    "holding_period_days": 2,
                    "classification": "compounder",
                },
                {
                    "signal_date": "2025-01-05",
                    "execution_date": "2025-01-06",
                    "ticker": "AAA",
                    "action": "SELL",
                    "reason": "RUNNER_STOP",
                    "execution_price": 110.0,
                    "shares": 5.0,
                    "entry_price": 100.0,
                    "realized_pnl": 50.0,
                    "realized_pnl_pct": 0.10,
                    "holding_period_days": 4,
                    "classification": "compounder",
                },
                {
                    "signal_date": "2025-02-01",
                    "execution_date": "2025-02-02",
                    "ticker": "BBB",
                    "action": "BUY",
                    "reason": "ENTRY_SIGNAL",
                    "execution_price": 50.0,
                    "shares": 10.0,
                    "entry_price": 50.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_pct": 0.0,
                    "holding_period_days": 0,
                    "classification": "compounder",
                },
                {
                    "signal_date": "2025-02-03",
                    "execution_date": "2025-02-04",
                    "ticker": "BBB",
                    "action": "SELL",
                    "reason": "HARD_STOP",
                    "execution_price": 45.0,
                    "shares": 10.0,
                    "entry_price": 50.0,
                    "realized_pnl": -50.0,
                    "realized_pnl_pct": -0.10,
                    "holding_period_days": 2,
                    "classification": "compounder",
                },
            ]
        )
        ledger = build_seed_trade_ledger(trades, seed=7)
        stocks = build_seed_stock_performance(ledger)
        runs = pd.DataFrame(
            [
                {
                    "seed": 7,
                    "total_return": 0.20,
                    "annualized_return": 0.20,
                    "max_drawdown": -0.10,
                    "sharpe_ratio": 1.0,
                    "benchmark_return_QQQ": 0.10,
                    "strategy_vs_QQQ": 0.10,
                    "number_of_trades": 5.0,
                    "cash_drag": 0.10,
                },
                {
                    "seed": 8,
                    "total_return": 0.30,
                    "annualized_return": 0.30,
                    "max_drawdown": -0.08,
                    "sharpe_ratio": 1.2,
                    "benchmark_return_QQQ": 0.10,
                    "strategy_vs_QQQ": 0.20,
                    "number_of_trades": 0.0,
                    "cash_drag": 1.0,
                },
            ],
            columns=RUN_COLUMNS,
        )
        rankings = build_seed_rankings(runs, stocks)

        aaa = stocks.loc[stocks["ticker"].eq("AAA")].iloc[0]
        self.assertAlmostEqual(float(aaa["realized_return"]), 0.15)
        self.assertEqual(int(aaa["completed_exit_count"]), 2)
        self.assertEqual(aaa["first_entry_date"], pd.Timestamp("2025-01-02"))
        self.assertEqual(aaa["last_exit_date"], pd.Timestamp("2025-01-06"))
        seed_seven = rankings.loc[rankings["seed"].eq(7)].iloc[0]
        self.assertEqual(int(seed_seven["seed_rank"]), 2)
        self.assertEqual(seed_seven["best_stock_ticker"], "AAA")
        self.assertEqual(seed_seven["worst_stock_ticker"], "BBB")
        self.assertAlmostEqual(float(seed_seven["best_stock_return"]), 0.15)
        self.assertAlmostEqual(float(seed_seven["worst_stock_return"]), -0.10)
        self.assertTrue(
            pd.isna(rankings.loc[rankings["seed"].eq(8), "best_stock_ticker"].iloc[0])
        )

    def test_catastrophic_summary_counts_unique_seeds_and_recoveries(self) -> None:
        episodes = pd.DataFrame(
            {
                "seed": [1, 2, 2],
                "ticker": ["AAA", "AAA", "BBB"],
                "resolution": ["CATASTROPHIC_STOP_30"] * 3,
                "trigger_date": pd.to_datetime(
                    ["2025-01-01", "2025-02-01", "2025-03-01"]
                ),
                "resolution_date": pd.to_datetime(
                    ["2025-01-02", "2025-02-02", "2025-03-02"]
                ),
                "trough_return": [-0.31, -0.35, -0.32],
                "execution_return": [-0.30, -0.36, -0.33],
                "reached_breakeven_after_exit": [True, False, True],
                "reached_positive_after_exit": [True, False, False],
                "max_return_after_exit_through_backtest": [0.20, -0.05, 0.0],
            }
        )
        summary = build_catastrophic_stop_summary(episodes).set_index("ticker")

        self.assertEqual(int(summary.loc["AAA", "catastrophic_stop_episodes"]), 2)
        self.assertEqual(int(summary.loc["AAA", "seeds_affected"]), 2)
        self.assertEqual(int(summary.loc["AAA", "later_breakeven_count"]), 1)
        self.assertAlmostEqual(
            float(summary.loc["AAA", "pct_later_reaching_positive"]), 50.0
        )


if __name__ == "__main__":
    unittest.main()
