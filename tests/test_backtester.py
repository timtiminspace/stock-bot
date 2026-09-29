"""State, timing, gap, partial-sale, re-entry, and cooldown checks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "marketsignallab" / "src"
sys.path.insert(0, str(SRC_DIR))

from backtester import PositionState, run_backtest
from config import DEFAULT_BACKTEST_CONFIG, BacktestConfig
from rolling_universe import RollingUniverse
from universe import Universe


def make_universe(classification: str) -> Universe:
    return Universe(
        approved_universe=frozenset({"AAA"}),
        rejected_universe=frozenset(),
        benchmark_universe=frozenset({"QQQ"}),
        classifications={"AAA": classification},
    )


def signal_row(
    date: str,
    open_price: float,
    close: float,
    entry: bool = False,
    spike: bool = False,
    weekly_exit: bool = False,
    relative_volume: float = 1.2,
    ma_50d: float = 90.0,
) -> dict:
    return {
        "date": pd.Timestamp(date),
        "ticker": "AAA",
        "open": open_price,
        "close": close,
        "ma_50d": ma_50d,
        "momentum_30d": 0.10,
        "relative_volume_30d": relative_volume,
        "entry_signal": entry,
        "spike_signal": spike,
        "weekly_underperformance_signal": weekly_exit,
    }


class BacktesterTests(unittest.TestCase):
    def run_rows(
        self,
        rows: list[dict],
        classification: str = "recovery_candidate",
        config: BacktestConfig = DEFAULT_BACKTEST_CONFIG,
        rolling_universe: RollingUniverse | None = None,
    ):
        return run_backtest(
            pd.DataFrame(rows),
            make_universe(classification),
            backtest_config=config,
            rolling_universe=rolling_universe,
        )

    def test_entry_uses_next_open_and_records_gap(self) -> None:
        result = self.run_rows(
            [
                signal_row("2025-01-01", 98.0, 100.0, entry=True),
                signal_row("2025-01-02", 90.0, 95.0),
            ]
        )
        buy = result.trades.iloc[0]
        self.assertEqual(buy["signal_date"], pd.Timestamp("2025-01-01"))
        self.assertEqual(buy["execution_date"], pd.Timestamp("2025-01-02"))
        self.assertEqual(buy["execution_price"], 90.0)
        self.assertAlmostEqual(buy["gap_return"], -0.10)
        self.assertAlmostEqual(
            buy["position_value"],
            DEFAULT_BACKTEST_CONFIG.starting_cash
            * DEFAULT_BACKTEST_CONFIG.target_position_fraction,
        )
        self.assertEqual(result.final_states["AAA"], PositionState.HELD_FULL.value)

    def test_recovery_spike_exit_enters_watch_and_can_reenter(self) -> None:
        result = self.run_rows(
            [
                signal_row("2025-01-01", 100.0, 100.0, entry=True),
                signal_row("2025-01-02", 100.0, 120.0, spike=True),
                signal_row("2025-01-03", 125.0, 132.0),
                signal_row("2025-01-04", 130.0, 131.0),
            ]
        )
        self.assertEqual(result.trades["action"].tolist(), ["BUY", "SELL", "BUY"])
        spike_sell = result.trades.iloc[1]
        self.assertEqual(spike_sell["reason"], "SPIKE_EXIT_RECOVERY")
        self.assertEqual(
            spike_sell["state_after"], PositionState.WATCHING_REENTRY.value
        )
        self.assertEqual(result.trades.iloc[2]["reason"], "REENTRY_SIGNAL")
        self.assertEqual(result.final_states["AAA"], PositionState.HELD_FULL.value)
        self.assertEqual(result.watchlist.iloc[0]["status"], "REENTERED")

    def test_compounder_sells_ninety_percent_and_keeps_runner(self) -> None:
        result = self.run_rows(
            [
                signal_row("2025-01-01", 100.0, 100.0, entry=True),
                signal_row("2025-01-02", 100.0, 120.0, spike=True),
                signal_row("2025-01-03", 125.0, 125.0),
            ],
            classification="compounder",
        )
        buy, partial_sell = result.trades.iloc[0], result.trades.iloc[1]
        self.assertEqual(partial_sell["action"], "SELL_PARTIAL")
        self.assertAlmostEqual(partial_sell["shares"], buy["shares"] * 0.90)
        self.assertAlmostEqual(
            partial_sell["position_value"], buy["shares"] * 0.10 * 125.0
        )
        self.assertEqual(result.final_states["AAA"], PositionState.HELD_RUNNER.value)

    def test_runner_weakness_sells_remaining_position_to_cooldown(self) -> None:
        result = self.run_rows(
            [
                signal_row("2025-01-01", 100.0, 100.0, entry=True),
                signal_row("2025-01-02", 100.0, 120.0, spike=True),
                signal_row("2025-01-03", 125.0, 125.0),
                signal_row("2025-01-04", 120.0, 118.0),
                signal_row("2025-01-05", 117.0, 116.0),
            ],
            classification="compounder",
        )
        runner_sell = result.trades.iloc[-1]
        self.assertEqual(runner_sell["reason"], "RUNNER_STOP")
        self.assertEqual(runner_sell["state_before"], PositionState.HELD_RUNNER.value)
        self.assertEqual(runner_sell["state_after"], PositionState.COOLDOWN.value)
        self.assertEqual(result.watchlist.iloc[0]["status"], "CANCELLED")

    def test_unprofitable_spike_does_not_trigger_profit_taking(self) -> None:
        result = self.run_rows(
            [
                signal_row("2025-01-01", 100.0, 100.0, entry=True),
                signal_row("2025-01-02", 100.0, 105.0, spike=True),
                signal_row("2025-01-03", 106.0, 107.0),
            ]
        )
        self.assertEqual(result.trades["action"].tolist(), ["BUY"])
        self.assertEqual(result.final_states["AAA"], PositionState.HELD_FULL.value)

    def test_too_hot_reentry_expires_without_buying(self) -> None:
        config = BacktestConfig(max_watch_days=2)
        result = self.run_rows(
            [
                signal_row("2025-01-01", 100.0, 100.0, entry=True),
                signal_row("2025-01-02", 100.0, 120.0, spike=True),
                signal_row("2025-01-03", 125.0, 150.0),
                signal_row("2025-01-04", 151.0, 152.0),
                signal_row("2025-01-05", 153.0, 154.0),
                signal_row("2025-01-06", 155.0, 156.0),
            ],
            config=config,
        )
        self.assertEqual(result.trades["action"].tolist(), ["BUY", "SELL"])
        self.assertEqual(result.watchlist.iloc[0]["status"], "EXPIRED")
        self.assertEqual(result.final_states["AAA"], PositionState.NOT_HELD.value)

    def test_ten_percent_loss_holds_until_recovery_to_positive_five(self) -> None:
        result = self.run_rows(
            [
                signal_row("2025-01-01", 100.0, 100.0, entry=True),
                signal_row("2025-01-02", 100.0, 85.0, weekly_exit=True),
                signal_row("2025-01-03", 84.0, 90.0, weekly_exit=True),
                signal_row("2025-01-04", 91.0, 106.0),
                signal_row("2025-01-05", 105.0, 107.0),
            ]
        )
        self.assertEqual(result.trades["action"].tolist(), ["BUY", "SELL"])
        recovery_sell = result.trades.iloc[1]
        self.assertEqual(recovery_sell["reason"], "LOSS_RECOVERY_EXIT")
        self.assertEqual(recovery_sell["signal_date"], pd.Timestamp("2025-01-04"))
        self.assertEqual(recovery_sell["execution_date"], pd.Timestamp("2025-01-05"))
        self.assertAlmostEqual(recovery_sell["realized_pnl_pct"], 0.05)
        episode = result.loss_recovery_log.iloc[0]
        self.assertEqual(episode["trigger_date"], pd.Timestamp("2025-01-02"))
        self.assertAlmostEqual(episode["trigger_return"], -0.15)
        self.assertEqual(episode["resolution"], "LOSS_RECOVERY_EXIT")

    def test_thirty_percent_loss_sells_half_and_sidelines_the_remainder(self) -> None:
        result = self.run_rows(
            [
                signal_row("2025-01-01", 100.0, 100.0, entry=True),
                signal_row("2025-01-02", 100.0, 85.0),
                signal_row("2025-01-03", 84.0, 69.0),
                signal_row("2025-01-04", 68.0, 70.0),
                signal_row("2025-01-05", 90.0, 100.0),
                signal_row("2025-01-06", 101.0, 101.0),
            ]
        )
        catastrophic_sell = result.trades.iloc[1]
        self.assertEqual(catastrophic_sell["reason"], "CATASTROPHIC_STOP_30_HALF_SALE")
        self.assertEqual(catastrophic_sell["action"], "SELL_HALF_AND_SIDELINE")
        self.assertAlmostEqual(
            catastrophic_sell["shares"], result.trades.iloc[0]["shares"] * 0.50
        )
        self.assertAlmostEqual(catastrophic_sell["realized_pnl_pct"], -0.32)
        episode = result.loss_recovery_log.iloc[0]
        self.assertEqual(episode["trough_date"], pd.Timestamp("2025-01-03"))
        self.assertAlmostEqual(episode["trough_return"], -0.31)
        self.assertTrue(bool(episode["reached_breakeven_after_exit"]))
        self.assertEqual(
            episode["first_breakeven_after_exit_date"], pd.Timestamp("2025-01-05")
        )
        self.assertTrue(bool(episode["reached_positive_after_exit"]))
        self.assertEqual(
            episode["first_positive_after_exit_date"], pd.Timestamp("2025-01-06")
        )
        self.assertEqual(result.final_states["AAA"], PositionState.HELD_SIDELINED.value)
        self.assertEqual(int(result.equity_curve.iloc[-1]["number_of_positions"]), 0)
        self.assertEqual(
            int(result.equity_curve.iloc[-1]["number_of_sidelined_positions"]), 1
        )

    def test_unresolved_loss_recovery_is_reported_at_backtest_end(self) -> None:
        result = self.run_rows(
            [
                signal_row("2025-01-01", 100.0, 100.0, entry=True),
                signal_row("2025-01-02", 100.0, 85.0),
                signal_row("2025-01-03", 84.0, 80.0),
            ]
        )
        self.assertEqual(result.trades["action"].tolist(), ["BUY"])
        self.assertEqual(result.loss_recovery_log.iloc[0]["resolution"], "OPEN_AT_END")
        self.assertEqual(result.final_states["AAA"], PositionState.HELD_FULL.value)

    def test_rolling_universe_blocks_entry_until_ticker_appears(self) -> None:
        rolling = RollingUniverse(
            {
                pd.Timestamp("2025-01-01"): pd.DataFrame(
                    {"ticker": ["BBB"], "eligible": [True]}
                ),
                pd.Timestamp("2025-01-03"): pd.DataFrame(
                    {"ticker": ["AAA"], "eligible": [True]}
                ),
            }
        )
        result = self.run_rows(
            [
                signal_row("2025-01-01", 100.0, 100.0, entry=True),
                signal_row("2025-01-02", 101.0, 101.0, entry=True),
                signal_row("2025-01-03", 102.0, 102.0, entry=True),
                signal_row("2025-01-04", 103.0, 103.0),
            ],
            rolling_universe=rolling,
        )

        self.assertEqual(result.trades["action"].tolist(), ["BUY"])
        buy = result.trades.iloc[0]
        self.assertEqual(buy["signal_date"], pd.Timestamp("2025-01-03"))
        self.assertEqual(buy["execution_date"], pd.Timestamp("2025-01-04"))
        self.assertEqual(buy["universe_snapshot_date"], pd.Timestamp("2025-01-03"))

    def test_snapshot_removal_does_not_force_an_existing_position_exit(self) -> None:
        rolling = RollingUniverse(
            {
                pd.Timestamp("2025-01-01"): pd.DataFrame(
                    {"ticker": ["AAA"], "eligible": [True]}
                ),
                pd.Timestamp("2025-01-03"): pd.DataFrame(
                    {"ticker": ["BBB"], "eligible": [True]}
                ),
            }
        )
        result = self.run_rows(
            [
                signal_row("2025-01-01", 100.0, 100.0, entry=True),
                signal_row("2025-01-02", 101.0, 101.0),
                signal_row("2025-01-03", 102.0, 102.0),
                signal_row("2025-01-04", 103.0, 103.0),
            ],
            rolling_universe=rolling,
        )

        self.assertEqual(result.trades["action"].tolist(), ["BUY"])
        self.assertEqual(result.final_states["AAA"], PositionState.HELD_FULL.value)
        self.assertEqual(result.equity_curve["universe_snapshot_date"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
