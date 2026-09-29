"""Long-only, next-open V1 backtester with explicit per-ticker states."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from config import (
    DEFAULT_BACKTEST_CONFIG,
    DEFAULT_STRATEGY_CONFIG,
    BacktestConfig,
    StrategyConfig,
)
from universe import Universe

if TYPE_CHECKING:
    from rolling_universe import RollingUniverse


class PositionState(str, Enum):
    NOT_HELD = "NOT_HELD"
    HELD_FULL = "HELD_FULL"
    HELD_RUNNER = "HELD_RUNNER"
    HELD_SIDELINED = "HELD_SIDELINED"
    WATCHING_REENTRY = "WATCHING_REENTRY"
    COOLDOWN = "COOLDOWN"
    REJECTED = "REJECTED"


@dataclass
class Position:
    ticker: str
    shares: float
    entry_price: float
    entry_date: pd.Timestamp
    classification: str
    runner_reference_price: float | None = None
    loss_recovery_active: bool = False
    loss_recovery_trigger_date: pd.Timestamp | None = None
    loss_recovery_trigger_return: float | None = None
    loss_recovery_trough_date: pd.Timestamp | None = None
    loss_recovery_trough_return: float | None = None
    reached_breakeven_before_exit: bool = False
    reached_positive_before_exit: bool = False
    sidelined: bool = False


@dataclass
class PendingOrder:
    ticker: str
    action: str
    reason: str
    signal_date: pd.Timestamp
    signal_price: float
    position_return_at_signal: float | None
    universe_snapshot_date: pd.Timestamp | None = None
    sell_fraction: float = 0.0
    cooldown_days: int = 0


@dataclass
class WatchRecord:
    ticker: str
    exit_date: pd.Timestamp
    exit_price: float
    watch_start_date: pd.Timestamp
    watch_end_date: pd.Timestamp
    status: str = "ACTIVE"


@dataclass(frozen=True)
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    watchlist: pd.DataFrame
    entry_candidate_log: pd.DataFrame
    loss_recovery_log: pd.DataFrame
    final_states: dict[str, str]


TRADE_COLUMNS = [
    "signal_date",
    "execution_date",
    "ticker",
    "action",
    "state_before",
    "state_after",
    "reason",
    "signal_price",
    "execution_price",
    "gap_return",
    "shares",
    "cash_before",
    "cash_after",
    "position_value",
    "entry_price",
    "position_return_at_signal",
    "realized_pnl",
    "realized_pnl_pct",
    "holding_period_days",
    "classification",
    "universe_snapshot_date",
]

EQUITY_COLUMNS = [
    "date",
    "cash",
    "positions_value",
    "active_positions_value",
    "sidelined_positions_value",
    "total_portfolio_value",
    "number_of_positions",
    "number_of_sidelined_positions",
    "total_number_of_positions",
    "drawdown",
    "universe_snapshot_date",
]

WATCH_COLUMNS = [
    "ticker",
    "exit_date",
    "exit_price",
    "watch_start_date",
    "watch_end_date",
    "status",
]

ENTRY_CANDIDATE_COLUMNS = [
    "date",
    "ticker",
    "selected",
    "selection_mode",
    "random_seed",
    "candidate_count",
    "open_slots",
    "reason",
]

LOSS_RECOVERY_COLUMNS = [
    "ticker",
    "entry_date",
    "entry_price",
    "trigger_date",
    "trigger_return",
    "trough_date",
    "trough_return",
    "resolution_signal_date",
    "resolution_date",
    "resolution",
    "return_at_resolution_signal",
    "execution_return",
    "sold_fraction",
    "shares_sold",
    "shares_remaining",
    "remaining_value_at_execution",
    "reached_breakeven_before_exit",
    "reached_positive_before_exit",
    "reached_breakeven_after_exit",
    "first_breakeven_after_exit_date",
    "reached_positive_after_exit",
    "first_positive_after_exit_date",
    "max_return_after_exit_through_backtest",
]

BACKTEST_REQUIRED_COLUMNS = {
    "date",
    "ticker",
    "open",
    "close",
    "ma_50d",
    "momentum_30d",
    "relative_volume_30d",
    "entry_signal",
    "spike_signal",
    "weekly_underperformance_signal",
}


class Backtester:
    """Execute close-generated signals at the next available daily open."""

    def __init__(
        self,
        signals: pd.DataFrame,
        universe: Universe,
        strategy_config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
        backtest_config: BacktestConfig = DEFAULT_BACKTEST_CONFIG,
        rolling_universe: RollingUniverse | None = None,
    ) -> None:
        missing = BACKTEST_REQUIRED_COLUMNS.difference(signals.columns)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Signal data is missing required columns: {names}")

        self.data = signals.copy()
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.data["ticker"] = self.data["ticker"].astype(str).str.upper()
        for column in ("open", "close"):
            values = cast(
                pd.Series,
                pd.to_numeric(cast(pd.Series, self.data[column]), errors="coerce"),
            )
            if bool(values.isna().any()) or bool(values.le(0).any()):
                raise ValueError(f"Signal data contains invalid {column} prices.")
            self.data[column] = values.astype(float)
        self.data = self.data.sort_values(["date", "ticker"]).reset_index(drop=True)
        if self.data.empty:
            raise ValueError("Cannot backtest an empty signal dataset.")
        if self.data.duplicated(["ticker", "date"]).any():
            raise ValueError("Signal data contains duplicate ticker/date rows.")

        self.universe = universe
        self.strategy_config = strategy_config
        self.config = backtest_config
        self.rolling_universe = rolling_universe
        self.cash = float(backtest_config.starting_cash)
        self.positions: dict[str, Position] = {}
        self.pending_orders: dict[str, PendingOrder] = {}
        self.cooldown_ends: dict[str, pd.Timestamp] = {}
        self.active_watches: dict[str, WatchRecord] = {}
        self.watch_history: list[WatchRecord] = []
        self.trade_rows: list[dict[str, Any]] = []
        self.equity_rows: list[dict[str, Any]] = []
        self.entry_candidate_rows: list[dict[str, Any]] = []
        self.loss_recovery_rows: list[dict[str, Any]] = []
        self.last_closes: dict[str, float] = {}
        self.states: dict[str, PositionState] = {
            ticker: PositionState.NOT_HELD for ticker in universe.approved_universe
        }
        self.states.update(
            {ticker: PositionState.REJECTED for ticker in universe.rejected_universe}
        )

        self.ticker_dates: dict[str, list[pd.Timestamp]] = {}
        for raw_ticker, group in self.data.groupby("ticker", sort=False):
            ticker = str(raw_ticker)
            date_series = cast(pd.Series, group["date"])
            self.ticker_dates[ticker] = [
                cast(pd.Timestamp, pd.Timestamp(cast(Any, date)))
                for date in date_series.sort_values().drop_duplicates().tolist()
            ]
        self.ticker_date_indexes: dict[str, dict[pd.Timestamp, int]] = {
            ticker: {date: index for index, date in enumerate(dates)}
            for ticker, dates in self.ticker_dates.items()
        }

    def run(self) -> BacktestResult:
        """Run the backtest and return trades, daily equity, and watch records."""
        for raw_date, daily_frame in self.data.groupby("date", sort=True):
            current_date = cast(pd.Timestamp, pd.Timestamp(cast(Any, raw_date)))
            daily_rows: dict[str, pd.Series] = {
                str(row["ticker"]): row for _, row in daily_frame.iterrows()
            }
            self._execute_pending_orders(current_date, daily_rows)
            self.last_closes.update(
                {ticker: float(row["close"]) for ticker, row in daily_rows.items()}
            )
            self._expire_temporary_states(current_date)
            self._generate_close_orders(current_date, daily_rows)
            self._record_equity(current_date)

        self._finalize_open_loss_recoveries()
        self._add_post_exit_recovery_outcomes()
        trades = pd.DataFrame(self.trade_rows, columns=pd.Index(TRADE_COLUMNS))
        equity = pd.DataFrame(self.equity_rows, columns=pd.Index(EQUITY_COLUMNS))
        watches = pd.DataFrame(
            [vars(record) for record in self.watch_history],
            columns=pd.Index(WATCH_COLUMNS),
        )
        candidate_log = pd.DataFrame(
            self.entry_candidate_rows,
            columns=pd.Index(ENTRY_CANDIDATE_COLUMNS),
        )
        loss_recovery_log = pd.DataFrame(
            self.loss_recovery_rows,
            columns=pd.Index(LOSS_RECOVERY_COLUMNS),
        )
        final_states = {ticker: state.value for ticker, state in self.states.items()}
        return BacktestResult(
            trades,
            equity,
            watches,
            candidate_log,
            loss_recovery_log,
            final_states,
        )

    def _execute_pending_orders(
        self,
        current_date: pd.Timestamp,
        daily_rows: dict[str, pd.Series],
    ) -> None:
        executable = [
            order
            for ticker, order in self.pending_orders.items()
            if ticker in daily_rows and current_date > order.signal_date
        ]
        executable.sort(
            key=lambda order: (not order.action.startswith("SELL"), order.ticker)
        )
        for order in executable:
            row = daily_rows[order.ticker]
            if order.action.startswith("SELL"):
                self._execute_sell(order, current_date, float(row["open"]))
            else:
                self._execute_buy(order, current_date, float(row["open"]), daily_rows)
            self.pending_orders.pop(order.ticker, None)

    def _execute_buy(
        self,
        order: PendingOrder,
        execution_date: pd.Timestamp,
        execution_price: float,
        daily_rows: dict[str, pd.Series],
    ) -> None:
        if (
            self.rolling_universe is not None
            and order.ticker
            not in self.rolling_universe.eligible_tickers(execution_date)
        ):
            return
        existing = self.positions.get(order.ticker)
        if (
            existing is None
            and self._active_position_count() >= self.config.max_positions
        ):
            return

        portfolio_value = self._portfolio_value_at_open(daily_rows)
        current_value = 0.0 if existing is None else existing.shares * execution_price
        target_value = portfolio_value * self.config.target_position_fraction
        desired_purchase = max(target_value - current_value, 0.0)
        minimum_cash = portfolio_value * self.config.min_cash_buffer_fraction
        available_cash = max(self.cash - minimum_cash, 0.0)
        purchase_value = min(desired_purchase, available_cash)
        if purchase_value <= 0 or execution_price <= 0:
            return

        shares = purchase_value / execution_price
        cash_before = self.cash
        state_before = self.states[order.ticker]
        self.cash -= purchase_value
        classification = (
            self.universe.classification_for(order.ticker)
            if self.rolling_universe is None
            else self.rolling_universe.position_classification_for(
                order.ticker,
                execution_date,
            )
        )

        if existing is None:
            position = Position(
                ticker=order.ticker,
                shares=shares,
                entry_price=execution_price,
                entry_date=execution_date,
                classification=classification,
            )
            self.positions[order.ticker] = position
        else:
            old_cost = existing.shares * existing.entry_price
            existing.shares += shares
            existing.entry_price = (old_cost + purchase_value) / existing.shares
            existing.entry_date = execution_date
            existing.runner_reference_price = None
            existing.loss_recovery_active = False
            existing.loss_recovery_trigger_date = None
            existing.loss_recovery_trigger_return = None
            existing.loss_recovery_trough_date = None
            existing.loss_recovery_trough_return = None
            existing.reached_breakeven_before_exit = False
            existing.reached_positive_before_exit = False
            existing.sidelined = False
            position = existing

        self.states[order.ticker] = PositionState.HELD_FULL
        if order.ticker in self.active_watches:
            self.active_watches.pop(order.ticker).status = "REENTERED"

        self._append_trade(
            order=order,
            execution_date=execution_date,
            execution_price=execution_price,
            shares=shares,
            cash_before=cash_before,
            state_before=state_before,
            position_value=position.shares * execution_price,
            entry_price=position.entry_price,
            realized_pnl=0.0,
            realized_pnl_pct=0.0,
            holding_period_days=0,
            classification=classification,
            action="BUY",
        )

    def _execute_sell(
        self,
        order: PendingOrder,
        execution_date: pd.Timestamp,
        execution_price: float,
    ) -> None:
        position = self.positions.get(order.ticker)
        if position is None or execution_price <= 0:
            return

        state_before = self.states[order.ticker]
        shares = position.shares * order.sell_fraction
        shares = min(shares, position.shares)
        proceeds = shares * execution_price
        cash_before = self.cash
        entry_price = position.entry_price
        realized_pnl = shares * (execution_price - entry_price)
        realized_pnl_pct = execution_price / entry_price - 1
        holding_period = self._trading_day_distance(
            order.ticker, position.entry_date, execution_date
        )
        if order.reason in {
            "LOSS_RECOVERY_EXIT",
            "CATASTROPHIC_STOP_30_HALF_SALE",
        }:
            self._record_loss_recovery_resolution(
                position,
                order,
                execution_date,
                execution_price,
            )
        self.cash += proceeds
        position.shares -= shares

        if order.action == "SELL_PARTIAL":
            self.states[order.ticker] = PositionState.HELD_RUNNER
            position.runner_reference_price = execution_price
            self._start_watch(order.ticker, execution_date, execution_price)
            position_value = position.shares * execution_price
            logged_action = "SELL_PARTIAL"
        elif order.action == "SELL_HALF_AND_SIDELINE":
            self.states[order.ticker] = PositionState.HELD_SIDELINED
            position.sidelined = True
            position.loss_recovery_active = False
            position.runner_reference_price = None
            self._cancel_watch(order.ticker)
            position_value = position.shares * execution_price
            logged_action = "SELL_HALF_AND_SIDELINE"
        else:
            self.positions.pop(order.ticker)
            position_value = 0.0
            logged_action = "SELL"
            if order.action == "SELL_TO_WATCH":
                self.states[order.ticker] = PositionState.WATCHING_REENTRY
                self._start_watch(order.ticker, execution_date, execution_price)
            elif order.action == "SELL_TO_COOLDOWN":
                self.states[order.ticker] = PositionState.COOLDOWN
                self.cooldown_ends[order.ticker] = self._future_ticker_date(
                    order.ticker, execution_date, order.cooldown_days
                )
                self._cancel_watch(order.ticker)
            else:
                self.states[order.ticker] = PositionState.NOT_HELD

        self._append_trade(
            order=order,
            execution_date=execution_date,
            execution_price=execution_price,
            shares=shares,
            cash_before=cash_before,
            state_before=state_before,
            position_value=position_value,
            entry_price=entry_price,
            realized_pnl=realized_pnl,
            realized_pnl_pct=realized_pnl_pct,
            holding_period_days=holding_period,
            classification=position.classification,
            action=logged_action,
        )

    def _append_trade(
        self,
        order: PendingOrder,
        execution_date: pd.Timestamp,
        execution_price: float,
        shares: float,
        cash_before: float,
        state_before: PositionState,
        position_value: float,
        entry_price: float,
        realized_pnl: float,
        realized_pnl_pct: float,
        holding_period_days: int,
        classification: str,
        action: str,
    ) -> None:
        self.trade_rows.append(
            {
                "signal_date": order.signal_date,
                "execution_date": execution_date,
                "ticker": order.ticker,
                "action": action,
                "state_before": state_before.value,
                "state_after": self.states[order.ticker].value,
                "reason": order.reason,
                "signal_price": order.signal_price,
                "execution_price": execution_price,
                "gap_return": execution_price / order.signal_price - 1,
                "shares": shares,
                "cash_before": cash_before,
                "cash_after": self.cash,
                "position_value": position_value,
                "entry_price": entry_price,
                "position_return_at_signal": order.position_return_at_signal,
                "realized_pnl": realized_pnl,
                "realized_pnl_pct": realized_pnl_pct,
                "holding_period_days": holding_period_days,
                "classification": classification,
                "universe_snapshot_date": order.universe_snapshot_date,
            }
        )

    def _generate_close_orders(
        self,
        current_date: pd.Timestamp,
        daily_rows: dict[str, pd.Series],
    ) -> None:
        eligible_for_entry = set(
            self.universe.approved_universe
            if self.rolling_universe is None
            else self.rolling_universe.eligible_tickers(current_date)
        )
        eligible_for_entry -= (
            self.universe.rejected_universe | self.universe.benchmark_universe
        )

        # Evaluate every held position before allocating tomorrow's entry slots.
        for ticker in sorted(self.universe.approved_universe):
            if ticker not in daily_rows or ticker in self.pending_orders:
                continue
            row = daily_rows[ticker]
            state = self.states[ticker]

            if state in {
                PositionState.HELD_FULL,
                PositionState.HELD_RUNNER,
            } and self._queue_position_exit(ticker, current_date, row, state):
                continue

        # Re-entries retain their state-machine priority over new positions.
        for ticker in sorted(self.universe.approved_universe):
            if ticker not in daily_rows or ticker in self.pending_orders:
                continue
            row = daily_rows[ticker]
            state = self.states[ticker]
            if (
                ticker in eligible_for_entry
                and ticker in self.active_watches
                and state
                in {
                    PositionState.WATCHING_REENTRY,
                    PositionState.HELD_RUNNER,
                }
                and self._queue_reentry(ticker, current_date, row)
            ):
                continue

        candidates = [
            ticker
            for ticker in sorted(eligible_for_entry)
            if ticker in daily_rows
            and ticker not in self.pending_orders
            and ticker not in self.positions
            and self.states.get(ticker) == PositionState.NOT_HELD
            and bool(daily_rows[ticker]["entry_signal"])
        ]
        open_slots = self._projected_open_slots()
        ordered_candidates = self._order_entry_candidates(
            candidates,
            current_date,
        )
        selected = set(ordered_candidates[:open_slots])
        candidate_count = len(candidates)

        for ticker in ordered_candidates:
            row = daily_rows[ticker]
            was_selected = ticker in selected
            if was_selected:
                self.pending_orders[ticker] = PendingOrder(
                    ticker=ticker,
                    action="BUY_ENTRY",
                    reason="ENTRY_SIGNAL",
                    signal_date=current_date,
                    signal_price=float(row["close"]),
                    position_return_at_signal=None,
                    universe_snapshot_date=self._snapshot_date(current_date),
                )
                reason = "SELECTED"
            elif open_slots <= 0:
                reason = "NO_OPEN_SLOTS"
            else:
                reason = "NOT_SELECTED_RANDOM"

            self.entry_candidate_rows.append(
                {
                    "date": current_date,
                    "ticker": ticker,
                    "selected": was_selected,
                    "selection_mode": "random",
                    "random_seed": self.config.random_seed,
                    "candidate_count": candidate_count,
                    "open_slots": open_slots,
                    "reason": reason,
                }
            )

    def _projected_open_slots(self) -> int:
        """Return slots expected after already queued next-open orders execute."""
        slot_releases = sum(
            order.action.startswith("SELL") and order.action != "SELL_PARTIAL"
            for order in self.pending_orders.values()
            if order.ticker in self.positions
            and not self.positions[order.ticker].sidelined
        )
        new_position_buys = sum(
            order.action.startswith("BUY") and order.ticker not in self.positions
            for order in self.pending_orders.values()
        )
        projected_positions = (
            self._active_position_count() - slot_releases + new_position_buys
        )
        return max(self.config.max_positions - projected_positions, 0)

    def _active_position_count(self) -> int:
        """Count positions participating in the configured active-position cap."""
        return sum(not position.sidelined for position in self.positions.values())

    def _order_entry_candidates(
        self,
        candidates: list[str],
        current_date: pd.Timestamp,
    ) -> list[str]:
        """Shuffle the full candidate pool reproducibly for one trading date."""
        ordered = sorted(candidates)
        date_seed = int(current_date.strftime("%Y%m%d"))
        random.Random(self.config.random_seed + date_seed).shuffle(ordered)
        return ordered

    def _queue_position_exit(
        self,
        ticker: str,
        signal_date: pd.Timestamp,
        row: pd.Series,
        state: PositionState,
    ) -> bool:
        position = self.positions[ticker]
        close = float(row["close"])
        position_return = close / position.entry_price - 1

        if position.loss_recovery_active:
            self._update_loss_recovery(position, signal_date, position_return)
            if position_return <= self.strategy_config.hard_stop_loss:
                self._queue_catastrophic_half_sale(
                    ticker,
                    signal_date,
                    close,
                    position_return,
                )
            elif position_return >= self.strategy_config.loss_recovery_exit:
                self._queue_sell(
                    ticker,
                    signal_date,
                    close,
                    "LOSS_RECOVERY_EXIT",
                    position_return,
                    1.0,
                    self.config.loss_recovery_cooldown_days,
                )
            return True

        if position_return <= self.strategy_config.loss_recovery_trigger:
            self._start_loss_recovery(position, signal_date, position_return)
            if position_return <= self.strategy_config.hard_stop_loss:
                self._queue_catastrophic_half_sale(
                    ticker,
                    signal_date,
                    close,
                    position_return,
                )
            return True

        if bool(row["weekly_underperformance_signal"]):
            reason = (
                "RUNNER_WEEKLY_UNDERPERFORMANCE"
                if state == PositionState.HELD_RUNNER
                else "WEEKLY_UNDERPERFORMANCE"
            )
            self._queue_sell(
                ticker,
                signal_date,
                close,
                reason,
                position_return,
                1.0,
                self.config.weekly_drop_cooldown_days,
            )
            return True

        if state == PositionState.HELD_RUNNER:
            runner_reference = position.runner_reference_price
            if runner_reference is None:
                raise RuntimeError(f"Runner position has no reference price: {ticker}")
            runner_return = close / runner_reference - 1
            if runner_return <= self.strategy_config.runner_stop_loss:
                reason = "RUNNER_STOP"
            elif not math.isnan(self._number(row, "ma_50d")) and close < self._number(
                row, "ma_50d"
            ):
                reason = "RUNNER_BELOW_MA50"
            else:
                return False
            self._queue_sell(
                ticker,
                signal_date,
                close,
                reason,
                position_return,
                1.0,
                self.config.runner_cooldown_days,
            )
            return True

        if (
            bool(row["spike_signal"])
            and position_return >= self.strategy_config.min_spike_profit
        ):
            if position.classification == "recovery_candidate":
                action = "SELL_TO_WATCH"
                sell_fraction = 1.0
                reason = "SPIKE_EXIT_RECOVERY"
            else:
                action = "SELL_PARTIAL"
                sell_fraction = self.config.compounder_sale_fraction
                reason = "SPIKE_PARTIAL_COMPOUNDER"
            self.pending_orders[ticker] = PendingOrder(
                ticker=ticker,
                action=action,
                reason=reason,
                signal_date=signal_date,
                signal_price=close,
                position_return_at_signal=position_return,
                universe_snapshot_date=self._snapshot_date(signal_date),
                sell_fraction=sell_fraction,
            )
            return True
        return False

    def _queue_catastrophic_half_sale(
        self,
        ticker: str,
        signal_date: pd.Timestamp,
        signal_price: float,
        position_return: float,
    ) -> None:
        self.pending_orders[ticker] = PendingOrder(
            ticker=ticker,
            action="SELL_HALF_AND_SIDELINE",
            reason="CATASTROPHIC_STOP_30_HALF_SALE",
            signal_date=signal_date,
            signal_price=signal_price,
            position_return_at_signal=position_return,
            universe_snapshot_date=self._snapshot_date(signal_date),
            sell_fraction=self.config.catastrophic_sale_fraction,
        )

    @staticmethod
    def _start_loss_recovery(
        position: Position,
        signal_date: pd.Timestamp,
        position_return: float,
    ) -> None:
        position.loss_recovery_active = True
        position.loss_recovery_trigger_date = signal_date
        position.loss_recovery_trigger_return = position_return
        position.loss_recovery_trough_date = signal_date
        position.loss_recovery_trough_return = position_return
        position.reached_breakeven_before_exit = position_return >= 0.0
        position.reached_positive_before_exit = position_return > 0.0

    @staticmethod
    def _update_loss_recovery(
        position: Position,
        signal_date: pd.Timestamp,
        position_return: float,
    ) -> None:
        if (
            position.loss_recovery_trough_return is None
            or position_return < position.loss_recovery_trough_return
        ):
            position.loss_recovery_trough_date = signal_date
            position.loss_recovery_trough_return = position_return
        position.reached_breakeven_before_exit |= position_return >= 0.0
        position.reached_positive_before_exit |= position_return > 0.0

    def _record_loss_recovery_resolution(
        self,
        position: Position,
        order: PendingOrder,
        execution_date: pd.Timestamp,
        execution_price: float,
    ) -> None:
        self.loss_recovery_rows.append(
            {
                "ticker": position.ticker,
                "entry_date": position.entry_date,
                "entry_price": position.entry_price,
                "trigger_date": position.loss_recovery_trigger_date,
                "trigger_return": position.loss_recovery_trigger_return,
                "trough_date": position.loss_recovery_trough_date,
                "trough_return": position.loss_recovery_trough_return,
                "resolution_signal_date": order.signal_date,
                "resolution_date": execution_date,
                "resolution": order.reason,
                "return_at_resolution_signal": order.position_return_at_signal,
                "execution_return": execution_price / position.entry_price - 1,
                "sold_fraction": order.sell_fraction,
                "shares_sold": position.shares * order.sell_fraction,
                "shares_remaining": position.shares * (1.0 - order.sell_fraction),
                "remaining_value_at_execution": (
                    position.shares * (1.0 - order.sell_fraction) * execution_price
                ),
                "reached_breakeven_before_exit": (
                    position.reached_breakeven_before_exit
                ),
                "reached_positive_before_exit": position.reached_positive_before_exit,
                "reached_breakeven_after_exit": False,
                "first_breakeven_after_exit_date": pd.NaT,
                "reached_positive_after_exit": False,
                "first_positive_after_exit_date": pd.NaT,
                "max_return_after_exit_through_backtest": None,
            }
        )

    def _finalize_open_loss_recoveries(self) -> None:
        if self.data.empty:
            return
        final_date = cast(
            pd.Timestamp, pd.Timestamp(cast(Any, self.data["date"].max()))
        )
        for position in self.positions.values():
            if not position.loss_recovery_active:
                continue
            final_close = self.last_closes.get(position.ticker, position.entry_price)
            self.loss_recovery_rows.append(
                {
                    "ticker": position.ticker,
                    "entry_date": position.entry_date,
                    "entry_price": position.entry_price,
                    "trigger_date": position.loss_recovery_trigger_date,
                    "trigger_return": position.loss_recovery_trigger_return,
                    "trough_date": position.loss_recovery_trough_date,
                    "trough_return": position.loss_recovery_trough_return,
                    "resolution_signal_date": pd.NaT,
                    "resolution_date": final_date,
                    "resolution": "OPEN_AT_END",
                    "return_at_resolution_signal": None,
                    "execution_return": final_close / position.entry_price - 1,
                    "sold_fraction": 0.0,
                    "shares_sold": 0.0,
                    "shares_remaining": position.shares,
                    "remaining_value_at_execution": None,
                    "reached_breakeven_before_exit": (
                        position.reached_breakeven_before_exit
                    ),
                    "reached_positive_before_exit": (
                        position.reached_positive_before_exit
                    ),
                    "reached_breakeven_after_exit": False,
                    "first_breakeven_after_exit_date": pd.NaT,
                    "reached_positive_after_exit": False,
                    "first_positive_after_exit_date": pd.NaT,
                    "max_return_after_exit_through_backtest": None,
                }
            )

    def _add_post_exit_recovery_outcomes(self) -> None:
        for episode in self.loss_recovery_rows:
            if episode["resolution"] == "OPEN_AT_END":
                continue
            ticker = str(episode["ticker"])
            resolution_date = cast(
                pd.Timestamp, pd.Timestamp(cast(Any, episode["resolution_date"]))
            )
            entry_price = float(cast(Any, episode["entry_price"]))
            future = self.data.loc[
                self.data["ticker"].eq(ticker) & self.data["date"].gt(resolution_date),
                ["date", "close"],
            ].sort_values("date")
            if future.empty:
                continue
            future_returns = future["close"].astype(float) / entry_price - 1
            episode["max_return_after_exit_through_backtest"] = float(
                future_returns.max()
            )
            breakeven = future.loc[future_returns.ge(0.0)]
            positive = future.loc[future_returns.gt(0.0)]
            if not breakeven.empty:
                episode["reached_breakeven_after_exit"] = True
                episode["first_breakeven_after_exit_date"] = breakeven.iloc[0]["date"]
            if not positive.empty:
                episode["reached_positive_after_exit"] = True
                episode["first_positive_after_exit_date"] = positive.iloc[0]["date"]

    def _queue_sell(
        self,
        ticker: str,
        signal_date: pd.Timestamp,
        signal_price: float,
        reason: str,
        position_return: float,
        sell_fraction: float,
        cooldown_days: int,
    ) -> None:
        self.pending_orders[ticker] = PendingOrder(
            ticker=ticker,
            action="SELL_TO_COOLDOWN",
            reason=reason,
            signal_date=signal_date,
            signal_price=signal_price,
            position_return_at_signal=position_return,
            universe_snapshot_date=self._snapshot_date(signal_date),
            sell_fraction=sell_fraction,
            cooldown_days=cooldown_days,
        )

    def _queue_reentry(
        self,
        ticker: str,
        signal_date: pd.Timestamp,
        row: pd.Series,
    ) -> bool:
        watch = self.active_watches[ticker]
        close = float(row["close"])
        return_since_exit = close / watch.exit_price - 1
        relative_volume = self._number(row, "relative_volume_30d")
        ma_50d = self._number(row, "ma_50d")
        momentum_30d = self._number(row, "momentum_30d")
        should_reenter = (
            return_since_exit >= self.strategy_config.reentry_min_return
            and return_since_exit < self.strategy_config.reentry_max_return
            and not math.isnan(relative_volume)
            and relative_volume < self.strategy_config.max_relative_volume_30d
            and not math.isnan(ma_50d)
            and close > ma_50d
            and not math.isnan(momentum_30d)
            and momentum_30d > 0
        )
        if not should_reenter:
            return False

        position = self.positions.get(ticker)
        position_return = None if position is None else close / position.entry_price - 1
        self.pending_orders[ticker] = PendingOrder(
            ticker=ticker,
            action="BUY_REENTRY",
            reason="REENTRY_SIGNAL",
            signal_date=signal_date,
            signal_price=close,
            position_return_at_signal=position_return,
            universe_snapshot_date=self._snapshot_date(signal_date),
        )
        return True

    def _expire_temporary_states(self, current_date: pd.Timestamp) -> None:
        for ticker, end_date in list(self.cooldown_ends.items()):
            if current_date >= end_date:
                self.states[ticker] = PositionState.NOT_HELD
                self.cooldown_ends.pop(ticker)

        for ticker, watch in list(self.active_watches.items()):
            if current_date > watch.watch_end_date:
                watch.status = "EXPIRED"
                self.active_watches.pop(ticker)
                if self.states[ticker] == PositionState.WATCHING_REENTRY:
                    self.states[ticker] = PositionState.NOT_HELD

    def _start_watch(
        self, ticker: str, execution_date: pd.Timestamp, execution_price: float
    ) -> None:
        self._cancel_watch(ticker)
        watch = WatchRecord(
            ticker=ticker,
            exit_date=execution_date,
            exit_price=execution_price,
            watch_start_date=execution_date,
            watch_end_date=self._future_ticker_date(
                ticker, execution_date, self.config.max_watch_days
            ),
        )
        self.active_watches[ticker] = watch
        self.watch_history.append(watch)

    def _cancel_watch(self, ticker: str) -> None:
        watch = self.active_watches.pop(ticker, None)
        if watch is not None and watch.status == "ACTIVE":
            watch.status = "CANCELLED"

    def _future_ticker_date(
        self, ticker: str, current_date: pd.Timestamp, trading_days: int
    ) -> pd.Timestamp:
        dates = self.ticker_dates[ticker]
        current_index = self.ticker_date_indexes[ticker][current_date]
        return dates[min(current_index + trading_days, len(dates) - 1)]

    def _trading_day_distance(
        self, ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> int:
        indexes = self.ticker_date_indexes[ticker]
        return max(indexes[end_date] - indexes[start_date], 0)

    def _portfolio_value_at_open(self, daily_rows: dict[str, pd.Series]) -> float:
        positions_value = 0.0
        for ticker, position in self.positions.items():
            if ticker in daily_rows:
                price = float(daily_rows[ticker]["open"])
            else:
                price = self.last_closes.get(ticker, position.entry_price)
            positions_value += position.shares * price
        return self.cash + positions_value

    @staticmethod
    def _number(row: pd.Series, column: str) -> float:
        """Convert a dynamically typed Pandas scalar to a numeric value."""
        return float(cast(Any, row[column]))

    def _record_equity(self, current_date: pd.Timestamp) -> None:
        active_positions_value = 0.0
        sidelined_positions_value = 0.0
        for ticker, position in self.positions.items():
            value = position.shares * self.last_closes.get(ticker, position.entry_price)
            if position.sidelined:
                sidelined_positions_value += value
            else:
                active_positions_value += value
        positions_value = active_positions_value + sidelined_positions_value
        total_value = self.cash + positions_value
        previous_peak = max(
            [self.config.starting_cash]
            + [float(row["total_portfolio_value"]) for row in self.equity_rows]
        )
        peak = max(previous_peak, total_value)
        drawdown = total_value / peak - 1
        self.equity_rows.append(
            {
                "date": current_date,
                "cash": self.cash,
                "positions_value": positions_value,
                "active_positions_value": active_positions_value,
                "sidelined_positions_value": sidelined_positions_value,
                "total_portfolio_value": total_value,
                "number_of_positions": self._active_position_count(),
                "number_of_sidelined_positions": sum(
                    position.sidelined for position in self.positions.values()
                ),
                "total_number_of_positions": len(self.positions),
                "drawdown": drawdown,
                "universe_snapshot_date": self._snapshot_date(current_date),
            }
        )

    def _snapshot_date(self, date: pd.Timestamp) -> pd.Timestamp | None:
        if self.rolling_universe is None:
            return None
        return self.rolling_universe.snapshot_date_for(date)


def run_backtest(
    signals: pd.DataFrame,
    universe: Universe,
    strategy_config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
    backtest_config: BacktestConfig = DEFAULT_BACKTEST_CONFIG,
    rolling_universe: RollingUniverse | None = None,
) -> BacktestResult:
    """Convenience wrapper for running the V1 backtester."""
    return Backtester(
        signals,
        universe,
        strategy_config,
        backtest_config,
        rolling_universe,
    ).run()
