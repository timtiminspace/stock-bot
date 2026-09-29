"""Performance metrics and diagnostic tables for V1 backtest results."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

MetricValue = float | str

TRADE_RANKING_COLUMNS = [
    "execution_date",
    "ticker",
    "reason",
    "classification",
    "shares",
    "entry_price",
    "execution_price",
    "realized_pnl",
    "realized_pnl_pct",
    "holding_period_days",
    "gap_return",
]


def _safe_mean(values: pd.Series) -> float:
    return 0.0 if values.empty else float(values.mean())


def _prepare_inputs(
    equity_curve: pd.DataFrame, trades: pd.DataFrame, prices: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if equity_curve.empty:
        raise ValueError("Cannot calculate metrics from an empty equity curve.")

    equity = equity_curve.copy()
    equity["date"] = pd.to_datetime(equity["date"])
    equity = equity.sort_values("date").reset_index(drop=True)

    trade_data = trades.copy()
    if "execution_date" in trade_data.columns:
        trade_data["execution_date"] = pd.to_datetime(trade_data["execution_date"])
    if "signal_date" in trade_data.columns:
        trade_data["signal_date"] = pd.to_datetime(trade_data["signal_date"])

    price_data = prices.copy()
    price_data["date"] = pd.to_datetime(price_data["date"])
    price_data["ticker"] = price_data["ticker"].astype(str).str.upper()
    price_data = price_data.sort_values(["ticker", "date"]).reset_index(drop=True)
    return equity, trade_data, price_data


def _max_drawdown_period(equity: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    values = equity["total_portfolio_value"].astype(float).to_numpy()
    running_peaks = np.maximum.accumulate(values)
    drawdowns = values / running_peaks - 1
    trough_position = int(np.argmin(drawdowns))
    peak_position = int(np.argmax(values[: trough_position + 1]))
    return (
        cast(pd.Timestamp, pd.Timestamp(equity.iloc[peak_position]["date"])),
        cast(pd.Timestamp, pd.Timestamp(equity.iloc[trough_position]["date"])),
    )


def _spike_exit_forward_returns(
    trades: pd.DataFrame, prices: pd.DataFrame, trading_days: int = 20
) -> list[float]:
    spike_exits = trades.loc[trades["reason"].str.contains("SPIKE", na=False)]
    returns: list[float] = []
    for _, trade in spike_exits.iterrows():
        future = prices.loc[
            prices["ticker"].eq(trade["ticker"])
            & prices["date"].gt(pd.Timestamp(trade["execution_date"]))
        ]
        if len(future) >= trading_days:
            future_close = float(future.iloc[trading_days - 1]["close"])
            returns.append(future_close / float(trade["execution_price"]) - 1)
    return returns


def _reentry_success_rate(trades: pd.DataFrame) -> float:
    reentries = trades.loc[
        trades["action"].eq("BUY") & trades["reason"].eq("REENTRY_SIGNAL")
    ]
    outcomes: list[bool] = []
    for _, reentry in reentries.iterrows():
        later_sells = trades.loc[
            trades["ticker"].eq(reentry["ticker"])
            & trades["action"].str.startswith("SELL")
            & trades["execution_date"].gt(reentry["execution_date"])
        ].sort_values("execution_date")
        if not later_sells.empty:
            outcomes.append(float(later_sells.iloc[0]["realized_pnl"]) > 0)
    return 0.0 if not outcomes else sum(outcomes) / len(outcomes)


def _loss_recovery_metrics(
    loss_recovery_log: pd.DataFrame | None,
) -> dict[str, float]:
    """Summarize close-based −10% recovery episodes and −30% stops."""
    names = {
        "number_of_loss_recovery_triggers": 0.0,
        "number_of_loss_recovery_exits": 0.0,
        "number_of_catastrophic_stops_30": 0.0,
        "unique_tickers_hitting_minus_30": 0.0,
        "number_of_open_loss_recoveries": 0.0,
        "loss_recovery_exit_rate": 0.0,
        "recovery_episodes_later_reaching_breakeven": 0.0,
        "recovery_episodes_later_reaching_positive": 0.0,
        "catastrophic_stops_later_reaching_breakeven": 0.0,
        "catastrophic_stops_later_reaching_positive": 0.0,
    }
    if loss_recovery_log is None or loss_recovery_log.empty:
        return names

    episodes = loss_recovery_log.copy()
    recovery_exits = episodes.loc[episodes["resolution"].eq("LOSS_RECOVERY_EXIT")]
    catastrophic = episodes.loc[
        episodes["resolution"].isin(
            ("CATASTROPHIC_STOP_30", "CATASTROPHIC_STOP_30_HALF_SALE")
        )
    ]
    open_episodes = episodes.loc[episodes["resolution"].eq("OPEN_AT_END")]
    completed = len(recovery_exits) + len(catastrophic)
    later_breakeven = (
        episodes["reached_breakeven_after_exit"].fillna(False).astype(bool)
    )
    later_positive = episodes["reached_positive_after_exit"].fillna(False).astype(bool)
    catastrophic_breakeven = (
        catastrophic["reached_breakeven_after_exit"].fillna(False).astype(bool)
    )
    catastrophic_positive = (
        catastrophic["reached_positive_after_exit"].fillna(False).astype(bool)
    )
    return {
        "number_of_loss_recovery_triggers": float(len(episodes)),
        "number_of_loss_recovery_exits": float(len(recovery_exits)),
        "number_of_catastrophic_stops_30": float(len(catastrophic)),
        "unique_tickers_hitting_minus_30": float(catastrophic["ticker"].nunique()),
        "number_of_open_loss_recoveries": float(len(open_episodes)),
        "loss_recovery_exit_rate": (
            0.0 if completed == 0 else float(len(recovery_exits) / completed)
        ),
        "recovery_episodes_later_reaching_breakeven": float(later_breakeven.sum()),
        "recovery_episodes_later_reaching_positive": float(later_positive.sum()),
        "catastrophic_stops_later_reaching_breakeven": float(
            catastrophic_breakeven.sum()
        ),
        "catastrophic_stops_later_reaching_positive": float(
            catastrophic_positive.sum()
        ),
    }


def calculate_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    starting_cash: float,
    benchmark_ticker: str = "QQQ",
    loss_recovery_log: pd.DataFrame | None = None,
) -> dict[str, MetricValue]:
    """Calculate scalar portfolio, capacity, trade, and benchmark metrics."""
    equity, trade_data, price_data = _prepare_inputs(equity_curve, trades, prices)
    final_value = float(equity.iloc[-1]["total_portfolio_value"])
    total_return = final_value / starting_cash - 1
    periods = max(len(equity) - 1, 1)
    annualized_return = (final_value / starting_cash) ** (252 / periods) - 1
    portfolio_values = equity["total_portfolio_value"].astype(float)
    daily_returns = (portfolio_values / portfolio_values.shift(1) - 1).dropna()
    sharpe_ratio = (
        0.0
        if daily_returns.empty or float(daily_returns.std()) == 0.0
        else float(np.sqrt(252) * daily_returns.mean() / daily_returns.std())
    )

    sells = trade_data.loc[trade_data["action"].str.startswith("SELL")].copy()
    wins = sells.loc[sells["realized_pnl"].gt(0)]
    losses = sells.loc[sells["realized_pnl"].lt(0)]
    gross_profit = float(wins["realized_pnl"].sum())
    gross_loss = abs(float(losses["realized_pnl"].sum()))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    benchmark = price_data.loc[
        price_data["ticker"].eq(benchmark_ticker.upper())
        & price_data["date"].between(equity["date"].min(), equity["date"].max())
    ].sort_values("date")
    benchmark_return = (
        0.0
        if benchmark.empty
        else float(benchmark.iloc[-1]["close"] / benchmark.iloc[0]["close"] - 1)
    )

    spike_exits = trade_data.loc[trade_data["reason"].str.contains("SPIKE", na=False)]
    spike_forward_returns = _spike_exit_forward_returns(trade_data, price_data)
    reentries = trade_data.loc[
        trade_data["action"].eq("BUY") & trade_data["reason"].eq("REENTRY_SIGNAL")
    ]
    runner_sells = sells.loc[sells["reason"].str.startswith("RUNNER", na=False)]
    drawdown_start, drawdown_end = _max_drawdown_period(equity)
    observed_max_positions = int(equity["number_of_positions"].max())
    days_at_max_positions = (
        0
        if observed_max_positions == 0
        else int(equity["number_of_positions"].eq(observed_max_positions).sum())
    )
    snapshots_used = (
        0
        if "universe_snapshot_date" not in equity.columns
        else int(equity["universe_snapshot_date"].dropna().nunique())
    )
    sidelined_counts = (
        equity["number_of_sidelined_positions"].astype(float)
        if "number_of_sidelined_positions" in equity.columns
        else pd.Series(0.0, index=equity.index)
    )
    sidelined_values = (
        equity["sidelined_positions_value"].astype(float)
        if "sidelined_positions_value" in equity.columns
        else pd.Series(0.0, index=equity.index)
    )
    final_sidelined_value = float(sidelined_values.iloc[-1])
    recovery_metrics = _loss_recovery_metrics(loss_recovery_log)

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": float(equity["drawdown"].min()),
        "max_drawdown_period_start": drawdown_start.date().isoformat(),
        "max_drawdown_period_end": drawdown_end.date().isoformat(),
        "average_positions_held": float(equity["number_of_positions"].mean()),
        "max_positions_held": float(observed_max_positions),
        "days_at_max_positions": float(days_at_max_positions),
        "number_of_universe_snapshots_used": float(snapshots_used),
        "average_sidelined_positions": float(sidelined_counts.mean()),
        "max_sidelined_positions": float(sidelined_counts.max()),
        "final_sidelined_positions": float(sidelined_counts.iloc[-1]),
        "final_sidelined_value": final_sidelined_value,
        "final_sidelined_portfolio_fraction": (
            0.0 if final_value == 0.0 else final_sidelined_value / final_value
        ),
        "number_of_trades": float(len(trade_data)),
        "win_rate": 0.0 if sells.empty else float(len(wins) / len(sells)),
        "average_win": _safe_mean(wins["realized_pnl_pct"]),
        "average_loss": _safe_mean(losses["realized_pnl_pct"]),
        "profit_factor": profit_factor,
        "average_holding_period": _safe_mean(sells["holding_period_days"]),
        "best_trade": 0.0 if sells.empty else float(sells["realized_pnl_pct"].max()),
        "worst_trade": 0.0 if sells.empty else float(sells["realized_pnl_pct"].min()),
        "cash_drag": float((equity["cash"] / equity["total_portfolio_value"]).mean()),
        "benchmark_return_QQQ": benchmark_return,
        "strategy_vs_QQQ": total_return - benchmark_return,
        "number_of_spike_exits": float(len(spike_exits)),
        "average_return_after_spike_exit": (
            0.0 if not spike_forward_returns else float(np.mean(spike_forward_returns))
        ),
        "number_of_reentries": float(len(reentries)),
        "reentry_success_rate": _reentry_success_rate(trade_data),
        "runner_performance": _safe_mean(runner_sells["realized_pnl_pct"]),
        **recovery_metrics,
    }


def _annual_return_tables(
    equity: pd.DataFrame,
    prices: pd.DataFrame,
    starting_cash: float,
    benchmark_ticker: str,
) -> dict[str, pd.DataFrame]:
    strategy_rows: list[dict[str, Any]] = []
    previous_strategy_value = starting_cash
    for year, group in equity.groupby(equity["date"].dt.year, sort=True):
        ending_value = float(group.iloc[-1]["total_portfolio_value"])
        strategy_rows.append(
            {
                "year": int(cast(Any, year)),
                "annual_return_strategy": ending_value / previous_strategy_value - 1,
            }
        )
        previous_strategy_value = ending_value
    strategy = pd.DataFrame(strategy_rows)

    benchmark_data = prices.loc[
        prices["ticker"].eq(benchmark_ticker.upper())
        & prices["date"].between(equity["date"].min(), equity["date"].max())
    ]
    benchmark_rows: list[dict[str, Any]] = []
    previous_benchmark_close: float | None = None
    for year, group in benchmark_data.groupby(
        benchmark_data["date"].dt.year, sort=True
    ):
        first_close = float(group.iloc[0]["close"])
        ending_close = float(group.iloc[-1]["close"])
        starting_close = (
            first_close
            if previous_benchmark_close is None
            else previous_benchmark_close
        )
        benchmark_rows.append(
            {
                "year": int(cast(Any, year)),
                "annual_return_QQQ": ending_close / starting_close - 1,
            }
        )
        previous_benchmark_close = ending_close
    benchmark = pd.DataFrame(benchmark_rows)

    comparison = strategy.merge(benchmark, on="year", how="outer").sort_values("year")
    comparison["strategy_vs_QQQ"] = (
        comparison["annual_return_strategy"] - comparison["annual_return_QQQ"]
    )
    return {
        "annual_returns_strategy": strategy,
        "annual_returns_QQQ": benchmark,
        "strategy_vs_QQQ_by_year": comparison,
    }


def _drawdown_by_year(equity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, group in equity.groupby(equity["date"].dt.year, sort=True):
        yearly = group.reset_index(drop=True)
        values = yearly["total_portfolio_value"].astype(float).to_numpy()
        running_peaks = np.maximum.accumulate(values)
        drawdowns = values / running_peaks - 1
        trough_position = int(np.argmin(drawdowns))
        peak_position = int(np.argmax(values[: trough_position + 1]))
        rows.append(
            {
                "year": int(cast(Any, year)),
                "max_drawdown": float(drawdowns[trough_position]),
                "drawdown_start": pd.Timestamp(yearly.iloc[peak_position]["date"]),
                "drawdown_end": pd.Timestamp(yearly.iloc[trough_position]["date"]),
            }
        )
    return pd.DataFrame(rows)


def _close_on_or_before(prices: pd.DataFrame, ticker: str, date: pd.Timestamp) -> float:
    history = prices.loc[prices["ticker"].eq(ticker) & prices["date"].le(date), "close"]
    return 0.0 if history.empty else float(history.iloc[-1])


def _signed_shares(trades: pd.DataFrame) -> pd.Series:
    direction = np.where(trades["action"].eq("BUY"), 1.0, -1.0)
    return trades["shares"].astype(float) * direction


def _ticker_performance_tables(
    trades: pd.DataFrame, prices: pd.DataFrame, ending_date: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    pnl_rows: list[dict[str, Any]] = []
    return_rows: list[dict[str, Any]] = []
    for ticker in sorted(set(trades["ticker"].astype(str))):
        ticker_trades = trades.loc[trades["ticker"].eq(ticker)].copy()
        buys = ticker_trades.loc[ticker_trades["action"].eq("BUY")]
        sells = ticker_trades.loc[ticker_trades["action"].str.startswith("SELL")]
        buy_value = float((buys["shares"] * buys["execution_price"]).sum())
        sell_value = float((sells["shares"] * sells["execution_price"]).sum())
        ending_shares = float(_signed_shares(ticker_trades).sum())
        ending_value = ending_shares * _close_on_or_before(prices, ticker, ending_date)
        total_pnl = ending_value + sell_value - buy_value
        realized_pnl = float(sells["realized_pnl"].sum())

        pnl_rows.append(
            {
                "ticker": ticker,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": total_pnl - realized_pnl,
                "total_pnl": total_pnl,
                "ending_position_value": ending_value,
                "gross_buy_value": buy_value,
                "gross_sell_value": sell_value,
            }
        )
        return_rows.append(
            {
                "ticker": ticker,
                "return_on_gross_buy_value": 0.0
                if buy_value == 0
                else total_pnl / buy_value,
                "average_exit_return": _safe_mean(sells["realized_pnl_pct"]),
                "best_exit_return": 0.0
                if sells.empty
                else float(sells["realized_pnl_pct"].max()),
                "worst_exit_return": 0.0
                if sells.empty
                else float(sells["realized_pnl_pct"].min()),
                "number_of_entries": len(buys),
                "number_of_exits": len(sells),
            }
        )
    return_columns = [
        "ticker",
        "return_on_gross_buy_value",
        "average_exit_return",
        "best_exit_return",
        "worst_exit_return",
        "number_of_entries",
        "number_of_exits",
    ]
    pnl_columns = [
        "ticker",
        "realized_pnl",
        "unrealized_pnl",
        "total_pnl",
        "ending_position_value",
        "gross_buy_value",
        "gross_sell_value",
    ]
    return {
        "returns_by_ticker": pd.DataFrame(
            return_rows, columns=pd.Index(return_columns)
        ),
        "pnl_by_ticker": pd.DataFrame(
            pnl_rows, columns=pd.Index(pnl_columns)
        ).sort_values("total_pnl"),
    }


def _trades_by_exit_reason(trades: pd.DataFrame) -> pd.DataFrame:
    sells = trades.loc[trades["action"].str.startswith("SELL")]
    rows: list[dict[str, Any]] = []
    for reason in sorted(set(sells["reason"].astype(str))):
        exits = sells.loc[sells["reason"].eq(reason)]
        rows.append(
            {
                "exit_reason": reason,
                "number_of_exits": len(exits),
                "win_rate": float(exits["realized_pnl"].gt(0).mean()),
                "average_return": float(exits["realized_pnl_pct"].mean()),
                "total_realized_pnl": float(exits["realized_pnl"].sum()),
            }
        )
    columns = [
        "exit_reason",
        "number_of_exits",
        "win_rate",
        "average_return",
        "total_realized_pnl",
    ]
    return pd.DataFrame(rows, columns=pd.Index(columns))


def _drawdown_attribution_by_ticker(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    start_date, end_date = _max_drawdown_period(equity)
    start_value = float(
        equity.loc[equity["date"].eq(start_date), "total_portfolio_value"].iloc[0]
    )
    end_value = float(
        equity.loc[equity["date"].eq(end_date), "total_portfolio_value"].iloc[0]
    )
    portfolio_loss = end_value - start_value
    rows: list[dict[str, Any]] = []

    for ticker in sorted(set(trades["ticker"].astype(str))):
        ticker_trades = trades.loc[trades["ticker"].eq(ticker)].copy()
        through_start = ticker_trades.loc[
            ticker_trades["execution_date"].le(start_date)
        ]
        through_end = ticker_trades.loc[ticker_trades["execution_date"].le(end_date)]
        during_drawdown = ticker_trades.loc[
            ticker_trades["execution_date"].gt(start_date)
            & ticker_trades["execution_date"].le(end_date)
        ]
        start_shares = float(_signed_shares(through_start).sum())
        end_shares = float(_signed_shares(through_end).sum())
        start_position_value = start_shares * _close_on_or_before(
            prices, ticker, start_date
        )
        end_position_value = end_shares * _close_on_or_before(prices, ticker, end_date)
        buys = during_drawdown.loc[during_drawdown["action"].eq("BUY")]
        sells = during_drawdown.loc[during_drawdown["action"].str.startswith("SELL")]
        buy_cost = float((buys["shares"] * buys["execution_price"]).sum())
        sell_proceeds = float((sells["shares"] * sells["execution_price"]).sum())
        ticker_pnl = (
            end_position_value - start_position_value + sell_proceeds - buy_cost
        )
        if abs(ticker_pnl) < 1e-9 and start_position_value == 0 and buy_cost == 0:
            continue
        rows.append(
            {
                "ticker": ticker,
                "drawdown_period_start": start_date,
                "drawdown_period_end": end_date,
                "start_position_value": start_position_value,
                "buy_cost_during_drawdown": buy_cost,
                "sell_proceeds_during_drawdown": sell_proceeds,
                "end_position_value": end_position_value,
                "drawdown_period_pnl": ticker_pnl,
                "contribution_to_portfolio_drawdown": ticker_pnl / start_value,
                "share_of_drawdown_loss": 0.0
                if portfolio_loss == 0
                else ticker_pnl / portfolio_loss,
            }
        )
    columns = [
        "ticker",
        "drawdown_period_start",
        "drawdown_period_end",
        "start_position_value",
        "buy_cost_during_drawdown",
        "sell_proceeds_during_drawdown",
        "end_position_value",
        "drawdown_period_pnl",
        "contribution_to_portfolio_drawdown",
        "share_of_drawdown_loss",
    ]
    return pd.DataFrame(rows, columns=pd.Index(columns)).sort_values(
        "drawdown_period_pnl"
    )


def calculate_metric_tables(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    starting_cash: float,
    benchmark_ticker: str = "QQQ",
) -> dict[str, pd.DataFrame]:
    """Build annual, ticker, exit, ranking, and drawdown-attribution tables."""
    equity, trade_data, price_data = _prepare_inputs(equity_curve, trades, prices)
    sells = trade_data.loc[trade_data["action"].str.startswith("SELL")]
    ranking_columns = [
        column for column in TRADE_RANKING_COLUMNS if column in sells.columns
    ]
    tables = _annual_return_tables(equity, price_data, starting_cash, benchmark_ticker)
    tables["drawdown_by_year"] = _drawdown_by_year(equity)
    tables.update(
        _ticker_performance_tables(
            trade_data,
            price_data,
            cast(
                pd.Timestamp,
                pd.Timestamp(cast(Any, equity.iloc[-1]["date"])),
            ),
        )
    )
    tables["trades_by_exit_reason"] = _trades_by_exit_reason(trade_data)
    if sells.empty:
        tables["worst_10_trades"] = pd.DataFrame(columns=pd.Index(ranking_columns))
        tables["best_10_trades"] = pd.DataFrame(columns=pd.Index(ranking_columns))
    else:
        tables["worst_10_trades"] = sells.nsmallest(10, "realized_pnl_pct")[
            ranking_columns
        ].reset_index(drop=True)
        tables["best_10_trades"] = sells.nlargest(10, "realized_pnl_pct")[
            ranking_columns
        ].reset_index(drop=True)
    tables["drawdown_attribution_by_ticker"] = _drawdown_attribution_by_ticker(
        equity, trade_data, price_data
    )
    return tables
