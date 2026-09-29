"""Run reproducible random-selection backtests over one prepared signal dataset."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
from backtester import run_backtest
from config import BacktestConfig, StrategyConfig
from data_loader import (
    DATA_DIR,
    DEFAULT_START_DATE,
    clean_yfinance_data,
    download_price_data,
    filter_price_dates,
    load_data,
    save_data,
)
from features import add_features
from metrics import MetricValue, calculate_metrics
from rolling_universe import RollingUniverse
from strategy import add_strategy_signals
from universe import load_universe

DEFAULT_ROLLING_UNIVERSE_DIR = DATA_DIR / "universe_snapshots"

RUN_COLUMNS = [
    "seed",
    "total_return",
    "annualized_return",
    "max_drawdown",
    "sharpe_ratio",
    "benchmark_return_QQQ",
    "strategy_vs_QQQ",
    "number_of_trades",
    "cash_drag",
    "average_sidelined_positions",
    "max_sidelined_positions",
    "final_sidelined_positions",
    "final_sidelined_value",
    "final_sidelined_portfolio_fraction",
    "number_of_loss_recovery_triggers",
    "number_of_loss_recovery_exits",
    "number_of_catastrophic_stops_30",
    "unique_tickers_hitting_minus_30",
    "number_of_open_loss_recoveries",
    "loss_recovery_exit_rate",
    "recovery_episodes_later_reaching_breakeven",
    "recovery_episodes_later_reaching_positive",
    "catastrophic_stops_later_reaching_breakeven",
    "catastrophic_stops_later_reaching_positive",
]

SUMMARY_COLUMNS = [
    "median_total_return",
    "p10_total_return",
    "p90_total_return",
    "median_max_drawdown",
    "pct_runs_beating_QQQ",
    "best_seed",
    "worst_seed",
    "median_catastrophic_stops_30",
    "p90_catastrophic_stops_30",
    "pct_runs_with_catastrophic_stop_30",
    "pct_catastrophic_stops_later_reaching_breakeven",
    "pct_catastrophic_stops_later_reaching_positive",
    "total_loss_recovery_triggers",
    "total_loss_recovery_exits",
    "total_catastrophic_stops_30",
    "total_open_loss_recoveries",
    "unique_tickers_hitting_minus_30_across_runs",
    "median_final_sidelined_positions",
    "median_final_sidelined_value",
    "median_final_sidelined_portfolio_fraction",
]

CATASTROPHIC_STOP_SUMMARY_COLUMNS = [
    "ticker",
    "catastrophic_stop_episodes",
    "seeds_affected",
    "first_trigger_date",
    "last_stop_date",
    "worst_trough_return",
    "average_execution_return",
    "later_breakeven_count",
    "later_positive_count",
    "pct_later_reaching_breakeven",
    "pct_later_reaching_positive",
    "best_post_exit_return_through_backtest",
]

TRADE_LEDGER_COLUMNS = [
    "seed",
    "trade_rank_best",
    "trade_rank_worst",
    "ticker",
    "entry_signal_date",
    "entry_date",
    "exit_signal_date",
    "exit_date",
    "entry_execution_price",
    "cost_basis_price",
    "exit_price",
    "shares_sold",
    "capital_at_risk",
    "realized_pnl",
    "realized_return",
    "exit_reason",
    "exit_action",
    "holding_period_days",
    "classification",
]

STOCK_PERFORMANCE_COLUMNS = [
    "seed",
    "stock_rank_best",
    "stock_rank_worst",
    "ticker",
    "first_entry_date",
    "last_exit_date",
    "completed_exit_count",
    "total_realized_pnl",
    "total_cost_basis",
    "realized_return",
    "best_trade_return",
    "worst_trade_return",
]

SEED_RANKING_DETAIL_COLUMNS = [
    "seed_rank",
    *RUN_COLUMNS,
    "completed_stock_count",
    "best_stock_ticker",
    "best_stock_entry_date",
    "best_stock_exit_date",
    "best_stock_return",
    "best_stock_realized_pnl",
    "best_stock_trade_count",
    "worst_stock_ticker",
    "worst_stock_entry_date",
    "worst_stock_exit_date",
    "worst_stock_return",
    "worst_stock_realized_pnl",
    "worst_stock_trade_count",
]


def _numeric_metric(metrics: dict[str, MetricValue], name: str) -> float:
    value = metrics[name]
    if isinstance(value, str):
        raise TypeError(f"Expected numeric metric {name}, received {value!r}.")
    return float(value)


def _load_or_download_prices(
    universe_tickers: list[str],
    *,
    refresh: bool,
    start: str,
    end: str | None,
) -> pd.DataFrame:
    prices_path = DATA_DIR / "prices.csv"
    if refresh or not prices_path.exists():
        prices = clean_yfinance_data(
            download_price_data(universe_tickers, start=start, end=end)
        )
        prices = filter_price_dates(prices, start=start, end=end)
        save_data(prices, "prices.csv")
        return prices
    else:
        prices = load_data("prices.csv")
    return filter_price_dates(prices, start=start, end=end)


def summarize_random_runs(
    runs: pd.DataFrame,
    loss_recovery_log: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the requested one-row Monte Carlo distribution summary."""
    if runs.empty:
        raise ValueError("Cannot summarize zero random backtest runs.")
    best = runs.sort_values("total_return", ascending=False, kind="stable").iloc[0]
    worst = runs.sort_values("total_return", ascending=True, kind="stable").iloc[0]
    catastrophic_counts = (
        runs["number_of_catastrophic_stops_30"]
        if "number_of_catastrophic_stops_30" in runs.columns
        else pd.Series(0.0, index=runs.index)
    )
    later_breakeven = (
        runs["catastrophic_stops_later_reaching_breakeven"]
        if "catastrophic_stops_later_reaching_breakeven" in runs.columns
        else pd.Series(0.0, index=runs.index)
    )
    later_positive = (
        runs["catastrophic_stops_later_reaching_positive"]
        if "catastrophic_stops_later_reaching_positive" in runs.columns
        else pd.Series(0.0, index=runs.index)
    )
    catastrophic_total = float(catastrophic_counts.sum())
    trigger_total = (
        0.0
        if "number_of_loss_recovery_triggers" not in runs.columns
        else float(runs["number_of_loss_recovery_triggers"].sum())
    )
    recovery_exit_total = (
        0.0
        if "number_of_loss_recovery_exits" not in runs.columns
        else float(runs["number_of_loss_recovery_exits"].sum())
    )
    open_total = (
        0.0
        if "number_of_open_loss_recoveries" not in runs.columns
        else float(runs["number_of_open_loss_recoveries"].sum())
    )
    unique_catastrophic_tickers = 0
    if loss_recovery_log is not None and not loss_recovery_log.empty:
        catastrophic_log = loss_recovery_log.loc[
            loss_recovery_log["resolution"].isin(
                ("CATASTROPHIC_STOP_30", "CATASTROPHIC_STOP_30_HALF_SALE")
            )
        ]
        unique_catastrophic_tickers = int(catastrophic_log["ticker"].nunique())
    summary = {
        "median_total_return": float(runs["total_return"].median()),
        "p10_total_return": float(runs["total_return"].quantile(0.10)),
        "p90_total_return": float(runs["total_return"].quantile(0.90)),
        "median_max_drawdown": float(runs["max_drawdown"].median()),
        "pct_runs_beating_QQQ": float(runs["strategy_vs_QQQ"].gt(0).mean() * 100.0),
        "best_seed": int(best["seed"]),
        "worst_seed": int(worst["seed"]),
        "median_catastrophic_stops_30": float(catastrophic_counts.median()),
        "p90_catastrophic_stops_30": float(catastrophic_counts.quantile(0.90)),
        "pct_runs_with_catastrophic_stop_30": float(
            catastrophic_counts.gt(0).mean() * 100.0
        ),
        "pct_catastrophic_stops_later_reaching_breakeven": (
            0.0
            if catastrophic_total == 0.0
            else float(later_breakeven.sum() / catastrophic_total * 100.0)
        ),
        "pct_catastrophic_stops_later_reaching_positive": (
            0.0
            if catastrophic_total == 0.0
            else float(later_positive.sum() / catastrophic_total * 100.0)
        ),
        "total_loss_recovery_triggers": trigger_total,
        "total_loss_recovery_exits": recovery_exit_total,
        "total_catastrophic_stops_30": catastrophic_total,
        "total_open_loss_recoveries": open_total,
        "unique_tickers_hitting_minus_30_across_runs": float(
            unique_catastrophic_tickers
        ),
        "median_final_sidelined_positions": float(
            runs["final_sidelined_positions"].median()
            if "final_sidelined_positions" in runs.columns
            else 0.0
        ),
        "median_final_sidelined_value": float(
            runs["final_sidelined_value"].median()
            if "final_sidelined_value" in runs.columns
            else 0.0
        ),
        "median_final_sidelined_portfolio_fraction": float(
            runs["final_sidelined_portfolio_fraction"].median()
            if "final_sidelined_portfolio_fraction" in runs.columns
            else 0.0
        ),
    }
    return pd.DataFrame([summary], columns=pd.Index(SUMMARY_COLUMNS))


def build_catastrophic_stop_summary(loss_recovery_log: pd.DataFrame) -> pd.DataFrame:
    """Aggregate every −30% episode by ticker across random seeds."""
    if loss_recovery_log.empty:
        return pd.DataFrame(columns=pd.Index(CATASTROPHIC_STOP_SUMMARY_COLUMNS))
    catastrophic = loss_recovery_log.loc[
        loss_recovery_log["resolution"].isin(
            ("CATASTROPHIC_STOP_30", "CATASTROPHIC_STOP_30_HALF_SALE")
        )
    ].copy()
    if catastrophic.empty:
        return pd.DataFrame(columns=pd.Index(CATASTROPHIC_STOP_SUMMARY_COLUMNS))

    summary = (
        catastrophic.groupby("ticker", as_index=False)
        .agg(
            catastrophic_stop_episodes=("ticker", "size"),
            seeds_affected=("seed", "nunique"),
            first_trigger_date=("trigger_date", "min"),
            last_stop_date=("resolution_date", "max"),
            worst_trough_return=("trough_return", "min"),
            average_execution_return=("execution_return", "mean"),
            later_breakeven_count=("reached_breakeven_after_exit", "sum"),
            later_positive_count=("reached_positive_after_exit", "sum"),
            best_post_exit_return_through_backtest=(
                "max_return_after_exit_through_backtest",
                "max",
            ),
        )
        .reset_index(drop=True)
    )
    summary["pct_later_reaching_breakeven"] = (
        summary["later_breakeven_count"] / summary["catastrophic_stop_episodes"] * 100.0
    )
    summary["pct_later_reaching_positive"] = (
        summary["later_positive_count"] / summary["catastrophic_stop_episodes"] * 100.0
    )
    return (
        summary.loc[:, CATASTROPHIC_STOP_SUMMARY_COLUMNS]
        .sort_values(["catastrophic_stop_episodes", "ticker"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_seed_trade_ledger(trades: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Match each realized exit to its active buy and rank exits for one seed."""
    if trades.empty:
        return pd.DataFrame(columns=pd.Index(TRADE_LEDGER_COLUMNS))

    data = trades.copy().reset_index(drop=True)
    data["execution_date"] = pd.to_datetime(data["execution_date"])
    data["signal_date"] = pd.to_datetime(data["signal_date"])
    data["_event_order"] = range(len(data))
    data = data.sort_values(["execution_date", "_event_order"])
    active_entries: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []

    for _, row in data.iterrows():
        ticker = str(row["ticker"])
        action = str(row["action"])
        if action == "BUY":
            active_entries[ticker] = {
                "signal_date": row["signal_date"],
                "execution_date": row["execution_date"],
                "execution_price": float(row["execution_price"]),
            }
            continue
        if not action.startswith("SELL"):
            continue

        entry = active_entries.get(ticker, {})
        shares = float(row["shares"])
        cost_basis_price = float(row["entry_price"])
        rows.append(
            {
                "seed": seed,
                "ticker": ticker,
                "entry_signal_date": entry.get("signal_date", pd.NaT),
                "entry_date": entry.get("execution_date", pd.NaT),
                "exit_signal_date": row["signal_date"],
                "exit_date": row["execution_date"],
                "entry_execution_price": entry.get("execution_price", pd.NA),
                "cost_basis_price": cost_basis_price,
                "exit_price": float(row["execution_price"]),
                "shares_sold": shares,
                "capital_at_risk": shares * cost_basis_price,
                "realized_pnl": float(row["realized_pnl"]),
                "realized_return": float(row["realized_pnl_pct"]),
                "exit_reason": str(row["reason"]),
                "exit_action": action,
                "holding_period_days": int(row["holding_period_days"]),
                "classification": str(row["classification"]),
            }
        )
        if action == "SELL":
            active_entries.pop(ticker, None)

    ledger = pd.DataFrame(rows)
    if ledger.empty:
        return pd.DataFrame(columns=pd.Index(TRADE_LEDGER_COLUMNS))
    ledger["trade_rank_best"] = (
        ledger["realized_return"].rank(method="first", ascending=False).astype(int)
    )
    ledger["trade_rank_worst"] = (
        ledger["realized_return"].rank(method="first", ascending=True).astype(int)
    )
    return ledger.loc[:, TRADE_LEDGER_COLUMNS].reset_index(drop=True)


def build_seed_stock_performance(trade_ledger: pd.DataFrame) -> pd.DataFrame:
    """Aggregate completed exits into per-seed, per-ticker realized performance."""
    if trade_ledger.empty:
        return pd.DataFrame(columns=pd.Index(STOCK_PERFORMANCE_COLUMNS))

    performance = (
        trade_ledger.groupby(["seed", "ticker"], as_index=False)
        .agg(
            first_entry_date=("entry_date", "min"),
            last_exit_date=("exit_date", "max"),
            completed_exit_count=("ticker", "size"),
            total_realized_pnl=("realized_pnl", "sum"),
            total_cost_basis=("capital_at_risk", "sum"),
            best_trade_return=("realized_return", "max"),
            worst_trade_return=("realized_return", "min"),
        )
        .reset_index(drop=True)
    )
    performance["realized_return"] = (
        performance["total_realized_pnl"] / performance["total_cost_basis"]
    )
    performance["stock_rank_best"] = performance.groupby("seed")[
        "realized_return"
    ].rank(method="first", ascending=False)
    performance["stock_rank_worst"] = performance.groupby("seed")[
        "realized_return"
    ].rank(method="first", ascending=True)
    performance["stock_rank_best"] = performance["stock_rank_best"].astype(int)
    performance["stock_rank_worst"] = performance["stock_rank_worst"].astype(int)
    return (
        performance.loc[:, STOCK_PERFORMANCE_COLUMNS]
        .sort_values(["seed", "stock_rank_best"])
        .reset_index(drop=True)
    )


def build_seed_rankings(
    runs: pd.DataFrame,
    stock_performance: pd.DataFrame,
) -> pd.DataFrame:
    """Rank seeds and attach each seed's best and worst realized ticker."""
    if runs.empty:
        return pd.DataFrame(columns=pd.Index(SEED_RANKING_DETAIL_COLUMNS))

    ranked = runs.sort_values(
        ["total_return", "seed"], ascending=[False, True]
    ).reset_index(drop=True)
    ranked["seed_rank"] = pd.Series(
        range(1, len(ranked) + 1), index=ranked.index, dtype="int64"
    )
    details: list[dict[str, object]] = []
    for _, run in ranked.iterrows():
        seed = int(run["seed"])
        stocks = stock_performance.loc[stock_performance["seed"].eq(seed)]
        detail: dict[str, object] = {
            "completed_stock_count": len(stocks),
            "best_stock_ticker": pd.NA,
            "best_stock_entry_date": pd.NaT,
            "best_stock_exit_date": pd.NaT,
            "best_stock_return": pd.NA,
            "best_stock_realized_pnl": pd.NA,
            "best_stock_trade_count": pd.NA,
            "worst_stock_ticker": pd.NA,
            "worst_stock_entry_date": pd.NaT,
            "worst_stock_exit_date": pd.NaT,
            "worst_stock_return": pd.NA,
            "worst_stock_realized_pnl": pd.NA,
            "worst_stock_trade_count": pd.NA,
        }
        if not stocks.empty:
            best = stocks.sort_values(
                "stock_rank_best", ascending=True, kind="stable"
            ).iloc[0]
            worst = stocks.sort_values(
                "stock_rank_worst", ascending=True, kind="stable"
            ).iloc[0]
            for label, stock in (("best", best), ("worst", worst)):
                detail.update(
                    {
                        f"{label}_stock_ticker": stock["ticker"],
                        f"{label}_stock_entry_date": stock["first_entry_date"],
                        f"{label}_stock_exit_date": stock["last_exit_date"],
                        f"{label}_stock_return": stock["realized_return"],
                        f"{label}_stock_realized_pnl": stock["total_realized_pnl"],
                        f"{label}_stock_trade_count": stock["completed_exit_count"],
                    }
                )
        details.append(detail)

    detail_table = pd.DataFrame(details)
    ranked = pd.concat([ranked, detail_table], axis=1)
    return ranked.loc[:, SEED_RANKING_DETAIL_COLUMNS]


def run_random_backtests(
    *,
    start: str,
    end: str | None,
    rolling_universe_dir: Path,
    runs: int,
    max_positions: int,
    refresh: bool = False,
    first_seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare signals once, then run the same rules over many random seeds."""
    if runs <= 0:
        raise ValueError("runs must be greater than zero.")
    if max_positions <= 0:
        raise ValueError("max_positions must be greater than zero.")

    rolling = RollingUniverse.from_directory(rolling_universe_dir)
    rolling_tickers = rolling.all_eligible_tickers()
    if not rolling_tickers:
        raise ValueError("Rolling-universe snapshots contain no eligible tickers.")

    base_universe = load_universe()
    universe = base_universe.with_trade_universe(rolling_tickers)
    effective_end = end
    if effective_end is None:
        effective_end = rolling.recommended_end_exclusive().date().isoformat()
        print(
            "No --end supplied; using "
            f"{effective_end} (exclusive) from the latest rolling snapshot."
        )
    requested_end = cast(pd.Timestamp, pd.Timestamp(cast(Any, effective_end)))
    requested_through = cast(
        pd.Timestamp,
        pd.Timestamp(cast(Any, requested_end.date() - timedelta(days=1))),
    )
    rolling.validate_coverage(requested_through)
    prices = _load_or_download_prices(
        universe.data_tickers(),
        refresh=refresh,
        start=start,
        end=effective_end,
    )
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    if prices.empty:
        raise ValueError("No price rows are available for the requested period.")

    last_price_date = cast(pd.Timestamp, pd.Timestamp(cast(Any, prices["date"].max())))
    rolling.validate_coverage(last_price_date)
    available = set(prices["ticker"])
    missing = sorted(universe.approved_universe - available)
    if missing:
        print(
            "Warning: no price data for rolling-universe tickers: " + ", ".join(missing)
        )

    strategy_config = StrategyConfig()
    signals = add_strategy_signals(
        add_features(prices),
        universe,
        strategy_config,
        benchmark_ticker=BacktestConfig().benchmark_ticker,
    )

    default_backtest = BacktestConfig()
    target_fraction = min(
        default_backtest.target_position_fraction,
        1.0 / max_positions,
    )
    rows: list[dict[str, float | int]] = []
    candidate_logs: list[pd.DataFrame] = []
    trade_ledgers: list[pd.DataFrame] = []
    loss_recovery_logs: list[pd.DataFrame] = []
    for seed in range(first_seed, first_seed + runs):
        config = replace(
            default_backtest,
            max_positions=max_positions,
            target_position_fraction=target_fraction,
            random_seed=seed,
        )
        result = run_backtest(
            signals,
            universe,
            strategy_config,
            config,
            rolling,
        )
        metrics = calculate_metrics(
            result.equity_curve,
            result.trades,
            prices,
            starting_cash=config.starting_cash,
            benchmark_ticker=config.benchmark_ticker,
            loss_recovery_log=result.loss_recovery_log,
        )
        candidate_logs.append(result.entry_candidate_log)
        trade_ledgers.append(build_seed_trade_ledger(result.trades, seed))
        seed_recovery_log = result.loss_recovery_log.copy()
        seed_recovery_log.insert(0, "seed", seed)
        loss_recovery_logs.append(seed_recovery_log)
        rows.append(
            {
                "seed": seed,
                **{
                    name: _numeric_metric(metrics, name)
                    for name in RUN_COLUMNS
                    if name != "seed"
                },
            }
        )
        print(
            f"Completed random backtest {seed - first_seed + 1}/{runs} (seed={seed})."
        )

    run_table = pd.DataFrame(rows, columns=pd.Index(RUN_COLUMNS))
    loss_recovery_log = pd.concat(loss_recovery_logs, ignore_index=True)
    summary = summarize_random_runs(run_table, loss_recovery_log)
    trade_ledger = pd.concat(trade_ledgers, ignore_index=True)
    stock_performance = build_seed_stock_performance(trade_ledger)
    seed_rankings = build_seed_rankings(run_table, stock_performance)
    catastrophic_summary = build_catastrophic_stop_summary(loss_recovery_log)
    save_data(run_table, "random_backtest_runs.csv")
    save_data(summary, "random_backtest_summary.csv")
    save_data(seed_rankings, "random_backtest_seed_rankings.csv")
    save_data(stock_performance, "random_backtest_seed_stock_performance.csv")
    save_data(trade_ledger, "random_backtest_seed_trades.csv")
    save_data(
        catastrophic_summary,
        "random_backtest_catastrophic_stop_summary.csv",
    )
    save_data(
        loss_recovery_log,
        "random_backtest_loss_recovery_log.csv",
    )
    save_data(
        pd.concat(candidate_logs, ignore_index=True),
        "entry_candidate_log.csv",
    )
    return run_table, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--rolling-universe-dir",
        type=Path,
        default=DEFAULT_ROLLING_UNIVERSE_DIR,
    )
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download price data before preparing the shared signal dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_table, summary = run_random_backtests(
        start=args.start,
        end=args.end,
        rolling_universe_dir=args.rolling_universe_dir,
        runs=args.runs,
        max_positions=args.max_positions,
        refresh=args.refresh,
        first_seed=args.first_seed,
    )
    print(f"Saved {len(run_table)} runs to {DATA_DIR / 'random_backtest_runs.csv'}.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
