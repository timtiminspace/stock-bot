"""Command-line entry point for the V1 swing-trading research pipeline."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
from backtester import BacktestResult, run_backtest
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
from metrics import MetricValue, calculate_metric_tables, calculate_metrics
from rolling_universe import RollingUniverse
from strategy import add_strategy_signals
from universe import Universe, load_universe

DEFAULT_ROLLING_UNIVERSE_DIR = DATA_DIR / "universe_snapshots"


def load_or_download_prices(
    universe: Universe,
    refresh: bool = False,
    start: str = DEFAULT_START_DATE,
    end: str | None = None,
) -> pd.DataFrame:
    """Load cached prices, or download the configured universe when requested."""
    prices_path = DATA_DIR / "prices.csv"
    if refresh or not prices_path.exists():
        raw_data = download_price_data(universe.data_tickers(), start=start, end=end)
        prices = clean_yfinance_data(raw_data)
    else:
        prices = load_data("prices.csv")

    prices = filter_price_dates(prices, start=start, end=end)
    save_data(prices, "prices.csv")
    return prices


def run_pipeline(
    refresh: bool = False,
    start: str = DEFAULT_START_DATE,
    end: str | None = None,
    rolling_universe_dir: Path = DEFAULT_ROLLING_UNIVERSE_DIR,
    random_seed: int = 42,
) -> tuple[BacktestResult, dict[str, MetricValue]]:
    """Build features/signals, backtest the strategy, and save all V1 outputs."""
    rolling_universe = RollingUniverse.from_directory(rolling_universe_dir)
    rolling_tickers = rolling_universe.all_eligible_tickers()
    if not rolling_tickers:
        raise ValueError(
            "Rolling-universe snapshots contain no eligible technology tickers. "
            "Rebuild them with config/company_evidence.csv before backtesting."
        )

    universe = load_universe().with_trade_universe(rolling_tickers)
    effective_end = end
    if effective_end is None:
        effective_end = rolling_universe.recommended_end_exclusive().date().isoformat()
        print(
            "No --end supplied; using "
            f"{effective_end} (exclusive) from the latest rolling snapshot."
        )
    requested_end = cast(pd.Timestamp, pd.Timestamp(cast(Any, effective_end)))
    requested_through = cast(
        pd.Timestamp,
        pd.Timestamp(cast(Any, requested_end.date() - timedelta(days=1))),
    )
    rolling_universe.validate_coverage(requested_through)

    prices = load_or_download_prices(
        universe,
        refresh=refresh,
        start=start,
        end=effective_end,
    )
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    last_price_date = cast(
        pd.Timestamp,
        pd.Timestamp(cast(Any, prices["date"].max())),
    )
    rolling_universe.validate_coverage(last_price_date)

    available = set(prices["ticker"])
    missing_trade_tickers = sorted(universe.approved_universe - available)
    if missing_trade_tickers:
        print(
            "Warning: no cached price data for rolling-universe tickers: "
            + ", ".join(missing_trade_tickers)
        )

    features = add_features(prices)
    save_data(features, "features.csv")

    strategy_config = StrategyConfig()
    backtest_config = replace(
        BacktestConfig(),
        random_seed=random_seed,
    )
    signals = add_strategy_signals(
        features,
        universe,
        strategy_config,
        benchmark_ticker=backtest_config.benchmark_ticker,
    )
    save_data(signals, "signals.csv")

    result = run_backtest(
        signals,
        universe,
        strategy_config,
        backtest_config,
        rolling_universe,
    )
    save_data(result.trades, "trades.csv")
    save_data(result.equity_curve, "equity_curve.csv")
    save_data(result.watchlist, "reentry_watchlist.csv")
    save_data(result.entry_candidate_log, "entry_candidate_log.csv")
    save_data(result.loss_recovery_log, "loss_recovery_log.csv")

    metrics = calculate_metrics(
        result.equity_curve,
        result.trades,
        prices,
        starting_cash=backtest_config.starting_cash,
        benchmark_ticker=backtest_config.benchmark_ticker,
        loss_recovery_log=result.loss_recovery_log,
    )
    metric_tables = calculate_metric_tables(
        result.equity_curve,
        result.trades,
        prices,
        starting_cash=backtest_config.starting_cash,
        benchmark_ticker=backtest_config.benchmark_ticker,
    )
    for name, table in metric_tables.items():
        save_data(table, f"{name}.csv")
    save_data(
        pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())}),
        "metrics_summary.csv",
    )
    return result, metrics


def print_metrics(metrics: dict[str, MetricValue]) -> None:
    """Print a stable, readable metrics summary."""
    percentage_metrics = {
        "total_return",
        "annualized_return",
        "max_drawdown",
        "win_rate",
        "average_win",
        "average_loss",
        "best_trade",
        "worst_trade",
        "cash_drag",
        "benchmark_return_QQQ",
        "strategy_vs_QQQ",
        "average_return_after_spike_exit",
        "reentry_success_rate",
        "runner_performance",
        "loss_recovery_exit_rate",
        "final_sidelined_portfolio_fraction",
    }
    print("\nV1 backtest metrics")
    for name, value in metrics.items():
        if isinstance(value, str):
            rendered = value
        elif name in percentage_metrics:
            rendered = f"{value:.2%}"
        else:
            rendered = f"{value:.2f}"
        print(f"{name}: {rendered}")


def print_drawdown_attribution(attribution: pd.DataFrame) -> None:
    """Print the largest ticker losses during the maximum drawdown window."""
    losers = attribution.loc[attribution["drawdown_period_pnl"].lt(0)].head(10)
    print("\nLargest contributors to maximum drawdown")
    if losers.empty:
        print("No losing ticker contribution was found.")
        return
    for _, row in losers.iterrows():
        print(
            f"{row['ticker']}: {float(row['drawdown_period_pnl']):.2f} PnL, "
            f"{float(row['contribution_to_portfolio_drawdown']):.2%} of peak equity"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download fresh prices instead of using data/prices.csv.",
    )
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--rolling-universe-dir",
        type=Path,
        default=DEFAULT_ROLLING_UNIVERSE_DIR,
        help=(
            "Dated point-in-time snapshots used for entry eligibility "
            "(default: data/universe_snapshots)."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Base seed used by deterministic random entry selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result, metrics = run_pipeline(
            refresh=args.refresh,
            start=args.start,
            end=args.end,
            rolling_universe_dir=args.rolling_universe_dir,
            random_seed=args.random_seed,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    print(
        f"Saved {len(result.trades)} trades and "
        f"{len(result.equity_curve)} daily equity rows under {DATA_DIR}."
    )
    print_metrics(metrics)
    print_drawdown_attribution(load_data("drawdown_attribution_by_ticker.csv"))


if __name__ == "__main__":
    main()
