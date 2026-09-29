"""Market-level signals for entries, spikes, and exits."""

from __future__ import annotations

import pandas as pd
from config import DEFAULT_STRATEGY_CONFIG, StrategyConfig
from universe import Universe

SIGNAL_REQUIRED_COLUMNS = {
    "date",
    "ticker",
    "close",
    "daily_return",
    "return_3d",
    "return_4d",
    "weekly_return_5d",
    "momentum_30d",
    "momentum_90d",
    "volatility_30d",
    "avg_volume_30d",
    "relative_volume_30d",
    "ma_50d",
}


def _validate_columns(data: pd.DataFrame, required: set[str]) -> None:
    missing_columns = required.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Feature data is missing required columns: {missing}")


def add_strategy_signals(
    features: pd.DataFrame,
    universe: Universe,
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
    benchmark_ticker: str = "QQQ",
) -> pd.DataFrame:
    """Add deterministic entry, spike, and benchmark-relative exit signals."""
    _validate_columns(features, SIGNAL_REQUIRED_COLUMNS)
    data = features.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    data["is_approved"] = data["ticker"].isin(sorted(universe.approved_universe))

    data["entry_signal"] = (
        data["is_approved"]
        & data["avg_volume_30d"].ge(config.min_avg_volume_30d)
        & data["close"].gt(data["ma_50d"])
        & data["momentum_30d"].gt(0)
        & data["momentum_90d"].gt(0)
        & data["volatility_30d"].le(config.max_volatility_30d)
    )

    data["spike_1d"] = data["daily_return"].ge(config.spike_return_threshold)
    data["spike_3d"] = data["return_3d"].ge(config.spike_return_threshold)
    data["spike_4d"] = data["return_4d"].ge(config.spike_return_threshold)
    data["high_relative_volume"] = data["relative_volume_30d"].ge(
        config.high_relative_volume
    )
    data["spike_signal"] = (
        data[["spike_1d", "spike_3d", "spike_4d"]].any(axis=1)
        & data["high_relative_volume"]
    )

    normalized_benchmark = benchmark_ticker.upper()
    benchmark = data.loc[
        data["ticker"].eq(normalized_benchmark), ["date", "weekly_return_5d"]
    ].rename(columns={"weekly_return_5d": "qqq_weekly_return"})
    if benchmark.empty:
        raise ValueError(f"Benchmark data is missing for {normalized_benchmark}.")

    data = data.merge(benchmark, on="date", how="left", validate="many_to_one")
    data["relative_weekly_return"] = (
        data["weekly_return_5d"] - data["qqq_weekly_return"]
    )
    data["weekly_underperformance_signal"] = data["weekly_return_5d"].le(
        config.weekly_drop_threshold
    ) & data["relative_weekly_return"].le(config.weekly_relative_threshold)
    data["exit_signal"] = data["spike_signal"] | data["weekly_underperformance_signal"]
    return data.sort_values(["ticker", "date"]).reset_index(drop=True)
