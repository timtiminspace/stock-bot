"""Feature engineering for clean daily OHLCV price data."""

import pandas as pd

REQUIRED_PRICE_COLUMNS = {"date", "ticker", "open", "high", "low", "close", "volume"}


def prepare_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, copy, and sort daily OHLCV data by ticker and date."""
    missing_columns = REQUIRED_PRICE_COLUMNS.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Price data is missing required columns: {missing}")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["ticker"] = data["ticker"].astype(str).str.upper()

    if data.duplicated(["ticker", "date"]).any():
        raise ValueError("Price data contains duplicate ticker/date rows.")
    ohlcv = data[["open", "high", "low", "close", "volume"]]
    if bool(ohlcv.isna().to_numpy().any()):
        raise ValueError("Price data contains missing OHLCV values.")

    return data.sort_values(["ticker", "date"]).reset_index(drop=True)


def _add_close_return(
    df: pd.DataFrame,
    *,
    column: str,
    periods: int,
) -> pd.DataFrame:
    """Add close-to-close return over a fixed number of trading rows."""
    data = prepare_price_data(df)
    previous_close = data.groupby("ticker", sort=False)["close"].shift(periods)
    data[column] = data["close"] / previous_close - 1
    return data


def add_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add each ticker's day-over-day closing-price return."""
    return _add_close_return(df, column="daily_return", periods=1)


def add_return_3d(df: pd.DataFrame) -> pd.DataFrame:
    """Add each ticker's three-trading-day closing-price return."""
    return _add_close_return(df, column="return_3d", periods=3)


def add_return_4d(df: pd.DataFrame) -> pd.DataFrame:
    """Add each ticker's four-trading-day closing-price return."""
    return _add_close_return(df, column="return_4d", periods=4)


def add_weekly_return_5d(df: pd.DataFrame) -> pd.DataFrame:
    """Add each ticker's five-trading-day closing-price return."""
    return _add_close_return(df, column="weekly_return_5d", periods=5)


def add_momentum_30d(df: pd.DataFrame) -> pd.DataFrame:
    """Add each ticker's 30-trading-day closing-price return."""
    return _add_close_return(df, column="momentum_30d", periods=30)


def add_momentum_90d(df: pd.DataFrame) -> pd.DataFrame:
    """Add each ticker's 90-trading-day closing-price return."""
    return _add_close_return(df, column="momentum_90d", periods=90)


def add_volatility_30d(df: pd.DataFrame) -> pd.DataFrame:
    """Add the rolling 30-day standard deviation of daily returns."""
    data = df.copy() if "daily_return" in df.columns else add_daily_returns(df)
    data = prepare_price_data(data)
    data["volatility_30d"] = data.groupby("ticker", sort=False)[
        "daily_return"
    ].transform(lambda returns: returns.rolling(window=30, min_periods=30).std())
    return data


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trailing volume averages and today's volume relative to that history."""
    data = prepare_price_data(df)
    data["avg_volume_30d"] = data.groupby("ticker", sort=False)["volume"].transform(
        lambda volume: volume.shift(1).rolling(window=30, min_periods=30).mean()
    )
    data["relative_volume_30d"] = data["volume"] / data["avg_volume_30d"]
    return data


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add 50- and 200-day moving averages and price-above-average flags."""
    data = prepare_price_data(df)
    grouped_close = data.groupby("ticker", sort=False)["close"]
    data["ma_50d"] = grouped_close.transform(
        lambda close: close.rolling(window=50, min_periods=50).mean()
    )
    data["ma_200d"] = grouped_close.transform(
        lambda close: close.rolling(window=200, min_periods=200).mean()
    )
    data["above_ma_50d"] = data["close"] > data["ma_50d"]
    data["above_ma_200d"] = data["close"] > data["ma_200d"]
    return data


def add_down_from_start(df: pd.DataFrame) -> pd.DataFrame:
    """Add return from the first closing price available for each ticker."""
    data = prepare_price_data(df)
    first_close = data.groupby("ticker", sort=False)["close"].transform("first")
    data["down_from_start"] = data["close"] / first_close - 1
    return data


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build every market feature required by the V1 strategy."""
    data = add_daily_returns(df)
    data = add_return_3d(data)
    data = add_return_4d(data)
    data = add_weekly_return_5d(data)
    data = add_momentum_30d(data)
    data = add_momentum_90d(data)
    data = add_volatility_30d(data)
    data = add_volume_features(data)
    data = add_moving_averages(data)
    return add_down_from_start(data)
