"""Download, normalize, save, and load daily market data."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_START_DATE = "2023-01-03"
EARLIEST_START_DATE = "2023-01-01"


def _validate_start_date(start: str) -> pd.Timestamp:
    requested_start = cast(pd.Timestamp, pd.Timestamp(cast(Any, start)))
    earliest_start = cast(pd.Timestamp, pd.Timestamp(cast(Any, EARLIEST_START_DATE)))
    first_trading_day = cast(pd.Timestamp, pd.Timestamp(cast(Any, DEFAULT_START_DATE)))
    if requested_start < earliest_start:
        raise ValueError(f"Price history cannot start before {earliest_start.date()}.")
    return max(requested_start, first_trading_day)


def download_price_data(
    tickers: Sequence[str],
    start: str = DEFAULT_START_DATE,
    end: str | None = None,
) -> pd.DataFrame:
    """Download unmodified daily data for one or more tickers from Yahoo Finance."""
    start_date = _validate_start_date(start)
    symbols = sorted(
        {str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()}
    )
    if not symbols:
        raise ValueError("At least one ticker is required for a price download.")
    raw_data = yf.download(
        symbols,
        start=start_date.date().isoformat(),
        end=end,
        interval="1d",
        auto_adjust=False,
        group_by="column",
        multi_level_index=True,
        progress=False,
    )
    if raw_data is None or raw_data.empty:
        raise RuntimeError("Yahoo Finance returned no price data.")

    return raw_data


def filter_price_dates(
    prices: pd.DataFrame,
    start: str = DEFAULT_START_DATE,
    end: str | None = None,
) -> pd.DataFrame:
    """Restrict prices to the configured inclusive start and exclusive end dates."""
    required_columns = {"date", "ticker"}
    missing = required_columns.difference(prices.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Price data is missing columns: {names}")

    start_date = _validate_start_date(start)
    end_date = None if end is None else cast(pd.Timestamp, pd.Timestamp(cast(Any, end)))
    if end_date is not None and end_date <= start_date:
        raise ValueError("The end date must be later than the start date.")

    data = prices.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.loc[data["date"].ge(start_date)]

    if end_date is not None:
        data = data.loc[data["date"].lt(end_date)]

    if data.empty:
        raise ValueError(f"No price data is available on or after {start_date.date()}.")

    return data.sort_values(["ticker", "date"]).reset_index(drop=True)


def clean_yfinance_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Convert yfinance's wide ticker columns to sorted long-form OHLCV rows."""
    if not isinstance(raw_data.columns, pd.MultiIndex):
        raise TypeError("Expected MultiIndex columns from a multi-ticker download.")

    ticker_level: str | int = (
        "Ticker" if "Ticker" in raw_data.columns.names else raw_data.columns.nlevels - 1
    )
    data = (
        raw_data.stack(level=ticker_level, future_stack=True)
        .rename_axis(["date", "ticker"])
        .reset_index()
    )
    data.columns = data.columns.str.lower()

    required = ["date", "ticker", "open", "high", "low", "close", "volume"]
    missing = set(required).difference(data.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Downloaded price data is missing columns: {names}")

    clean_data = data.loc[:, required].copy()
    return clean_data.sort_values(by=["ticker", "date"]).reset_index(drop=True)


def save_data(df: pd.DataFrame, filename: str) -> None:
    """Save a generated DataFrame under the project data directory."""
    df.to_csv(DATA_DIR / filename, index=False)


def load_data(filename: str) -> pd.DataFrame:
    """Load a generated CSV and parse its date columns when present."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Data file does not exist: {path}")

    header = pd.read_csv(path, nrows=0)
    date_columns = [
        column
        for column in (
            "date",
            "signal_date",
            "execution_date",
            "exit_date",
            "watch_start_date",
            "watch_end_date",
            "universe_snapshot_date",
        )
        if column in header.columns
    ]

    return pd.read_csv(path, parse_dates=date_columns)
