"""Build and consume immutable point-in-time stock-universe snapshots."""

from __future__ import annotations

import argparse
import re
import time
from collections.abc import Iterable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd
from classification import (
    UNKNOWN_CATEGORY,
    UNKNOWN_CATEGORY_ALIASES,
    build_category_history,
    classify_records,
)
from providers.alpha_vantage_listing import AlphaVantageListingClient

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_DIR / "config"
DATA_DIR = PROJECT_DIR / "data"
RAW_LISTING_DIR = DATA_DIR / "raw" / "alpha_vantage"
SNAPSHOT_DIR = DATA_DIR / "universe_snapshots"
CATEGORY_HISTORY_PATH = DATA_DIR / "category_history.csv"
DEFAULT_EVIDENCE_PATH = CONFIG_DIR / "company_evidence.csv"

ALLOWED_EXCHANGES = {"NASDAQ", "NYSE", "AMEX"}
ALLOWED_POSITION_CLASSIFICATIONS = {"recovery_candidate", "compounder"}
DEFAULT_MIN_CLASSIFICATION_CONFIDENCE = 0.60
NON_COMMON_SECURITY_PATTERN = re.compile(
    r"\b(?:ETF|ETN|FUND|WARRANTS?|UNITS?|RIGHTS?|PREFERRED|PREFERENCE|NOTES?)\b",
    flags=re.IGNORECASE,
)

SNAPSHOT_BASE_COLUMNS = [
    "snapshot_date",
    "ticker",
    "name",
    "exchange",
    "asset_type",
    "status",
    "ipo_date",
    "delisting_date",
]


class ListingClient(Protocol):
    """The provider surface required by snapshot construction."""

    def listing_status(self, date: str, state: str = "active") -> pd.DataFrame: ...


def normalize_ticker(ticker: object) -> str:
    """Normalize provider symbols to the yfinance-compatible project convention."""
    return str(ticker).strip().upper().replace(".", "-")


def normalize_exchange(exchange: object) -> str:
    """Collapse common US exchange spellings into three stable identifiers."""
    value = re.sub(r"\s+", " ", str(exchange).strip().upper())
    aliases = {
        "NASDAQ GLOBAL SELECT": "NASDAQ",
        "NASDAQ GLOBAL MARKET": "NASDAQ",
        "NASDAQ CAPITAL MARKET": "NASDAQ",
        "NASDAQGS": "NASDAQ",
        "NASDAQGM": "NASDAQ",
        "NASDAQCM": "NASDAQ",
        "NEW YORK STOCK EXCHANGE": "NYSE",
        "NYSE MKT": "AMEX",
        "NYSE AMERICAN": "AMEX",
        "NYSEAMERICAN": "AMEX",
        "AMERICAN STOCK EXCHANGE": "AMEX",
    }
    return aliases.get(value, value)


def _snake_case_column(column: object) -> str:
    text = str(column).strip().replace("-", "_").replace(" ", "_")
    text = re.sub(r"(?<!^)(?=[A-Z])", "_", text).lower()
    return re.sub(r"_+", "_", text).strip("_")


def _normalized_listing_columns(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    normalized.columns = [_snake_case_column(column) for column in normalized.columns]
    aliases = {
        "symbol": "ticker",
        "company_name": "name",
        "exchange_short_name": "exchange",
        "assettype": "asset_type",
        "ipodate": "ipo_date",
        "delistingdate": "delisting_date",
    }
    for source, destination in aliases.items():
        if source in normalized.columns and destination not in normalized.columns:
            normalized = normalized.rename(columns={source: destination})
    return normalized


def _common_equity_mask(values: pd.Series) -> pd.Series:
    text = values.fillna("").astype(str).str.strip().str.lower()
    is_equity = text.str.contains(r"\b(?:stock|equity|common|adr)\b", regex=True)
    is_forbidden = text.str.contains(
        r"\b(?:etf|etn|fund|warrant|unit|right|preferred|note|bond)\b", regex=True
    )
    return is_equity & ~is_forbidden


def clean_listing_roster(
    roster: pd.DataFrame,
    snapshot_date: str,
    *,
    rejected_tickers: Iterable[str] = (),
    benchmark_tickers: Iterable[str] = (),
) -> pd.DataFrame:
    """Normalize a provider roster and retain common US exchange-listed equities."""
    data = _normalized_listing_columns(roster)
    if "ticker" not in data.columns:
        columns = ", ".join(data.columns)
        raise ValueError(f"Expected a symbol/ticker column. Received: {columns}")

    parsed_date = cast(pd.Timestamp, pd.Timestamp(cast(Any, snapshot_date)))
    if str(parsed_date) == "NaT":
        raise ValueError("snapshot_date must be a valid date.")

    data["ticker"] = data["ticker"].map(normalize_ticker)
    data = data.loc[data["ticker"].ne("")]

    if "asset_type" in data.columns:
        data = data.loc[_common_equity_mask(data["asset_type"])]

    if "exchange" in data.columns:
        data["exchange"] = data["exchange"].map(normalize_exchange)
        data = data.loc[data["exchange"].isin(sorted(ALLOWED_EXCHANGES))]

    if "name" in data.columns:
        names = data["name"].fillna("").astype(str)
        data = data.loc[~names.str.contains(NON_COMMON_SECURITY_PATTERN, na=False)]

    excluded = {
        normalize_ticker(ticker)
        for ticker in (*tuple(rejected_tickers), *tuple(benchmark_tickers))
    }
    data = data.loc[~data["ticker"].isin(excluded)]

    defaults: dict[str, object] = {
        "name": "",
        "exchange": "",
        "asset_type": "",
        "status": "",
        "ipo_date": "",
        "delisting_date": "",
    }
    for column, default in defaults.items():
        if column not in data.columns:
            data[column] = default

    exchange_rank = {"NASDAQ": 0, "NYSE": 1, "AMEX": 2}
    data["_exchange_rank"] = data["exchange"].map(exchange_rank).fillna(99)
    data = data.sort_values(["_exchange_rank", "ticker"])
    data = data.drop_duplicates(subset=["ticker"], keep="first")
    data = data.drop(columns=["_exchange_rank"])
    data.insert(0, "snapshot_date", parsed_date.date().isoformat())

    leading = [column for column in SNAPSHOT_BASE_COLUMNS if column in data.columns]
    trailing = [column for column in data.columns if column not in leading]
    return data[leading + trailing].reset_index(drop=True)


def _read_tickers(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = pd.read_csv(path)
    if "ticker" not in data.columns:
        raise ValueError(f"Universe file has no ticker column: {path}")
    return {normalize_ticker(ticker) for ticker in data["ticker"]}


def _bool_values(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    return (
        values.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y"})
    )


def _eligible_rows(snapshot: pd.DataFrame) -> pd.DataFrame:
    if "eligible" in snapshot.columns:
        return snapshot.loc[_bool_values(cast(pd.Series, snapshot["eligible"]))]
    if "primary_category" in snapshot.columns:
        categories = snapshot["primary_category"].fillna("").astype(str).str.lower()
        unknowns = {category.lower() for category in UNKNOWN_CATEGORY_ALIASES}
        return snapshot.loc[~categories.isin(unknowns) & categories.ne("")]
    raise ValueError(
        "Rolling-universe snapshot has no eligible or primary_category column."
    )


class RollingUniverse:
    """In-memory resolver for the latest universe snapshot on or before a date."""

    def __init__(self, snapshots: dict[pd.Timestamp, pd.DataFrame]) -> None:
        if not snapshots:
            raise ValueError("Rolling universe contains no dated snapshots.")
        self._snapshots = dict(sorted(snapshots.items()))
        self._dates = tuple(self._snapshots)

    @classmethod
    def from_directory(cls, directory: Path) -> RollingUniverse:
        """Load YYYY-MM-DD.csv snapshots from a directory."""
        if not directory.exists():
            raise FileNotFoundError(
                f"Rolling-universe directory does not exist: {directory}"
            )

        snapshots: dict[pd.Timestamp, pd.DataFrame] = {}
        for path in sorted(directory.glob("*.csv")):
            try:
                snapshot_date = cast(
                    pd.Timestamp, pd.Timestamp(cast(Any, path.stem))
                ).normalize()
            except ValueError:
                continue
            data = pd.read_csv(path)
            if "ticker" not in data.columns:
                raise ValueError(
                    f"Rolling-universe snapshot has no ticker column: {path}"
                )
            if not {"eligible", "primary_category"}.intersection(data.columns):
                raise ValueError(
                    f"Rolling-universe snapshot has no eligibility field: {path}"
                )
            data["ticker"] = data["ticker"].map(normalize_ticker)
            data = data.loc[data["ticker"].ne("")].copy()
            if bool(cast(Any, data["ticker"].duplicated().any())):
                raise ValueError(
                    f"Rolling-universe snapshot has duplicate tickers: {path}"
                )
            data["snapshot_date"] = snapshot_date
            snapshots[snapshot_date] = data.reset_index(drop=True)
        if not snapshots:
            raise ValueError(
                f"No YYYY-MM-DD.csv snapshots were found under {directory}."
            )
        return cls(snapshots)

    @property
    def snapshot_dates(self) -> tuple[pd.Timestamp, ...]:
        return self._dates

    @property
    def latest_snapshot_date(self) -> pd.Timestamp:
        """Return the newest immutable snapshot in the loaded series."""
        return self._dates[-1]

    def recommended_end_exclusive(self) -> pd.Timestamp:
        """Return the exclusive end after the month following the latest snapshot."""
        following_month_end = cast(
            pd.Timestamp,
            self.latest_snapshot_date + pd.offsets.MonthEnd(1),
        )
        return cast(
            pd.Timestamp,
            pd.Timestamp(cast(Any, following_month_end.date() + timedelta(days=1))),
        )

    def snapshot_date_for(self, date: str | pd.Timestamp) -> pd.Timestamp | None:
        """Return the newest snapshot date that cannot look into the future."""
        requested = cast(pd.Timestamp, pd.Timestamp(cast(Any, date))).normalize()
        available = [
            snapshot_date for snapshot_date in self._dates if snapshot_date <= requested
        ]
        return available[-1] if available else None

    def snapshot_for(self, date: str | pd.Timestamp) -> pd.DataFrame:
        """Return a defensive copy of the point-in-time snapshot for a date."""
        snapshot_date = self.snapshot_date_for(date)
        if snapshot_date is None:
            return pd.DataFrame(columns=pd.Index(["snapshot_date", "ticker"]))
        return self._snapshots[snapshot_date].copy()

    def eligible_tickers(self, date: str | pd.Timestamp) -> frozenset[str]:
        """Return tickers explicitly eligible in the latest non-future snapshot."""
        snapshot = self.snapshot_for(date)
        if snapshot.empty:
            return frozenset()
        eligible = _eligible_rows(snapshot)
        return frozenset(eligible["ticker"].astype(str))

    def all_eligible_tickers(self) -> frozenset[str]:
        """Return the historical superset required for price downloads/signals."""
        tickers: set[str] = set()
        for snapshot_date in self._dates:
            tickers.update(self.eligible_tickers(snapshot_date))
        return frozenset(tickers)

    def position_classification_for(
        self,
        ticker: str,
        date: str | pd.Timestamp,
    ) -> str:
        """Return the non-future position classification for an eligible ticker."""
        snapshot = self.snapshot_for(date)
        eligible = _eligible_rows(snapshot)
        normalized = normalize_ticker(ticker)
        rows = eligible.loc[eligible["ticker"].eq(normalized)]
        if rows.empty:
            raise KeyError(
                f"Ticker {normalized} is not eligible in the snapshot for {date}."
            )
        column = next(
            (
                name
                for name in ("position_classification", "classification")
                if name in rows.columns
            ),
            None,
        )
        value = "compounder" if column is None else str(rows.iloc[-1][column]).strip()
        return value if value in ALLOWED_POSITION_CLASSIFICATIONS else "compounder"

    def validate_coverage(
        self,
        through_date: str | pd.Timestamp,
        *,
        max_snapshot_age_days: int = 40,
    ) -> None:
        """Reject a snapshot series that becomes stale before the backtest ends."""
        if max_snapshot_age_days < 0:
            raise ValueError("max_snapshot_age_days cannot be negative.")
        requested = cast(
            pd.Timestamp, pd.Timestamp(cast(Any, through_date))
        ).normalize()
        latest = self.snapshot_date_for(requested)
        if latest is None:
            raise ValueError(
                f"Rolling universe has no snapshot on or before {requested.date()}."
            )
        age_days = int((requested - latest).days)
        if age_days > max_snapshot_age_days:
            raise ValueError(
                "Rolling-universe coverage is incomplete: latest usable snapshot "
                f"is {latest.date()}, but price data continues through "
                f"{requested.date()} ({age_days} days stale). Finish or resume the "
                "snapshot build before backtesting."
            )


def _attach_classification(
    roster: pd.DataFrame,
    evidence: pd.DataFrame | None,
    *,
    snapshot_date: str,
    include_unknown_technology: bool,
    min_classification_confidence: float,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not 0.0 <= min_classification_confidence <= 1.0:
        raise ValueError("min_classification_confidence must be between 0 and 1.")
    if evidence is None:
        data = roster.copy()
        data["primary_category"] = UNKNOWN_CATEGORY
        data["secondary_categories"] = ""
        data["classification_confidence"] = 0.0
        data["classifier_version"] = "unclassified"
        data["eligible"] = include_unknown_technology
        data["position_classification"] = "compounder"
        return data, None

    selected_evidence = evidence.copy()
    selected_evidence["ticker"] = selected_evidence["ticker"].map(normalize_ticker)
    if "valid_from" in selected_evidence.columns:
        as_of = pd.Timestamp(snapshot_date)
        selected_evidence["_valid_from"] = pd.to_datetime(
            selected_evidence["valid_from"], errors="coerce"
        )
        if "valid_to" in selected_evidence.columns:
            selected_evidence["_valid_to"] = pd.to_datetime(
                selected_evidence["valid_to"], errors="coerce"
            )
        else:
            selected_evidence["_valid_to"] = pd.NaT
        selected_evidence = selected_evidence.loc[
            selected_evidence["_valid_from"].notna()
            & selected_evidence["_valid_from"].le(as_of)
            & (
                selected_evidence["_valid_to"].isna()
                | selected_evidence["_valid_to"].ge(as_of)
            )
        ]
        selected_evidence = selected_evidence.sort_values("_valid_from")
        selected_evidence = selected_evidence.drop_duplicates(
            subset=["ticker"], keep="last"
        ).drop(columns=["_valid_from", "_valid_to"])

    classified = classify_records(selected_evidence)
    if classified["ticker"].duplicated().any():
        raise ValueError("Classification evidence contains duplicate tickers.")

    enrichment_columns = [
        "ticker",
        "primary_category",
        "secondary_categories",
        "classification_confidence",
        "classifier_version",
    ]
    position_source = next(
        (
            column
            for column in ("position_classification", "classification")
            if column in classified.columns
        ),
        None,
    )
    if position_source is not None:
        enrichment_columns.append(position_source)

    data = roster.merge(
        classified[enrichment_columns],
        on="ticker",
        how="left",
        validate="one_to_one",
    )
    data["primary_category"] = data["primary_category"].fillna(UNKNOWN_CATEGORY)
    data["secondary_categories"] = data["secondary_categories"].fillna("")
    numeric_confidence = cast(
        pd.Series,
        pd.to_numeric(
            cast(pd.Series, data["classification_confidence"]), errors="coerce"
        ),
    )
    data["classification_confidence"] = numeric_confidence.astype("Float64")
    data["classification_confidence"] = data["classification_confidence"].fillna(0.0)
    data["classifier_version"] = data["classifier_version"].fillna("unclassified")
    normalized_categories = data["primary_category"].astype(str).str.lower()
    unknown_categories = {category.lower() for category in UNKNOWN_CATEGORY_ALIASES}
    data["eligible"] = include_unknown_technology | (
        ~normalized_categories.isin(unknown_categories)
        & data["classification_confidence"].ge(min_classification_confidence)
    )

    if position_source is None:
        data["position_classification"] = "compounder"
    else:
        data["position_classification"] = data[position_source].fillna("compounder")
        if position_source != "position_classification":
            data = data.drop(columns=[position_source])
    data.loc[
        ~data["position_classification"].isin(sorted(ALLOWED_POSITION_CLASSIFICATIONS)),
        "position_classification",
    ] = "compounder"
    return data, classified


def build_listing_snapshot(
    snapshot_date: str,
    *,
    client: ListingClient | None = None,
    snapshot_dir: Path = SNAPSHOT_DIR,
    raw_dir: Path = RAW_LISTING_DIR,
    rejected_tickers: Iterable[str] = (),
    benchmark_tickers: Iterable[str] = (),
    evidence: pd.DataFrame | None = None,
    category_history_path: Path | None = CATEGORY_HISTORY_PATH,
    include_unknown_technology: bool = False,
    min_classification_confidence: float = DEFAULT_MIN_CLASSIFICATION_CONFIDENCE,
    overwrite: bool = False,
    refresh_listing_data: bool = False,
) -> pd.DataFrame:
    """Fetch, classify, and save one immutable point-in-time universe snapshot."""
    if evidence is not None:
        if "ticker" not in evidence.columns:
            raise ValueError("Classification evidence must contain a ticker column.")
        if evidence.empty:
            raise ValueError(
                "Classification evidence contains no company rows. Populate the "
                "evidence CSV before building eligible snapshots; no provider "
                "request was made."
            )
    parsed_date = cast(pd.Timestamp, pd.Timestamp(cast(Any, snapshot_date)))
    if str(parsed_date) == "NaT":
        raise ValueError("snapshot_date must be a valid date.")
    normalized_date = parsed_date.date().isoformat()
    output_path = snapshot_dir / f"{normalized_date}.csv"
    if output_path.exists() and not overwrite:
        return pd.read_csv(output_path, parse_dates=["snapshot_date"])

    raw_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"listing_status_{normalized_date}_active.csv"
    if raw_path.exists() and not refresh_listing_data:
        raw = pd.read_csv(raw_path)
    else:
        listing_client = client or AlphaVantageListingClient()
        raw = listing_client.listing_status(normalized_date, state="active")
        raw.to_csv(raw_path, index=False)

    clean = clean_listing_roster(
        raw,
        normalized_date,
        rejected_tickers=rejected_tickers,
        benchmark_tickers=benchmark_tickers,
    )
    snapshot, classified = _attach_classification(
        clean,
        evidence,
        snapshot_date=normalized_date,
        include_unknown_technology=include_unknown_technology,
        min_classification_confidence=min_classification_confidence,
    )
    snapshot.to_csv(output_path, index=False)

    if (
        classified is not None
        and evidence is not None
        and category_history_path is not None
    ):
        history = build_category_history(classified, valid_from=normalized_date)
        category_history_path.parent.mkdir(parents=True, exist_ok=True)
        if category_history_path.exists():
            existing = pd.read_csv(category_history_path)
            history = pd.concat([existing, history], ignore_index=True)
            history = history.drop_duplicates(
                subset=["ticker", "valid_from", "classifier_version"], keep="last"
            )
        history.to_csv(category_history_path, index=False)

    eligible_count = int(_bool_values(cast(pd.Series, snapshot["eligible"])).sum())
    print(
        f"Saved {len(snapshot)} listed equities ({eligible_count} technology-eligible) "
        f"to {output_path}."
    )
    return snapshot


def month_ends(start: str, end: str) -> list[str]:
    """Return inclusive calendar month-end dates for a requested range."""
    start_date = cast(pd.Timestamp, pd.Timestamp(cast(Any, start)))
    end_date = cast(pd.Timestamp, pd.Timestamp(cast(Any, end)))
    if str(start_date) == "NaT" or str(end_date) == "NaT":
        raise ValueError("start and end must be valid dates.")
    if end_date < start_date:
        raise ValueError("end must be on or after start.")
    first_month_end = start_date + pd.offsets.MonthEnd(0)
    dates: list[str] = []
    current = cast(pd.Timestamp, first_month_end)
    while current <= end_date:
        dates.append(current.date().isoformat())
        current = cast(pd.Timestamp, current + pd.offsets.MonthEnd(1))
    return dates


def build_monthly_snapshots(
    start: str,
    end: str,
    *,
    client: ListingClient | None = None,
    snapshot_dir: Path = SNAPSHOT_DIR,
    raw_dir: Path = RAW_LISTING_DIR,
    rejected_tickers: Iterable[str] = (),
    benchmark_tickers: Iterable[str] = (),
    evidence: pd.DataFrame | None = None,
    category_history_path: Path | None = CATEGORY_HISTORY_PATH,
    include_unknown_technology: bool = False,
    min_classification_confidence: float = DEFAULT_MIN_CLASSIFICATION_CONFIDENCE,
    overwrite: bool = False,
    refresh_listing_data: bool = False,
    request_delay_seconds: float = 0.0,
) -> list[Path]:
    """Build a cached month-end snapshot series, stopping on the first gap."""
    if request_delay_seconds < 0:
        raise ValueError("request_delay_seconds cannot be negative.")
    built: list[Path] = []
    dates = month_ends(start, end)
    for index, snapshot_date in enumerate(dates):
        path = snapshot_dir / f"{snapshot_date}.csv"
        existed = path.exists()
        build_listing_snapshot(
            snapshot_date,
            client=client,
            snapshot_dir=snapshot_dir,
            raw_dir=raw_dir,
            rejected_tickers=rejected_tickers,
            benchmark_tickers=benchmark_tickers,
            evidence=evidence,
            category_history_path=category_history_path,
            include_unknown_technology=include_unknown_technology,
            min_classification_confidence=min_classification_confidence,
            overwrite=overwrite,
            refresh_listing_data=refresh_listing_data,
        )
        built.append(path)
        if not existed and request_delay_seconds and index < len(dates) - 1:
            time.sleep(request_delay_seconds)
    return built


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse snapshot-builder command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--snapshot-date")
    mode.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_LISTING_DIR)
    parser.add_argument(
        "--evidence-csv",
        type=Path,
        default=DEFAULT_EVIDENCE_PATH,
        help="Effective-dated technology evidence (default: config/company_evidence.csv).",
    )
    parser.add_argument("--category-history", type=Path, default=CATEGORY_HISTORY_PATH)
    parser.add_argument("--include-unknown-technology", action="store_true")
    parser.add_argument(
        "--min-classification-confidence",
        type=float,
        default=DEFAULT_MIN_CLASSIFICATION_CONFIDENCE,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--refresh-listings",
        action="store_true",
        help="Redownload raw listing CSVs instead of reusing the local cache.",
    )
    parser.add_argument("--request-delay", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.start and not args.end:
        parser.error("--end is required when --start is used.")
    if args.end and not args.start:
        parser.error("--end requires --start.")
    if args.refresh_listings and not args.overwrite:
        parser.error("--refresh-listings requires --overwrite.")
    return args


def main() -> None:
    """Build one or many point-in-time listing snapshots."""
    args = parse_args()
    if not args.evidence_csv.exists():
        raise FileNotFoundError(
            f"Technology evidence file does not exist: {args.evidence_csv}. "
            "Build it with tech_universe_builder.py first."
        )
    evidence = pd.read_csv(args.evidence_csv)
    if args.start and "valid_from" not in evidence.columns:
        raise ValueError(
            "Historical snapshot ranges require a valid_from column in the evidence "
            "CSV so future descriptions cannot leak into earlier classifications."
        )
    rejected_tickers = _read_tickers(CONFIG_DIR / "rejected_universe.csv")
    benchmark_tickers = _read_tickers(CONFIG_DIR / "benchmark_universe.csv")
    if args.snapshot_date:
        build_listing_snapshot(
            args.snapshot_date,
            snapshot_dir=args.snapshot_dir,
            raw_dir=args.raw_dir,
            rejected_tickers=rejected_tickers,
            benchmark_tickers=benchmark_tickers,
            evidence=evidence,
            category_history_path=args.category_history,
            include_unknown_technology=args.include_unknown_technology,
            min_classification_confidence=args.min_classification_confidence,
            overwrite=args.overwrite,
            refresh_listing_data=args.refresh_listings,
        )
    else:
        build_monthly_snapshots(
            args.start,
            args.end,
            snapshot_dir=args.snapshot_dir,
            raw_dir=args.raw_dir,
            rejected_tickers=rejected_tickers,
            benchmark_tickers=benchmark_tickers,
            evidence=evidence,
            category_history_path=args.category_history,
            include_unknown_technology=args.include_unknown_technology,
            min_classification_confidence=args.min_classification_confidence,
            overwrite=args.overwrite,
            refresh_listing_data=args.refresh_listings,
            request_delay_seconds=args.request_delay,
        )


if __name__ == "__main__":
    main()
