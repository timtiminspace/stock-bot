"""Build an approximate technology seed universe from reproducible CSV sources."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
from classification import UNKNOWN_CATEGORY_ALIASES, classify_record
from rolling_universe import normalize_ticker

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_DIR / "config"
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_SEED_SOURCE_DIR = DATA_DIR / "seed_sources"
DEFAULT_SNAPSHOT_DIR = DATA_DIR / "universe_snapshots"
DEFAULT_CACHE_PATH = DATA_DIR / "cache" / "yfinance_company_info.csv"
DEFAULT_OUTPUT_PATH = CONFIG_DIR / "company_evidence.csv"

OUTPUT_COLUMNS = [
    "ticker",
    "company_name",
    "primary_category",
    "secondary_categories",
    "confidence",
    "evidence_source",
    "source_tags",
    "valid_from",
    "valid_to",
    "position_classification",
]

CACHE_COLUMNS = [
    "ticker",
    "company_name",
    "sector",
    "industry",
    "long_business_summary",
    "fetched_on",
    "fetch_error",
]

SYMBOL_COLUMNS = ("ticker", "symbol", "holding", "name")
COMPANY_NAME_COLUMNS = ("company_name", "company", "security_name", "name")
INVALID_SYMBOLS = {"", "CASH", "USD", "US DOLLAR", "NAN", "NONE", "TOTAL"}
ALLOWED_POSITION_CLASSIFICATIONS = {"recovery_candidate", "compounder"}


@dataclass
class SeedCompany:
    """Evidence gathered for one normalized symbol before classification."""

    ticker: str
    company_name: str = ""
    position_classification: str = "compounder"
    source_tags: set[str] = field(default_factory=set)
    source_text: list[str] = field(default_factory=list)


def _normalized_column_lookup(data: pd.DataFrame) -> dict[str, str]:
    """Map accepted lowercase headings to their original string column names."""
    lookup: dict[str, str] = {}
    for column in data.columns:
        if not isinstance(column, str):
            raise TypeError("Seed CSV column names must be text values.")
        lookup[column.strip().lower()] = column
    return lookup


def _symbol_column(data: pd.DataFrame) -> str:
    lookup = _normalized_column_lookup(data)
    for name in SYMBOL_COLUMNS:
        if name in lookup:
            return lookup[name]
    expected = ", ".join(SYMBOL_COLUMNS)
    raise ValueError(f"Seed CSV needs one of these symbol columns: {expected}.")


def _company_name_column(data: pd.DataFrame, symbol_column: str) -> str | None:
    lookup = _normalized_column_lookup(data)
    for name in COMPANY_NAME_COLUMNS:
        column = lookup.get(name)
        if column is not None and column != symbol_column:
            return column
    return None


def _clean_ticker(value: object) -> str:
    ticker = normalize_ticker(value)
    if ticker in INVALID_SYMBOLS:
        return ""
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9-]{0,14}", ticker):
        return ""
    return ticker


def _clean_text(value: object) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _ingest_seed_source(
    companies: dict[str, SeedCompany],
    data: pd.DataFrame,
    *,
    source_tag: str,
    extra_text_columns: tuple[str, ...] = (),
) -> None:
    """Merge one source table into the deduplicated ticker evidence mapping."""
    symbol_column = _symbol_column(data)
    name_column = _company_name_column(data, symbol_column)
    lookup = _normalized_column_lookup(data)
    text_columns = [lookup[name] for name in extra_text_columns if name in lookup]
    classification_column = next(
        (
            lookup[name]
            for name in ("position_classification", "classification")
            if name in lookup
        ),
        None,
    )

    for _, row in data.iterrows():
        ticker = _clean_ticker(row[symbol_column])
        if not ticker:
            continue
        company = companies.setdefault(ticker, SeedCompany(ticker=ticker))
        company.source_tags.add(source_tag)
        if name_column is not None:
            candidate_name = _clean_text(row[name_column])
            if candidate_name and not company.company_name:
                company.company_name = candidate_name
        if classification_column is not None:
            classification = _clean_text(row[classification_column]).lower()
            if classification in ALLOWED_POSITION_CLASSIFICATIONS:
                company.position_classification = classification
        for column in text_columns:
            text = _clean_text(row[column])
            if text:
                company.source_text.append(text)


def collect_seed_companies(
    *,
    manual_universe_path: Path,
    seed_source_dir: Path,
) -> dict[str, SeedCompany]:
    """Read the manual seed universe and every local ETF/source CSV."""
    companies: dict[str, SeedCompany] = {}
    if manual_universe_path.exists():
        _ingest_seed_source(
            companies,
            pd.read_csv(manual_universe_path),
            source_tag="manual:tech_seed_universe",
            extra_text_columns=("theme", "sector", "industry"),
        )

    if seed_source_dir.exists():
        for path in sorted(seed_source_dir.glob("*.csv")):
            _ingest_seed_source(
                companies,
                pd.read_csv(path),
                source_tag=f"seed:{path.stem}",
                extra_text_columns=("theme", "sector", "industry", "category"),
            )
    return companies


def load_listed_tickers(
    snapshot_dir: Path,
    as_of: str,
) -> tuple[set[str] | None, Path | None]:
    """Load the latest Alpha Vantage roster snapshot on or before an as-of date."""
    requested = cast(pd.Timestamp, pd.Timestamp(cast(Any, as_of))).normalize()
    matches: list[tuple[pd.Timestamp, Path]] = []
    if snapshot_dir.exists():
        for path in snapshot_dir.glob("*.csv"):
            try:
                snapshot_date = cast(
                    pd.Timestamp, pd.Timestamp(cast(Any, path.stem))
                ).normalize()
            except ValueError:
                continue
            if snapshot_date <= requested:
                matches.append((snapshot_date, path))
    if not matches:
        return None, None

    _, selected_path = max(matches, key=lambda item: item[0])
    snapshot = pd.read_csv(selected_path)
    if "ticker" not in snapshot.columns:
        raise ValueError(f"Listing snapshot has no ticker column: {selected_path}")
    tickers = {_clean_ticker(value) for value in snapshot["ticker"]}
    return {ticker for ticker in tickers if ticker}, selected_path


def _empty_cache() -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index(CACHE_COLUMNS))


def load_yfinance_cache(cache_path: Path) -> pd.DataFrame:
    """Load previously fetched company metadata without making a network call."""
    if not cache_path.exists():
        return _empty_cache()
    cache = pd.read_csv(cache_path).reindex(columns=CACHE_COLUMNS)
    cache["ticker"] = cache["ticker"].map(_clean_ticker)
    return cache.loc[cache["ticker"].ne("")].drop_duplicates("ticker", keep="last")


def enrich_yfinance_metadata(
    tickers: list[str],
    *,
    cache_path: Path,
    fetch_limit: int,
) -> pd.DataFrame:
    """Fetch a capped set of uncached yfinance profiles and persist each result."""
    if fetch_limit < 0:
        raise ValueError("fetch_limit cannot be negative.")
    cache = load_yfinance_cache(cache_path)
    cached_tickers = set(cache["ticker"].astype(str))
    missing = [ticker for ticker in tickers if ticker not in cached_tickers]

    if fetch_limit > 0:
        import yfinance as yf

        rows: list[dict[str, str]] = []
        for ticker in missing[:fetch_limit]:
            try:
                info = cast(dict[str, Any], yf.Ticker(ticker).get_info())
                rows.append(
                    {
                        "ticker": ticker,
                        "company_name": str(
                            info.get("longName") or info.get("shortName") or ""
                        ).strip(),
                        "sector": str(info.get("sector") or "").strip(),
                        "industry": str(info.get("industry") or "").strip(),
                        "long_business_summary": str(
                            info.get("longBusinessSummary") or ""
                        ).strip(),
                        "fetched_on": datetime.now(UTC).date().isoformat(),
                        "fetch_error": "",
                    }
                )
            except Exception as error:  # noqa: BLE001 - provider errors vary by version.
                rows.append(
                    {
                        "ticker": ticker,
                        "company_name": "",
                        "sector": "",
                        "industry": "",
                        "long_business_summary": "",
                        "fetched_on": datetime.now(UTC).date().isoformat(),
                        "fetch_error": type(error).__name__,
                    }
                )

            cache = pd.concat([cache, pd.DataFrame([rows[-1]])], ignore_index=True)
            cache = cache.drop_duplicates("ticker", keep="last")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache.to_csv(cache_path, index=False)
    return cache.reset_index(drop=True)


def classify_seed_company(
    company: SeedCompany,
    profile: dict[str, object] | None = None,
) -> tuple[str, str, float, str]:
    """Classify broad technology labels from local source tags and cached metadata."""
    profile = profile or {}
    classification = classify_record(
        {
            "name": company.company_name,
            "sector": _clean_text(profile.get("sector")),
            "industry": " ".join(
                [*company.source_text, _clean_text(profile.get("industry"))]
            ),
            "description": " ".join(
                [
                    *sorted(company.source_tags),
                    _clean_text(profile.get("long_business_summary")),
                ]
            ),
        }
    )
    primary = str(classification["primary_category"])
    secondary = str(classification["secondary_categories"])
    confidence = float(cast(Any, classification["classification_confidence"]))

    profile_name = _clean_text(profile.get("company_name"))
    company_name = profile_name or company.company_name or company.ticker
    return primary, secondary, confidence, company_name


def build_tech_seed_universe(
    *,
    manual_universe_path: Path = CONFIG_DIR / "tech_seed_universe.csv",
    seed_source_dir: Path = DEFAULT_SEED_SOURCE_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    valid_from: str = "2023-01-03",
    listed_tickers: set[str] | None = None,
    profiles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build and save deduplicated approximate technology classification evidence."""
    effective_date = cast(pd.Timestamp, pd.Timestamp(cast(Any, valid_from)))
    if str(effective_date) == "NaT":
        raise ValueError("valid_from must be a valid date.")

    companies = collect_seed_companies(
        manual_universe_path=manual_universe_path,
        seed_source_dir=seed_source_dir,
    )
    if not companies:
        raise ValueError(
            "No technology seed tickers were found in the manual universe or "
            "seed-source CSVs."
        )
    if listed_tickers is not None:
        normalized_listed = {_clean_ticker(ticker) for ticker in listed_tickers}
        companies = {
            ticker: company
            for ticker, company in companies.items()
            if ticker in normalized_listed
        }

    profile_records: dict[str, dict[str, object]] = {}
    if profiles is not None and not profiles.empty:
        for record in profiles.to_dict(orient="records"):
            normalized = {str(key): value for key, value in record.items()}
            ticker = _clean_ticker(normalized.get("ticker"))
            if ticker:
                profile_records[ticker] = normalized

    rows: list[dict[str, object]] = []
    for ticker in sorted(companies):
        company = companies[ticker]
        primary, secondary, confidence, company_name = classify_seed_company(
            company,
            profile_records.get(ticker),
        )
        rows.append(
            {
                "ticker": ticker,
                "company_name": company_name,
                "primary_category": primary,
                "secondary_categories": secondary,
                "confidence": confidence,
                "evidence_source": "approximate_seed_universe",
                "source_tags": "|".join(sorted(company.source_tags)),
                "valid_from": effective_date.date().isoformat(),
                "valid_to": "",
                "position_classification": company.position_classification,
            }
        )

    result = pd.DataFrame(rows, columns=pd.Index(OUTPUT_COLUMNS))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-source-dir", type=Path, default=DEFAULT_SEED_SOURCE_DIR)
    parser.add_argument(
        "--manual-universe",
        type=Path,
        default=CONFIG_DIR / "tech_seed_universe.csv",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--valid-from", default="2023-01-03")
    parser.add_argument(
        "--snapshot-date",
        default=None,
        help="Validate membership against the latest Alpha snapshot on/before this date.",
    )
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument(
        "--enrich-yfinance",
        action="store_true",
        help="Fetch uncached yfinance sector, industry, and summary metadata.",
    )
    parser.add_argument(
        "--enrichment-limit",
        type=int,
        default=100,
        help="Maximum number of uncached yfinance profiles fetched in this run.",
    )
    parser.add_argument("--yfinance-cache", type=Path, default=DEFAULT_CACHE_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    companies = collect_seed_companies(
        manual_universe_path=args.manual_universe,
        seed_source_dir=args.seed_source_dir,
    )

    listed_tickers: set[str] | None = None
    if args.snapshot_date is not None:
        listed_tickers, snapshot_path = load_listed_tickers(
            args.snapshot_dir,
            args.snapshot_date,
        )
        if listed_tickers is None or snapshot_path is None:
            raise ValueError(
                "No Alpha Vantage listing snapshot exists on or before "
                f"{args.snapshot_date}."
            )
        print(f"Validating seed tickers against {snapshot_path}.")

    profiles = load_yfinance_cache(args.yfinance_cache)
    if args.enrich_yfinance:
        profiles = enrich_yfinance_metadata(
            sorted(companies),
            cache_path=args.yfinance_cache,
            fetch_limit=args.enrichment_limit,
        )

    result = build_tech_seed_universe(
        manual_universe_path=args.manual_universe,
        seed_source_dir=args.seed_source_dir,
        output_path=args.output,
        valid_from=args.valid_from,
        listed_tickers=listed_tickers,
        profiles=profiles,
    )
    unknown_count = int(
        result["primary_category"]
        .astype(str)
        .str.lower()
        .isin(UNKNOWN_CATEGORY_ALIASES)
        .sum()
    )
    print(
        f"Saved {len(result)} approximate technology seeds "
        f"({unknown_count} unknown_tech) to {args.output}."
    )


if __name__ == "__main__":
    main()
