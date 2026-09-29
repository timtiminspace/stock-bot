"""Load runtime trading, rejection, and benchmark universes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
ALLOWED_CLASSIFICATIONS = {"recovery_candidate", "compounder"}


def _load_ticker_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Universe file does not exist: {path}")

    data = pd.read_csv(path)
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    data = data.copy()
    data["ticker"] = data["ticker"].astype(str).str.strip().str.upper()
    data = data.loc[data["ticker"].ne("")]
    if data["ticker"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate tickers.")
    return data.reset_index(drop=True)


@dataclass(frozen=True)
class Universe:
    """Ticker sets controlling what the strategy may trade or compare."""

    approved_universe: frozenset[str]
    rejected_universe: frozenset[str]
    benchmark_universe: frozenset[str]
    classifications: dict[str, str]

    def classification_for(self, ticker: str) -> str:
        normalized = ticker.upper()
        if normalized not in self.classifications:
            raise KeyError(f"Ticker is not approved: {normalized}")
        return self.classifications[normalized]

    def data_tickers(self) -> list[str]:
        tickers = (
            self.approved_universe | self.benchmark_universe
        ) - self.rejected_universe
        return sorted(tickers)

    def with_trade_universe(
        self,
        tickers: Iterable[str],
        classifications: dict[str, str] | None = None,
    ) -> Universe:
        """Return a copy whose entry whitelist is a point-in-time ticker superset."""
        approved = {
            str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()
        }
        approved -= self.rejected_universe | self.benchmark_universe
        supplied = {
            str(ticker).strip().upper(): classification
            for ticker, classification in (classifications or {}).items()
        }
        position_classifications = {
            ticker: supplied.get(
                ticker,
                self.classifications.get(ticker, "compounder"),
            )
            for ticker in approved
        }
        invalid = set(position_classifications.values()) - ALLOWED_CLASSIFICATIONS
        if invalid:
            names = ", ".join(sorted(invalid))
            raise ValueError(f"Unsupported rolling-universe classifications: {names}")
        return Universe(
            approved_universe=frozenset(approved),
            rejected_universe=self.rejected_universe,
            benchmark_universe=self.benchmark_universe,
            classifications=position_classifications,
        )


def load_universe(
    config_dir: Path = CONFIG_DIR,
    seed_path: Path | None = None,
) -> Universe:
    """Load the manual technology seed, rejection, and benchmark definitions."""
    approved = _load_ticker_csv(
        seed_path or config_dir / "tech_seed_universe.csv",
        {"ticker", "company", "theme", "classification"},
    )
    rejected = _load_ticker_csv(config_dir / "rejected_universe.csv", {"ticker"})
    benchmark = _load_ticker_csv(config_dir / "benchmark_universe.csv", {"ticker"})

    approved["classification"] = (
        approved["classification"].astype(str).str.strip().str.lower()
    )
    invalid_classifications = set(approved["classification"]) - ALLOWED_CLASSIFICATIONS
    if invalid_classifications:
        invalid = ", ".join(sorted(invalid_classifications))
        raise ValueError(f"Unsupported approved-stock classifications: {invalid}")

    approved_tickers = frozenset(approved["ticker"])
    rejected_tickers = frozenset(rejected["ticker"])
    benchmark_tickers = frozenset(benchmark["ticker"])
    if approved_tickers & rejected_tickers:
        raise ValueError("A ticker cannot be both approved and rejected.")
    if approved_tickers & benchmark_tickers:
        raise ValueError("Benchmark tickers cannot be approved for trading.")

    classifications = dict(zip(approved["ticker"], approved["classification"]))
    return Universe(
        approved_universe=approved_tickers,
        rejected_universe=rejected_tickers,
        benchmark_universe=benchmark_tickers,
        classifications=classifications,
    )
