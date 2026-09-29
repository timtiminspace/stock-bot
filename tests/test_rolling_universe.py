"""Tests for listing cleaning, immutable snapshots, and as-of resolution."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "marketsignallab" / "src"
sys.path.insert(0, str(SRC_DIR))

from rolling_universe import (
    RollingUniverse,
    build_listing_snapshot,
    clean_listing_roster,
    month_ends,
    normalize_ticker,
)


class FakeListingClient:
    def __init__(self, roster: pd.DataFrame) -> None:
        self.roster = roster
        self.calls: list[tuple[str, str]] = []

    def listing_status(self, date: str, state: str = "active") -> pd.DataFrame:
        self.calls.append((date, state))
        return self.roster.copy()


class RollingUniverseTests(unittest.TestCase):
    def test_normalizes_yfinance_ticker_convention(self) -> None:
        self.assertEqual(normalize_ticker(" brk.b "), "BRK-B")

    def test_clean_roster_excludes_non_common_and_forbidden_securities(self) -> None:
        roster = pd.DataFrame(
            [
                {
                    "symbol": "nvda",
                    "name": "NVIDIA",
                    "exchange": "NASDAQ",
                    "assetType": "Stock",
                },
                {
                    "symbol": "BRK.B",
                    "name": "Berkshire Hathaway",
                    "exchange": "NYSE",
                    "assetType": "Equity",
                },
                {
                    "symbol": "UTL",
                    "name": "Unitil Corporation",
                    "exchange": "NYSE",
                    "assetType": "Stock",
                },
                {
                    "symbol": "QQQ",
                    "name": "Nasdaq ETF",
                    "exchange": "NASDAQ",
                    "assetType": "ETF",
                },
                {
                    "symbol": "ABC-U",
                    "name": "ABC Units",
                    "exchange": "NYSE",
                    "assetType": "Stock",
                },
                {
                    "symbol": "BAD",
                    "name": "Rejected Corp",
                    "exchange": "AMEX",
                    "assetType": "Stock",
                },
                {
                    "symbol": "OTC",
                    "name": "OTC Corp",
                    "exchange": "OTC",
                    "assetType": "Stock",
                },
            ]
        )

        clean = clean_listing_roster(
            roster,
            "2023-01-31",
            rejected_tickers={"BAD"},
            benchmark_tickers={"QQQ"},
        )

        self.assertEqual(clean["ticker"].tolist(), ["NVDA", "BRK-B", "UTL"])
        self.assertEqual(clean["snapshot_date"].unique().tolist(), ["2023-01-31"])

    def test_snapshot_saves_raw_and_classified_outputs_and_reuses_cache(self) -> None:
        roster = pd.DataFrame(
            [
                {
                    "symbol": "CRWD",
                    "name": "CrowdStrike",
                    "exchange": "NASDAQ",
                    "assetType": "Stock",
                },
                {
                    "symbol": "KO",
                    "name": "Coca-Cola",
                    "exchange": "NYSE",
                    "assetType": "Stock",
                },
            ]
        )
        evidence = pd.DataFrame(
            [
                {
                    "ticker": "CRWD",
                    "industry": "Cybersecurity",
                    "description": "Endpoint security",
                },
                {
                    "ticker": "KO",
                    "industry": "Beverages",
                    "description": "Sells soft drinks",
                },
            ]
        )
        client = FakeListingClient(roster)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot_dir = root / "snapshots"
            raw_dir = root / "raw"
            history_path = root / "category_history.csv"
            first = build_listing_snapshot(
                "2023-01-31",
                client=client,
                snapshot_dir=snapshot_dir,
                raw_dir=raw_dir,
                evidence=evidence,
                category_history_path=history_path,
            )
            second = build_listing_snapshot(
                "2023-01-31",
                client=client,
                snapshot_dir=snapshot_dir,
                raw_dir=raw_dir,
                evidence=evidence,
                category_history_path=history_path,
            )
            rebuilt = build_listing_snapshot(
                "2023-01-31",
                client=client,
                snapshot_dir=snapshot_dir,
                raw_dir=raw_dir,
                evidence=evidence,
                category_history_path=history_path,
                overwrite=True,
            )

            self.assertEqual(client.calls, [("2023-01-31", "active")])
            self.assertTrue((raw_dir / "listing_status_2023-01-31_active.csv").exists())
            self.assertTrue((snapshot_dir / "2023-01-31.csv").exists())
            self.assertTrue(history_path.exists())
            self.assertEqual(first.loc[first["eligible"], "ticker"].tolist(), ["CRWD"])
            self.assertEqual(second["ticker"].tolist(), first["ticker"].tolist())
            self.assertEqual(rebuilt["ticker"].tolist(), first["ticker"].tolist())

    def test_empty_evidence_fails_before_requesting_provider_data(self) -> None:
        client = FakeListingClient(pd.DataFrame())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "contains no company rows"):
                build_listing_snapshot(
                    "2023-01-31",
                    client=client,
                    snapshot_dir=root / "snapshots",
                    raw_dir=root / "raw",
                    evidence=pd.DataFrame(columns=["ticker"]),
                )
        self.assertEqual(client.calls, [])

    def test_resolver_uses_latest_snapshot_not_after_requested_date(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pd.DataFrame({"ticker": ["AAA"], "eligible": [True]}).to_csv(
                directory / "2023-01-31.csv", index=False
            )
            pd.DataFrame({"ticker": ["BBB"], "eligible": [True]}).to_csv(
                directory / "2023-02-28.csv", index=False
            )
            rolling = RollingUniverse.from_directory(directory)

            self.assertIsNone(rolling.snapshot_date_for("2023-01-01"))
            self.assertEqual(rolling.eligible_tickers("2023-02-15"), {"AAA"})
            self.assertEqual(rolling.eligible_tickers("2023-03-01"), {"BBB"})
            self.assertEqual(
                rolling.snapshot_date_for("2023-03-01"), pd.Timestamp("2023-02-28")
            )

    def test_recommended_end_covers_the_month_after_latest_snapshot(self) -> None:
        rolling = RollingUniverse(
            {
                pd.Timestamp("2023-12-31"): pd.DataFrame(
                    {"ticker": ["AAA"], "eligible": [True]}
                ),
                pd.Timestamp("2024-01-31"): pd.DataFrame(
                    {"ticker": ["AAA"], "eligible": [True]}
                ),
            }
        )

        self.assertEqual(rolling.latest_snapshot_date, pd.Timestamp("2024-01-31"))
        self.assertEqual(
            rolling.recommended_end_exclusive(),
            pd.Timestamp("2024-03-01"),
        )

    def test_position_classification_is_resolved_without_future_leakage(self) -> None:
        rolling = RollingUniverse(
            {
                pd.Timestamp("2023-01-31"): pd.DataFrame(
                    {
                        "ticker": ["AAA"],
                        "eligible": [True],
                        "position_classification": ["recovery_candidate"],
                    }
                ),
                pd.Timestamp("2023-02-28"): pd.DataFrame(
                    {
                        "ticker": ["AAA"],
                        "eligible": [True],
                        "position_classification": ["compounder"],
                    }
                ),
            }
        )

        self.assertEqual(
            rolling.position_classification_for("AAA", "2023-02-15"),
            "recovery_candidate",
        )
        self.assertEqual(
            rolling.position_classification_for("AAA", "2023-03-01"),
            "compounder",
        )

    def test_snapshot_directory_requires_explicit_eligibility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pd.DataFrame({"ticker": ["AAA"]}).to_csv(
                directory / "2023-01-31.csv", index=False
            )

            with self.assertRaisesRegex(ValueError, "no eligibility field"):
                RollingUniverse.from_directory(directory)

    def test_future_dated_evidence_cannot_make_a_ticker_eligible(self) -> None:
        roster = pd.DataFrame(
            [
                {
                    "symbol": "AAA",
                    "name": "Example",
                    "exchange": "NASDAQ",
                    "assetType": "Stock",
                }
            ]
        )
        future_evidence = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "description": "Cybersecurity software",
                    "valid_from": "2024-01-01",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = build_listing_snapshot(
                "2023-01-31",
                client=FakeListingClient(roster),
                snapshot_dir=root / "snapshots",
                raw_dir=root / "raw",
                evidence=future_evidence,
                category_history_path=None,
            )

        self.assertFalse(bool(snapshot.iloc[0]["eligible"]))
        self.assertEqual(snapshot.iloc[0]["primary_category"], "unknown_technology")

    def test_preclassified_approximate_seed_evidence_is_supported(self) -> None:
        roster = pd.DataFrame(
            [
                {
                    "symbol": "CRWD",
                    "name": "CrowdStrike",
                    "exchange": "NASDAQ",
                    "assetType": "Stock",
                },
                {
                    "symbol": "MYST",
                    "name": "Mystery Tech",
                    "exchange": "NYSE",
                    "assetType": "Stock",
                },
            ]
        )
        evidence = pd.DataFrame(
            [
                {
                    "ticker": "CRWD",
                    "primary_category": "cyber",
                    "secondary_categories": "cloud",
                    "confidence": 0.90,
                    "evidence_source": "approximate_seed_universe",
                    "valid_from": "2023-01-01",
                },
                {
                    "ticker": "MYST",
                    "primary_category": "unknown_tech",
                    "secondary_categories": "",
                    "confidence": 0.90,
                    "evidence_source": "approximate_seed_universe",
                    "valid_from": "2023-01-01",
                },
            ]
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = build_listing_snapshot(
                "2023-01-31",
                client=FakeListingClient(roster),
                snapshot_dir=root / "snapshots",
                raw_dir=root / "raw",
                evidence=evidence,
                category_history_path=None,
            )

        eligible = snapshot.loc[snapshot["eligible"], "ticker"].tolist()
        self.assertEqual(eligible, ["CRWD"])
        self.assertEqual(snapshot.iloc[0]["classifier_version"], "approximate-seed-v1")

    def test_month_end_generation_is_inclusive_and_validated(self) -> None:
        self.assertEqual(
            month_ends("2023-01-01", "2023-03-31"),
            ["2023-01-31", "2023-02-28", "2023-03-31"],
        )
        with self.assertRaises(ValueError):
            month_ends("2023-03-01", "2023-01-01")

    def test_stale_snapshot_series_is_rejected_for_later_backtest_data(self) -> None:
        rolling = RollingUniverse(
            {pd.Timestamp("2023-11-30"): pd.DataFrame({"ticker": ["AAA"]})}
        )
        with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
            rolling.validate_coverage("2024-03-31")
        rolling.validate_coverage("2023-12-31")


if __name__ == "__main__":
    unittest.main()
