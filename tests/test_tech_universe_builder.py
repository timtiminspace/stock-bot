"""Tests for the approximate technology seed-universe builder."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "marketsignallab" / "src"
sys.path.insert(0, str(SRC_DIR))

from tech_universe_builder import OUTPUT_COLUMNS, build_tech_seed_universe


class TechUniverseBuilderTests(unittest.TestCase):
    def test_reads_supported_columns_deduplicates_and_aggregates_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seeds = root / "seed_sources"
            seeds.mkdir()
            manual = root / "tech_seed_universe.csv"
            output = root / "company_evidence.csv"

            pd.DataFrame(
                {
                    "ticker": ["AAPL", "CRWD"],
                    "company": ["Apple", "CrowdStrike"],
                    "theme": ["hardware", "cybersecurity"],
                    "classification": ["compounder", "recovery_candidate"],
                }
            ).to_csv(manual, index=False)
            pd.DataFrame({"symbol": ["NVDA", "CRWD"]}).to_csv(
                seeds / "semiconductor_etf.csv", index=False
            )
            pd.DataFrame({"holding": ["SNOW"]}).to_csv(seeds / "cloud.csv", index=False)
            pd.DataFrame({"name": ["PLTR"]}).to_csv(seeds / "ai.csv", index=False)

            result = build_tech_seed_universe(
                manual_universe_path=manual,
                seed_source_dir=seeds,
                output_path=output,
                valid_from="2023-01-31",
                listed_tickers={"CRWD", "NVDA", "SNOW", "PLTR"},
            )

            self.assertEqual(result.columns.tolist(), OUTPUT_COLUMNS)
            self.assertEqual(
                result["ticker"].tolist(), ["CRWD", "NVDA", "PLTR", "SNOW"]
            )
            self.assertNotIn("AAPL", set(result["ticker"]))
            crwd = result.set_index("ticker").loc["CRWD"]
            self.assertIn("manual:tech_seed_universe", crwd["source_tags"])
            self.assertIn("seed:semiconductor_etf", crwd["source_tags"])
            self.assertEqual(crwd["primary_category"], "cybersecurity")
            self.assertEqual(crwd["position_classification"], "recovery_candidate")
            self.assertTrue(
                result["evidence_source"].eq("approximate_seed_universe").all()
            )
            self.assertTrue(output.exists())

    def test_cached_profile_metadata_can_classify_an_unknown_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seeds = root / "seed_sources"
            seeds.mkdir()
            pd.DataFrame({"ticker": ["XYZ"]}).to_csv(
                seeds / "broad_technology.csv", index=False
            )
            manual = root / "missing_manual.csv"
            profiles = pd.DataFrame(
                {
                    "ticker": ["XYZ"],
                    "company_name": ["Example Cloud"],
                    "sector": ["Technology"],
                    "industry": ["Cloud software"],
                    "long_business_summary": ["Enterprise software as a service"],
                }
            )

            result = build_tech_seed_universe(
                manual_universe_path=manual,
                seed_source_dir=seeds,
                output_path=root / "out.csv",
                profiles=profiles,
            )

            self.assertEqual(result.iloc[0]["company_name"], "Example Cloud")
            self.assertEqual(result.iloc[0]["primary_category"], "saas")
            self.assertGreaterEqual(float(result.iloc[0]["confidence"]), 0.60)


if __name__ == "__main__":
    unittest.main()
