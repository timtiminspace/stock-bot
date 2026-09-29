"""Tests for deterministic multi-label technology classification."""

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "marketsignallab" / "src"
sys.path.insert(0, str(SRC_DIR))

from classification import (
    UNKNOWN_CATEGORY,
    build_category_history,
    category_as_of,
    classify_record,
)


class ClassificationTests(unittest.TestCase):
    def test_expected_technology_examples_are_multi_labelled(self) -> None:
        examples = {
            "CRWD": (
                {
                    "industry": "Cybersecurity",
                    "description": "Cloud endpoint cybersecurity and threat intelligence",
                },
                {"cybersecurity"},
            ),
            "NVDA": (
                {
                    "industry": "Semiconductors",
                    "description": "Designs GPUs and AI accelerators",
                },
                {"semiconductors", "ai_infrastructure"},
            ),
            "SNOW": (
                {
                    "industry": "Data analytics",
                    "description": "Cloud data platform and data warehouse",
                },
                {"data_analytics", "cloud_infrastructure"},
            ),
            "MSFT": (
                {
                    "description": "Public cloud computing, generative AI, and software as a service",
                },
                {"cloud_infrastructure", "ai_software", "saas"},
            ),
            "IONQ": (
                {
                    "description": "Develops quantum computers and quantum computing systems"
                },
                {"quantum_computing"},
            ),
        }

        for ticker, (record, expected) in examples.items():
            with self.subTest(ticker=ticker):
                result = classify_record(record)
                labels = {str(result["primary_category"])} | set(
                    str(result["secondary_categories"]).split("|")
                )
                labels.discard("")
                self.assertTrue(expected.issubset(labels))

    def test_non_technology_text_remains_unknown(self) -> None:
        result = classify_record(
            {"industry": "Restaurants", "description": "Operates coffee shops"}
        )
        self.assertEqual(result["primary_category"], UNKNOWN_CATEGORY)
        self.assertEqual(result["classification_confidence"], 0.0)

    def test_category_history_never_uses_a_future_record(self) -> None:
        first = build_category_history(
            pd.DataFrame([{"ticker": "AAA", "description": "Cybersecurity software"}]),
            valid_from="2023-01-01",
        )
        second = build_category_history(
            pd.DataFrame(
                [{"ticker": "AAA", "description": "Quantum computing platform"}]
            ),
            valid_from="2024-01-01",
        )
        history = pd.concat([first, second], ignore_index=True)

        in_2023 = category_as_of(history, "AAA", "2023-06-01")
        in_2024 = category_as_of(history, "AAA", "2024-06-01")

        self.assertIsNotNone(in_2023)
        self.assertIsNotNone(in_2024)
        self.assertEqual(in_2023["primary_category"], "cybersecurity")
        self.assertEqual(in_2024["primary_category"], "quantum_computing")


if __name__ == "__main__":
    unittest.main()
