"""Tests for point-in-time Alpha Vantage listing downloads."""

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

SRC_DIR = Path(__file__).resolve().parents[1] / "marketsignallab" / "src"
sys.path.insert(0, str(SRC_DIR))

from providers.alpha_vantage_listing import (
    AlphaVantageHTTPError,
    AlphaVantageListingClient,
    AlphaVantageUsageError,
)


class FakeResponse:
    def __init__(self, body: str) -> None:
        self.body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class AlphaVantageListingTests(unittest.TestCase):
    @patch("providers.alpha_vantage_listing.urlopen")
    def test_parses_listing_csv_and_sends_point_in_time_parameters(
        self, open_url
    ) -> None:
        open_url.return_value = FakeResponse(
            "symbol,name,exchange,assetType,status\nNVDA,NVIDIA,NASDAQ,Stock,Active\n"
        )
        client = AlphaVantageListingClient("test-key")

        result = client.listing_status("2023-01-31")

        self.assertEqual(result["symbol"].tolist(), ["NVDA"])
        request = open_url.call_args.args[0]
        self.assertIn("function=LISTING_STATUS", request.full_url)
        self.assertIn("date=2023-01-31", request.full_url)
        self.assertIn("state=active", request.full_url)

    @patch("providers.alpha_vantage_listing.urlopen")
    def test_turns_quota_json_into_clear_usage_error(self, open_url) -> None:
        open_url.return_value = FakeResponse(
            '{"Information":"API rate limit reached for this key"}'
        )

        with self.assertRaisesRegex(AlphaVantageUsageError, "rate limit"):
            AlphaVantageListingClient("test-key").listing_status("2023-01-31")

    @patch("providers.alpha_vantage_listing.urlopen")
    def test_preserves_http_status_for_subscription_failures(self, open_url) -> None:
        open_url.side_effect = HTTPError(
            "https://example.invalid",
            402,
            "Payment Required",
            {},
            io.BytesIO(b"premium endpoint"),
        )

        with self.assertRaises(AlphaVantageHTTPError) as raised:
            AlphaVantageListingClient("test-key").listing_status("2023-01-31")
        self.assertEqual(raised.exception.status_code, 402)


if __name__ == "__main__":
    unittest.main()
