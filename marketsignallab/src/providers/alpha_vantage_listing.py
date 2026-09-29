"""Point-in-time US listing rosters from Alpha Vantage."""

from __future__ import annotations

import json
import os
from io import StringIO
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"


class AlphaVantageError(RuntimeError):
    """Raised when Alpha Vantage cannot return a usable listing roster."""


class AlphaVantageConfigurationError(AlphaVantageError):
    """Raised when the Alpha Vantage API key is not configured."""


class AlphaVantageHTTPError(AlphaVantageError):
    """Raised when Alpha Vantage returns an HTTP failure."""

    def __init__(self, status_code: int, detail: str = "") -> None:
        self.status_code = status_code
        self.detail = detail
        suffix = f": {detail}" if detail else "."
        super().__init__(
            f"Alpha Vantage LISTING_STATUS request failed with HTTP "
            f"{status_code}{suffix}"
        )


class AlphaVantageUsageError(AlphaVantageError):
    """Raised for quota, rate-limit, entitlement, or invalid-key messages."""


def require_alpha_vantage_api_key(api_key: str | None = None) -> str:
    """Return an explicit or environment-provided Alpha Vantage API key."""
    configured = (api_key or os.getenv("ALPHAVANTAGE_API_KEY", "")).strip()
    if not configured:
        raise AlphaVantageConfigurationError(
            "ALPHAVANTAGE_API_KEY is not set. Export it in your shell or pass an "
            "API key when constructing AlphaVantageListingClient."
        )
    return configured


def _error_message_from_payload(text: str) -> str | None:
    """Extract Alpha Vantage's JSON usage/error messages when present."""
    stripped = text.strip()
    if not stripped:
        return "Alpha Vantage returned an empty response."

    try:
        payload: Any = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        for key in ("Error Message", "Information", "Note", "message", "error"):
            value = payload.get(key)
            if value:
                return str(value).strip()
        return "Alpha Vantage returned JSON instead of a listing-status CSV."

    lowered = stripped.lower()
    usage_markers = (
        "rate limit",
        "call frequency",
        "thank you for using alpha vantage",
        "invalid api call",
        "premium endpoint",
    )
    if any(marker in lowered for marker in usage_markers):
        return stripped[:500]
    if stripped.startswith("<"):
        return "Alpha Vantage returned HTML instead of a listing-status CSV."
    return None


class AlphaVantageListingClient:
    """Download historical active or delisted US security rosters as CSV."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = require_alpha_vantage_api_key(api_key)
        self.timeout = timeout

    def listing_status(self, date: str, state: str = "active") -> pd.DataFrame:
        """Return the LISTING_STATUS roster for one point-in-time date."""
        snapshot_date = cast(pd.Timestamp, pd.Timestamp(cast(Any, date)))
        if str(snapshot_date) == "NaT":
            raise ValueError("date must be a valid calendar date.")
        if snapshot_date < pd.Timestamp("2010-01-01"):
            raise ValueError(
                "Alpha Vantage listing dates must be on or after 2010-01-01."
            )

        normalized_state = state.strip().lower()
        if normalized_state not in {"active", "delisted"}:
            raise ValueError("state must be either 'active' or 'delisted'.")

        params = urlencode(
            {
                "function": "LISTING_STATUS",
                "date": snapshot_date.date().isoformat(),
                "state": normalized_state,
                "apikey": self.api_key,
            }
        )
        request = Request(
            f"{ALPHA_VANTAGE_BASE_URL}?{params}",
            headers={
                "Accept": "text/csv, application/json",
                "User-Agent": "MarketSignalLab/1.0",
            },
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                text = response.read().decode("utf-8-sig")
        except HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace").strip()[:500]
            except (AttributeError, OSError):
                detail = ""
            raise AlphaVantageHTTPError(exc.code, detail) from exc
        except (URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", "request timed out")
            raise AlphaVantageError(
                f"Alpha Vantage LISTING_STATUS request failed: {reason}"
            ) from exc
        except UnicodeDecodeError as exc:
            raise AlphaVantageError(
                "Alpha Vantage returned a response that was not valid UTF-8."
            ) from exc

        error_message = _error_message_from_payload(text)
        if error_message is not None:
            raise AlphaVantageUsageError(error_message)

        try:
            roster = pd.read_csv(StringIO(text))
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            raise AlphaVantageError(
                "Alpha Vantage returned an invalid listing-status CSV."
            ) from exc

        normalized_columns = {
            str(column).strip().lower().replace("_", "") for column in roster.columns
        }
        if "symbol" not in normalized_columns:
            columns = ", ".join(str(column) for column in roster.columns)
            raise AlphaVantageError(
                "Alpha Vantage listing-status CSV has no symbol column. "
                f"Received: {columns}"
            )
        return roster
