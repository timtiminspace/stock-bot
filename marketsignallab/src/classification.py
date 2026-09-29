"""Deterministic, evidence-based technology category classification."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, cast

import pandas as pd

CLASSIFIER_VERSION = "keyword-v1"
UNKNOWN_CATEGORY = "unknown_technology"
UNKNOWN_CATEGORY_ALIASES = {UNKNOWN_CATEGORY, "unknown_tech"}

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "cybersecurity": (
        "cybersecurity",
        "cyber security",
        "endpoint security",
        "zero trust",
        "identity security",
        "threat intelligence",
        "cloud security",
        "network security",
        "firewall",
        "data security",
        "edge security",
        "security analytics",
        "vulnerability management",
    ),
    "ai_software": (
        "ai",
        "artificial intelligence",
        "machine learning",
        "generative ai",
        "large language model",
        "predictive ai",
        "computer vision",
        "natural language processing",
        "ai software",
        "cloud ai",
        "data ai",
        "observability ai",
    ),
    "ai_infrastructure": (
        "ai infrastructure",
        "ai accelerator",
        "ai accelerators",
        "gpu",
        "graphics processing unit",
        "inference workloads",
        "training workloads",
        "accelerated computing",
        "ai server",
        "ai servers",
    ),
    "semiconductors": (
        "semiconductor",
        "semiconductors",
        "integrated circuit",
        "microprocessor",
        "processor design",
        "wafer fabrication",
        "foundry",
        "fabless",
        "chipmaker",
    ),
    "cloud_infrastructure": (
        "cloud infrastructure",
        "cloud computing",
        "public cloud",
        "hybrid cloud",
        "cloud data platform",
        "data center infrastructure",
        "infrastructure as a service",
        "platform as a service",
        "observability platform",
        "observability",
        "cloud ai",
    ),
    "saas": (
        "software as a service",
        "saas",
        "subscription software",
        "cloud-based software",
        "cloud software platform",
        "enterprise application software",
    ),
    "data_analytics": (
        "data analytics",
        "analytics platform",
        "data platform",
        "database platform",
        "data warehouse",
        "data lake",
        "business intelligence",
    ),
    "networking": (
        "network switch",
        "network switches",
        "networking equipment",
        "network infrastructure",
        "ethernet",
        "routing platform",
        "wireless networking",
        "networking",
    ),
    "consumer_hardware": (
        "personal computer",
        "gaming hardware",
        "computer peripherals",
        "consumer electronics",
        "smartphone hardware",
        "wearable device",
        "pc hardware",
        "pc peripherals",
        "gaming pc hardware",
    ),
    "enterprise_hardware": (
        "enterprise hardware",
        "enterprise server",
        "data center server",
        "storage hardware",
        "computer workstation",
        "server systems",
        "ai servers",
        "servers networking",
    ),
    "robotics_automation": (
        "industrial robot",
        "industrial robotics",
        "robotic automation",
        "warehouse automation",
        "autonomous robot",
        "factory automation",
    ),
    "quantum_computing": (
        "quantum computing",
        "quantum computer",
        "quantum processor",
        "quantum hardware",
        "quantum software",
        "quantum",
    ),
}

EVIDENCE_FIELD_WEIGHTS: dict[str, float] = {
    "industry": 3.0,
    "sic_description": 2.5,
    "description": 1.5,
    "name": 0.75,
    "company": 0.75,
    "sector": 0.5,
}

CLASSIFICATION_COLUMNS = [
    "primary_category",
    "secondary_categories",
    "classification_confidence",
    "classifier_version",
]

CATEGORY_HISTORY_COLUMNS = [
    "ticker",
    "cik",
    "valid_from",
    "valid_to",
    "primary_category",
    "secondary_categories",
    "classification_confidence",
    "evidence_source",
    "classifier_version",
]


def _text(value: object) -> str:
    """Return clean scalar evidence text without asking Pandas to type an object."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def _contains_keyword(text: str, keyword: str) -> bool:
    if keyword == "ai":
        return re.search(r"\bai\b", text) is not None
    return keyword in text


def _score_text(text: str, weight: float = 1.0) -> dict[str, float]:
    """Score each category once per distinct matching phrase."""
    normalized = text.lower()
    return {
        category: weight
        * float(
            sum(1 for keyword in keywords if _contains_keyword(normalized, keyword))
        )
        for category, keywords in CATEGORY_KEYWORDS.items()
        if any(_contains_keyword(normalized, keyword) for keyword in keywords)
    }


def _rank_scores(scores: Mapping[str, float]) -> dict[str, object]:
    """Convert deterministic category scores into stable multi-label output."""
    positive = [(category, score) for category, score in scores.items() if score > 0]
    if not positive:
        return {
            "primary_category": UNKNOWN_CATEGORY,
            "secondary_categories": "",
            "classification_confidence": 0.0,
            "classifier_version": CLASSIFIER_VERSION,
        }

    ranked = sorted(positive, key=lambda item: (-item[1], item[0]))
    primary, primary_score = ranked[0]
    secondary = [
        category
        for category, score in ranked[1:]
        if score >= max(1.0, primary_score * 0.25)
    ][:3]
    confidence = min(0.98, 0.50 + 0.08 * primary_score)
    if len(ranked) > 1 and ranked[1][1] == primary_score:
        confidence = max(0.50, confidence - 0.10)

    return {
        "primary_category": primary,
        "secondary_categories": "|".join(secondary),
        "classification_confidence": round(confidence, 4),
        "classifier_version": CLASSIFIER_VERSION,
    }


def classify_text(text: str) -> dict[str, object]:
    """Classify one cached text block into deterministic technology labels."""
    return _rank_scores(_score_text(text))


def classify_record(record: Mapping[str, object]) -> dict[str, object]:
    """Classify a company record, weighting structured industry evidence highest."""
    scores: dict[str, float] = {}
    for field, weight in EVIDENCE_FIELD_WEIGHTS.items():
        field_scores = _score_text(_text(record.get(field)), weight)
        for category, score in field_scores.items():
            scores[category] = scores.get(category, 0.0) + score
    return _rank_scores(scores)


def classify_records(records: pd.DataFrame) -> pd.DataFrame:
    """Append categories, preserving explicit cached seed classifications."""
    if "ticker" not in records.columns:
        raise ValueError("Classification input must contain a ticker column.")

    data = records.copy()
    data["ticker"] = data["ticker"].astype(str).str.strip().str.upper()
    if bool(cast(Any, data["ticker"].eq("").any())):
        raise ValueError("Classification input contains an empty ticker.")
    if data.empty:
        for column in CLASSIFICATION_COLUMNS:
            data[column] = pd.Series(dtype="object")
        return data

    results: list[dict[str, object]] = []
    for raw_row in data.to_dict(orient="records"):
        row = {str(key): value for key, value in raw_row.items()}
        raw_category = row.get("primary_category")
        supplied_category = (
            ""
            if raw_category is None
            or (isinstance(raw_category, float) and math.isnan(raw_category))
            else str(raw_category).strip()
        )
        if supplied_category:
            raw_confidence = row.get(
                "classification_confidence", row.get("confidence", 0.0)
            )
            try:
                confidence = float(cast(Any, raw_confidence))
                if math.isnan(confidence):
                    confidence = 0.0
            except (TypeError, ValueError):
                confidence = 0.0
            raw_secondary = row.get("secondary_categories")
            secondary = (
                ""
                if raw_secondary is None
                or (isinstance(raw_secondary, float) and math.isnan(raw_secondary))
                else str(raw_secondary).strip()
            )
            results.append(
                {
                    "primary_category": supplied_category,
                    "secondary_categories": secondary,
                    "classification_confidence": min(max(confidence, 0.0), 1.0),
                    "classifier_version": str(
                        row.get("classifier_version") or "approximate-seed-v1"
                    ),
                }
            )
        else:
            results.append(classify_record(row))
    classified = pd.DataFrame(results, index=data.index)
    for column in CLASSIFICATION_COLUMNS:
        data[column] = classified[column]
    return data


def build_category_history(
    evidence: pd.DataFrame,
    *,
    valid_from: str,
    evidence_source: str = "cached_evidence",
) -> pd.DataFrame:
    """Build an effective-dated category-history table from cached evidence."""
    effective_date = cast(pd.Timestamp, pd.Timestamp(cast(Any, valid_from)))
    if str(effective_date) == "NaT":
        raise ValueError("valid_from must be a valid date.")

    classified = classify_records(evidence)
    if bool(cast(Any, classified["ticker"].duplicated().any())):
        raise ValueError("Category evidence contains duplicate tickers.")

    history = pd.DataFrame(index=classified.index)
    history["ticker"] = classified["ticker"]
    history["cik"] = (
        classified["cik"]
        if "cik" in classified.columns
        else pd.Series("", index=classified.index)
    )
    history["valid_from"] = effective_date.date().isoformat()
    history["valid_to"] = ""
    history["primary_category"] = classified["primary_category"]
    history["secondary_categories"] = classified["secondary_categories"]
    history["classification_confidence"] = classified["classification_confidence"]
    history["evidence_source"] = (
        classified["evidence_source"]
        if "evidence_source" in classified.columns
        else evidence_source
    )
    history["classifier_version"] = classified["classifier_version"]
    return cast(
        pd.DataFrame,
        history.loc[:, CATEGORY_HISTORY_COLUMNS].reset_index(drop=True),
    )


def category_as_of(
    category_history: pd.DataFrame,
    ticker: str,
    as_of: str | pd.Timestamp,
) -> pd.Series | None:
    """Return the most recent non-future category record for a ticker."""
    required = {"ticker", "valid_from", "valid_to"}
    missing = required.difference(category_history.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Category history is missing columns: {names}")

    data = category_history.copy()
    data["ticker"] = data["ticker"].astype(str).str.strip().str.upper()
    data["valid_from"] = pd.to_datetime(data["valid_from"], errors="coerce")
    data["valid_to"] = pd.to_datetime(data["valid_to"], errors="coerce")
    snapshot_date = pd.Timestamp(as_of)
    rows = data.loc[
        data["ticker"].eq(ticker.strip().upper())
        & data["valid_from"].notna()
        & data["valid_from"].le(snapshot_date)
        & (data["valid_to"].isna() | data["valid_to"].ge(snapshot_date))
    ].sort_values("valid_from")
    if rows.empty:
        return None
    return rows.iloc[-1]
