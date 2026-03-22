"""Tool implementations (fixed — not the primary optimization surface)."""

import re

from agents.agent4.original_agent.data import (
    COMPARABLE_SALES,
    DEFAULT_MARKET,
    MARKET_TRENDS,
    PROPERTY_DB,
)


def _normalize_addr(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _extract_zip(address: str) -> str | None:
    m = re.search(r"\b(\d{5})(?:-\d{4})?\b", address)
    return m.group(1) if m else None


def lookup_property(address: str) -> dict:
    norm = _normalize_addr(address)
    zip_guess = _extract_zip(address)
    # Exact / substring match on DB keys
    for key, rec in PROPERTY_DB.items():
        if key in norm or norm in key:
            return {"found": True, "record": rec}
    # Token overlap scoring
    best_key = None
    best_score = 0
    tokens = set(re.findall(r"[a-z0-9]+", norm))
    for key, rec in PROPERTY_DB.items():
        kt = set(re.findall(r"[a-z0-9]+", key))
        overlap = len(tokens & kt)
        if overlap > best_score:
            best_score = overlap
            best_key = key
    if best_key and best_score >= 3:
        return {"found": True, "record": PROPERTY_DB[best_key], "match": "fuzzy"}
    return {
        "found": False,
        "address_query": address,
        "zip_from_address": zip_guess,
        "hint": "No direct record; use user-provided attributes and comps for the ZIP.",
    }


def get_comparable_sales(
    zip_code: str,
    property_type: str,
    square_footage: int | None = None,
    limit: int = 10,
) -> dict:
    z = re.sub(r"\D", "", zip_code)[:5]
    pt = property_type.lower().strip()
    rows = [
        c for c in COMPARABLE_SALES if c["zip_code"] == z and c["property_type"] == pt
    ]
    if not rows:
        rows = [c for c in COMPARABLE_SALES if c["zip_code"] == z]
    if not rows:
        rows = [c for c in COMPARABLE_SALES if c["property_type"] == pt]
    if square_footage and rows:

        def sqft_dist(r: dict) -> float:
            return abs(r["square_footage"] - square_footage)

        rows = sorted(rows, key=sqft_dist)
    lim = max(1, min(limit, 20))
    return {
        "zip_code": z,
        "property_type": pt,
        "count": len(rows[:lim]),
        "comparables": rows[:lim],
    }


def get_market_trends(zip_code: str) -> dict:
    z = re.sub(r"\D", "", zip_code)[:5]
    base = MARKET_TRENDS.get(z, {**DEFAULT_MARKET})
    out = {**base, "zip_code": z}
    return out


TOOL_FUNCTIONS = {
    "lookup_property": lookup_property,
    "get_comparable_sales": get_comparable_sales,
    "get_market_trends": get_market_trends,
}
