"""Tools for the returns_concierge agent.

Vague tool docstrings on purpose — Overmind should specify when each is
worth calling (e.g. don't inspect photos that weren't attached).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from overmind import tool

_DATA_DIR = Path(__file__).parent / "data"


def _load(name: str) -> Any:
    path = _DATA_DIR / name
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


@tool("lookup_order")
def lookup_order(order_id: str) -> dict[str, Any]:
    """Look up an order."""
    data = _load("orders.json")
    return data.get(order_id, {"order_id": order_id, "found": False})


@tool("check_return_window")
def check_return_window(order_id: str) -> dict[str, Any]:
    """Check the return window."""
    data = _load("orders.json")
    order = data.get(order_id)
    if not order:
        return {"order_id": order_id, "in_window": False, "reason": "order_not_found"}
    delivered = order.get("delivered_at")
    final_sale = order.get("final_sale", False)
    if not delivered:
        return {"order_id": order_id, "in_window": False, "reason": "not_yet_delivered"}
    if final_sale:
        return {"order_id": order_id, "in_window": False, "reason": "final_sale"}
    delivered_dt = datetime.fromisoformat(delivered.replace("Z", "+00:00"))
    today = datetime(2026, 5, 1, tzinfo=delivered_dt.tzinfo)
    days = (today - delivered_dt).days
    return {
        "order_id": order_id,
        "in_window": days <= 30,
        "days_since_delivery": days,
        "policy_window_days": 30,
    }


@tool("inspect_condition_photos")
def inspect_condition_photos(photos_url: str | None) -> dict[str, Any]:
    """Inspect condition photos."""
    if not photos_url:
        return {"has_photos": False, "condition": "unknown"}
    data = _load("photo_inspections.json")
    return data.get(photos_url, {"has_photos": True, "condition": "good"})


@tool("get_customer_ltv")
def get_customer_ltv(customer_id: str) -> dict[str, Any]:
    """Get customer LTV."""
    data = _load("customers.json")
    return data.get(customer_id, {"customer_id": customer_id, "ltv_usd": 0, "tier": "standard", "abuse_flags": []})


@tool("issue_refund_or_replacement")
def issue_refund_or_replacement(order_id: str, kind: str, amount: float) -> dict[str, Any]:
    """Issue a refund or replacement."""
    return {"ok": True, "order_id": order_id, "kind": kind, "amount": amount}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_return_window",
            "description": "Check the return window.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_condition_photos",
            "description": "Inspect condition photos.",
            "parameters": {
                "type": "object",
                "properties": {"photos_url": {"type": "string"}},
                "required": ["photos_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer_ltv",
            "description": "Get customer LTV.",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund_or_replacement",
            "description": "Issue a refund or replacement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "kind": {"type": "string"},
                    "amount": {"type": "number"},
                },
                "required": ["order_id", "kind", "amount"],
            },
        },
    },
]


TOOL_FNS = {
    "lookup_order": lookup_order,
    "check_return_window": check_return_window,
    "inspect_condition_photos": inspect_condition_photos,
    "get_customer_ltv": get_customer_ltv,
    "issue_refund_or_replacement": issue_refund_or_replacement,
}
