"""Tools for the ap_invoice agent.

Tool docstrings are deliberately thin so Overmind has room to teach the model
when each lookup is actually relevant (e.g. skip PO fetch when no PO# given).
"""

from __future__ import annotations

import json
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


@tool("lookup_vendor")
def lookup_vendor(vendor_name: str) -> dict[str, Any]:
    """Look up vendor."""
    data = _load("vendors.json")
    key = vendor_name.lower().strip()
    for vid, v in data.items():
        if v["name"].lower() == key or vid.lower() == key:
            return v
    return {"vendor_id": None, "name": vendor_name, "approved": False}


@tool("fetch_purchase_order")
def fetch_purchase_order(po_number: str) -> dict[str, Any]:
    """Fetch a purchase order."""
    data = _load("pos.json")
    return data.get(po_number, {"po_number": po_number, "found": False})


@tool("gl_code_classifier")
def gl_code_classifier(line_description: str) -> dict[str, Any]:
    """Classify a line item to a GL code."""
    desc = (line_description or "").lower()
    rules = [
        (("aws", "gcp", "azure", "cloud", "hosting"), "6105", "Cloud Infrastructure"),
        (("saas", "subscription", "license", "seats"), "6110", "Software Subscriptions"),
        (("legal", "attorney", "counsel"), "6210", "Legal Fees"),
        (("consult", "advisory", "audit"), "6220", "Professional Services"),
        (("travel", "hotel", "flight", "uber", "lyft"), "6310", "Travel"),
        (("meals", "lunch", "dinner", "catering"), "6320", "Meals & Entertainment"),
        (("office", "stationery", "supplies"), "6410", "Office Supplies"),
        (("marketing", "ads", "campaign"), "6510", "Marketing"),
    ]
    for keywords, code, name in rules:
        if any(k in desc for k in keywords):
            return {"gl_code": code, "gl_name": name, "confidence": 0.85}
    return {"gl_code": "6900", "gl_name": "Other Expenses", "confidence": 0.4}


@tool("fraud_signals")
def fraud_signals(invoice: dict[str, Any]) -> dict[str, Any]:
    """Run fraud signals on an invoice."""
    history = _load("invoice_history.json")
    inv_no = invoice.get("invoice_number", "")
    vendor_name = invoice.get("vendor_name", "")
    total = float(invoice.get("total", 0) or 0)
    submitted = invoice.get("submitted_at", "")

    flags: list[str] = []
    seen = history.get(vendor_name, [])
    if inv_no in seen:
        flags.append("duplicate_invoice_number")
    if total > 0 and total == round(total) and total >= 1000 and total % 1000 == 0:
        flags.append("suspiciously_round_total")
    if submitted and submitted.endswith(("Sat", "Sun")):
        flags.append("weekend_submission")
    if total > 25000:
        flags.append("large_amount")
    return {"flags": flags, "risk": "high" if len(flags) >= 2 else ("medium" if flags else "low")}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_vendor",
            "description": "Look up vendor.",
            "parameters": {
                "type": "object",
                "properties": {"vendor_name": {"type": "string"}},
                "required": ["vendor_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_purchase_order",
            "description": "Fetch a purchase order.",
            "parameters": {
                "type": "object",
                "properties": {"po_number": {"type": "string"}},
                "required": ["po_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gl_code_classifier",
            "description": "Classify a line item to a GL code.",
            "parameters": {
                "type": "object",
                "properties": {"line_description": {"type": "string"}},
                "required": ["line_description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fraud_signals",
            "description": "Run fraud signals on an invoice.",
            "parameters": {
                "type": "object",
                "properties": {"invoice": {"type": "object"}},
                "required": ["invoice"],
            },
        },
    },
]


TOOL_FNS = {
    "lookup_vendor": lookup_vendor,
    "fetch_purchase_order": fetch_purchase_order,
    "gl_code_classifier": gl_code_classifier,
    "fraud_signals": fraud_signals,
}
