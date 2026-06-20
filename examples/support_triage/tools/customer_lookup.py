from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from overmind import tool

_CUSTOMERS = Path(__file__).resolve().parent.parent / "data" / "customers.json"


@tool("lookup_customer")
def lookup_customer(customer_id: str) -> dict[str, Any]:
    """Look up a customer by id."""
    if not _CUSTOMERS.exists():
        return {"error": "no customer data available"}
    data = json.loads(_CUSTOMERS.read_text())
    for c in data:
        if c["id"] == customer_id:
            return c
    return {"error": f"customer {customer_id} not found"}
