"""Tools for the lead qualifier agent.

The docstrings / schemas here are deliberately minimal — Overmind should be
able to rewrite them into something clearer and more actionable.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import httpx
from overmind import tool

_DATA_DIR = Path(__file__).parent / "data"


@tool("lookup_company_size")
def lookup_company_size(domain: str) -> dict[str, Any]:
    """Look up company size from internal CRM data."""
    path = _DATA_DIR / "companies.csv"
    key = domain.lower().strip().removeprefix("www.")
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["domain"].lower() == key:
                return {
                    "domain": row["domain"],
                    "employees": int(row["employees"]),
                    "industry": row["industry"],
                    "is_competitor": row["is_competitor"] == "true",
                }
    return {
        "domain": domain,
        "employees": None,
        "industry": None,
        "is_competitor": False,
    }


@tool("exa_search_company")
def exa_search_company(company: str) -> dict[str, Any]:
    """Search the web for a company."""
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        return {"results": [], "error": "EXA_API_KEY not set"}
    try:
        resp = httpx.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={
                "query": f"{company} company overview",
                "numResults": 3,
                "useAutoprompt": True,
                "contents": {"text": {"maxCharacters": 500}},
            },
            timeout=20.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "results": [
                {
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "snippet": (r.get("text") or "")[:400],
                }
                for r in data.get("results", [])
            ]
        }
    except Exception as e:
        return {"results": [], "error": str(e)}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_company_size",
            "description": "Look up company size from internal CRM data.",
            "parameters": {
                "type": "object",
                "properties": {"domain": {"type": "string"}},
                "required": ["domain"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exa_search_company",
            "description": "Search the web for a company.",
            "parameters": {
                "type": "object",
                "properties": {"company": {"type": "string"}},
                "required": ["company"],
            },
        },
    },
]


TOOL_FNS = {
    "lookup_company_size": lookup_company_size,
    "exa_search_company": exa_search_company,
}
