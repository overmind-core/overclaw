"""Tools for the clinical_coder agent.

Tool docstrings are deliberately thin — Overmind should rewrite them so the
model knows when each lookup is actually useful.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from overmind import tool

_DATA_DIR = Path(__file__).parent / "data"


def _load(name: str) -> dict[str, Any]:
    path = _DATA_DIR / name
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


@tool("search_icd10")
def search_icd10(query: str) -> dict[str, Any]:
    """Search ICD-10 diagnosis codes."""
    data = _load("icd10.json")
    q = (query or "").lower()
    hits = [
        {"code": code, "description": desc}
        for code, desc in data.items()
        if q and (q in desc.lower() or q in code.lower())
    ]
    return {"results": hits[:5]}


@tool("search_cpt")
def search_cpt(query: str) -> dict[str, Any]:
    """Search CPT procedure codes."""
    data = _load("cpt.json")
    q = (query or "").lower()
    hits = [
        {"code": code, "description": desc}
        for code, desc in data.items()
        if q and (q in desc.lower() or q in code.lower())
    ]
    return {"results": hits[:5]}


@tool("check_payer_policy")
def check_payer_policy(payer: str, cpt_code: str) -> dict[str, Any]:
    """Check payer policy for a CPT code."""
    data = _load("payer_policies.json")
    payer_data = data.get(payer.lower(), {})
    pol = payer_data.get(cpt_code, {})
    return {
        "payer": payer,
        "cpt_code": cpt_code,
        "auth_required": pol.get("auth_required", False),
        "documentation_requirements": pol.get("docs", []),
        "notes": pol.get("notes", "no policy on file"),
    }


@tool("lookup_member_eligibility")
def lookup_member_eligibility(member_id: str) -> dict[str, Any]:
    """Look up member eligibility."""
    data = _load("members.json")
    return data.get(member_id, {"member_id": member_id, "active": False})


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_icd10",
            "description": "Search ICD-10 diagnosis codes.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_cpt",
            "description": "Search CPT procedure codes.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_payer_policy",
            "description": "Check payer policy for a CPT code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payer": {"type": "string"},
                    "cpt_code": {"type": "string"},
                },
                "required": ["payer", "cpt_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_member_eligibility",
            "description": "Look up member eligibility.",
            "parameters": {
                "type": "object",
                "properties": {"member_id": {"type": "string"}},
                "required": ["member_id"],
            },
        },
    },
]


TOOL_FNS = {
    "search_icd10": search_icd10,
    "search_cpt": search_cpt,
    "check_payer_policy": check_payer_policy,
    "lookup_member_eligibility": lookup_member_eligibility,
}
