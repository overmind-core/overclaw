"""Tools for the oncall_triage investigator subagent.

Tool descriptions are deliberately weak; in particular, nothing tells the model
that `search_runbook` should usually be called first, before paging humans or
fetching deploys.
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


@tool("fetch_logs")
def fetch_logs(service: str, minutes: int = 15) -> dict[str, Any]:
    """Fetch logs for a service."""
    data = _load("logs.json")
    entries = data.get(service, [])
    return {"service": service, "minutes": minutes, "entries": entries[-50:]}


@tool("query_metrics")
def query_metrics(service: str, metric: str) -> dict[str, Any]:
    """Query metrics."""
    data = _load("metrics.json")
    svc = data.get(service, {})
    return {"service": service, "metric": metric, "data": svc.get(metric, [])}


@tool("search_runbook")
def search_runbook(service: str) -> dict[str, Any]:
    """Search the runbook."""
    data = _load("runbooks.json")
    return data.get(service, {"service": service, "runbook": None})


@tool("get_recent_deploys")
def get_recent_deploys(service: str) -> dict[str, Any]:
    """Get recent deploys."""
    data = _load("deploys.json")
    return {"service": service, "deploys": data.get(service, [])}


@tool("post_status_update")
def post_status_update(message: str, severity: str) -> dict[str, Any]:
    """Post a status update."""
    return {"posted": True, "message": message, "severity": severity}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_logs",
            "description": "Fetch logs for a service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "minutes": {"type": "integer"},
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_metrics",
            "description": "Query metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "metric": {"type": "string"},
                },
                "required": ["service", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_runbook",
            "description": "Search the runbook.",
            "parameters": {
                "type": "object",
                "properties": {"service": {"type": "string"}},
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_deploys",
            "description": "Get recent deploys.",
            "parameters": {
                "type": "object",
                "properties": {"service": {"type": "string"}},
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "post_status_update",
            "description": "Post a status update.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "severity": {"type": "string"},
                },
                "required": ["message", "severity"],
            },
        },
    },
]


TOOL_FNS = {
    "fetch_logs": fetch_logs,
    "query_metrics": query_metrics,
    "search_runbook": search_runbook,
    "get_recent_deploys": get_recent_deploys,
    "post_status_update": post_status_update,
}
