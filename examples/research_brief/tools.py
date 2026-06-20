"""Tools for the research_brief researcher subagent.

`web_search` is real (EXA) when EXA_API_KEY is set, otherwise falls back to a
local stub. `keyword_metrics_lookup` is fully local. Tool descriptions are
deliberately weak; in particular, nothing tells the model to cap source counts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from overmind import tool

_DATA_DIR = Path(__file__).parent / "data"


def _load(name: str) -> Any:
    path = _DATA_DIR / name
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


@tool("web_search")
def web_search(query: str, num_results: int = 20) -> dict[str, Any]:
    """Search the web."""
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        stub = _load("stub_search.json")
        hits = stub.get(query.lower(), stub.get("__default__", []))
        return {"results": hits[:num_results]}
    try:
        resp = httpx.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={
                "query": query,
                "numResults": num_results,
                "useAutoprompt": True,
                "contents": {"text": {"maxCharacters": 600}},
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
                    "snippet": (r.get("text") or "")[:500],
                }
                for r in data.get("results", [])
            ]
        }
    except Exception as e:
        return {"results": [], "error": str(e)}


@tool("fetch_url")
def fetch_url(url: str) -> dict[str, Any]:
    """Fetch a URL."""
    stub = _load("stub_pages.json")
    if url in stub:
        return {"url": url, "content": stub[url]}
    try:
        resp = httpx.get(url, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
        return {"url": url, "content": resp.text[:4000]}
    except Exception as e:
        return {"url": url, "error": str(e)}


@tool("keyword_metrics_lookup")
def keyword_metrics_lookup(keyword: str) -> dict[str, Any]:
    """Look up keyword metrics."""
    data = _load("keyword_metrics.json")
    return data.get(keyword.lower(), {"keyword": keyword, "monthly_searches": 0, "difficulty": None})


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num_results": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "keyword_metrics_lookup",
            "description": "Look up keyword metrics.",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string"}},
                "required": ["keyword"],
            },
        },
    },
]


TOOL_FNS = {
    "web_search": web_search,
    "fetch_url": fetch_url,
    "keyword_metrics_lookup": keyword_metrics_lookup,
}
