from __future__ import annotations

import os
from typing import Any

import httpx
from overmind import tool


@tool("search_public_docs")
def search_public_docs(query: str) -> dict[str, Any]:
    """Search public docs on the web."""
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        return {"results": [], "error": "EXA_API_KEY not set"}
    try:
        resp = httpx.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={
                "query": query,
                "numResults": 3,
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
