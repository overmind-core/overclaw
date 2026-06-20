from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from overmind import tool

_KB_DIR = Path(__file__).resolve().parent.parent / "kb"


def _load_kb() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    if not _KB_DIR.exists():
        return docs
    for p in sorted(_KB_DIR.glob("*.json")):
        docs.append(json.loads(p.read_text()))
    return docs


@tool("search_kb")
def search_kb(query: str) -> dict[str, Any]:
    """Search the knowledge base."""
    q = query.lower()
    docs = _load_kb()
    scored: list[tuple[int, dict[str, Any]]] = []
    for doc in docs:
        hay = (doc.get("title", "") + " " + doc.get("body", "") + " " + " ".join(doc.get("tags", []))).lower()
        score = sum(1 for tok in q.split() if tok in hay)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda x: -x[0])
    return {
        "results": [
            {
                "id": d["id"],
                "title": d["title"],
                "snippet": d["body"][:300],
                "tags": d.get("tags", []),
            }
            for _, d in scored[:3]
        ]
    }
