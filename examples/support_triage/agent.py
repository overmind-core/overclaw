"""Support ticket triage agent (Anthropic tool use).

Deliberate sub-optimalities:
- System prompt has no priority or escalation rubric
- Tool descriptions are one-liners - the model picks the wrong tool or order
- No tone / length guidance for `suggested_response`
- Uses a large Claude model for what's essentially a classifier
- Prefers `search_public_docs` (web) over `search_kb` (authoritative internal)
- No schema enforcement on the returned JSON
"""

from __future__ import annotations

import json
import os
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv
from prompts import SYSTEM_PROMPT
from tools import TOOL_FNS, TOOL_SCHEMAS

load_dotenv()

_MODEL = os.environ.get("SUPPORT_TRIAGE_MODEL", "claude-sonnet-4-6")
_MAX_TOOL_ROUNDS = 8


def _client() -> Anthropic:
    return Anthropic()


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        text = text.removeprefix("json")
        text = text.rsplit("```", 1)[0]

    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        candidates.append(text[start : end + 1])

    for chunk in candidates:
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            continue

    return {
        "category": "other",
        "priority": "P3",
        "escalate": False,
        "suggested_response": text,
        "tags": [],
    }


def run(input_data: dict[str, Any]) -> dict[str, Any]:
    customer_id = input_data.get("customer_id", "")
    subject = input_data.get("subject", "")
    body = input_data.get("body", "")

    user_text = (
        f"Ticket\nCustomer: {customer_id}\nSubject: {subject}\n\n{body}\n\nTriage this ticket and return the JSON."
    )

    client = _client()
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_text}]

    for _ in range(_MAX_TOOL_ROUNDS):
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            return _extract_json("\n".join(text_parts))

        tool_results: list[dict[str, Any]] = []
        for block in resp.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            fn = TOOL_FNS.get(block.name)
            try:
                result = fn(**block.input) if fn else {"error": f"unknown tool {block.name}"}
            except Exception as e:
                result = {"error": str(e)}
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })
        messages.append({"role": "user", "content": tool_results})

    return {
        "category": "other",
        "priority": "P3",
        "escalate": False,
        "suggested_response": "Our team is looking into your ticket and will follow up shortly.",
        "tags": ["max-rounds"],
    }


if __name__ == "__main__":
    # Trace this run to Overmind. ``agent_id`` (the capability's UUID from the
    # Console) is the identifier — resolved first and stable through renames.
    # ``agent_name`` is a display label the server resolves via its alias table.
    import overmind

    overmind.init(
        agent_id=os.environ.get("OVERMIND_AGENT_ID"),  # e.g. "6f1c…"
        agent_name="Support Triage",
        service_name="support-triage",
        environment=os.environ.get("ENVIRONMENT", "development"),
        providers=["anthropic"],  # auto-instrument the Anthropic SDK
    )

    result = run({
        "customer_id": "cust_123",
        "subject": "Refund never arrived",
        "body": "I returned my order two weeks ago and still have not been refunded.",
    })
    print(json.dumps(result, indent=2))
    overmind.force_flush_traces()
