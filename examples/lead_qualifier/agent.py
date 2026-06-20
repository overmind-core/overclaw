"""Lead qualifier agent.

Classifies an inbound sales lead using company enrichment.

Deliberate sub-optimalities Overmind should be able to improve:
- System prompt has no calibration bands for hot/warm/cold
- Tool descriptions are vague
- Uses an expensive model for a small classification task
- No schema validation; JSON parsing is best-effort
- Always calls exa_search even when the CRM lookup already answered it
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from overmind import entry_point, init
from prompts import SYSTEM_PROMPT
from tools import TOOL_FNS, TOOL_SCHEMAS

load_dotenv()

init(service_name="lead-qualifier")

_MODEL = os.environ.get("LEAD_QUALIFIER_MODEL", "gpt-4o")
_MAX_TOOL_ROUNDS = 6


def _client() -> OpenAI:
    return OpenAI()


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
        "category": "warm",
        "lead_score": 50,
        "reasoning": text,
        "next_action": "follow up",
    }


@entry_point("Lead Qualifier")
def run(input_data: dict[str, Any]) -> dict[str, Any]:
    company = input_data.get("company_name", "")
    inquiry = input_data.get("inquiry_text", "")
    role = input_data.get("contact_role", "unknown")

    user_msg = f"Company: {company}\nContact role: {role}\nInquiry: {inquiry}\n\nQualify this lead."

    client = _client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    for _ in range(_MAX_TOOL_ROUNDS):
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            temperature=0.2,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return _extract_json(msg.content or "")

        for tc in msg.tool_calls:
            fn = TOOL_FNS.get(tc.function.name)
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = fn(**args) if fn else {"error": f"unknown tool {tc.function.name}"}
            except Exception as e:
                result = {"error": str(e)}
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    return {
        "category": "warm",
        "lead_score": 50,
        "reasoning": "Max tool rounds reached without a final answer.",
        "next_action": "manual review",
    }
