"""Returns / RMA concierge agent.

Decides refund / replace / deny / escalate on a customer return request and
drafts the customer-facing response.

Deliberate sub-optimalities:
- System prompt has no policy table for final-sale items, abuse patterns, or
  LTV-based goodwill thresholds.
- `customer_message` has no tone or length guidance — replies come back
  robotic and inconsistent.
- Always calls `inspect_condition_photos`, even when the customer didn't
  attach photos (sometimes hallucinates damage from a None URL).
- Defaults to gpt-4o; 90% of requests are within-window same-decision cases.
- Schema drift: `amount` returned as "19.99", 19.99, or "$19.99" depending on
  the case.
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

init(service_name="returns-concierge")

_MODEL = os.environ.get("RETURNS_MODEL", "gpt-4o")
_MAX_TOOL_ROUNDS = 8


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
        "decision": "escalate",
        "amount": 0,
        "restocking_fee": 0,
        "customer_message": text,
        "reasoning": "parse_error",
    }


@entry_point("Returns Concierge")
def run(input_data: dict[str, Any]) -> dict[str, Any]:
    user_msg = f"Return request:\n{json.dumps(input_data, indent=2)}\n\nMake a decision and reply to the customer."

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
            temperature=0.3,
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
        "decision": "escalate",
        "amount": 0,
        "restocking_fee": 0,
        "customer_message": "Our team is looking into your request and will follow up.",
        "reasoning": "Max tool rounds reached without a final answer.",
    }
