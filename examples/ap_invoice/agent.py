"""Accounts Payable invoice triage agent.

Given a parsed invoice (vendor, line items, totals, optional PO), decides
approve / hold / reject, assigns GL codes, lists exceptions and an approver tier.

Deliberate sub-optimalities:
- System prompt has no 3-way-match rule, no segregation-of-duties thresholds,
  no calibration on what a P1/P2/P3 approver tier actually means.
- Tool ordering is wrong: model calls `fraud_signals` after deciding, so
  duplicate-invoice bait doesn't trigger holds.
- `fetch_purchase_order` description doesn't say to skip when no PO number is
  present — so the model fetches anyway and wastes a round.
- Defaults to gpt-4o for every invoice; 80% are clean PO matches.
- No JSON schema; `gl_codes` sometimes returned as comma-string.
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

init(service_name="ap-invoice")

_MODEL = os.environ.get("AP_INVOICE_MODEL", "gpt-4o")
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
        "decision": "hold",
        "gl_codes": [],
        "exceptions": ["parse_error"],
        "approver_tier": "T2",
        "reasoning": text,
    }


@entry_point("AP Invoice Processor")
def run(input_data: dict[str, Any]) -> dict[str, Any]:
    invoice = input_data.get("invoice", input_data)
    user_msg = f"Process this invoice:\n{json.dumps(invoice, indent=2)}\n\nDecide approve / hold / reject."

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
        "decision": "hold",
        "gl_codes": [],
        "exceptions": ["max_rounds"],
        "approver_tier": "T2",
        "reasoning": "Max tool rounds reached without a final answer.",
    }
