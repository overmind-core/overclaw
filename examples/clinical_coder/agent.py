"""Clinical coding + prior-auth triage agent.

Given a clinical note and a planned procedure, produces ICD-10 / CPT codes,
modifiers, prior-auth determination and a denial-risk score.

Deliberate sub-optimalities Overmind should be able to improve:
- System prompt has no calibration bands for `denial_risk` (0-100 with no buckets)
- Tool descriptions don't say when to skip the payer-policy call
- Always calls all four tools, even for simple routine codes
- Uses an expensive frontier model for what is mostly a classifier
- No JSON schema enforcement; modifiers come back as free text
- No PHI-redaction guardrail in the system prompt
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

init(service_name="clinical-coder")

_MODEL = os.environ.get("CLINICAL_CODER_MODEL", "gpt-4o")
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
        "icd10": [],
        "cpt": [],
        "modifiers": [],
        "auth_required": False,
        "denial_risk": 50,
        "reasoning": text,
    }


@entry_point("Clinical Coder")
def run(input_data: dict[str, Any]) -> dict[str, Any]:
    note = input_data.get("clinical_note", "")
    procedure = input_data.get("procedure", "")
    payer = input_data.get("payer", "unknown")
    member_id = input_data.get("member_id", "unknown")

    user_msg = (
        f"Member ID: {member_id}\n"
        f"Payer: {payer}\n"
        f"Procedure: {procedure}\n\n"
        f"Clinical note:\n{note}\n\n"
        "Code this encounter."
    )

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
            temperature=0.1,
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
        "icd10": [],
        "cpt": [],
        "modifiers": [],
        "auth_required": True,
        "denial_risk": 60,
        "reasoning": "Max tool rounds reached without a final answer.",
    }
