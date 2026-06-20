"""On-call incident triage agent (multi-agent: router -> investigator -> responder).

Given a raw alert payload, produces a structured triage: severity, hypothesis,
suggested actions, public status message, escalation decision.

Deliberate sub-optimalities:
- Router prompt is a verbose paragraph (rambling > directive).
- Investigator has _MAX_TOOL_ROUNDS = 12 and no guidance to call search_runbook
  first; ends up paging humans / fetching logs before reading the runbook.
- Responder uses claude-opus to write a 2-line Slack status; obvious downgrade.
- Hand-off between subagents is free-text — fragile / lossy.
- No SEV rubric anywhere; severity is wildly miscalibrated on seed eval.
- No JSON schema; `suggested_actions` sometimes a string, sometimes a list.
"""

from __future__ import annotations

import json
import os
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI
from overmind import entry_point, init, workflow
from prompts import INVESTIGATOR_PROMPT, RESPONDER_PROMPT, ROUTER_PROMPT
from tools import TOOL_FNS, TOOL_SCHEMAS

load_dotenv()

init(service_name="oncall-triage")

_ROUTER_MODEL = os.environ.get("ONCALL_ROUTER_MODEL", "gpt-4o")
_INVESTIGATOR_MODEL = os.environ.get("ONCALL_INVESTIGATOR_MODEL", "gpt-4o")
_RESPONDER_MODEL = os.environ.get("ONCALL_RESPONDER_MODEL", "claude-sonnet-4-6")
_MAX_TOOL_ROUNDS = 12


def _openai() -> OpenAI:
    return OpenAI()


def _anthropic() -> Anthropic:
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
        "severity": "SEV3",
        "hypothesis": text,
        "suggested_actions": [],
        "status_message": "We are investigating an issue.",
        "escalate": False,
    }


@workflow("router")
def _route(alert: dict[str, Any]) -> str:
    client = _openai()
    resp = client.chat.completions.create(
        model=_ROUTER_MODEL,
        messages=[
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": f"Alert payload:\n{json.dumps(alert, indent=2)}"},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content or ""


@workflow("investigator")
def _investigate(brief: str, alert: dict[str, Any]) -> str:
    client = _openai()
    user_msg = f"Router brief:\n{brief}\n\nOriginal alert payload:\n{json.dumps(alert, indent=2)}\n\nInvestigate."
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": INVESTIGATOR_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    for _ in range(_MAX_TOOL_ROUNDS):
        resp = client.chat.completions.create(
            model=_INVESTIGATOR_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            temperature=0.2,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content or ""

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
    return "Investigation truncated at max rounds."


@workflow("responder")
def _respond(findings: str, alert: dict[str, Any]) -> dict[str, Any]:
    client = _anthropic()
    resp = client.messages.create(
        model=_RESPONDER_MODEL,
        max_tokens=1200,
        system=RESPONDER_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Investigator findings:\n{findings}\n\n"
                    f"Original alert:\n{json.dumps(alert, indent=2)}\n\n"
                    "Return the final triage JSON."
                ),
            }
        ],
    )
    text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return _extract_json("\n".join(text_parts))


@entry_point("On-Call Triage")
def run(input_data: dict[str, Any]) -> dict[str, Any]:
    alert = input_data.get("alert", input_data)
    brief = _route(alert)
    findings = _investigate(brief, alert)
    return _respond(findings, alert)
