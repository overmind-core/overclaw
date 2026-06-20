"""SEO content brief generator (multi-agent: researcher -> outliner -> editor).

Given a topic, audience, and primary keyword, produces a content brief
(title options, target keywords, outline, FAQs, internal links, sources).

Deliberate sub-optimalities:
- Researcher pulls 20 search results then summarises all of them — wastes
  tokens. Overmind should cap source count and add a relevance filter.
- Outliner uses gpt-4o; Editor uses gpt-4o. Both downgradable.
- No audience-persona block in any prompt — outputs read generic.
- Stages communicate via free-text Markdown — Overmind can impose a JSON
  hand-off contract for at least the researcher -> outliner stage.
- `keyword_metrics_lookup` description is "useful for SEO stuff" so the model
  uses it inconsistently.
- No JSON schema enforcement on the final brief.
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from overmind import entry_point, init, workflow
from prompts import EDITOR_PROMPT, OUTLINER_PROMPT, RESEARCHER_PROMPT
from tools import TOOL_FNS, TOOL_SCHEMAS

load_dotenv()

init(service_name="research-brief")

_RESEARCHER_MODEL = os.environ.get("RESEARCH_BRIEF_RESEARCHER_MODEL", "gpt-4o")
_OUTLINER_MODEL = os.environ.get("RESEARCH_BRIEF_OUTLINER_MODEL", "gpt-4o")
_EDITOR_MODEL = os.environ.get("RESEARCH_BRIEF_EDITOR_MODEL", "gpt-4o")
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
        "title_options": [],
        "target_keywords": [],
        "outline": [],
        "faqs": [],
        "internal_link_suggestions": [],
        "sources": [],
        "raw": text,
    }


@workflow("researcher")
def _research(topic: str, audience: str, keyword: str) -> str:
    client = _client()
    user_msg = f"Topic: {topic}\nTarget audience: {audience}\nPrimary keyword: {keyword}\n\nDo the research."
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": RESEARCHER_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    for _ in range(_MAX_TOOL_ROUNDS):
        resp = client.chat.completions.create(
            model=_RESEARCHER_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            temperature=0.3,
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
    return "Research truncated at max rounds."


@workflow("outliner")
def _outline(notes: str, topic: str, audience: str) -> str:
    client = _client()
    resp = client.chat.completions.create(
        model=_OUTLINER_MODEL,
        messages=[
            {"role": "system", "content": OUTLINER_PROMPT},
            {
                "role": "user",
                "content": (f"Topic: {topic}\nAudience: {audience}\n\nResearch notes:\n{notes}"),
            },
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content or ""


@workflow("editor")
def _edit(notes: str, outline: str, topic: str, audience: str, keyword: str) -> dict[str, Any]:
    client = _client()
    resp = client.chat.completions.create(
        model=_EDITOR_MODEL,
        messages=[
            {"role": "system", "content": EDITOR_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}\nAudience: {audience}\nPrimary keyword: {keyword}\n\n"
                    f"Research notes:\n{notes}\n\nOutline:\n{outline}\n\n"
                    "Return the final brief JSON."
                ),
            },
        ],
        temperature=0.3,
    )
    return _extract_json(resp.choices[0].message.content or "")


@entry_point("Research Brief Generator")
def run(input_data: dict[str, Any]) -> dict[str, Any]:
    topic = input_data.get("topic", "")
    audience = input_data.get("target_audience", "general readers")
    keyword = input_data.get("primary_keyword", topic)

    notes = _research(topic, audience, keyword)
    outline = _outline(notes, topic, audience)
    return _edit(notes, outline, topic, audience, keyword)
