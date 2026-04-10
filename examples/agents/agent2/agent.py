"""
Sample Agent: Lead Qualification

Takes sales lead information (company name, contact email, inquiry text) and
produces a structured qualification assessment.  This file is the optimization
target — OverClaw will iteratively improve it.
"""

import json

from overclaw.core.tracer import call_llm, call_tool

from examples.agents.agent2.config import MODEL, SYSTEM_PROMPT, TOOLS
from examples.agents.agent2.tools import TOOL_FUNCTIONS

# === AGENT METADATA ===

AGENT_DESCRIPTION = """\
Lead Qualification Agent: Takes sales lead information (company_name,
contact_email, inquiry) and returns a structured JSON assessment with:
lead_score (0-100), category (hot/warm/cold), priority (high/medium/low),
recommended_action (schedule_demo/send_info/nurture/disqualify), and reasoning.
"""


# === AGENT LOGIC ===


def format_input(input_data: dict) -> str:
    parts = []
    if "company_name" in input_data:
        parts.append(f"Company: {input_data['company_name']}")
    if "contact_email" in input_data:
        parts.append(f"Contact: {input_data['contact_email']}")
    if "inquiry" in input_data:
        parts.append(f"Inquiry: {input_data['inquiry']}")
    return "\n".join(parts)


def parse_output(content: str) -> dict:
    """Extract JSON from the model's response."""
    if not content:
        return _fallback_output("Empty response")
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    return _fallback_output(content)


def _fallback_output(reason: str) -> dict:
    return {
        "lead_score": 0,
        "category": "unknown",
        "priority": "unknown",
        "recommended_action": "unknown",
        "reasoning": f"Parse failure: {reason[:200]}",
    }


def run(input_data: dict) -> dict:
    """Main agent entry point. Takes lead info, returns structured qualification."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_input(input_data)},
    ]

    response = call_llm(model=MODEL, messages=messages, tools=TOOLS)
    message = response.choices[0].message

    max_tool_rounds = 5
    for _ in range(max_tool_rounds):
        if not message.tool_calls:
            break

        assistant_msg: dict = {"role": "assistant", "content": message.content}
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
        messages.append(assistant_msg)

        for tc in message.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            if fn_name in TOOL_FUNCTIONS:
                result = call_tool(fn_name, fn_args, TOOL_FUNCTIONS[fn_name])
            else:
                result = {"error": f"Unknown tool: {fn_name}"}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

        response = call_llm(model=MODEL, messages=messages, tools=TOOLS)
        message = response.choices[0].message

    return parse_output(message.content or "")
