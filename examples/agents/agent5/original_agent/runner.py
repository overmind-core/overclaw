"""Main agent entry: single LLM call, no tools."""

from overclaw.core.tracer import call_llm

from agents.agent5.original_agent.config import MODEL, SYSTEM_PROMPT
from agents.agent5.original_agent.parsing import format_input, parse_output


def run(input_data: dict) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_input(input_data)},
    ]
    response = call_llm(model=MODEL, messages=messages)
    return parse_output(response.choices[0].message.content or "")
