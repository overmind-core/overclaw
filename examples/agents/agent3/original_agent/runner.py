"""Main agent entry: single LLM call, no tools."""

from overclaw.core.tracer import call_llm

from agents.agent3.original_agent.config import MODEL, SYSTEM_PROMPT, TOOLS
from agents.agent3.original_agent.parsing import format_input, parse_output


def run(input_data: dict) -> dict:
    """Main agent entry point. Single LLM call; no tools."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_input(input_data)},
    ]

    response = call_llm(model=MODEL, messages=messages, tools=TOOLS)
    message = response.choices[0].message
    return parse_output(message.content or "")
