"""Main agent entry: LLM loop with tool calls."""

import json

from overclaw.core.tracer import call_llm, call_tool

from examples.agents.agent4.original_agent.config import MODEL, SYSTEM_PROMPT, TOOLS
from examples.agents.agent4.original_agent.parsing import format_input, parse_output
from examples.agents.agent4.original_agent.tools_impl import TOOL_FUNCTIONS


def run(input_data: dict) -> dict:
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
        assistant_msg = {"role": "assistant", "content": message.content}
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
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)}
            )
        response = call_llm(model=MODEL, messages=messages, tools=TOOLS)
        message = response.choices[0].message
    return parse_output(message.content or "")
