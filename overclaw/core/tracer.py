"""
Tracing infrastructure for capturing LLM calls, tool calls, and agent execution.

Provides a thread-local Tracer that records every LLM and tool invocation
as structured Span objects. The optimizer sets the active tracer before each
agent run and collects the resulting Trace afterwards.
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import litellm

from overclaw.core.litellm_params import llm_completion

_local = threading.local()


@dataclass
class Span:
    span_type: str
    name: str
    start_time: float
    end_time: float = 0.0
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)
    error: str | None = None

    def finish(self):
        self.end_time = time.time()
        self.latency_ms = (self.end_time - self.start_time) * 1000


@dataclass
class Trace:
    trace_id: str
    input_data: dict = field(default_factory=dict)
    output_data: dict = field(default_factory=dict)
    spans: list = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    score: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class Tracer:
    """Collects spans for a single agent execution run."""

    def __init__(self, trace_id: str):
        self.trace = Trace(trace_id=trace_id, start_time=time.time())

    def add_span(self, span: Span):
        self.trace.spans.append(span)

    def set_input(self, data: dict):
        self.trace.input_data = data

    def set_output(self, data: dict):
        self.trace.output_data = data

    def finish(self):
        self.trace.end_time = time.time()
        self.trace.total_latency_ms = (
            self.trace.end_time - self.trace.start_time
        ) * 1000


def set_current_tracer(tracer: Tracer | None):
    _local.tracer = tracer


def get_current_tracer() -> Tracer | None:
    return getattr(_local, "tracer", None)


def call_llm(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    **kwargs,
) -> Any:
    """Traced wrapper around litellm.completion. Captures model, tokens, cost, latency."""
    tracer = get_current_tracer()
    span = Span(span_type="llm_call", name=model, start_time=time.time())

    try:
        response = llm_completion(model, messages, tools=tools, **kwargs)
        span.finish()

        usage = response.usage
        cost = 0.0
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            pass

        tool_calls_data = []
        msg = response.choices[0].message
        if msg.tool_calls:
            tool_calls_data = [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in msg.tool_calls
            ]

        span.metadata = {
            "model": model,
            "messages_count": len(messages),
            "tools_provided": [t["function"]["name"] for t in (tools or [])],
            "response_content": msg.content,
            "tool_calls": tool_calls_data,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "cost": cost,
        }

        if tracer:
            tracer.add_span(span)
            tracer.trace.total_tokens += usage.total_tokens if usage else 0
            tracer.trace.total_cost += cost

        return response

    except Exception as e:
        span.finish()
        span.error = str(e)
        if tracer:
            tracer.add_span(span)
        raise


def call_tool(name: str, args: dict, fn: Any) -> Any:
    """Traced wrapper for tool execution. Captures args, result, and latency."""
    tracer = get_current_tracer()
    span = Span(span_type="tool_call", name=name, start_time=time.time())

    try:
        result = fn(**args)
        span.finish()
        span.metadata = {"args": args, "result": result}
        if tracer:
            tracer.add_span(span)
        return result

    except Exception as e:
        span.finish()
        span.error = str(e)
        span.metadata = {"args": args}
        if tracer:
            tracer.add_span(span)
        raise
