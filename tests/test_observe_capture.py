"""Capture modes on ``observe``/``start_span`` — the payload surface the
adapters used to hand-roll: none/messages modes, format hooks, ignored
arguments, and message normalisation."""

from __future__ import annotations

import asyncio
import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import overmind.tracing as tracing
from overmind import attrs
from overmind.tracing import SpanType, normalize_messages, observe, start_span, tool


@pytest.fixture
def exporter(monkeypatch):
    provider = TracerProvider()
    inmem = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(inmem))
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer("overmind", "test"))
    return inmem


def _only_span(exporter):
    (span,) = exporter.get_finished_spans()
    return span


# ---------------------------------------------------------------------------
# capture modes
# ---------------------------------------------------------------------------


def test_capture_auto_records_scrubbed_inputs_and_outputs(exporter):
    @observe("op")
    def op(query, api_key):
        return {"hits": [query]}

    op("refunds", api_key="sk-1")
    span = _only_span(exporter)
    assert json.loads(span.attributes["inputs"]) == {"query": "refunds", "api_key": "<redacted>"}
    assert json.loads(span.attributes["outputs"]) == {"hits": ["refunds"]}


def test_capture_none_records_no_payloads(exporter):
    @observe("op", capture="none")
    def op(secret):
        return secret

    op("s3cret")
    span = _only_span(exporter)
    assert "inputs" not in span.attributes
    assert "outputs" not in span.attributes
    assert span.attributes[attrs.STATUS] == "success"


def test_capture_messages_normalises_input_and_list_output(exporter):
    @observe("llm", type="llm", capture="messages")
    def call(messages, temperature=0.2):
        return [{"role": "ai", "content": "hi there"}]

    call([{"role": "human", "content": "hello"}])
    span = _only_span(exporter)
    assert json.loads(span.attributes["inputs"]) == {"messages": [{"role": "user", "content": "hello"}]}
    assert json.loads(span.attributes["outputs"]) == {"messages": [{"role": "assistant", "content": "hi there"}]}


def test_capture_validates_mode():
    with pytest.raises(ValueError, match="capture"):
        observe(capture="everything")


def test_ignore_drops_named_arguments(exporter):
    @observe("op", ignore=("browser_session", "llm"))
    def op(query, browser_session, llm=None):
        return query

    op("q", object())
    span = _only_span(exporter)
    assert json.loads(span.attributes["inputs"]) == {"query": "q"}


def test_format_hooks_shape_payloads(exporter):
    @observe(
        "op",
        format_input=lambda bound: {"url_count": len(bound["urls"])},
        format_output=lambda result, bound: {"pages": len(result), "query": bound.get("query")},
    )
    def scrape(urls, query=None):
        return ["page"] * 3

    scrape(["a", "b"], query="refunds")
    span = _only_span(exporter)
    assert json.loads(span.attributes["inputs"]) == {"url_count": 2}
    assert json.loads(span.attributes["outputs"]) == {"pages": 3, "query": "refunds"}


def test_failing_format_hook_is_best_effort(exporter):
    @observe("op", format_output=lambda result, bound: 1 / 0)
    def op():
        return "ok"

    assert op() == "ok"
    span = _only_span(exporter)
    assert "outputs" not in span.attributes
    assert span.attributes[attrs.STATUS] == "success"


# ---------------------------------------------------------------------------
# span type coercion + naming
# ---------------------------------------------------------------------------


def test_string_span_types_accepted(exporter):
    @observe("op", type="tool")
    def op():
        return 1

    op()
    assert _only_span(exporter).attributes[attrs.SPAN_TYPE] == SpanType.TOOL.value


def test_unknown_span_type_raises():
    with pytest.raises(ValueError, match="span type"):
        observe(type="banana")


def test_default_span_name_is_qualname(exporter):
    class Agent:
        @observe()
        def step(self):
            return 1

    Agent().step()
    span = _only_span(exporter)
    assert span.name.endswith("Agent.step")


def test_tool_name_stays_short(exporter):
    class Tools:
        @tool()
        def search_kb(self, query):
            return query

    Tools().search_kb("q")
    span = _only_span(exporter)
    assert span.attributes[attrs.TOOL_NAME] == "search_kb"
    assert span.name.endswith("Tools.search_kb")


def test_start_span_accepts_string_type_and_attributes(exporter):
    with start_span("model.call", span_type="llm", attributes={"inputs": {"messages": []}}):
        pass
    span = _only_span(exporter)
    assert span.attributes[attrs.SPAN_TYPE] == "llm_call"
    assert span.attributes[attrs.PROVENANCE] == "agent"
    assert json.loads(span.attributes["inputs"]) == {"messages": []}


# ---------------------------------------------------------------------------
# dynamic span naming — polymorphic dispatchers
# ---------------------------------------------------------------------------


def test_tool_callable_name_emits_per_action_tool_spans(exporter):
    """The browser-use shape: one dispatcher method executing named actions
    must emit per-action tool spans with the full @tool shape."""

    class Tools:
        @tool(name=lambda self, action, **params: action)
        def act(self, action: str, **params):
            return f"{action}:ok"

    tools = Tools()
    assert tools.act("navigate", url="https://ex.io") == "navigate:ok"
    assert tools.act("done") == "done:ok"

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert set(spans) == {"navigate", "done"}
    for name, span in spans.items():
        assert span.attributes[attrs.SPAN_TYPE] == "tool_call"
        assert span.attributes[attrs.TOOL_NAME] == name
        assert span.attributes[attrs.PROVENANCE] == "environment"
        assert span.attributes[attrs.CODE_FUNCTION_NAME].endswith("Tools.act")
    assert json.loads(spans["navigate"].attributes["inputs"]) == {"action": "navigate", "url": "https://ex.io"}
    assert spans["navigate"].attributes[attrs.TOOL_ARG_KEYS] == ("action", "url")


def test_tool_callable_name_supports_async(exporter):
    @tool(name=lambda action: f"do:{action}")
    async def act(action: str) -> str:
        return action

    assert asyncio.run(act("extract")) == "extract"
    span = _only_span(exporter)
    assert span.name == "do:extract"
    assert span.attributes[attrs.TOOL_NAME] == "do:extract"


def test_tool_name_callable_failure_falls_back_to_function(exporter):
    @tool(name=lambda action: action.name)  # str has no .name
    def act(action: str) -> str:
        return action

    assert act("x") == "x"
    span = _only_span(exporter)
    assert span.name.endswith("act")
    assert span.attributes[attrs.TOOL_NAME] == "act"
    assert span.attributes[attrs.STATUS] == "success"


# ---------------------------------------------------------------------------
# normalize_messages
# ---------------------------------------------------------------------------


def test_normalize_messages_handles_dicts_objects_and_parts():
    class _Msg:
        def __init__(self):
            self.type = "ai"
            self.content = [{"text": "part one"}, {"image": "..."}, "part two"]
            self.tool_calls = [{"name": "search", "args": {"q": "x"}}]

    out = normalize_messages([
        {"role": "human", "content": "hello"},
        _Msg(),
        {"role": "tool", "content": "result", "tool_call_id": "tc-1"},
    ])
    assert out == [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "part one\npart two",
            "tool_calls": [{"name": "search", "args": {"q": "x"}}],
        },
        {"role": "tool", "content": "result", "tool_call_id": "tc-1"},
    ]


def test_normalize_messages_empty_and_none():
    assert normalize_messages(None) == []
    assert normalize_messages([]) == []
