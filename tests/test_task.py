"""Wire-contract tests for ``overmind.task()`` and the ``name=`` anchor identity.

``task()`` must stamp ``overmind.behaviour.key`` on an ``entry_point`` unit
span (decorator and context-manager forms, sync and async); the ``name=``
parameter on decorators must stamp ``overmind.anchor.name`` while the
qualname stays the default when ``name`` is absent. Uses the repo's
in-memory span exporter pattern.
"""

from __future__ import annotations

import asyncio

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from overmind import attrs
from overmind.tracing import SpanType, function, task, tool


@pytest.fixture
def inmem(monkeypatch):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr("overmind.tracing.get_tracer", lambda: provider.get_tracer("test"))
    return provider, exporter


def _only_span(inmem):
    provider, exporter = inmem
    provider.force_flush()
    (span,) = exporter.get_finished_spans()
    return span


# ---------------------------------------------------------------------------
# task() — decorator form
# ---------------------------------------------------------------------------


def test_task_decorator_stamps_key_and_entry_point_type(inmem):
    @task("invoice-triage")
    def run_agent(query):
        return query

    assert run_agent("q") == "q"
    span = _only_span(inmem)
    assert span.attributes[attrs.BEHAVIOUR_KEY] == "invoice-triage"
    assert span.attributes[attrs.SPAN_TYPE] == SpanType.ENTRY_POINT.value
    assert span.attributes["inputs"] == '{"query": "q"}'
    assert span.attributes["outputs"] == '"q"'


def test_task_decorator_capture_io_false_preserves_binding_metadata(inmem):
    @task("invoice-triage", capture_io=False)
    def run_agent(query):
        return query

    assert run_agent("q") == "q"
    span = _only_span(inmem)
    assert span.attributes[attrs.BEHAVIOUR_KEY] == "invoice-triage"
    assert span.attributes[attrs.SPAN_TYPE] == SpanType.ENTRY_POINT.value
    assert "inputs" not in span.attributes
    assert "outputs" not in span.attributes


def test_task_decorator_stamps_key_on_async_function(inmem):
    @task("invoice-triage")
    async def run_agent(query):
        return query

    assert asyncio.run(run_agent("q")) == "q"
    span = _only_span(inmem)
    assert span.attributes[attrs.BEHAVIOUR_KEY] == "invoice-triage"
    assert span.attributes[attrs.SPAN_TYPE] == SpanType.ENTRY_POINT.value


def test_task_decorator_selects_dynamic_key_and_preserves_identity_and_io(inmem):
    @task(key_from=lambda request: request["task_key"])
    def run_agent(request):
        return request["query"]

    assert run_agent({"task_key": "invoice-triage", "query": "q"}) == "q"
    span = _only_span(inmem)
    assert span.attributes[attrs.BEHAVIOUR_KEY] == "invoice-triage"
    assert span.attributes[attrs.CODE_NAMESPACE] == __name__
    assert span.attributes[attrs.CODE_FUNCTION_NAME].endswith("run_agent")
    assert span.attributes["inputs"] == '{"request": {"task_key": "invoice-triage", "query": "q"}}'
    assert span.attributes["outputs"] == '"q"'


def test_task_decorator_selects_dynamic_key_on_async_function(inmem):
    @task(key_from=lambda query: query["task_key"])
    async def run_agent(query):
        return query["value"]

    assert asyncio.run(run_agent({"task_key": "invoice-triage", "value": "q"})) == "q"
    span = _only_span(inmem)
    assert span.attributes[attrs.BEHAVIOUR_KEY] == "invoice-triage"
    assert span.attributes[attrs.SPAN_TYPE] == SpanType.ENTRY_POINT.value


def test_task_decorator_accepts_name_kwarg(inmem):
    @task("invoice-triage", name="triage_run")
    def run_agent(query):
        return query

    run_agent("q")
    span = _only_span(inmem)
    assert span.attributes[attrs.BEHAVIOUR_KEY] == "invoice-triage"
    assert span.attributes[attrs.ANCHOR_NAME] == "triage_run"
    assert span.name == "triage_run"


# ---------------------------------------------------------------------------
# task() — context-manager form
# ---------------------------------------------------------------------------


def test_task_context_manager_stamps_key_and_entry_point_type(inmem):
    with task("payment-collect"):
        pass

    span = _only_span(inmem)
    assert span.attributes[attrs.BEHAVIOUR_KEY] == "payment-collect"
    assert span.attributes[attrs.SPAN_TYPE] == SpanType.ENTRY_POINT.value


def test_task_context_manager_stamps_anchor_name_when_given(inmem):
    with task("payment-collect", name="collect_run"):
        pass

    span = _only_span(inmem)
    assert span.attributes[attrs.BEHAVIOUR_KEY] == "payment-collect"
    assert span.attributes[attrs.ANCHOR_NAME] == "collect_run"


def test_task_context_manager_matches_decorator_identity_options(inmem):
    with task("payment-collect", agent_id="agent-1", project_id="project-1"):
        pass

    span = _only_span(inmem)
    assert span.attributes[attrs.AGENT_ID] == "agent-1"
    assert span.attributes[attrs.PROJECT_ID] == "project-1"


# ---------------------------------------------------------------------------
# name= anchor identity on decorators
# ---------------------------------------------------------------------------


def test_function_name_stamps_anchor_and_renames_span(inmem):
    @function(name="fetch_invoices")
    def load():
        return 1

    load()
    span = _only_span(inmem)
    assert span.attributes[attrs.ANCHOR_NAME] == "fetch_invoices"
    assert span.name == "fetch_invoices"


def test_tool_name_stamps_anchor(inmem):
    @tool(name="lookup_user")
    def lookup(user_id):
        return user_id

    lookup(1)
    span = _only_span(inmem)
    assert span.attributes[attrs.ANCHOR_NAME] == "lookup_user"
    assert span.attributes[attrs.TOOL_NAME] == "lookup_user"


def test_unnamed_decorator_keeps_qualname_default_without_anchor(inmem):
    @function()
    def plain():
        return 1

    plain()
    span = _only_span(inmem)
    assert attrs.ANCHOR_NAME not in span.attributes
    assert span.attributes[attrs.CODE_FUNCTION_NAME].endswith("plain")
    assert span.name == "plain"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_task_rejects_empty_or_non_string_key():
    with pytest.raises(ValueError, match="non-empty string key"):
        task("")
    with pytest.raises(ValueError, match="non-empty string key"):
        task("   ")
    with pytest.raises(ValueError, match="non-empty string key"):
        task(None)


def test_task_requires_exactly_one_static_key_or_selector():
    with pytest.raises(ValueError, match="exactly one"):
        task()
    with pytest.raises(ValueError, match="exactly one"):
        task("invoice-triage", key_from=lambda: "other")
    with pytest.raises(ValueError, match="callable"):
        task(key_from="invoice-triage")


def test_task_rejects_invalid_dynamic_key_before_exporting_a_span(inmem):
    @task(key_from=lambda _query: " ")
    def run_agent(_query):
        return "unreachable"

    with pytest.raises(ValueError, match="key_from must return a non-empty string"):
        run_agent("q")
    assert not inmem[1].get_finished_spans()


def test_task_selector_failure_is_clear_and_before_span_creation(inmem):
    def select_key(_query):
        raise LookupError("missing task")

    @task(key_from=select_key)
    def run_agent(_query):
        return "unreachable"

    with pytest.raises(RuntimeError, match=r"task\(\) key_from failed: missing task"):
        run_agent("q")
    assert not inmem[1].get_finished_spans()


def test_task_rejects_nested_dynamic_task(inmem):
    @task("outer")
    def run_outer():
        @task(key_from=lambda value: value)
        def run_inner(value):
            return value

        with pytest.raises(RuntimeError, match="cannot open task 'inner' inside task 'outer'"):
            run_inner("inner")

    run_outer()
    spans = inmem[1].get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes[attrs.BEHAVIOUR_KEY] == "outer"
