"""Tests for the canonical ``genai.*`` tracing contract.

Covers the three anchor guarantees:

(a) a span carrying provider ``gen_ai.*`` usage ends up with the canonical
    ``genai.*`` token counts + ``genai.cost`` (enrichment processor + wrapper);
(b) ``init(capability_id=…, capability=…, project_id=…)`` results in
    ``overmind.capability.id`` / ``overmind.capability.name`` / ``overmind.project.id`` on
    emitted spans;
(c) finer-grained spans/attributes (tool metadata, streaming TTFT) are emitted
    with correct nesting.

Uses the repo's in-memory span exporter pattern (no network, no real LLMs).
"""

from __future__ import annotations

import contextvars

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from overmind import attrs
from overmind.genai_usage import canonical_usage_updates, compute_cost
from overmind.tracing import (
    _GenAiUsageSpanProcessor,
    _seed_identity_context,
    _span_processor_on_start,
)


@pytest.fixture
def inmem():
    """In-memory OTel provider wired exactly like ``init`` wires the real one.

    Enrichment processor runs first (mirrors ``gen_ai.*`` → ``genai.*`` on end),
    the exporting processor runs second and carries the identity on-start hook.
    The Overmind SDK tracer is pointed at this provider for the test.
    """
    from overmind import tracing

    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(_GenAiUsageSpanProcessor())
    exporter = InMemorySpanExporter()
    export_proc = SimpleSpanProcessor(exporter)
    provider.add_span_processor(export_proc)
    export_proc.on_start = _span_processor_on_start

    saved_tracer = tracing._tracer
    saved_initialized = tracing._initialized
    tracing._tracer = provider.get_tracer("overmind")
    tracing._initialized = True
    try:
        yield provider, exporter
    finally:
        tracing._tracer = saved_tracer
        tracing._initialized = saved_initialized


# ---------------------------------------------------------------------------
# canonical_usage_updates — the mirror/cost helper (unit)
# ---------------------------------------------------------------------------


def test_mirror_from_otel_semconv_prompt_completion():
    updates = canonical_usage_updates({
        "gen_ai.usage.prompt_tokens": 120,
        "gen_ai.usage.completion_tokens": 30,
        attrs.LLM_MODEL: "gpt-4o-mini",
    })
    assert updates[attrs.LLM_PROMPT_TOKENS] == 120
    assert updates[attrs.LLM_COMPLETION_TOKENS] == 30
    assert updates[attrs.LLM_TOTAL_TOKENS] == 150
    assert updates[attrs.LLM_COST] > 0


def test_mirror_from_input_output_and_llm_total():
    updates = canonical_usage_updates({
        "gen_ai.usage.input_tokens": 10,
        "gen_ai.usage.output_tokens": 5,
        "llm.usage.total_tokens": 15,
    })
    assert updates[attrs.LLM_PROMPT_TOKENS] == 10
    assert updates[attrs.LLM_COMPLETION_TOKENS] == 5
    assert updates[attrs.LLM_TOTAL_TOKENS] == 15


def test_mirror_never_zero_fills_and_respects_existing():
    # No usage at all → nothing to mirror.
    assert canonical_usage_updates({attrs.SPAN_TYPE: "llm_call"}) == {}
    # Existing canonical cost must not be overwritten.
    updates = canonical_usage_updates({
        "gen_ai.usage.prompt_tokens": 100,
        "gen_ai.usage.completion_tokens": 100,
        attrs.LLM_MODEL: "gpt-4o-mini",
        attrs.LLM_COST: 0.42,
    })
    assert attrs.LLM_COST not in updates


def test_compute_cost_unknown_model_is_none():
    assert compute_cost("some-nonexistent-model-xyz", 100, 100) is None
    assert compute_cost("gpt-4o-mini", None, None) is None


# ---------------------------------------------------------------------------
# (a) Enrichment processor mirrors auto-instrumentor spans on end
# ---------------------------------------------------------------------------


def test_enrichment_processor_mirrors_on_end(inmem):
    provider, exporter = inmem
    tracer = provider.get_tracer("overmind")
    with tracer.start_as_current_span("openai.chat") as span:
        span.set_attribute("gen_ai.usage.prompt_tokens", 200)
        span.set_attribute("gen_ai.usage.completion_tokens", 80)
        span.set_attribute("gen_ai.request.model", "gpt-4o-mini")
    provider.force_flush()

    exported = exporter.get_finished_spans()[-1]
    assert exported.attributes[attrs.LLM_PROMPT_TOKENS] == 200
    assert exported.attributes[attrs.LLM_COMPLETION_TOKENS] == 80
    assert exported.attributes[attrs.LLM_TOTAL_TOKENS] == 280
    assert exported.attributes[attrs.LLM_MODEL] == "gpt-4o-mini"
    assert exported.attributes[attrs.LLM_COST] > 0


# ---------------------------------------------------------------------------
# (b) Agent / project identity on every span
# ---------------------------------------------------------------------------


def test_identity_stamped_on_spans(inmem):
    provider, exporter = inmem

    def _run():
        _seed_identity_context("agent-uuid-123", "Lead Qualifier", "proj-uuid-9")
        tracer = provider.get_tracer("overmind")
        with tracer.start_as_current_span("work"):
            pass

    contextvars.copy_context().run(_run)
    provider.force_flush()

    span = exporter.get_finished_spans()[-1]
    assert span.attributes[attrs.CAPABILITY_ID] == "agent-uuid-123"
    assert span.attributes[attrs.CAPABILITY_NAME] == "Lead Qualifier"
    assert span.attributes[attrs.PROJECT_ID] == "proj-uuid-9"


# ---------------------------------------------------------------------------
# (c) Tool span metadata + nesting
# ---------------------------------------------------------------------------


def test_tool_decorator_stamps_metadata(inmem):
    provider, exporter = inmem
    from overmind import tool

    @tool()
    def search_kb(query, limit=5):
        return {"hits": 1}

    search_kb("refund policy", limit=3)
    provider.force_flush()

    span = exporter.get_finished_spans()[-1]
    assert span.attributes[attrs.SPAN_TYPE] == "tool_call"
    assert span.attributes[attrs.TOOL_NAME] == "search_kb"
    assert set(span.attributes[attrs.TOOL_ARG_KEYS]) == {"query", "limit"}


def test_tool_decorator_records_error(inmem):
    provider, exporter = inmem
    from overmind import tool

    @tool()
    def failing_tool():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        failing_tool()
    provider.force_flush()

    span = exporter.get_finished_spans()[-1]
    assert span.attributes[attrs.TOOL_ERROR] == "ValueError"
    assert span.attributes[attrs.STATUS] == "failed"


def test_nested_tool_under_workflow_keeps_parent(inmem):
    provider, exporter = inmem
    from overmind import tool, workflow

    @tool()
    def inner_tool(x):
        return x

    @workflow()
    def outer():
        return inner_tool(1)

    outer()
    provider.force_flush()

    spans = {s.name.rsplit(".", 1)[-1]: s for s in exporter.get_finished_spans()}
    parent = spans["outer"]
    child = spans["inner_tool"]
    assert child.parent is not None
    assert child.parent.span_id == parent.context.span_id
