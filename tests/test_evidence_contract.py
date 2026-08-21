"""Wire-contract tests for the evaluation evidence attributes.

The platform's evaluation framework reads ``overmind.provenance`` /
``overmind.unit_kind`` / ``overmind.delivery`` / ``overmind.grounded_by``
against exactly these keys and values — pinned, never rename.  Emitters live
in ``overmind/tracing.py`` (see docs/tracing-attributes.md §7).
"""

from __future__ import annotations

import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import overmind.tracing as tracing
from overmind import attrs
from overmind.tracing import SpanType, deliver, entry_point, function, mark_unit, observe, retrieval, start_span, tool


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


def test_attribute_keys_are_pinned():
    assert attrs.PROVENANCE == "overmind.provenance"
    assert attrs.UNIT_KIND == "overmind.unit_kind"
    assert attrs.DELIVERY == "overmind.delivery"
    assert attrs.GROUNDED_BY == "overmind.grounded_by"


# ---------------------------------------------------------------------------
# Capture-time provenance auto-tagging
# ---------------------------------------------------------------------------


def test_tool_span_auto_tags_environment(exporter):
    @tool()
    def search(query: str) -> list[str]:
        return [query]

    search("q")
    assert _only_span(exporter).attributes[attrs.PROVENANCE] == "environment"


def test_retrieval_span_auto_tags_environment(exporter):
    @retrieval()
    def fetch_docs(query: str) -> list[str]:
        return [query]

    fetch_docs("q")
    assert _only_span(exporter).attributes[attrs.PROVENANCE] == "environment"


def test_llm_span_auto_tags_agent(exporter):
    with start_span("completion", span_type=SpanType.LLM):
        pass
    assert _only_span(exporter).attributes[attrs.PROVENANCE] == "agent"


def test_function_span_has_no_provenance(exporter):
    @function()
    def compute() -> int:
        return 1

    compute()
    assert attrs.PROVENANCE not in _only_span(exporter).attributes


def test_explicit_provenance_overrides_auto_tag(exporter):
    @observe(provenance="user")
    def capture_message(text: str) -> str:
        return text

    capture_message("hi")
    assert _only_span(exporter).attributes[attrs.PROVENANCE] == "user"


def test_invalid_provenance_raises():
    with pytest.raises(ValueError, match="provenance"):
        observe(provenance="alien")


# ---------------------------------------------------------------------------
# Unit markers
# ---------------------------------------------------------------------------


def test_entry_point_marks_run(exporter):
    @entry_point()
    def run() -> int:
        return 1

    run()
    assert _only_span(exporter).attributes[attrs.UNIT_KIND] == "run"


def test_mark_unit_stamps_current_span(exporter):
    with start_span("step"):
        mark_unit("turn")
    assert _only_span(exporter).attributes[attrs.UNIT_KIND] == "turn"


def test_mark_unit_validates_kind():
    with pytest.raises(ValueError, match="kind"):
        mark_unit("phase")


# ---------------------------------------------------------------------------
# deliver()
# ---------------------------------------------------------------------------


def test_deliver_marks_delivery_and_grounding(exporter):
    with start_span("evidence") as evidence:
        pass
    evidence_id = format(evidence.get_span_context().span_id, "016x")

    deliver({"answer": 42}, grounded_by=[evidence, "deadbeefdeadbeef"])

    span = exporter.get_finished_spans()[-1]
    assert span.attributes[attrs.DELIVERY] is True
    assert span.attributes[attrs.PROVENANCE] == "agent"
    assert json.loads(span.attributes[attrs.GROUNDED_BY]) == [evidence_id, "deadbeefdeadbeef"]
    assert json.loads(span.attributes["outputs"]) == {"answer": 42}


def test_deliver_without_grounding_omits_grounded_by(exporter):
    deliver("done")
    span = _only_span(exporter)
    assert span.attributes[attrs.DELIVERY] is True
    assert attrs.GROUNDED_BY not in span.attributes
