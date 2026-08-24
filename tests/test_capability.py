"""Capability scoping — identity stamping, exit restore, and handoff turns.

``overmind.capability`` attaches identity to the OTel context; the on-start
processor (wired by ``init``) stamps it on every span created inside, and
marks the first span of a mid-trace handoff ``overmind.unit_kind = "turn"``.
The wire keys are pinned in tests/test_evidence_contract.py.
"""

from __future__ import annotations

import asyncio
import contextvars

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import overmind.tracing as tracing
from overmind import attrs
from overmind.tracing import (
    _span_processor_on_start,
    capability,
    entry_point,
    function,
    set_agent_id,
    set_agent_name,
    start_span,
    tool,
)


@pytest.fixture
def exporter(monkeypatch):
    """In-memory provider wired like ``init``: identity on-start hook installed."""
    provider = TracerProvider()
    inmem = InMemorySpanExporter()
    proc = SimpleSpanProcessor(inmem)
    provider.add_span_processor(proc)
    proc.on_start = _span_processor_on_start
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer("overmind", "test"))
    monkeypatch.setattr(tracing, "_initialized", True)
    return inmem


def _in_fresh_context(fn):
    """Run *fn* in a copied context so ambient attach() calls don't leak."""
    return contextvars.copy_context().run(fn)


def _by_name(exporter, name):
    (span,) = [s for s in exporter.get_finished_spans() if s.name == name]
    return span


def test_capability_requires_identity():
    with pytest.raises(ValueError, match="name and/or id"):
        capability()


def test_spans_inside_scope_carry_identity(exporter):
    def _run():
        with capability("DOM Element Locator", id="cap-1"):
            with start_span("inside"):
                pass
        with start_span("outside"):
            pass

    _in_fresh_context(_run)
    inside = _by_name(exporter, "inside")
    assert inside.attributes[attrs.AGENT_NAME] == "DOM Element Locator"
    assert inside.attributes[attrs.AGENT_ID] == "cap-1"
    outside = _by_name(exporter, "outside")
    assert attrs.AGENT_NAME not in outside.attributes
    assert attrs.AGENT_ID not in outside.attributes


def test_nested_scope_restores_outer_identity(exporter):
    def _run():
        with capability("Outer", id="cap-out"):
            with capability("Inner", id="cap-in"), start_span("inner"):
                pass
            with start_span("after-inner"):
                pass

    _in_fresh_context(_run)
    assert _by_name(exporter, "inner").attributes[attrs.AGENT_ID] == "cap-in"
    after = _by_name(exporter, "after-inner")
    assert after.attributes[attrs.AGENT_ID] == "cap-out"
    assert after.attributes[attrs.AGENT_NAME] == "Outer"


def test_name_only_scope_clears_outer_id(exporter):
    def _run():
        set_agent_id("cap-out")
        set_agent_name("Outer")
        with capability("Inner"), start_span("inner"):
            pass

    _in_fresh_context(_run)
    inner = _by_name(exporter, "inner")
    assert inner.attributes[attrs.AGENT_NAME] == "Inner"
    assert attrs.AGENT_ID not in inner.attributes


def test_handoff_mid_trace_stamps_turn_on_first_span_only(exporter):
    def _run():
        set_agent_name("Browser Automation Agent")
        with start_span("run-root"), capability("DOM Element Locator"):
            with start_span("boundary"):
                pass
            with start_span("second"):
                pass

    _in_fresh_context(_run)
    assert _by_name(exporter, "boundary").attributes[attrs.UNIT_KIND] == "turn"
    assert attrs.UNIT_KIND not in _by_name(exporter, "second").attributes


def test_same_capability_is_not_a_handoff(exporter):
    def _run():
        set_agent_name("Browser Automation Agent")
        with start_span("run-root"), capability("Browser Automation Agent"):
            with start_span("step"):
                pass

    _in_fresh_context(_run)
    assert attrs.UNIT_KIND not in _by_name(exporter, "step").attributes


def test_scope_without_active_trace_stamps_no_turn(exporter):
    def _run():
        set_agent_name("Outer")
        with capability("Inner"), start_span("root"):
            pass

    _in_fresh_context(_run)
    assert attrs.UNIT_KIND not in _by_name(exporter, "root").attributes


def test_mixed_identity_grains_are_not_a_handoff(exporter):
    def _run():
        set_agent_name("Outer")
        with start_span("run-root"), capability(id="cap-9"):
            with start_span("inside"):
                pass

    _in_fresh_context(_run)
    inside = _by_name(exporter, "inside")
    assert attrs.UNIT_KIND not in inside.attributes
    assert inside.attributes[attrs.AGENT_ID] == "cap-9"
    assert attrs.AGENT_NAME not in inside.attributes


def test_entry_point_handoff_boundary_keeps_turn(exporter):
    @entry_point("sub-run")
    def sub_run() -> int:
        return 1

    def _run():
        set_agent_name("Outer")
        with start_span("run-root"), capability("Inner"):
            sub_run()

    _in_fresh_context(_run)
    assert _by_name(exporter, "sub-run").attributes[attrs.UNIT_KIND] == "turn"


def test_entry_point_outside_handoff_still_marks_run(exporter):
    @entry_point("run")
    def run() -> int:
        return 1

    _in_fresh_context(run)
    assert _by_name(exporter, "run").attributes[attrs.UNIT_KIND] == "run"


def test_capability_as_decorator_composes_with_tool(exporter):
    @capability("Page Markdown Extractor")
    @tool()
    def extract(url: str) -> str:
        return url

    _in_fresh_context(lambda: extract("https://example.com"))
    span = _by_name(exporter, "extract")
    assert span.attributes[attrs.AGENT_NAME] == "Page Markdown Extractor"
    assert span.attributes[attrs.PROVENANCE] == "environment"


def test_async_decorator_and_context_manager(exporter):
    @capability("Async Cap", id="cap-async")
    async def work() -> None:
        with start_span("decorated"):
            pass

    async def _main():
        await work()
        async with capability("Async CM"):
            with start_span("managed"):
                pass

    _in_fresh_context(lambda: asyncio.run(_main()))
    assert _by_name(exporter, "decorated").attributes[attrs.AGENT_ID] == "cap-async"
    assert _by_name(exporter, "managed").attributes[attrs.AGENT_NAME] == "Async CM"


def test_observe_agent_id_routes_through_capability_scope(exporter):
    @function("delegate", agent_id="cap-2")
    def delegate() -> None:
        with start_span("child"):
            pass

    def _run():
        set_agent_id("cap-1")
        with start_span("run-root"):
            delegate()

    _in_fresh_context(_run)
    span = _by_name(exporter, "delegate")
    assert span.attributes[attrs.AGENT_ID] == "cap-2"
    assert span.attributes[attrs.UNIT_KIND] == "turn"
    assert _by_name(exporter, "child").attributes[attrs.AGENT_ID] == "cap-2"


def test_observe_same_agent_id_is_not_a_handoff(exporter):
    @function("delegate", agent_id="cap-1")
    def delegate() -> None:
        pass

    def _run():
        set_agent_id("cap-1")
        with start_span("run-root"):
            delegate()

    _in_fresh_context(_run)
    assert attrs.UNIT_KIND not in _by_name(exporter, "delegate").attributes
