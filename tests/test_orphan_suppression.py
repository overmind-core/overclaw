"""Orphan-fragment suppression (`_OrphanSpanSampler`).

A declared ``function`` span that starts a new local trace — no parent, no
boundary declaration — is sampled out (with its children), unless
``init(export_orphan_spans=True)``. Deliberate roots always export: boundary
declarations (entry point / ``unit=``), other declared span types, foreign
spans with no declaration, and remote-parent (TRACEPARENT) continuations.
"""

from __future__ import annotations

import contextvars

import pytest
from opentelemetry import trace
from opentelemetry.context import attach, detach
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

import overmind.tracing as tracing
from overmind.tracing import _span_processor_on_start, observe, start_span, task


@pytest.fixture
def exporter(monkeypatch):
    provider = TracerProvider(sampler=tracing._OrphanSpanSampler())
    inmem = InMemorySpanExporter()
    provider.add_span_processor(tracing._TurnLifecycleSpanProcessor())
    proc = SimpleSpanProcessor(inmem)
    provider.add_span_processor(proc)
    proc.on_start = _span_processor_on_start
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer("overmind", "test"))
    monkeypatch.setattr(tracing, "_initialized", True)
    monkeypatch.setattr(tracing, "_turn_registry", tracing._TurnRegistry())
    monkeypatch.setattr(tracing, "_export_orphan_spans", False)
    monkeypatch.setattr(tracing, "_orphan_suppressed_logged", False)
    return inmem


def _in_fresh_context(fn):
    return contextvars.copy_context().run(fn)


def _names(exporter):
    return sorted(s.name for s in exporter.get_finished_spans())


def test_observed_function_outside_any_boundary_is_not_exported(exporter, caplog):
    @observe()
    def reconstruct_agent_state():
        return "state"

    with caplog.at_level("WARNING"):
        assert _in_fresh_context(reconstruct_agent_state) == "state"

    assert _names(exporter) == []
    (record,) = (r for r in caplog.records if "not exported" in r.message)
    assert "overmind.run" in record.message
    assert "export_orphan_spans=True" in record.message


def test_suppression_warns_once_per_process(exporter, caplog):
    @observe("orphan")
    def orphan():
        pass

    with caplog.at_level("WARNING"):
        _in_fresh_context(orphan)
        _in_fresh_context(orphan)

    assert len([r for r in caplog.records if "not exported" in r.message]) == 1


def test_children_of_a_suppressed_orphan_fall_with_it(exporter):
    def _main():
        with start_span("orphan-root"), start_span("child", span_type="tool"):
            pass

    _in_fresh_context(_main)
    assert _names(exporter) == []


def test_entry_point_root_and_its_interior_spans_export(exporter):
    @observe("inner")
    def inner():
        pass

    def _main():
        with start_span("root", span_type="entry_point"):
            inner()

    _in_fresh_context(_main)
    assert _names(exporter) == ["inner", "root"]


def test_manual_unit_declarations_always_export(exporter):
    def _main():
        with start_span("one-shot", unit="turn"):
            pass

    _in_fresh_context(_main)
    assert _names(exporter) == ["one-shot"]


def test_turn_registry_span_at_root_exports(exporter):
    def _main():
        with task("phase", unit="turn"), start_span("step"):
            pass

    _in_fresh_context(_main)
    tracing.force_flush_traces()
    assert _names(exporter) == ["phase", "step"]


@pytest.mark.parametrize("span_type", ["tool", "workflow", "llm", "retrieval"])
def test_non_function_declared_roots_export(exporter, span_type):
    def _main():
        with start_span("deliberate", span_type=span_type):
            pass

    _in_fresh_context(_main)
    assert _names(exporter) == ["deliberate"]


def test_foreign_root_without_declaration_exports(exporter):
    """Auto-instrumented spans carry no overmind declaration — a bare provider
    call at the root stays visible."""

    def _main():
        with tracing.get_tracer().start_as_current_span("openai.chat"):
            pass

    _in_fresh_context(_main)
    assert _names(exporter) == ["openai.chat"]


@pytest.mark.parametrize("sampled", [True, False])
def test_remote_parent_continuations_always_export(exporter, sampled):
    remote = SpanContext(
        trace_id=0x1CE1CE1CE1CE1CE1CE1CE1CE1CE1CE1C,
        span_id=0x51DE51DE51DE51DE,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED if sampled else TraceFlags.DEFAULT),
    )

    @observe("subprocess_work")
    def subprocess_work():
        pass

    def _main():
        token = attach(trace.set_span_in_context(NonRecordingSpan(remote)))
        try:
            subprocess_work()
        finally:
            detach(token)

    _in_fresh_context(_main)
    (span,) = exporter.get_finished_spans()
    assert span.name == "subprocess_work"
    assert span.parent.span_id == remote.span_id


def test_export_orphan_spans_opt_out_exports_everything(exporter, monkeypatch):
    monkeypatch.setattr(tracing, "_export_orphan_spans", True)

    @observe("orphan")
    def orphan():
        pass

    _in_fresh_context(orphan)
    assert _names(exporter) == ["orphan"]
