"""Declared task mapping — ``overmind.task`` stamps ``overmind.behaviour.key``."""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import overmind.tracing as tracing
from overmind import attrs
from overmind.tracing import _span_processor_on_start, start_span, task


@pytest.fixture
def exporter(monkeypatch):
    provider = TracerProvider()
    inmem = InMemorySpanExporter()
    provider.add_span_processor(tracing._TurnLifecycleSpanProcessor())
    proc = SimpleSpanProcessor(inmem)
    provider.add_span_processor(proc)
    proc.on_start = _span_processor_on_start
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer("overmind", "test"))
    monkeypatch.setattr(tracing, "_initialized", True)
    monkeypatch.setattr(tracing, "_turn_registry", tracing._TurnRegistry())
    return inmem


def _in_fresh_context(fn):
    return contextvars.copy_context().run(fn)


def _by_name(exporter, name):
    (span,) = (s for s in exporter.get_finished_spans() if s.name == name)
    return span


def test_task_requires_key():
    with pytest.raises(ValueError, match="requires a key"):
        task("")


def test_spans_inside_scope_carry_key(exporter):
    def _run():
        with task("lead-tool-loop"), start_span("inside"):
            pass
        with start_span("outside"):
            pass

    _in_fresh_context(_run)
    assert _by_name(exporter, "inside").attributes[attrs.BEHAVIOUR_KEY] == "lead-tool-loop"
    assert attrs.BEHAVIOUR_KEY not in _by_name(exporter, "outside").attributes


def test_nested_scope_restores_outer_key(exporter):
    def _run():
        with task("outer"):
            with task("inner"), start_span("inner"):
                pass
            with start_span("after-inner"):
                pass

    _in_fresh_context(_run)
    assert _by_name(exporter, "inner").attributes[attrs.BEHAVIOUR_KEY] == "inner"
    assert _by_name(exporter, "after-inner").attributes[attrs.BEHAVIOUR_KEY] == "outer"


def test_stamps_current_span_when_entered_inside(exporter):
    def _run():
        with start_span("unit"), task("run-startup"):
            pass

    _in_fresh_context(_run)
    assert _by_name(exporter, "unit").attributes[attrs.BEHAVIOUR_KEY] == "run-startup"


def test_never_stamps_run_boundary_span(exporter):
    def _run():
        with start_span("run-root", unit="run"):
            with task("step-a"), start_span("inside-a"):
                pass
            with task("step-b"), start_span("inside-b"):
                pass

    _in_fresh_context(_run)
    assert attrs.BEHAVIOUR_KEY not in _by_name(exporter, "run-root").attributes
    assert _by_name(exporter, "inside-a").attributes[attrs.BEHAVIOUR_KEY] == "step-a"
    assert _by_name(exporter, "inside-b").attributes[attrs.BEHAVIOUR_KEY] == "step-b"


def test_never_stamps_entry_point_span(exporter):
    def _run():
        with start_span("entry", span_type="entry_point"), task("last-phase"):
            pass

    _in_fresh_context(_run)
    assert attrs.BEHAVIOUR_KEY not in _by_name(exporter, "entry").attributes


def test_decorator_and_async(exporter):
    @task("generate-suggestions")
    def work() -> None:
        with start_span("decorated"):
            pass

    @task("async-task")
    async def async_work() -> None:
        with start_span("async-decorated"):
            pass

    async def _main():
        await async_work()
        async with task("async-cm"):
            with start_span("managed"):
                pass

    _in_fresh_context(work)
    _in_fresh_context(lambda: asyncio.run(_main()))
    assert _by_name(exporter, "decorated").attributes[attrs.BEHAVIOUR_KEY] == "generate-suggestions"
    assert _by_name(exporter, "async-decorated").attributes[attrs.BEHAVIOUR_KEY] == "async-task"
    assert _by_name(exporter, "managed").attributes[attrs.BEHAVIOUR_KEY] == "async-cm"


def test_noop_when_not_initialized():
    def _run():
        with task("lead-tool-loop"):
            pass

    _in_fresh_context(_run)


def test_turn_unit_rejects_other_units():
    with pytest.raises(ValueError, match='unit must be "turn"'):
        task("investment-debate", unit="run")


def test_turn_scope_opens_turn_span(exporter):
    def _run():
        with start_span("run-root", span_type="entry_point"):
            with task("investment-debate", unit="turn"), start_span("child"):
                pass

    _in_fresh_context(_run)
    turn = _by_name(exporter, "investment-debate")
    assert turn.attributes[attrs.UNIT_KIND] == "turn"
    assert turn.attributes[attrs.BEHAVIOUR_KEY] == "investment-debate"
    assert turn.attributes[attrs.SPAN_TYPE] == "function"
    root = _by_name(exporter, "run-root")
    assert turn.parent.span_id == root.get_span_context().span_id
    child = _by_name(exporter, "child")
    assert child.parent.span_id == turn.get_span_context().span_id
    assert child.attributes[attrs.BEHAVIOUR_KEY] == "investment-debate"


def test_turn_reentry_reuses_open_span(exporter):
    def _run():
        with start_span("run-root", span_type="entry_point"):
            with task("investment-debate", unit="turn"), start_span("round-1"):
                pass
            with task("research-manager", unit="turn"), start_span("verdict"):
                pass
            with task("investment-debate", unit="turn"), start_span("round-2"):
                pass

    _in_fresh_context(_run)
    debate = _by_name(exporter, "investment-debate")
    manager = _by_name(exporter, "research-manager")
    assert debate.get_span_context().span_id != manager.get_span_context().span_id
    for child in ("round-1", "round-2"):
        assert _by_name(exporter, child).parent.span_id == debate.get_span_context().span_id
    assert _by_name(exporter, "verdict").parent.span_id == manager.get_span_context().span_id


def test_run_boundary_end_closes_turn_spans_at_last_activity(exporter):
    def _run():
        with start_span("run-root", span_type="entry_point"):
            with task("investment-debate", unit="turn"):
                pass
            t0 = time.time_ns()
            with task("investment-debate", unit="turn"):
                pass
            t1 = time.time_ns()
            assert not exporter.get_finished_spans()  # turn still open mid-run
        return t0, t1

    t0, t1 = _in_fresh_context(_run)
    turn = _by_name(exporter, "investment-debate")
    assert t0 <= turn.end_time <= t1
    assert turn.end_time < _by_name(exporter, "run-root").end_time


def test_force_flush_ends_orphan_turn_spans(exporter):
    def _run():
        with task("orphan", unit="turn"):
            pass

    _in_fresh_context(_run)
    assert not exporter.get_finished_spans()
    tracing.force_flush_traces()
    assert _by_name(exporter, "orphan").attributes[attrs.UNIT_KIND] == "turn"


def test_turn_unit_noop_when_not_initialized():
    def _run():
        with task("investment-debate", unit="turn"):
            pass

    _in_fresh_context(_run)


def test_concurrent_entry_shares_one_turn_span(exporter):
    def _run():
        with start_span("run-root", span_type="entry_point"):

            def enter():
                with task("shared", unit="turn"):
                    time.sleep(0.005)

            threads = [threading.Thread(target=contextvars.copy_context().run, args=(enter,)) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

    _in_fresh_context(_run)
    assert len([s for s in exporter.get_finished_spans() if s.name == "shared"]) == 1
