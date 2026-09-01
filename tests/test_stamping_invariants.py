"""Stamping invariants over enumerated scope compositions.

Every nesting of capability / task / turn-task / entry-point / observe /
plain spans (all sequences up to depth 3, each closed with a leaf span) must
satisfy the system invariants on the exported spans:

I1  at most one ``unit_kind="run"`` span per trace (the root boundary);
I2  a run-boundary span never carries ``overmind.behaviour.key``;
I3  a turn span's behaviour key is never overwritten after creation;
I4  every span created inside a task scope carries the innermost scope's key,
    unless a capability scope entered after it cleared the ambient key (I6);
I5  a handoff stamps ``unit_kind="turn"`` on exactly the first span of the
    new capability scope;
I6  a span inside capability scope B never carries a behaviour key from a
    task scope opened under capability A — keys are capability-scoped, so
    entering a capability scope clears the ambient key and exiting restores it.

The interpreter mirrors the documented contract (not the implementation) to
compute each created span's expected unit kind and key, so shape enumeration
is deterministic — no randomness.
"""

from __future__ import annotations

import contextvars
import dataclasses
import itertools
import threading
from collections import Counter

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import overmind.tracing as tracing
from overmind import attrs
from overmind.tracing import (
    _seed_identity_context,
    _span_processor_on_start,
    capability,
    observe,
    start_span,
    task,
)


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
    (span,) = [s for s in exporter.get_finished_spans() if s.name == name]
    return span


# ---------------------------------------------------------------------------
# Shape enumeration + contract-mirroring interpreter
# ---------------------------------------------------------------------------

OPS = ("plain", "run", "task", "turn", "cap", "obs")


def _all_shapes(max_depth: int = 3) -> list[tuple[str, ...]]:
    return [shape for depth in range(1, max_depth + 1) for shape in itertools.product(OPS, repeat=depth)]


@dataclasses.dataclass
class _Spec:
    """Expected final stamps for one created span."""

    unit: str | None
    key: str | None


def _run_shape(shape: tuple[str, ...]) -> dict[str, _Spec]:
    """Execute *shape* outermost-first with a leaf span innermost, mirroring
    the wire contract to compute per-span expectations.

    Interpreter state per level: the innermost task key, the active
    capability identity, whether an in-process trace is open, and the
    handoff one-shot (shared by reference, like the real pending-turn cell).
    """
    expected: dict[str, _Spec] = {}

    def created(name: str, state: dict, own_unit: str | None = None, own_key: str | None = None) -> _Spec:
        unit = own_unit
        pending = state["pending"]
        if pending is not None and not pending["consumed"]:
            pending["consumed"] = True
            unit = "turn"  # I5: first span of a handoff scope
        elif unit == "run" and state["in_trace"]:
            unit = "turn"  # I1: a nested run declaration is a unit, not a second run
        key = own_key or (state["key"] if unit != "run" else None)  # I2
        spec = _Spec(unit=unit, key=key)
        expected[name] = spec
        return spec

    def interpret(index: int, state: dict, ambient: _Spec | None) -> None:
        if index == len(shape):
            created("leaf", state)
            with start_span("leaf"):
                pass
            return
        op = shape[index]
        name = f"{op}-{index}"
        inner = dict(state)
        if op == "plain":
            spec = created(name, state)
            inner["in_trace"] = True
            with start_span(name):
                interpret(index + 1, inner, spec)
        elif op == "run":
            spec = created(name, state, own_unit="run")
            inner["in_trace"] = True
            with start_span(name, span_type="entry_point"):
                interpret(index + 1, inner, spec)
        elif op == "obs":
            spec = created(name, state)
            inner["in_trace"] = True

            @observe(name)
            def observed() -> None:
                interpret(index + 1, inner, spec)

            observed()
        elif op == "task":
            key = f"key-{index}"
            # Allowlist labelling: only an unkeyed non-boundary ambient span.
            if ambient is not None and ambient.unit is None and ambient.key is None:
                ambient.key = key
            inner["key"] = key
            with task(key):
                interpret(index + 1, inner, ambient)
        elif op == "turn":
            key = f"turn-{index}"
            spec = created(key, state, own_unit="turn", own_key=key)
            inner["key"] = key
            inner["in_trace"] = True
            with task(key, unit="turn"):
                interpret(index + 1, inner, spec)
        elif op == "cap":
            cap_name = f"Cap-{index}"
            handoff = state["in_trace"] and state["identity"] != cap_name
            inner["identity"] = cap_name
            inner["pending"] = {"consumed": False} if handoff else None
            inner["key"] = None  # I6: entering a capability clears the ambient key
            with capability(cap_name):
                interpret(index + 1, inner, ambient)

    def _main() -> None:
        _seed_identity_context(None, "Outer", None)
        interpret(0, {"key": None, "identity": "Outer", "pending": None, "in_trace": False}, None)

    _in_fresh_context(_main)
    return expected


@pytest.mark.parametrize("shape", _all_shapes(), ids=lambda shape: "-".join(shape))
def test_composition_invariants(exporter, shape):
    expected = _run_shape(shape)
    tracing.force_flush_traces()  # close any still-open turn spans
    spans = exporter.get_finished_spans()

    assert sorted(s.name for s in spans) == sorted(expected)
    for span in spans:
        spec = expected[span.name]
        assert span.attributes.get(attrs.UNIT_KIND) == spec.unit, span.name
        assert span.attributes.get(attrs.BEHAVIOUR_KEY) == spec.key, span.name

    # Recheck I1 + I2 straight from the export, independent of the mirror.
    runs_per_trace = Counter(
        s.context.trace_id for s in spans if s.attributes.get(attrs.UNIT_KIND) == "run"
    )
    assert all(count == 1 for count in runs_per_trace.values())
    for span in spans:
        if span.attributes.get(attrs.UNIT_KIND) == "run":
            assert attrs.BEHAVIOUR_KEY not in span.attributes


# ---------------------------------------------------------------------------
# Targeted regressions and thread interleavings
# ---------------------------------------------------------------------------


def test_nested_task_inside_turn_task_does_not_rebind_turn(exporter):
    """A plain task("b") nested in task("a", unit="turn") must not overwrite
    the turn span's key — the same hijack class as the root-boundary bug."""

    def _run():
        with start_span("run-root", span_type="entry_point"):
            with task("a", unit="turn"):
                with task("b"), start_span("inside-b"):
                    pass

    _in_fresh_context(_run)
    turn = _by_name(exporter, "a")
    assert turn.attributes[attrs.UNIT_KIND] == "turn"
    assert turn.attributes[attrs.BEHAVIOUR_KEY] == "a"
    assert _by_name(exporter, "inside-b").attributes[attrs.BEHAVIOUR_KEY] == "b"


def test_handoff_scope_never_inherits_outer_task_key(exporter):
    """A handoff entered inside task("x") must not leak "x" onto the handoff
    turn or its children — the key is capability-scoped; after the inner
    scope exits, spans under the outer scope carry "x" again."""

    def _run():
        _seed_identity_context(None, "Outer", None)
        with start_span("run-root", span_type="entry_point"), task("x"):
            with capability("Inner"):
                with start_span("handoff-turn"), start_span("inner-child"):
                    pass
            with start_span("outer-again"):
                pass

    _in_fresh_context(_run)
    handoff = _by_name(exporter, "handoff-turn")
    assert handoff.attributes[attrs.UNIT_KIND] == "turn"
    assert attrs.BEHAVIOUR_KEY not in handoff.attributes
    assert attrs.BEHAVIOUR_KEY not in _by_name(exporter, "inner-child").attributes
    assert _by_name(exporter, "outer-again").attributes[attrs.BEHAVIOUR_KEY] == "x"


def test_task_never_stamps_turn_boundary_span(exporter):
    def _run():
        with start_span("unit", unit="turn"), task("late-key"), start_span("child"):
            pass

    _in_fresh_context(_run)
    assert attrs.BEHAVIOUR_KEY not in _by_name(exporter, "unit").attributes
    assert _by_name(exporter, "child").attributes[attrs.BEHAVIOUR_KEY] == "late-key"


def test_task_never_overwrites_existing_key(exporter):
    def _run():
        with start_span("unit"):
            with task("first"):
                pass
            with task("second"), start_span("child"):
                pass

    _in_fresh_context(_run)
    assert _by_name(exporter, "unit").attributes[attrs.BEHAVIOUR_KEY] == "first"
    assert _by_name(exporter, "child").attributes[attrs.BEHAVIOUR_KEY] == "second"


def test_run_declared_inside_task_scope_carries_no_key(exporter):
    def _run():
        with task("outer-key"), start_span("root", span_type="entry_point"):
            with start_span("inside"):
                pass

    _in_fresh_context(_run)
    root = _by_name(exporter, "root")
    assert root.attributes[attrs.UNIT_KIND] == "run"
    assert attrs.BEHAVIOUR_KEY not in root.attributes
    assert _by_name(exporter, "inside").attributes[attrs.BEHAVIOUR_KEY] == "outer-key"


def test_nested_entry_point_demotes_to_turn(exporter):
    def _run():
        with start_span("root", span_type="entry_point"):
            with start_span("sub-run", span_type="entry_point"):
                pass

    _in_fresh_context(_run)
    assert _by_name(exporter, "root").attributes[attrs.UNIT_KIND] == "run"
    assert _by_name(exporter, "sub-run").attributes[attrs.UNIT_KIND] == "turn"


def test_threaded_task_scopes_stamp_only_their_own_children(exporter):
    def _run():
        with start_span("run-root", span_type="entry_point"):

            def worker(i: int) -> None:
                with task(f"key-{i}"), start_span(f"child-{i}"):
                    pass

            threads = [
                threading.Thread(target=contextvars.copy_context().run, args=(worker, i)) for i in range(8)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

    _in_fresh_context(_run)
    assert attrs.BEHAVIOUR_KEY not in _by_name(exporter, "run-root").attributes
    for i in range(8):
        assert _by_name(exporter, f"child-{i}").attributes[attrs.BEHAVIOUR_KEY] == f"key-{i}"


def test_threaded_nested_tasks_never_rebind_shared_turn(exporter):
    def _run():
        with start_span("run-root", span_type="entry_point"):

            def worker(i: int) -> None:
                with task("shared", unit="turn"):
                    with task(f"inner-{i}"), start_span(f"step-{i}"):
                        pass

            threads = [
                threading.Thread(target=contextvars.copy_context().run, args=(worker, i)) for i in range(8)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

    _in_fresh_context(_run)
    turn = _by_name(exporter, "shared")
    assert turn.attributes[attrs.UNIT_KIND] == "turn"
    assert turn.attributes[attrs.BEHAVIOUR_KEY] == "shared"
    for i in range(8):
        assert _by_name(exporter, f"step-{i}").attributes[attrs.BEHAVIOUR_KEY] == f"inner-{i}"
