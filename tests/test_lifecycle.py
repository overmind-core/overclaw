"""``overmind.run(...)`` lifecycle scope: boundary span, capability identity,
intent/conversation/tags, delivery via the handle, error status, and
turn-span closure on exit. The decorator form additionally resolves callable
parameters from the wrapped call's arguments and stamps the function's code
identity on the boundary span."""

from __future__ import annotations

import asyncio
import contextvars

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import overmind.tracing as tracing
from overmind import attrs
from overmind.lifecycle import RunHandle, run
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
    (span,) = [s for s in exporter.get_finished_spans() if s.name == name]
    return span


def test_run_opens_an_entry_point_run_boundary(exporter):
    def _main():
        with run("my-run") as handle:
            assert isinstance(handle, RunHandle)
            assert handle.span.is_recording()

    _in_fresh_context(_main)
    root = _by_name(exporter, "my-run")
    assert root.attributes[attrs.SPAN_TYPE] == "entry_point"
    assert root.attributes[attrs.UNIT_KIND] == "run"
    assert attrs.BEHAVIOUR_KEY not in root.attributes


def test_run_stamps_capability_intent_conversation_and_tags(exporter):
    def _main():
        with run(
            "my-run",
            capability="Research Agent",
            intent="Answer the question",
            conversation_id="conv-1",
            tags={"ticker": "AAPL"},
        ):
            with start_span("child"):
                pass

    _in_fresh_context(_main)
    root = _by_name(exporter, "my-run")
    child = _by_name(exporter, "child")
    for span in (root, child):
        assert span.attributes[attrs.AGENT_NAME] == "Research Agent"
        assert span.attributes["conversation.id"] == "conv-1"
    assert root.attributes["ticker"] == "AAPL"
    (event,) = [e for e in root.events if e.name == attrs.EVAL_INTENT_EVENT]
    assert "Answer the question" in event.attributes[attrs.EVAL_PAYLOAD]


def test_run_capability_falls_back_to_env(exporter, monkeypatch):
    monkeypatch.setenv("OVERMIND_AGENT_NAME", "Env Agent")
    monkeypatch.setenv("OVERMIND_AGENT_ID", "env-id")

    def _main():
        with run():
            pass

    _in_fresh_context(_main)
    root = _by_name(exporter, "run")
    assert root.attributes[attrs.AGENT_NAME] == "Env Agent"
    assert root.attributes[attrs.AGENT_ID] == "env-id"


def test_run_without_identity_needs_no_capability_scope(exporter, monkeypatch):
    monkeypatch.delenv("OVERMIND_AGENT_NAME", raising=False)
    monkeypatch.delenv("OVERMIND_AGENT_ID", raising=False)

    def _main():
        with run():
            pass

    _in_fresh_context(_main)
    root = _by_name(exporter, "run")
    assert attrs.AGENT_NAME not in root.attributes


def test_handle_delivers_inside_the_producing_unit(exporter):
    def _main():
        with run("my-run") as handle:
            with task("portfolio-manager", unit="turn"):
                handle.deliver({"decision": "BUY"})

    _in_fresh_context(_main)
    turn = _by_name(exporter, "portfolio-manager")
    delivery = _by_name(exporter, "deliver")
    assert turn.attributes[attrs.UNIT_KIND] == "turn"
    assert delivery.parent.span_id == turn.context.span_id
    assert delivery.attributes[attrs.DELIVERY] is True
    assert delivery.attributes[attrs.BEHAVIOUR_KEY] == "portfolio-manager"


def test_open_turn_spans_close_when_the_run_ends(exporter):
    def _main():
        with run("my-run"):
            with task("phase-a", unit="turn"):
                pass
            # phase-a's turn span stays open here (re-entrant registry)…

    _in_fresh_context(_main)
    # …and must be finished once the run boundary ended.
    turn = _by_name(exporter, "phase-a")
    root = _by_name(exporter, "my-run")
    assert turn.parent.span_id == root.context.span_id


def test_exception_marks_the_run_failed_and_reraises(exporter):
    def _main():
        with run("my-run"):
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        _in_fresh_context(_main)
    root = _by_name(exporter, "my-run")
    assert root.attributes[attrs.STATUS] == "failed"
    assert root.attributes[attrs.ERROR_TYPE] == "ValueError"


def test_run_is_a_noop_when_uninitialised(monkeypatch):
    monkeypatch.setattr(tracing, "_initialized", False)

    def _main():
        with run("my-run", intent="x") as handle:
            handle.deliver("payload")  # must not raise

    _in_fresh_context(_main)


# ---------------------------------------------------------------------------
# Decorator form — callable parameters + boundary code identity
# ---------------------------------------------------------------------------


def test_run_decorator_resolves_callable_params_from_bound_args(exporter):
    """The browser-use shape: a method entry point whose intent and
    conversation id are runtime values on self."""

    class Agent:
        def __init__(self, task: str, task_id: str) -> None:
            self.task = task
            self.task_id = task_id

        @run(
            "agent-run",
            intent=lambda self, *a, **k: self.task,
            conversation_id=lambda self, *a, **k: self.task_id,
        )
        def start(self) -> str:
            return "done"

    agent = Agent("Book the flight", "task-7")
    assert _in_fresh_context(agent.start) == "done"
    root = _by_name(exporter, "agent-run")
    assert root.attributes[attrs.SPAN_TYPE] == "entry_point"
    assert root.attributes[attrs.UNIT_KIND] == "run"
    assert root.attributes["conversation.id"] == "task-7"
    (event,) = [e for e in root.events if e.name == attrs.EVAL_INTENT_EVENT]
    assert "Book the flight" in event.attributes[attrs.EVAL_PAYLOAD]


def test_run_decorator_stamps_boundary_code_identity(exporter):
    """One decoration satisfies an entry-point scan-contract anchor: the run
    boundary itself carries the function's code identity."""

    class Agent:
        @run()
        def start(self) -> int:
            return 1

    assert _in_fresh_context(Agent().start) == 1
    (root,) = exporter.get_finished_spans()
    assert root.name.endswith("Agent.start")  # default name is the qualname
    assert root.attributes[attrs.CODE_NAMESPACE] == __name__
    assert root.attributes[attrs.CODE_FUNCTION_NAME].endswith("Agent.start")
    assert root.attributes[attrs.UNIT_KIND] == "run"


def test_run_decorator_supports_async(exporter):
    @run("async-run", intent=lambda prompt: prompt, tags=lambda prompt: {"chars": len(prompt)})
    async def main(prompt: str) -> str:
        return prompt.upper()

    assert _in_fresh_context(lambda: asyncio.run(main("hi"))) == "HI"
    root = _by_name(exporter, "async-run")
    assert root.attributes[attrs.UNIT_KIND] == "run"
    assert root.attributes["chars"] == 2
    (event,) = [e for e in root.events if e.name == attrs.EVAL_INTENT_EVENT]
    assert "hi" in event.attributes[attrs.EVAL_PAYLOAD]


def test_run_decorator_does_not_auto_deliver(exporter):
    @run("r")
    def main() -> dict:
        return {"answer": 42}

    assert _in_fresh_context(main) == {"answer": 42}
    assert [s.name for s in exporter.get_finished_spans()] == ["r"]
    assert attrs.DELIVERY not in _by_name(exporter, "r").attributes


def test_run_decorator_callable_failure_degrades_to_none(exporter):
    @run("r", conversation_id=lambda: {}["missing"])
    def main() -> str:
        return "ok"

    assert _in_fresh_context(main) == "ok"
    assert "conversation.id" not in _by_name(exporter, "r").attributes


def test_run_decorator_is_noop_when_uninitialised(monkeypatch):
    monkeypatch.setattr(tracing, "_initialized", False)

    @run("r", intent=lambda x: x)
    def main(x: str) -> str:
        return x

    assert _in_fresh_context(lambda: main("ok")) == "ok"


def test_bare_run_decoration_raises():
    with pytest.raises(TypeError, match="parentheses"):

        @run
        def main() -> None: ...
