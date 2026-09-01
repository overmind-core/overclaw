"""``overmind.integrations.langgraph.bind``: slug convention, behaviour
overrides and opt-outs, turn-unit re-entry, code identity on function nodes,
declarative delivery, and structural validation — all against a minimal fake
of langgraph's StateGraph/StateNodeSpec/RunnableCallable shape, so the suite
stays hermetic."""

from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import overmind.tracing as tracing
from overmind import attrs
from overmind.integrations.langgraph import bind, slug
from overmind.tracing import _span_processor_on_start, start_span


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


# --- minimal langgraph shape -------------------------------------------------


class FakeRunnable:
    """RunnableCallable stand-in: ``func`` invoked sync, ``afunc`` awaited."""

    def __init__(self, func=None, afunc=None) -> None:
        self.func = func
        self.afunc = afunc

    def invoke(self, state, config=None):
        return self.func(state)

    async def ainvoke(self, state, config=None):
        return await self.afunc(state)


@dataclass
class FakeSpec:
    runnable: Any


class FakeGraph:
    def __init__(self, **nodes) -> None:
        self.nodes = {name: FakeSpec(FakeRunnable(func=fn)) for name, fn in nodes.items()}

    def invoke_node(self, name, state):
        return self.nodes[name].runnable.invoke(state)


def market_analyst(state):
    return {"report": "up"}


def portfolio_manager(state):
    return {"final_trade_decision": "BUY"}


# --- slug convention ----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Market Analyst", "market-analyst"),
        ("tools_market", "tools-market"),
        ("Msg Clear Fundamentals", "msg-clear-fundamentals"),
        ("  Portfolio  Manager!  ", "portfolio-manager"),
    ],
)
def test_slug_convention(name, expected):
    assert slug(name) == expected


# --- binding behaviour ---------------------------------------------------------


def test_nodes_run_inside_slugged_turn_units_with_code_identity(exporter):
    graph = bind(FakeGraph(**{"Market Analyst": market_analyst}))

    def _main():
        with start_span("root", span_type="entry_point"):
            assert graph.invoke_node("Market Analyst", {}) == {"report": "up"}

    _in_fresh_context(_main)
    tracing.force_flush_traces()

    turn = _by_name(exporter, "market-analyst")
    assert turn.attributes[attrs.UNIT_KIND] == "turn"
    assert turn.attributes[attrs.BEHAVIOUR_KEY] == "market-analyst"

    node_span = _by_name(exporter, "Market Analyst")
    assert node_span.parent.span_id == turn.context.span_id
    assert node_span.attributes[attrs.CODE_NAMESPACE] == market_analyst.__module__
    assert node_span.attributes[attrs.CODE_FUNCTION_NAME] == "market_analyst"
    assert node_span.attributes[attrs.BEHAVIOUR_KEY] == "market-analyst"
    assert "inputs" not in node_span.attributes  # capture="none"


def test_behaviours_override_and_share_one_turn_across_nodes(exporter):
    graph = bind(
        FakeGraph(**{"Market Analyst": market_analyst, "tools_market": lambda s: s}),
        behaviours={"Market Analyst": "analyst-tool-loop", "tools_market": "analyst-tool-loop"},
    )

    def _main():
        with start_span("root", span_type="entry_point"):
            graph.invoke_node("Market Analyst", {})
            graph.invoke_node("tools_market", {})
            graph.invoke_node("Market Analyst", {})

    _in_fresh_context(_main)
    tracing.force_flush_traces()

    # Non-contiguous activity of one behaviour shares a single turn span.
    turn = _by_name(exporter, "analyst-tool-loop")
    assert turn.attributes[attrs.UNIT_KIND] == "turn"
    node_spans = [s for s in exporter.get_finished_spans() if s.name == "Market Analyst"]
    assert len(node_spans) == 2
    assert all(s.parent.span_id == turn.context.span_id for s in node_spans)


def test_none_behaviour_opts_the_node_out(exporter):
    original = market_analyst
    graph = FakeGraph(**{"Msg Clear Market": original})
    bind(graph, behaviours={"Msg Clear Market": None})
    assert graph.nodes["Msg Clear Market"].runnable.func is original


def test_deliver_node_emits_delivery_inside_its_own_unit(exporter):
    graph = bind(
        FakeGraph(**{"Portfolio Manager": portfolio_manager}),
        deliver="Portfolio Manager",
    )

    def _main():
        with start_span("root", span_type="entry_point"):
            graph.invoke_node("Portfolio Manager", {})

    _in_fresh_context(_main)
    tracing.force_flush_traces()

    turn = _by_name(exporter, "portfolio-manager")
    delivery = _by_name(exporter, "deliver")
    assert delivery.parent.span_id == turn.context.span_id
    assert delivery.attributes[attrs.DELIVERY] is True
    assert "BUY" in delivery.attributes["outputs"]


def test_partial_node_anchors_identity_of_the_underlying_function(exporter):
    """Agent factories often return ``functools.partial(node_fn, name=...)`` —
    the partial unwraps so the user function still anchors code identity."""
    import functools

    def trader_node(state, name):
        return {"decision": name}

    graph = FakeGraph()
    graph.nodes["Trader"] = FakeSpec(FakeRunnable(func=functools.partial(trader_node, name="Trader")))
    bind(graph)

    def _main():
        with start_span("root", span_type="entry_point"):
            assert graph.invoke_node("Trader", {}) == {"decision": "Trader"}

    _in_fresh_context(_main)
    tracing.force_flush_traces()

    turn = _by_name(exporter, "trader")
    node_span = _by_name(exporter, "Trader")
    assert node_span.parent.span_id == turn.context.span_id
    assert node_span.attributes[attrs.CODE_FUNCTION_NAME].endswith("trader_node")


def test_bound_method_node_gets_turn_scope_but_no_identity_span(exporter):
    class ToolNodeish:
        def _func(self, state):
            return {"messages": []}

    tool_node = ToolNodeish()
    graph = FakeGraph()
    graph.nodes["tools_market"] = FakeSpec(FakeRunnable(func=tool_node._func))
    bind(graph)

    def _main():
        with start_span("root", span_type="entry_point"):
            graph.invoke_node("tools_market", {})

    _in_fresh_context(_main)
    tracing.force_flush_traces()

    turn = _by_name(exporter, "tools-market")
    assert turn.attributes[attrs.UNIT_KIND] == "turn"
    # No observe span was interposed for library-owned callables.
    assert not [s for s in exporter.get_finished_spans() if s.name == "tools_market"]


def test_wrappers_unwrap_to_the_original_callable(exporter):
    """LangGraph's compile-time subgraph scan follows ``__wrapped__`` when
    reading a node callable's source and closure. Without it, a wrapper closing
    over a bound method (whose ``__self__`` is the runnable being scanned)
    sends the scan into an infinite loop."""
    import inspect

    class ToolNodeish:
        def _func(self, state):
            return {"messages": []}

    tool_node = ToolNodeish()
    graph = FakeGraph(**{"Market Analyst": market_analyst})
    graph.nodes["tools_market"] = FakeSpec(FakeRunnable(func=tool_node._func))
    bind(graph)

    assert inspect.unwrap(graph.nodes["tools_market"].runnable.func) == tool_node._func
    assert inspect.unwrap(graph.nodes["Market Analyst"].runnable.func) is market_analyst
    # getsource resolves through the chain — what the scan actually reads.
    assert "def _func" in inspect.getsource(graph.nodes["tools_market"].runnable.func)


def test_async_native_node_is_bound_through_afunc(exporter):
    async def async_node(state):
        return {"ok": True}

    graph = FakeGraph()
    graph.nodes["Async Node"] = FakeSpec(FakeRunnable(afunc=async_node))
    bind(graph)

    def _main():
        async def _run():
            with start_span("root", span_type="entry_point"):
                return await graph.nodes["Async Node"].runnable.ainvoke({})

        return asyncio.run(_run())

    assert _in_fresh_context(_main) == {"ok": True}
    tracing.force_flush_traces()

    turn = _by_name(exporter, "async-node")
    node_span = _by_name(exporter, "Async Node")
    assert turn.attributes[attrs.UNIT_KIND] == "turn"
    assert node_span.parent.span_id == turn.context.span_id
    assert node_span.attributes[attrs.CODE_FUNCTION_NAME].endswith("async_node")


# --- validation -----------------------------------------------------------------


def test_bind_rejects_objects_without_node_specs():
    class Compiledish:
        nodes = {"a": object()}  # no .runnable

    with pytest.raises(TypeError, match="uncompiled langgraph StateGraph"):
        bind(Compiledish())
    with pytest.raises(TypeError):
        bind(object())


def test_bind_rejects_unknown_deliver_and_behaviour_names():
    with pytest.raises(ValueError, match="deliver="):
        bind(FakeGraph(**{"A": market_analyst}), deliver="missing")
    with pytest.raises(ValueError, match="unknown nodes"):
        bind(FakeGraph(**{"A": market_analyst}), behaviours={"B": "x"})
