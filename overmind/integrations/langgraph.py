"""Declarative Overmind binding for LangGraph ``StateGraph``s.

``bind(graph)`` wraps every node so each invocation runs inside
``task(key, unit="turn")`` — the node's behaviour becomes a scoring unit, and
re-entrant phases (tool loops, debate rounds) share one turn span. Nodes
backed by a user function additionally get an ``@observe(capture="none")``
span carrying that function's code identity, the anchor evidence the
platform's step judges bind against.

Call it on the built graph, after the ``add_node`` calls and before
``compile()``::

    workflow = build_my_state_graph()
    langgraph.bind(workflow, behaviours={...})
    app = workflow.compile()

This module never imports langgraph — it works structurally on the graph's
``nodes`` mapping — so it is import-safe in environments without it.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping
from typing import Any

from overmind.tracing import deliver as _deliver
from overmind.tracing import identity_slug, observe, task


def slug(name: str) -> str:
    """Default behaviour key for a node name (``"Market Analyst"`` →
    ``"market-analyst"``, ``"tools_market"`` → ``"tools-market"``)."""
    return identity_slug(name)


def _with_identity(target: Callable, name: str) -> Callable:
    """Interpose ``@observe(capture="none")`` on the user function behind the
    node callable, so its code identity anchors the node's spans. Partials
    unwrap (agent factories often return ``partial(node_fn, name=...)``);
    bound methods of runnable classes (e.g. ``ToolNode._func``) and
    langchain/langgraph-owned functions (executor plumbing) are library code,
    not anchors, and pass through untouched."""
    if isinstance(target, functools.partial):
        inner = _with_identity(target.func, name)
        if inner is target.func:
            return target
        return functools.partial(inner, *target.args, **target.keywords)
    if inspect.isfunction(target) and not target.__module__.startswith(("langchain", "langgraph")):
        return observe(span_name=name, capture="none")(target)
    return target


# ``functools.wraps`` on the wrappers is load-bearing, not cosmetic: LangGraph's
# compile-time subgraph scan (``find_subgraph_pregel``) analyses a node callable's
# source and closure, following bound methods to their ``__self__``. A bare
# closure over e.g. ``ToolNode._func`` leads the scan back to the ToolNode it is
# already expanding — an infinite loop. ``__wrapped__`` makes ``inspect``
# unwrap to the original callable, so the scan sees exactly what it saw
# before binding.
def _wrap_sync(call: Callable, key: str, delivers: bool) -> Callable:
    @functools.wraps(call)
    def bound(*args, **kwargs):
        with task(key, unit="turn"):
            result = call(*args, **kwargs)
            if delivers:
                _deliver(result)
            return result

    return bound


def _wrap_async(call: Callable, key: str, delivers: bool) -> Callable:
    @functools.wraps(call)
    async def bound(*args, **kwargs):
        async with task(key, unit="turn"):
            result = await call(*args, **kwargs)
            if delivers:
                _deliver(result)
            return result

    return bound


def _bind_node(runnable: Any, name: str, key: str, delivers: bool) -> None:
    """Rebind the node runnable's ``func`` / ``afunc`` in place. LangGraph's
    ``RunnableCallable`` fixed its call contract at construction, so an
    argument-transparent wrapper slots in without re-coercion. ``func`` is
    always invoked synchronously and ``afunc`` always awaited, so the wrapper
    flavour follows the attribute, not introspection."""
    for attr, wrap in (("func", _wrap_sync), ("afunc", _wrap_async)):
        target = getattr(runnable, attr, None)
        if target is None:
            continue
        setattr(runnable, attr, wrap(_with_identity(target, name), key, delivers))


def bind(
    graph,
    *,
    behaviours: Mapping[str, str | None] | None = None,
    deliver: str | None = None,
):
    """Bind every node of a LangGraph ``StateGraph`` to its behaviour's turn unit.

    Args:
        graph: The ``StateGraph`` (before ``compile()``).
        behaviours: Node name → behaviour key overrides. Unlisted nodes
            default to ``slug(node_name)``; a ``None`` value opts the node
            out (its spans stay interior to whatever unit encloses them).
        deliver: Name of the node whose return value is the run's terminal
            deliverable — its completion emits ``deliver()`` inside the
            node's own turn unit. Omit to deliver manually via the
            :func:`overmind.run` handle.

    Returns the same graph, nodes wrapped in place.
    """
    nodes = getattr(graph, "nodes", None)
    if not isinstance(nodes, Mapping) or any(not hasattr(spec, "runnable") for spec in nodes.values()):
        raise TypeError(
            "bind() expects an uncompiled langgraph StateGraph (call it after "
            f"add_node() and before compile()), got {type(graph).__name__}"
        )
    if deliver is not None and deliver not in nodes:
        raise ValueError(f"deliver={deliver!r} is not a node of the graph")
    overrides = behaviours or {}
    unknown = set(overrides) - set(nodes)
    if unknown:
        raise ValueError(f"behaviours= names unknown nodes: {sorted(unknown)}")

    for name, spec in nodes.items():
        key = overrides[name] if name in overrides else slug(name)
        if not key:
            continue
        _bind_node(spec.runnable, name, key, delivers=name == deliver)
    return graph


__all__ = ["bind", "slug"]
