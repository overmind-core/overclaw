"""Roundtrip test for the local JSONL file span exporter.

Verifies that a span emitted under the ``overmind`` tracer scope and
exported via :class:`overmind.tracing_file_exporter.JsonlFileSpanExporter`
can be read back by
:func:`overmind.optimize.trace_reader.parse_trace_file_per_line` into a
:class:`overmind.optimize.trace_reader.ParsedTrace` whose ``tool_trace``
contains the expected ``{name, args, result, ...}`` entry.

If this regresses, every Tool Usage score will fall back to "unscored"
again and the optimizer's analyzer will lock focus onto tool descriptions
indefinitely.
"""

from __future__ import annotations

import json
from pathlib import Path

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from overmind.optimize.trace_reader import parse_trace_file_per_line
from overmind.tracing_file_exporter import JsonlFileSpanExporter


def _new_provider(path: Path) -> TracerProvider:
    """Build a TracerProvider whose only exporter writes to *path*.

    Use ``SimpleSpanProcessor`` here (not Batch) so spans flush
    synchronously on span end — keeps the test deterministic and avoids
    the need for an ``OS``-level shutdown sleep.
    """
    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(SimpleSpanProcessor(JsonlFileSpanExporter(path)))
    return provider


def test_file_exporter_writes_overmind_scope(tmp_path: Path):
    trace_path = tmp_path / "case_0000.jsonl"
    provider = _new_provider(trace_path)
    tracer = provider.get_tracer("overmind")

    with tracer.start_as_current_span("entry"), tracer.start_as_current_span("my_tool") as child:
        child.set_attribute("name", "my_tool")
        child.set_attribute("type", "function")
        child.set_attribute("inputs", json.dumps({"x": 1}))
        child.set_attribute("outputs", json.dumps({"y": 2}))

    provider.force_flush()
    provider.shutdown()

    assert trace_path.exists(), "exporter should have written a JSONL line"
    text = trace_path.read_text(encoding="utf-8").strip()
    assert text, "file should be non-empty"

    parsed = parse_trace_file_per_line(trace_path)
    merged_tool_trace = [t for trace in parsed for t in trace.tool_trace]

    names = [t["name"] for t in merged_tool_trace]
    assert "my_tool" in names, f"tool_trace should include 'my_tool', got: {names}"

    tool_entry = next(t for t in merged_tool_trace if t["name"] == "my_tool")
    assert tool_entry.get("args", {}).get("x") == 1
    assert tool_entry.get("result", {}).get("y") == 2


def test_file_exporter_handles_missing_dir(tmp_path: Path):
    nested = tmp_path / "deep" / "nested" / "case.jsonl"
    provider = _new_provider(nested)
    tracer = provider.get_tracer("overmind")

    with tracer.start_as_current_span("entry"), tracer.start_as_current_span("tool_a") as child:
        child.set_attribute("name", "tool_a")
        child.set_attribute("type", "function")

    provider.force_flush()
    provider.shutdown()

    assert nested.exists()
    parsed = parse_trace_file_per_line(nested)
    assert any("tool_a" in [t["name"] for t in trace.tool_trace] for trace in parsed)


# Intentionally no teardown that touches the global tracer provider.
# OTel does not support resetting ``set_tracer_provider`` cleanly; clobbering
# the singleton here would break any later test that calls
# ``otel_trace.get_tracer_provider().force_flush(...)``. The locally-built
# provider in each test stays installed for the rest of the test session,
# which is harmless: nothing else exports spans here.
