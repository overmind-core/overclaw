"""Tests for the local JSONL file span exporter.

Verifies that a span emitted under the ``overmind`` tracer scope and exported
via :class:`overmind.tracing_file_exporter.JsonlFileSpanExporter` is written to
disk as a well-formed OTLP-JSON envelope containing the span's name and
attributes.

The exporter is the local sink wired into the agent subprocess by the CLI
daemon's runner (``OVERMIND_TRACE_FILE``); the Overmind server ingests those
spans out of band, so this test asserts the writer's output shape rather than
parsing it back with any local reader.
"""

from __future__ import annotations

import json
from pathlib import Path

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

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


def _spans_in(path: Path) -> list[dict]:
    """Flatten every span object across all OTLP-JSON envelope lines in *path*."""
    spans: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        envelope = json.loads(line)
        for rs in envelope.get("resource_spans", []):
            for ss in rs.get("scope_spans", []):
                spans.extend(ss.get("spans", []))
    return spans


def _attr(span: dict, key: str) -> dict | None:
    """Return the ``value`` object for *key* in an OTLP span's attribute list."""
    for attr in span.get("attributes", []):
        if attr.get("key") == key:
            return attr.get("value")
    return None


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

    spans = _spans_in(trace_path)
    names = [s.get("name") for s in spans]
    assert "my_tool" in names, f"exported spans should include 'my_tool', got: {names}"

    tool_span = next(s for s in spans if s.get("name") == "my_tool")
    assert _attr(tool_span, "name") == {"string_value": "my_tool"}
    inputs = _attr(tool_span, "inputs")
    assert inputs is not None and json.loads(inputs["string_value"]) == {"x": 1}
    outputs = _attr(tool_span, "outputs")
    assert outputs is not None and json.loads(outputs["string_value"]) == {"y": 2}


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
    names = [s.get("name") for s in _spans_in(nested)]
    assert "tool_a" in names, f"exported spans should include 'tool_a', got: {names}"
