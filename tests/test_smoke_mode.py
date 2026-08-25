"""Tests for the ``OVERMIND_TRACE_FILE`` exporter and ``OVERMIND_SMOKE`` stub."""

from __future__ import annotations

import importlib.util
import json

import pytest


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


@pytest.fixture(autouse=True)
def reset_sdk_state():
    """Reset SDK state before each test, restoring it afterwards (mirrors
    tests/test_sdk.py so other modules' no-op tracer isn't disturbed)."""
    from opentelemetry import trace as otel_trace
    from opentelemetry.util._once import Once

    import overmind.tracing as sdk

    saved_initialized = sdk._initialized
    saved_tracer = sdk._tracer
    saved_provider = otel_trace._TRACER_PROVIDER
    saved_once = otel_trace._TRACER_PROVIDER_SET_ONCE
    sdk._initialized = False
    sdk._tracer = None
    otel_trace._TRACER_PROVIDER = None
    otel_trace._TRACER_PROVIDER_SET_ONCE = Once()
    yield
    sdk._initialized = saved_initialized
    sdk._tracer = saved_tracer
    otel_trace._TRACER_PROVIDER = saved_provider
    otel_trace._TRACER_PROVIDER_SET_ONCE = saved_once


def test_file_exporter_roundtrip(tmp_path, monkeypatch):
    import overmind
    from overmind import attrs

    trace_file = tmp_path / "trace.jsonl"
    monkeypatch.setenv("OVERMIND_TRACE_FILE", str(trace_file))
    monkeypatch.delenv("OVERMIND_API_KEY", raising=False)

    overmind.init()

    @overmind.task("k")
    def run():
        return "done"

    run()
    overmind.tracing.force_flush_traces()

    lines = [json.loads(line) for line in trace_file.read_text().splitlines() if line.strip()]
    assert lines
    task_spans = [line for line in lines if line["attributes"].get(attrs.BEHAVIOUR_KEY) == "k"]
    assert len(task_spans) == 1
    span = task_spans[0]
    assert span["trace_id"] and len(span["trace_id"]) == 32
    assert span["span_id"] and len(span["span_id"]) == 16
    assert isinstance(span["events"], list)


@pytest.mark.skipif(not _installed("openai"), reason="openai not installed")
def test_smoke_stub_openai_returns_canned_response(monkeypatch):
    from overmind.smoke import activate_smoke_mode, deactivate_smoke_mode

    monkeypatch.setenv("OVERMIND_SMOKE", "1")
    monkeypatch.setenv("OVERMIND_SMOKE_RESPONSE", "hello from smoke")
    try:
        activate_smoke_mode()
        from openai.resources.chat.completions import Completions

        instance = Completions.__new__(Completions)
        response = instance.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
        assert response.choices[0].message.content == "hello from smoke"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
    finally:
        deactivate_smoke_mode()


@pytest.mark.skipif(not _installed("openai"), reason="openai not installed")
def test_smoke_deactivate_restores_original():
    from openai.resources.chat.completions import Completions

    from overmind.smoke import activate_smoke_mode, deactivate_smoke_mode

    original = Completions.create
    activate_smoke_mode()
    assert Completions.create is not original
    deactivate_smoke_mode()
    assert Completions.create is original
