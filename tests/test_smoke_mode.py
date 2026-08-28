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

    assert overmind.init() is True

    with (
        overmind.run("smoke-run", capability_id="11111111-1111-1111-1111-111111111111"),
        overmind.task("k", unit="turn"),
    ):
        pass

    spans = [json.loads(line) for line in trace_file.read_text().splitlines() if line.strip()]
    assert spans
    unit_spans = [span for span in spans if span["attributes"].get(attrs.BEHAVIOUR_KEY) == "k"]
    assert len(unit_spans) == 1
    span = unit_spans[0]
    assert len(span["trace_id"]) == 32
    assert len(span["span_id"]) == 16
    assert span["attributes"][attrs.UNIT_KIND] == "turn"
    assert span["attributes"][attrs.AGENT_ID] == "11111111-1111-1111-1111-111111111111"
    assert isinstance(span["events"], list)
    assert span["end_time_ns"] >= span["start_time_ns"]


def test_file_exporter_reuses_a_provider_someone_else_installed(tmp_path, monkeypatch):
    """The optimise runner wrapper installs its own file-exporter provider;
    init() must ride it instead of building a second one OTel would ignore."""
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider

    import overmind

    monkeypatch.setenv("OVERMIND_TRACE_FILE", str(tmp_path / "trace.jsonl"))
    monkeypatch.delenv("OVERMIND_API_KEY", raising=False)
    installed = TracerProvider()
    otel_trace.set_tracer_provider(installed)

    assert overmind.init() is True
    assert otel_trace.get_tracer_provider() is installed


@pytest.mark.skipif(not _installed("openai"), reason="openai not installed")
def test_smoke_stub_openai_returns_canned_response(monkeypatch):
    from overmind.smoke import activate_smoke_mode, deactivate_smoke_mode

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
def test_smoke_activate_is_idempotent_and_deactivate_restores(monkeypatch):
    from openai.resources.chat.completions import Completions

    from overmind.smoke import activate_smoke_mode, deactivate_smoke_mode

    original = Completions.create
    try:
        activate_smoke_mode()
        patched = Completions.create
        assert patched is not original
        activate_smoke_mode()
        assert Completions.create is patched
    finally:
        deactivate_smoke_mode()
    assert Completions.create is original


def test_smoke_skips_a_provider_whose_layout_does_not_match(monkeypatch, caplog):
    """Legacy openai<1 has no ``openai.resources``: skip that provider, never crash."""
    from overmind import smoke

    monkeypatch.setattr(smoke, "_installed", lambda module: True)
    monkeypatch.setattr(smoke, "_patch_openai", lambda: (_ for _ in ()).throw(ImportError("no openai.resources")))
    monkeypatch.setattr(smoke, "_patch_anthropic", lambda: None)
    monkeypatch.setattr(smoke, "_patch_google_genai", lambda: None)

    with caplog.at_level("DEBUG", logger="overmind.smoke"):
        smoke.activate_smoke_mode()

    assert any("smoke patch skipped" in record.message for record in caplog.records)


def test_init_activates_smoke_mode(tmp_path, monkeypatch):
    import overmind
    from overmind import smoke

    calls: list[int] = []
    monkeypatch.setattr(smoke, "activate_smoke_mode", lambda: calls.append(1))
    monkeypatch.setenv("OVERMIND_SMOKE", "1")
    monkeypatch.setenv("OVERMIND_TRACE_FILE", str(tmp_path / "trace.jsonl"))
    monkeypatch.delenv("OVERMIND_API_KEY", raising=False)

    overmind.init()

    assert calls == [1]
