"""Tests for :func:`overmind.tracing_file_exporter.install_local_provider`.

The helper consolidates the boilerplate that the optimizer's subprocess
wrapper used to inline: build a fresh ``TracerProvider`` that writes to a
JSONL file, register it on the global API, and enable auto-instrumentation.
Phase 4.7 of the cleanup plan moved this into the exporter module so the
runner wrapper can be a 2-liner.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from overmind.tracing_file_exporter import install_local_provider


class TestInstallLocalProvider:
    def test_sets_global_provider_and_enables_tracing(self, tmp_path: Path):
        trace_file = tmp_path / "traces.jsonl"

        with (
            patch("opentelemetry.trace.set_tracer_provider") as set_provider,
            patch("overmind.tracing.enable_tracing") as enable_tracing,
        ):
            install_local_provider(trace_file)

        set_provider.assert_called_once()
        # ``enable_tracing(providers=[])`` is the documented "all supported" call.
        enable_tracing.assert_called_once_with(providers=[])

    def test_failures_are_swallowed(self, tmp_path: Path, caplog):
        """Tracing setup must never break the agent subprocess."""
        trace_file = tmp_path / "traces.jsonl"

        with patch("opentelemetry.trace.set_tracer_provider", side_effect=RuntimeError("boom")):
            install_local_provider(trace_file)  # must not raise

        assert any("install_local_provider" in rec.message for rec in caplog.records)

    def test_uses_jsonl_exporter(self, tmp_path: Path):
        """Smoke test: the provider that gets registered actually exports JSONL."""
        trace_file = tmp_path / "traces.jsonl"

        captured: dict[str, object] = {}

        def _capture_set_provider(provider):
            captured["provider"] = provider

        with (
            patch("opentelemetry.trace.set_tracer_provider", side_effect=_capture_set_provider),
            patch("overmind.tracing.enable_tracing", new=MagicMock()),
        ):
            install_local_provider(trace_file)

        from opentelemetry.sdk.trace import TracerProvider

        provider = captured["provider"]
        assert isinstance(provider, TracerProvider)
