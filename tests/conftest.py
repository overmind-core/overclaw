"""Shared fixtures for the Overmind test suite."""

from __future__ import annotations

import os

from opentelemetry import trace as _otel_trace
from opentelemetry.sdk.trace import TracerProvider

import overmind.tracing as _overmind_tracing

# ---------------------------------------------------------------------------
# OTel / overmind SDK bootstrap for tests
# ---------------------------------------------------------------------------
#
# ``overmind.start_span`` / ``overmind.observe`` / ``overmind.set_tag`` all
# go through ``overmind.tracing.get_tracer()`` which raises until
# ``overmind.init()`` is called.  We don't want test runs to actually
# export anything, so we install a no-op ``TracerProvider`` (no exporter)
# and flip the SDK's internal ``_initialized`` flag manually.

_provider = TracerProvider()
_otel_trace.set_tracer_provider(_provider)
_overmind_tracing._tracer = _provider.get_tracer("overmind", "test")
_overmind_tracing._initialized = True
os.environ["OVERMIND_API_KEY"] = "test"
