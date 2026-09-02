"""Core-install isolation: with the cli/inference extras absent (litellm,
typer, rich, psutil blocked via sys.modules), ``import overmind`` and the
whole tracing surface must work, cost enrichment must degrade with an
install hint, and the CLI must raise an error naming the extra.

Runs in a subprocess so the blocked modules can't leak in from the test
session's already-imported state.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

_SCRIPT = textwrap.dedent("""
    import logging
    import sys

    for blocked in ("litellm", "typer", "rich", "psutil"):
        sys.modules[blocked] = None

    import overmind  # noqa: E402 — must import with core deps only
    import overmind.tracing as tracing
    from overmind import attrs

    # Tracing surface end-to-end against an in-memory exporter.
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = TracerProvider(sampler=tracing._OrphanSpanSampler())
    inmem = InMemorySpanExporter()
    proc = SimpleSpanProcessor(inmem)
    provider.add_span_processor(proc)
    proc.on_start = tracing._span_processor_on_start
    tracing._tracer = provider.get_tracer("overmind", "test")
    tracing._initialized = True

    @overmind.entry_point()
    def handle(question: str) -> str:
        overmind.deliver("answer")
        return "answer"

    with overmind.run("core-only-run") as handle_run:
        pass
    assert handle("q") == "answer"
    names = {s.name for s in inmem.get_finished_spans()}
    assert {"core-only-run", "handle", "deliver"} <= names, names

    # Cost enrichment degrades with the inference-extra hint, never raises.
    from overmind import genai_usage

    logging.basicConfig(level=logging.INFO)
    assert genai_usage.compute_cost("gpt-4o", 10, 10) is None
    assert genai_usage._litellm_missing_logged, "expected the overmind[inference] hint"

    # The inference client rides on core deps (requests ships with the OTLP exporter).
    assert overmind.Client is not None

    # The CLI names its extra instead of dying on a bare ModuleNotFoundError.
    try:
        import overmind.__main__  # noqa: F401
    except ImportError as exc:
        assert "overmind[cli]" in str(exc), str(exc)
    else:
        raise AssertionError("overmind.__main__ imported without the cli extra")

    print("EXTRAS-ISOLATION-OK")
""")


def test_tracing_surface_works_without_extras():
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "EXTRAS-ISOLATION-OK" in result.stdout
