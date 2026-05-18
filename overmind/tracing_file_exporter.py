"""OTel ``SpanExporter`` that writes spans to a local JSONL trace file.

Each call to :meth:`JsonlFileSpanExporter.export` (one per batch flushed by the
configured ``BatchSpanProcessor``) appends a single JSON object — the
OTLP ``resource_spans`` envelope — to the file. The format is exactly what
:func:`overmind.optimize.trace_reader.parse_trace_file_per_line` consumes,
so the local optimizer can read back per-case tool traces without any
backend roundtrip.

This is the local sibling of the OTLP HTTP exporter installed by
:func:`overmind.tracing.init`: it is activated inside the optimizer's
agent subprocess when the runner sets the ``OVERMIND_TRACE_FILE`` env
var. No API key, no network, no cloud dependency — the file is the only
sink.

Concurrent writers across processes are safe as long as each process
gets its own ``trace_file`` path (the optimizer allocates one per case).
Within a process, exports are serialised by the ``BatchSpanProcessor``,
and each export ends with an ``fsync`` so the parent can read every
span after the subprocess exits.

:func:`install_local_provider` is the one-call bootstrap used by the
optimizer's runner subprocess: it wires up a fresh ``TracerProvider`` that
exports *only* to a JSONL file (no cloud, no API key) and enables the
supported LLM-provider auto-instrumentations.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

logger = logging.getLogger(__name__)


def install_local_provider(path: str | os.PathLike[str]) -> None:
    """Install a JSONL-file-only :class:`TracerProvider` on the global API.

    Used by the optimizer's subprocess wrapper to enable tracing without
    any cloud roundtrip — the entire span stream is written to *path*.
    Best-effort: any failure to set up tracing is swallowed so the agent
    process can still run.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        from overmind.tracing import enable_tracing

        provider = TracerProvider(resource=Resource.create({"service.name": "overmind-optimize-subprocess"}))
        provider.add_span_processor(BatchSpanProcessor(JsonlFileSpanExporter(path)))
        trace.set_tracer_provider(provider)
        enable_tracing(providers=[])
    except Exception:
        logger.exception("install_local_provider: tracing setup failed")


class JsonlFileSpanExporter(SpanExporter):
    """Append OTLP-encoded span batches to a JSONL file.

    The encoded envelope is the protobuf ``TracesData`` message converted
    to a plain dict via :func:`google.protobuf.json_format.MessageToDict`
    with ``preserving_proto_field_name=True`` so keys are snake_case
    (matching what :mod:`overmind.optimize.trace_reader` parses).
    """

    def __init__(self, path: str | os.PathLike[str]):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not spans:
            return SpanExportResult.SUCCESS
        try:
            envelope = self._encode(spans)
        except Exception as exc:
            logger.warning(f"JsonlFileSpanExporter: failed to encode {len(spans)} span(s): {exc}")
            return SpanExportResult.FAILURE
        line = json.dumps(envelope, default=str, separators=(",", ":")) + "\n"
        try:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
        except OSError as exc:
            logger.warning(f"JsonlFileSpanExporter: write to {self._path} failed: {exc}")
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    def shutdown(self) -> None:
        return None

    @staticmethod
    def _encode(spans: Sequence[ReadableSpan]) -> dict:
        from google.protobuf.json_format import MessageToDict
        from opentelemetry.exporter.otlp.proto.common._internal.trace_encoder import encode_spans

        traces_data = encode_spans(spans)
        return MessageToDict(traces_data, preserving_proto_field_name=True)


__all__ = ["JsonlFileSpanExporter", "install_local_provider"]
