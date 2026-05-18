"""Parse overmind OTel JSONL trace files into structured data.

The overmind (via ``OVERMIND_TRACE_FILE``) writes one JSON object
per line in OpenTelemetry ``resource_spans`` format.  Each line contains
two scope groups:

  1. **overmind** — function spans created by ``@observe()``, carrying
     ``inputs`` and ``outputs`` as JSON-encoded attribute strings.
  2. **opentelemetry.instrumentation.openai.v1** — LLM client spans
     with ``gen_ai.*`` / ``llm.*`` attributes (model, prompts, tokens,
     completions, tool calls).

This module converts those spans into the ``tool_trace`` list-of-dicts
format consumed by :class:`overmind.optimize.evaluator.SpecEvaluator`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class ParsedTrace:
    """Aggregated data extracted from an overmind trace file."""

    tool_trace: list[dict] = field(default_factory=list)
    """tool_trace items: {name, args, result, error, latency_ms, source?}"""

    total_tokens: int = 0
    total_cost: float = 0.0
    spans: list[dict] = field(default_factory=list)
    """All raw function spans (overmind scope)."""

    llm_spans: list[dict] = field(default_factory=list)
    """All raw LLM spans (openai instrumentation scope)."""

    source_tags: list[dict] = field(default_factory=list)
    """Per-call provenance tags harvested from the shadow sidecar file.

    Each tag is ``{"name": str, "source": str, "reason": str, "ts": float}``.
    Empty when the run happened in normal subprocess mode.  Extended by
    :func:`attach_shadow_provenance` after parsing the trace.
    """


# ---------------------------------------------------------------------------
# Attribute helpers
# ---------------------------------------------------------------------------


def _attrs_to_dict(attributes: list[dict]) -> dict[str, str | int | float | bool]:
    """Convert OTel attributes list to a flat dict."""
    result: dict = {}
    for attr in attributes:
        key = attr.get("key", "")
        value = attr.get("value", {})
        if "string_value" in value:
            result[key] = value["string_value"]
        elif "int_value" in value:
            result[key] = int(value["int_value"])
        elif "bool_value" in value:
            result[key] = value["bool_value"]
        elif "double_value" in value:
            result[key] = float(value["double_value"])
    return result


def _parse_json_attr(attrs: dict, key: str, default=None):
    """Parse a JSON-encoded string attribute, returning the parsed value."""
    raw = attrs.get(key)
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _nano_to_ms(start_ns: str | int, end_ns: str | int) -> float:
    """Convert nanosecond timestamps to millisecond duration."""
    try:
        return (int(end_ns) - int(start_ns)) / 1_000_000
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


def parse_trace_file(path: str | Path) -> ParsedTrace:
    """Parse an overmind JSONL trace file and return a single aggregated trace.

    All JSONL lines are merged into one :class:`ParsedTrace`.
    """
    traces = parse_trace_file_per_line(path)
    merged = ParsedTrace()
    for t in traces:
        merged.tool_trace.extend(t.tool_trace)
        merged.total_tokens += t.total_tokens
        merged.total_cost += t.total_cost
        merged.spans.extend(t.spans)
        merged.llm_spans.extend(t.llm_spans)
    return merged


def parse_trace_file_per_line(path: str | Path) -> list[ParsedTrace]:
    """Parse an overmind JSONL trace file, returning one ParsedTrace per line.

    Each JSONL line corresponds to one subprocess invocation (one datapoint).
    Returns an empty list if the file doesn't exist or is empty.
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"Trace file not found: {path}")
        return []

    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.warning(f"Could not read trace file {path}: {exc}")
        return []

    if not text:
        return []

    results: list[ParsedTrace] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        per_line = ParsedTrace()
        _process_resource_spans(data, per_line)
        results.append(per_line)

    return results


def _process_resource_spans(data: dict, result: ParsedTrace) -> None:
    """Process a single JSONL line (one ``resource_spans`` object)."""
    for rs in data.get("resource_spans", []):
        for ss in rs.get("scope_spans", []):
            scope_name = ss.get("scope", {}).get("name", "")
            spans = ss.get("spans", [])

            if scope_name == "overmind":
                _process_overmind_spans(spans, result)
            elif "openai" in scope_name or "opentelemetry.instrumentation" in scope_name:
                _process_llm_spans(spans, result)


def _process_overmind_spans(spans: list[dict], result: ParsedTrace) -> None:
    """Extract function spans from the overmind scope."""
    for span in spans:
        attrs = _attrs_to_dict(span.get("attributes", []))
        name = attrs.get("name", span.get("name", ""))
        span_type = attrs.get("type", "function")

        inputs = _parse_json_attr(attrs, "inputs", {})
        outputs = _parse_json_attr(attrs, "outputs", {})

        latency_ms = _nano_to_ms(
            span.get("start_time_unix_nano", 0),
            span.get("end_time_unix_nano", 0),
        )

        status = span.get("status", {})
        error = None
        if status.get("code") == "STATUS_CODE_ERROR":
            error = status.get("message", "unknown error")

        parsed = {
            "name": name,
            "type": span_type,
            "inputs": inputs,
            "outputs": outputs,
            "latency_ms": latency_ms,
            "error": error,
            "span_id": span.get("span_id"),
            "parent_span_id": span.get("parent_span_id"),
            "trace_id": span.get("trace_id"),
        }
        result.spans.append(parsed)

        # Build tool_trace item — every function span except the top-level
        # entry point (which has no parent_span_id) becomes a tool_trace entry.
        if span.get("parent_span_id"):
            args = inputs if isinstance(inputs, dict) else {}
            tool_item = {
                "name": name,
                "args": args,
                "result": outputs,
                "error": error,
                "latency_ms": latency_ms,
            }
            result.tool_trace.append(tool_item)


def _process_llm_spans(spans: list[dict], result: ParsedTrace) -> None:
    """Extract token counts and cost from LLM instrumentation spans."""
    for span in spans:
        attrs = _attrs_to_dict(span.get("attributes", []))

        # Prefer the OTel GenAI semconv key the SDK emits; fall back to the
        # legacy ``llm.usage.total_tokens`` for spans produced by older
        # auto-instrumentors.
        total_tokens = attrs.get(
            "gen_ai.usage.total_tokens",
            attrs.get("llm.usage.total_tokens", 0),
        )
        if isinstance(total_tokens, str):
            try:
                total_tokens = int(total_tokens)
            except ValueError:
                total_tokens = 0
        result.total_tokens += total_tokens

        parsed_llm = {
            "name": span.get("name", ""),
            "model": attrs.get("gen_ai.request.model", ""),
            "response_model": attrs.get("gen_ai.response.model", ""),
            "total_tokens": total_tokens,
            # Auto-instrumentors sometimes emit the older
            # ``prompt_tokens`` / ``completion_tokens`` names; accept both.
            "input_tokens": attrs.get(
                "gen_ai.usage.input_tokens",
                attrs.get("gen_ai.usage.prompt_tokens", 0),
            ),
            "output_tokens": attrs.get(
                "gen_ai.usage.output_tokens",
                attrs.get("gen_ai.usage.completion_tokens", 0),
            ),
            "finish_reason": attrs.get("gen_ai.completion.0.finish_reason", ""),
            "span_id": span.get("span_id"),
            "parent_span_id": span.get("parent_span_id"),
        }
        result.llm_spans.append(parsed_llm)


# ---------------------------------------------------------------------------
# Shadow provenance decoration
# ---------------------------------------------------------------------------


def attach_shadow_provenance(traces: list[ParsedTrace], sidecar_tags: list[list[dict]]) -> None:
    """Copy per-case shadow sidecar tags onto matching :class:`ParsedTrace`s.

    The shadow runtime writes one provenance JSONL file per subprocess call
    (see :mod:`overmind.optimize.shadow_runtime`).  *sidecar_tags* has the
    same length and ordering as *traces* — index ``i`` in *sidecar_tags*
    belongs to ``traces[i]``.  When the shadow bootstrap is not active the
    caller passes empty lists and nothing changes.

    Tool-trace rows whose ``name`` matches a tag's ``name`` (or the literal
    LLM identifier ``"llm:<model>"``) also gain a ``"source"`` key so
    downstream scoring can down-weight simulated rows.
    """
    for trace, tags in zip(traces, sidecar_tags):
        if not tags:
            continue
        trace.source_tags.extend(tags)
        if not trace.tool_trace:
            continue
        tag_by_name = {t.get("name", ""): t for t in tags}
        for row in trace.tool_trace:
            tag = tag_by_name.get(row.get("name", ""))
            if tag:
                row["source"] = tag.get("source", "")
