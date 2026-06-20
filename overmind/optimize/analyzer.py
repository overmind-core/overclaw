"""
Analysis engine: examines per-test-case results, identifies failure patterns,
and generates improved agent code.

Uses a **two-pass** approach:
  Pass 1 (Diagnosis): Analyze failures, tool usage, and score breakdowns to
      produce a structured diagnosis with specific change instructions.
  Pass 2 (Code Generation): Given the diagnosis, produce the updated agent code.

Supports generating multiple candidates in parallel (best-of-N).
Supports multi-file agents via the ``AgentBundle`` virtual representation.
"""

from __future__ import annotations

import ast
import contextvars
import json
import logging
import os
import random
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from overmind import SpanType, attrs, set_tag
from overmind.prompts.analyzer import (
    _BUNDLE_OUTPUT_INSTRUCTION,
    _SINGLE_FILE_OUTPUT_INSTRUCTION,
    AGENTIC_CODEGEN_FOCUS,
    AGENTIC_CODEGEN_INSTRUCTION,
    CODEGEN_FOCUS_DIRECTIVE,
    CODEGEN_PROMPT,
    COMPONENT_IMPACT_SECTION,
    DIAGNOSIS_FOCUS_DIRECTIVE,
    DIAGNOSIS_PROMPT,
    DIAGNOSIS_SYSTEM_PROMPT,
    FAILURE_CLUSTERS_SECTION,
    FOCUS_LABELS,
    MULTI_FILE_AWARENESS_SECTION,
    SINGLE_PASS_PROMPT,
)
from overmind.tracing import observe_safe
from overmind.utils.llm import llm_completion
from overmind.utils.llm_parse import parse_json_object

if TYPE_CHECKING:
    from overmind.optimize.failure_registry import FailureRegistry
    from overmind.utils.code import AgentBundle

_log = logging.getLogger("overmind.optimize.analyzer")


# ---------------------------------------------------------------------------
# Argument-bundle dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DiagnosisContext:
    """Inputs to :func:`_run_diagnosis` that stay constant across iterations.

    The diagnosis prompt is parameterised by a long list of inputs — agent
    code, per-case results, eval spec, policy text, prior attempts, etc.
    Bundling them in this dataclass lets callers build the context once and
    pass it through to every diagnosis call, instead of threading a dozen
    keyword arguments through ``generate_candidates`` and the codegen
    helpers.

    Iteration-specific knobs (focus area, case fraction, max-cases budget)
    stay as explicit parameters on :func:`_run_diagnosis` so callers can
    vary them per call.
    """

    agent_code: str
    case_results: list[dict]
    evaluation_results: dict
    model: str
    entrypoint_fn: str
    eval_spec: dict | None = None
    failed_attempts: list[dict] | None = None
    successful_changes: list[dict] | None = None
    allow_model_change: bool = False
    temperature: float = 0.7
    iteration_seed: int = 42
    policy_context: str = ""
    bundle: AgentBundle | None = None
    cluster_context: str = ""
    component_weights_context: str = ""


@dataclass
class CodegenSettings:
    """Inputs to the codegen phase used by :func:`generate_candidates`.

    Separated from :class:`DiagnosisContext` because the codegen step has
    its own model and step-budget knobs that the diagnosis step doesn't
    need.
    """

    codegen_model: str = ""
    codegen_max_steps: int = 50
    policy_constraints: str = ""
    agent_files: dict[str, str] | None = None
    num_candidates: int = 3
    return_plans_only: bool = False
    focus_weights: dict[str, float] | None = field(default=None)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _measure_system_prompt(agent_code: str) -> tuple[int, int]:
    """Extract SYSTEM_PROMPT from agent code and measure its size."""
    m = re.search(
        r'SYSTEM_PROMPT\s*=\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')',
        agent_code,
        re.DOTALL,
    )
    if m:
        prompt_text = m.group(1)
        return len(prompt_text), prompt_text.count("\n") + 1
    return 0, 0


def _format_scoring_mechanics(eval_spec: dict | None) -> str:
    if not eval_spec:
        return "(no evaluation spec available)"

    lines: list[str] = []
    sw = eval_spec.get("structure_weight", 20)
    lines.append(
        f"**Structure: {sw} pts** — All expected fields present and non-empty. Score = (present / total) * {sw}."
    )

    for field_name, config in eval_spec.get("output_fields", {}).items():
        label = field_name.replace("_", " ").title()
        weight = config.get("weight", 0)
        ftype = config.get("type", "unknown")

        if ftype == "enum":
            vals = ", ".join(config.get("values", []))
            if config.get("partial_credit"):
                ps = config.get("partial_score", 0)
                lines.append(
                    f"**{label}: {weight} pts** (enum: {vals}) — Exact match = {weight}. Valid but wrong = {ps}."
                )
            else:
                lines.append(f"**{label}: {weight} pts** (enum: {vals}) — Exact match = {weight}. Any mismatch = 0.")
        elif ftype == "number":
            bands = config.get("tolerance_bands", [])
            field_range = config.get("range", [])
            if bands:
                parts = [f"±{b['within']} → {b['score_pct'] * 100:.0f}%" for b in bands]
                range_note = ""
                if field_range and len(field_range) == 2:
                    lo, hi = field_range
                    range_note = f" Field range: [{lo:,}–{hi:,}]."
                    tightest = min(b["within"] for b in bands)
                    widest = max(b["within"] for b in bands)
                    if hi > 0 and tightest > 0:
                        pct_tight = tightest / hi * 100
                        if pct_tight < 1.0:
                            range_note += (
                                f" WARNING: tolerances are ABSOLUTE values — "
                                f"the tightest band (±{tightest:g}) is "
                                f"{pct_tight:.4f}% of the field range, "
                                f"so the output must NEARLY EXACTLY match "
                                f"the expected value. Any value beyond "
                                f"±{widest:g} absolute scores 0."
                            )
                lines.append(
                    f"**{label}: {weight} pts** (number) — "
                    f"ABSOLUTE proximity bands: {', '.join(parts)}. "
                    f"Beyond = 0.{range_note}"
                )
            else:
                tol = config.get("tolerance", 10)
                lines.append(
                    f"**{label}: {weight} pts** (number) — "
                    f"Within ±{tol} absolute = full, "
                    f"±{tol * 2} absolute = half, beyond = 0."
                )
        elif ftype == "text":
            mode = config.get("eval_mode", "non_empty")
            mode_desc = {
                "non_empty": "non-empty check (any text scores full points)",
                "similarity": "token similarity vs expected (Jaccard + coverage)",
                "keyword_coverage": "fraction of expected keywords present",
                "llm_judge": "per-field LLM comparison vs expected text",
                "skip": "not scored",
            }.get(mode, f"{mode} check")
            lines.append(f"**{label}: {weight} pts** (text) — {mode_desc}.")
        elif ftype == "boolean":
            lines.append(f"**{label}: {weight} pts** (boolean) — Exact match only.")

    tw = eval_spec.get("tool_usage_weight", 0)
    if tw > 0:
        lines.append(f"**Tool Usage: {tw} pts** — Correct tool calls, arguments, chaining.")

    jw = eval_spec.get("llm_judge_weight", 0)
    if jw > 0:
        lines.append(f"**LLM Judge: {jw} pts** — Semantic correctness, consistency, reasoning.")

    lines.append(
        "**Type Correctness: penalty** — Each field with wrong type "
        "(e.g., string where number expected) deducts 2 pts (capped at -10)."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool-trace formatting (schema-aware, two-tier)
# ---------------------------------------------------------------------------
#
# Tool-heavy agents (multi-agent frameworks, deep chains, search-and-rerank
# pipelines) can record dozens of tool calls per case whose ``args``/``result``
# JSON payloads are kilobytes each.  Dumping that raw into the analyzer prompt
# scales linearly with (cases × calls × payload size) and routinely blows past
# the model's context window on real workloads.
#
# The signal the analyzer actually needs is:
#   1. Which tools were called, in what order (control-flow shape).
#   2. Which calls *failed*.
#   3. For the most diagnostic cases, the literal args / results so the model
#      can spot wrong values, malformed inputs, off-by-one indexes, etc.
#
# We satisfy 1+2 cheaply for every case (a compact run-length-encoded name
# sequence plus any error calls in full) and reserve the expensive full
# args/results form for the worst N cases, where deep inspection pays.


# Number of worst-scoring cases that get the full per-call tool trace.
# Configurable via env so projects with tiny model context windows can
# dial it down to 1, and projects with huge context windows can lift it
# without editing source.
_FULL_TRACE_TOP_N_DEFAULT = 3

# Conservative chars-per-token estimate.  Anthropic/OpenAI English text
# is ~3.5-4 chars/token; code is ~2.5-3 chars/token; mixed JSON/code
# (which is what analyzer prompts are) sits near 2.8.  We pick the low
# end so we over-shrink rather than under-shrink — the alternative is
# a hard API rejection that wastes a whole iteration.
_CHARS_PER_TOKEN_CONSERVATIVE = 2.8

# Default analyzer context budget in *tokens* (not chars).  Set well
# under the typical 200k-token ceiling for Claude / GPT-4o-class models
# to leave headroom for the model's reply.  Wider-context models
# (Gemini 2M, GPT-5 1M) can opt in via ``OVERMIND_ANALYZER_PROMPT_BUDGET_TOKENS``.
_PROMPT_BUDGET_TOKENS_DEFAULT = 160_000


def _full_trace_top_n() -> int:
    try:
        return max(0, int(os.environ.get("OVERMIND_ANALYZER_FULL_TRACE_TOP_N", _FULL_TRACE_TOP_N_DEFAULT)))
    except (TypeError, ValueError):
        return _FULL_TRACE_TOP_N_DEFAULT


def _budget_tokens_for(model: str | None) -> int:
    """Resolve a token budget for the analyzer prompt.

    Priority order:
      1. ``OVERMIND_ANALYZER_PROMPT_BUDGET_TOKENS`` env var (explicit user
         override; useful for wider-context models or paranoid setups).
      2. ``litellm.get_model_info(model)["max_input_tokens"]`` × safety
         factor (0.85) when the model is known and litellm has it.
      3. :data:`_PROMPT_BUDGET_TOKENS_DEFAULT` (160k) as a fallback.

    Returned value is always **input** tokens; callers convert to chars
    via :data:`_CHARS_PER_TOKEN_CONSERVATIVE`.
    """
    env_val = os.environ.get("OVERMIND_ANALYZER_PROMPT_BUDGET_TOKENS")
    if env_val:
        try:
            return max(8_000, int(env_val))
        except (TypeError, ValueError):
            pass
    if model:
        try:
            import litellm  # type: ignore

            info = litellm.get_model_info(model) or {}
            mit = info.get("max_input_tokens") or info.get("max_tokens")
            if isinstance(mit, int) and mit > 0:
                return max(8_000, int(mit * 0.85))
        except Exception:
            # litellm doesn't know this model (custom provider, typo, or
            # offline catalog).  Fall through to the default budget — a
            # mis-sized prompt is a degraded experience, not a crash.
            _log.debug("litellm.get_model_info(%r) failed; using default prompt budget", model, exc_info=True)
    return _PROMPT_BUDGET_TOKENS_DEFAULT


def _budget_chars_for(model: str | None) -> int:
    """Char-space budget derived from :func:`_budget_tokens_for`."""
    tokens = _budget_tokens_for(model)
    return max(20_000, int(tokens * _CHARS_PER_TOKEN_CONSERVATIVE))


def _summarize_value_shape(
    v: Any,
    *,
    max_str_chars: int = 80,
    max_list_items: int = 3,
) -> str:
    """Return a compact, schema-aware representation of *v*.

    Preserves structural information (keys, types, counts) that helps the
    analyzer reason about tool behavior while dropping bulk content (long
    strings, large arrays of records) that balloon tokens without adding
    diagnostic signal.

    Examples:
        {"flights": [...50 dicts...]}       -> "{flights: [50 of {airline, dep, ...}]}"
        "very long error message ..."       -> "<str 256 chars: 'very long…essage'>"
        [1, 2, 3, 4, 5]                     -> "[5: 1, 2, 3, …+2]"
        42                                  -> "42"
    """
    if v is None or isinstance(v, bool | int | float):
        try:
            return json.dumps(v)
        except (TypeError, ValueError):
            return repr(v)
    if isinstance(v, str):
        if len(v) <= max_str_chars:
            return json.dumps(v)
        head = v[: max(8, max_str_chars // 2)]
        tail = v[-max(4, max_str_chars // 4) :]
        return f"<str {len(v)} chars: {json.dumps(head)[:-1]}…{json.dumps(tail)[1:]}>"
    if isinstance(v, list):
        n = len(v)
        if n == 0:
            return "[]"
        first = v[0]
        if isinstance(first, dict):
            keys = sorted(first.keys())
            keys_part = ", ".join(keys[:6]) + ("…" if len(keys) > 6 else "")
            return f"[{n} of {{{keys_part}}}]"
        if isinstance(first, list):
            return f"[{n} of list]"
        sample = ", ".join(
            _summarize_value_shape(x, max_str_chars=max_str_chars, max_list_items=max_list_items)
            for x in v[:max_list_items]
        )
        more = f", …+{n - max_list_items}" if n > max_list_items else ""
        return f"[{n}: {sample}{more}]"
    if isinstance(v, dict):
        if not v:
            return "{}"
        keys = sorted(v.keys())
        # For small dicts, recurse one level so the analyzer sees the shape
        # of nested data (e.g. ``{flights: [30 of {airline, dep}]}``).  For
        # wide dicts, just list keys.
        if len(keys) <= 6:
            inner_budget = max(40, max_str_chars * 2)
            parts = []
            for k in keys:
                child = _summarize_value_shape(
                    v[k],
                    max_str_chars=max_str_chars,
                    max_list_items=max_list_items,
                )
                if len(child) > inner_budget:
                    child = child[:inner_budget] + "…"
                parts.append(f"{k}: {child}")
            return "{" + ", ".join(parts) + "}"
        keys_part = ", ".join(keys[:8]) + ("…" if len(keys) > 8 else "")
        return "{" + keys_part + "}"
    s = repr(v)
    return s if len(s) <= max_str_chars else s[:max_str_chars] + "…"


def _format_tool_call_compact(tc: dict) -> str:
    """One-line summary of a single tool call (no bulk payload).

    Used for cases beyond ``_full_trace_top_n()`` and for surfacing the
    *individual* failing calls inside an otherwise summarized case.
    """
    name = tc.get("name", "?")
    args = tc.get("args", {})
    if isinstance(args, dict) and args:
        keys = sorted(args.keys())
        args_part = "{" + ", ".join(keys[:6]) + ("…" if len(keys) > 6 else "") + "}"
    elif isinstance(args, dict):
        args_part = "{}"
    else:
        args_part = _summarize_value_shape(args)
    err = tc.get("error")
    if err:
        err_s = str(err)
        if len(err_s) > 160:
            err_s = err_s[:160] + "…"
        return f"{name}({args_part}) → ERROR: {err_s}"
    result_shape = _summarize_value_shape(tc.get("result"))
    return f"{name}({args_part}) → {result_shape}"


def _format_tool_call_full(tc: dict, *, value_chars: int = 240) -> str:
    """Schema-aware "full" tool-call line for the deepest-inspection cases.

    Replaces the previous blunt 200-char chop on JSON-dumped args/result
    with shape-aware truncation that preserves dict keys and array counts
    while still bounding total bytes.
    """
    name = tc.get("name", "?")
    args = tc.get("args", {})
    if isinstance(args, dict) and args:
        per_kv = max(40, value_chars // max(len(args), 1))
        args_inner = ", ".join(
            f"{k}={_summarize_value_shape(v, max_str_chars=per_kv)}" for k, v in sorted(args.items())
        )
        args_str = "{" + args_inner + "}"
    else:
        args_str = _summarize_value_shape(args, max_str_chars=value_chars)
    err = tc.get("error")
    if err:
        err_s = str(err)
        if len(err_s) > value_chars:
            err_s = err_s[:value_chars] + "…"
        return f"{name}({args_str}) → ERROR: {err_s}"
    result_str = _summarize_value_shape(tc.get("result"), max_str_chars=value_chars)
    return f"{name}({args_str}) → {result_str}"


def _format_tool_trace_compact(tool_trace: list[dict]) -> list[str]:
    """Return compact lines summarizing *tool_trace* without bulk payloads.

    Run-length-compresses consecutive same-named calls so a chain of
    e.g. 12 retries of ``search`` renders as ``search×12``.  Always
    surfaces failing calls in full (compact form) so the analyzer can
    see error details even for non-top cases.
    """
    if not tool_trace:
        return []
    compressed: list[str] = []
    prev_name: str | None = None
    run = 0
    for tc in tool_trace:
        n = tc.get("name", "?")
        if n == prev_name:
            run += 1
        else:
            if prev_name is not None:
                compressed.append(f"{prev_name}×{run}" if run > 1 else prev_name)
            prev_name = n
            run = 1
    if prev_name is not None:
        compressed.append(f"{prev_name}×{run}" if run > 1 else prev_name)

    lines: list[str] = [f"  Tool calls ({len(tool_trace)}): " + " → ".join(compressed)]
    err_calls = [tc for tc in tool_trace if tc.get("error")]
    if err_calls:
        lines.append("  Tool errors:")
        for tc in err_calls[:5]:
            lines.append(f"    - {_format_tool_call_compact(tc)}")
        if len(err_calls) > 5:
            lines.append(f"    … +{len(err_calls) - 5} more error calls")
    return lines


def _render_within_budget(
    builder: Callable[[int], str],
    *,
    initial_max_cases: int,
    model: str | None = None,
    budget_chars: int | None = None,
) -> str:
    """Iteratively render a prompt, shrinking ``max_cases`` until it fits.

    ``builder(max_cases)`` must return a fully-assembled prompt string.
    We start at ``initial_max_cases`` and back off through a ladder of
    progressively smaller case counts until the prompt size fits the
    model's context budget (resolved via :func:`_budget_chars_for`).
    The final returned prompt carries a one-line notice if any shrinking
    happened, so the analyzer model knows the input is partial.

    This is a defense-in-depth guard layered on top of the two-tier
    per-case format (compact summaries for all but the worst
    ``_full_trace_top_n()`` cases).  Together they prevent verbose
    datasets — multi-agent tool-heavy systems especially — from busting
    the analyzer model's context window.
    """
    if budget_chars is None:
        budget_chars = _budget_chars_for(model)
    ladder = [initial_max_cases, 12, 8, 5, 3, 2, 1]
    seen: list[int] = []
    last_size = 0
    for n in ladder:
        if n <= 0 or n in seen:
            continue
        seen.append(n)
        prompt = builder(n)
        last_size = len(prompt)
        if os.environ.get("OVERMIND_ANALYZER_DEBUG_PROMPT"):
            import sys as _sys

            _sys.stderr.write(
                f"[overmind/analyzer] _render_within_budget: "
                f"max_cases={n} size={last_size:,} budget={budget_chars:,} "
                f"model={model}\n"
            )
        if last_size <= budget_chars:
            if seen and seen[0] != n:
                notice = (
                    f"\n\n> _NOTE: prompt-size guard reduced max_cases "
                    f"{seen[0]} → {n} (final size {last_size:,} chars / "
                    f"budget {budget_chars:,}) to fit the analyzer model's context._"
                )
                _log.info(
                    "analyzer prompt sized to fit: %s chars at max_cases=%d (budget=%d, model=%s)",
                    f"{last_size:,}",
                    n,
                    budget_chars,
                    model or "<unspecified>",
                )
                return prompt + notice
            _log.debug(
                "analyzer prompt: %s chars at max_cases=%d (budget=%d)",
                f"{last_size:,}",
                n,
                budget_chars,
            )
            return prompt
    # Final attempt: even 1 case overflowed.  Return the smallest with a
    # very explicit warning so the analyzer / caller knows the input is
    # degraded.  The caller still ships the prompt — if the API rejects
    # it, the retry-on-ContextWindowExceededError path will catch the
    # failure and degrade gracefully.
    prompt = builder(1)
    _log.warning(
        "analyzer prompt still %s chars at max_cases=1 (budget %d); model=%s — diagnosis quality may be reduced.",
        f"{len(prompt):,}",
        budget_chars,
        model or "<unspecified>",
    )
    notice = (
        f"\n\n> _WARNING: prompt-size guard could not fit any case fully — "
        f"size {len(prompt):,} chars exceeds budget {budget_chars:,}. "
        f"Diagnose at your own risk._"
    )
    return prompt + notice


def _format_per_case_results(
    case_results: list[dict],
    eval_spec: dict | None,
    *,
    max_cases: int = 20,
    case_fraction: float = 1.0,
    iteration_seed: int = 42,
    full_trace_top_n: int | None = None,
) -> str:
    """Format per-case results for the analyzer.

    Expected values are shown for failing numeric fields so the diagnosis
    can identify scoring patterns (e.g., expected values correlating with
    tool-returned data).  Anti-overfitting rules in the prompt prevent
    hardcoding specific case values.

    Two-tier tool-trace rendering bounds prompt growth for tool-heavy
    agents: the worst ``full_trace_top_n`` cases (default
    :data:`OVERMIND_ANALYZER_FULL_TRACE_TOP_N` or 3) get the full per-call
    detail with schema-aware value truncation; every other case gets a
    compact run-length sequence plus any error calls in full.
    """
    if not case_results:
        return "(no results available)"

    if full_trace_top_n is None:
        full_trace_top_n = _full_trace_top_n()

    sorted_cases = sorted(case_results, key=lambda c: c.get("score", {}).get("total", 0))

    # Partially blind diagnosis: only show a fraction of cases, with a
    # different random subset each iteration to prevent memorization.
    if 0 < case_fraction < 1.0 and len(sorted_cases) > 4:
        n_keep = max(3, int(len(sorted_cases) * case_fraction))
        worst = sorted_cases[:3]
        rest = sorted_cases[3:]
        n_from_rest = max(0, n_keep - 3)
        if n_from_rest < len(rest):
            rest = random.Random(iteration_seed).sample(rest, n_from_rest)
        sorted_cases = worst + sorted(rest, key=lambda c: c.get("score", {}).get("total", 0))

    if len(sorted_cases) > max_cases:
        worst = sorted_cases[: max_cases - 5]
        best = sorted_cases[-5:]
        visible = worst + best
        omitted = len(sorted_cases) - len(visible)
    else:
        visible = sorted_cases
        omitted = 0

    fields = list((eval_spec or {}).get("output_fields", {}).keys())
    struct_max = (eval_spec or {}).get("structure_weight", 20)
    # Tier boundary: first ``full_trace_top_n`` *visible* cases (which are
    # the worst-scoring ones thanks to the sort above) get full tool-call
    # detail; every other case gets a compact summary so prompt size scales
    # sublinearly with the dataset size.
    full_trace_cutoff = min(full_trace_top_n, len(visible))

    lines: list[str] = []
    for i, case in enumerate(visible):
        if omitted and i == len(visible) - 5:
            lines.append(f"... ({omitted} mid-range cases omitted) ...")
            lines.append("")

        score = case.get("score", {})
        total = score.get("total", 0)
        output = case.get("output", {})
        input_data = case.get("input", {})

        if isinstance(input_data, dict):
            input_summary = ", ".join(f"{k}={json.dumps(v)}" for k, v in input_data.items())[:400]
        else:
            input_summary = str(input_data)[:400]

        lines.append(f"**Case {i + 1} \u2014 {total:.0f}/100**")
        lines.append(f"  Input: {input_summary}")

        if not isinstance(output, dict):
            lines.append(f"  Output (text): {str(output)[:200]}")
            continue

        for fname in fields:
            act = output.get(fname, "MISSING")
            fs = score.get(fname, 0)
            cfg = (eval_spec or {}).get("output_fields", {}).get(fname, {})
            mx = cfg.get("weight", 0)
            passed = mx > 0 and fs >= mx * 0.8
            mark = "\u2713" if passed else "\u2717"
            if passed:
                lines.append(f"  [{mark}] {fname}: PASS ({fs:.1f}/{mx})")
            else:
                ftype = cfg.get("type", "unknown")
                if ftype == "enum":
                    valid_vals = cfg.get("values", [])
                    got_str = str(act or "").lower().strip()
                    if got_str in [v.lower() for v in valid_vals]:
                        hint = f"valid but wrong value: {act!r}"
                    elif act in (None, "", "MISSING"):
                        hint = "MISSING"
                    else:
                        hint = f"invalid value: {act!r}"
                    lines.append(f"  [{mark}] {fname}: FAIL — {hint} ({fs:.1f}/{mx})")
                elif ftype == "number":
                    if act in (None, "", "MISSING"):
                        lines.append(f"  [{mark}] {fname}: FAIL — MISSING ({fs:.1f}/{mx})")
                    else:
                        pct = fs / mx * 100 if mx > 0 else 0
                        exp_val = case.get("expected", {}).get(fname)
                        if exp_val is not None:
                            try:
                                diff = float(act) - float(exp_val)
                                sign = "+" if diff >= 0 else ""
                                lines.append(
                                    f"  [{mark}] {fname}: FAIL — "
                                    f"got {act}, expected {exp_val}, "
                                    f"diff={sign}{diff:.0f} "
                                    f"({pct:.0f}% credit, {fs:.1f}/{mx})"
                                )
                            except (ValueError, TypeError):
                                lines.append(
                                    f"  [{mark}] {fname}: FAIL — "
                                    f"got {act!r}, off target ({pct:.0f}% "
                                    f"credit, {fs:.1f}/{mx})"
                                )
                        else:
                            lines.append(
                                f"  [{mark}] {fname}: FAIL — got {act!r}, off target ({pct:.0f}% credit, {fs:.1f}/{mx})"
                            )
                elif ftype == "text":
                    if act and str(act).strip():
                        lines.append(f"  [{mark}] {fname}: FAIL — present but insufficient ({fs:.1f}/{mx})")
                    else:
                        lines.append(f"  [{mark}] {fname}: FAIL — empty/missing ({fs:.1f}/{mx})")
                else:
                    lines.append(f"  [{mark}] {fname}: FAIL — got {act!r} ({fs:.1f}/{mx})")

        struct_score = score.get("structure", 0)
        s_mark = "\u2713" if struct_score >= struct_max * 0.8 else "\u2717"
        lines.append(f"  [{s_mark}] structure: {struct_score:.1f}/{struct_max}")

        tool_trace = case.get("tool_trace", [])
        if tool_trace:
            if i < full_trace_cutoff:
                lines.append("  Tool calls (full):")
                for t_idx, tc in enumerate(tool_trace, 1):
                    lines.append(f"    {t_idx}. {_format_tool_call_full(tc)}")
            else:
                lines.extend(_format_tool_trace_compact(tool_trace))
        elif case.get("tool_calls"):
            lines.append(f"  Tools used: {', '.join(case['tool_calls'])}")

        lines.append("")

    return "\n".join(lines)


# Per-arg-value bound used by the aggregated "argument value distribution"
# section.  Without a bound, agents that pass large context objects (e.g.
# multi-agent framework ``RunContextWrapper`` instances, tens of kilobytes
# each, in *every* tool call) blow this section to multiple MB and bust
# the analyzer model's context window.  Configurable via
# ``OVERMIND_ANALYZER_ARG_VALUE_MAX_CHARS``.
_ARG_VALUE_MAX_CHARS_DEFAULT = 80
# Cap on the per-(tool, param) line in the aggregation output.
_ARG_DISTRIBUTION_LINE_MAX_CHARS_DEFAULT = 400


def _arg_value_max_chars() -> int:
    try:
        return max(16, int(os.environ.get("OVERMIND_ANALYZER_ARG_VALUE_MAX_CHARS", _ARG_VALUE_MAX_CHARS_DEFAULT)))
    except (TypeError, ValueError):
        return _ARG_VALUE_MAX_CHARS_DEFAULT


def _arg_distribution_line_max_chars() -> int:
    try:
        return max(
            80,
            int(
                os.environ.get(
                    "OVERMIND_ANALYZER_ARG_DISTRIBUTION_LINE_MAX_CHARS", _ARG_DISTRIBUTION_LINE_MAX_CHARS_DEFAULT
                )
            ),
        )
    except (TypeError, ValueError):
        return _ARG_DISTRIBUTION_LINE_MAX_CHARS_DEFAULT


def _stringify_arg_value(v: Any, max_chars: int) -> str:
    """Compact, bounded textual representation of an arg value for aggregation.

    We delegate the shape work to :func:`_summarize_value_shape` (which
    already preserves keys / counts and truncates long strings) and then
    enforce a hard char cap so a single misbehaving value (e.g. a giant
    framework wrapper printed via ``__repr__``) cannot blow the budget.
    """
    s = _summarize_value_shape(v, max_str_chars=max_chars, max_list_items=3)
    if len(s) > max_chars:
        s = s[: max_chars - 1] + "…"
    return s


def _format_tool_usage_analysis(case_results: list[dict]) -> str:
    """Aggregate tool usage patterns across all cases.

    The argument-value distribution section is value-bounded (each value
    stringified with :func:`_stringify_arg_value`, lines truncated at
    :func:`_arg_distribution_line_max_chars`) so a single tool that passes
    multi-kilobyte context objects cannot dominate the analyzer prompt.
    """
    if not case_results:
        return "(no tool data)"

    max_val_chars = _arg_value_max_chars()
    max_line_chars = _arg_distribution_line_max_chars()

    tool_calls_count: dict[str, int] = {}
    arg_values: dict[str, dict[str, set[str]]] = {}
    missing_tools: dict[str, int] = {}
    errors: list[str] = []
    total_cases = len(case_results)

    all_tool_names: set[str] = set()
    for case in case_results:
        trace = case.get("tool_trace", [])
        for tc in trace:
            name = tc.get("name", "")
            all_tool_names.add(name)
            tool_calls_count[name] = tool_calls_count.get(name, 0) + 1
            args = tc.get("args", {})
            if isinstance(args, dict):
                for param, val in args.items():
                    # Store bounded representations as we go so the
                    # aggregation set itself stays small.  Cap distinct
                    # values per (tool, param) at 32 to avoid pathological
                    # high-cardinality params (timestamps, UUIDs, …)
                    # filling memory before truncation.
                    bucket = arg_values.setdefault(name, {}).setdefault(param, set())
                    if len(bucket) < 32:
                        bucket.add(_stringify_arg_value(val, max_val_chars))
            if tc.get("error"):
                err_msg = str(tc["error"])
                if len(err_msg) > 200:
                    err_msg = err_msg[:200] + "…"
                errors.append(f"{name}: {err_msg}")

    for case in case_results:
        called = {tc.get("name") for tc in case.get("tool_trace", [])}
        for tool_name in all_tool_names:
            if tool_name not in called:
                missing_tools[tool_name] = missing_tools.get(tool_name, 0) + 1

    lines: list[str] = []
    lines.append(f"**Tool call frequency** (across {total_cases} cases):")
    for name, count in sorted(tool_calls_count.items(), key=lambda x: -x[1]):
        skip_count = missing_tools.get(name, 0)
        skip_note = f" (skipped in {skip_count} cases)" if skip_count else ""
        lines.append(f"  - {name}: called {count} times{skip_note}")

    lines.append("")
    lines.append("**Argument value distribution** (values shape-summarized, line-capped):")
    for tool_name, params in arg_values.items():
        for param, unique in params.items():
            if len(unique) <= 10:
                rendered = f"{sorted(unique)}"
            else:
                sample = sorted(unique)[:5]
                rendered = f"{len(unique)} unique values (sample: {sample})"
            line = f"  - {tool_name}.{param}: {rendered}"
            if len(line) > max_line_chars:
                line = line[: max_line_chars - 1] + "…"
            lines.append(line)

    if errors:
        lines.append("")
        lines.append("**Tool errors:**")
        for err in errors[:10]:
            lines.append(f"  - {err}")
        if len(errors) > 10:
            lines.append(f"  - … +{len(errors) - 10} more error calls")

    return "\n".join(lines) or "(no tool data)"


def _format_score_breakdown(evaluation: dict, eval_spec: dict | None) -> str:
    lines: list[str] = []
    for key, val in evaluation.items():
        if key.startswith("avg_") and key != "avg_total" and isinstance(val, (int, float)):
            nice = key.replace("avg_", "").replace("_", " ").title()
            field_key = key.replace("avg_", "")
            max_val = 0.0
            if eval_spec:
                if field_key == "structure":
                    max_val = float(eval_spec.get("structure_weight", 20))
                elif field_key in eval_spec.get("output_fields", {}):
                    max_val = float(eval_spec["output_fields"][field_key].get("weight", 0))
                elif field_key == "tool_usage":
                    max_val = float(eval_spec.get("tool_usage_weight", 0))
                elif field_key == "llm_judge":
                    max_val = float(eval_spec.get("llm_judge_weight", 0))
            pct = f" ({val / max_val * 100:.0f}%)" if max_val else ""
            mx_str = f" / {max_val:.0f}" if max_val else ""
            lines.append(f"  {nice}: {val:.1f}{mx_str}{pct}")
    return "\n".join(lines) or "  (no breakdown)"


def _find_weakest_dimension(evaluation: dict, eval_spec: dict | None) -> tuple[str, float, float]:
    """Return (display_name, avg_score, max_score) of the worst-relative dimension.

    Dimensions that the spec configures but the evaluator did not score
    for this run (key ``avg_<dim>`` absent from *evaluation*) are
    skipped, not defaulted to ``0``. Treating an absent dimension as
    ``0/max`` would give it a gap of ``1.0`` and force it to always win
    the "weakest dimension" selection, which then pins every subsequent
    iteration's focus to that dimension regardless of the agent's real
    failure surface. The evaluator emits a one-time warning when this
    happens so users still know the dimension isn't being measured.
    """
    if not eval_spec:
        return ("unknown", 0.0, 0.0)

    worst_name = "Structure"
    worst_gap = 0.0
    worst_score = evaluation.get("avg_structure", 0.0)
    worst_max = float(eval_spec.get("structure_weight", 20))

    if worst_max > 0:
        worst_gap = 1 - (worst_score / worst_max)

    for field_name, config in eval_spec.get("output_fields", {}).items():
        max_val = float(config.get("weight", 0))
        if max_val <= 0:
            continue
        avg_key = f"avg_{field_name}"
        if avg_key not in evaluation:
            continue
        avg_val = float(evaluation[avg_key])
        gap = 1 - (avg_val / max_val)
        if gap > worst_gap:
            worst_gap = gap
            worst_name = field_name.replace("_", " ").title()
            worst_score = avg_val
            worst_max = max_val

    # Also check tool_usage and llm_judge
    for dim_key, spec_key in [
        ("tool_usage", "tool_usage_weight"),
        ("llm_judge", "llm_judge_weight"),
    ]:
        max_val = float(eval_spec.get(spec_key, 0))
        if max_val <= 0:
            continue
        avg_key = f"avg_{dim_key}"
        if avg_key not in evaluation:
            continue
        avg_val = float(evaluation[avg_key])
        gap = 1 - (avg_val / max_val)
        if gap > worst_gap:
            worst_gap = gap
            worst_name = dim_key.replace("_", " ").title()
            worst_score = avg_val
            worst_max = max_val

    return worst_name, worst_score, worst_max


def _format_fixed_elements(eval_spec: dict | None) -> str:
    if not eval_spec or not eval_spec.get("fixed_elements"):
        return "- Tool implementation functions and their logic"
    return "\n".join(f"- {e}" for e in eval_spec["fixed_elements"])


def _format_optimizable_elements(eval_spec: dict | None) -> str:
    if not eval_spec or not eval_spec.get("optimizable_elements"):
        return "- Prompts, tool descriptions, agent logic"
    return "\n".join(f"- {e}" for e in eval_spec["optimizable_elements"])


def _format_dimension_deltas(deltas: dict[str, float]) -> str:
    """Format dimension deltas into a compact gains/losses summary."""
    if not deltas:
        return ""
    gains = [f"{k} +{v:.1f}" for k, v in deltas.items() if v > 0]
    losses = [f"{k} {v:.1f}" for k, v in deltas.items() if v < 0]
    parts: list[str] = []
    if gains:
        parts.append("Gains: " + ", ".join(gains))
    if losses:
        parts.append("Losses: " + ", ".join(losses))
    return " | ".join(parts)


def _format_failed_attempts(failed: list[dict] | None, max_entries: int = 8) -> str:
    if not failed:
        return "(none yet)"
    recent = failed[-max_entries:]
    lines: list[str] = []

    all_suggestions = []
    for att in recent:
        all_suggestions.extend(att.get("suggestions", []))
    sugg_lower = [s.lower() for s in all_suggestions]

    repeated_patterns: list[str] = []
    for keyword, label in [
        ("recomput", "deterministic recomputation/override of LLM output"),
        ("post-process", "post-processing overrides"),
        ("_recompute", "helper function to recompute values"),
        ("overwrite", "overwriting LLM output with formulas"),
        ("unconditionally set", "unconditionally overriding LLM fields"),
    ]:
        count = sum(1 for s in sugg_lower if keyword in s)
        if count >= 2:
            repeated_patterns.append(f"'{label}' attempted {count}x and FAILED every time")

    if repeated_patterns:
        lines.append("⚠️  REPEATEDLY FAILED APPROACHES (try something fundamentally different):")
        for rp in repeated_patterns:
            lines.append(f"  - {rp}")
        lines.append(
            "Do NOT propose variations of the above approaches. "
            "They consistently make things worse. Try a completely "
            "different strategy (e.g., improving system prompt instructions, "
            "restructuring input formatting, improving tool descriptions)."
        )
        lines.append("")

    for i, att in enumerate(recent, 1):
        reason = att.get("reason", "no improvement")
        score = att.get("score", 0)
        lines.append(f"Attempt {i} (score: {score:.1f}, {reason}):")
        for s in att.get("suggestions", []):
            lines.append(f"  - {s}")
        delta_str = _format_dimension_deltas(att.get("dimension_deltas", {}))
        if delta_str:
            lines.append(f"  Dimensions: {delta_str}")
    return "\n".join(lines)


def _format_successful_changes(succ: list[dict] | None, max_entries: int = 8) -> str:
    if not succ:
        return "(none yet)"
    recent = succ[-max_entries:]
    lines: list[str] = []
    for i, ch in enumerate(recent, 1):
        lines.append(f"Round {i} ({ch.get('improvement', '')}):")
        for s in ch.get("suggestions", []):
            lines.append(f"  - {s}")
        delta_str = _format_dimension_deltas(ch.get("dimension_deltas", {}))
        if delta_str:
            lines.append(f"  Dimensions: {delta_str}")
    return "\n".join(lines)


def _detect_agent_model(code: str) -> tuple[str, str]:
    m = re.search(r"""(?:MODEL|model)\s*[:=]\s*["']([^"']+)["']""", code)
    name = m.group(1) if m else "unknown"
    if any(x in name.lower() for x in ["mini", "nano", "small", "haiku", "flash"]):
        return name, "lightweight"
    if any(x in name.lower() for x in ["pro", "opus"]):
        return name, "very capable"
    return name, "capable"


def _extract_code_and_analysis(
    text: str,
    agent_code: str = "",
) -> tuple[str, list[str], str | None]:
    """Parse the model response into (analysis, suggestions, code | None)."""
    analysis = ""
    suggestions: list[str] = []

    fingerprints = _build_fingerprints(agent_code)

    json_m = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if json_m:
        parsed = parse_json_object(json_m.group(1).strip(), on_fail="default", default=None)
        if isinstance(parsed, dict):
            analysis = parsed.get("analysis", parsed.get("root_cause", ""))
            suggestions = parsed.get(
                "suggestions",
                [c.get("action", "") for c in parsed.get("changes", [])],
            )

    if not analysis:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start : end + 1]
            # Skip candidates that look like agent code or are excessively long —
            # they'd otherwise blow up parsing or smuggle code into the analysis.
            if not _matches_fingerprint(candidate, fingerprints) and len(candidate) < 3000:
                parsed = parse_json_object(candidate, on_fail="default", default=None)
                if isinstance(parsed, dict):
                    analysis = parsed.get("analysis", parsed.get("root_cause", ""))
                    suggestions = parsed.get(
                        "suggestions",
                        [c.get("action", "") for c in parsed.get("changes", [])],
                    )

    code: str | None = None

    all_blocks: list[str] = []
    for m in re.finditer(r"```[a-zA-Z]*\s*\n(.*?)```", text, re.DOTALL):
        block = m.group(1).strip()
        if json_m and m.start() == json_m.start():
            continue
        all_blocks.append(block)

    best_len = 0
    for block in all_blocks:
        if _matches_fingerprint(block, fingerprints) and len(block) > best_len:
            code = block
            best_len = len(block)

    if not code:
        for fm in reversed(list(re.finditer(r"```[a-zA-Z]*\s*\n", text))):
            if json_m and fm.start() == json_m.start():
                continue
            after = text[fm.end() :]
            if "```" in after:
                after = after[: after.rfind("```")]
            candidate = after.strip()
            if _matches_fingerprint(candidate, fingerprints):
                code = candidate
                break

    if not code:
        search_from = text
        if json_m:
            search_from = text[json_m.end() :]
        search_from = re.sub(r"^```\s*\n?", "", search_from.strip()).strip()
        if _matches_fingerprint(search_from, fingerprints):
            code = search_from

    return analysis, suggestions, code


def _build_fingerprints(agent_code: str) -> list[str]:
    if not agent_code:
        return []
    fps: list[str] = []
    for pattern in [
        r"((?:def|func|function|export\s+(?:async\s+)?function)\s+\w+\s*\([^)]*\))",
        r"(run\s*[:=])",
    ]:
        m = re.search(pattern, agent_code)
        if m:
            fps.append(m.group(1).split("(")[0].strip())
            break
    if not fps:
        fps.append("run")
    return fps


def _matches_fingerprint(text: str, fingerprints: list[str]) -> bool:
    if not fingerprints:
        return len(text) > 100
    return all(fp in text for fp in fingerprints)


# ---------------------------------------------------------------------------
# Multi-file (bundle) output parsing
# ---------------------------------------------------------------------------


def _parse_file_updates(
    text: str,
) -> tuple[str, list[str], dict[str, str]]:
    """Parse an LLM response that uses the whole-file ``### FILE:`` format.

    Returns ``(analysis, suggestions, file_updates)`` where
    *file_updates* maps ``relative_path → complete_new_source``.
    """
    analysis = ""
    suggestions: list[str] = []

    json_m = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if json_m:
        parsed = parse_json_object(json_m.group(1).strip(), on_fail="default", default=None)
        if isinstance(parsed, dict):
            analysis = parsed.get("analysis", parsed.get("root_cause", ""))
            suggestions = parsed.get(
                "suggestions",
                [c.get("action", "") for c in parsed.get("changes", [])],
            )

    file_updates: dict[str, str] = {}

    # Primary pattern: ### FILE: path/to/file.py
    file_pattern = r"###\s*FILE:\s*(\S+)\s*\n```[a-zA-Z]*\s*\n(.*?)```"
    for m in re.finditer(file_pattern, text, re.DOTALL):
        file_path = m.group(1).strip()
        code = m.group(2).strip()
        if code:
            file_updates[file_path] = code

    # Fallback: # ===== FILE: path [TAG] ===== followed by code fence
    if not file_updates:
        fallback_pattern = (
            r"#\s*=+\s*FILE:\s*(\S+)\s*\[.*?\]\s*=+\s*\n"
            r"```[a-zA-Z]*\s*\n(.*?)```"
        )
        for m in re.finditer(fallback_pattern, text, re.DOTALL):
            file_path = m.group(1).strip()
            code = m.group(2).strip()
            if code:
                file_updates[file_path] = code

    return analysis, suggestions, file_updates


def _build_agent_code_section(
    agent_code: str,
    bundle: AgentBundle | None = None,
) -> str:
    """Build the ``{agent_code_section}`` content for prompts.

    When *bundle* is provided, renders the full virtual bundle with
    positional piece IDs.  Otherwise wraps *agent_code* in a simple
    code fence (backward compatibility).
    """
    if bundle is not None:
        return bundle.to_prompt_text()
    return f"```\n{agent_code}\n```"


def _get_output_format_instruction(bundle: AgentBundle | None = None) -> str:
    """Return the appropriate output format instruction."""
    if bundle is not None and bundle.is_multi_file():
        return _BUNDLE_OUTPUT_INSTRUCTION
    return _SINGLE_FILE_OUTPUT_INSTRUCTION


def _get_entry_file(
    agent_code: str,
    bundle: AgentBundle | None = None,
) -> str:
    """Return the entry file path for prompt injection."""
    if bundle is not None:
        return bundle.entry_file
    return "the agent module"


# ---------------------------------------------------------------------------
# Two-pass generation
# ---------------------------------------------------------------------------


@observe_safe(span_name="diagnosis", type=SpanType.FUNCTION)
def _run_diagnosis(
    *args: Any,
    ctx: DiagnosisContext | None = None,
    focus_area: str | None = None,
    case_fraction: float = 1.0,
    max_cases: int = 20,
    **kwargs: Any,
) -> dict | None:
    """Pass 1: produce a structured diagnosis from the per-case results.

    The function accepts two equivalent call shapes:

    * A pre-built :class:`DiagnosisContext` (preferred for new code)::

          _run_diagnosis(ctx=DiagnosisContext(...), focus_area="tool_description")

    * Legacy keyword arguments that the rest of the codebase / tests use::

          _run_diagnosis(agent_code=..., case_results=..., ..., entrypoint_fn="run")

    The legacy keyword form is folded into a :class:`DiagnosisContext`
    internally so there's exactly one execution path below.
    """
    if ctx is None:
        ctx = _build_diagnosis_context_from_kwargs(args, kwargs)

    agent_model, capability = _detect_agent_model(ctx.agent_code)
    weak_name, weak_score, weak_max = _find_weakest_dimension(ctx.evaluation_results, ctx.eval_spec)

    mcr = (
        "You MAY suggest changing the MODEL constant."
        if ctx.allow_model_change
        else "Do NOT suggest changing the MODEL constant."
    )

    prompt_chars, prompt_lines = _measure_system_prompt(ctx.agent_code)

    def _build_prompt(_max_cases: int) -> str:
        ac_section = _build_agent_code_section(ctx.agent_code, ctx.bundle)
        per_case = _format_per_case_results(
            ctx.case_results,
            ctx.eval_spec,
            max_cases=_max_cases,
            case_fraction=case_fraction,
            iteration_seed=ctx.iteration_seed,
        )
        tu_section = _format_tool_usage_analysis(ctx.case_results)
        if os.environ.get("OVERMIND_ANALYZER_DEBUG_PROMPT"):
            import sys as _sys

            _sys.stderr.write(
                f"[overmind/analyzer] sections @max_cases={_max_cases}: "
                f"agent_code={len(ac_section):,} "
                f"per_case={len(per_case):,} "
                f"tool_usage={len(tu_section):,}\n"
            )
        p = DIAGNOSIS_PROMPT.format(
            agent_code_section=ac_section,
            entry_file=_get_entry_file(ctx.agent_code, ctx.bundle),
            entrypoint_fn=ctx.entrypoint_fn,
            scoring_mechanics=_format_scoring_mechanics(ctx.eval_spec),
            per_case_results=per_case,
            tool_usage_analysis=tu_section,
            policy_context=ctx.policy_context or "(no policy defined)",
            avg_score=ctx.evaluation_results.get("avg_total", 0),
            weakest_dimension=weak_name,
            weakest_dim_score=weak_score,
            weakest_dim_max=weak_max,
            score_breakdown=_format_score_breakdown(ctx.evaluation_results, ctx.eval_spec),
            successful_changes=_format_successful_changes(ctx.successful_changes),
            failed_attempts=_format_failed_attempts(ctx.failed_attempts),
            model_change_rule=mcr,
            agent_model=agent_model,
            model_capability=capability,
            prompt_char_count=prompt_chars,
            prompt_line_count=prompt_lines,
        )
        if ctx.bundle is not None and ctx.bundle.is_multi_file():
            p += MULTI_FILE_AWARENESS_SECTION
        if ctx.cluster_context:
            p += FAILURE_CLUSTERS_SECTION.format(formatted_clusters=ctx.cluster_context)
        if ctx.component_weights_context:
            p += COMPONENT_IMPACT_SECTION.format(component_lines=ctx.component_weights_context)
        if focus_area:
            labels = {k: v.format(entrypoint_fn=ctx.entrypoint_fn) if "{" in v else v for k, v in FOCUS_LABELS.items()}
            focus_desc = labels.get(focus_area, focus_area)
            p += DIAGNOSIS_FOCUS_DIRECTIVE.format(
                focus_area=focus_area,
                focus_desc=focus_desc,
            )
        return p

    system_msg = DIAGNOSIS_SYSTEM_PROMPT.format(
        scoring_mechanics=_format_scoring_mechanics(ctx.eval_spec),
        optimizable_elements=_format_optimizable_elements(ctx.eval_spec),
        fixed_elements=_format_fixed_elements(ctx.eval_spec),
    )

    # Retry-on-context-window-exceeded: even after the proactive budget
    # guard, the model's actual tokenizer can disagree with our
    # chars/token estimate (Anthropic's tokenizer treats code denser
    # than English).  Halve the char budget on overflow and re-render
    # with fewer cases; capped at 3 retries to bound latency.
    budget_chars = _budget_chars_for(ctx.model)
    last_exc: Exception | None = None
    for attempt in range(4):
        prompt = _render_within_budget(
            _build_prompt,
            initial_max_cases=max_cases,
            model=ctx.model,
            budget_chars=budget_chars,
        )
        try:
            resp = llm_completion(
                ctx.model,
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=max(ctx.temperature * 0.5, 0.1),
                max_tokens=4000,
            )
            content = resp.choices[0].message.content or ""
            parsed = parse_json_object(content, on_fail="default", default=None)
            if parsed is not None:
                return parsed
            last_exc = RuntimeError("Diagnosis response contained no JSON object.")
            break
        except Exception as exc:
            last_exc = exc
            exc_name = type(exc).__name__
            msg = str(exc).lower()
            is_ctx_overflow = (
                "contextwindowexceeded" in exc_name.lower()
                or "context_length" in msg
                or "prompt is too long" in msg
                or "maximum context length" in msg
            )
            if is_ctx_overflow and attempt < 3:
                new_budget = max(20_000, int(budget_chars * 0.5))
                _log.warning(
                    "analyzer prompt rejected by model (attempt %d, model=%s): "
                    "%s — halving char budget %d → %d and retrying.",
                    attempt + 1,
                    ctx.model,
                    str(exc)[:200],
                    budget_chars,
                    new_budget,
                )
                budget_chars = new_budget
                continue
            break

    if last_exc is not None:
        global _LAST_DIAGNOSIS_ERROR
        _LAST_DIAGNOSIS_ERROR = f"{type(last_exc).__name__}: {last_exc}"
        _log.warning(
            "diagnosis LLM call failed model=%s focus=%s error=%s: %s",
            ctx.model,
            focus_area,
            type(last_exc).__name__,
            str(last_exc)[:300],
        )
    return None


# Legacy positional-parameter names, in declaration order, used to fold the
# pre-dataclass kwargs surface into a :class:`DiagnosisContext`.  Kept in
# sync with the long-form signature documented in the test suite.
_LEGACY_DIAGNOSIS_POSITIONAL = (
    "agent_code",
    "case_results",
    "evaluation_results",
    "model",
    "eval_spec",
    "failed_attempts",
    "successful_changes",
    "allow_model_change",
    "temperature",
)


def _build_diagnosis_context_from_kwargs(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> DiagnosisContext:
    """Fold the legacy keyword surface of ``_run_diagnosis`` into a context.

    Accepts both positional and keyword arguments in the historical order
    so test callers and the optimizer's existing call sites keep working
    while the body uses a single :class:`DiagnosisContext`.
    """
    merged: dict[str, Any] = dict(zip(_LEGACY_DIAGNOSIS_POSITIONAL, args, strict=False))
    merged.update(kwargs)

    return DiagnosisContext(
        agent_code=merged["agent_code"],
        case_results=merged["case_results"],
        evaluation_results=merged["evaluation_results"],
        model=merged["model"],
        entrypoint_fn=merged["entrypoint_fn"],
        eval_spec=merged.get("eval_spec"),
        failed_attempts=merged.get("failed_attempts"),
        successful_changes=merged.get("successful_changes"),
        allow_model_change=merged.get("allow_model_change", False),
        temperature=merged.get("temperature", 0.7),
        iteration_seed=merged.get("iteration_seed", 42),
        policy_context=merged.get("policy_context", ""),
        bundle=merged.get("bundle"),
        cluster_context=merged.get("cluster_context", ""),
        component_weights_context=merged.get("component_weights_context", ""),
    )


_LAST_DIAGNOSIS_ERROR: str | None = None


def get_last_diagnosis_error() -> str | None:
    """Return the most recent diagnosis-LLM exception, or ``None``."""
    return _LAST_DIAGNOSIS_ERROR


def reset_last_diagnosis_error() -> None:
    """Clear the cached diagnosis error before a fresh ``run_diagnose_phase``."""
    global _LAST_DIAGNOSIS_ERROR
    _LAST_DIAGNOSIS_ERROR = None


@observe_safe(span_name="single_shot_codegen", type=SpanType.FUNCTION)
def _run_codegen(
    agent_code: str,
    diagnosis: dict,
    model: str,
    eval_spec: dict | None,
    temperature: float,
    policy_constraints: str = "",
    *,
    entrypoint_fn: str,
    focus_area: str | None = None,
    bundle: AgentBundle | None = None,
) -> str | dict | None:
    """Pass 2: Generate updated code from a diagnosis.

    When *focus_area* is set, the codegen is steered to prioritize changes
    targeting that element while still applying the full diagnosis.

    Returns
    -------
    str
        Complete file code (single-file mode).
    dict
        ``{"file_updates": {rel_path: complete_new_source}}`` when
        operating in bundle mode.
    None
        On failure.
    """
    focus_directive = ""
    if focus_area:
        labels = {k: v.format(entrypoint_fn=entrypoint_fn) if "{" in v else v for k, v in FOCUS_LABELS.items()}
        focus_desc = labels.get(focus_area, focus_area)
        focus_directive = CODEGEN_FOCUS_DIRECTIVE.format(
            focus_area=focus_area,
            focus_desc=focus_desc,
        )

    agent_tokens = len(agent_code) // 3
    codegen_max_tokens = max(4000, min(16000, int(agent_tokens * 2.0)))

    use_bundle = bundle is not None and bundle.is_multi_file()

    prompt = (
        CODEGEN_PROMPT.format(
            agent_code_section=_build_agent_code_section(agent_code, bundle),
            entry_file=_get_entry_file(agent_code, bundle),
            entrypoint_fn=entrypoint_fn,
            diagnosis_json=json.dumps(diagnosis, indent=2),
            optimizable_elements=_format_optimizable_elements(eval_spec),
            fixed_elements=_format_fixed_elements(eval_spec),
            policy_constraints=policy_constraints or "(none)",
            output_format_instruction=_get_output_format_instruction(bundle),
        )
        + focus_directive
    )

    try:
        resp = llm_completion(
            model,
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=codegen_max_tokens,
        )
        content = resp.choices[0].message.content or ""

        if use_bundle:
            _, _, file_updates = _parse_file_updates(content)
            if file_updates:
                return {"file_updates": file_updates}
            # Last resort: single-file extraction (the LLM ignored the
            # whole-file bundle instruction and returned a single code
            # fence — accept it for the entry file).
            _, _, code = _extract_code_and_analysis(content, agent_code)
            return code

        _, _, code = _extract_code_and_analysis(content, agent_code)
        return code
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Agentic codegen (coding-agent-based code generation)
# ---------------------------------------------------------------------------


def _extract_imports_from_source(source: str, known_files: set[str]) -> list[str]:
    """Extract imports from *source* that reference files in *known_files*.

    Supports both Python (AST-based) and JS/TS (regex-based) sources.
    The language is auto-detected: if AST parsing succeeds, Python path
    is used; otherwise the JS regex path runs as a fallback.
    """
    stems: dict[str, str] = {}
    for kf in known_files:
        parts = kf.replace("/", ".").replace("\\", ".")
        for ext in (".py", ".js", ".ts", ".mjs", ".mts"):
            if parts.endswith(ext):
                parts = parts[: -len(ext)]
                break
        for segment in parts.split("."):
            stems[segment] = kf

    imported: list[str] = []

    # Try Python AST first
    try:
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for seg in alias.name.split("."):
                        if seg in stems and stems[seg] not in imported:
                            imported.append(stems[seg])
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for seg in module.split("."):
                    if seg in stems and stems[seg] not in imported:
                        imported.append(stems[seg])
                for alias in node.names:
                    if alias.name in stems and stems[alias.name] not in imported:
                        imported.append(stems[alias.name])
        return imported
    except SyntaxError:
        pass

    # Fallback: JS/TS regex-based import extraction
    for m in re.finditer(
        r"""(?:import\s+.*?\s+from\s+|require\s*\(\s*)['"]([^'"]+)['"]""",
        source,
    ):
        mod_path = m.group(1)
        if not mod_path.startswith("."):
            continue
        clean = mod_path.lstrip("./").replace("/", ".")
        for ext in (".js", ".ts", ".mjs", ".mts"):
            if clean.endswith(ext):
                clean = clean[: -len(ext)]
                break
        for seg in clean.split("."):
            if seg in stems and stems[seg] not in imported:
                imported.append(stems[seg])

    return imported


def _build_import_graph(agent_files: dict[str, str]) -> str:
    """Build a human-readable import graph for the agent files."""
    known = set(agent_files.keys())
    lines: list[str] = []
    for rel, src in agent_files.items():
        imports = _extract_imports_from_source(src, known - {rel})
        if imports:
            lines.append(f"- `{rel}` imports from: {', '.join(f'`{i}`' for i in imports)}")
    return "\n".join(lines) if lines else "(single file — no cross-file imports)"


def _build_agentic_instruction(
    diagnosis: dict,
    eval_spec: dict | None,
    policy_constraints: str,
    entrypoint_fn: str,
    entry_file: str,
    agent_files: dict[str, str],
    focus_area: str | None = None,
    *,
    optimizable_files: set[str] | None = None,
) -> str:
    """Build the user instruction for the coding agent from a diagnosis."""
    if optimizable_files is not None:
        file_listing = "\n".join(
            f"- `{rel}` ({len(src.splitlines())} lines) "
            f"{'[OPTIMIZABLE]' if rel in optimizable_files else '[READ-ONLY]'}"
            for rel, src in agent_files.items()
        )
    else:
        file_listing = "\n".join(f"- `{rel}` ({len(src.splitlines())} lines)" for rel, src in agent_files.items())

    import_graph = _build_import_graph(agent_files)

    policy_section = f"- Policy constraints: {policy_constraints}" if policy_constraints else ""

    focus_directive = ""
    if focus_area:
        labels = {k: v.format(entrypoint_fn=entrypoint_fn) if "{" in v else v for k, v in FOCUS_LABELS.items()}
        focus_desc = labels.get(focus_area, focus_area)
        focus_directive = AGENTIC_CODEGEN_FOCUS.format(
            focus_area=focus_area,
            focus_desc=focus_desc,
        )

    return AGENTIC_CODEGEN_INSTRUCTION.format(
        diagnosis_json=json.dumps(diagnosis, indent=2),
        entrypoint_fn=entrypoint_fn,
        policy_constraints_section=policy_section,
        entry_file=entry_file,
        file_listing=file_listing,
        import_graph=import_graph,
        focus_directive=focus_directive,
    )


def _run_codegen_agentic(
    agent_files: dict[str, str],
    diagnosis: dict,
    model: str,
    eval_spec: dict | None = None,
    policy_constraints: str = "",
    *,
    entrypoint_fn: str,
    entry_file: str,
    focus_area: str | None = None,
    max_steps: int = 50,
    optimizable_files: set[str] | None = None,
) -> dict:
    """Run the coding agent to generate one candidate.

    Returns a candidate dict compatible with the optimizer's expectations.
    """
    from overmind.coding_agent import apply_code_changes

    instruction = _build_agentic_instruction(
        diagnosis,
        eval_spec,
        policy_constraints,
        entrypoint_fn,
        entry_file,
        agent_files,
        focus_area=focus_area,
        optimizable_files=optimizable_files,
    )

    try:
        result = apply_code_changes(
            agent_files=agent_files,
            instruction=instruction,
            model=model,
            entry_file=entry_file,
            max_steps=max_steps,
        )
    except Exception as exc:
        _log.warning(f"Agentic codegen failed: {exc}")
        return {
            "analysis": f"Coding agent error: {exc}",
            "suggestions": [],
            "updated_code": None,
            "method": "agentic_error",
            "_debug": {"error": str(exc)},
        }

    suggestions = [c.get("action", "") for c in diagnosis.get("changes", [])]

    if not result.file_updates:
        entry_source = agent_files.get(entry_file)
        return {
            "analysis": diagnosis.get("root_cause", ""),
            "suggestions": suggestions,
            "updated_code": entry_source,
            "method": "agentic_no_changes",
            "diagnosis": diagnosis,
            "_debug": {
                "steps": result.steps_taken,
                "usage": result.usage,
            },
        }

    entry_source = result.file_updates.get(entry_file, agent_files.get(entry_file))

    is_multi_file = len(agent_files) > 1

    return {
        "analysis": diagnosis.get("root_cause", ""),
        "suggestions": suggestions,
        "updated_code": entry_source,
        "bundle_updates": ({"file_updates": result.file_updates} if is_multi_file else None),
        "_resolved_files": result.file_updates if is_multi_file else None,
        "method": f"agentic({focus_area or 'general'})",
        "diagnosis": diagnosis,
        "_debug": {
            "steps": result.steps_taken,
            "usage": result.usage,
            "files_updated": len(result.file_updates),
        },
    }


# ---------------------------------------------------------------------------
# Automated focus targeting
# ---------------------------------------------------------------------------


def _extract_focus_from_method(method: str) -> str | None:
    """Extract the focus area name from a candidate's method string."""
    for focus in (
        "tool_description",
        "agent_logic",
        "format_input",
        "system_prompt",
        "tool_implementation",
        "helper_module",
        "error_handling",
    ):
        if focus in method:
            return focus
    return None


_ALL_FOCUS_AREAS = (
    "tool_description",
    "agent_logic",
    "format_input",
    "system_prompt",
    "tool_implementation",
    "helper_module",
    "error_handling",
)


def compute_focus_weights(
    case_results: list[dict],
    evaluation_results: dict,
    eval_spec: dict | None = None,
    failure_registry: FailureRegistry | None = None,
    successful_changes: list[dict] | None = None,
    failed_attempts: list[dict] | None = None,
    *,
    is_multi_file: bool = False,
) -> dict[str, float]:
    """Score each focus area 0-1 based on multi-signal failure analysis.

    Signals:
    1. Tool trace errors → tool_description + tool_implementation
    2. Field-specific failures → agent_logic / format_input / helper_module
    3. Runtime errors / crashes → error_handling
    4. Historical effectiveness → boost what worked, dampen what didn't
    5. Failure cluster mechanisms (when available)
    """
    weights: dict[str, float] = {k: 0.0 for k in _ALL_FOCUS_AREAS}

    if not case_results:
        return weights

    n_cases = max(len(case_results), 1)

    # Signal 1: Tool trace errors → tool_description + tool_implementation
    tool_errors = 0
    missing_tools = 0
    for case in case_results:
        trace = case.get("tool_trace", [])
        for t in trace:
            if t.get("error"):
                tool_errors += 1
        expected_tools = (eval_spec or {}).get("tool_config", {}).get("expected_tools", [])
        called = {t.get("name") for t in trace}
        for et in expected_tools:
            name = et if isinstance(et, str) else et.get("name", "")
            if name and name not in called:
                missing_tools += 1

    tool_signal = (tool_errors + missing_tools) / n_cases
    weights["tool_description"] += min(tool_signal, 1.0)
    weights["tool_implementation"] += min(tool_signal * 0.8, 1.0)

    # Signal 2: Field-specific failure analysis
    if eval_spec:
        struct_score = evaluation_results.get("avg_structure", 0)
        struct_max = float(eval_spec.get("structure_weight", 20))
        if struct_max > 0 and struct_score / struct_max < 0.8:
            weights["format_input"] += 1.0 - (struct_score / struct_max)

        fields = eval_spec.get("output_fields", {})
        n_fields = max(len(fields), 1)
        severe_field_failures = 0
        for fname, cfg in fields.items():
            avg = evaluation_results.get(f"avg_{fname}", 0)
            mx = float(cfg.get("weight", 0))
            if mx > 0 and avg / mx < 0.7:
                gap = 1.0 - avg / mx
                weights["agent_logic"] += gap / n_fields
                if avg / mx < 0.5:
                    severe_field_failures += 1

        if is_multi_file and severe_field_failures > 0:
            weights["helper_module"] += min(severe_field_failures * 0.3, 1.0)

    # Signal 3: Runtime errors / crashes → error_handling
    crash_count = sum(1 for c in case_results if c.get("output") is None or c.get("output") == {})
    if crash_count > 0:
        weights["error_handling"] += min(crash_count / n_cases, 1.0)

    # Signal 4: Historical effectiveness of focus areas
    for change in (successful_changes or [])[-10:]:
        focus = _extract_focus_from_method(change.get("method", ""))
        if not focus:
            for sug in change.get("suggestions", []):
                sug_lower = str(sug).lower()
                for f in _ALL_FOCUS_AREAS:
                    if f.replace("_", " ") in sug_lower:
                        focus = f
                        break
                if focus:
                    break
        if focus and focus in weights:
            weights[focus] += 0.12

    for attempt in (failed_attempts or [])[-5:]:
        focus = _extract_focus_from_method(attempt.get("method", ""))
        if not focus:
            for sug in attempt.get("suggestions", []):
                sug_lower = str(sug).lower()
                for f in _ALL_FOCUS_AREAS:
                    if f.replace("_", " ") in sug_lower:
                        focus = f
                        break
                if focus:
                    break
        if focus and focus in weights:
            weights[focus] -= 0.08

    # Signal 5: Failure cluster mechanisms
    if failure_registry is not None:
        cluster_weights = failure_registry.compute_component_weights()
        for k, v in cluster_weights.items():
            if k in weights:
                weights[k] += v * 0.5

    # Normalize to 0-1 range
    max_w = max(weights.values()) if weights else 1.0
    if max_w > 0:
        for k in weights:  # noqa: PLC0206
            weights[k] = max(0.0, min(1.0, weights[k] / max_w))

    return weights


def format_component_weights(weights: dict[str, float]) -> str:
    """Format component weights into a human-readable prompt section."""
    labels = {
        "tool_description": "tool_description (tool schemas, parameter descriptions)",
        "agent_logic": "agent_logic (control flow, orchestration, validation)",
        "format_input": "format_input (input data structuring for the LLM)",
        "system_prompt": "system_prompt (system prompt instructions)",
        "tool_implementation": "tool_implementation (tool execution logic in supporting modules)",
        "helper_module": "helper_module (utility functions, data processing helpers)",
        "error_handling": "error_handling (retry logic, fallbacks, input validation)",
    }
    sorted_w = sorted(weights.items(), key=lambda x: -x[1])
    lines: list[str] = []
    for k, v in sorted_w:
        pct = v * 100
        label = labels.get(k, k)
        bar = "\u2588" * int(pct / 5) + "\u2591" * (20 - int(pct / 5))
        lines.append(f"- {label}: {pct:.0f}% {bar}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@observe_safe(span_name="overmind.optimize.generate_candidates", type=SpanType.FUNCTION)
def generate_candidates(
    agent_code: str,
    case_results: list[dict],
    evaluation_results: dict,
    model: str,
    eval_spec: dict | None = None,
    failed_attempts: list[dict] | None = None,
    successful_changes: list[dict] | None = None,
    allow_model_change: bool = False,
    num_candidates: int = 3,
    temperature: float = 0.7,
    diagnosis_case_fraction: float = 1.0,
    iteration_seed: int = 42,
    policy_context: str = "",
    policy_constraints: str = "",
    *,
    entrypoint_fn: str,
    bundle: AgentBundle | None = None,
    agent_files: dict[str, str] | None = None,
    codegen_model: str = "",
    codegen_max_steps: int = 50,
    cluster_context: str = "",
    component_weights_context: str = "",
    focus_weights: dict[str, float] | None = None,
    return_plans_only: bool = False,
) -> list[dict]:
    """Generate *num_candidates* improved agent versions.

    Uses a shared-diagnosis approach: one diagnosis call identifies all
    failure patterns and change instructions, then N parallel codegen calls
    each apply those instructions with a different focus area for diversity.
    Falls back to single-pass if diagnosis fails.

    When *agent_files* is provided, the codegen phase uses an agentic tool
    loop (read/edit/grep/etc.) instead of single-shot LLM code generation.
    This produces higher-quality, targeted edits rather than full file rewrites.

    When *bundle* is provided, prompts use the virtual bundle representation
    and outputs are parsed as targeted piece updates.

    When *focus_weights* is provided, focus areas are assigned by descending
    weight instead of the default static round-robin order.
    """

    agent_model, capability = _detect_agent_model(agent_code)
    weak_name, weak_score, weak_max = _find_weakest_dimension(evaluation_results, eval_spec)
    mcr = (
        "You MAY change the MODEL constant if a different model would clearly help."
        if allow_model_change
        else "Do NOT change the MODEL constant."
    )

    FOCUS_AREAS_DEFAULT = list(_ALL_FOCUS_AREAS)

    # Resolve effective focus ordering: use dynamic weights if provided,
    # otherwise fall back to the default static order.
    if focus_weights:
        sorted_focuses = sorted(focus_weights.items(), key=lambda x: -x[1])
        FOCUS_AREAS = [k for k, v in sorted_focuses if v > 0.05]
        if not FOCUS_AREAS:
            FOCUS_AREAS = list(FOCUS_AREAS_DEFAULT)
    else:
        FOCUS_AREAS = list(FOCUS_AREAS_DEFAULT)

    use_bundle = bundle is not None and bundle.is_multi_file()

    @observe_safe(span_name="single_pass_fallback", type=SpanType.FUNCTION)
    def _gen_single_pass() -> dict:
        agent_tokens = len(agent_code) // 3
        sp_max_tokens = max(4000, min(16000, int(agent_tokens * 2.0)))

        def _build_sp_prompt(_max_cases: int) -> str:
            return SINGLE_PASS_PROMPT.format(
                agent_code_section=_build_agent_code_section(agent_code, bundle),
                entry_file=_get_entry_file(agent_code, bundle),
                entrypoint_fn=entrypoint_fn,
                scoring_mechanics=_format_scoring_mechanics(eval_spec),
                per_case_results=_format_per_case_results(
                    case_results,
                    eval_spec,
                    max_cases=_max_cases,
                ),
                tool_usage_analysis=_format_tool_usage_analysis(case_results),
                policy_context=policy_context or "(no policy defined)",
                avg_score=evaluation_results.get("avg_total", 0),
                weakest_dimension=weak_name,
                weakest_dim_score=weak_score,
                weakest_dim_max=weak_max,
                score_breakdown=_format_score_breakdown(evaluation_results, eval_spec),
                successful_changes=_format_successful_changes(successful_changes),
                failed_attempts=_format_failed_attempts(failed_attempts),
                fixed_elements=_format_fixed_elements(eval_spec),
                optimizable_elements=_format_optimizable_elements(eval_spec),
                model_change_rule=mcr,
                agent_model=agent_model,
                model_capability=capability,
                output_format_instruction=_get_output_format_instruction(bundle),
            )

        prompt = _render_within_budget(
            _build_sp_prompt,
            initial_max_cases=20,
            model=model,
        )
        try:
            resp = llm_completion(
                model,
                [{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=sp_max_tokens,
            )
            raw = resp.choices[0].message.content or ""
            finish_reason = resp.choices[0].finish_reason or "unknown"

            if use_bundle:
                analysis_str, suggs, file_updates = _parse_file_updates(raw)
                if file_updates:
                    return {
                        "analysis": analysis_str,
                        "suggestions": suggs,
                        "updated_code": None,
                        "bundle_updates": {
                            "file_updates": file_updates,
                        },
                        "method": "single_pass_bundle",
                        "_debug": {
                            "response_len": len(raw),
                            "finish_reason": finish_reason,
                            "files_updated": len(file_updates),
                        },
                    }

            analysis_str, suggs, code = _extract_code_and_analysis(raw, agent_code)
            return {
                "analysis": analysis_str,
                "suggestions": suggs,
                "updated_code": code,
                "method": "single_pass" if code else "failed",
                "_debug": {
                    "response_len": len(raw),
                    "finish_reason": finish_reason,
                    "has_code_fence": "```" in raw,
                    "code_extracted": code is not None,
                },
            }
        except Exception as exc:
            return {
                "analysis": f"Error: {exc}",
                "suggestions": [],
                "updated_code": None,
                "method": "error",
                "_debug": {"error": str(exc)},
            }

    # --- Adaptive context: reduce case count and history at high scores ---
    avg_score = evaluation_results.get("avg_total", 0)
    adaptive_max_cases = 20
    adaptive_history_cap = 8
    if avg_score >= 80:
        adaptive_max_cases = 6
        adaptive_history_cap = 3
    elif avg_score >= 70:
        adaptive_max_cases = 10
        adaptive_history_cap = 4

    trimmed_failed = failed_attempts[-adaptive_history_cap:] if failed_attempts else None
    trimmed_successful = successful_changes[-adaptive_history_cap:] if successful_changes else None

    # --- Shared diagnosis (single LLM call) ---
    shared_diag_ctx = DiagnosisContext(
        agent_code=agent_code,
        case_results=case_results,
        evaluation_results=evaluation_results,
        model=model,
        entrypoint_fn=entrypoint_fn,
        eval_spec=eval_spec,
        failed_attempts=trimmed_failed,
        successful_changes=trimmed_successful,
        allow_model_change=allow_model_change,
        temperature=temperature,
        iteration_seed=iteration_seed,
        policy_context=policy_context,
        bundle=bundle,
        cluster_context=cluster_context,
        component_weights_context=component_weights_context,
    )
    diag = _run_diagnosis(
        ctx=shared_diag_ctx,
        focus_area=None,
        case_fraction=diagnosis_case_fraction,
        max_cases=adaptive_max_cases,
    )

    if not diag:
        all_results = [_gen_single_pass() for _ in range(num_candidates)]
        if not any(r.get("updated_code") for r in all_results):
            return [
                {
                    "analysis": "No candidates produced valid code.",
                    "suggestions": [],
                    "updated_code": None,
                    "method": "failed",
                    "_debug": [r.get("_debug", {}) for r in all_results],
                }
            ]
        return all_results

    # --- Independent diagnosis for the last candidate (diversity) ---
    # When generating 3+ candidates, give the last one a completely
    # independent diagnosis with a different case subset and higher
    # temperature so it explores a different improvement direction.
    independent_diag: dict | None = None
    if num_candidates >= 3:
        independent_diag_ctx = DiagnosisContext(
            agent_code=agent_code,
            case_results=case_results,
            evaluation_results=evaluation_results,
            model=model,
            entrypoint_fn=entrypoint_fn,
            eval_spec=eval_spec,
            failed_attempts=trimmed_failed,
            successful_changes=trimmed_successful,
            allow_model_change=allow_model_change,
            temperature=min(temperature + 0.15, 1.0),
            iteration_seed=iteration_seed + 9973,
            policy_context=policy_context,
            bundle=bundle,
            cluster_context=cluster_context,
            component_weights_context=component_weights_context,
        )
        independent_diag = _run_diagnosis(
            ctx=independent_diag_ctx,
            focus_area=None,
            case_fraction=max(0.5, diagnosis_case_fraction - 0.2),
            max_cases=adaptive_max_cases,
        )

    # --- Parallel codegen forks with different focus areas ---
    # Ensure exploration diversity: when the dominant focus area has >70%
    # weight, force at least one candidate to use a non-dominant focus area
    # so the pipeline doesn't get stuck in a single strategy.
    focus_assignments: list[str | None] = []
    for idx in range(num_candidates):
        if idx < len(FOCUS_AREAS):
            focus_assignments.append(FOCUS_AREAS[idx])
        else:
            focus_assignments.append(None)

    if focus_weights and num_candidates >= 2 and len(FOCUS_AREAS) >= 1:
        dominant = FOCUS_AREAS[0]
        dominant_weight = focus_weights.get(dominant, 0)
        if dominant_weight > 0.70:
            non_dominant = [fa for fa in FOCUS_AREAS_DEFAULT if fa != dominant]
            if non_dominant:
                explore_idx = num_candidates - 2 if num_candidates >= 3 else 0
                explore_focus = random.Random(iteration_seed).choice(non_dominant)
                focus_assignments[explore_idx] = explore_focus

    # Resolve the entry file path for the agentic path
    _entry_file = (
        bundle.entry_file if bundle is not None else next(iter(agent_files), "agent.py") if agent_files else "agent.py"
    )

    # Choose the codegen model — fall back to the diagnosis model
    effective_codegen_model = codegen_model or model

    # Resolve optimizable file set for the agentic path
    _opt_files: set[str] | None = None
    if bundle is not None:
        _opt_files = bundle.optimizable_files

    # ---- Plans-only short-circuit ----
    # When the host coding agent (Cursor / Codex / Claude Code) is going to
    # do the per-candidate code edits in parallel git worktrees, skip the
    # in-process codegen forks and just return one plan per candidate. The
    # plan bundles the diagnosis (or independent diagnosis for the last
    # candidate when N>=3), the focus area, and a ready-to-use prompt the
    # host agent can hand to a sub-coding-agent.
    if return_plans_only:
        plans: list[dict] = []
        suggestions = [c.get("action", "") for c in diag.get("changes", [])]
        for idx, focus in enumerate(focus_assignments):
            is_last = idx == len(focus_assignments) - 1
            use_diag = independent_diag if (is_last and independent_diag) else diag
            use_suggestions = (
                [c.get("action", "") for c in (independent_diag or {}).get("changes", [])]
                if (is_last and independent_diag)
                else suggestions
            )
            instruction = _build_agentic_instruction(
                use_diag,
                eval_spec,
                policy_constraints,
                entrypoint_fn,
                _entry_file,
                agent_files or {_entry_file: agent_code},
                focus_area=focus,
                optimizable_files=_opt_files,
            )
            method_label = "plan(independent)" if (is_last and independent_diag) else f"plan({focus or 'general'})"
            plans.append({
                "candidate_id": f"c{idx}",
                "method": method_label,
                "focus_area": focus or "general",
                "diagnosis": use_diag,
                "suggestions": use_suggestions,
                "edit_instructions": instruction,
            })
        return plans

    # ---- Agentic codegen path ----
    if agent_files:

        def _agentic_fork(
            focus: str | None,
            use_diag: dict | None = None,
        ) -> dict:
            return _run_codegen_agentic(
                agent_files=agent_files,
                diagnosis=use_diag or diag,
                model=effective_codegen_model,
                eval_spec=eval_spec,
                policy_constraints=policy_constraints,
                entrypoint_fn=entrypoint_fn,
                entry_file=_entry_file,
                focus_area=focus,
                max_steps=codegen_max_steps,
                optimizable_files=_opt_files,
            )

        all_results: list[dict] = []
        if num_candidates <= 1:
            all_results.append(_agentic_fork(None))
        else:
            _log.info(
                f"Spawning {num_candidates} agentic codegen fork(s) in parallel (workers={min(num_candidates, 5)})"
            )
            with ThreadPoolExecutor(max_workers=min(num_candidates, 5)) as pool:
                # Snapshot the parent OTel context so spans created inside
                # each fork (e.g. overmind_llm_completion) nest under the
                # active workflow span instead of becoming orphan roots.
                parent_ctx = contextvars.copy_context()
                futures = []
                for idx, focus in enumerate(focus_assignments):
                    is_last = idx == len(focus_assignments) - 1
                    if is_last and independent_diag:
                        futures.append(
                            pool.submit(
                                parent_ctx.copy().run,
                                _agentic_fork,
                                None,
                                independent_diag,
                            )
                        )
                    else:
                        futures.append(pool.submit(parent_ctx.copy().run, _agentic_fork, focus))
                all_results = []
                for i, fut in enumerate(futures):
                    try:
                        all_results.append(fut.result())
                    except Exception:
                        _log.exception(f"Agentic codegen fork {i} failed")
                        all_results.append({
                            "analysis": "agentic fork crashed",
                            "suggestions": [],
                            "updated_code": None,
                            "method": "failed",
                            "_debug": {"fork_crash": True},
                        })

        has_any_output = any(r.get("updated_code") or r.get("bundle_updates") for r in all_results)
        if not has_any_output:
            sp_result = _gen_single_pass()
            if sp_result.get("updated_code") or sp_result.get("bundle_updates"):
                return [sp_result]
            return [
                {
                    "analysis": "No candidates produced valid code.",
                    "suggestions": [],
                    "updated_code": None,
                    "method": "failed",
                    "_debug": [r.get("_debug", {}) for r in all_results],
                }
            ]
        return all_results

    # ---- Legacy single-shot codegen path (no agent_files provided) ----

    def _codegen_for_focus(focus: str | None, use_diag: dict | None = None) -> dict:
        effective_diag = use_diag or diag
        effective_suggestions = [c.get("action", "") for c in effective_diag.get("changes", [])]
        result = _run_codegen(
            agent_code,
            effective_diag,
            model,
            eval_spec,
            temperature,
            policy_constraints=policy_constraints,
            entrypoint_fn=entrypoint_fn,
            focus_area=focus,
            bundle=bundle,
        )
        is_independent = use_diag is not None

        if isinstance(result, dict):
            return {
                "analysis": effective_diag.get("root_cause", ""),
                "suggestions": effective_suggestions,
                "updated_code": None,
                "bundle_updates": result,
                "method": (f"two_pass_bundle({'independent' if is_independent else focus or 'general'})"),
                "diagnosis": effective_diag,
                "_debug": {
                    "two_pass": True,
                    "bundle_mode": True,
                    "shared_diagnosis": not is_independent,
                    "focus": focus,
                    "files_updated": len(result.get("file_updates", {})),
                },
            }

        code = result
        return {
            "analysis": effective_diag.get("root_cause", ""),
            "suggestions": effective_suggestions,
            "updated_code": code,
            "method": (f"two_pass({'independent' if is_independent else focus or 'general'})" if code else "failed"),
            "diagnosis": effective_diag,
            "_debug": {
                "two_pass": True,
                "shared_diagnosis": not is_independent,
                "focus": focus,
                "code_extracted": code is not None,
            },
        }

    all_results: list[dict] = []
    if num_candidates <= 1:
        all_results.append(_codegen_for_focus(None))
    else:
        _log.info(f"Spawning {num_candidates} legacy codegen fork(s) in parallel (workers={min(num_candidates, 5)})")
        with ThreadPoolExecutor(max_workers=min(num_candidates, 5)) as pool:
            # Snapshot the parent OTel context so spans created inside each
            # fork nest under the active workflow span instead of becoming
            # orphan roots.
            parent_ctx = contextvars.copy_context()
            futures = []
            for idx, focus in enumerate(focus_assignments):
                is_last = idx == len(focus_assignments) - 1
                if is_last and independent_diag:
                    futures.append(
                        pool.submit(
                            parent_ctx.copy().run,
                            _codegen_for_focus,
                            None,
                            independent_diag,
                        )
                    )
                else:
                    futures.append(pool.submit(parent_ctx.copy().run, _codegen_for_focus, focus))
            all_results = []
            for i, fut in enumerate(futures):
                try:
                    all_results.append(fut.result())
                except Exception:
                    _log.exception(f"Legacy codegen fork {i} failed")
                    all_results.append({
                        "analysis": "codegen crashed",
                        "suggestions": [],
                        "updated_code": None,
                        "method": "failed",
                        "_debug": {"fork_crash": True},
                    })

    has_any_output = any(r.get("updated_code") or r.get("bundle_updates") for r in all_results)

    valid_count = sum(1 for r in all_results if r.get("updated_code") or r.get("bundle_updates"))
    methods = [r.get("method", "unknown") for r in all_results]
    set_tag(attrs.CANDIDATES_REQUESTED, int(num_candidates))
    set_tag(attrs.CANDIDATES_PRODUCED, valid_count)
    # ``methods`` is a list of strings — ``set_tag`` passes it through
    # natively; no need to pre-encode as JSON here.
    set_tag(attrs.CANDIDATES_METHODS, methods)
    set_tag(attrs.CANDIDATES_HAS_DIAGNOSIS, diag is not None)
    set_tag(attrs.CANDIDATES_USE_BUNDLE, bool(use_bundle))
    if diag:
        # Note: root_cause text intentionally not tagged — it can echo agent
        # code / policy snippets which we don't want to ship to the trace UI.
        set_tag(attrs.CANDIDATES_HAS_ROOT_CAUSE, bool(diag.get("root_cause")))

    if not has_any_output:
        sp_result = _gen_single_pass()
        if sp_result.get("updated_code") or sp_result.get("bundle_updates"):
            set_tag(attrs.CANDIDATES_FALLBACK, "single_pass")
            return [sp_result]
        set_tag(attrs.CANDIDATES_FALLBACK, "failed")
        return [
            {
                "analysis": "No candidates produced valid code.",
                "suggestions": [],
                "updated_code": None,
                "method": "failed",
                "_debug": [r.get("_debug", {}) for r in all_results],
            }
        ]
    return all_results
