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

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import logging

from overclaw.utils.llm import llm_completion
from overclaw.prompts.analyzer import (
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
    _BUNDLE_OUTPUT_INSTRUCTION,
    _SINGLE_FILE_OUTPUT_INSTRUCTION,
)

if TYPE_CHECKING:
    from overclaw.optimize.failure_registry import FailureRegistry
    from overclaw.utils.code import AgentBundle

_log = logging.getLogger("overclaw.optimize.analyzer")


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
        f"**Structure: {sw} pts** — All expected fields present and non-empty. "
        f"Score = (present / total) * {sw}."
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
                    f"**{label}: {weight} pts** (enum: {vals}) — "
                    f"Exact match = {weight}. Valid but wrong = {ps}."
                )
            else:
                lines.append(
                    f"**{label}: {weight} pts** (enum: {vals}) — "
                    f"Exact match = {weight}. Any mismatch = 0."
                )
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
            lines.append(f"**{label}: {weight} pts** (text) — {mode} check.")
        elif ftype == "boolean":
            lines.append(f"**{label}: {weight} pts** (boolean) — Exact match only.")

    tw = eval_spec.get("tool_usage_weight", 0)
    if tw > 0:
        lines.append(
            f"**Tool Usage: {tw} pts** — Correct tool calls, arguments, chaining."
        )

    jw = eval_spec.get("llm_judge_weight", 0)
    if jw > 0:
        lines.append(
            f"**LLM Judge: {jw} pts** — Semantic correctness, consistency, reasoning."
        )

    return "\n".join(lines)


def _format_per_case_results(
    case_results: list[dict],
    eval_spec: dict | None,
    *,
    max_cases: int = 20,
    case_fraction: float = 1.0,
    iteration_seed: int = 42,
) -> str:
    """Format per-case results for the analyzer.

    Expected values are shown for failing numeric fields so the diagnosis
    can identify scoring patterns (e.g., expected values correlating with
    tool-returned data).  Anti-overfitting rules in the prompt prevent
    hardcoding specific case values.
    """
    if not case_results:
        return "(no results available)"

    sorted_cases = sorted(
        case_results, key=lambda c: c.get("score", {}).get("total", 0)
    )

    # Partially blind diagnosis: only show a fraction of cases, with a
    # different random subset each iteration to prevent memorization.
    if 0 < case_fraction < 1.0 and len(sorted_cases) > 4:
        n_keep = max(3, int(len(sorted_cases) * case_fraction))
        worst = sorted_cases[:3]
        rest = sorted_cases[3:]
        n_from_rest = max(0, n_keep - 3)
        if n_from_rest < len(rest):
            rest = random.Random(iteration_seed).sample(rest, n_from_rest)
        sorted_cases = worst + sorted(
            rest, key=lambda c: c.get("score", {}).get("total", 0)
        )

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

    lines: list[str] = []
    for i, case in enumerate(visible):
        if omitted and i == len(visible) - 5:
            lines.append(f"... ({omitted} mid-range cases omitted) ...")
            lines.append("")

        score = case.get("score", {})
        total = score.get("total", 0)
        output = case.get("output", {})
        input_data = case.get("input", {})

        input_summary = ", ".join(
            f"{k}={json.dumps(v)}" for k, v in input_data.items()
        )[:400]

        lines.append(f"**Case {i + 1} \u2014 {total:.0f}/100**")
        lines.append(f"  Input: {input_summary}")

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
                        lines.append(
                            f"  [{mark}] {fname}: FAIL — MISSING ({fs:.1f}/{mx})"
                        )
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
                                f"  [{mark}] {fname}: FAIL — "
                                f"got {act!r}, off target ({pct:.0f}% credit, "
                                f"{fs:.1f}/{mx})"
                            )
                elif ftype == "text":
                    if act and str(act).strip():
                        lines.append(
                            f"  [{mark}] {fname}: FAIL — present but "
                            f"insufficient ({fs:.1f}/{mx})"
                        )
                    else:
                        lines.append(
                            f"  [{mark}] {fname}: FAIL — empty/missing ({fs:.1f}/{mx})"
                        )
                else:
                    lines.append(
                        f"  [{mark}] {fname}: FAIL — got {act!r} ({fs:.1f}/{mx})"
                    )

        struct_score = score.get("structure", 0)
        s_mark = "\u2713" if struct_score >= struct_max * 0.8 else "\u2717"
        lines.append(f"  [{s_mark}] structure: {struct_score:.1f}/{struct_max}")

        tool_trace = case.get("tool_trace", [])
        if tool_trace:
            lines.append("  Tool calls:")
            for t_idx, tc in enumerate(tool_trace, 1):
                args_str = json.dumps(tc.get("args", {}))
                if len(args_str) > 200:
                    args_str = args_str[:200] + "\u2026"
                result_str = json.dumps(tc.get("result", {}))
                if len(result_str) > 200:
                    result_str = result_str[:200] + "\u2026"
                err = tc.get("error")
                if err:
                    lines.append(
                        f"    {t_idx}. {tc.get('name', '?')}({args_str}) "
                        f"\u2192 ERROR: {err}"
                    )
                else:
                    lines.append(
                        f"    {t_idx}. {tc.get('name', '?')}({args_str}) "
                        f"\u2192 {result_str}"
                    )
        elif case.get("tool_calls"):
            lines.append(f"  Tools used: {', '.join(case['tool_calls'])}")

        lines.append("")

    return "\n".join(lines)


def _format_tool_usage_analysis(case_results: list[dict]) -> str:
    """Aggregate tool usage patterns across all cases."""
    if not case_results:
        return "(no tool data)"

    tool_calls_count: dict[str, int] = {}
    arg_values: dict[str, dict[str, list]] = {}
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
            for param, val in tc.get("args", {}).items():
                arg_values.setdefault(name, {}).setdefault(param, []).append(str(val))
            if tc.get("error"):
                errors.append(f"{name}: {tc['error']}")

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
    lines.append("**Argument value distribution:**")
    for tool_name, params in arg_values.items():
        for param, vals in params.items():
            unique = set(vals)
            if len(unique) <= 10:
                lines.append(f"  - {tool_name}.{param}: {sorted(unique)}")
            else:
                sample = sorted(unique)[:5]
                lines.append(
                    f"  - {tool_name}.{param}: {len(unique)} unique values "
                    f"(sample: {sample})"
                )

    if errors:
        lines.append("")
        lines.append("**Tool errors:**")
        for err in errors[:10]:
            lines.append(f"  - {err}")

    return "\n".join(lines) or "(no tool data)"


def _format_score_breakdown(evaluation: dict, eval_spec: dict | None) -> str:
    lines: list[str] = []
    for key, val in evaluation.items():
        if (
            key.startswith("avg_")
            and key != "avg_total"
            and isinstance(val, (int, float))
        ):
            nice = key.replace("avg_", "").replace("_", " ").title()
            field_key = key.replace("avg_", "")
            max_val = 0.0
            if eval_spec:
                if field_key == "structure":
                    max_val = float(eval_spec.get("structure_weight", 20))
                elif field_key in eval_spec.get("output_fields", {}):
                    max_val = float(
                        eval_spec["output_fields"][field_key].get("weight", 0)
                    )
                elif field_key == "tool_usage":
                    max_val = float(eval_spec.get("tool_usage_weight", 0))
                elif field_key == "llm_judge":
                    max_val = float(eval_spec.get("llm_judge_weight", 0))
            pct = f" ({val / max_val * 100:.0f}%)" if max_val else ""
            mx_str = f" / {max_val:.0f}" if max_val else ""
            lines.append(f"  {nice}: {val:.1f}{mx_str}{pct}")
    return "\n".join(lines) or "  (no breakdown)"


def _find_weakest_dimension(
    evaluation: dict, eval_spec: dict | None
) -> tuple[str, float, float]:
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
        avg_val = evaluation.get(f"avg_{field_name}", 0.0)
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
        avg_val = evaluation.get(f"avg_{dim_key}", 0.0)
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
            repeated_patterns.append(
                f"'{label}' attempted {count}x and FAILED every time"
            )

    if repeated_patterns:
        lines.append(
            "⚠️  REPEATEDLY FAILED APPROACHES (try something fundamentally different):"
        )
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
        try:
            parsed = json.loads(json_m.group(1).strip())
            analysis = parsed.get("analysis", parsed.get("root_cause", ""))
            suggestions = parsed.get(
                "suggestions",
                [c.get("action", "") for c in parsed.get("changes", [])],
            )
        except json.JSONDecodeError:
            pass

    if not analysis:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                candidate = text[start : end + 1]
                if (
                    not _matches_fingerprint(candidate, fingerprints)
                    and len(candidate) < 3000
                ):
                    parsed = json.loads(candidate)
                    analysis = parsed.get("analysis", parsed.get("root_cause", ""))
                    suggestions = parsed.get(
                        "suggestions",
                        [c.get("action", "") for c in parsed.get("changes", [])],
                    )
        except (json.JSONDecodeError, ValueError):
            pass

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
        try:
            parsed = json.loads(json_m.group(1).strip())
            analysis = parsed.get("analysis", parsed.get("root_cause", ""))
            suggestions = parsed.get(
                "suggestions",
                [c.get("action", "") for c in parsed.get("changes", [])],
            )
        except json.JSONDecodeError:
            pass

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


def _parse_bundle_updates(
    text: str,
) -> tuple[str, list[str], dict[str, str], list[tuple[str, str]]]:
    """Legacy parser for piece-ID format. Delegates to ``_parse_file_updates``
    first, falling back to piece-ID parsing for backward compatibility.

    Returns ``(analysis, suggestions, piece_updates, new_pieces)``.
    """
    analysis, suggestions, file_updates = _parse_file_updates(text)
    if file_updates:
        return analysis, suggestions, file_updates, []

    piece_updates: dict[str, str] = {}
    new_pieces: list[tuple[str, str]] = []

    exact_pattern = r"###\s*\[P(\d+)\]\s*\n```[a-zA-Z]*\s*\n(.*?)```"
    for m in re.finditer(exact_pattern, text, re.DOTALL):
        pid = f"P{m.group(1)}"
        code = m.group(2).strip()
        if code:
            piece_updates[pid] = code

    if not piece_updates:
        relaxed_pattern = r"\[P(\d+)\]\s*\n```[a-zA-Z]*\s*\n(.*?)```"
        for m in re.finditer(relaxed_pattern, text, re.DOTALL):
            pid = f"P{m.group(1)}"
            code = m.group(2).strip()
            if code:
                piece_updates[pid] = code

    new_pattern = r"###\s*\[NEW\]\s*IN:\s*(\S+)\s*\n```[a-zA-Z]*\s*\n(.*?)```"
    for m in re.finditer(new_pattern, text, re.DOTALL):
        file_path = m.group(1).strip()
        code = m.group(2).strip()
        if code:
            new_pieces.append((file_path, code))

    return analysis, suggestions, piece_updates, new_pieces


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


def _run_diagnosis(
    agent_code: str,
    case_results: list[dict],
    evaluation_results: dict,
    model: str,
    eval_spec: dict | None,
    failed_attempts: list[dict] | None,
    successful_changes: list[dict] | None,
    allow_model_change: bool,
    temperature: float,
    focus_area: str | None = None,
    case_fraction: float = 1.0,
    iteration_seed: int = 42,
    policy_context: str = "",
    *,
    entrypoint_fn: str,
    max_cases: int = 20,
    bundle: AgentBundle | None = None,
    cluster_context: str = "",
    component_weights_context: str = "",
) -> dict | None:
    """Pass 1: Produce a structured diagnosis.

    If *focus_area* is set, the diagnosis is steered to prioritize changes
    targeting that element (e.g. "tool_description", "agent_logic").
    When *bundle* is provided, the prompt uses the virtual bundle
    representation instead of a flat code string.
    """
    agent_model, capability = _detect_agent_model(agent_code)
    weak_name, weak_score, weak_max = _find_weakest_dimension(
        evaluation_results, eval_spec
    )

    mcr = (
        "You MAY suggest changing the MODEL constant."
        if allow_model_change
        else "Do NOT suggest changing the MODEL constant."
    )

    prompt_chars, prompt_lines = _measure_system_prompt(agent_code)

    prompt = DIAGNOSIS_PROMPT.format(
        agent_code_section=_build_agent_code_section(agent_code, bundle),
        entry_file=_get_entry_file(agent_code, bundle),
        entrypoint_fn=entrypoint_fn,
        scoring_mechanics=_format_scoring_mechanics(eval_spec),
        per_case_results=_format_per_case_results(
            case_results,
            eval_spec,
            max_cases=max_cases,
            case_fraction=case_fraction,
            iteration_seed=iteration_seed,
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
        model_change_rule=mcr,
        agent_model=agent_model,
        model_capability=capability,
        prompt_char_count=prompt_chars,
        prompt_line_count=prompt_lines,
    )

    if bundle is not None and bundle.is_multi_file():
        prompt += MULTI_FILE_AWARENESS_SECTION

    if cluster_context:
        prompt += FAILURE_CLUSTERS_SECTION.format(
            formatted_clusters=cluster_context,
        )

    if component_weights_context:
        prompt += COMPONENT_IMPACT_SECTION.format(
            component_lines=component_weights_context,
        )

    if focus_area:
        labels = {
            k: v.format(entrypoint_fn=entrypoint_fn) if "{" in v else v
            for k, v in FOCUS_LABELS.items()
        }
        focus_desc = labels.get(focus_area, focus_area)
        prompt += DIAGNOSIS_FOCUS_DIRECTIVE.format(
            focus_area=focus_area,
            focus_desc=focus_desc,
        )

    system_msg = DIAGNOSIS_SYSTEM_PROMPT.format(
        scoring_mechanics=_format_scoring_mechanics(eval_spec),
        optimizable_elements=_format_optimizable_elements(eval_spec),
        fixed_elements=_format_fixed_elements(eval_spec),
    )

    try:
        resp = llm_completion(
            model,
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=max(temperature * 0.5, 0.1),
            max_tokens=4000,
        )
        content = resp.choices[0].message.content or ""
        json_m = re.search(r"```json\s*\n(.*?)```", content, re.DOTALL)
        if json_m:
            return json.loads(json_m.group(1).strip())
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
    except Exception:
        pass
    return None


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
        ``{"piece_updates": {pid: code}, "new_pieces": [(file, code)]}``
        when operating in bundle mode.
    None
        On failure.
    """
    focus_directive = ""
    if focus_area:
        labels = {
            k: v.format(entrypoint_fn=entrypoint_fn) if "{" in v else v
            for k, v in FOCUS_LABELS.items()
        }
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
            # Fallback: try legacy piece-ID parsing
            _, _, piece_updates, new_pieces = _parse_bundle_updates(content)
            if piece_updates:
                return {
                    "piece_updates": piece_updates,
                    "new_pieces": new_pieces,
                }
            # Last resort: try single-file extraction
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
    """Extract imports from *source* that reference files in *known_files*."""
    import ast as _ast

    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return []

    stems = {}
    for kf in known_files:
        parts = kf.replace("/", ".").replace("\\", ".")
        if parts.endswith(".py"):
            parts = parts[:-3]
        for segment in parts.split("."):
            stems[segment] = kf

    imported: list[str] = []
    for node in _ast.iter_child_nodes(tree):
        if isinstance(node, _ast.Import):
            for alias in node.names:
                for seg in alias.name.split("."):
                    if seg in stems and stems[seg] not in imported:
                        imported.append(stems[seg])
        elif isinstance(node, _ast.ImportFrom):
            module = node.module or ""
            for seg in module.split("."):
                if seg in stems and stems[seg] not in imported:
                    imported.append(stems[seg])
            for alias in node.names:
                if alias.name in stems and stems[alias.name] not in imported:
                    imported.append(stems[alias.name])
    return imported


def _build_import_graph(agent_files: dict[str, str]) -> str:
    """Build a human-readable import graph for the agent files."""
    known = set(agent_files.keys())
    lines: list[str] = []
    for rel, src in agent_files.items():
        imports = _extract_imports_from_source(src, known - {rel})
        if imports:
            lines.append(
                f"- `{rel}` imports from: {', '.join(f'`{i}`' for i in imports)}"
            )
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
        file_listing = "\n".join(
            f"- `{rel}` ({len(src.splitlines())} lines)"
            for rel, src in agent_files.items()
        )

    import_graph = _build_import_graph(agent_files)

    policy_section = (
        f"- Policy constraints: {policy_constraints}" if policy_constraints else ""
    )

    focus_directive = ""
    if focus_area:
        labels = {
            k: v.format(entrypoint_fn=entrypoint_fn) if "{" in v else v
            for k, v in FOCUS_LABELS.items()
        }
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
    from overclaw.coding_agent import apply_code_changes

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
        _log.warning("Agentic codegen failed: %s", exc)
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
        "bundle_updates": (
            {"file_updates": result.file_updates} if is_multi_file else None
        ),
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
        expected_tools = (
            (eval_spec or {}).get("tool_config", {}).get("expected_tools", [])
        )
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
    crash_count = sum(
        1 for c in case_results if c.get("output") is None or c.get("output") == {}
    )
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
        for k in weights:
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
    weak_name, weak_score, weak_max = _find_weakest_dimension(
        evaluation_results, eval_spec
    )
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

    def _gen_single_pass() -> dict:
        agent_tokens = len(agent_code) // 3
        sp_max_tokens = max(4000, min(16000, int(agent_tokens * 2.0)))
        prompt = SINGLE_PASS_PROMPT.format(
            agent_code_section=_build_agent_code_section(agent_code, bundle),
            entry_file=_get_entry_file(agent_code, bundle),
            entrypoint_fn=entrypoint_fn,
            scoring_mechanics=_format_scoring_mechanics(eval_spec),
            per_case_results=_format_per_case_results(case_results, eval_spec),
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
                # Fallback: try legacy piece-ID format
                _, _, piece_updates, new_pieces = _parse_bundle_updates(raw)
                if piece_updates:
                    return {
                        "analysis": analysis_str if analysis_str else "",
                        "suggestions": suggs if suggs else [],
                        "updated_code": None,
                        "bundle_updates": {
                            "piece_updates": piece_updates,
                            "new_pieces": new_pieces,
                        },
                        "method": "single_pass_bundle_legacy",
                        "_debug": {
                            "response_len": len(raw),
                            "finish_reason": finish_reason,
                            "pieces_updated": len(piece_updates),
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

    trimmed_failed = (
        failed_attempts[-adaptive_history_cap:] if failed_attempts else None
    )
    trimmed_successful = (
        successful_changes[-adaptive_history_cap:] if successful_changes else None
    )

    # --- Shared diagnosis (single LLM call) ---
    diag = _run_diagnosis(
        agent_code,
        case_results,
        evaluation_results,
        model,
        eval_spec,
        trimmed_failed,
        trimmed_successful,
        allow_model_change,
        temperature,
        focus_area=None,
        case_fraction=diagnosis_case_fraction,
        iteration_seed=iteration_seed,
        policy_context=policy_context,
        entrypoint_fn=entrypoint_fn,
        max_cases=adaptive_max_cases,
        bundle=bundle,
        cluster_context=cluster_context,
        component_weights_context=component_weights_context,
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
        independent_diag = _run_diagnosis(
            agent_code,
            case_results,
            evaluation_results,
            model,
            eval_spec,
            trimmed_failed,
            trimmed_successful,
            allow_model_change,
            min(temperature + 0.15, 1.0),
            focus_area=None,
            case_fraction=max(0.5, diagnosis_case_fraction - 0.2),
            iteration_seed=iteration_seed + 9973,
            policy_context=policy_context,
            entrypoint_fn=entrypoint_fn,
            max_cases=adaptive_max_cases,
            bundle=bundle,
            cluster_context=cluster_context,
            component_weights_context=component_weights_context,
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
        bundle.entry_file
        if bundle is not None
        else next(iter(agent_files), "agent.py")
        if agent_files
        else "agent.py"
    )

    # Choose the codegen model — fall back to the diagnosis model
    effective_codegen_model = codegen_model or model

    # Resolve optimizable file set for the agentic path
    _opt_files: set[str] | None = None
    if bundle is not None:
        _opt_files = bundle.optimizable_files

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
            with ThreadPoolExecutor(max_workers=min(num_candidates, 5)) as pool:
                futures = []
                for idx, focus in enumerate(focus_assignments):
                    is_last = idx == len(focus_assignments) - 1
                    if is_last and independent_diag:
                        futures.append(
                            pool.submit(_agentic_fork, None, independent_diag)
                        )
                    else:
                        futures.append(pool.submit(_agentic_fork, focus))
                all_results = [f.result() for f in futures]

        has_any_output = any(
            r.get("updated_code") or r.get("bundle_updates") for r in all_results
        )
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
        effective_suggestions = [
            c.get("action", "") for c in effective_diag.get("changes", [])
        ]
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
                "method": (
                    f"two_pass_bundle("
                    f"{'independent' if is_independent else focus or 'general'})"
                ),
                "diagnosis": effective_diag,
                "_debug": {
                    "two_pass": True,
                    "bundle_mode": True,
                    "shared_diagnosis": not is_independent,
                    "focus": focus,
                    "files_updated": len(result.get("file_updates", {})),
                    "pieces_updated": len(result.get("piece_updates", {})),
                },
            }

        code = result
        return {
            "analysis": effective_diag.get("root_cause", ""),
            "suggestions": effective_suggestions,
            "updated_code": code,
            "method": (
                f"two_pass({'independent' if is_independent else focus or 'general'})"
                if code
                else "failed"
            ),
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
        with ThreadPoolExecutor(max_workers=min(num_candidates, 5)) as pool:
            futures = []
            for idx, focus in enumerate(focus_assignments):
                is_last = idx == len(focus_assignments) - 1
                if is_last and independent_diag:
                    futures.append(
                        pool.submit(_codegen_for_focus, None, independent_diag)
                    )
                else:
                    futures.append(pool.submit(_codegen_for_focus, focus))
            all_results = [f.result() for f in futures]

    has_any_output = any(
        r.get("updated_code") or r.get("bundle_updates") for r in all_results
    )
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


def analyze_and_improve(
    agent_code: str,
    traces: list[dict],
    evaluation_results: dict,
    model: str,
    eval_spec: dict | None = None,
    failed_attempts: list[dict] | None = None,
    successful_changes: list[dict] | None = None,
    allow_model_change: bool = False,
    case_results: list[dict] | None = None,
    num_candidates: int = 1,
    temperature: float = 0.7,
    policy_context: str = "",
    policy_constraints: str = "",
    *,
    entrypoint_fn: str,
    bundle: AgentBundle | None = None,
) -> dict:
    """Backward-compatible single-candidate wrapper."""
    candidates = generate_candidates(
        agent_code=agent_code,
        case_results=case_results or [],
        evaluation_results=evaluation_results,
        model=model,
        eval_spec=eval_spec,
        failed_attempts=failed_attempts,
        successful_changes=successful_changes,
        allow_model_change=allow_model_change,
        num_candidates=num_candidates,
        temperature=temperature,
        policy_context=policy_context,
        policy_constraints=policy_constraints,
        entrypoint_fn=entrypoint_fn,
        bundle=bundle,
    )
    return candidates[0]
