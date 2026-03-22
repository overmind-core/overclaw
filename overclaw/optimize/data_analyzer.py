"""Seed dataset analysis and coverage gap detection.

Used by Path B of the data pipeline: when the user already has seed data,
this module validates it against the eval spec, runs an LLM-based coverage
analysis, and produces a structured gap report that can drive targeted
augmentation.
"""

from __future__ import annotations

import json
import logging

from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from overclaw.core.branding import BRAND
from overclaw.core.progress import make_spinner_progress
from overclaw.optimize.data import (
    _format_input_schema,
    _format_output_schema,
    _llm_call,
    _safe_parse_json,
    validate_case_against_spec,
)
from overclaw.prompts.data_analyzer import DATA_QUALITY_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 0: Schema validation of seed data
# ---------------------------------------------------------------------------


def validate_seed_data(
    cases: list[dict],
    eval_spec: dict,
    console: Console | None = None,
) -> dict:
    """Validate every case against the eval spec and print a report.

    Returns a summary dict:
      valid_count, invalid_count, issues (list of {index, errors})
    """
    console = console or Console()
    issues: list[dict] = []

    for i, case in enumerate(cases):
        errors = validate_case_against_spec(case, eval_spec)
        if errors:
            issues.append({"index": i, "errors": errors})

    valid = len(cases) - len(issues)
    summary = {
        "total_cases": len(cases),
        "valid_count": valid,
        "invalid_count": len(issues),
        "issues": issues,
    }

    if not issues:
        console.print(
            f"  [bold {BRAND}]✓[/bold {BRAND}]"
            f"  [dim]Validated {len(cases)} cases — all valid[/dim]"
        )
    else:
        console.print(
            f"  [yellow]⚠[/yellow]"
            f"  [dim]Validated {len(cases)} cases —"
            f" {valid} valid, {len(issues)} invalid[/dim]"
        )
        for item in issues[:5]:
            console.print(
                f"    [dim]Case {item['index']}:[/dim]  " + "  ·  ".join(item["errors"])
            )
        if len(issues) > 5:
            console.print(f"    [dim]… and {len(issues) - 5} more[/dim]")

    return summary


# ---------------------------------------------------------------------------
# Phase 1B: LLM-based coverage analysis
# ---------------------------------------------------------------------------


def analyze_seed_coverage(
    cases: list[dict],
    eval_spec: dict,
    policy_context: str | None,
    agent_description: str,
    model: str,
    console: Console | None = None,
) -> dict:
    """Run an LLM-based analysis of seed data coverage and quality.

    Returns a structured analysis dict with coverage_gaps, quality scores,
    and augmentation recommendations.
    """
    console = console or Console()

    input_schema_text = _format_input_schema(eval_spec)
    output_schema_text = _format_output_schema(eval_spec)

    policy_section = ""
    if policy_context:
        policy_section = f"\n<PolicyRules>\n{policy_context}\n</PolicyRules>\n"

    sample_cases = cases if len(cases) <= 30 else cases[:15] + cases[-15:]
    dataset_json = json.dumps(sample_cases, indent=2, default=str)
    if len(dataset_json) > 12000:
        dataset_json = dataset_json[:12000] + "\n... (truncated)"

    prompt = DATA_QUALITY_ANALYSIS_PROMPT.format(
        agent_description=agent_description,
        input_schema_text=input_schema_text,
        output_schema_text=output_schema_text,
        policy_section=policy_section,
        dataset_json=dataset_json,
    )

    with make_spinner_progress(console, transient=True) as progress:
        progress.add_task("  Analyzing dataset coverage…")
        raw = _llm_call(model, prompt, temperature=0.3, max_tokens=4000)
    if not raw:
        logger.warning("Coverage analysis LLM call returned nothing")
        return _fallback_analysis(cases, eval_spec)

    parsed = _safe_parse_json(raw)
    if not isinstance(parsed, dict):
        logger.warning("Could not parse coverage analysis response")
        return _fallback_analysis(cases, eval_spec)

    _display_analysis(parsed, console)
    return parsed


def _fallback_analysis(cases: list[dict], eval_spec: dict) -> dict:
    """Mechanical-only coverage analysis when LLM fails."""
    output_fields = eval_spec.get("output_fields", {})
    uncovered_enum: dict[str, list[str]] = {}
    enum_seen: dict[str, set[str]] = {}

    for case in cases:
        out = case.get("expected_output", {})
        for field, cfg in output_fields.items():
            if cfg.get("type") == "enum":
                val = out.get(field)
                if val is not None:
                    enum_seen.setdefault(field, set()).add(str(val))

    for field, cfg in output_fields.items():
        if cfg.get("type") == "enum":
            allowed = set(cfg.get("values", []))
            seen = enum_seen.get(field, set())
            missing = allowed - seen
            if missing:
                uncovered_enum[field] = sorted(missing)

    gaps = []
    for field, missing_vals in uncovered_enum.items():
        gaps.append(
            {
                "area": f"enum:{field}",
                "description": f"Missing values: {', '.join(missing_vals)}",
                "severity": "medium",
            }
        )

    return {
        "overall_quality_score": 5,
        "case_count": len(cases),
        "difficulty_distribution": {},
        "coverage_gaps": gaps,
        "uncovered_policy_rules": [],
        "uncovered_edge_cases": [],
        "uncovered_enum_values": uncovered_enum,
        "quality_issues": [],
        "augmentation_recommendation": "LLM analysis unavailable; mechanical check found gaps above.",
        "suggested_additional_cases": max(5, len(cases) // 2),
    }


def _display_analysis(analysis: dict, console: Console) -> None:
    """Pretty-print the coverage analysis to the console."""
    score = analysis.get("overall_quality_score", "?")
    case_count = analysis.get("case_count", "?")
    score_color = (
        "green"
        if isinstance(score, (int, float)) and score >= 7
        else "yellow"
        if isinstance(score, (int, float)) and score >= 4
        else "red"
    )

    console.print()
    console.print(
        Rule(
            f"[bold {BRAND}]Dataset Analysis[/bold {BRAND}]  "
            f"[dim]{case_count} cases[/dim]",
            style="dim",
        )
    )
    console.print()

    # Quality score bar
    score_val = score if isinstance(score, (int, float)) else 0
    bar_filled = int(score_val * 2)
    bar = f"[{score_color}]{'█' * bar_filled}[/{score_color}][dim]{'░' * (20 - bar_filled)}[/dim]"
    console.print(f"  Quality  [{score_color}]{score}/10[/{score_color}]  {bar}")

    dist = analysis.get("difficulty_distribution", {})
    if dist:
        parts = [f"{k}: {v}" for k, v in dist.items() if v]
        console.print(f"  [dim]Difficulty  {' · '.join(parts)}[/dim]")

    console.print()

    gaps = analysis.get("coverage_gaps", [])
    if gaps:
        table = Table(
            border_style=f"{BRAND}",
            show_header=True,
            header_style=f"bold {BRAND}",
        )
        table.add_column("Severity")
        table.add_column("Area")
        table.add_column("Description")
        for gap in gaps[:10]:
            sev = gap.get("severity", "medium")
            sev_style = {"high": "red", "medium": "yellow", "low": "dim"}.get(
                sev, "dim"
            )
            table.add_row(
                f"[{sev_style}]{sev}[/{sev_style}]",
                f"[dim]{gap.get('area', '')}[/dim]",
                gap.get("description", ""),
            )
        console.print(table)
    else:
        console.print(
            f"  [bold {BRAND}]✓[/bold {BRAND}]  [dim]No significant coverage gaps[/dim]"
        )

    uncovered_rules = analysis.get("uncovered_policy_rules", [])
    if uncovered_rules:
        console.print(f"\n  [yellow]Uncovered rules ({len(uncovered_rules)})[/yellow]")
        for rule in uncovered_rules[:5]:
            console.print(f"    [dim]·[/dim]  {rule}")
        if len(uncovered_rules) > 5:
            console.print(f"    [dim]… and {len(uncovered_rules) - 5} more[/dim]")

    quality_issues = analysis.get("quality_issues", [])
    if quality_issues:
        console.print(f"\n  [yellow]Quality issues ({len(quality_issues)})[/yellow]")
        for qi in quality_issues[:5]:
            console.print(
                f"    [dim]Case {qi.get('case_index', '?')}[/dim]  {qi.get('issue', '')}"
            )
        if len(quality_issues) > 5:
            console.print(f"    [dim]… and {len(quality_issues) - 5} more[/dim]")

    rec = analysis.get("augmentation_recommendation", "")
    if rec:
        console.print(f"\n  [dim]Recommendation[/dim]  {rec}")
