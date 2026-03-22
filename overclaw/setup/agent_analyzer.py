"""LLM-based analysis of agent source code.

Produces a structural analysis, proposed evaluation spec, AND a tool schema
analysis that detects parameter issues, missing constraints, and tool
dependencies.

Supports both single-file and multi-file agents via ``AgentBundle``.
"""

import json
import logging
from pathlib import Path

from rich.console import Console
from rich.table import Table

from overclaw.core.bundler import AgentBundle
from overclaw.core.litellm_params import llm_completion
from overclaw.core.registry import project_root, project_root_from_agent_file
from overclaw.core.progress import make_spinner_progress
from overclaw.prompts.agent_analyzer import ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


def _build_setup_code_section(agent_path: str, bundle: AgentBundle | None) -> str:
    """Build the prompt-ready code section for analysis.

    Returns the bundle's prompt text for multi-file agents, or the entry
    file wrapped in a python fence for single-file agents.
    """
    if bundle and bundle.is_multi_file():
        return bundle.to_prompt_text()
    code = Path(agent_path).read_text()
    return f"```python\n{code}\n```"


def analyze_agent(
    agent_path: str,
    model: str,
    console: Console,
    *,
    entrypoint_fn: str,
) -> dict:
    """Analyze an agent file and return structured metadata with proposed criteria.

    Automatically detects multi-file agents by resolving local imports
    from the entry file and builds an ``AgentBundle`` when multiple
    project-local files are found.
    """
    code = Path(agent_path).read_text()

    bundle: AgentBundle | None = None
    try:
        pr = project_root_from_agent_file(agent_path)
        if pr is None:
            pr = project_root()
        root = str(pr)
        bundle = AgentBundle.from_entry_point(
            entry_path=agent_path,
            project_root=root,
            entrypoint_fn=entrypoint_fn,
        )
        if bundle.is_multi_file():
            logger.info(
                "Multi-file agent detected: %d files, %d pieces",
                len(bundle.original_files),
                len(bundle.pieces),
            )
    except Exception:
        logger.debug(
            "Bundle construction failed; falling back to single-file", exc_info=True
        )
        bundle = None

    agent_code_section = _build_setup_code_section(agent_path, bundle)

    with make_spinner_progress(console) as progress:
        task = progress.add_task("  Analyzing agent code and tool definitions…")

        response = llm_completion(
            model,
            [
                {
                    "role": "user",
                    "content": ANALYSIS_PROMPT.format(
                        agent_code_section=agent_code_section,
                        entrypoint_fn=entrypoint_fn,
                    ),
                }
            ],
            temperature=0.2,
            max_tokens=6000,
        )
        progress.update(task, completed=True)

    content = response.choices[0].message.content

    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            analysis = json.loads(content[start:end])
        else:
            raise ValueError("No JSON object found in LLM response")
    except (json.JSONDecodeError, ValueError) as exc:
        console.print(f"[red]Failed to parse analysis: {exc}[/red]")
        console.print("[dim]Raw response:[/dim]")
        console.print(content[:500])
        raise SystemExit(1)

    _display_analysis(analysis, console)

    analysis["_agent_path"] = agent_path
    analysis["_agent_code"] = code
    analysis["_agent_code_section"] = agent_code_section
    return analysis


def _display_analysis(analysis: dict, console: Console):
    """Pretty-print the analysis, proposed criteria, and tool analysis."""
    console.print("\n  [bold]Agent Description[/bold]")
    console.print(f"  {analysis.get('description', 'N/A')}\n")

    # --- Output schema table ---
    schema_table = Table(title="Detected Output Schema", border_style="blue")
    schema_table.add_column("Field", style="bold")
    schema_table.add_column("Type")
    schema_table.add_column("Details")

    for field_name, info in analysis.get("output_schema", {}).items():
        ftype = info.get("type", "unknown")
        if ftype == "enum":
            details = f"Values: {', '.join(info.get('values', []))}"
        elif ftype == "number":
            r = info.get("range", [])
            details = f"Range: {r[0]}\u2013{r[1]}" if len(r) == 2 else ""
        elif ftype == "text":
            details = info.get("description", "free text")
        else:
            details = info.get("description", "")
        schema_table.add_row(field_name, ftype, details)

    console.print(schema_table)

    # --- Proposed criteria table ---
    criteria = analysis.get("proposed_criteria", {})
    fields_criteria = criteria.get("fields", {})
    if fields_criteria:
        console.print()
        criteria_table = Table(
            title="Proposed Evaluation Criteria", border_style="green"
        )
        criteria_table.add_column("Field", style="bold")
        criteria_table.add_column("Importance")
        criteria_table.add_column("Scoring Detail")

        for field_name, fc in fields_criteria.items():
            importance = fc.get("importance", "important")
            output_schema = analysis.get("output_schema", {})
            ftype = output_schema.get(field_name, {}).get("type", "text")

            if ftype == "enum":
                detail = (
                    "partial credit"
                    if fc.get("partial_credit", True)
                    else "exact match only"
                )
            elif ftype == "number":
                detail = f"tolerance \u00b1{fc.get('tolerance', 10)}"
            elif ftype == "text":
                mode = fc.get("eval_mode", "non_empty")
                detail = "check non-empty" if mode == "non_empty" else "skip"
            else:
                detail = "exact match"

            criteria_table.add_row(field_name, importance, detail)

        sw = criteria.get("structure_weight", 20)
        criteria_table.add_row(
            "[dim]structure[/dim]",
            "[dim]\u2014[/dim]",
            f"[dim]{sw} pts for completeness[/dim]",
        )
        console.print(criteria_table)

    # --- Tool analysis table ---
    tool_analysis = analysis.get("tool_analysis", {})
    tools = tool_analysis.get("tools", {})
    if tools:
        console.print()
        tool_table = Table(title="Tool Schema Analysis", border_style="yellow")
        tool_table.add_column("Tool", style="bold")
        tool_table.add_column("Quality")
        tool_table.add_column("Issues")
        tool_table.add_column("Param Constraints")

        for tool_name, info in tools.items():
            quality = info.get("description_quality", "unknown")
            q_style = "green" if quality == "good" else "yellow"
            issues = info.get("issues", [])
            issues_str = "; ".join(issues[:2]) if issues else "none"
            constraints = info.get("param_constraints", {})
            constraints_str = (
                ", ".join(f"{p}: {v}" for p, v in constraints.items())[:60] or "none"
            )
            tool_table.add_row(
                tool_name,
                f"[{q_style}]{quality}[/{q_style}]",
                issues_str[:60],
                constraints_str,
            )

        console.print(tool_table)

    # --- Tool dependencies ---
    deps = tool_analysis.get("dependencies", [])
    if deps:
        console.print()
        console.print("  [bold]Tool Dependencies[/bold]")
        for dep in deps:
            console.print(
                f"    {dep.get('from_tool', '?')}.{dep.get('from_field', '?')} "
                f"\u2192 {dep.get('to_tool', '?')}.{dep.get('to_param', '?')} "
                f"[dim]({dep.get('description', '')})[/dim]"
            )

    # --- Orchestration issues ---
    orch_issues = tool_analysis.get("orchestration_issues", [])
    if orch_issues:
        console.print()
        console.print("  [yellow]Tool Orchestration Issues[/yellow]")
        for issue in orch_issues:
            console.print(f"    [yellow]\u26a0[/yellow] {issue}")

    # --- Consistency rules ---
    rules = analysis.get("consistency_rules", [])
    if rules:
        console.print()
        console.print("  [bold]Cross-Field Consistency Rules[/bold]")
        for rule in rules:
            console.print(
                f"    {rule.get('field_a', '?')} \u2194 {rule.get('field_b', '?')}: "
                f"{rule.get('description', '')} "
                f"[dim](penalty: {rule.get('penalty', 0)})[/dim]"
            )
