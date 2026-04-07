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
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from overclaw.utils.code import AgentBundle
from overclaw.utils.display import BRAND, make_spinner_progress
from overclaw.utils.llm import llm_completion
from overclaw.core.registry import project_root, project_root_from_agent_file
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

    # ---- Agent description ----
    desc = analysis.get("description", "N/A")
    console.print()
    console.print(
        Panel(
            f"[dim]{desc}[/dim]",
            title="[bold]Agent Description[/bold]",
            border_style=BRAND,
            padding=(1, 3),
        )
    )

    # ---- Output schema ----
    console.print()
    schema_table = Table(
        title="Detected Output Schema",
        border_style="blue",
        show_lines=True,
        padding=(0, 1),
    )
    schema_table.add_column("Field", style="bold", min_width=12)
    schema_table.add_column("Type", min_width=8)
    schema_table.add_column("Details", ratio=1)

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

    # ---- Proposed evaluation criteria ----
    criteria = analysis.get("proposed_criteria", {})
    fields_criteria = criteria.get("fields", {})
    if fields_criteria:
        console.print()
        criteria_table = Table(
            title="Proposed Evaluation Criteria",
            border_style="green",
            show_lines=True,
            padding=(0, 1),
        )
        criteria_table.add_column("Field", style="bold", min_width=12)
        criteria_table.add_column("Importance", min_width=10)
        criteria_table.add_column("Scoring Detail", ratio=1)

        for field_name, fc in fields_criteria.items():
            importance = fc.get("importance", "important")
            imp_style = (
                "red"
                if importance == "critical"
                else "yellow"
                if importance == "important"
                else "dim"
            )
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

            criteria_table.add_row(
                field_name, f"[{imp_style}]{importance}[/{imp_style}]", detail
            )

        sw = criteria.get("structure_weight", 20)
        criteria_table.add_row(
            "[dim]structure[/dim]",
            "[dim]\u2014[/dim]",
            f"[dim]{sw} pts for completeness[/dim]",
        )
        console.print(criteria_table)

    # ---- Tool analysis ----
    tool_analysis = analysis.get("tool_analysis", {})
    tools = tool_analysis.get("tools", {})
    deps = tool_analysis.get("dependencies", [])
    orch_issues = tool_analysis.get("orchestration_issues", [])

    if tools or deps or orch_issues:
        console.print()
        console.print(Rule("[bold]Tool Analysis[/bold]", style="dim"))

    if tools:
        console.print()
        for tool_name, info in tools.items():
            quality = info.get("description_quality", "unknown")
            q_style = "green" if quality == "good" else "yellow"
            issues = info.get("issues", [])
            constraints = info.get("param_constraints", {})

            lines = f"  [bold]Quality:[/bold]  [{q_style}]{quality}[/{q_style}]"
            if constraints:
                constraint_parts = [
                    f"[cyan]{p}[/cyan]: {v}" for p, v in constraints.items()
                ]
                lines += f"\n  [bold]Params:[/bold]   {', '.join(constraint_parts)}"
            if issues:
                lines += "\n  [bold]Issues:[/bold]"
                for issue in issues:
                    lines += f"\n    [yellow]\u26a0[/yellow] [dim]{issue}[/dim]"

            console.print(
                Panel(
                    lines,
                    title=f"[bold]{tool_name}[/bold]",
                    border_style="yellow",
                    padding=(0, 2),
                )
            )

    if deps:
        console.print()
        console.print("  [bold]Dependencies[/bold]")
        for dep in deps:
            console.print(
                f"    [cyan]{dep.get('from_tool', '?')}[/cyan]."
                f"{dep.get('from_field', '?')} \u2192 "
                f"[cyan]{dep.get('to_tool', '?')}[/cyan]."
                f"{dep.get('to_param', '?')}"
            )
            if dep.get("description"):
                console.print(f"    [dim]{dep['description']}[/dim]")

    if orch_issues:
        console.print()
        console.print("  [bold yellow]Orchestration Issues[/bold yellow]")
        for issue in orch_issues:
            console.print(f"    [yellow]\u26a0[/yellow] [dim]{issue}[/dim]")

    # ---- Consistency rules ----
    rules = analysis.get("consistency_rules", [])
    if rules:
        console.print()
        console.print(Rule("[bold]Cross-Field Consistency Rules[/bold]", style="dim"))
        console.print()
        for rule in rules:
            field_a = rule.get("field_a", "?")
            field_b = rule.get("field_b", "?")
            penalty = rule.get("penalty", 0)
            desc_text = rule.get("description", "")
            console.print(
                f"    [cyan]{field_a}[/cyan] \u2194 [cyan]{field_b}[/cyan]  "
                f"[dim](penalty: {penalty})[/dim]"
            )
            if desc_text:
                console.print(f"    [dim]{desc_text}[/dim]")

    console.print()
