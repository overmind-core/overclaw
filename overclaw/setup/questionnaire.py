"""Conversational refinement of evaluation criteria.

Instead of a rigid multiple-choice form, this module collects free-form
domain knowledge from the user and feeds it to an LLM that produces
refined evaluation criteria.
"""

import json

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from overclaw.core.branding import overmind_prompt
from overclaw.core.litellm_params import llm_completion
from overclaw.core.progress import make_spinner_progress
from overclaw.prompts.questionnaire import REFINEMENT_PROMPT


def run_questionnaire(analysis: dict, model: str, console: Console) -> dict:
    """Collect domain knowledge conversationally and refine criteria via LLM.

    Returns a modified copy of *analysis* with updated ``proposed_criteria``.
    """
    output_schema = analysis.get("output_schema", {})
    field_names = ", ".join(f.replace("_", " ") for f in output_schema)

    console.print(
        "\n  [dim]Tell me about your expectations so I can build better "
        "evaluation criteria.[/dim]\n"
    )

    # --- Open-ended questions ---
    feedback = overmind_prompt(
        console,
        "What would you like to change about the proposed criteria?\n ",
    )

    expectations = overmind_prompt(
        console,
        f"Your agent outputs: [bold]{field_names}[/bold].\n"
        "  Describe what a good output looks like — what matters most?\n ",
    )

    critical_mistakes = overmind_prompt(
        console,
        "What mistakes should cost the agent the most points?\n ",
    )

    console.print(
        "\n  [dim]Any other context about your domain or scoring preferences? "
        "(press Enter to skip)[/dim]"
    )
    additional_context = Prompt.ask(" ", default="")

    # --- Send to LLM for refinement ---
    original_criteria = analysis.get("proposed_criteria", {})

    # Strip internal keys before sending to the LLM
    analysis_for_llm = {k: v for k, v in analysis.items() if not k.startswith("_")}

    with make_spinner_progress(console) as progress:
        task = progress.add_task("\n  Refining evaluation criteria…")

        response = llm_completion(
            model,
            [
                {
                    "role": "user",
                    "content": REFINEMENT_PROMPT.format(
                        analysis_json=json.dumps(analysis_for_llm, indent=2),
                        criteria_json=json.dumps(original_criteria, indent=2),
                        feedback=feedback,
                        expectations=expectations,
                        critical_mistakes=critical_mistakes,
                        additional_context=additional_context or "(none)",
                    ),
                }
            ],
            temperature=0.2,
            max_tokens=2000,
        )
        progress.update(task, completed=True)

    content = response.choices[0].message.content

    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            refined_criteria = json.loads(content[start:end])
        else:
            raise ValueError("No JSON found in LLM response")
    except (json.JSONDecodeError, ValueError) as exc:
        console.print(
            f"\n  [yellow]Could not parse refined criteria ({exc}). "
            f"Using original proposal.[/yellow]"
        )
        refined_criteria = original_criteria

    # Show what changed
    _display_refined(refined_criteria, analysis, console)

    # Return a modified copy of the analysis
    refined_analysis = {**analysis, "proposed_criteria": refined_criteria}
    return refined_analysis


def _display_refined(criteria: dict, analysis: dict, console: Console):
    """Show the refined criteria table."""
    fields_criteria = criteria.get("fields", {})
    if not fields_criteria:
        return

    console.print()
    table = Table(title="Refined Evaluation Criteria", border_style="green")
    table.add_column("Field", style="bold")
    table.add_column("Importance")
    table.add_column("Scoring Detail")

    output_schema = analysis.get("output_schema", {})
    for field_name, fc in fields_criteria.items():
        importance = fc.get("importance", "important")
        ftype = output_schema.get(field_name, {}).get("type", "text")

        if ftype == "enum":
            detail = (
                "partial credit"
                if fc.get("partial_credit", True)
                else "exact match only"
            )
        elif ftype == "number":
            detail = f"tolerance ±{fc.get('tolerance', 10)}"
        elif ftype == "text":
            mode = fc.get("eval_mode", "non_empty")
            detail = "check non-empty" if mode == "non_empty" else "skip"
        else:
            detail = "exact match"

        table.add_row(field_name, importance, detail)

    sw = criteria.get("structure_weight", 20)
    table.add_row(
        "[dim]structure[/dim]",
        "[dim]—[/dim]",
        f"[dim]{sw} pts for completeness[/dim]",
    )
    console.print(table)
