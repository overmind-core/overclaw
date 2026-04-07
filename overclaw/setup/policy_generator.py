"""Policy elicitation and generation.

Conducts a focused conversation to extract domain knowledge, then uses an LLM
to produce a structured ``policies.md`` alongside a machine-readable policy
dict that gets embedded into the eval spec.

The policy has two layers:
  1. **Domain Knowledge** — business rules, terminology, edge cases, and
     context that the agent needs to reason correctly.  This is the ground
     truth the user owns and the optimizer tests *against*.
  2. **Agent Behavior** — output constraints, tool-calling expectations, and
     quality heuristics derived from the agent's code and eval spec.  This is
     what the optimizer uses to *measure* the agent.
"""

import json
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

from overclaw.utils.display import BRAND, overmind_prompt
from overclaw.utils.llm import llm_completion
from overclaw.utils.display import make_spinner_progress
from overclaw.prompts.policy_generator import (
    POLICY_FROM_CODE_PROMPT,
    POLICY_FROM_DOCUMENT_PROMPT,
    POLICY_GENERATION_PROMPT,
    POLICY_IMPROVE_PROMPT,
    POLICY_REFINE_PROMPT,
)

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _extract_markdown_and_json(text: str) -> tuple[str, dict]:
    """Pull out the markdown policy and the JSON summary from an LLM response.

    Markdown extraction tries, in order:
      1. ```markdown ... ``` (canonical)
      2. ```md ... ```       (common abbreviation)
      3. The largest non-JSON fenced block that looks like a policy heading
      4. Any text block containing a level-1 heading as a last resort
    """
    import re

    md_content = ""

    md_blocks = re.findall(r"```markdown\s*\n(.*?)```", text, re.DOTALL)
    if md_blocks:
        md_content = md_blocks[0].strip()

    if not md_content:
        md_blocks = re.findall(r"```md\s*\n(.*?)```", text, re.DOTALL)
        if md_blocks:
            md_content = md_blocks[0].strip()

    if not md_content:
        for m in re.finditer(
            r"```(?!json|changes)[a-zA-Z]*\s*\n(.*?)```", text, re.DOTALL
        ):
            block = m.group(1).strip()
            if block.startswith("#"):
                md_content = block
                break

    if not md_content:
        heading_m = re.search(r"(#\s+Agent Policy.*)", text, re.DOTALL)
        json_fence_m = re.search(r"```json", text)
        if heading_m:
            start = heading_m.start()
            end = json_fence_m.start() if json_fence_m else len(text)
            md_content = text[start:end].strip()

    json_blocks = re.findall(r"```json\s*\n(.*?)```", text, re.DOTALL)
    policy_data: dict = {}
    for block in json_blocks:
        try:
            candidate = json.loads(block.strip())
            if (
                "domain_rules" in candidate
                or "decision_rules" in candidate
                or "purpose" in candidate
            ):
                policy_data = candidate
                break
        except json.JSONDecodeError:
            continue

    if not policy_data:
        start = text.rfind("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                policy_data = json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

    # Migrate legacy format if needed
    policy_data = _migrate_legacy_policy(policy_data)

    return md_content, policy_data


def _migrate_legacy_policy(data: dict) -> dict:
    """Convert old single-layer policy dicts to the two-layer format."""
    if not data or "domain_rules" in data:
        return data

    migrated = {
        "purpose": data.get("purpose", ""),
        "domain_rules": data.get("decision_rules", []),
        "domain_edge_cases": data.get("edge_cases", []),
        "terminology": {},
        "output_constraints": data.get("hard_constraints", []),
        "tool_requirements": [],
        "decision_mapping": [],
        "quality_expectations": data.get("quality_expectations", []),
    }

    # Normalize edge case keys
    for i, ec in enumerate(migrated["domain_edge_cases"]):
        if isinstance(ec, dict) and "expected" in ec and "correct_handling" not in ec:
            migrated["domain_edge_cases"][i] = {
                "scenario": ec.get("scenario", ""),
                "correct_handling": ec["expected"],
            }

    return migrated


def _default_policy_data() -> dict:
    return {
        "purpose": "",
        "domain_rules": [],
        "domain_edge_cases": [],
        "terminology": {},
        "output_constraints": [],
        "tool_requirements": [],
        "decision_mapping": [],
        "quality_expectations": [],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def elicit_policy(
    analysis: dict,
    model: str,
    console: Console,
) -> tuple[str, dict]:
    """Conversationally elicit domain knowledge and generate a policy.

    Returns ``(markdown_text, structured_policy_dict)``.
    """
    description = analysis.get("description", "AI agent")
    agent_name = description.split(":")[0].strip() if ":" in description else "Agent"

    console.print()
    console.print(Rule(style="dim"))
    console.print()
    console.print(
        Panel(
            "[bold]Agent Policy Definition[/bold]\n"
            "[dim]Define the domain knowledge and rules that govern your "
            "agent's behaviour. The optimizer will test the agent against "
            "these rules.[/dim]",
            border_style=BRAND,
        )
    )

    console.print(
        "  [dim]Answer in plain language — I'll structure it into a formal "
        "policy document you can review and edit.[/dim]\n"
    )

    console.print(
        "  [bold cyan]Section 1: Domain Knowledge[/bold cyan]\n"
        "  [dim]These are the real-world rules your agent must follow — "
        "business logic, not code.[/dim]\n"
    )

    decision_rules = overmind_prompt(
        console,
        "[bold]What business rules should this agent follow when making "
        "decisions?[/bold]\n"
        "  [dim]e.g. 'Enterprise leads with recent pricing page visits are "
        "always high priority', 'Refunds over $500 need manager approval'[/dim]\n ",
    )

    hard_constraints = overmind_prompt(
        console,
        "[bold]What domain-specific mistakes are unacceptable?[/bold]\n"
        "  [dim]e.g. 'Never recommend a product that is out of stock', "
        "'A cold lead must never get schedule_demo'[/dim]\n ",
    )

    console.print(
        "\n  [dim]Describe any tricky real-world edge cases and what the "
        "correct handling should be. Press Enter to skip.[/dim]"
    )
    edge_cases = Prompt.ask(" ", default="")

    console.print(
        "\n  [dim]Define any key terms, categories, or thresholds the agent "
        "must understand. Press Enter to skip.[/dim]"
    )
    terminology = Prompt.ask(" ", default="")

    analysis_for_llm = {k: v for k, v in analysis.items() if not k.startswith("_")}

    with make_spinner_progress(console) as progress:
        task = progress.add_task("  Generating agent policy…")

        response = llm_completion(
            model,
            [
                {
                    "role": "user",
                    "content": POLICY_GENERATION_PROMPT.format(
                        analysis_json=json.dumps(analysis_for_llm, indent=2),
                        decision_rules=decision_rules,
                        hard_constraints=hard_constraints,
                        edge_cases=edge_cases or "(none provided)",
                        terminology=terminology or "(none provided)",
                        agent_name=agent_name,
                    ),
                }
            ],
            temperature=0.3,
            max_tokens=4000,
        )
        progress.update(task, completed=True)

    content = response.choices[0].message.content or ""
    md_text, policy_data = _extract_markdown_and_json(content)

    if not policy_data:
        policy_data = _default_policy_data()
    if not md_text:
        md_text = f"# Agent Policy: {agent_name}\n\n(Policy generation failed — please edit manually.)"

    display_policy(md_text, policy_data, console)
    return md_text, policy_data


def generate_policy_from_document(
    analysis: dict,
    document_path: str,
    model: str,
    console: Console,
) -> tuple[str, dict]:
    """Restructure an existing document into the canonical policy format.

    Returns ``(markdown_text, structured_policy_dict)``.
    """
    user_doc = Path(document_path).read_text()
    description = analysis.get("description", "AI agent")
    agent_name = description.split(":")[0].strip() if ":" in description else "Agent"
    analysis_for_llm = {k: v for k, v in analysis.items() if not k.startswith("_")}

    with make_spinner_progress(console) as progress:
        task = progress.add_task("  Structuring policy from document…")

        response = llm_completion(
            model,
            [
                {
                    "role": "user",
                    "content": POLICY_FROM_DOCUMENT_PROMPT.format(
                        analysis_json=json.dumps(analysis_for_llm, indent=2),
                        user_document=user_doc,
                        agent_name=agent_name,
                    ),
                }
            ],
            temperature=0.2,
            max_tokens=4000,
        )
        progress.update(task, completed=True)

    content = response.choices[0].message.content or ""
    md_text, policy_data = _extract_markdown_and_json(content)

    if not policy_data:
        policy_data = _default_policy_data()
    if not md_text:
        md_text = user_doc

    display_policy(md_text, policy_data, console)
    return md_text, policy_data


def generate_policy_from_code(
    analysis: dict,
    model: str,
    console: Console,
) -> tuple[str, dict]:
    """Infer a minimal policy from agent code alone (fast/auto mode).

    Returns ``(markdown_text, structured_policy_dict)``.
    """
    agent_code_section = analysis.get(
        "_agent_code_section",
        f"```python\n{analysis.get('_agent_code', '')}\n```",
    )
    description = analysis.get("description", "AI agent")
    agent_name = description.split(":")[0].strip() if ":" in description else "Agent"
    analysis_for_llm = {k: v for k, v in analysis.items() if not k.startswith("_")}

    with make_spinner_progress(console) as progress:
        task = progress.add_task("  Inferring policy from agent code…")

        response = llm_completion(
            model,
            [
                {
                    "role": "user",
                    "content": POLICY_FROM_CODE_PROMPT.format(
                        analysis_json=json.dumps(analysis_for_llm, indent=2),
                        agent_code_section=agent_code_section,
                        agent_name=agent_name,
                    ),
                }
            ],
            temperature=0.2,
            max_tokens=3000,
        )
        progress.update(task, completed=True)

    content = response.choices[0].message.content or ""
    md_text, policy_data = _extract_markdown_and_json(content)

    if not policy_data:
        policy_data = _default_policy_data()
    if not md_text:
        md_text = (
            f"<!-- Auto-generated from code analysis. "
            f"Edit to improve optimization quality. -->\n\n"
            f"# Agent Policy: {agent_name}\n\n"
            f"(Auto-generated — edit this file to add domain knowledge.)"
        )

    return md_text, policy_data


def improve_existing_policy(
    analysis: dict,
    existing_policy_path: str,
    model: str,
    console: Console,
) -> tuple[str, dict, str]:
    """Analyze an existing policy against agent code and suggest improvements.

    Returns ``(improved_markdown, improved_policy_dict, change_summary)``.
    """
    import re

    existing_md = Path(existing_policy_path).read_text()
    agent_code_section = analysis.get(
        "_agent_code_section",
        f"```python\n{analysis.get('_agent_code', '')}\n```",
    )
    description = analysis.get("description", "AI agent")
    agent_name = description.split(":")[0].strip() if ":" in description else "Agent"
    analysis_for_llm = {k: v for k, v in analysis.items() if not k.startswith("_")}

    with make_spinner_progress(console) as progress:
        task = progress.add_task("  Analyzing policy against agent code…")

        response = llm_completion(
            model,
            [
                {
                    "role": "user",
                    "content": POLICY_IMPROVE_PROMPT.format(
                        analysis_json=json.dumps(analysis_for_llm, indent=2),
                        agent_code_section=agent_code_section,
                        existing_policy=existing_md,
                        agent_name=agent_name,
                    ),
                }
            ],
            temperature=0.2,
            max_tokens=4000,
        )
        progress.update(task, completed=True)

    content = response.choices[0].message.content or ""

    changes_blocks = re.findall(r"```changes\s*\n(.*?)```", content, re.DOTALL)
    change_summary = changes_blocks[0].strip() if changes_blocks else ""

    if not change_summary:
        lines = content.split("```")[0].strip().splitlines()
        bullet_lines = [ln.strip() for ln in lines if ln.strip().startswith("-")]
        change_summary = "\n".join(bullet_lines[:5])

    md_text, policy_data = _extract_markdown_and_json(content)

    if not policy_data:
        policy_data = _default_policy_data()
    if not md_text:
        md_text = existing_md

    return md_text, policy_data, change_summary


def refine_policy(
    current_md: str,
    current_data: dict,
    analysis: dict,
    model: str,
    console: Console,
) -> tuple[str, dict]:
    """Refine an existing policy based on user feedback.

    Returns ``(updated_markdown, updated_policy_dict)``.
    """
    console.print(
        "\n  [dim]Tell me what you'd like to change about the policy.[/dim]\n"
    )

    feedback = overmind_prompt(
        console,
        "[bold]What would you like to add, remove, or change?[/bold]\n ",
    )

    console.print(
        "\n  [dim]Any additional domain rules or edge cases to include? "
        "Press Enter to skip.[/dim]"
    )
    additions = Prompt.ask(" ", default="")

    description = analysis.get("description", "AI agent")
    agent_name = description.split(":")[0].strip() if ":" in description else "Agent"

    analysis_for_llm = {k: v for k, v in analysis.items() if not k.startswith("_")}

    prompt = POLICY_REFINE_PROMPT.format(
        analysis_json=json.dumps(analysis_for_llm, indent=2),
        current_md=current_md,
        current_data_json=json.dumps(current_data, indent=2),
        feedback=feedback,
        additions=additions or "(none)",
        agent_name=agent_name,
    )

    with make_spinner_progress(console) as progress:
        task = progress.add_task("  Refining policy…")

        response = llm_completion(
            model,
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4000,
        )
        progress.update(task, completed=True)

    content = response.choices[0].message.content or ""
    md_text, policy_data = _extract_markdown_and_json(content)

    if not policy_data:
        policy_data = current_data
    if not md_text:
        md_text = current_md

    display_policy(md_text, policy_data, console)
    return md_text, policy_data


def save_policy(md_text: str, path: str) -> None:
    """Write the Markdown policy to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(md_text + "\n")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def display_policy(md_text: str, policy_data: dict, console: Console) -> None:
    """Show the generated policy to the user."""
    console.print()
    console.print(Rule(style="dim"))
    console.print()
    console.print(
        Panel(
            Markdown(md_text),
            title=f"[bold {BRAND}]Generated Agent Policy[/bold {BRAND}]",
            border_style=BRAND,
            padding=(1, 2),
        )
    )

    n_domain = len(policy_data.get("domain_rules", []))
    n_constraints = len(policy_data.get("output_constraints", []))
    n_edge = len(policy_data.get("domain_edge_cases", []))
    n_terms = len(policy_data.get("terminology", {}))

    parts = [
        f"{n_domain} domain rule(s)",
        f"{n_constraints} output constraint(s)",
        f"{n_edge} edge case(s)",
    ]
    if n_terms:
        parts.append(f"{n_terms} term(s) defined")

    console.print(f"\n  [dim]Policy contains {', '.join(parts)}[/dim]")
