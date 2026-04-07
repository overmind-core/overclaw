"""Policy loading and formatting for injection into pipeline stages.

Each consumer in the pipeline (analyzer, data generator, evaluator) needs a
different slice of the policy.  This module loads the structured policy from
``eval_spec.json`` and formats it for each consumer, keeping token budgets
tight.

The policy has two layers:
  1. **Domain Knowledge** — business rules, edge cases, terminology.
     Ground truth the user owns; the optimizer tests *against* these.
  2. **Agent Behavior** — output constraints, tool requirements, decision
     mapping, quality expectations.  Derived from code; the optimizer uses
     these to *measure* the agent.
"""

from __future__ import annotations

from pathlib import Path

from overclaw.core.paths import agent_setup_spec_dir


def default_policy_path(agent_name: str) -> str:
    """Canonical location for an agent's policy file (inside setup_spec/)."""
    return str(agent_setup_spec_dir(agent_name) / "policies.md")


def load_policy_data(eval_spec: dict) -> dict | None:
    """Extract the structured policy block from an eval spec.

    Returns ``None`` when no policy was embedded.
    """
    return eval_spec.get("policy")


def load_policy_markdown(agent_name: str) -> str | None:
    """Read the raw ``policies.md`` file if it exists."""
    path = Path(default_policy_path(agent_name))
    if path.exists():
        return path.read_text()
    return None


def _is_two_layer(policy: dict) -> bool:
    """Check whether the policy uses the new two-layer format."""
    return "domain_rules" in policy


def _get_rules(policy: dict) -> list[str]:
    """Get decision/domain rules regardless of format."""
    if _is_two_layer(policy):
        return policy.get("domain_rules", [])
    return policy.get("decision_rules", [])


def _get_constraints(policy: dict) -> list[str]:
    """Get hard/output constraints regardless of format."""
    if _is_two_layer(policy):
        return policy.get("output_constraints", [])
    return policy.get("hard_constraints", [])


def _get_edge_cases(policy: dict) -> list:
    """Get edge cases regardless of format."""
    if _is_two_layer(policy):
        return policy.get("domain_edge_cases", [])
    return policy.get("edge_cases", [])


def _get_quality_expectations(policy: dict) -> list[str]:
    if _is_two_layer(policy):
        return policy.get("quality_expectations", [])
    return policy.get("quality_expectations", [])


# ---------------------------------------------------------------------------
# Formatters — one per pipeline consumer
# ---------------------------------------------------------------------------


def format_for_diagnosis(policy: dict) -> str:
    """Policy context for the diagnosis prompt.

    Includes domain rules + output constraints + decision mapping so the
    analyzer can check whether failures are policy violations.
    """
    if not policy:
        return ""

    lines: list[str] = []

    purpose = policy.get("purpose", "")
    if purpose:
        lines.append(f"**Purpose:** {purpose}")
        lines.append("")

    rules = _get_rules(policy)
    if rules:
        lines.append("**Domain Rules (ground truth the agent must follow):**")
        for i, rule in enumerate(rules, 1):
            lines.append(f"  {i}. {rule}")
        lines.append("")

    constraints = _get_constraints(policy)
    if constraints:
        lines.append("**Output Constraints (the agent must never violate):**")
        for c in constraints:
            lines.append(f"  - {c}")
        lines.append("")

    mapping = policy.get("decision_mapping", [])
    if mapping:
        lines.append("**Decision Mapping (how domain signals map to outputs):**")
        for m in mapping:
            lines.append(f"  - {m}")
        lines.append("")

    terminology = policy.get("terminology", {})
    if terminology:
        lines.append("**Key Terminology:**")
        for term, defn in terminology.items():
            lines.append(f"  - **{term}**: {defn}")
        lines.append("")

    tool_reqs = policy.get("tool_requirements", [])
    if tool_reqs:
        lines.append("**Tool Requirements:**")
        for t in tool_reqs:
            lines.append(f"  - {t}")
        lines.append("")

    # Legacy support
    priorities = policy.get("priority_order", [])
    if priorities:
        lines.append(
            "**Priority order (when rules conflict):** " + " > ".join(priorities)
        )

    return "\n".join(lines)


def format_for_codegen(policy: dict) -> str:
    """Minimal policy context for the codegen prompt.

    Output constraints + tool requirements so generated code respects them.
    """
    if not policy:
        return ""

    lines: list[str] = []

    constraints = _get_constraints(policy)
    if constraints:
        lines.append("Output constraints the agent must satisfy:")
        for c in constraints:
            lines.append(f"  - {c}")

    tool_reqs = policy.get("tool_requirements", [])
    if tool_reqs:
        if lines:
            lines.append("")
        lines.append("Tool requirements:")
        for t in tool_reqs:
            lines.append(f"  - {t}")

    mapping = policy.get("decision_mapping", [])
    if mapping:
        if lines:
            lines.append("")
        lines.append("Decision mapping rules:")
        for m in mapping:
            lines.append(f"  - {m}")

    return "\n".join(lines) if lines else ""


def format_for_synthetic_data(policy: dict) -> str:
    """Full domain rules + edge cases for synthetic data generation.

    Richer context is justified here since this prompt runs once.  Domain
    knowledge is critical for generating realistic test cases.
    """
    if not policy:
        return ""

    lines: list[str] = []

    rules = _get_rules(policy)
    if rules:
        lines.append(
            "The agent follows these domain rules. Generate test "
            "cases that exercise each rule, including boundary "
            "conditions:"
        )
        for i, rule in enumerate(rules, 1):
            lines.append(f"  {i}. {rule}")
        lines.append("")

    edge_cases = _get_edge_cases(policy)
    if edge_cases:
        lines.append("Known edge cases — generate test cases for each:")
        for ec in edge_cases:
            if isinstance(ec, dict):
                scenario = ec.get("scenario", "")
                handling = ec.get("correct_handling", ec.get("expected", ""))
            else:
                scenario = str(ec)
                handling = ""
            if handling:
                lines.append(f"  - {scenario} → {handling}")
            else:
                lines.append(f"  - {scenario}")
        lines.append("")

    terminology = policy.get("terminology", {})
    if terminology:
        lines.append("Key terminology the test data should reflect:")
        for term, defn in terminology.items():
            lines.append(f"  - **{term}**: {defn}")
        lines.append("")

    constraints = _get_constraints(policy)
    if constraints:
        lines.append("Output constraints — include cases that test these boundaries:")
        for c in constraints:
            lines.append(f"  - {c}")

    return "\n".join(lines)


def format_for_judge(policy: dict) -> str:
    """Policy-derived rubric for the LLM judge.

    Domain rules are the primary evaluation criteria; output constraints and
    quality expectations provide the scoring rubric.
    """
    if not policy:
        return ""

    lines: list[str] = []

    rules = _get_rules(policy)
    if rules:
        lines.append("Check whether the agent correctly applied these domain rules:")
        for i, rule in enumerate(rules, 1):
            lines.append(f"  {i}. {rule}")
        lines.append("")

    mapping = policy.get("decision_mapping", [])
    if mapping:
        lines.append("Check whether outputs are consistent with these mappings:")
        for m in mapping:
            lines.append(f"  - {m}")
        lines.append("")

    constraints = _get_constraints(policy)
    if constraints:
        lines.append("Check whether the agent violated any output constraints:")
        for c in constraints:
            lines.append(f"  - {c}")
        lines.append("")

    expectations = _get_quality_expectations(policy)
    if expectations:
        lines.append("Quality expectations:")
        for e in expectations:
            lines.append(f"  - {e}")

    return "\n".join(lines)
