"""Generate evaluation spec from analysis and (optionally) user preferences."""

import json
from pathlib import Path

IMPORTANCE_MULTIPLIERS = {"critical": 3, "important": 2, "minor": 1}


def generate_spec_from_proposal(
    analysis: dict, policy_data: dict | None = None
) -> dict:
    """Build an eval spec directly from the LLM's proposed criteria.

    When *policy_data* is provided, it is embedded into the spec so that
    downstream consumers (optimizer, data generator, judge) can access it
    without re-reading the Markdown file.
    """
    output_schema = analysis.get("output_schema", {})
    criteria = analysis.get("proposed_criteria", {})
    fields_criteria = criteria.get("fields", {})
    structure_weight = criteria.get("structure_weight", 20)

    field_importance = {
        name: fc.get("importance", "important") for name, fc in fields_criteria.items()
    }

    field_settings: dict[str, dict] = {}
    for field_name, fc in fields_criteria.items():
        ftype = output_schema.get(field_name, {}).get("type", "text")
        settings: dict = {}
        if ftype == "enum":
            settings["partial_credit"] = fc.get("partial_credit", True)
        elif ftype == "number":
            settings["tolerance"] = fc.get("tolerance", 10)
        elif ftype == "text":
            settings["eval_mode"] = fc.get("eval_mode", "non_empty")
        field_settings[field_name] = settings

    return _build_spec(
        analysis,
        output_schema,
        field_importance,
        field_settings,
        structure_weight,
        policy_data=policy_data,
    )


def _build_spec(
    analysis: dict,
    output_schema: dict,
    field_importance: dict[str, str],
    field_settings: dict[str, dict],
    structure_weight: int,
    policy_data: dict | None = None,
) -> dict:
    """Shared logic for assembling the final spec dict."""
    tool_analysis = analysis.get("tool_analysis", {})
    has_tools = bool(tool_analysis.get("tools"))

    # Reserve points for tool_usage if tools exist
    tool_usage_weight = 0
    if has_tools:
        tool_usage_weight = 10

    available_points = 100 - structure_weight - tool_usage_weight

    # Compute raw importance weights
    raw_weights: dict[str, int] = {}
    total_raw = 0
    for field_name in output_schema:
        importance = field_importance.get(field_name, "important")
        raw = IMPORTANCE_MULTIPLIERS.get(importance, 2)
        raw_weights[field_name] = raw
        total_raw += raw

    # Allocate points proportionally, then fix rounding
    weights: dict[str, int] = {}
    for field_name in output_schema:
        raw = raw_weights.get(field_name, 2)
        weights[field_name] = round((raw / max(total_raw, 1)) * available_points)

    weight_sum = sum(weights.values())
    if weight_sum != available_points and weights:
        first = next(iter(weights))
        weights[first] += available_points - weight_sum

    # Build per-field spec
    output_fields: dict[str, dict] = {}
    for field_name, info in output_schema.items():
        ftype = info.get("type", "text")
        weight = weights.get(field_name, 0)
        settings = field_settings.get(field_name, {})

        field_spec: dict = {
            "type": ftype,
            "description": info.get("description", ""),
            "weight": weight,
            "importance": field_importance.get(field_name, "important"),
        }

        if ftype == "enum":
            field_spec["values"] = info.get("values", [])
            field_spec["partial_credit"] = settings.get("partial_credit", True)
            field_spec["partial_score"] = max(1, round(weight * 0.2))

        elif ftype == "number":
            field_spec["range"] = info.get("range", [0, 100])
            tolerance = settings.get("tolerance", 10)
            field_spec["tolerance_bands"] = [
                {"within": max(1, tolerance // 2), "score_pct": 1.0},
                {"within": tolerance, "score_pct": 0.8},
                {"within": int(tolerance * 1.5), "score_pct": 0.5},
                {"within": int(tolerance * 2.5), "score_pct": 0.25},
            ]

        elif ftype == "text":
            field_spec["eval_mode"] = settings.get("eval_mode", "non_empty")

        output_fields[field_name] = field_spec

    # Build tool_config from tool_analysis
    tool_config: dict = {}
    if tool_analysis:
        tool_config["expected_tools"] = tool_analysis.get("expected_tools", [])
        tool_config["dependencies"] = tool_analysis.get("dependencies", [])

        param_constraints: dict[str, dict] = {}
        for tool_name, tool_info in tool_analysis.get("tools", {}).items():
            constraints = tool_info.get("param_constraints", {})
            if constraints:
                param_constraints[tool_name] = constraints
        tool_config["param_constraints"] = param_constraints

    # Extract consistency rules
    consistency_rules = analysis.get("consistency_rules", [])

    spec = {
        "agent_description": analysis.get("description", ""),
        "agent_path": analysis.get("_agent_path", ""),
        "input_schema": analysis.get("input_schema", {}),
        "output_fields": output_fields,
        "structure_weight": structure_weight,
        "total_points": 100,
        "optimizable_elements": analysis.get("optimizable_elements", []),
        "fixed_elements": analysis.get("fixed_elements", []),
    }

    if tool_config:
        spec["tool_config"] = tool_config
        spec["tool_usage_weight"] = tool_usage_weight

    if consistency_rules:
        spec["consistency_rules"] = consistency_rules

    if policy_data:
        spec["policy"] = policy_data

    return spec


def save_spec(spec: dict, path: str):
    """Write the evaluation spec to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(spec, f, indent=2)
