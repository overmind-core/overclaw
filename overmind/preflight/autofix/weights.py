"""Renormalize ``output_fields`` weights so the eval spec sums to ``total_points``."""

from __future__ import annotations

from overmind.preflight.state import IssueRecord, PatchRecord
from overmind.preflight.workspace import WorkingState


def apply_invalid_weights(state: WorkingState, issue: IssueRecord) -> list[PatchRecord]:
    spec = state.eval_spec
    fields = spec.get("output_fields") or {}
    if not isinstance(fields, dict) or not fields:
        return []

    total_declared = float(spec.get("total_points", 100))
    structure = float(spec.get("structure_weight", 0))
    tools = float(spec.get("tool_usage_weight", 0))
    judge = float(spec.get("llm_judge_weight", 0))
    field_budget = total_declared - (structure + tools + judge)
    if field_budget <= 0:
        # Pathological spec; force a reasonable structure weight back in
        # rather than producing zero-budget fields.
        spec["structure_weight"] = 20.0
        structure = 20.0
        field_budget = total_declared - (structure + tools + judge)
        if field_budget <= 0:
            field_budget = total_declared * 0.6
            spec["structure_weight"] = total_declared - field_budget - tools - judge

    current_field_sum = sum(float((cfg or {}).get("weight", 0)) for cfg in fields.values())
    importance_mult = {"critical": 3, "important": 2, "minor": 1}

    new_weights: dict[str, float] = {}
    if current_field_sum <= 0:
        # Allocate from importance only.
        weights_raw = {
            name: importance_mult.get((cfg or {}).get("importance", "important"), 2) for name, cfg in fields.items()
        }
        total_raw = sum(weights_raw.values()) or 1
        for name, raw in weights_raw.items():
            new_weights[name] = field_budget * (raw / total_raw)
    else:
        # Preserve the user-authored ratios but rescale to fit the budget.
        scale = field_budget / current_field_sum
        for name, cfg in fields.items():
            new_weights[name] = float((cfg or {}).get("weight", 0)) * scale

    rounded = {name: round(w, 1) for name, w in new_weights.items()}
    residual = round(field_budget - sum(rounded.values()), 1)
    if rounded:
        first = next(iter(rounded))
        rounded[first] = round(rounded[first] + residual, 1)

    for name, w in rounded.items():
        cfg = fields.get(name)
        if isinstance(cfg, dict):
            cfg["weight"] = w

    return [
        PatchRecord(
            iteration=0,
            kind=issue.kind,
            file=str(state.eval_spec_path),
            before_hash="",
            after_hash="",
            reason=issue.reason,
            diff_summary=(f"Renormalised {len(rounded)} field weights to fit total_points={total_declared}."),
        )
    ]
