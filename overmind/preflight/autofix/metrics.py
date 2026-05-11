"""Drop ``output_fields`` whose scorer raises so the spec produces a finite total."""

from __future__ import annotations

from overmind.preflight.autofix.weights import apply_invalid_weights
from overmind.preflight.state import IssueRecord, PatchRecord
from overmind.preflight.workspace import WorkingState

_SAFE_TYPES = {"text", "enum", "number", "boolean"}


def apply_metric_broken(state: WorkingState, issue: IssueRecord) -> list[PatchRecord]:
    """Sanitize the eval spec when the scorer raised on a row.

    Strategy:

    1. Replace any field type the evaluator does not know how to score
       with ``"text"`` (so the field is at least scorable in fallback
       mode).  Common offenders: ``"string"``, ``"object"``, ``"array"``.
    2. Trim invalid metric metadata — empty ``values`` on enum fields,
       missing ``range`` on number fields.
    3. Force a renormalize of weights afterwards.

    These are the deterministic plumbing fixes that map "scorer raised"
    to a green run.
    """
    fields = state.eval_spec.get("output_fields") or {}
    if not isinstance(fields, dict):
        return []

    changes: list[str] = []
    for name, cfg in list(fields.items()):
        if not isinstance(cfg, dict):
            fields[name] = {"type": "text", "weight": 0, "importance": "minor"}
            changes.append(f"{name} → coerced cfg to dict")
            continue
        ftype = (cfg.get("type") or "").lower()
        if ftype not in _SAFE_TYPES:
            cfg["type"] = "text"
            changes.append(f"{name}: type {ftype!r} → 'text'")
        if cfg.get("type") == "enum":
            values = cfg.get("values")
            if not isinstance(values, list) or not values:
                cfg["type"] = "text"
                cfg.pop("values", None)
                changes.append(f"{name}: enum without values → 'text'")
        if cfg.get("type") == "number":
            rng = cfg.get("range")
            if not (isinstance(rng, list) and len(rng) == 2 and all(isinstance(v, (int, float)) for v in rng)):
                cfg["range"] = [0, 1]
                changes.append(f"{name}: missing/invalid range → [0, 1]")

    if not changes:
        return []

    patches = [
        PatchRecord(
            iteration=0,
            kind=issue.kind,
            file=str(state.eval_spec_path),
            before_hash="",
            after_hash="",
            reason=issue.reason,
            diff_summary="Sanitised output_fields: " + "; ".join(changes[:6]) + ("…" if len(changes) > 6 else ""),
        )
    ]
    patches.extend(
        apply_invalid_weights(
            state,
            IssueRecord(
                kind="invalid_weights",
                severity="fix",
                target="eval_spec",
                reason="Renormalised after sanitising metrics.",
            ),
        )
    )
    return patches
