"""Drop dataset rows that don't conform to the eval spec contract."""

from __future__ import annotations

from overmind.preflight.state import IssueRecord, PatchRecord
from overmind.preflight.workspace import WorkingState


def _row_violates(row: dict, expected_input_keys: set[str]) -> str | None:
    """Return a short reason if *row* doesn't fit the schema, else ``None``."""
    if not isinstance(row, dict):
        return "row_not_dict"
    inp = row.get("input")
    if inp is None and "input_data" in row:
        inp = row.get("input_data")
    if inp is None:
        return "missing_input"
    if not expected_input_keys:
        return None
    if isinstance(inp, dict):
        actual = set(inp.keys())
        if actual != expected_input_keys:
            extras = sorted(actual - expected_input_keys)
            missing = sorted(expected_input_keys - actual)
            parts = []
            if missing:
                parts.append(f"missing={missing}")
            if extras:
                parts.append(f"extras={extras}")
            return "input_keys " + ", ".join(parts)
    return None


def apply_dataset_row_invalid(state: WorkingState, issue: IssueRecord) -> list[PatchRecord]:
    """Drop rows that don't match ``input_schema``.

    The classifier currently emits one of these per offending row, but
    we sweep the whole dataset on the first hit so the next iteration
    sees a clean slice.  Idempotent — if there's nothing to drop, no
    patch is recorded.
    """
    expected_input_keys = set((state.eval_spec.get("input_schema") or {}).keys())

    kept: list[dict] = []
    dropped_idx: list[int] = []
    for idx, row in enumerate(state.dataset):
        reason = _row_violates(row, expected_input_keys)
        if reason is None:
            kept.append(row)
        else:
            dropped_idx.append(idx)

    if not dropped_idx:
        return []

    state.dataset = kept
    return [
        PatchRecord(
            iteration=0,
            kind=issue.kind,
            file=str(state.dataset_path),
            before_hash="",
            after_hash="",
            reason=issue.reason,
            diff_summary=f"Dropped {len(dropped_idx)} dataset row(s) violating input_schema (indices: {dropped_idx[:8]}{'…' if len(dropped_idx) > 8 else ''}).",
        )
    ]
