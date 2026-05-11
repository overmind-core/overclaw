"""Reconcile ``input_schema`` / ``output_fields`` with the entrypoint contract."""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

from overmind.core.registry import resolve_agent
from overmind.preflight.autofix.weights import apply_invalid_weights
from overmind.preflight.state import IssueRecord, PatchRecord
from overmind.preflight.workspace import WorkingState

# ---------------------------------------------------------------------------
# input_schema ↔ entrypoint signature
# ---------------------------------------------------------------------------


def _entrypoint_signature(agent_name: str) -> list[str] | None:
    """Best-effort param list of the registered entrypoint.

    Tries live import first (most accurate, picks up wrapped/decorated
    functions), then falls back to AST so a broken import still yields
    something usable.  Returns ``None`` only when both paths fail.
    """
    try:
        agent_path, fn_name = resolve_agent(agent_name)
    except SystemExit:
        return None
    p = Path(agent_path)

    try:
        sys.path.insert(0, str(p.parent))
        module_name = p.stem
        try:
            mod = __import__(module_name)
        finally:
            sys.path.pop(0)
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            sig = inspect.signature(fn)
            params = [
                name
                for name, param in sig.parameters.items()
                if param.kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ]
            return params
    except Exception:  # noqa: S110 - fall through to AST-based parsing below
        pass

    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except (SyntaxError, OSError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
            args = node.args
            return [a.arg for a in (args.posonlyargs + args.args + args.kwonlyargs)]
    return None


def apply_entrypoint_signature(state: WorkingState, issue: IssueRecord) -> list[PatchRecord]:
    """Rewrite ``eval_spec.input_schema`` to match the entrypoint signature.

    Triggered by classifier when the runtime raises a TypeError about an
    unexpected / missing keyword argument — that's a deterministic
    signal that the dataset/spec disagrees with the function signature.
    """
    params = _entrypoint_signature(state.agent_name)
    if not params:
        return []

    schema = state.eval_spec.get("input_schema")
    if not isinstance(schema, dict):
        schema = {}
        state.eval_spec["input_schema"] = schema

    existing_keys = list(schema.keys())
    if set(existing_keys) == set(params):
        return []

    new_schema: dict[str, dict] = {}
    for name in params:
        if name in schema and isinstance(schema[name], dict):
            new_schema[name] = schema[name]
        else:
            new_schema[name] = {"type": "text", "description": f"Auto-derived from entrypoint signature ({name})."}
    state.eval_spec["input_schema"] = new_schema

    return [
        PatchRecord(
            iteration=0,
            kind=issue.kind,
            file=str(state.eval_spec_path),
            before_hash="",
            after_hash="",
            reason=issue.reason,
            diff_summary=(f"Rebuilt input_schema from entrypoint signature: {existing_keys} → {params}."),
        )
    ]


# ---------------------------------------------------------------------------
# output_fields ↔ what the entrypoint actually returns
# ---------------------------------------------------------------------------


def apply_output_schema_mismatch(state: WorkingState, issue: IssueRecord) -> list[PatchRecord]:
    """Drop ``output_fields`` entries the agent never returned, then renormalize weights."""
    missing = list(issue.details.get("scored_but_missing") or [])
    fields = state.eval_spec.get("output_fields") or {}
    if not isinstance(fields, dict) or not missing:
        return []

    dropped: list[str] = []
    for name in missing:
        if name in fields:
            del fields[name]
            dropped.append(name)
    if not dropped:
        return []

    patches = [
        PatchRecord(
            iteration=0,
            kind=issue.kind,
            file=str(state.eval_spec_path),
            before_hash="",
            after_hash="",
            reason=issue.reason,
            diff_summary=f"Dropped {len(dropped)} output_fields the agent never returned: {dropped}.",
        )
    ]
    # Dropping fields invalidates the weight totals — rebalance now so
    # the next smoke run doesn't re-trip the weights check.
    rebal = apply_invalid_weights(
        state,
        IssueRecord(
            kind="invalid_weights",
            severity="fix",
            target="eval_spec",
            reason="Renormalised after dropping output_fields.",
        ),
    )
    patches.extend(rebal)
    return patches
