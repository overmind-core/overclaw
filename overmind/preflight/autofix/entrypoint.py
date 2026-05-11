"""LLM-driven repair of the registered Overmind entrypoint *harness*.

The harness is the thin file that ``/overmind-register-agent`` writes to
bridge dataset inputs ↔ the native agent's call signature ↔ the
``output_fields`` the eval_spec scores. It contains no business logic
and is invisible to ``overmind optimize`` (which mutates the native
agent code that the harness imports). That makes the harness pure
plumbing — a perfect target for autonomous preflight repair.

Scope guarantees enforced here:

- Only the harness file (``state.entrypoint_path``) may be modified.
- We snapshot it before invoking the coding agent and revert if the
  agent touched anything else, if the file ended up empty, or if it
  did not actually change.
- The repair budget is capped via ``state.max_entrypoint_repairs`` so
  a pathological loop cannot burn unbounded tokens.
- After the harness changes we re-instrument the per-agent copy so the
  next smoke iteration sees the updated source.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from overmind.preflight.state import IssueRecord, PatchRecord
from overmind.preflight.workspace import WorkingState

logger = logging.getLogger("overmind.preflight.autofix.entrypoint")

_DEFAULT_MODEL = "anthropic/claude-sonnet-4-20250514"
_MAX_STEPS = 12


def apply_entrypoint_repair(state: WorkingState, issue: IssueRecord) -> list[PatchRecord]:
    """Attempt a bounded LLM rewrite of the harness file.

    Returns ``[]`` (so the runner falls back to the next handler — e.g.
    ``apply_output_schema_mismatch``'s spec-drop) when:

    - The harness path is unknown (agent unresolved).
    - We have already used the per-run repair budget.
    - No coding-agent model is configured (no ``ANALYZER_MODEL`` and no
      provider key) — preflight refuses to silently no-op an LLM call.
    - The coding agent finishes without changing the file.
    - The coding agent edited any file other than the harness; we
      revert from the snapshot and report no patch.
    """
    entrypoint_path = state.entrypoint_path
    if entrypoint_path is None or not entrypoint_path.is_file():
        return []

    if state.entrypoint_repair_attempts >= state.max_entrypoint_repairs:
        logger.info(
            "Skipping entrypoint repair: budget exhausted (%d/%d)",
            state.entrypoint_repair_attempts,
            state.max_entrypoint_repairs,
        )
        return []

    model = os.environ.get("ANALYZER_MODEL") or _DEFAULT_MODEL
    if not _has_provider_credentials(model):
        logger.info("Skipping entrypoint repair: no provider credentials for %s", model)
        return []

    state.entrypoint_repair_attempts += 1

    snapshot_dir = Path(tempfile.mkdtemp(prefix="overmind_preflight_ep_"))
    snapshot_file = snapshot_dir / entrypoint_path.name
    snapshot_file.write_bytes(entrypoint_path.read_bytes())
    before_hash = state.file_hash(entrypoint_path)

    instruction = _build_instruction(state, issue, entrypoint_path)

    try:
        from overmind.coding_agent.agent import run as coding_agent_run

        coding_agent_run(
            instruction=instruction,
            model=model,
            cwd=str(entrypoint_path.parent),
            worktree=str(entrypoint_path.parent),
            extra_instructions=[
                "STRICT FILE SCOPE: you may ONLY edit the file "
                f"{entrypoint_path.name}. Do not create, delete, or "
                "modify any other file under any circumstance. Do not "
                "edit modules the harness imports — those contain "
                "native agent logic that a separate optimisation pass "
                "is responsible for.",
                "STRICT BEHAVIOURAL SCOPE: do not invent new business "
                "logic. Your only job is to make the harness "
                "(a) accept the dataset's input keys, "
                "(b) call the existing native agent function, and "
                "(c) return a dict containing every key the eval_spec "
                "scores. Prefer minimal, targeted edits.",
            ],
            max_steps=_MAX_STEPS,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Entrypoint repair raised: %s", exc)
        # Revert any partial mutation.
        entrypoint_path.write_bytes(snapshot_file.read_bytes())
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        return []

    other_files_touched = _detect_collateral_writes(snapshot_dir, state, entrypoint_path)
    if other_files_touched:
        logger.warning(
            "Entrypoint repair touched off-limits files (%s); reverting.",
            other_files_touched,
        )
        entrypoint_path.write_bytes(snapshot_file.read_bytes())
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        return []

    new_bytes = entrypoint_path.read_bytes() if entrypoint_path.is_file() else b""
    if not new_bytes.strip():
        logger.warning("Entrypoint repair emptied the file; reverting.")
        entrypoint_path.write_bytes(snapshot_file.read_bytes())
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        return []

    after_hash = state.file_hash(entrypoint_path)
    if after_hash == before_hash:
        logger.info("Entrypoint repair produced no changes.")
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        return []

    # smoke runs the instrumented *copy*, not the user source — so we
    # must mirror the rewritten harness across before re-instrumenting.
    _sync_to_instrumented_copy(state, entrypoint_path)
    shutil.rmtree(snapshot_dir, ignore_errors=True)

    return [
        PatchRecord(
            iteration=0,
            kind=issue.kind,
            file=str(entrypoint_path),
            before_hash=before_hash,
            after_hash=after_hash,
            reason=issue.reason,
            diff_summary=(
                "LLM-driven repair of the registered Overmind entrypoint "
                f"harness via {model} "
                f"(attempt {state.entrypoint_repair_attempts}/"
                f"{state.max_entrypoint_repairs})."
            ),
        )
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def _has_provider_credentials(model: str) -> bool:
    prefix = model.split("/", 1)[0].lower() if "/" in model else ""
    env_var = _PROVIDER_KEYS.get(prefix)
    if env_var:
        return bool(os.environ.get(env_var))
    # Unknown / custom router: assume yes if *any* recognised key is set.
    return any(os.environ.get(v) for v in _PROVIDER_KEYS.values())


def _sync_to_instrumented_copy(state: WorkingState, entrypoint_path: Path) -> None:
    """Mirror the rewritten harness into ``state.instrumented_dir``.

    The instrumented dir is a flat copy of the project root rooted at
    the file generated by ``instrument_agent_files``. We compute the
    rel-path from the project root and overwrite the matching file in
    the instrumented copy. The runner then triggers ``instrument_directory``
    so ``@observe()`` decorators get reapplied.
    """
    if not state.instrumented_dir or not state.instrumented_dir.is_dir():
        return
    try:
        from overmind.core.registry import project_root_from_agent_file

        proj_root = project_root_from_agent_file(str(entrypoint_path))
    except Exception:
        proj_root = None

    if proj_root and entrypoint_path.is_relative_to(proj_root):
        rel = entrypoint_path.relative_to(proj_root)
    else:
        rel = Path(entrypoint_path.name)

    target = state.instrumented_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(entrypoint_path.read_bytes())
    # Tell the runner to re-decorate before the next smoke pass.
    state.reinstrument_requests.add(str(rel))


def _detect_collateral_writes(snapshot_dir: Path, state: WorkingState, entrypoint_path: Path) -> list[str]:
    """The coding agent's worktree is the harness's parent directory.

    We only allow it to edit the harness itself. As a defence-in-depth
    check, scan the parent dir for files modified after the snapshot
    timestamp, excluding the harness.
    """
    snapshot_mtime = snapshot_dir.stat().st_mtime
    touched: list[str] = []
    for sibling in entrypoint_path.parent.iterdir():
        if sibling == entrypoint_path:
            continue
        if not sibling.is_file():
            continue
        try:
            if sibling.stat().st_mtime > snapshot_mtime:
                touched.append(sibling.name)
        except OSError:
            continue
    return touched


def _build_instruction(state: WorkingState, issue: IssueRecord, entrypoint_path: Path) -> str:
    """Compose the user instruction for the coding agent.

    Embeds the eval_spec contract, the failing case (when known), and
    the current harness source so the model has every piece of context
    it needs in one shot.
    """
    spec = state.eval_spec or {}
    input_schema = spec.get("input_schema") or {}
    output_fields = list((spec.get("output_fields") or {}).keys())

    sample_input: Any = None
    sample_expected: Any = None
    row_idx = issue.details.get("row_index")
    if isinstance(row_idx, int) and 0 <= row_idx < len(state.dataset):
        row = state.dataset[row_idx]
        if isinstance(row, dict):
            sample_input = row.get("input")
            sample_expected = row.get("expected_output") or row.get("expected")

    raw_error = issue.details.get("raw") or ""
    scored_but_missing = issue.details.get("scored_but_missing") or []
    actually_returned = issue.details.get("actually_returned") or []

    parts = [
        "You are repairing an Overmind agent entrypoint *harness* file.",
        f"File to edit (and ONLY file you may edit): {entrypoint_path}",
        "",
        "## What this file is",
        "The harness is the thin wrapper Overmind calls to invoke the",
        "native agent. It must accept the dataset's input keys, call",
        "the native agent that lives in another module, and return a",
        "dict containing every key the eval_spec scores. It contains no",
        "business logic.",
        "",
        "## Eval spec contract",
        f"Input schema (keys the harness must accept): {sorted(input_schema.keys())}",
        f"Output fields (keys the harness must return): {output_fields}",
    ]
    if scored_but_missing:
        parts += [
            f"Currently missing from harness output: {scored_but_missing}",
            f"Currently returned by harness: {actually_returned}",
        ]
    if sample_input is not None:
        parts += [
            "",
            "## Failing case",
            f"Input:    {sample_input!r}",
            f"Expected: {sample_expected!r}",
        ]
    if raw_error:
        parts += [
            "",
            "## Failure detail",
            "```",
            str(raw_error)[:1500],
            "```",
        ]
    parts += [
        "",
        "## Rules",
        "- Edit ONLY the harness file. Do not modify, create, or delete any other file.",
        "- Do not invent or change business logic in the underlying native agent.",
        "- Keep imports of the native agent unchanged unless the import path is itself broken.",
        "- The function signature must accept the input_schema keys as keyword arguments.",
        "- Return a Python dict with every output_fields key present.",
        "- Apply your edit using the available tools (Edit / Write). Do not just describe the change.",
        "",
        "Make the minimum change required, then stop.",
    ]
    return "\n".join(parts)
