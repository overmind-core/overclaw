"""Repair the instrumented copy when it can't import or hasn't been wired.

Three handlers live here because they all touch the same surface
(``.overmind/agents/<name>/instrumented/``):

- :func:`apply_instrumentation_broken` — re-runs ``instrument_directory``
  (idempotent thanks to ``is_instrumented`` short-circuits).
- :func:`apply_import_error`           — same fix; an import failure
  during smoke usually means a stale or partial copy.
- :func:`apply_dep_missing`            — appends the missing package to
  the instrumented copy's ``requirements.txt`` so the next subprocess
  install picks it up.
"""

from __future__ import annotations

from overmind.preflight.state import IssueRecord, PatchRecord
from overmind.preflight.workspace import WorkingState
from overmind.utils.instrument import instrument_directory


def _reinstrument(state: WorkingState, issue: IssueRecord, kind_label: str) -> list[PatchRecord]:
    if not state.instrumented_dir.is_dir():
        return []
    modified = instrument_directory(str(state.instrumented_dir))
    if not modified:
        return []
    return [
        PatchRecord(
            iteration=0,
            kind=issue.kind,
            file=str(state.instrumented_dir),
            before_hash="",
            after_hash="",
            reason=issue.reason,
            diff_summary=f"{kind_label}: re-instrumented {modified} file(s) under instrumented copy.",
        )
    ]


def apply_instrumentation_broken(state: WorkingState, issue: IssueRecord) -> list[PatchRecord]:
    return _reinstrument(state, issue, "instrumentation_broken")


def apply_import_error(state: WorkingState, issue: IssueRecord) -> list[PatchRecord]:
    return _reinstrument(state, issue, "import_error")


def apply_dep_missing(state: WorkingState, issue: IssueRecord) -> list[PatchRecord]:
    """Append the missing package to the instrumented requirements file.

    The runner.ensure_environment path picks the manifest up on the
    next iteration's smoke run, so this is enough to unblock most "I
    forgot to add ``litellm``" failures without touching the user's
    repo.  Idempotent — refuses to add the same line twice.
    """
    module = issue.details.get("module")
    if not isinstance(module, str) or not module:
        return []

    if not state.instrumented_dir.is_dir():
        return []

    req_path = state.instrumented_dir / "requirements.txt"
    existing_lines: list[str] = []
    if req_path.is_file():
        existing_lines = [ln.strip() for ln in req_path.read_text().splitlines()]

    # Map the import name to PyPI name when the runner already knows one.
    from overmind.optimize.runner import _IMPORT_TO_PYPI

    pkg = _IMPORT_TO_PYPI.get(module, module)
    if pkg in existing_lines or any(ln.startswith((pkg + "==", pkg + ">")) for ln in existing_lines):
        return []

    new_lines = existing_lines + [pkg]
    req_path.write_text("\n".join(new_lines).rstrip() + "\n")
    state.deps_to_add.add(pkg)

    return [
        PatchRecord(
            iteration=0,
            kind=issue.kind,
            file=str(req_path),
            before_hash="",
            after_hash="",
            reason=issue.reason,
            diff_summary=f"Added missing dependency '{pkg}' to instrumented requirements.txt.",
        )
    ]
