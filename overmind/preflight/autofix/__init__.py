"""Autonomous, deterministic fixes for the failure kinds emitted by the classifier.

Every public function in this package is a pure-function transform with
the signature::

    apply(state: WorkingState, issue: IssueRecord) -> list[PatchRecord]

The runner in :mod:`overmind.preflight.runner` snapshots the affected
files, dispatches each issue to the registered handler, applies the
returned patches to disk, and records them.

A handler must:

- Be idempotent — running it twice on a healthy state produces no
  changes and emits no patches.
- Touch the **minimum** set of files necessary.
- Never read or write secrets, never call an LLM, and never modify the
  user's source tree (only the per-agent ``.overmind/agents/<name>/``
  workspace and the registered eval_spec / dataset files).
"""

from __future__ import annotations

from collections.abc import Callable

from overmind.preflight.autofix.dataset import apply_dataset_row_invalid
from overmind.preflight.autofix.entrypoint import apply_entrypoint_repair
from overmind.preflight.autofix.instrument import (
    apply_dep_missing,
    apply_import_error,
    apply_instrumentation_broken,
)
from overmind.preflight.autofix.metrics import apply_metric_broken
from overmind.preflight.autofix.schema import (
    apply_entrypoint_signature,
    apply_output_schema_mismatch,
)
from overmind.preflight.autofix.weights import apply_invalid_weights
from overmind.preflight.classifier import (
    KIND_DATASET_ROW_INVALID,
    KIND_DEP_MISSING,
    KIND_ENTRYPOINT_REPAIR,
    KIND_ENTRYPOINT_SIGNATURE,
    KIND_IMPORT_ERROR,
    KIND_INSTRUMENTATION_BROKEN,
    KIND_INVALID_WEIGHTS,
    KIND_METRIC_BROKEN,
    KIND_OUTPUT_SCHEMA_MISMATCH,
)
from overmind.preflight.state import IssueRecord, PatchRecord
from overmind.preflight.workspace import WorkingState

# Public dispatcher used by the runner.
HANDLERS: dict[str, Callable[[WorkingState, IssueRecord], list[PatchRecord]]] = {
    KIND_INVALID_WEIGHTS: apply_invalid_weights,
    KIND_ENTRYPOINT_REPAIR: apply_entrypoint_repair,
    KIND_OUTPUT_SCHEMA_MISMATCH: apply_output_schema_mismatch,
    KIND_ENTRYPOINT_SIGNATURE: apply_entrypoint_signature,
    KIND_METRIC_BROKEN: apply_metric_broken,
    KIND_DATASET_ROW_INVALID: apply_dataset_row_invalid,
    KIND_INSTRUMENTATION_BROKEN: apply_instrumentation_broken,
    KIND_IMPORT_ERROR: apply_import_error,
    KIND_DEP_MISSING: apply_dep_missing,
}


# Order matters: handlers earlier in this list run before later ones in
# the runner's per-iteration loop. Entrypoint repair is attempted *before*
# the schema-drop fallback so we prefer teaching the harness to return
# the right keys over silently dropping fields from the eval_spec.
HANDLER_ORDER: list[str] = [
    KIND_ENTRYPOINT_REPAIR,
    KIND_INVALID_WEIGHTS,
    KIND_METRIC_BROKEN,
    KIND_DATASET_ROW_INVALID,
    KIND_ENTRYPOINT_SIGNATURE,
    KIND_OUTPUT_SCHEMA_MISMATCH,
    KIND_INSTRUMENTATION_BROKEN,
    KIND_IMPORT_ERROR,
    KIND_DEP_MISSING,
]


def sort_issues(issues: list[IssueRecord]) -> list[IssueRecord]:
    """Stable-sort issues so the runner applies handlers in the right order."""
    rank = {kind: i for i, kind in enumerate(HANDLER_ORDER)}
    return sorted(issues, key=lambda i: rank.get(i.kind, len(HANDLER_ORDER)))


def fix(state: WorkingState, issue: IssueRecord) -> list[PatchRecord]:
    """Dispatch *issue* to the matching handler.

    Returns an empty list when no handler is registered (the runner
    treats that as "this issue is not autonomously fixable" and records
    it under ``issues_remaining``).
    """
    handler = HANDLERS.get(issue.kind)
    if handler is None:
        return []
    return handler(state, issue)


__all__ = ["HANDLERS", "HANDLER_ORDER", "fix", "sort_issues"]
