"""Autonomous, deterministic fixes for the failure kinds emitted by the classifier.

Every public handler has the signature::

    apply(state: WorkingState, issue: IssueRecord) -> list[PatchRecord]

Handlers must be:
- Idempotent — running twice on a healthy state produces no changes.
- Minimal — touch only what's necessary to make the next smoke run pass.
- Pure Python — no LLM calls, no subprocess spawning.
"""

from __future__ import annotations

from collections.abc import Callable

from overmind.preflight.autofix.dataset import apply_dataset_row_invalid
from overmind.preflight.autofix.instrument import (
    apply_dep_missing,
    apply_import_error,
    apply_instrumentation_broken,
)
from overmind.preflight.autofix.metrics import apply_consistency_rules_invalid, apply_metric_broken
from overmind.preflight.autofix.schema import (
    apply_entrypoint_signature,
    apply_output_schema_mismatch,
)
from overmind.preflight.autofix.weights import apply_invalid_weights
from overmind.preflight.classifier import (
    KIND_CONSISTENCY_RULES_INVALID,
    KIND_DATASET_ROW_INVALID,
    KIND_DEP_MISSING,
    KIND_ENTRYPOINT_SIGNATURE,
    KIND_IMPORT_ERROR,
    KIND_INSTRUMENTATION_BROKEN,
    KIND_INVALID_WEIGHTS,
    KIND_METRIC_BROKEN,
    KIND_OUTPUT_SCHEMA_MISMATCH,
)
from overmind.preflight.state import IssueRecord, PatchRecord
from overmind.preflight.workspace import WorkingState

# Handler dispatch table.
HANDLERS: dict[str, Callable[[WorkingState, IssueRecord], list[PatchRecord]]] = {
    KIND_INVALID_WEIGHTS: apply_invalid_weights,
    KIND_CONSISTENCY_RULES_INVALID: apply_consistency_rules_invalid,
    KIND_OUTPUT_SCHEMA_MISMATCH: apply_output_schema_mismatch,
    KIND_ENTRYPOINT_SIGNATURE: apply_entrypoint_signature,
    KIND_METRIC_BROKEN: apply_metric_broken,
    KIND_DATASET_ROW_INVALID: apply_dataset_row_invalid,
    KIND_INSTRUMENTATION_BROKEN: apply_instrumentation_broken,
    KIND_IMPORT_ERROR: apply_import_error,
    KIND_DEP_MISSING: apply_dep_missing,
}

# Handlers run in this order within a fix pass.
HANDLER_ORDER: list[str] = [
    KIND_INVALID_WEIGHTS,
    KIND_CONSISTENCY_RULES_INVALID,
    KIND_METRIC_BROKEN,
    KIND_DATASET_ROW_INVALID,
    KIND_ENTRYPOINT_SIGNATURE,
    KIND_OUTPUT_SCHEMA_MISMATCH,
    KIND_INSTRUMENTATION_BROKEN,
    KIND_IMPORT_ERROR,
    KIND_DEP_MISSING,
]


def sort_issues(issues: list[IssueRecord]) -> list[IssueRecord]:
    """Stable-sort issues so handlers run in the intended order."""
    rank = {kind: i for i, kind in enumerate(HANDLER_ORDER)}
    return sorted(issues, key=lambda i: rank.get(i.kind, len(HANDLER_ORDER)))


def fix(state: WorkingState, issue: IssueRecord) -> list[PatchRecord]:
    """Dispatch *issue* to the matching handler.

    Returns an empty list when no handler is registered for this kind.
    """
    handler = HANDLERS.get(issue.kind)
    if handler is None:
        return []
    return handler(state, issue)


__all__ = ["HANDLERS", "HANDLER_ORDER", "fix", "sort_issues"]
