"""Deterministic classification of preflight smoke-run failures.

The classifier inspects the structured output of one
:class:`overmind.preflight.smoke.SmokeRunResult` and emits a list of
:class:`~overmind.preflight.state.IssueRecord` items.  No LLM calls.

Issue *kinds* are mapped to autofixers in
:mod:`overmind.preflight.autofix`.  Three kinds intentionally short-
circuit the loop:

- ``missing_secret``  → block, ask the user.
- ``runtime_crash``   → record but do not patch (that is optimize's job).
- ``quality``         → record and exit green; pipeline is healthy.

Everything else is autonomously fixable.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from overmind.preflight.state import IssueRecord

if TYPE_CHECKING:
    from overmind.preflight.smoke import SmokeRunResult


# ---------------------------------------------------------------------------
# Issue kinds (must match autofix dispatcher table)
# ---------------------------------------------------------------------------

KIND_MISSING_SECRET = "missing_secret"
KIND_IMPORT_ERROR = "import_error"
KIND_DEP_MISSING = "missing_dependency"
KIND_ENTRYPOINT_SIGNATURE = "entrypoint_signature"
KIND_RUNTIME_CRASH = "runtime_crash"
KIND_OUTPUT_SCHEMA_MISMATCH = "output_schema_mismatch"
KIND_METRIC_BROKEN = "metric_broken"
KIND_INVALID_WEIGHTS = "invalid_weights"
KIND_DATASET_ROW_INVALID = "dataset_row_invalid"
KIND_INSTRUMENTATION_BROKEN = "instrumentation_broken"
# LLM-driven repair of the registered Overmind entrypoint *harness* file.
# Distinct from KIND_RUNTIME_CRASH (which targets the native agent code and
# is left to overmind optimize): emitted only when the failure originates
# inside the harness or when the harness's normalisation layer is provably
# returning the wrong shape.
KIND_ENTRYPOINT_REPAIR = "entrypoint_repair"
KIND_QUALITY = "quality"


# Patterns we recognise in stderr / error strings
_AUTH_PATTERNS = re.compile(
    r"(?:"
    r"AuthenticationError"
    r"|InvalidAPIKey"
    r"|api[_ ]key"
    r"|401|403"
    r"|invalid_api_key"
    r"|permission[_ ]denied"
    r"|missing[_ ]api[_ ]key"
    r"|provider.*key.*missing"
    r"|api.+credential"
    r")",
    re.IGNORECASE,
)

# E.g. "OPENAI_API_KEY" or "ANTHROPIC_API_KEY" mentioned in the error.
_ENV_KEY_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,}_(?:API_KEY|TOKEN|SECRET|KEY))\b")

_IMPORT_RE = re.compile(
    r"(?:ModuleNotFoundError|ImportError)(?::|.*?)\s+(?:No module named\s+)?['\"]?([A-Za-z0-9_.]+)['\"]?",
    re.MULTILINE,
)

_TYPE_ERROR_KW_RE = re.compile(
    r"got\s+an\s+unexpected\s+keyword\s+argument\s+['\"]([^'\"]+)['\"]"
    r"|missing\s+\d+\s+required\s+(?:positional|keyword(?:-only)?)\s+arguments?:\s*([^\n]+)",
    re.IGNORECASE,
)


def _extract_missing_secrets(blob: str) -> list[str]:
    return sorted(set(_ENV_KEY_RE.findall(blob)))


def _looks_like_auth(blob: str) -> bool:
    return bool(_AUTH_PATTERNS.search(blob or ""))


def _extract_missing_module(blob: str) -> str | None:
    m = _IMPORT_RE.search(blob or "")
    return m.group(1) if m else None


def _extract_signature_problem(blob: str) -> dict[str, str] | None:
    m = _TYPE_ERROR_KW_RE.search(blob or "")
    if not m:
        return None
    if m.group(1):
        return {"reason": "unexpected_keyword", "param": m.group(1)}
    if m.group(2):
        return {"reason": "missing_required", "params": m.group(2).strip()}
    return None


def _row_invalid_issue(row_index: int, reason: str, details: dict) -> IssueRecord:
    return IssueRecord(
        kind=KIND_DATASET_ROW_INVALID,
        severity="fix",
        target="dataset",
        reason=reason,
        details={"row_index": row_index, **details},
    )


def _traceback_implicates_entrypoint(blob: str, entrypoint_path: str | None) -> bool:
    """Return True iff *blob* names the harness file in a traceback frame.

    We deliberately check both the absolute path (which is how
    AgentRunner reports it) and the bare basename, so the heuristic
    works whether the smoke subprocess preserved absolute paths or not.
    """
    if not entrypoint_path or not blob:
        return False
    if entrypoint_path in blob:
        return True
    name = entrypoint_path.rsplit("/", 1)[-1]
    return bool(name) and name in blob


def classify(
    result: SmokeRunResult,
    *,
    eval_spec: dict | None = None,
    entrypoint_path: str | None = None,
) -> list[IssueRecord]:
    """Return a deduplicated, severity-ranked list of issues from *result*.

    *entrypoint_path* — absolute path to the registered Overmind harness
    file. When provided, runtime crashes whose traceback names the
    harness are reclassified from ``runtime_crash`` (a quality-only
    signal owned by optimize) into ``entrypoint_repair`` (a fixable
    plumbing signal owned by preflight).
    """
    issues: list[IssueRecord] = []

    # ------------------------------------------------------------------
    # Pre-execution problems (env provisioning, importability)
    # ------------------------------------------------------------------
    if result.preflight_error:
        blob = result.preflight_error
        missing_secrets = _extract_missing_secrets(blob)
        if _looks_like_auth(blob) or missing_secrets:
            for key in missing_secrets:
                issues.append(
                    IssueRecord(
                        kind=KIND_MISSING_SECRET,
                        severity="block",
                        target="env",
                        reason=f"Provider rejected the request — {key} appears missing.",
                        details={"env_var": key, "raw": blob[:400]},
                    )
                )
            if not missing_secrets:
                issues.append(
                    IssueRecord(
                        kind=KIND_MISSING_SECRET,
                        severity="block",
                        target="env",
                        reason="Provider authentication failed (no specific env var detected).",
                        details={"raw": blob[:400]},
                    )
                )
        else:
            module = _extract_missing_module(blob)
            if module:
                issues.append(
                    IssueRecord(
                        kind=KIND_DEP_MISSING,
                        severity="fix",
                        target="instrumented",
                        reason=f"Missing dependency: {module}",
                        details={"module": module, "raw": blob[:400]},
                    )
                )
            else:
                issues.append(
                    IssueRecord(
                        kind=KIND_IMPORT_ERROR,
                        severity="fix",
                        target="instrumented",
                        reason="Failed to provision agent environment before any case ran.",
                        details={"raw": blob[:400]},
                    )
                )

    # ------------------------------------------------------------------
    # Per-case failures
    # ------------------------------------------------------------------
    for idx, case in enumerate(result.cases):
        if case.success and case.score is not None:
            continue
        if case.success and case.score is None:
            issues.append(
                IssueRecord(
                    kind=KIND_METRIC_BROKEN,
                    severity="fix",
                    target="eval_spec",
                    reason=f"Scorer raised on case {idx}: {(case.scorer_error or 'unknown')[:200]}",
                    details={"row_index": idx, "raw": (case.scorer_error or "")[:400]},
                )
            )
            continue

        err = case.error or ""

        # Authentication / missing secrets surface first — they're terminal.
        if _looks_like_auth(err):
            for key in _extract_missing_secrets(err):
                issues.append(
                    IssueRecord(
                        kind=KIND_MISSING_SECRET,
                        severity="block",
                        target="env",
                        reason=f"Provider rejected case {idx} — {key} appears missing.",
                        details={"env_var": key, "row_index": idx, "raw": err[:400]},
                    )
                )
            if not _extract_missing_secrets(err):
                issues.append(
                    IssueRecord(
                        kind=KIND_MISSING_SECRET,
                        severity="block",
                        target="env",
                        reason=f"Case {idx} failed authentication (no specific key matched).",
                        details={"row_index": idx, "raw": err[:400]},
                    )
                )
            continue

        module = _extract_missing_module(err)
        if module:
            issues.append(
                IssueRecord(
                    kind=KIND_DEP_MISSING,
                    severity="fix",
                    target="instrumented",
                    reason=f"Missing dependency surfaced on case {idx}: {module}",
                    details={"module": module, "row_index": idx, "raw": err[:400]},
                )
            )
            continue

        sig = _extract_signature_problem(err)
        if sig:
            issues.append(
                IssueRecord(
                    kind=KIND_ENTRYPOINT_SIGNATURE,
                    severity="fix",
                    target="eval_spec",
                    reason=f"Entrypoint signature mismatch on case {idx}: {sig['reason']}",
                    details={"row_index": idx, **sig, "raw": err[:400]},
                )
            )
            continue

        # Generic crash. If the traceback implicates the harness file we
        # registered as the Overmind entrypoint, treat it as plumbing and
        # let the LLM-driven entrypoint repair handler attempt a fix.
        # Otherwise, this is a bug inside the native agent code — leave
        # it to overmind optimize.
        if _traceback_implicates_entrypoint(err, entrypoint_path):
            issues.append(
                IssueRecord(
                    kind=KIND_ENTRYPOINT_REPAIR,
                    severity="fix",
                    target="entrypoint",
                    reason=f"Case {idx} crashed inside the entrypoint harness: {err[:200]}",
                    details={"row_index": idx, "raw": err[:1500]},
                )
            )
        else:
            issues.append(
                IssueRecord(
                    kind=KIND_RUNTIME_CRASH,
                    severity="quality",
                    target="agent",
                    reason=f"Case {idx} crashed: {err[:200]}",
                    details={"row_index": idx, "raw": err[:400]},
                )
            )

    # ------------------------------------------------------------------
    # Eval-spec hygiene (run regardless of per-case failures)
    # ------------------------------------------------------------------
    if eval_spec is not None:
        weight_issue = _check_weights(eval_spec)
        if weight_issue:
            issues.append(weight_issue)

    # ------------------------------------------------------------------
    # Output schema vs entrypoint contract
    # ------------------------------------------------------------------
    if eval_spec is not None and result.successful_outputs():
        mismatch = _check_output_schema(eval_spec, result.successful_outputs())
        if mismatch:
            # Try to teach the harness to return the missing keys *first*;
            # if the LLM repair can't, the schema-drop fallback below
            # still keeps the spec scorable.
            if entrypoint_path:
                issues.append(
                    IssueRecord(
                        kind=KIND_ENTRYPOINT_REPAIR,
                        severity="fix",
                        target="entrypoint",
                        reason=(
                            "Harness output is missing keys the eval_spec scores: "
                            f"{mismatch.details.get('scored_but_missing')}"
                        ),
                        details={
                            "scored_but_missing": mismatch.details.get("scored_but_missing"),
                            "actually_returned": mismatch.details.get("actually_returned"),
                            "hint": "schema_drop_fallback",
                        },
                    )
                )
            issues.append(mismatch)

    # ------------------------------------------------------------------
    # Instrumentation health
    # ------------------------------------------------------------------
    if result.cases and any(c.success for c in result.cases) and result.span_count == 0:
        issues.append(
            IssueRecord(
                kind=KIND_INSTRUMENTATION_BROKEN,
                severity="fix",
                target="instrumented",
                reason="Agent ran successfully but no @observe() spans were captured.",
                details={"hint": "instrument_directory may need a re-run, or all functions skipped."},
            )
        )

    # ------------------------------------------------------------------
    # Quality signal — pipeline runs but score is low
    # ------------------------------------------------------------------
    if result.baseline_score is not None and result.baseline_score == 0.0 and any(c.success for c in result.cases):
        issues.append(
            IssueRecord(
                kind=KIND_QUALITY,
                severity="quality",
                target="agent",
                reason="Pipeline runs end-to-end but baseline score is 0 — leave for optimize.",
                details={"baseline_score": 0.0},
            )
        )

    return _dedupe(issues)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_weights(eval_spec: dict) -> IssueRecord | None:
    """Detect weights that don't sum to ``total_points``."""
    total_declared = float(eval_spec.get("total_points", 100))
    fields = eval_spec.get("output_fields") or {}
    field_sum = sum(float((cfg or {}).get("weight", 0)) for cfg in fields.values())
    structure = float(eval_spec.get("structure_weight", 0))
    tools = float(eval_spec.get("tool_usage_weight", 0))
    judge = float(eval_spec.get("llm_judge_weight", 0))
    actual = field_sum + structure + tools + judge
    if abs(actual - total_declared) <= 1.0:
        return None
    return IssueRecord(
        kind=KIND_INVALID_WEIGHTS,
        severity="fix",
        target="eval_spec",
        reason=f"eval_spec weights sum to {actual:.1f} but total_points is {total_declared:.1f}.",
        details={"actual": actual, "expected": total_declared},
    )


def _check_output_schema(eval_spec: dict, outputs: list[dict]) -> IssueRecord | None:
    """Compare ``output_fields`` keys to what the agent actually returned.

    If the agent returns *fewer* keys than scored, we can drop the
    missing ones from the spec.  If it returns *more*, those are
    optional bonus fields — we don't mutate the spec for those.
    """
    spec_fields = set((eval_spec.get("output_fields") or {}).keys())
    if not spec_fields:
        return None
    seen: set[str] = set()
    for out in outputs:
        if isinstance(out, dict):
            seen.update(out.keys())
    missing_in_output = spec_fields - seen
    if not missing_in_output or not seen:
        return None
    return IssueRecord(
        kind=KIND_OUTPUT_SCHEMA_MISMATCH,
        severity="fix",
        target="eval_spec",
        reason=(
            f"eval_spec scores {sorted(missing_in_output)} but the agent never returned them. "
            "These fields will be dropped so the scorer can compute a finite total."
        ),
        details={
            "scored_but_missing": sorted(missing_in_output),
            "actually_returned": sorted(seen),
        },
    )


def _dedupe(issues: list[IssueRecord]) -> list[IssueRecord]:
    """Collapse identical issues so the loop doesn't retry the same fix forever."""
    seen: set[tuple] = set()
    out: list[IssueRecord] = []
    for issue in issues:
        key = (issue.kind, issue.target, issue.reason)
        if key in seen:
            continue
        seen.add(key)
        out.append(issue)
    return out


# ---------------------------------------------------------------------------
# Severity helpers used by the runner
# ---------------------------------------------------------------------------


def has_blockers(issues: list[IssueRecord]) -> bool:
    return any(i.severity == "block" for i in issues)


def has_fixable(issues: list[IssueRecord]) -> bool:
    return any(i.severity == "fix" for i in issues)


def missing_secret_keys(issues: list[IssueRecord]) -> list[str]:
    keys: list[str] = []
    for issue in issues:
        if issue.kind != KIND_MISSING_SECRET:
            continue
        env_var = issue.details.get("env_var")
        if isinstance(env_var, str) and env_var and env_var not in keys:
            keys.append(env_var)
    return keys
