"""Pure-function code-complexity heuristics used by the optimizer.

These helpers are state-free: every parameter is supplied explicitly, no
``Optimizer``-instance state is read.  That makes them safe to call from
either the in-process optimizer or the step-driven CLI, and easy to
exercise directly in tests without spinning up a full optimizer fixture.

Public functions
----------------
:func:`prompt_size`
    Length of the ``SYSTEM_PROMPT`` literal inside *code* (or ``0``).
:func:`count_conditional_branches`
    ``if`` / ``elif`` branches as a proxy for post-processing complexity.
:func:`count_function_defs`
    ``def`` / ``async def`` declarations.
:func:`detect_data_leakage`
    Count of training-set expected-output literals that appear in the
    candidate code but not the baseline.
:func:`compute_complexity_penalty`
    Aggregated penalty across prompt bloat, code growth, branch growth,
    branch-to-function ratio, and data leakage.
"""

from __future__ import annotations

import re
from typing import Any

# Tokens that look like expected-output literals but are actually domain-neutral
# vocabulary.  Skipped during leakage detection so the optimizer doesn't
# penalise a candidate for spelling ``"true"`` or ``"1"``.
_LEAKAGE_IGNORE = frozenset({
    "",
    "true",
    "false",
    "none",
    "null",
    "yes",
    "no",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "10",
    "100",
})


def prompt_size(code: str) -> int:
    """Return the character length of the ``SYSTEM_PROMPT`` literal in *code*.

    Returns ``0`` when the agent doesn't declare a top-level
    ``SYSTEM_PROMPT`` triple-quoted block.
    """
    m = re.search(
        r'SYSTEM_PROMPT\s*=\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')',
        code,
        re.DOTALL,
    )
    return len(m.group(1)) if m else 0


def count_conditional_branches(code: str) -> int:
    """Count ``if`` / ``elif`` branches as a proxy for post-processing complexity."""
    return sum(
        1 for line in code.splitlines() if line.strip().startswith(("if ", "elif ", "if(", "elif("))
    )


def count_function_defs(code: str) -> int:
    """Count top-level and nested ``def`` / ``async def`` declarations."""
    return sum(1 for line in code.splitlines() if line.strip().startswith(("def ", "async def ")))


def detect_data_leakage(
    candidate_code: str,
    baseline_code: str,
    train_set: list[dict],
    *,
    known_domain_values: set[str] | None = None,
) -> int:
    """Count expected-output literals that appear in *candidate_code* but not in
    *baseline_code*.

    Excludes:

    * tokens in :data:`_LEAKAGE_IGNORE` (numeric labels, ``true``/``false``, …)
    * any value in *known_domain_values* — typically the enum vocabulary
      pulled from the eval spec (these are legal domain references)
    * literals shorter than six characters
    """
    new_lines = set(candidate_code.splitlines()) - set(baseline_code.splitlines())
    new_code_text = "\n".join(new_lines)
    if not new_code_text.strip():
        return 0

    known = {v.lower().strip() for v in (known_domain_values or set())}

    leaked = 0
    seen: set[str] = set()
    for case in train_set:
        expected = case.get("expected_output", case.get("expected", {}))
        if not isinstance(expected, dict):
            continue
        for val in expected.values():
            if not isinstance(val, str):
                continue
            normalised = val.strip().lower()
            if (
                len(normalised) < 6
                or normalised in _LEAKAGE_IGNORE
                or normalised in known
                or normalised in seen
            ):
                continue
            if val in new_code_text:
                leaked += 1
                seen.add(normalised)
    return leaked


def compute_complexity_penalty(
    candidate_code: str,
    *,
    baseline_code: str | None,
    best_code: str,
    best_score: float,
    train_set: list[dict] | None = None,
    raw_score: float | None = None,
    max_code_growth_ratio: float = 2.5,
    known_domain_values: set[str] | None = None,
) -> float:
    """Penalize candidates with excessive prompt, code, or logic growth.

    Five dimensions (all use quadratic ramps so small overshoots get tiny
    penalties while large overshoots are still meaningful):

    1. ``SYSTEM_PROMPT`` bloat (vs *baseline_code* or *best_code*)
    2. Total code-size growth (vs *baseline_code*, size-adaptive threshold)
    3. New conditional branches (vs *baseline_code*, size-adaptive)
    4. Branch-to-function ratio — many new branches without new functions is a
       strong overfitting signal.
    5. Hardcoded expected-output literals (data leakage from *train_set*)

    When *raw_score* is supplied the total penalty is capped at 60% of the raw
    improvement over *best_score*, ensuring genuine improvements always yield
    at least partial net gain.
    """
    penalty = 0.0
    reference = baseline_code or best_code

    # 1. Prompt bloat (vs original baseline, threshold 2.0x).
    baseline_prompt = prompt_size(reference)
    cand_prompt = prompt_size(candidate_code)
    if baseline_prompt > 0:
        prompt_ratio = cand_prompt / baseline_prompt
        if prompt_ratio > 2.0:
            overshoot = prompt_ratio - 2.0
            penalty += min(3.0, overshoot**2 * 2.0)

    # 2. Total code growth (vs original baseline, size-adaptive).
    if baseline_code:
        baseline_lines = len(baseline_code.splitlines())
        candidate_lines = len(candidate_code.splitlines())
        if baseline_lines > 0:
            max_ratio = max_code_growth_ratio
            if baseline_lines < 150:
                max_ratio += 1.0
            elif baseline_lines < 300:
                max_ratio += 0.5
            code_ratio = candidate_lines / baseline_lines
            if code_ratio > max_ratio:
                overshoot = code_ratio - max_ratio
                penalty += min(5.0, overshoot**2 * 1.5)

    # 3. New conditional branches (vs original baseline, size-adaptive).
    new_branches = 0
    if baseline_code:
        baseline_branches = count_conditional_branches(baseline_code)
        candidate_branches = count_conditional_branches(candidate_code)
        new_branches = candidate_branches - baseline_branches
        branch_threshold = max(8, baseline_branches // 3)
        if new_branches > branch_threshold:
            overshoot = new_branches - branch_threshold
            penalty += min(4.0, overshoot**2 * 0.03)

    # 4. Conditional-to-structural ratio.
    if baseline_code and new_branches > 5:
        baseline_funcs = count_function_defs(baseline_code)
        candidate_funcs = count_function_defs(candidate_code)
        new_funcs = candidate_funcs - baseline_funcs
        if new_funcs <= 0:
            penalty += min(2.0, (new_branches - 5) * 0.15)

    # 5. Hardcoded expected-output literals from training data.
    if train_set and baseline_code:
        leakage = detect_data_leakage(
            candidate_code,
            baseline_code,
            train_set,
            known_domain_values=known_domain_values,
        )
        if leakage > 0:
            penalty += min(5.0, leakage * 1.5)

    # Cap penalty at 60% of the raw improvement so genuine gains always
    # produce at least partial net progress.
    if raw_score is not None and penalty > 0:
        raw_improvement = raw_score - best_score
        if raw_improvement > 0:
            max_allowed = raw_improvement * 0.6
            penalty = min(penalty, max_allowed)

    return penalty


# Keep mypy quiet about the unused Any import (used by callers via re-export).
_ = Any
