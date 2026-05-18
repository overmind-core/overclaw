"""Reusable pipeline building blocks for the optimizer.

This package is the destination for pieces of :mod:`overmind.optimize.optimizer`
that don't need any of the ``Optimizer`` instance state — pure functions that
can be tested in isolation and reused by either the in-process
``Optimizer.run()`` path or the step-driven CLI.

Submodules
----------
:mod:`overmind.optimize.pipeline.scoring`
    Code-complexity heuristics and the complexity-penalty math used by
    candidate scoring.

See the cleanup plan (Phase 5.1) for the full target structure.  This
package is intentionally small today; future extractions will land here
without changing the import surface for callers.
"""

from overmind.optimize.pipeline.scoring import (
    compute_complexity_penalty,
    count_conditional_branches,
    count_function_defs,
    detect_data_leakage,
    prompt_size,
)

__all__ = [
    "compute_complexity_penalty",
    "count_conditional_branches",
    "count_function_defs",
    "detect_data_leakage",
    "prompt_size",
]
