"""Stateless step library backing the ``overmind optimize-step`` CLI.

Each step is a small, JSON-in / JSON-out function that loads a
``SkillRunState`` from disk, reconstructs a (possibly partial)
``Optimizer`` instance, performs one phase of work, and writes the
updated state back. This lets a host coding agent (Cursor / Codex /
Claude Code) drive the iteration loop, parallel candidate fan-out, and
early stopping from a SKILL.md, while all heavy lifting (subprocess
isolation, scoring, regression gating) stays inside the existing
``overmind.optimize`` library.

Architecturally this is **additive** — ``overmind optimize`` keeps using
``Optimizer.run()`` as a single in-process driver. The skill path uses
the same ``Optimizer`` methods, just one phase per CLI invocation.
"""

from overmind.optimize.steps.state import SkillRunState

__all__ = ["SkillRunState"]
