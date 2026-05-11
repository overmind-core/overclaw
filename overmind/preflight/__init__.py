"""End-to-end pipeline validation between dataset generation and ``overmind optimize``.

Preflight runs the agent against a tiny slice of the dataset, exercises
the full evaluator, classifies any failures into deterministic kinds,
and autonomously repairs every "plumbing" failure (eval-spec / dataset /
schema / instrumentation) until the pipeline is healthy enough that
``overmind optimize`` is guaranteed to start without infrastructure
errors.

Public surface (used by the CLI and skills):

- :class:`PreflightReport` — JSON-serialisable result envelope.
- :func:`run_preflight` — the convergence loop (run → classify → patch).
- :func:`scan_secrets`  — static scan for env vars the agent needs.
- :func:`set_secret`    — write a single env var into the per-agent ``.env``.
- :func:`load_report` / :func:`is_preflight_green` — used by the optimize
  gate to decide whether the pipeline is allowed to run.
"""

from overmind.preflight.runner import run_preflight
from overmind.preflight.secrets_scan import scan_secrets, set_secret
from overmind.preflight.state import (
    IssueRecord,
    PatchRecord,
    PreflightReport,
    is_preflight_green,
    load_report,
    preflight_report_path,
)

__all__ = [
    "IssueRecord",
    "PatchRecord",
    "PreflightReport",
    "is_preflight_green",
    "load_report",
    "preflight_report_path",
    "run_preflight",
    "scan_secrets",
    "set_secret",
]
