"""Local execution support for server-orchestrated optimization.

The optimization "brain" (analysis, eval-criteria, diagnosis, candidate
generation, scoring) now lives on the Overmind server. What remains here is
the minimal machinery the CLI daemon needs to run an agent locally:

* :mod:`overmind.optimize.runner` — language-agnostic, subprocess-isolated
  agent executor used by the daemon's ``run_agent`` handler and
  ``overmind agent validate``.
* :mod:`overmind.optimize.shadow_runtime` — record/replay + simulation
  bootstrap injected into the runner's subprocess wrapper.
* :mod:`overmind.optimize.data` — dataset loading / field normalization
  helpers shared by ``overmind agent validate``.
"""
