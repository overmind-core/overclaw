"""`overmind start` (the slim daemon) must not drag in the local optimizer.

The CLI used to eagerly ``import overmind.commands.optimize_cmd`` at module top,
which pulls in ``overmind.optimize.optimizer`` (the ~150KB local optimization
loop) for *every* command — including ``start``, whose only job is to poll the
server and execute primitive commands. These tests pin the lazy-import boundary:
loading the CLI, building the arg parser, and importing the daemon entrypoint
must all stay clear of the heavy optimizer modules.
"""

from __future__ import annotations

import json
import subprocess
import sys

# Heavy modules the slim daemon must never import just to start.
HEAVY = {"overmind.optimize.optimizer", "overmind.commands.optimize_cmd"}


def _heavy_modules_after(snippet: str) -> set[str]:
    """Run ``snippet`` in a clean interpreter; return any HEAVY modules it loaded."""
    script = (
        f"{snippet}\n"
        "import sys, json\n"
        "print(json.dumps(sorted(m for m in sys.modules)))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    loaded = set(json.loads(proc.stdout.splitlines()[-1]))
    return HEAVY & loaded


def test_importing_cli_does_not_load_optimizer():
    leaked = _heavy_modules_after("import overmind.cli")
    assert not leaked, f"importing overmind.cli leaked: {sorted(leaked)}"


def test_building_parser_does_not_load_optimizer():
    leaked = _heavy_modules_after("import overmind.cli as c; c._build_parser()")
    assert not leaked, f"building the parser leaked: {sorted(leaked)}"


def test_daemon_entrypoint_import_is_slim():
    leaked = _heavy_modules_after("import overmind.daemon.main")
    assert not leaked, f"importing the daemon entrypoint leaked: {sorted(leaked)}"
