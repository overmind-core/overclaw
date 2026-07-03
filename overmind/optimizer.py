#!/usr/bin/env python3
"""Overmind optimizer client a tiny, agent and codebase-agnostic command runner.

The optimizer FSM (experiment -> iterations -> candidates -> commands) now lives
server-side. This file is the *client*: it registers with the
backend, polls for queued optimizer commands, runs whatever shell the server
hands it (from the repo root — you launch the optimizer there — optionally
applying a candidate git diff first), and reports the result back. The server's
clone path is server-side only and is never used here, so the same binary
optimizes any repo.

Usage:
    curl -s https://static.overmindlab.ai/cli.py | OVERMIND_API_KEY=<api-key> \\
    OVERMIND_API_URL=http://localhost:8000 \\
    python - optimizer

Logging: INFO by default (startup, registration, every command + its outcome,
periodic idle heartbeat). Set ``OPTIMIZER_LOG_LEVEL=DEBUG`` to see the full
detail — request payloads, the exact shell, cwd/timeout, traceparent, candidate
diff apply/revert, and stdout/stderr tails. The API key is never logged.

Env:
    OVERMIND_API_URL              backend base url (default https://api.overmindlab.ai)
    OVERMIND_API_KEY              API key, sent as ``X-Api-Key`` (required)
    OVERMIND_CWD                  optional local override for the working dir;
                                  defaults to the current dir (run from the repo root)
    OPTIMIZER_POLL_INTERVAL        idle poll seconds (default 5)
    OPTIMIZER_HEARTBEAT_INTERVAL   idle "still alive" log seconds (default 60)
    OPTIMIZER_LOG_LEVEL            DEBUG/INFO/WARNING/ERROR (default INFO);
                                   falls back to LOG_LEVEL, then INFO

ponytail: the server can hand this client ARBITRARY shell, and we run it with
``shell=True`` (only a per-command timeout guards it). The production daemon
allowlists git-only commands (investment-team/overmind/daemon/safety.py); add a
command allowlist / sandbox here before pointing it at untrusted backends.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import secrets
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby

import psutil
import requests

API_URL = os.getenv("OVERMIND_API_URL", "https://api.overmindlab.ai").rstrip("/")
API_KEY = os.getenv("OVERMIND_API_KEY", "")
# Local working dir for git + command runs. We ALWAYS run from the repo root: the
# optimizer is launched there, so this defaults to the current directory. The
# server-provided path (``cmd["cwd"]`` = its own clone_path) is server-side only
# and must never be used to locate the repo on the client.
WORK_DIR = os.getenv("OVERMIND_CWD", "") or None
IDLE_INTERVAL = float(os.getenv("OPTIMIZER_POLL_INTERVAL", "5"))
HEARTBEAT_INTERVAL = float(os.getenv("OPTIMIZER_HEARTBEAT_INTERVAL", "60"))
BUSY_INTERVAL = 0.5  # poll fast while there is work to drain
MAX_BACKOFF = 30.0  # cap the exponential backoff on transport errors
OUTPUT_TAIL = 8000  # chars of stdout/stderr to report back
LOG_SNIPPET = 200  # chars of an error surfaced at INFO (full body rides DEBUG)

logger = logging.getLogger("optimizer.client")


def configure_logging(level_name: str | None = None) -> None:
    """Wire up root logging from ``level_name`` (default ``OPTIMIZER_LOG_LEVEL``/INFO).

    Without this no handler exists and INFO/DEBUG would be swallowed — so an
    operator could never "turn it up" to see what the client is doing.
    """
    level_name = (level_name or os.getenv("OPTIMIZER_LOG_LEVEL") or os.getenv("LOG_LEVEL") or "INFO").upper()
    level = logging.getLevelName(level_name)
    if not isinstance(level, int):  # unknown name -> safe default, but say so
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not isinstance(logging.getLevelName(level_name), int):
        logger.warning("unknown log level %r; defaulting to INFO", level_name)


class OptimizerAPI:
    """Thin HTTP transport for the three CLI endpoints (raw ``requests``)."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"X-Api-Key": api_key, "Content-Type": "application/json"})
        # NB: never log api_key — only confirm whether one was supplied.
        logger.debug(
            "OptimizerAPI ready: base_url=%s api_key=%s",
            base_url,
            "set" if api_key else "MISSING",
        )

    def register(self) -> str:
        uname = os.uname()
        memory = psutil.virtual_memory()
        payload = {
            "hostname": socket.gethostname(),
            "cli_version": "optimizer/0.1",
            "metadata": {
                "pid": os.getpid(),
                "cpu.count": os.cpu_count(),  # traditional API, usually logical
                "cpu.logical_cores": multiprocessing.cpu_count(),
                # New in psutil 5.4.0+: returns (physical, logical)
                "cpu.physical_cores": psutil.cpu_count(logical=False),
                "memory.total": getattr(memory, "total", None),
                "memory.available": getattr(memory, "available", None),
                "memory.used": getattr(memory, "used", None),
                "memory.free": getattr(memory, "free", None),
                "memory.active": getattr(memory, "active", None),
                "memory.inactive": getattr(memory, "inactive", None),
                "memory.wired": getattr(memory, "wired", None),
                # This is rough, but it tells if running inside a container/VM by looking for the cgroup file (Linux only).
                "containerized": os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"),
                "architecture": getattr(uname, "machine", None),
                "release": getattr(uname, "release", None),
                "python.version": sys.version,
            },
        }
        logger.debug("POST %s/api/cli/sessions/ payload=%s", self.base_url, payload)
        resp = self.session.post(f"{self.base_url}/api/cli/sessions/", json=payload, timeout=30)
        resp.raise_for_status()
        session_id = resp.json()["id"]
        logger.debug("register -> session_id=%s", session_id)
        return session_id

    def poll(self, session_id: str) -> list[dict]:
        logger.debug("POST .../sessions/%s/poll/ (heartbeat + lease)", session_id)
        resp = self.session.post(f"{self.base_url}/api/cli/sessions/{session_id}/poll/", json={}, timeout=30)
        resp.raise_for_status()
        commands = resp.json().get("commands", [])
        logger.debug("poll leased %d command(s): %s", len(commands), [c.get("id") for c in commands])
        return commands

    def submit_result(self, command_id: str, *, success: bool, result: dict, error: str) -> None:
        logger.debug(
            "POST .../commands/%s/result/ success=%s trace_id=%s",
            command_id,
            success,
            result.get("trace_id"),
        )
        resp = self.session.post(
            f"{self.base_url}/api/cli/commands/{command_id}/result/",
            json={"success": success, "result": result, "error": error},
            timeout=60,
        )
        resp.raise_for_status()
        logger.debug("result accepted for command=%s", command_id)


def _new_traceparent() -> tuple[str, str]:
    """Return ``(traceparent_header, trace_id)`` so the child's trace id is knowable.

    Best-effort W3C trace context with no OTel dependency on the client: the
    overmind SDK inside the agent process adopts ``TRACEPARENT`` if it is set, so
    we can report the 32-hex trace id we minted without parsing the child's logs.
    """
    trace_id = secrets.token_hex(16)  # 32 hex chars
    span_id = secrets.token_hex(8)  # 16 hex chars
    return f"00-{trace_id}-{span_id}-01", trace_id


def _git_apply(diff_text: str, cwd: str | None, *, reverse: bool = False) -> subprocess.CompletedProcess:
    args = ["git", "apply", "--whitespace=nowarn"]
    if reverse:
        args.append("-R")
    args.append("-")  # read the patch from stdin
    if diff_text and not diff_text.endswith("\n"):
        diff_text += "\n"
    return subprocess.run(args, input=diff_text, cwd=cwd, text=True, capture_output=True)


def _current_branch(cwd: str | None) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _setup_iteration_branch(iteration_id: str, patch: str, cwd: str | None) -> None:
    """Checkout a branch named after the iteration and apply the patch as a commit.

    If the branch already exists (daemon restart / retry), switch to it and skip
    the apply step — the commit is already there.
    """
    proc = subprocess.run(
        ["git", "checkout", "-b", iteration_id],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        # Branch already exists; just switch to it.
        subprocess.run(["git", "checkout", iteration_id], cwd=cwd, check=True)
        logger.debug("iteration %s: reusing existing branch", iteration_id)
        return

    if not patch:
        logger.debug("iteration %s: no patch — baseline branch ready", iteration_id)
        return

    apply_proc = _git_apply(patch, cwd)
    if apply_proc.returncode != 0:
        err = apply_proc.stderr.strip()
        raise RuntimeError(f"git apply failed for iteration {iteration_id}: {err}")

    subprocess.run(["git", "add", "-A"], cwd=cwd, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"apply patch for iteration {iteration_id}"],
        cwd=cwd,
        check=True,
    )
    logger.info("iteration %s: patch applied and committed", iteration_id)


def run_command(cmd: dict, cwd: str | None = None) -> tuple[bool, dict, str]:
    """Run one server command and capture its output.

    The patch has already been committed to the iteration branch by
    ``_setup_iteration_branch`` before this is called. We only verify we are
    on the right branch (sanity check), then run the shell command.
    """
    cmd_id = cmd.get("id", "?")
    cwd = cwd if cwd is not None else WORK_DIR
    command = cmd.get("command") or ""
    iteration_id = cmd.get("iteration_id", "")
    timeout = int(cmd.get("timeout") or 600)
    traceparent, trace_id = _new_traceparent()
    env = {**os.environ, "TRACEPARENT": traceparent}

    if not command:
        logger.error("command %s has empty command text; reporting failure", cmd_id)
        return False, {"trace_id": trace_id}, "empty command"

    # Sanity check: the branch must match the iteration we set up.
    if iteration_id:
        branch = _current_branch(cwd)
        if branch != iteration_id:
            msg = f"branch mismatch: on '{branch}' but expected '{iteration_id}'"
            logger.error("command %s: %s", cmd_id, msg)
            return False, {"trace_id": trace_id}, msg

    logger.info(
        "running command %s (datapoint #%s, trace=%s)",
        cmd_id,
        cmd.get("datapoint_index", "?"),
        trace_id,
    )
    logger.debug(
        "command %s: cwd=%s timeout=%ss traceparent=%s",
        cmd_id,
        cwd or os.getcwd(),
        timeout,
        traceparent,
    )
    logger.debug("command %s shell:\n%s", cmd_id, command)

    started = time.monotonic()
    try:
        proc = subprocess.run(command, shell=True, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        elapsed = time.monotonic() - started
        success = proc.returncode == 0
        result = {
            "output": (proc.stdout or "")[-OUTPUT_TAIL:],
            "stdout": (proc.stdout or "")[-OUTPUT_TAIL:],
            "exit_code": proc.returncode,
            "trace_id": trace_id,
        }
        error = "" if success else (proc.stderr or "")[-OUTPUT_TAIL:]
        if success:
            logger.info("command %s ok in %.1fs (exit=0)", cmd_id, elapsed)
        else:
            logger.error(
                "command %s failed in %.1fs (exit=%s): %s",
                cmd_id,
                elapsed,
                proc.returncode,
                (error or "(no stderr)")[:LOG_SNIPPET],
            )
        logger.debug("command %s stdout tail:\n%s", cmd_id, result["output"])
        if error:
            logger.debug("command %s stderr tail:\n%s", cmd_id, error)
        return success, result, error
    except subprocess.TimeoutExpired:
        logger.error("command %s timed out after %ss", cmd_id, timeout)
        return False, {"trace_id": trace_id, "exit_code": -1}, f"timed out after {timeout}s"
    except OSError as exc:  # failed to spawn (bad cwd, missing shell, etc.)
        logger.exception("command %s failed to spawn", cmd_id)
        return False, {"trace_id": trace_id}, str(exc)[:OUTPUT_TAIL]


def poll_once(api: OptimizerAPI, session_id: str, cwd: str | None = None) -> int:
    """Run all leased commands this tick, grouped by iteration, in parallel.

    For each iteration batch: checkout the iteration branch with the patch
    applied as a single commit, then run all commands for that iteration
    concurrently (up to 8 at a time).
    """
    cwd = cwd if cwd is not None else WORK_DIR
    commands = api.poll(session_id)
    if not commands:
        return 0

    # Group by iteration_id preserving server order (commands arrive ordered by created_at).
    sorted_cmds = sorted(commands, key=lambda c: c.get("iteration_id", ""))
    total = 0
    for iteration_id, batch_iter in groupby(sorted_cmds, key=lambda c: c.get("iteration_id", "")):
        batch = list(batch_iter)
        patch = batch[0].get("iteration_patch", "")

        if iteration_id:
            try:
                _setup_iteration_branch(iteration_id, patch, cwd)
            except Exception as exc:
                logger.error("iteration %s: branch setup failed: %s", iteration_id, exc)
                for cmd in batch:
                    api.submit_result(cmd["id"], success=False, result={}, error=f"branch setup failed: {exc}")
                total += len(batch)
                continue

        with ThreadPoolExecutor(max_workers=8) as pool:
            future_to_cmd = {pool.submit(run_command, cmd, cwd): cmd for cmd in batch}
            for future in as_completed(future_to_cmd):
                cmd = future_to_cmd[future]
                try:
                    success, result, error = future.result()
                except Exception as exc:
                    logger.exception("command %s raised unexpectedly", cmd.get("id"))
                    success, result, error = False, {}, str(exc)
                api.submit_result(cmd["id"], success=success, result=result, error=error)

        total += len(batch)
    return total


def _register_with_retry(api: OptimizerAPI, idle_interval: float = IDLE_INTERVAL) -> str:
    """Register, retrying with backoff so the daemon survives a backend that is
    not up yet (or restarting). Logs each attempt so a stuck startup is visible.
    """
    backoff = idle_interval
    attempt = 0
    while True:
        attempt += 1
        try:
            return api.register()
        except requests.RequestException as exc:
            logger.warning("registration attempt %d failed: %s; retrying in %.1fs", attempt, exc, backoff)
            logger.debug("registration error detail", exc_info=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)


def run_optimizer(
    *,
    api_url: str | None = None,
    api_key: str | None = None,
    cwd: str | None = None,
    poll_interval: float | None = None,
    heartbeat_interval: float | None = None,
) -> None:
    """Register with the backend and loop forever draining queued commands.

    All arguments fall back to the module-level env-derived defaults (``API_URL``,
    ``OVERMIND_API_KEY``, ``OVERMIND_CWD``, ...) so this still works unmodified as
    the standalone curl-installed script.
    """
    api_url = (api_url or API_URL).rstrip("/")
    api_key = api_key or API_KEY
    cwd = cwd if cwd is not None else WORK_DIR
    idle_interval = poll_interval if poll_interval is not None else IDLE_INTERVAL
    heartbeat_interval = heartbeat_interval if heartbeat_interval is not None else HEARTBEAT_INTERVAL

    if not api_key:
        logger.error("OVERMIND_API_KEY is required")
        raise SystemExit(2)

    logger.info(
        "optimizer starting: api=%s host=%s idle=%.1fs heartbeat=%.0fs",
        api_url,
        socket.gethostname(),
        idle_interval,
        heartbeat_interval,
    )
    api = OptimizerAPI(api_url, api_key)
    session_id = _register_with_retry(api, idle_interval)
    logger.info("registered: session=%s", session_id)

    backoff = idle_interval
    last_heartbeat = time.monotonic()
    while True:
        try:
            ran = poll_once(api, session_id, cwd)
            now = time.monotonic()
            if ran:
                logger.info("ran %d command(s) this tick", ran)
                last_heartbeat = now
            elif now - last_heartbeat >= heartbeat_interval:
                logger.info("idle — connected, waiting for commands")
                last_heartbeat = now
            backoff = BUSY_INTERVAL if ran else idle_interval
        except requests.RequestException as exc:
            logger.warning("transport error talking to backend: %s; backing off %.1fs", exc, backoff)
            logger.debug("transport error detail", exc_info=True)
            backoff = min(max(backoff, idle_interval) * 2, MAX_BACKOFF)
        except Exception:
            logger.exception("unexpected error in poll loop; backing off %.1fs", backoff)
            backoff = min(max(backoff, idle_interval) * 2, MAX_BACKOFF)
        time.sleep(backoff)


if __name__ == "__main__":
    configure_logging()
    mode = sys.argv[1] if len(sys.argv) >= 2 else ""
    if mode == "optimizer":
        run_optimizer()
    else:
        print(__doc__)
        raise SystemExit(2)
