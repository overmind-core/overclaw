"""CLI daemon main loop — one command at a time (Overclaw)."""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

import overmind
from overmind.client import get_client, get_project_id, is_configured
from overmind.daemon.handlers import CommandHandler

logger = logging.getLogger(__name__)

POLL_INTERVAL = 2.0
HEARTBEAT_INTERVAL = 30.0
MAX_RETRIES = 5
BACKOFF_BASE = 2
BACKOFF_CAP = 60


def _daemon_project_root() -> Path:
    """Project root the daemon operates on: the directory it was launched in.

    Walk up from cwd to the nearest ancestor that has ``.overmind/``, but never
    cross the enclosing git repository boundary. Without the boundary, a stray
    ``~/.overmind`` hijacks a daemon launched inside an un-inited repo: the bundler
    roots at ``$HOME``, the agent's entry path resolves to a non-existent file, and
    ``upload_bundle`` dies with a cryptic ``KeyError`` on that path. Falls back to
    the git root, then cwd.
    """
    cwd = Path.cwd().resolve()
    boundary = cwd
    for parent in [cwd, *cwd.parents]:
        if (parent / ".git").exists():
            boundary = parent
            break
    for parent in [cwd, *cwd.parents]:
        if (parent / ".overmind").is_dir():
            return parent
        if parent == boundary:
            break
    return boundary


def _heartbeat_loop(client, session_id: str, stop: threading.Event) -> None:
    """Send periodic liveness beats while the daemon is idle or busy."""
    while not stop.wait(HEARTBEAT_INTERVAL):
        try:
            client.heartbeat(session_id)
        except Exception:
            logger.warning("Heartbeat failed for session %s", session_id, exc_info=True)

def _read_session_id_from_file() -> str | None:
    """Read session id from file."""
    session_id_file = Path.home() / ".overmind" / "session_id"
    if session_id_file.exists():
        return session_id_file.read_text().strip()
    return None

def write_session_id_to_file(session_id: str) -> None:
    """Write session id to file."""
    session_id_file = Path.home() / ".overmind" / "session_id"
    session_id_file.parent.mkdir(parents=True, exist_ok=True)
    session_id_file.write_text(session_id)

def run_daemon(
    *,
    agent_name: str | None = None,
    session_id: str | None = None,
) -> None:
    """Run the CLI daemon until interrupted."""
    if not is_configured():
        raise SystemExit("OVERMIND_API_URL and OVERMIND_API_KEY are required")

    client = get_client()
    if client is None:
        raise SystemExit("Could not create Overmind API client")

    project_id = get_project_id()
    if not project_id:
        raise SystemExit("OVERMIND_PROJECT_ID is required")

    root = _daemon_project_root()
    api_url = os.getenv("OVERMIND_API_URL") or "https://api.overmindlab.ai"
    logger.info(
        "Overmind daemon starting (v%s) — project=%s root=%s api=%s",
        overmind.__version__,
        project_id,
        root,
        api_url,
    )

    # try to read session id from file
    if not session_id:
        session_id = _read_session_id_from_file()

    if session_id:
        sid = session_id
        logger.info("Reusing CLI session %s", sid)
    else:
        logger.info("Registering CLI session on %s …", socket.gethostname())
        session = client.create_cli_session(
            project_id,
            hostname=socket.gethostname(),
            cli_version=overmind.__version__,
            agent_name=agent_name or "",
        )
        sid = str(session.id)
        logger.info("Registered CLI session %s", sid)
        write_session_id_to_file(sid)

    handlers = CommandHandler(client, root=root, agent_name=agent_name)
    logger.info(
        "Listening for commands on session %s (polling every %.0fs). Press Ctrl+C to stop.",
        sid,
        POLL_INTERVAL,
    )

    stop_heartbeat = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(client, sid, stop_heartbeat),
        name="overmind-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()

    try:
        while True:
            try:
                if not single_loop(client, sid, handlers):
                    time.sleep(POLL_INTERVAL)
            except KeyboardInterrupt:
                break
    finally:
        logger.info("Daemon stopping (session %s)", sid)
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=1.0)


def single_loop(client, session_id: str, handlers: CommandHandler) -> bool:
    """Process at most one command. Returns True if work was done."""
    try:
        command = client.fetch_next_command(session_id)
    except Exception:
        logger.warning("Failed to fetch next command", exc_info=True)
        return False

    if command is None:
        return False

    command_id = str(command.get("id", ""))
    logger.info("Executing command %s kind=%s", command_id, command.get("kind"))
    try:
        success, result, error = handlers.dispatch(command)
    except Exception as exc:
        success, result, error = False, {}, str(exc)
        logger.exception("Command %s crashed", command_id)

    if success:
        logger.info("Command %s completed", command_id)
    else:
        logger.warning("Command %s failed: %s", command_id, error or "(no detail)")

    try:
        _submit_result_with_retry(
            client,
            session_id,
            command_id,
            success=success,
            result=result,
            error=error,
        )
    except Exception:
        logger.exception("Could not report result for command %s", command_id)
    return True


def _submit_result_with_retry(
    client,
    session_id: str,
    command_id: str,
    *,
    success: bool,
    result: dict[str, Any],
    error: str,
) -> None:
    attempt = 0
    while True:
        try:
            client.submit_command_result(
                session_id,
                command_id,
                success=success,
                result=result,
                error=error,
            )
            return
        except Exception:
            attempt += 1
            if attempt >= MAX_RETRIES:
                raise
            delay = min(BACKOFF_CAP, BACKOFF_BASE**attempt)
            logger.warning(
                "Failed to report result for %s (attempt %d) — retrying in %ss",
                command_id,
                attempt,
                delay,
            )
            time.sleep(delay)
