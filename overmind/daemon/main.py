"""Daemon entry point: register a session and poll for work.

``run_daemon`` is the foreground loop behind ``overmind start`` — it keeps a
session connected (each poll doubles as a heartbeat) and executes whatever the
server's orchestrator enqueues. ``poll_once`` is the single claim→execute→report
cycle, reused by the optimize coordinator so both share one code path.
"""

from __future__ import annotations

import logging
import socket
import threading
import time

from overmind.daemon import handlers
from overmind.daemon.api import DaemonAPI, resolve_base_url, resolve_token

logger = logging.getLogger("overmind.daemon")

POLL_INTERVAL_S = 2.0
_ERROR_BACKOFF_MAX_S = 15.0


def _cli_version() -> str:
    try:
        from importlib.metadata import version

        return version("overmind")
    except Exception:
        return ""


def poll_once(api: DaemonAPI, session_id: str, ctx: handlers.HandlerContext, *, agent_name: str = "") -> int:
    """Claim and execute one batch of commands. Returns the number handled."""
    resp = api.poll(session_id, agent_name=agent_name, cli_version=_cli_version())
    commands = resp.get("commands") or []
    for command in commands:
        success, result, error = handlers.dispatch(command, ctx)
        try:
            api.submit_result(command["id"], success=success, result=result, error=error)
        except Exception:
            logger.exception("failed to submit result for command %s", command.get("id"))
    return len(commands)


def _init_tracing(token: str) -> None:
    """Best-effort: route the daemon's per-command agent-run spans to the backend.

    Idempotent and failure-tolerant (no key / offline) — telemetry must never
    stop the daemon from executing work.
    """
    try:
        from overmind import tracing

        tracing.init(
            overmind_api_key=token,
            overmind_base_url=resolve_base_url(),
            service_name="overmind-daemon",
        )
    except Exception:
        logger.debug("daemon tracing init skipped", exc_info=True)


def run_daemon(agent_name: str = "", *, poll_interval: float = POLL_INTERVAL_S, stop_event: threading.Event | None = None) -> None:
    token = resolve_token()
    if not token:
        raise SystemExit("OVERMIND_API_KEY is not set. Run `overmind init` first.")

    _init_tracing(token)
    api = DaemonAPI(resolve_base_url(), token)
    ctx = handlers.HandlerContext.create(agent_name=agent_name)
    session = api.register_session(
        agent_name=agent_name, cli_version=_cli_version(), hostname=socket.gethostname()
    )
    session_id = session["id"]
    print(
        f"Overmind daemon connected (session {session_id[:8]}). "
        "Polling for optimization work — press Ctrl-C to stop."
    )

    backoff = poll_interval
    try:
        while stop_event is None or not stop_event.is_set():
            try:
                handled = poll_once(api, session_id, ctx, agent_name=agent_name)
                backoff = poll_interval
            except Exception:
                logger.exception("poll cycle failed; backing off")
                time.sleep(backoff)
                backoff = min(backoff * 2, _ERROR_BACKOFF_MAX_S)
                continue
            time.sleep(poll_interval if handled == 0 else 0.2)
    except KeyboardInterrupt:
        print("\nStopping Overmind daemon.")
    finally:
        api.close()
