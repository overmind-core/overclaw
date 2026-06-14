"""CLI daemon main loop — polling command runner.

The daemon registers a :class:`ClientSession` with the server, then polls the
``/api/cli/sessions/{id}/poll/`` endpoint every couple of seconds. Each poll
doubles as a liveness heartbeat and returns the commands the server wants run
on this machine. Commands are executed on a single background worker (handler
state — applied patches, agent servers — must not run concurrently) while the
main thread keeps polling, so a long-running command never starves the
heartbeat. The server keeps returning outstanding commands until a result is
submitted, so a dropped connection or a restarted server self-heals on the next
poll with no special reconnect logic.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import overmind
from overmind.client import get_client, get_project_id, is_configured
from overmind.daemon.handlers import CommandHandlers

logger = logging.getLogger(__name__)

POLL_INTERVAL = 2.0
MAX_RETRIES = 3


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

    if session_id:
        sid = session_id
    else:
        session = client.create_cli_session(
            project_id,
            hostname=socket.gethostname(),
            cli_version=overmind.__version__,
            agent_name=agent_name or "",
        )
        sid = str(session.id)
        logger.info("Registered CLI session %s", sid)

    handlers = CommandHandlers(client, agent_name=agent_name)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="overmind-cmd")
    lock = threading.Lock()
    in_flight: set[str] = set()
    completed: set[str] = set()

    def execute(command: dict[str, Any]) -> None:
        command_id = str(command.get("id", ""))
        logger.info("Running command %s kind=%s", command_id, command.get("kind"))
        try:
            success, result, error = handlers.dispatch(command)
            _submit_result_with_retry(
                client,
                command_id,
                success=success,
                result=result,
                error=error,
            )
        except Exception:
            logger.exception("Command %s crashed", command_id)
        finally:
            with lock:
                in_flight.discard(command_id)
                completed.add(command_id)

    logger.info("Polling for commands on session %s (every %.0fs)", sid, POLL_INTERVAL)
    while True:
        try:
            commands = client.poll_session(
                sid,
                agent_name=agent_name or "",
                cli_version=overmind.__version__,
            )
            for command in commands:
                command_id = str(command.get("id", ""))
                if not command_id:
                    continue
                with lock:
                    if command_id in in_flight or command_id in completed:
                        continue
                    in_flight.add(command_id)
                executor.submit(execute, command)
        except KeyboardInterrupt:
            break
        except Exception:
            logger.warning("Poll failed; retrying in %.0fs", POLL_INTERVAL, exc_info=True)

        try:
            time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            break

    executor.shutdown(wait=False, cancel_futures=True)


def _submit_result_with_retry(
    client,
    command_id: str,
    *,
    success: bool,
    result: dict[str, Any],
    error: str,
) -> None:
    for attempt in range(MAX_RETRIES):
        try:
            client.submit_command_result(
                command_id,
                success=success,
                result=result,
                error=error,
            )
            return
        except Exception:
            if attempt == MAX_RETRIES - 1:
                logger.exception("Failed to submit result for command %s", command_id)
            time.sleep(2**attempt)
