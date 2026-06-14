"""Subprocess lifecycle for the CLI daemon."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class DaemonProcess:
    """Manage a persistent ``overmind daemon`` child process."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(
        self,
        *,
        session_id: str,
        agent_name: str | None = None,
    ) -> subprocess.Popen:
        if self.running:
            return self._proc  # type: ignore[return-value]

        cmd = [
            sys.executable,
            "-m",
            "overmind",
            "daemon",
            "--session-id",
            session_id,
        ]
        if agent_name:
            cmd.extend(["--agent", agent_name])

        env = os.environ.copy()
        log_dir = Path.cwd() / ".overmind" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "daemon.log"

        logger.info("Starting daemon subprocess: %s", " ".join(cmd))
        with open(log_path, "a", encoding="utf-8") as log_file:
            self._proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(Path.cwd()),
            )
        time.sleep(0.5)
        if self._proc.poll() is not None:
            raise RuntimeError(f"Daemon exited immediately (code {self._proc.returncode}). See {log_path}")
        return self._proc

    def stop(self) -> None:
        if not self._proc:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
