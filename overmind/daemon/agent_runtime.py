"""Per-agent runtime configuration and long-running server management."""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import tomlkit

from overmind.core.constants import OVERMIND_DIR_NAME
from overmind.core.registry import project_root

logger = logging.getLogger(__name__)


@dataclass
class AgentRuntimeConfig:
    mode: str = "entrypoint"
    start_command: str = ""
    health_url: str = ""
    input_url: str = ""
    cwd: str = ""


def _runtime_path(agent_name: str) -> Path:
    return project_root() / OVERMIND_DIR_NAME / "agents" / agent_name / "runtime.toml"


def load_runtime_config(agent_name: str) -> AgentRuntimeConfig:
    """Load optional runtime config for an agent (defaults to subprocess entrypoint mode)."""
    path = _runtime_path(agent_name)
    if not path.is_file():
        return AgentRuntimeConfig()
    doc = tomlkit.loads(path.read_text(encoding="utf-8"))
    return AgentRuntimeConfig(
        mode=str(doc.get("mode") or "entrypoint"),
        start_command=str(doc.get("start_command") or ""),
        health_url=str(doc.get("health_url") or ""),
        input_url=str(doc.get("input_url") or ""),
        cwd=str(doc.get("cwd") or ""),
    )


def save_runtime_config(agent_name: str, config: AgentRuntimeConfig) -> None:
    path = _runtime_path(agent_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = tomlkit.document()
    doc["mode"] = config.mode
    if config.start_command:
        doc["start_command"] = config.start_command
    if config.health_url:
        doc["health_url"] = config.health_url
    if config.input_url:
        doc["input_url"] = config.input_url
    if config.cwd:
        doc["cwd"] = config.cwd
    path.write_text(tomlkit.dumps(doc), encoding="utf-8")


class AgentServerManager:
    """Start/stop/restart a long-running agent HTTP server."""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self.config = load_runtime_config(agent_name)
        self.root = project_root()
        self._proc: subprocess.Popen | None = None

    @property
    def is_server_mode(self) -> bool:
        return self.config.mode == "server" and bool(self.config.start_command)

    def start(self) -> None:
        if not self.is_server_mode:
            return
        self.stop()
        cwd = Path(self.config.cwd or self.root)
        self._proc = subprocess.Popen(
            self.config.start_command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        self._wait_healthy(timeout=60)

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            else:
                self._proc.terminate()
            self._proc.wait(timeout=15)
        except Exception:
            with contextlib.suppress(Exception):
                self._proc.kill()
        self._proc = None

    def restart(self) -> None:
        self.stop()
        self.start()

    def _wait_healthy(self, *, timeout: int) -> None:
        if not self.config.health_url:
            time.sleep(2)
            return
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = httpx.get(self.config.health_url, timeout=3.0)
                if resp.status_code < 500:
                    return
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError(f"Agent server did not become healthy: {self.config.health_url}")

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.config.input_url:
            raise RuntimeError("input_url is required for server-mode agents")
        resp = httpx.post(self.config.input_url, json=payload, timeout=300.0)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {"output": data}
