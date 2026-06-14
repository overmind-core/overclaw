"""Command handlers for the CLI daemon.

The server is the brain: it analyzes code, generates eval criteria, diagnoses
failures, and produces candidate git patches. The daemon is deliberately thin —
it only does the things that must happen on the machine where the agent's code
and dependencies live:

* ``upload_bundle`` — resolve the entrypoint's local import graph and upload it.
* ``run_agent``     — run the agent over a dataset (baseline / candidate / smoke).
* ``apply_patch``   — ``git apply`` a server-generated patch to the working tree.
* ``reset``         — reverse-apply the last patch to restore the baseline tree.
* ``cancel``        — no-op acknowledgement.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from overmind.core.registry import load_registry, project_root
from overmind.daemon.agent_runtime import AgentServerManager
from overmind.utils.code import AgentBundle

logger = logging.getLogger(__name__)


class CommandHandlers:
    """Execute server-dispatched commands locally."""

    def __init__(
        self,
        client: Any,
        *,
        agent_name: str | None = None,
    ) -> None:
        self.client = client
        self.agent_name = agent_name
        self.root = project_root()
        self._servers: dict[str, AgentServerManager] = {}
        # Last patch applied per workflow run, so ``reset`` can reverse it.
        self._applied_patches: dict[str, str] = {}

    def dispatch(self, event: dict[str, Any]) -> tuple[bool, dict[str, Any], str]:
        kind = event.get("kind", "")
        handlers = {
            "upload_bundle": self.handle_upload_bundle,
            "run_agent": self.handle_run_agent,
            "apply_patch": self.handle_apply_patch,
            "reset": self.handle_reset,
            "cancel": self.handle_cancel,
        }
        handler = handlers.get(kind)
        if handler is None:
            return False, {}, f"Unknown command kind: {kind}"
        return handler(event, event.get("payload", {}))

    # ── Helpers ────────────────────────────────────────────────────────────

    def _agent_config(self, payload: dict[str, Any]) -> tuple[str, str, str]:
        config = payload.get("config", {})
        agent_name = self.agent_name or config.get("agent_name", "")
        if not agent_name:
            raise ValueError("agent_name is required")
        registry = load_registry()
        if agent_name not in registry:
            raise ValueError(f"Agent {agent_name!r} not registered")
        entry = registry[agent_name]
        return agent_name, entry["file_path"], entry["fn_name"]

    def _relative_entry(self, agent_path: str) -> str:
        return str(Path(agent_path).resolve().relative_to(self.root))

    def _server_manager(self, agent_name: str) -> AgentServerManager:
        if agent_name not in self._servers:
            self._servers[agent_name] = AgentServerManager(agent_name)
        return self._servers[agent_name]

    # ── upload_bundle ────────────────────────────────────────────────────────

    def handle_upload_bundle(
        self,
        event: dict[str, Any],
        payload: dict[str, Any],
    ) -> tuple[bool, dict[str, Any], str]:
        run_id = event.get("workflow_run_id", "")
        try:
            _name, agent_path, fn = self._agent_config(payload)
            bundle = AgentBundle.from_entry_point(
                entry_path=agent_path,
                project_root=str(self.root),
                entrypoint_fn=fn,
                max_resolved_files=48,
                max_total_chars=80_000,
            )
            bundle_data = {
                "files": dict(bundle.original_files),
                "entrypoint_fn": fn,
                "entry_path": agent_path,
                "entry_file": self._relative_entry(agent_path),
                "project_root": str(self.root),
            }
            self.client.upload_workflow_artifact(
                run_id,
                kind="code_bundle",
                content=bundle_data,
                name="agent_bundle",
            )
            return True, {"bundle": bundle_data}, ""
        except Exception as exc:
            logger.exception("upload_bundle failed")
            return False, {}, str(exc)

    # ── run_agent ──────────────────────────────────────────────────────────

    def handle_run_agent(
        self,
        _event: dict[str, Any],
        payload: dict[str, Any],
    ) -> tuple[bool, dict[str, Any], str]:
        """Run the agent over the dataset in ``payload`` and return per-case results.

        Shared by the ``baseline_run`` / ``candidate_run`` / ``smoke_run`` client
        blocks — ``mode`` distinguishes them and the server scores the returned
        ``results`` (and ingests OTLP traces out of band).
        """
        from overmind.optimize.runner import AgentRunner, RunnerConfig

        mode = payload.get("mode", "baseline")
        try:
            agent_name, agent_path, fn = self._agent_config(payload)
            dataset = payload.get("dataset") or []

            server_mgr = self._server_manager(agent_name)
            if server_mgr.is_server_mode:
                if not server_mgr._proc:
                    server_mgr.start()
                results = [
                    self._invoke_server(server_mgr, i, case)
                    for i, case in enumerate(dataset)
                ]
                return True, {"results": results, "mode": mode, "count": len(results)}, ""

            runner = AgentRunner(
                agent_dir=self.root,
                entry_file=self._relative_entry(agent_path),
                entrypoint_fn=fn,
                config=RunnerConfig(timeout=300),
                env_dir=self.root,
            )
            runner.ensure_environment()
            results = []
            for i, case in enumerate(dataset):
                inp = case.get("input", case) if isinstance(case, dict) else {}
                run_result = runner.run(inp if isinstance(inp, dict) else {})
                results.append({
                    "index": i,
                    "input": inp,
                    "output": run_result.data,
                    "success": run_result.success,
                    "passed": run_result.success,
                    "error": run_result.error,
                })
            runner.cleanup()
            return True, {"results": results, "mode": mode, "count": len(results)}, ""
        except Exception as exc:
            logger.exception("run_agent failed (mode=%s)", mode)
            return False, {}, str(exc)

    @staticmethod
    def _invoke_server(
        server_mgr: AgentServerManager,
        index: int,
        case: Any,
    ) -> dict[str, Any]:
        inp = case.get("input", case) if isinstance(case, dict) else {}
        inp = inp if isinstance(inp, dict) else {}
        try:
            output = server_mgr.invoke(inp)
            return {
                "index": index,
                "input": inp,
                "output": output,
                "success": True,
                "passed": True,
                "error": "",
            }
        except Exception as exc:
            return {
                "index": index,
                "input": inp,
                "output": {},
                "success": False,
                "passed": False,
                "error": str(exc),
            }

    # ── apply_patch / reset ────────────────────────────────────────────────

    def handle_apply_patch(
        self,
        event: dict[str, Any],
        payload: dict[str, Any],
    ) -> tuple[bool, dict[str, Any], str]:
        run_id = event.get("workflow_run_id", "")
        try:
            diff = payload.get("diff", "")
            if not diff or not str(diff).strip():
                return True, {"applied": False, "reason": "empty_diff"}, ""

            rc, detail = self._git_apply(str(diff), reverse=False)
            if rc != 0:
                return False, {}, f"git apply failed: {detail}"

            self._applied_patches[run_id] = str(diff)

            agent_name, _, _ = self._agent_config(payload)
            server_mgr = self._server_manager(agent_name)
            if server_mgr.is_server_mode:
                server_mgr.restart()
            return True, {"applied": True, "restarted_server": server_mgr.is_server_mode}, ""
        except Exception as exc:
            logger.exception("apply_patch failed")
            return False, {}, str(exc)

    def handle_reset(
        self,
        event: dict[str, Any],
        payload: dict[str, Any],
    ) -> tuple[bool, dict[str, Any], str]:
        """Reverse-apply the last patch so the working tree returns to baseline."""
        run_id = event.get("workflow_run_id", "")
        try:
            diff = self._applied_patches.pop(run_id, "")
            if not diff.strip():
                return True, {"reset": True, "reason": "nothing_to_revert"}, ""

            rc, detail = self._git_apply(diff, reverse=True)
            if rc != 0:
                return False, {}, f"git apply -R failed: {detail}"

            agent_name = self.agent_name or payload.get("config", {}).get("agent_name", "")
            if agent_name and agent_name in self._servers:
                server_mgr = self._servers[agent_name]
                if server_mgr.is_server_mode:
                    server_mgr.restart()
            return True, {"reset": True}, ""
        except Exception as exc:
            logger.exception("reset failed")
            return False, {}, str(exc)

    def _git_apply(self, diff: str, *, reverse: bool) -> tuple[int, str]:
        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as tmp:
            tmp.write(diff)
            patch_path = tmp.name
        try:
            args = ["git", "apply", "--whitespace=nowarn"]
            if reverse:
                args.append("-R")
            args.append(patch_path)
            proc = subprocess.run(
                args,
                cwd=self.root,
                capture_output=True,
                text=True,
            )
            return proc.returncode, (proc.stderr or proc.stdout or "").strip()
        finally:
            Path(patch_path).unlink(missing_ok=True)

    # ── cancel ───────────────────────────────────────────────────────────────

    def handle_cancel(
        self,
        event: dict[str, Any],
        _payload: dict[str, Any],
    ) -> tuple[bool, dict[str, Any], str]:
        run_id = event.get("workflow_run_id", "")
        self._applied_patches.pop(run_id, None)
        for server_mgr in self._servers.values():
            server_mgr.stop()
        return True, {"cancelled": True}, ""
