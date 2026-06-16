"""Command handlers for the Overclaw CLI daemon."""

from __future__ import annotations

import logging
import tempfile
from functools import wraps
from pathlib import Path
from typing import Any

from overmind import SpanType, set_tag, start_span
from overmind import attrs as oc_attrs
from overmind.core.platform_agent import run_agent_from_platform
from overmind.core.registry import load_registry, project_root
from overmind.daemon import safety
from overmind.utils.code import AgentBundle

logger = logging.getLogger(__name__)


def _apply_diff(diff: str, root: str | Path, *, reverse: bool) -> tuple[int, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as tmp:
        tmp.write(diff)
        patch_path = tmp.name
    try:
        args = ["apply", "--whitespace=nowarn"]
        if reverse:
            args.append("-R")
        args.append(patch_path)
        proc = safety.run_git(*args, cwd=root)
        return proc.returncode, (proc.stderr or proc.stdout or "").strip()
    finally:
        Path(patch_path).unlink(missing_ok=True)


def _tag_command(kind: str, event: dict[str, Any], payload: dict[str, Any], run_id: str) -> None:
    """Stamp command + optimize-run correlation onto the active Overmind span."""
    set_tag(oc_attrs.COMMAND, kind)
    command_id = event.get("id") or payload.get("command_id")
    if command_id:
        set_tag(oc_attrs.JOB_ID, str(command_id))
    if run_id:
        set_tag(oc_attrs.WORKFLOW_RUN_ID, run_id)
    if payload.get("agent_id"):
        set_tag(oc_attrs.AGENT_ID, str(payload["agent_id"]))
    if payload.get("iteration") is not None:
        set_tag(oc_attrs.WORKFLOW_ITERATION, str(payload["iteration"]))
    if payload.get("subset"):
        set_tag(oc_attrs.RUN_KIND, str(payload["subset"]))
    if payload.get("candidate_index") is not None:
        set_tag(oc_attrs.OPTIMIZE_CANDIDATE_INDEX, str(payload["candidate_index"]))


def _observed_command(kind: str):
    """Wrap a command handler in an Overmind span — one span per executed command.

    Built on ``overmind.start_span`` / ``set_tag`` so every server-dispatched command is
    traced and correlated to its optimize run. Holding a recording span for the duration
    of the handler also lets ``run_agent`` propagate ``TRACEPARENT`` into the agent
    subprocess, so the agent's own ``agent.run`` spans re-parent under the command span.
    Status is derived from the handler's ``(ok, result, error)`` return because the
    handlers report failure by returning rather than raising.
    """

    def decorator(method):
        @wraps(method)
        def wrapper(self, event: dict[str, Any], payload: dict[str, Any]):
            with start_span(f"overmind.command.{kind}", span_type=SpanType.WORKFLOW):
                _tag_command(kind, event, payload, self._run_id(event, payload))
                ok, result, error = method(self, event, payload)
                set_tag(oc_attrs.STATUS, "success" if ok else "failed")
                if not ok and error:
                    set_tag(oc_attrs.ERROR_MESSAGE, str(error)[:500])
                return ok, result, error

        return wrapper

    return decorator


class CommandHandler:
    """Safety-enforced handler for server-dispatched optimize commands."""

    def __init__(
        self,
        client: Any = None,
        *,
        root: str | Path | None = None,
        agent_name: str | None = None,
    ) -> None:
        self.client = client
        self.agent_name = agent_name
        self.root = Path(root).resolve() if root else project_root()
        self._applied_patches: dict[str, str] = {}

    def dispatch(self, event: dict[str, Any]) -> tuple[bool, dict[str, Any], str]:
        kind = event.get("kind", "")
        handlers = {
            "upload_bundle": self.handle_upload_bundle,
            "apply_patch": self.handle_apply_patch,
            "reset": self.handle_reset,
            "run_agent": self.handle_run_agent,
        }
        handler = handlers.get(kind)
        if handler is None:
            return False, {}, f"Unknown command kind: {kind}"
        return handler(event, event.get("payload", {}))

    @staticmethod
    def _run_id(event: dict[str, Any], payload: dict[str, Any]) -> str:
        return str(
            event.get("optimize_run_id")
            or payload.get("optimize_run_id")
            or event.get("workflow_run_id")
            or payload.get("workflow_run_id")
            or ""
        )

    @_observed_command("upload_bundle")
    def handle_upload_bundle(
        self,
        event: dict[str, Any],
        payload: dict[str, Any],
    ) -> tuple[bool, dict[str, Any], str]:
        agent_path = (payload.get("agent_path") or "").strip()
        entrypoint_fn = (payload.get("entrypoint_fn") or "run").strip()
        if not agent_path:
            try:
                _name, agent_path, entrypoint_fn = self._agent_config(payload)
            except ValueError as exc:
                return False, {}, str(exc)

        entry = Path(agent_path)
        if not entry.is_absolute():
            entry = self.root / entry

        try:
            bundle = AgentBundle.from_entry_point(
                entry_path=str(entry),
                project_root=str(self.root),
                entrypoint_fn=entrypoint_fn,
                max_resolved_files=48,
                max_total_chars=80_000,
            )
            bundle_data = {
                "files": dict(bundle.original_files),
                "entrypoint_fn": entrypoint_fn,
                "entry_path": str(entry.relative_to(self.root)),
            }
            return True, {"bundle": bundle_data}, ""
        except Exception as exc:
            logger.exception("upload_bundle failed")
            return False, {}, str(exc)

    def _agent_config(self, payload: dict[str, Any]) -> tuple[str, str, str]:
        config = payload.get("config", {})
        name = self.agent_name or config.get("agent_name", "")
        if not name:
            raise ValueError("agent_name is required")
        registry = load_registry()
        if name not in registry:
            raise ValueError(f"Agent {name!r} not registered")
        entry = registry[name]
        return name, entry["file_path"], entry["fn_name"]

    @_observed_command("apply_patch")
    def handle_apply_patch(
        self,
        event: dict[str, Any],
        payload: dict[str, Any],
    ) -> tuple[bool, dict[str, Any], str]:
        diff = payload.get("diff", "")
        if not diff or not str(diff).strip():
            return False, {}, "Empty diff"
        try:
            rc, detail = _apply_diff(str(diff), self.root, reverse=False)
        except safety.UnsafeCommandError as exc:
            return False, {}, str(exc)
        if rc != 0:
            return False, {}, detail or "git apply failed"

        self._applied_patches[self._run_id(event, payload)] = str(diff)
        result: dict[str, Any] = {"applied": True}
        if payload.get("candidate_id") is not None:
            result["candidate_id"] = payload.get("candidate_id")
        return True, result, ""

    @_observed_command("reset")
    def handle_reset(
        self,
        event: dict[str, Any],
        payload: dict[str, Any],
    ) -> tuple[bool, dict[str, Any], str]:
        diff = self._applied_patches.pop(self._run_id(event, payload), "")
        if not diff.strip():
            return True, {"reset": True, "reason": "nothing_to_revert"}, ""
        try:
            rc, detail = _apply_diff(diff, self.root, reverse=True)
        except safety.UnsafeCommandError as exc:
            return False, {}, str(exc)
        if rc != 0:
            return False, {}, detail or "git apply -R failed"
        return True, {"reset": True}, ""

    @_observed_command("run_agent")
    def handle_run_agent(
        self,
        event: dict[str, Any],
        payload: dict[str, Any],
    ) -> tuple[bool, dict[str, Any], str]:
        try:
            correlation = dict(payload.get("trace_correlation") or {})
            run_id = self._run_id(event, payload)
            if run_id:
                correlation.setdefault(oc_attrs.WORKFLOW_RUN_ID, run_id)
            payload = {**payload, "trace_correlation": correlation}
            result = run_agent_from_platform(payload=payload, root=Path(self.root))
            return True, result, ""
        except Exception as exc:
            logger.exception("run_agent failed")
            return False, {}, str(exc)
