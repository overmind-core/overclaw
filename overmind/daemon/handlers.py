"""Command handlers — execute one server-issued primitive against the repo.

Each handler returns ``(success, result, error)``. ``result`` is the JSON the
server stores on the command (and folds into scoring/diagnosis); ``error`` is a
short human-readable failure string that drives the server's retry logic.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

from overmind.daemon import safety

logger = logging.getLogger("overmind.daemon.handlers")

_MAX_TAIL = 2000


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _run_trace(payload: dict):
    """Best-effort span around a server-driven agent run.

    Stamps the optimization context (run / agent / candidate / iteration /
    datapoint / project ids + ``trace_type``) so the backend can correlate the
    agent's trace to the run and candidate that produced it. Returns a no-op
    context manager when the SDK isn't initialized, so an offline daemon never
    fails a command over telemetry.
    """
    try:
        from overmind import attrs, tracing

        tracing.get_tracer()  # raises if the SDK isn't initialized
    except Exception:
        return nullcontext()

    candidate_index = _as_int(payload.get("candidate_index"), -1)
    trace_type = payload.get("trace_type") or ("original" if candidate_index < 0 else "replay")
    return tracing.start_child_span(
        "overmind.optimize.run_command",
        span_type=tracing.SpanType.ENTRY_POINT,
        attributes={
            attrs.OPTIMIZE_RUN_ID: str(payload.get("run_id") or ""),
            attrs.AGENT_ID: str(payload.get("agent_id") or ""),
            attrs.OPTIMIZE_PROJECT_ID: str(payload.get("project_id") or ""),
            attrs.OPTIMIZE_ITERATION: _as_int(payload.get("iteration"), -1),
            attrs.OPTIMIZE_CANDIDATE_INDEX: candidate_index,
            attrs.OPTIMIZE_DATAPOINT_INDEX: _as_int(payload.get("datapoint_index"), -1),
            attrs.OPTIMIZE_TRACE_TYPE: trace_type,
        },
    )


@dataclass
class HandlerContext:
    """Per-daemon execution context: where the repo is + a runner cache."""

    project_root: Path
    repo_root: Path
    agent_name: str = ""
    _runners: dict = field(default_factory=dict)
    # The branch the user was on when we first auto-stashed their uncommitted work,
    # so the final cleanup can put them back exactly where they started (req 12).
    _autostash_branch: str | None = None

    @classmethod
    def create(cls, *, agent_name: str = "") -> HandlerContext:
        from overmind.core.registry import project_root

        root = project_root()
        try:
            top = safety.run_git(root, ["rev-parse", "--show-toplevel"]).strip()
            repo_root = Path(top) if top else root
        except Exception:
            repo_root = root
        return cls(project_root=root, repo_root=repo_root, agent_name=agent_name)

    def runner_for(self, agent_dir: Path, entry_file: str, fn_name: str):
        from overmind.optimize.runner import AgentRunner

        key = (str(agent_dir), entry_file, fn_name)
        runner = self._runners.get(key)
        if runner is None:
            runner = AgentRunner(agent_dir=agent_dir, entry_file=entry_file, entrypoint_fn=fn_name)
            self._runners[key] = runner
        return runner


def dispatch(command: dict, ctx: HandlerContext) -> tuple[bool, dict, str]:
    kind = command.get("kind", "")
    payload = command.get("payload") or {}
    handler = _HANDLERS.get(kind)
    if handler is None:
        return False, {}, f"unknown command kind: {kind!r}"
    try:
        return handler(payload, ctx)
    except Exception as exc:
        logger.exception("command handler failed kind=%s", kind)
        return False, {}, str(exc)[:_MAX_TAIL]


def _resolve_entry(payload: dict, ctx: HandlerContext) -> tuple[Path, str, str]:
    """Resolve a command to ``(agent_dir, entry_file, fn_name)``.

    Prefers the server-supplied ``agent_path`` when it exists locally, then
    falls back to the local registry (by synced agent id, then name).
    """
    fn = (payload.get("entrypoint_fn") or "").strip() or "run"
    if ":" in fn:
        fn = fn.rsplit(":", 1)[-1]

    agent_path = (payload.get("agent_path") or "").strip()
    if agent_path:
        p = Path(agent_path)
        if not p.is_absolute():
            p = (ctx.project_root / p).resolve()
        if p.is_file():
            return p.parent, p.name, fn

    from overmind.core.registry import load_registry

    registry = load_registry()
    agent_id = payload.get("agent_id")
    name = payload.get("agent_name") or ctx.agent_name
    entry = None
    if agent_id:
        entry = next((e for e in registry.values() if e.get("id") == str(agent_id)), None)
    if entry is None and name and name in registry:
        entry = registry[name]
    if entry and entry.get("file_path") and Path(entry["file_path"]).is_file():
        fp = Path(entry["file_path"])
        return fp.parent, fp.name, entry.get("fn_name") or fn

    raise RuntimeError(
        f"cannot resolve agent entrypoint (agent_path={agent_path!r}, agent_id={agent_id!r}); "
        "start the daemon from the agent's project root and register it with "
        "`overmind agent register <name> <module:function>`"
    )


def _upload_bundle(payload: dict, ctx: HandlerContext) -> tuple[bool, dict, str]:
    from overmind.core.registry import project_root_from_agent_file
    from overmind.utils.code import AgentBundle

    agent_dir, entry_file, fn = _resolve_entry(payload, ctx)
    entry_path = agent_dir / entry_file
    root = project_root_from_agent_file(entry_path) or ctx.project_root
    bundle = AgentBundle.from_entry_point(str(entry_path), str(root), fn)
    files = [{"path": rel, "content": src} for rel, src in bundle.original_files.items()]
    return True, {"bundle": {"entry_file": bundle.entry_file, "files": files}}, ""


def _span_trace_id(span) -> str:
    """32-char hex trace id of *span*, or "" when telemetry is disabled.

    This is the same id the OTLP exporter ships the agent's trace under, so the
    server can line each run up with the run / candidate that produced it
    (``baseline_trace_ids`` → a replay's ``original_trace_id``). Best-effort: a
    daemon with no SDK (offline) just reports no trace id and the run continues.
    """
    try:
        sctx = span.get_span_context()
        if sctx and sctx.is_valid:
            return format(sctx.trace_id, "032x")
    except Exception:
        pass
    return ""


def _run_command(payload: dict, ctx: HandlerContext) -> tuple[bool, dict, str]:
    agent_dir, entry_file, fn = _resolve_entry(payload, ctx)
    runner = ctx.runner_for(agent_dir, entry_file, fn)
    runner.ensure_environment()
    with _run_trace(payload) as span:
        out = runner.run(payload.get("input") or {})
        trace_id = _span_trace_id(span)
    if out.success:
        result = {"output": out.data, "stdout": (out.stdout or "")[-_MAX_TAIL:]}
    else:
        result = {"output": None, "stderr": (out.stderr or "")[-_MAX_TAIL:]}
    # Carry the trace id back even on failure so a failed replay stays correlatable.
    if trace_id:
        result["trace_id"] = trace_id
    if out.success:
        return True, result, ""
    error = out.error or (out.stderr or "")[-_MAX_TAIL:] or "agent run failed"
    return False, result, error[:_MAX_TAIL]


# Identity stamped onto the ephemeral candidate/winner commits, passed via env so
# we never need a non-allowlisted `git -c user.email=…`.
_COMMIT_ENV = {
    "GIT_AUTHOR_NAME": "Overmind Optimizer",
    "GIT_AUTHOR_EMAIL": "overmind-bot@users.noreply.github.com",
    "GIT_COMMITTER_NAME": "Overmind Optimizer",
    "GIT_COMMITTER_EMAIL": "overmind-bot@users.noreply.github.com",
}


# Tag our auto-stash so we only ever pop work *we* set aside, never a stash the
# user created themselves.
_AUTOSTASH_MSG = "overmind-autostash"


def _current_branch(repo_root: Path) -> str:
    try:
        return safety.run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    except Exception:
        return ""


def _is_dirty(repo_root: Path) -> bool:
    try:
        return bool(safety.run_git(repo_root, ["status", "--porcelain"]).strip())
    except Exception:
        return False


def _has_autostash(repo_root: Path) -> bool:
    try:
        return _AUTOSTASH_MSG in safety.run_git(repo_root, ["stash", "list"])
    except Exception:
        return False


def _ensure_user_stash(repo_root: Path, ctx: HandlerContext) -> None:
    """Set the user's uncommitted work aside once, before we move off their branch.

    Req 12: a dirty repo must not block the run. Rather than refuse to switch
    branches, we stash the user's changes (including untracked files) and record
    the branch they were on, so :func:`_restore_user_stash` can put everything
    back exactly as we found it when the run finishes.
    """
    if ctx._autostash_branch is not None:
        return  # already stashed for this run
    if not _is_dirty(repo_root):
        return
    origin = _current_branch(repo_root)
    safety.run_git(repo_root, ["stash", "push", "--include-untracked", "-m", _AUTOSTASH_MSG])
    ctx._autostash_branch = origin or ""
    logger.info(
        "auto-stashed your uncommitted changes (%s); they'll be restored when the run finishes",
        _AUTOSTASH_MSG,
    )


def _restore_user_stash(repo_root: Path, ctx: HandlerContext) -> None:
    """Return the user to their original branch and pop the auto-stash, if any."""
    branch = ctx._autostash_branch
    if branch is None:
        return
    if branch:
        _git_ok(repo_root, ["checkout", branch])
    if _has_autostash(repo_root):
        try:
            safety.run_git(repo_root, ["stash", "pop"])
        except Exception as exc:  # noqa: BLE001 — surface but don't crash cleanup
            logger.warning(
                "could not auto-restore your stashed changes: %s — run `git stash pop` to recover",
                exc,
            )
    ctx._autostash_branch = None


def _git_ok(repo_root: Path, args: list[str]) -> bool:
    """Run an allowlisted git command, swallowing failure (e.g. deleting a missing branch)."""
    try:
        safety.run_git(repo_root, args)
        return True
    except Exception:
        return False


def _sync_base(repo_root: Path, base_branch: str, base_sha: str) -> None:
    """Park (raise) unless the local *base_branch* is at the optimizer's *base_sha*.

    We never ``reset --hard`` the user's branch: if their local default has drifted
    from the commit the server cloned at, the candidate diffs may not apply, so we
    ask them to sync rather than silently rewriting their history.
    """
    if not base_sha or not base_branch:
        return

    def ref_sha() -> str:
        try:
            return safety.run_git(repo_root, ["rev-parse", base_branch]).strip()
        except Exception:
            return ""

    if ref_sha() == base_sha:
        return
    _git_ok(repo_root, ["fetch"])
    if ref_sha() == base_sha:
        return
    raise RuntimeError(
        f"your local '{base_branch}' is not at the commit the optimizer started from "
        f"({base_sha[:10]}); run `git checkout {base_branch} && git pull` to sync, then resume"
    )


def _branches_with_prefix(repo_root: Path, prefix: str) -> list[str]:
    out = safety.run_git(repo_root, ["branch", "--format=%(refname:short)"])
    return [line.strip() for line in out.splitlines() if line.strip().startswith(prefix)]


def _apply_patch(payload: dict, ctx: HandlerContext) -> tuple[bool, dict, str]:
    """Mirror a server branch: be on ``branch`` (off ``base_branch``), ``diff`` committed.

    ``reset_to_base`` recreates ``branch`` from ``base_branch`` (candidate / smoke
    fix); without it the diff is committed onto the current branch (winner
    accumulation). With neither branch field set we fall back to the legacy
    apply-to-working-tree behavior.
    """
    base = (payload.get("base_branch") or "").strip()
    branch = (payload.get("branch") or "").strip()
    diff = payload.get("diff") or ""
    repo = ctx.repo_root

    if not base and not branch:  # legacy working-tree apply
        if not diff.strip():
            return True, {"applied": False, "reason": "empty diff"}, ""
        _apply_or_raise(repo, diff)
        return True, {"applied": True}, ""

    # Tolerate a dirty working tree (req 12): stash the user's uncommitted work
    # before we touch branches instead of refusing to run.
    _ensure_user_stash(repo, ctx)
    _sync_base(repo, base, (payload.get("base_sha") or "").strip())

    target = branch or base
    if payload.get("reset_to_base") and base:
        safety.run_git(repo, ["checkout", "-B", target, base])
    else:
        safety.run_git(repo, ["checkout", target])

    applied = False
    if diff.strip():
        _apply_or_raise(repo, diff)
        safety.run_git(repo, ["add", "-A"])
        safety.run_git(repo, ["commit", "-m", "overmind: optimize"], env=_COMMIT_ENV)
        applied = True
    head = safety.run_git(repo, ["rev-parse", "HEAD"]).strip()
    return True, {"applied": applied, "branch": target, "head": head}, ""


def _apply_or_raise(repo_root: Path, diff: str) -> None:
    """``git apply`` the diff, raising a clear, actionable error if it doesn't land.

    This is the *one* failure req 12 cares about: a dirty tree is fine, but if the
    optimizer's patch can't be applied we stop and tell the user to resolve it."""
    try:
        safety.run_git(repo_root, ["apply", "--whitespace=nowarn", "-"], stdin=diff)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "the optimizer's patch did not apply to your working tree "
            f"({str(exc)[:300]}); resolve the conflict (or commit/stash local edits to the "
            "same files) and resume the run"
        ) from exc


def _reset(payload: dict, ctx: HandlerContext) -> tuple[bool, dict, str]:
    """Return to ``base_branch`` and drop ``branch`` / every ``cleanup_prefix*`` branch.

    With no branch fields we fall back to the legacy reverse-apply of ``diff``.
    """
    base = (payload.get("base_branch") or "").strip()
    branch = (payload.get("branch") or "").strip()
    cleanup_prefix = (payload.get("cleanup_prefix") or "").strip()
    repo = ctx.repo_root

    if not base and not branch and not cleanup_prefix:  # legacy reverse-apply
        diff = payload.get("diff") or ""
        if not diff.strip():
            return True, {"reset": False, "reason": "empty diff"}, ""
        safety.run_git(repo, ["apply", "-R", "--whitespace=nowarn", "-"], stdin=diff)
        return True, {"reset": True}, ""

    if base:
        safety.run_git(repo, ["checkout", "-f", base])  # -f discards agent artifacts
    deleted: list[str] = []
    targets = [branch] if branch else []
    if cleanup_prefix:
        targets += _branches_with_prefix(repo, cleanup_prefix)
    for name in targets:
        if name and name != base and _git_ok(repo, ["branch", "-D", name]):
            deleted.append(name)
    # End-of-run cleanup (cleanup_prefix set): return the user to the branch they
    # started on and restore the work we stashed at the start (req 12).
    if cleanup_prefix:
        _restore_user_stash(repo, ctx)
    return True, {"reset": True, "checked_out": base, "deleted": deleted}, ""


_HANDLERS = {
    "upload_bundle": _upload_bundle,
    "run_command": _run_command,
    "apply_patch": _apply_patch,
    "reset": _reset,
}
