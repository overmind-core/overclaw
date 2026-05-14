"""Build an :class:`AgentBundle` from an optimize :class:`Config`.

This small factory decouples the optimizer from the details of how the
code bundle is assembled so the logic can be unit-tested in isolation.
Given a loaded :class:`Config` (which carries the agent path, entrypoint
function, and scope globs resolved from ``eval_spec.json``), produce the
:class:`AgentBundle` used by both baseline/candidate evaluation and the
analyzer prompt.
"""

from __future__ import annotations

import logging
from pathlib import Path

from overmind.core.registry import project_root, project_root_from_agent_file
from overmind.optimize.config import Config
from overmind.utils.code import AgentBundle, BundleConfigError
from overmind.utils.ignore import build_ignore_predicate

logger = logging.getLogger(__name__)


def _compose_ignore(root: Path, exclude_globs: list[str]):
    """Return a predicate that combines ``.overmindignore`` + config excludes."""
    base = build_ignore_predicate(root)
    excludes = [g for g in exclude_globs if g]

    def predicate(rel_path: str) -> bool:
        if base(rel_path):
            return True
        for glob in excludes:
            if Path(rel_path).match(glob):
                return True
        return False

    return predicate


def _expand_context_scope(root: Path, patterns: list[str]) -> dict[str, str]:
    """Expand ``context_scope`` globs into a ``{rel_path: source}`` map.

    ``context_scope`` is the user-facing "read-only context" knob in
    ``eval_spec.json``. It used to be a dead-letter field — the
    receiving ``prefetched_files`` plumbing already existed on
    :meth:`AgentBundle.from_entry_point`, but the factory never called
    into it. Anything declared under ``scope.context_paths`` was
    therefore silently dropped from the bundle and from every candidate
    worktree, causing ``ImportError`` / ``FileNotFoundError`` at
    evaluation time for agents that genuinely needed those files.

    Patterns are interpreted relative to *root*. A literal-file pattern
    (``Path.is_file()`` is True) is taken verbatim; otherwise it's fed
    through ``Path.glob`` and every matching regular file is collected.
    Missing patterns are skipped silently — the bundle predates strict
    validation here and we don't want to break existing specs whose
    globs match nothing (e.g. optional fixtures).
    """
    out: dict[str, str] = {}
    for pattern in patterns:
        if not pattern:
            continue
        abs_p = root / pattern
        candidates: list[Path]
        if abs_p.is_file():
            candidates = [abs_p]
        else:
            candidates = [p for p in root.glob(pattern) if p.is_file()]
        for candidate in candidates:
            try:
                rel = str(candidate.relative_to(root))
            except ValueError:
                continue
            if rel in out:
                continue
            try:
                out[rel] = candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.debug(f"Skipping context file {rel}: {exc}")
    return out


def build_agent_bundle(config: Config) -> AgentBundle | None:
    """Build the optimization bundle from *config*.

    Returns ``None`` when bundling fails (e.g. entry file cannot be read or
    the project root is unresolvable) so the optimizer can fall back to a
    single-file view.
    """
    agent_path = Path(config.agent_path).resolve()
    if not agent_path.is_file():
        logger.warning(f"Agent file missing: {agent_path}")
        return None

    try:
        root = project_root_from_agent_file(str(agent_path))
        if root is None:
            root = project_root()
        root = Path(root).resolve()
    except Exception as exc:
        logger.warning(f"Could not resolve project root: {exc}")
        return None

    try:
        entry_rel = str(agent_path.relative_to(root))
    except ValueError:
        entry_rel = agent_path.name

    optimizable_paths = list(config.optimizable_scope) or [entry_rel]

    read_only_paths = list(config.read_only_scope) or None

    prefetched_files = _expand_context_scope(root, list(config.context_scope))

    search_paths = list(config.bundle_search_paths) or None

    try:
        bundle = AgentBundle.from_entry_point(
            entry_path=str(agent_path),
            project_root=str(root),
            entrypoint_fn=config.entrypoint_fn,
            optimizable_paths=optimizable_paths,
            read_only_paths=read_only_paths,
            max_total_chars=config.max_total_chars,
            max_resolved_files=config.max_resolved_files,
            should_ignore_rel=_compose_ignore(root, config.exclude_scope),
            prefetched_files=prefetched_files or None,
            search_paths=search_paths,
        )
    except BundleConfigError as exc:
        # Misconfiguration — surface the message verbatim so the user
        # sees the explicit guidance from the raise site (which already
        # tells them to use read_only_paths if that was their intent).
        logger.error(f"Bundle configuration error: {exc}")
        return None
    except Exception as exc:
        logger.warning(f"Bundle construction failed: {exc}", exc_info=True)
        return None

    logger.info(
        f"Built bundle: entry={entry_rel} files={len(bundle.original_files)} "
        f"pieces={len(bundle.pieces)} optimizable={optimizable_paths} "
        f"read_only={sorted(bundle.read_only_files) if bundle.read_only_files else []} "
        f"context={sorted(prefetched_files) if prefetched_files else []} "
        f"search_paths={search_paths or []}"
    )
    return bundle
