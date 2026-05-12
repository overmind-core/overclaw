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
from overmind.utils.code import AgentBundle
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

    try:
        bundle = AgentBundle.from_entry_point(
            entry_path=str(agent_path),
            project_root=str(root),
            entrypoint_fn=config.entrypoint_fn,
            optimizable_paths=optimizable_paths,
            max_total_chars=config.max_total_chars,
            max_resolved_files=config.max_resolved_files,
            should_ignore_rel=_compose_ignore(root, config.exclude_scope),
        )
    except Exception as exc:
        logger.warning(f"Bundle construction failed: {exc}", exc_info=True)
        return None

    logger.info(
        f"Built bundle: entry={entry_rel} files={len(bundle.original_files)} "
        f"pieces={len(bundle.pieces)} optimizable={optimizable_paths}"
    )
    return bundle
