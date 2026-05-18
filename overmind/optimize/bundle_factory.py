"""Build an :class:`AgentBundle` from an optimize :class:`Config`.

This small factory decouples the optimizer from the details of how the
code bundle is assembled so the logic can be unit-tested in isolation.
Given a loaded :class:`Config` (which carries the agent path, entrypoint
function, and scope globs resolved from ``eval_spec.json``), produce the
:class:`AgentBundle` used by both baseline/candidate evaluation and the
analyzer prompt.

Scope model (post-collapse): two user-facing path lists.

* ``scope.optimizable_paths`` — files candidates may edit.
* ``scope.read_only_paths`` — files materialized into the bundle and
  enforced at accept time via a content diff (candidates may not edit).

The transitive import closure of the entry is auto-bundled (read-only
by default unless the file is also in ``optimizable_paths``). Project-
level drops (test directories, vendored code, infra) belong in
``.overmindignore`` or are covered by Overmind's hard-coded
``_ALWAYS_SKIP_DIRS`` set in :mod:`overmind.utils.ignore`.

The legacy ``scope.exclude_paths`` field is still parsed by
:func:`overmind.optimize.config.apply_eval_spec_scope` for one release
of backward compatibility. BFS-reachable matches are auto-promoted to
read-only here (mirroring the user's evident "not editable" intent);
non-BFS-reachable matches are no-ops. The shim and per-match warnings
will be removed once existing specs in the wild have been migrated.
"""

from __future__ import annotations

import logging
from pathlib import Path

from overmind.core.registry import project_root, project_root_from_agent_file
from overmind.optimize.config import Config
from overmind.utils.code import AgentBundle, BundleConfigError
from overmind.utils.ignore import build_ignore_predicate

logger = logging.getLogger(__name__)


def _first_legacy_exclude_match(rel: str, exclude_globs: list[str]) -> str | None:
    """Return the first ``_legacy_exclude_paths`` glob that matches *rel*.

    Used only by :func:`_auto_promote_legacy_excluded_runtime_deps` to
    name the offending pattern in the per-match deprecation warning so
    users can find the entry in their ``eval_spec.json`` without
    scrolling. Returns ``None`` when no glob matches.
    """
    if not rel or not exclude_globs:
        return None
    rel_path = Path(rel)
    for glob in exclude_globs:
        if not glob:
            continue
        try:
            if rel_path.match(glob):
                return glob
        except (TypeError, ValueError):
            continue
    return None


def _auto_promote_legacy_excluded_runtime_deps(
    bundle: AgentBundle,
    legacy_exclude_paths: list[str],
) -> list[tuple[str, str]]:
    """Promote BFS-reached files matched by legacy ``exclude_paths`` to read-only.

    Background. ``scope.exclude_paths`` used to be forwarded to the
    import-graph BFS in :func:`resolve_local_files` as a hard skip.
    That produced silently-broken candidate worktrees when an excluded
    path turned out to be a transitive runtime import of the entry —
    the candidate process would fail at ``importlib.exec_module`` time
    with ``ModuleNotFoundError`` and the optimizer would record a
    uniformly-low score across every candidate with no targeted
    diagnostic.

    The field is deprecated in this release in favor of the
    two-scope model (``optimizable_paths`` + ``read_only_paths`` plus
    ``.overmindignore`` for project-level drops). For one release we
    still accept ``exclude_paths`` in eval_spec and run this auto-
    promotion against BFS-reachable matches: the file lands in the
    bundle as read-only (preserving the user's "not editable" intent)
    and we emit a per-match warning naming the offending glob so users
    can migrate.

    Returns the ``(rel_path, matched_glob)`` pairs that were promoted.
    """
    if not legacy_exclude_paths:
        return []
    promoted: list[tuple[str, str]] = []
    for rel in list(bundle.original_files):
        match = _first_legacy_exclude_match(rel, legacy_exclude_paths)
        if match is None:
            continue
        if rel in bundle.optimizable_files:
            bundle.optimizable_files.discard(rel)
        bundle.read_only_files.add(rel)
        promoted.append((rel, match))
    return promoted


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
    read_only_paths = list(config.read_only_scope or [])
    legacy_excludes = list(getattr(config, "_legacy_exclude_paths", []) or [])

    search_paths = list(config.bundle_search_paths) or None

    # BFS uses only env-level ignore (``.overmindignore``, ``.git``,
    # ``.venv``, ``__pycache__``, …). The user-facing scope knobs are
    # never forwarded as hard skips: any file the entry's import graph
    # reaches is a runtime dependency of the candidate process and
    # must be present in every worktree. Editability is decided by
    # ``optimizable_paths`` / ``read_only_paths`` after BFS, not by
    # dropping files pre-BFS.
    env_ignore = build_ignore_predicate(root)

    try:
        bundle = AgentBundle.from_entry_point(
            entry_path=str(agent_path),
            project_root=str(root),
            entrypoint_fn=config.entrypoint_fn,
            optimizable_paths=optimizable_paths,
            read_only_paths=read_only_paths or None,
            max_total_chars=config.max_total_chars,
            max_resolved_files=config.max_resolved_files,
            should_ignore_rel=env_ignore,
            search_paths=search_paths,
        )
    except BundleConfigError as exc:
        logger.error(f"Bundle configuration error: {exc}")
        return None
    except Exception as exc:
        logger.warning(f"Bundle construction failed: {exc}", exc_info=True)
        return None

    auto_promoted = _auto_promote_legacy_excluded_runtime_deps(
        bundle, legacy_excludes
    )
    for rel, matched_glob in auto_promoted:
        logger.warning(
            "Runtime dependency %r is matched by deprecated scope."
            "exclude_paths glob %r; auto-promoting to read-only so the "
            "candidate worktree can still import it. Move this pattern "
            "into scope.read_only_paths in your eval_spec.json (or, for "
            "project-level drops the entry does not import, into "
            ".overmindignore). The exclude_paths field will be removed "
            "in the next release.",
            rel,
            matched_glob,
        )

    logger.info(
        f"Built bundle: entry={entry_rel} files={len(bundle.original_files)} "
        f"pieces={len(bundle.pieces)} optimizable={optimizable_paths} "
        f"read_only={sorted(bundle.read_only_files) if bundle.read_only_files else []} "
        f"search_paths={search_paths or []} "
        f"auto_promoted_legacy_excludes={[r for r, _ in auto_promoted]}"
    )
    return bundle
