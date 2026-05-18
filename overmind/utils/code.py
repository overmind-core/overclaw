"""
Agent code bundling and static analysis for multi-file optimization.

Resolves all project-local code reachable from an agent's entry file,
stores complete file sources, and provides whole-file replacement logic
to apply targeted updates back to original files.

This module bridges the gap between multi-file agent codebases and
the single-prompt optimization loop.  It produces a compact virtual
representation of only the code the agent actually uses, tagged with
origin information, and maps LLM-generated updates back into the
original file tree.

Usage::

    bundle = AgentBundle.from_entry_point(
        entry_path="agents/my_agent/agent.py",
        project_root="/path/to/project",
        entrypoint_fn="run",
    )

    # Render for LLM prompt
    prompt_text = bundle.to_prompt_text()

    # After LLM produces updated files, apply them
    modified_files = bundle.apply_file_updates(file_updates)
"""

from __future__ import annotations

import ast
import sys
import textwrap
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from overmind.utils.ignore import IgnorePredicate

# ---------------------------------------------------------------------------
# Code piece representation (internal analytics — not used in prompts)
# ---------------------------------------------------------------------------


@dataclass
class CodePiece:
    """A single extractable unit of code with its origin metadata."""

    piece_id: str
    file_path: str
    symbol_name: str
    symbol_type: str  # "imports" | "constant" | "function" | "class"
    source: str
    optimizable: bool
    line_start: int  # 1-indexed, inclusive
    line_end: int  # 1-indexed, inclusive
    base_indent: int = 0


class BundleConfigError(ValueError):
    """Raised for configuration errors that would silently produce a
    broken bundle.

    Currently fires when the entry file matches an ignore pattern
    (``.overmindignore`` or one of Overmind's hard-coded env-level
    skips). Without this guard the BFS bails on the first file, the
    bundle silently collapses to single-file mode, and the user spends
    an afternoon wondering why candidate worktrees only contain
    ``overmind_entrypoint.py``. Inheriting from :class:`ValueError`
    preserves call-site error handling for code that already catches
    ``ValueError``.
    """


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _node_name(node: ast.AST) -> str | None:
    """Return the top-level name bound by *node*, or ``None``."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                return target.id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _node_start_line(node: ast.AST) -> int:
    """First line of *node*, accounting for decorators."""
    if hasattr(node, "decorator_list") and node.decorator_list:
        return node.decorator_list[0].lineno
    return node.lineno


def _detect_base_indent(source_lines: list[str], start: int) -> int:
    """Detect the indentation level of the first non-empty line."""
    for line in source_lines[start:]:
        stripped = line.lstrip()
        if stripped:
            return len(line) - len(stripped)
    return 0


def _source_segment(lines: list[str], start: int, end: int) -> str:
    """Extract source from *lines* (0-indexed start, 0-indexed exclusive end)."""
    return "".join(lines[start:end])


def _names_referenced_in(node: ast.AST) -> set[str]:
    """Collect all ``Name`` identifiers referenced inside *node*."""
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            root = child
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                names.add(root.id)
    return names


def has_entrypoint_ast(source: str, fn_name: str) -> bool:
    """Check via AST whether *source* defines a top-level function *fn_name*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
            return True
    return False


# ---------------------------------------------------------------------------
# Import resolution
# ---------------------------------------------------------------------------


def _lang_tag_for_path(rel_path: str) -> str:
    """Return a Markdown code fence language tag for *rel_path*."""
    ext = Path(rel_path).suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".mjs": "javascript",
        ".ts": "typescript",
        ".mts": "typescript",
    }.get(ext, "python")


_STDLIB_TOP: frozenset[str] = getattr(sys, "stdlib_module_names", frozenset()) | frozenset(sys.builtin_module_names)


def _is_local_module(
    module_name: str,
    from_file: Path,
    project_root: Path,
    *,
    search_paths: Sequence[Path] | None = None,
) -> bool:
    """Return True if *module_name* resolves to a project-local file.

    Delegates to :func:`_resolve_module_to_file` so a module is "local"
    iff the resolver can actually find a ``.py`` file (or package
    ``__init__.py``) for it under *project_root*, searching from both
    the importing file's own directory and the project root. The
    earlier depth-1 ``project_root / top`` heuristic missed every
    nested layout (e.g. ``src/<pkg>/`` or ``python_backend/<pkg>/``);
    delegating removes the divergence between the "is local" predicate
    and the "find the file" resolver they were both meant to embody.

    ``search_paths`` lets callers extend the resolver with additional
    directories under *project_root* that should be treated as
    sys.path-style roots — used to support hyphenated layouts
    (``python-backend/``) and ``src/``-layouts whose package
    directories aren't direct children of *project_root*.

    Stdlib top-level names are short-circuited before invoking the
    filesystem-walking resolver — this is a hot path during BFS so
    avoiding hundreds of stat calls per ``import os`` is worth the
    duplicated check.
    """
    top = module_name.split(".")[0]
    if top in _STDLIB_TOP:
        return False
    return _resolve_module_to_file(module_name, from_file, project_root, search_paths=search_paths) is not None


def _resolve_module_to_file(
    module_name: str,
    from_file: Path,
    project_root: Path,
    *,
    search_paths: Sequence[Path] | None = None,
) -> Path | None:
    """Try to resolve a dotted module to a ``.py`` file under *project_root*.

    ``search_paths`` extends the resolver's base list with extra
    directories under *project_root*, in the same role as entries on
    ``sys.path``. Order: importing file's own directory → each search
    path (in order) → project root. The project root remains the last
    fallback so explicit search-path declarations always win over
    accidental same-named files at the repo root.
    """
    parts = module_name.split(".")

    bases: list[Path] = []
    pkg_dir = from_file.parent
    if pkg_dir != project_root:
        bases.append(pkg_dir)
    if search_paths:
        for sp in search_paths:
            sp_resolved = sp if sp.is_absolute() else (project_root / sp)
            try:
                sp_resolved = sp_resolved.resolve()
            except OSError:
                continue
            try:
                sp_resolved.relative_to(project_root)
            except ValueError:
                # Refuse search paths that escape the project root —
                # they'd let imports pull in files outside the bundle's
                # scope, which is both a confidentiality and a
                # reproducibility hazard.
                continue
            if sp_resolved not in bases:
                bases.append(sp_resolved)
    if project_root not in bases:
        bases.append(project_root)

    for base in bases:
        candidate = base / "/".join(parts)
        py_path = candidate.with_suffix(".py")
        if py_path.exists():
            try:
                py_path.relative_to(project_root)
            except ValueError:
                continue
            return py_path
        init_path = candidate / "__init__.py"
        if init_path.exists():
            try:
                init_path.relative_to(project_root)
            except ValueError:
                continue
            return init_path

    return None


def discover_search_paths(project_root: Path) -> list[Path]:
    """Auto-discover sys.path-style search roots under *project_root*.

    Looks at signals that real-world Python projects use to declare
    importable layouts that aren't direct children of the repo root:

    * ``src/`` directory at the project root (the standard "src layout"
      from setuptools/PEP 518). When present, ``src/`` itself is added
      so ``import mypkg`` resolves to ``src/mypkg/``.
    * ``[tool.setuptools.package-dir]`` and
      ``[tool.setuptools.packages.find].where`` in ``pyproject.toml``
      (the most common modern declaration of source roots).

    Results are de-duplicated and absolute, ordered by signal priority
    (pyproject.toml first because it's an explicit declaration; ``src/``
    second as the convention fallback). Users can still override or
    extend the list via ``scope.search_paths`` in ``eval_spec.json``.
    """
    discovered: list[Path] = []
    seen: set[Path] = set()

    def _add(p: Path) -> None:
        try:
            p_resolved = p.resolve()
        except OSError:
            return
        if not p_resolved.is_dir():
            return
        try:
            p_resolved.relative_to(project_root.resolve())
        except ValueError:
            return
        if p_resolved in seen:
            return
        seen.add(p_resolved)
        discovered.append(p_resolved)

    pyproject = project_root / "pyproject.toml"
    if pyproject.is_file():
        try:
            import tomllib  # Python 3.11+
        except ImportError:  # pragma: no cover - older runtimes
            tomllib = None  # type: ignore[assignment]
        if tomllib is not None:
            try:
                data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            tool = data.get("tool", {}) if isinstance(data, dict) else {}
            setuptools_cfg = tool.get("setuptools", {}) if isinstance(tool, dict) else {}
            if isinstance(setuptools_cfg, dict):
                pkg_dir = setuptools_cfg.get("package-dir", {})
                if isinstance(pkg_dir, dict):
                    for raw in pkg_dir.values():
                        if isinstance(raw, str) and raw:
                            _add(project_root / raw)
                find_cfg = setuptools_cfg.get("packages", {})
                if isinstance(find_cfg, dict):
                    where = find_cfg.get("find", {}).get("where") if isinstance(find_cfg.get("find"), dict) else None
                    if isinstance(where, list):
                        for raw in where:
                            if isinstance(raw, str) and raw:
                                _add(project_root / raw)

    src_dir = project_root / "src"
    if src_dir.is_dir():
        _add(src_dir)

    return discovered


def _intermediate_init_files(leaf_path: Path, project_root: Path) -> list[Path]:
    """Return every ``__init__.py`` between *leaf_path*'s package and the
    project root, walking strictly upward.

    Motivation: when the BFS resolves ``pkg.subpkg.module`` it lands on
    ``pkg/subpkg/module.py`` and stops. The intermediate
    ``pkg/__init__.py`` and ``pkg/subpkg/__init__.py`` are silently
    skipped. Real-world packages put meaningful code in ``__init__.py``
    (re-exports of the public API, version constants, side-effecting
    setup like environment loading). Skipping them means the bundled
    candidate worktree imports a *different* package than the user
    runs against, so candidate behaviour can diverge from baseline for
    reasons that have nothing to do with the LLM's edits.

    PEP 420 namespace packages have no ``__init__.py`` at the directory
    level — we only return files that actually exist, so namespace
    layouts continue to work unchanged.

    The root-level ``__init__.py`` (directly under *project_root*) is
    not included: it isn't on any package's import path and pulling it
    in would surprise users whose repo root happens to have one for
    unrelated reasons (e.g. an editable install).
    """
    results: list[Path] = []
    parent = leaf_path.parent
    project_root = project_root.resolve()
    seen: set[Path] = set()
    while True:
        try:
            parent_resolved = parent.resolve()
        except OSError:
            break
        if parent_resolved == project_root:
            break
        try:
            parent_resolved.relative_to(project_root)
        except ValueError:
            break
        init = parent_resolved / "__init__.py"
        if init.is_file() and init not in seen:
            seen.add(init)
            results.append(init)
        new_parent = parent_resolved.parent
        if new_parent == parent_resolved:
            break
        parent = new_parent
    return results


from overmind.code.syspath_eval import (
    detect_entry_search_paths as _detect_entry_search_paths,
)
from overmind.code.syspath_eval import (
    eval_path_expr as _eval_path_expr,
)


def _collect_import_targets(source: str) -> list[str]:
    """Return dotted module names from all import statements in *source*.

    Handles both absolute imports and relative imports (``from . import x``).
    For relative imports without an explicit module (level-only), the
    importing file's package is used as the base.

    Also honors a top-level ``__overmind_imports__`` assignment as a
    static hint for dynamic imports. Authors of plugin loaders,
    entrypoints that use ``importlib.import_module``, and lazy proxies
    can opt in by declaring the modules they load — the names are
    treated as if they appeared in a normal ``import`` statement. This
    is a cheap substitute for full runtime import tracing; we don't
    pretend to capture *every* dynamic import, but we give shims a way
    to surface the ones the static walker can't
    see.

    Example::

        __overmind_imports__ = ["airline.agents", "airline.tools"]
        mod = importlib.import_module(sys.argv[1])  # dynamic, hidden

    Both list and tuple literals are accepted; only string-literal
    elements are honored (anything computed at runtime is ignored
    since we can't safely evaluate it during static analysis).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    targets: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                targets.append(node.module)
                # Also surface ``from PKG import NAME`` as a submodule
                # candidate (``PKG.NAME``). When NAME is a re-exported
                # function/class this resolves to nothing and is dropped
                # silently; when NAME is a real submodule it lets the BFS
                # follow ``from langextract import visualization`` style
                # imports without which the worktree ships an incomplete
                # package and candidates fail at import time.
                if not (node.level and node.level > 0):
                    for alias in node.names:
                        if alias.name and alias.name != "*":
                            targets.append(f"{node.module}.{alias.name}")
            elif node.level and node.level > 0 and node.names:
                for alias in node.names:
                    targets.append(alias.name)
        elif isinstance(node, ast.Assign):
            # Surface ``__overmind_imports__ = ["pkg.mod", ...]`` hints.
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "__overmind_imports__"
                    and isinstance(node.value, (ast.List, ast.Tuple))
                ):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            targets.append(elt.value)
    return targets


def _resolve_relative_import(
    node: ast.ImportFrom,
    from_file: Path,
    project_root: Path,
) -> list[Path]:
    """Resolve a relative ImportFrom node to concrete file paths."""
    results: list[Path] = []
    pkg_dir = from_file.parent

    for _ in range(max(0, (node.level or 0) - 1)):
        pkg_dir = pkg_dir.parent

    if node.module:
        parts = node.module.split(".")
        candidate = pkg_dir / "/".join(parts)
        py_path = candidate.with_suffix(".py")
        if py_path.exists():
            try:
                py_path.relative_to(project_root)
                results.append(py_path)
            except ValueError:
                pass
        init_path = candidate / "__init__.py"
        if init_path.exists():
            try:
                init_path.relative_to(project_root)
                results.append(init_path)
            except ValueError:
                pass
    else:
        for alias in node.names:
            candidate = pkg_dir / alias.name
            py_path = candidate.with_suffix(".py")
            if py_path.exists():
                try:
                    py_path.relative_to(project_root)
                    results.append(py_path)
                except ValueError:
                    pass
            init_path = candidate / "__init__.py"
            if init_path.exists():
                try:
                    init_path.relative_to(project_root)
                    results.append(init_path)
                except ValueError:
                    pass

    return results


def resolve_local_files(
    entry_path: str,
    project_root: str,
    *,
    max_depth: int = 6,
    max_files: int | None = None,
    should_ignore_rel: IgnorePredicate | None = None,
    search_paths: Sequence[str | Path] | None = None,
) -> dict[str, str]:
    """Resolve project-local files reachable from *entry_path* (breadth-first).

    Returns ``{relative_path: source_code}`` with the entry file first, then
    dependencies in roughly increasing distance from the entry file.

    Parameters
    ----------
    max_files:
        If set, stop after this many files have been collected. Used for LLM
        prompts (e.g. setup analysis) where pulling an entire vendored package
        tree would exceed context limits.
    should_ignore_rel:
        If set, skip any project-relative path for which this returns True (no
        read, no import traversal from that file).
    search_paths:
        Extra ``sys.path``-style directories (relative to *project_root* or
        absolute, but they must resolve under *project_root*) that the import
        resolver should treat as additional package roots. Used to make
        hyphenated directories (``python-backend/``) and ``src/`` layouts
        discoverable without forcing users to declare every file in
        ``optimizable_paths``. When ``None``, auto-discovery via
        :func:`discover_search_paths` is applied.
    """
    root = Path(project_root).resolve()
    entry = Path(entry_path).resolve()
    result: dict[str, str] = {}
    visited: set[Path] = set()
    queue: deque[tuple[Path, int]] = deque()

    effective_search_paths: list[Path] = []
    seen_sp: set[Path] = set()

    def _add_sp(raw: Path | str) -> None:
        p = raw if isinstance(raw, Path) else Path(raw)
        if not p.is_absolute():
            p = root / p
        try:
            p_resolved = p.resolve()
        except OSError:
            return
        try:
            p_resolved.relative_to(root)
        except ValueError:
            return
        if not p_resolved.is_dir() or p_resolved in seen_sp:
            return
        seen_sp.add(p_resolved)
        effective_search_paths.append(p_resolved)

    if search_paths is None:
        # Auto-discovery path: project-level signals (``src/``,
        # ``pyproject.toml``) plus runtime hints from the entry's
        # ``sys.path`` mutations. The latter is what lets the analyzer
        # produce a multi-file bundle for hyphenated / sibling-package
        # layouts without any user declaration.
        for p in discover_search_paths(root):
            _add_sp(p)
        for p in _detect_entry_search_paths(entry, root):
            _add_sp(p)
    else:
        # Explicit list opts out of autodiscovery — the caller takes
        # full responsibility for what counts as a package root.
        for raw in search_paths:
            _add_sp(raw)

    try:
        entry_rel = str(entry.relative_to(root))
    except ValueError:
        return {}

    # Fail loud if the user accidentally listed the entry file in an
    # ignore pattern. The alternative (silent single-file fallback)
    # was the source of the original "candidate worktree only has the
    # harness" debugging sink.
    if should_ignore_rel and should_ignore_rel(entry_rel):
        raise BundleConfigError(
            f"Entry file {entry_rel!r} is matched by an ignore pattern "
            f"(.overmindignore or one of Overmind's env-level skips). The "
            f"entry must be reachable so the dependency BFS can walk from "
            f"it; if the entry is the optimization harness and should not "
            f"be edited, list it in scope.read_only_paths instead."
        )

    queue.append((entry, 0))

    while queue:
        if max_files is not None and len(result) >= max_files:
            break
        file_path, depth = queue.popleft()
        if depth > max_depth or file_path in visited:
            continue
        if not file_path.exists() or not file_path.is_file():
            continue
        try:
            file_path.relative_to(root)
        except ValueError:
            continue

        rel = str(file_path.relative_to(root))
        if should_ignore_rel and should_ignore_rel(rel):
            visited.add(file_path)
            continue

        visited.add(file_path)
        source = file_path.read_text(encoding="utf-8")
        result[rel] = source

        # Resolve absolute imports
        for mod_name in _collect_import_targets(source):
            if not _is_local_module(mod_name, file_path, root, search_paths=effective_search_paths):
                continue
            resolved = _resolve_module_to_file(mod_name, file_path, root, search_paths=effective_search_paths)
            if resolved and resolved not in visited:
                try:
                    rrel = str(resolved.relative_to(root))
                except ValueError:
                    continue
                if should_ignore_rel and should_ignore_rel(rrel):
                    continue
                queue.append((resolved, depth + 1))
                # Pull in intermediate ``__init__.py`` files for every
                # resolved leaf. Otherwise BFS stops at the leaf and
                # candidates run against a different package than the
                # user's baseline (re-exports, version constants, and
                # side-effecting setup live in ``__init__.py``).
                for init in _intermediate_init_files(resolved, root):
                    if init in visited:
                        continue
                    try:
                        irel = str(init.relative_to(root))
                    except ValueError:
                        continue
                    if should_ignore_rel and should_ignore_rel(irel):
                        continue
                    queue.append((init, depth + 1))

        # Resolve relative imports via AST
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                for resolved in _resolve_relative_import(node, file_path, root):
                    if resolved not in visited:
                        try:
                            rrel = str(resolved.relative_to(root))
                        except ValueError:
                            continue
                        if should_ignore_rel and should_ignore_rel(rrel):
                            continue
                        queue.append((resolved, depth + 1))
                        for init in _intermediate_init_files(resolved, root):
                            if init in visited:
                                continue
                            try:
                                irel = str(init.relative_to(root))
                            except ValueError:
                                continue
                            if should_ignore_rel and should_ignore_rel(irel):
                                continue
                            queue.append((init, depth + 1))

    return result


# ---------------------------------------------------------------------------
# Piece extraction (internal — used for analytics, not for prompts)
# ---------------------------------------------------------------------------


def _extract_import_block(source: str, tree: ast.Module) -> tuple[str, int, int] | None:
    """Extract the contiguous import block at the top of a module.

    Returns ``(source_text, start_line_1indexed, end_line_1indexed)`` or None.
    """
    lines = source.splitlines(keepends=True)
    import_nodes = [n for n in ast.iter_child_nodes(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    if not import_nodes:
        return None

    start = import_nodes[0].lineno
    end = import_nodes[-1].end_lineno or import_nodes[-1].lineno
    return _source_segment(lines, start - 1, end), start, end


def extract_pieces(
    rel_path: str,
    source: str,
    *,
    optimizable: bool = True,
    used_names: set[str] | None = None,
) -> list[CodePiece]:
    """Extract top-level code pieces from *source*.

    If *used_names* is provided, only pieces whose symbol name is in the set
    (or that are referenced by those pieces) are included.  When ``None``,
    all top-level symbols are extracted.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [
            CodePiece(
                piece_id="",
                file_path=rel_path,
                symbol_name="__full_file__",
                symbol_type="constant",
                source=source,
                optimizable=optimizable,
                line_start=1,
                line_end=source.count("\n") + 1,
            )
        ]

    lines = source.splitlines(keepends=True)
    pieces: list[CodePiece] = []

    # --- Imports (always included) ---
    imp = _extract_import_block(source, tree)
    if imp:
        imp_src, imp_start, imp_end = imp
        pieces.append(
            CodePiece(
                piece_id="",
                file_path=rel_path,
                symbol_name="__imports__",
                symbol_type="imports",
                source=imp_src.rstrip("\n"),
                optimizable=optimizable,
                line_start=imp_start,
                line_end=imp_end,
            )
        )

    # --- Build a map of all top-level definitions ---
    top_level_nodes: list[tuple[str, ast.AST]] = []
    for node in ast.iter_child_nodes(tree):
        name = _node_name(node)
        if name:
            top_level_nodes.append((name, node))

    # --- Resolve which names to include ---
    if used_names is not None:
        included = set(used_names)
        node_map = {name: node for name, node in top_level_nodes}
        changed = True
        while changed:
            changed = False
            for name in list(included):
                if name not in node_map:
                    continue
                refs = _names_referenced_in(node_map[name])
                for ref in refs:
                    if ref in node_map and ref not in included:
                        included.add(ref)
                        changed = True
    else:
        included = {name for name, _ in top_level_nodes}

    # --- Extract each included symbol ---
    for name, node in top_level_nodes:
        if name not in included:
            continue

        start_line = _node_start_line(node)
        end_line = node.end_lineno or start_line
        seg = _source_segment(lines, start_line - 1, end_line)
        indent = _detect_base_indent(lines, start_line - 1)

        sym_type = "constant"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sym_type = "function"
        elif isinstance(node, ast.ClassDef):
            sym_type = "class"

        pieces.append(
            CodePiece(
                piece_id="",
                file_path=rel_path,
                symbol_name=name,
                symbol_type=sym_type,
                source=seg.rstrip("\n"),
                optimizable=optimizable,
                line_start=start_line,
                line_end=end_line,
                base_indent=indent,
            )
        )

    return pieces


# ---------------------------------------------------------------------------
# Agent bundle
# ---------------------------------------------------------------------------


@dataclass
class AgentBundle:
    """Virtual representation of a multi-file agent for the optimization prompt.

    Holds original file sources tagged with optimizability.  The LLM sees
    complete files (not fragments) and returns complete updated files.

    ``read_only_files`` contains paths that are materialized into the
    bundle (and into candidate worktrees) but must not be edited by
    candidates. The accept step diffs these files against the baseline
    content captured here and rejects any candidate that mutated them.
    """

    entry_file: str  # relative path to entry point
    entry_function: str
    pieces: list[CodePiece] = field(default_factory=list)
    original_files: dict[str, str] = field(default_factory=dict)
    project_root: str = ""
    optimizable_files: set[str] = field(default_factory=set)
    read_only_files: set[str] = field(default_factory=set)

    # --- Construction ---------------------------------------------------

    @classmethod
    def from_entry_point(
        cls,
        entry_path: str,
        project_root: str,
        entrypoint_fn: str,
        *,
        optimizable_paths: Sequence[str] | None = None,
        read_only_paths: Sequence[str] | None = None,
        max_total_chars: int = 150_000,
        max_resolved_files: int | None = None,
        should_ignore_rel: IgnorePredicate | None = None,
        search_paths: Sequence[str | Path] | None = None,
    ) -> AgentBundle:
        """Build a bundle by resolving all local dependencies from *entry_path*.

        Parameters
        ----------
        entry_path:
            Absolute path to the agent's entry file.
        project_root:
            Absolute path to the project root.
        entrypoint_fn:
            Name of the entry function the optimizer invokes.
        optimizable_paths:
            Relative paths (under *project_root*) of files the LLM may modify.
            When ``None``, every resolved local dependency file is optimizable.
            Pass ``[entry_file]`` (relative path) for read-only dependency context
            with signature compression (e.g. setup-time analysis prompts).
        read_only_paths:
            Relative paths (or globs under *project_root*) of files that must be
            materialized into the bundle (and into candidate worktrees) but must
            not be edited by candidates. Files listed here are removed from the
            ``optimizable_files`` set even when they overlap ``optimizable_paths``
            or the entry file. The accept step enforces non-modification via a
            content diff against ``original_files``.
        max_total_chars:
            Token budget expressed as characters.  Read-only files beyond
            this budget are demoted to signature-only representation.
        max_resolved_files:
            Optional cap on how many project-local files to follow from the
            entry point (breadth-first). ``None`` means no limit.
        should_ignore_rel:
            Skip matching paths during BFS (see :func:`resolve_local_files`).
        search_paths:
            Extra sys.path-style directories under *project_root* the
            resolver should treat as package roots. Forwarded to
            :func:`resolve_local_files`. When ``None``, the resolver
            auto-discovers via :func:`discover_search_paths`
            (``src/`` + ``pyproject.toml`` declarations).
        """
        root = Path(project_root).resolve()
        entry = Path(entry_path).resolve()
        entry_rel = str(entry.relative_to(root))

        local_files = resolve_local_files(
            entry_path,
            project_root,
            max_files=max_resolved_files,
            should_ignore_rel=should_ignore_rel,
            search_paths=search_paths,
        )

        if optimizable_paths is None:
            opt_set = set(local_files.keys())
        else:
            # Expand glob patterns and collect concrete relative paths.
            expanded: set[str] = set()
            for pattern in optimizable_paths:
                abs_p = root / pattern
                if abs_p.is_file():
                    expanded.add(pattern)
                else:
                    # Treat as a glob pattern relative to project root.
                    matched = list(root.glob(pattern))
                    if matched:
                        for m in matched:
                            try:
                                expanded.add(str(m.relative_to(root)))
                            except ValueError:
                                pass
                    else:
                        # Keep as-is; may already be in local_files or resolve later.
                        expanded.add(pattern)
            opt_set = expanded
            opt_set.add(entry_rel)
            for rel in list(opt_set):
                if rel in local_files:
                    continue
                abs_p = root / rel
                if abs_p.is_file():
                    if should_ignore_rel and should_ignore_rel(rel):
                        continue
                    local_files[rel] = abs_p.read_text(encoding="utf-8")

        # Materialize read_only_paths into local_files and remove them from
        # the optimizable set. This is the mechanism that lets users keep
        # the entrypoint (or any harness file) in the bundle / worktree
        # while guaranteeing it stays editable only by them, not by
        # candidates. The accept step is responsible for the actual
        # enforcement; here we just stamp the intent on the bundle.
        read_only_set: set[str] = set()
        if read_only_paths:
            for pattern in read_only_paths:
                abs_p = root / pattern
                if abs_p.is_file():
                    read_only_set.add(pattern)
                else:
                    for m in root.glob(pattern):
                        if not m.is_file():
                            continue
                        try:
                            read_only_set.add(str(m.relative_to(root)))
                        except ValueError:
                            pass
            for rel in read_only_set:
                if rel in local_files:
                    continue
                abs_p = root / rel
                if abs_p.is_file():
                    local_files[rel] = abs_p.read_text(encoding="utf-8")
            # read_only wins over optimizable: subtract overlap so callers
            # don't get conflicting signals about which files candidates
            # may modify.
            opt_set -= read_only_set

        bundle = cls(
            entry_file=entry_rel,
            entry_function=entrypoint_fn,
            original_files=dict(local_files),
            project_root=project_root,
            optimizable_files=set(opt_set),
            read_only_files=set(read_only_set),
        )

        # Extract pieces for internal analytics (symbol tracking)
        ordered_paths = [entry_rel] + [p for p in local_files if p != entry_rel]

        total_chars = 0
        for rel_path in ordered_paths:
            source = local_files[rel_path]
            is_opt = rel_path in opt_set

            pieces = extract_pieces(rel_path, source, optimizable=is_opt)

            for p in pieces:
                total_chars += len(p.source)

            if total_chars > max_total_chars and not is_opt:
                sig_pieces = _signatures_only(pieces)
                bundle.pieces.extend(sig_pieces)
            else:
                bundle.pieces.extend(pieces)

        bundle._assign_ids()

        return bundle

    # --- ID assignment --------------------------------------------------

    def _assign_ids(self) -> None:
        """Assign positional IDs ``P0``, ``P1``, … to all pieces."""
        for idx, piece in enumerate(self.pieces):
            piece.piece_id = f"P{idx}"

    # --- Prompt rendering -----------------------------------------------

    def to_prompt_text(self) -> str:
        """Render the bundle as whole-file sections for the LLM prompt.

        Each file is shown in full (or signature-only for compressed
        read-only deps), clearly delimited with optimizability tags.
        """
        sections: list[str] = []

        ordered_paths = [self.entry_file] + [p for p in self.original_files if p != self.entry_file]

        for rel_path in ordered_paths:
            source = self.original_files.get(rel_path)
            if source is None:
                continue

            is_opt = rel_path in self.optimizable_files
            tag = "OPTIMIZABLE" if is_opt else "READ-ONLY"

            file_pieces = self.pieces_for_file(rel_path)
            has_signature_only = any(
                p.symbol_name == "__signature__" or p.source.rstrip().endswith("...")
                for p in file_pieces
                if p.symbol_type in ("function", "class")
            )

            lang_tag = _lang_tag_for_path(rel_path)
            if has_signature_only and not is_opt:
                sig_text = "\n\n".join(p.source for p in file_pieces)
                sections.append(f"\n# ===== FILE: {rel_path} [{tag}] =====\n```{lang_tag}\n{sig_text}\n```")
            else:
                sections.append(f"\n# ===== FILE: {rel_path} [{tag}] =====\n```{lang_tag}\n{source}\n```")

        return "\n".join(sections)

    def get_entry_code(self) -> str:
        """Return the full original source of the entry file."""
        return self.original_files.get(self.entry_file, "")

    def get_all_optimizable_code(self) -> str:
        """Concatenated source of all optimizable files (for metrics)."""
        return "\n\n".join(
            source for rel_path, source in self.original_files.items() if rel_path in self.optimizable_files
        )

    def get_optimizable_piece_ids(self) -> list[str]:
        """Return piece IDs of all optimizable pieces."""
        return [p.piece_id for p in self.pieces if p.optimizable]

    # --- Piece lookup ---------------------------------------------------

    def piece_by_id(self, piece_id: str) -> CodePiece | None:
        """Look up a piece by its positional ID."""
        for p in self.pieces:
            if p.piece_id == piece_id:
                return p
        return None

    def pieces_for_file(self, rel_path: str) -> list[CodePiece]:
        """Return all pieces belonging to *rel_path*."""
        return [p for p in self.pieces if p.file_path == rel_path]

    # --- Whole-file update application ----------------------------------

    def apply_file_updates(
        self,
        file_updates: dict[str, str],
    ) -> dict[str, str] | None:
        """Apply whole-file updates to the bundle.

        Parameters
        ----------
        file_updates:
            Mapping of ``{relative_path: complete_new_source}``.

        Returns
        -------
        dict or None
            ``{relative_path: validated_source}`` for files that actually
            changed, or ``None`` if any file has a syntax error.
        """
        modified: dict[str, str] = {}

        for rel_path, new_source in file_updates.items():
            if rel_path not in self.optimizable_files:
                continue

            if rel_path.endswith(".py"):
                try:
                    ast.parse(new_source)
                except SyntaxError:
                    return None
            elif rel_path.endswith((".js", ".mjs")):
                pass  # JS syntax checked at candidate validation time
            elif rel_path.endswith((".ts", ".mts")):
                pass  # TS syntax checked at candidate validation time

            original = self.original_files.get(rel_path, "")
            if new_source.rstrip() != original.rstrip():
                modified[rel_path] = new_source

        return modified

    def get_full_file_set(
        self,
        updates: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Return the complete file set, with optional updates merged in.

        Useful for creating temp directories for validation/execution.
        """
        result = dict(self.original_files)
        if updates:
            result.update(updates)
        return result

    def is_multi_file(self) -> bool:
        """Return True if the bundle spans more than one file."""
        return len(self.original_files) > 1

    def optimizable_file_count(self) -> int:
        """Count distinct files that have optimizable pieces."""
        return len(self.optimizable_files & set(self.original_files.keys()))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signatures_only(pieces: list[CodePiece]) -> list[CodePiece]:
    """Demote pieces to signature-only versions for context compression."""
    result: list[CodePiece] = []
    for p in pieces:
        if p.symbol_type == "imports":
            result.append(p)
            continue
        if p.symbol_type in ("function", "class"):
            sig = _extract_signature(p.source)
            if sig:
                result.append(
                    CodePiece(
                        piece_id=p.piece_id,
                        file_path=p.file_path,
                        symbol_name=p.symbol_name,
                        symbol_type=p.symbol_type,
                        source=sig,
                        optimizable=p.optimizable,
                        line_start=p.line_start,
                        line_end=p.line_end,
                        base_indent=p.base_indent,
                    )
                )
                continue
        result.append(p)
    return result


def _extract_signature(source: str) -> str | None:
    """Extract the function/class signature + docstring from *source*."""
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines = source.splitlines()
            sig_end = node.body[0].lineno - 1 if node.body else node.lineno
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, (ast.Constant, ast.Str))
            ):
                sig_end = node.body[0].end_lineno or sig_end
            sig_lines = lines[:sig_end]
            sig_lines.append("    ...")
            return "\n".join(sig_lines)

        if isinstance(node, ast.ClassDef):
            lines = source.splitlines()
            result_lines: list[str] = []
            in_class = False
            for i, line in enumerate(lines):
                if not in_class:
                    result_lines.append(line)
                    if line.strip().startswith("class "):
                        in_class = True
                elif in_class:
                    stripped = line.strip()
                    if stripped.startswith(("def ", "async def ")):
                        result_lines.append(line)
                        result_lines.append("        ...")
                    elif stripped.startswith(('"""', "'''")):
                        result_lines.append(line)
                    elif stripped and not stripped.startswith("#"):
                        if "=" in stripped:
                            result_lines.append(line)
            return "\n".join(result_lines)

    return None
