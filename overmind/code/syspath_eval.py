"""Static detection of ``sys.path`` mutations at module load.

Many real-world agents bridge non-standard layouts (hyphenated package
roots, sibling-package monorepos, projects without ``src/`` or
``pyproject.toml``) by mutating ``sys.path`` at module top, then importing
local code statically.  The bundle BFS would otherwise treat those imports
as external (PyPI) modules and the analyzer LLM is reduced to guessing
layout from docstrings.

This module re-implements the small subset of the Python runtime needed to
discover those entries statically:

:func:`eval_path_expr`
    Best-effort partial evaluator for the path-expression grammar agents
    typically use (``__file__``, ``Path``, ``/``-division, ``.parent``,
    ``.parents[N]``, ``os.path.{join,dirname,abspath,realpath}``, ``str``
    / ``Path`` wrappers, and previously-bound module-level constants).

:func:`detect_entry_search_paths`
    Walks the entry module top-level (and ``if`` / ``try`` bodies),
    folds constants discovered via :func:`eval_path_expr`, and returns
    every directory passed to ``sys.path.insert / append / extend`` or
    ``sys.path += [...]`` — restricted to directories that exist and live
    strictly under *project_root*.

The functions never raise: a failure to read or parse the entry yields
``[]``; an unsupported AST node yields ``None``.
"""

from __future__ import annotations

import ast
from pathlib import Path

__all__ = ["detect_entry_search_paths", "eval_path_expr"]


def eval_path_expr(
    node: ast.AST,
    *,
    file: Path,
    constants: dict[str, Path],
) -> Path | None:
    """Best-effort partial evaluation of a path expression.

    Returns ``None`` for any node outside the whitelisted grammar — this
    is always the safe failure mode (silently-skipped paths are far less
    harmful than imagined-up paths).
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            cand = Path(node.value)
            return cand if cand.is_absolute() else (file.parent / cand)
        return None

    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return file
        return constants.get(node.id)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = eval_path_expr(node.left, file=file, constants=constants)
        if left is None:
            return None
        right = node.right
        if isinstance(right, ast.Constant) and isinstance(right.value, str):
            return left / right.value
        return None

    if isinstance(node, ast.Attribute) and node.attr == "parent":
        base = eval_path_expr(node.value, file=file, constants=constants)
        return base.parent if base is not None else None

    if isinstance(node, ast.Subscript):
        target = node.value
        if isinstance(target, ast.Attribute) and target.attr == "parents":
            base = eval_path_expr(target.value, file=file, constants=constants)
            if base is None:
                return None
            slice_node = node.slice
            if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, int):
                try:
                    return base.parents[slice_node.value]
                except IndexError:
                    return None
        return None

    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id in ("Path", "str"):
            if not node.args:
                return None
            return eval_path_expr(node.args[0], file=file, constants=constants)
        if isinstance(func, ast.Attribute) and func.attr in ("resolve", "absolute"):
            return eval_path_expr(func.value, file=file, constants=constants)
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "os"
            and func.value.attr == "path"
        ):
            if not node.args:
                return None
            if func.attr == "join":
                base = eval_path_expr(node.args[0], file=file, constants=constants)
                if base is None:
                    return None
                for seg in node.args[1:]:
                    if not (isinstance(seg, ast.Constant) and isinstance(seg.value, str)):
                        return None
                    base = base / seg.value
                return base
            if func.attr == "dirname":
                base = eval_path_expr(node.args[0], file=file, constants=constants)
                return base.parent if base is not None else None
            if func.attr in ("abspath", "realpath"):
                return eval_path_expr(node.args[0], file=file, constants=constants)
        return None

    return None


def detect_entry_search_paths(entry_path: Path, project_root: Path) -> list[Path]:
    """Statically detect ``sys.path`` mutations at the top of *entry_path*.

    Returned paths are absolute, resolved, deduped (insertion-ordered),
    must live strictly under *project_root* (never escape the bundle
    boundary), and must point at existing directories.  Any failure to
    read or parse the entry yields ``[]``; the function never raises.
    """
    try:
        source = entry_path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    entry = entry_path.resolve()
    root = project_root.resolve()
    constants: dict[str, Path] = {}
    discovered: list[Path] = []
    seen: set[Path] = set()

    def _record(value: Path | None) -> None:
        if value is None:
            return
        try:
            resolved = value.resolve()
        except OSError:
            return
        if not resolved.is_dir():
            return
        try:
            resolved.relative_to(root)
        except ValueError:
            return
        if resolved == root or resolved in seen:
            return
        seen.add(resolved)
        discovered.append(resolved)

    def _handle_syspath_call(call: ast.Call) -> None:
        func = call.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "sys"
            and func.value.attr == "path"
        ):
            return
        if func.attr == "insert" and len(call.args) >= 2:
            arg = call.args[1]
            _record(eval_path_expr(arg, file=entry, constants=constants))
        elif func.attr == "append" and len(call.args) >= 1:
            _record(eval_path_expr(call.args[0], file=entry, constants=constants))
        elif func.attr == "extend" and call.args and isinstance(call.args[0], ast.List):
            for elt in call.args[0].elts:
                _record(eval_path_expr(elt, file=entry, constants=constants))

    def _is_syspath_target(target: ast.AST) -> bool:
        return (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "sys"
            and target.attr == "path"
        )

    def _scan_list_for_paths(expr: ast.AST) -> None:
        if isinstance(expr, (ast.List, ast.Tuple)):
            for elt in expr.elts:
                _record(eval_path_expr(elt, file=entry, constants=constants))
        elif isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            _scan_list_for_paths(expr.left)
            _scan_list_for_paths(expr.right)

    def _walk(stmts: list[ast.stmt]) -> None:
        for node in stmts:
            if isinstance(node, ast.Assign):
                value = eval_path_expr(node.value, file=entry, constants=constants)
                if value is not None:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            constants[target.id] = value
                if any(_is_syspath_target(t) for t in node.targets):
                    _scan_list_for_paths(node.value)
                continue
            if isinstance(node, ast.AugAssign) and _is_syspath_target(node.target):
                _scan_list_for_paths(node.value)
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                _handle_syspath_call(node.value)
                continue
            if isinstance(node, ast.If):
                _walk(node.body)
                _walk(node.orelse)
                continue
            if isinstance(node, ast.Try):
                _walk(node.body)
                for handler in node.handlers:
                    _walk(handler.body)
                _walk(node.orelse)
                _walk(node.finalbody)
                continue

    _walk(tree.body)
    return discovered
