"""Static AST scan for instrumentation candidates: entrypoints, routes, agent classes, LLM calls, tools.

Pure AST — no imports of user code, no LLM, no network.
"""

from __future__ import annotations

import ast
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from overmind.instrumentation_checker import _git_revision

_SKIP_DIRS = {
    "venv",
    ".venv",
    "node_modules",
    "site-packages",
    "build",
    "dist",
    "__pycache__",
    ".worktrees",
}


def _skip_dir(name: str) -> bool:
    if name.startswith("."):
        return True
    if name in _SKIP_DIRS:
        return True
    return name.startswith("venv") or name.startswith(".venv")


_ROUTE_DECORATOR_METHODS = {
    "get",
    "post",
    "put",
    "patch",
    "delete",
    "head",
    "options",
    "route",
    "websocket",
}
_ROUTE_DECORATOR_OBJECTS = {"app", "router"}

_LLM_CALL_CHAINS = {
    ("chat", "completions", "create"),
    ("messages", "create"),
    ("generate_content",),
    ("responses", "create"),
}

_TOOL_DECORATOR_NAMES = {"tool", "function_tool"}

_AGENT_BASE_MARKERS = ("Agent", "Crew", "Graph", "Workflow")

_FRAMEWORK_IMPORTS = {
    "fastapi": "fastapi",
    "flask": "flask",
    "click": "click",
    "typer": "typer",
    "langgraph": "langgraph",
    "crewai": "crewai",
    "openai": "openai",
    "anthropic": "anthropic",
    "google.genai": "google.genai",
    "agno": "agno",
}


def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Attribute):
        base = _decorator_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _is_route_decorator(node: ast.expr) -> bool:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    attr = node.func
    if attr.attr not in _ROUTE_DECORATOR_METHODS:
        return False
    return isinstance(attr.value, ast.Name)


def _is_entry_decorator(name: str) -> bool:
    return name in {"click.command", "command", "typer.command", "app.command"}


def _is_tool_decorator(name: str) -> bool:
    tail = name.rsplit(".", 1)[-1]
    return tail in _TOOL_DECORATOR_NAMES or name == "overmind.tool"


def _render_signature(args: ast.arguments) -> str:
    parts: list[str] = []
    for arg in args.posonlyargs:
        parts.append(arg.arg)
    if args.posonlyargs:
        parts.append("/")
    for arg in args.args:
        parts.append(arg.arg)
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")
    for arg in args.kwonlyargs:
        parts.append(arg.arg)
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return "(" + ", ".join(parts) + ")"


def _docstring(node: ast.AST) -> str | None:
    doc = ast.get_docstring(node)
    return doc[:300] if doc else None


def _call_chain_matches(call: ast.Call) -> bool:
    chain: list[str] = []
    node: ast.expr = call.func
    while isinstance(node, ast.Attribute):
        chain.append(node.attr)
        node = node.value
    chain.reverse()
    for known in _LLM_CALL_CHAINS:
        n = len(known)
        if tuple(chain[-n:]) == known:
            return True
    return False


def _function_has_llm_call(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(isinstance(child, ast.Call) and _call_chain_matches(child) for child in ast.walk(node))


def _referenced_in_tools_list(tree: ast.Module, name: str) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.keyword)):
            continue
        if isinstance(node, ast.keyword):
            if node.arg != "tools" or not isinstance(node.value, ast.List):
                continue
            elts = node.value.elts
        else:
            targets_match = any(isinstance(t, ast.Name) and t.id == "tools" for t in node.targets)
            if not targets_match or not isinstance(node.value, ast.List):
                continue
            elts = node.value.elts
        for elt in elts:
            if isinstance(elt, ast.Name) and elt.id == name:
                return True
    return False


def _base_name(base: ast.expr) -> str:
    if isinstance(base, ast.Attribute):
        return base.attr
    if isinstance(base, ast.Name):
        return base.id
    return ""


def _main_guard_callable(tree: ast.Module) -> str | None:
    """Return the entry-point marker for a top-level `if __name__ == "__main__":` guard.

    Returns the called function's name when the guard body is a simple call, else "__main__".
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        ):
            continue
        for stmt in node.body:
            call = stmt.value if isinstance(stmt, ast.Expr) else None
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                return call.func.id
        return "__main__"
    return None


def _detect_frameworks(tree: ast.Module) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix, label in _FRAMEWORK_IMPORTS.items():
                    if alias.name == prefix or alias.name.startswith(prefix + "."):
                        found.add(label)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for prefix, label in _FRAMEWORK_IMPORTS.items():
                if node.module == prefix or node.module.startswith(prefix + "."):
                    found.add(label)
    return found


def _scan_file(path: Path, root: Path) -> tuple[list[dict[str, Any]], set[str]]:
    try:
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return [], set()

    frameworks = _detect_frameworks(tree)
    main_guard_callable = _main_guard_callable(tree)
    symbols: list[dict[str, Any]] = []

    def _add(qualname: str, kind: str, node: ast.FunctionDef | ast.AsyncFunctionDef, decorators: list[str]) -> None:
        symbols.append(
            {
                "qualname": qualname,
                "kind": kind,
                "signature": _render_signature(node.args),
                "docstring": _docstring(node),
                "decorators": decorators,
                "lineno": node.lineno,
            }
        )

    def _visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                if any(_base_name(base) in {"Agent", "Crew", "Graph", "Workflow"} for base in child.bases) or any(
                    any(marker in _base_name(base) for marker in _AGENT_BASE_MARKERS) for base in child.bases
                ):
                    qualname = ".".join((*scope, child.name))
                    symbols.append(
                        {
                            "qualname": qualname,
                            "kind": "agent_class",
                            "signature": None,
                            "docstring": _docstring(child),
                            "decorators": [_decorator_name(d) for d in child.decorator_list],
                            "lineno": child.lineno,
                        }
                    )
                _visit(child, (*scope, child.name))
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = ".".join((*scope, child.name))
                decorator_names = [_decorator_name(d) for d in child.decorator_list]

                if any(_is_entry_decorator(name) for name in decorator_names) or child.name == "main":
                    _add(qualname, "entry", child, decorator_names)
                elif any(_is_route_decorator(d) for d in child.decorator_list):
                    _add(qualname, "route", child, decorator_names)
                elif any(_is_tool_decorator(name) for name in decorator_names) or _referenced_in_tools_list(
                    tree, child.name
                ):
                    _add(qualname, "tool", child, decorator_names)
                elif _function_has_llm_call(child):
                    _add(qualname, "llm_call", child, decorator_names)

                _visit(child, (*scope, child.name))

    _visit(tree, ())

    if main_guard_callable and not any(
        sym["kind"] == "entry" and sym["qualname"] == main_guard_callable for sym in symbols
    ):
        symbols.append(
            {
                "qualname": main_guard_callable,
                "kind": "entry",
                "signature": None,
                "docstring": None,
                "decorators": [],
                "lineno": 1,
            }
        )

    symbols.sort(key=lambda s: s["lineno"])
    return symbols, frameworks


def scan(root: str = ".") -> dict[str, Any]:
    root_path = Path(root).resolve()
    py_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if not _skip_dir(d)]
        for filename in filenames:
            if filename.endswith(".py"):
                py_files.append(Path(dirpath) / filename)
    py_files.sort()

    files_result: list[dict[str, Any]] = []
    frameworks: set[str] = set()
    with ThreadPoolExecutor() as pool:
        results = list(pool.map(lambda p: (p, *_scan_file(p, root_path)), py_files))

    for path, symbols, file_frameworks in results:
        frameworks |= file_frameworks
        if symbols:
            files_result.append({"path": str(path.relative_to(root_path)), "symbols": symbols})

    files_result.sort(key=lambda f: f["path"])

    return {
        "schema_version": 1,
        "repo_sha": _git_revision(root_path),
        "frameworks_detected": sorted(frameworks),
        "files": files_result,
    }
