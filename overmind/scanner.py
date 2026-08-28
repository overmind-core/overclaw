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
    "tests",
    "test",
    "testing",
    "e2e",
    "fixtures",
    "docs",
    "examples",
    "benchmarks",
    "scripts",
    "migrations",
}


def _skip_dir(name: str) -> bool:
    if name.startswith("."):
        return True
    if name in _SKIP_DIRS:
        return True
    return name.startswith("venv") or name.startswith(".venv")


def _skip_file(name: str) -> bool:
    return name == "conftest.py" or name.startswith("test_") or name.endswith("_test.py")


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
_LLM_CALL_CHAINS = {
    ("chat", "completions", "create"),
    ("messages", "create"),
    ("generate_content",),
    ("responses", "create"),
}

_LLM_CLIENT_MODULE_PREFIXES = ("openai", "anthropic", "google.genai", "google.generativeai", "litellm", "agno")

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


def _call_tail_name(func: ast.expr) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _call_base_names(func: ast.expr) -> set[str]:
    return {n.id for n in ast.walk(func) if isinstance(n, ast.Name)}


def _is_llm_client_module(module: str) -> bool:
    for prefix in _LLM_CLIENT_MODULE_PREFIXES:
        if module == prefix or module.startswith(prefix + "."):
            return True
    return module.startswith("langchain")


def _collect_llm_imports(tree: ast.Module) -> tuple[set[str], bool]:
    """Names imported from known LLM client packages, plus whether litellm was imported."""
    names: set[str] = set()
    litellm_imported = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_llm_client_module(alias.name):
                    names.add(alias.asname or alias.name.split(".")[0])
                    litellm_imported = litellm_imported or alias.name.split(".")[0] == "litellm"
        elif isinstance(node, ast.ImportFrom) and node.module and _is_llm_client_module(node.module):
            names.update(alias.asname or alias.name for alias in node.names)
            litellm_imported = litellm_imported or node.module.split(".")[0] == "litellm"
    return names, litellm_imported


def _function_has_llm_call(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    imported_names: set[str],
    litellm_imported: bool,
) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if _call_chain_matches(child):
            return True
        if imported_names and _call_base_names(child.func) & imported_names:
            return True
        if litellm_imported and _call_tail_name(child.func) in {"completion", "acompletion"}:
            return True
    return False


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


def _module_scope(path: Path, root: Path) -> tuple[str, ...]:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return tuple(parts)


def _scan_file(path: Path, root: Path) -> tuple[list[dict[str, Any]], set[str]]:
    try:
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
        lines = text.splitlines()
    except (OSError, SyntaxError, UnicodeDecodeError):
        return [], set()

    frameworks = _detect_frameworks(tree)
    main_guard_callable = _main_guard_callable(tree)
    llm_imports, litellm_imported = _collect_llm_imports(tree)
    symbols: list[dict[str, Any]] = []
    module_scope = _module_scope(path, root)

    def _qualname(scope: tuple[str, ...], name: str) -> str:
        return ".".join((*module_scope, *scope, name))

    def _add(qualname: str, kind: str, node: ast.FunctionDef | ast.AsyncFunctionDef, decorators: list[str]) -> None:
        symbols.append({
            "qualname": qualname,
            "kind": kind,
            "signature": _render_signature(node.args),
            "docstring": _docstring(node),
            "decorators": decorators,
            "lineno": node.lineno,
            "source_line": lines[node.lineno - 1].strip() if 0 < node.lineno <= len(lines) else "",
        })

    def _visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                if any(marker in _base_name(base) for base in child.bases for marker in _AGENT_BASE_MARKERS):
                    qualname = _qualname(scope, child.name)
                    symbols.append({
                        "qualname": qualname,
                        "kind": "agent_class",
                        "signature": None,
                        "docstring": _docstring(child),
                        "decorators": [_decorator_name(d) for d in child.decorator_list],
                        "lineno": child.lineno,
                        "source_line": lines[child.lineno - 1].strip() if 0 < child.lineno <= len(lines) else "",
                    })
                _visit(child, (*scope, child.name))
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = _qualname(scope, child.name)
                decorator_names = [_decorator_name(d) for d in child.decorator_list]

                if any(_is_entry_decorator(name) for name in decorator_names) or (child.name == "main" and not scope):
                    _add(qualname, "entry", child, decorator_names)
                elif any(_is_route_decorator(d) for d in child.decorator_list):
                    _add(qualname, "route", child, decorator_names)
                elif any(_is_tool_decorator(name) for name in decorator_names) or _referenced_in_tools_list(
                    tree, child.name
                ):
                    _add(qualname, "tool", child, decorator_names)
                elif _function_has_llm_call(child, llm_imports, litellm_imported):
                    _add(qualname, "llm_call", child, decorator_names)

                _visit(child, (*scope, child.name))

    _visit(tree, ())

    main_guard_qualname = _qualname((), main_guard_callable) if main_guard_callable else ""
    if main_guard_qualname and not any(
        sym["kind"] == "entry" and sym["qualname"] == main_guard_qualname for sym in symbols
    ):
        symbols.append({
            "qualname": main_guard_qualname,
            "kind": "entry",
            "signature": None,
            "docstring": None,
            "decorators": [],
            "lineno": 1,
            "source_line": "",
        })

    symbols.sort(key=lambda s: s["lineno"])
    return symbols, frameworks


def _count_py_files(path: Path) -> int:
    return sum(1 for _, _, filenames in os.walk(path) for f in filenames if f.endswith(".py"))


def scan(root: str = ".") -> dict[str, Any]:
    root_path = Path(root).resolve()
    py_files: list[Path] = []
    skipped_dirs = 0
    skipped_files = 0
    for dirpath, dirnames, filenames in os.walk(root_path):
        kept: list[str] = []
        for d in dirnames:
            if _skip_dir(d):
                skipped_dirs += 1
                skipped_files += _count_py_files(Path(dirpath) / d)
            else:
                kept.append(d)
        dirnames[:] = kept
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            if _skip_file(filename):
                skipped_files += 1
                continue
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
        "skipped": {"directories": skipped_dirs, "files": skipped_files},
    }
