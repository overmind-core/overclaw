"""AST-based instrumentation for overmind observe() tracing.

Transforms agent source code to:
  1. Add ``from overmind import init as overmind_init, observe`` and
     ``overmind_init()`` at the top of the file.
  2. Add ``@observe()`` decorator above every function definition.

The transform operates on source text (not AST rewriting) so it
preserves formatting, comments, and decorators.
"""

from __future__ import annotations

import ast
import re


def instrument_source(source: str) -> str:
    """Apply all instrumentation transforms to *source* and return the result."""
    source = _add_overmind_imports(source)
    source = _replace_call_llm(source)
    source = _replace_call_tool(source)
    source = _add_observe_decorators(source)
    return source


# ---------------------------------------------------------------------------
# 1. Add overmind imports + init at the top
# ---------------------------------------------------------------------------

_OVERMIND_IMPORT = "from overmind import init as overmind_init, observe"
_OVERMIND_INIT = "overmind_init()"


def _add_overmind_imports(source: str) -> str:
    # If already has the full import with observe, nothing to do.
    if _OVERMIND_IMPORT in source:
        return source

    # If overmind is imported but without ``observe``, patch the import.
    if "from overmind import" in source and "observe" not in source:
        return re.sub(
            r"(from overmind import .+)",
            r"\1, observe",
            source,
            count=1,
        )

    if "from overmind import" in source or "import overmind" in source:
        # Already imported (and observe is present or unused) — skip.
        return source

    lines = source.split("\n")

    # Use AST to reliably skip past the module docstring and any
    # ``from __future__`` imports (which must stay at the very top).
    # The previous line-by-line heuristic could inject our import
    # *inside* a multi-line docstring, leaving ``observe`` undefined
    # and crashing the instrumented module at import time.
    insert_idx = 0
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None

    if tree is not None and tree.body:
        body = tree.body
        i = 0
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            insert_idx = first.end_lineno or insert_idx
            i = 1
        while i < len(body):
            node = body[i]
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                insert_idx = node.end_lineno or insert_idx
                i += 1
            else:
                break

    # Bookend our injected block with blank lines so deinstrumentation
    # yields a clean round-trip (the strip regexes consume the trailing
    # ``\n`` of the line they remove; having the blanks here means the
    # blank line that was originally adjacent to the removed lines
    # survives).
    block = [_OVERMIND_IMPORT, _OVERMIND_INIT]
    prev_is_blank = insert_idx == 0 or lines[insert_idx - 1].strip() == ""
    next_is_blank = insert_idx >= len(lines) or lines[insert_idx].strip() == ""
    if not prev_is_blank:
        block.insert(0, "")
    if not next_is_blank:
        block.append("")

    for offset, text in enumerate(block):
        lines.insert(insert_idx + offset, text)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Replace call_llm(...) → litellm.completion(...)
# ---------------------------------------------------------------------------


def _replace_call_llm(source: str) -> str:
    """Replace ``call_llm(`` with ``litellm.completion(`` everywhere."""
    if "call_llm(" not in source:
        return source

    source = source.replace("call_llm(", "litellm.completion(")

    if "import litellm" not in source:
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith(("import ", "from ")):
                continue
            if line.strip() == _OVERMIND_INIT:
                continue
            if line.strip() == "" or line.strip().startswith("#"):
                continue
            lines.insert(i, "import litellm")
            break
        source = "\n".join(lines)

    return source


# ---------------------------------------------------------------------------
# 3. Replace call_tool(name, args, fn) → fn(**args)
# ---------------------------------------------------------------------------

_CALL_TOOL_RE = re.compile(
    r"""call_tool\(\s*
        (?P<name>[^,]+),\s*   # first arg: name (ignored)
        (?P<args>[^,]+),\s*   # second arg: args dict
        (?P<fn>[^)]+)          # third arg: callable
    \)""",
    re.VERBOSE,
)


def _replace_call_tool(source: str) -> str:
    """Replace ``call_tool(name, args, fn)`` with ``fn(**args)``."""
    if "call_tool(" not in source:
        return source

    def _sub(m: re.Match) -> str:
        args = m.group("args").strip()
        fn = m.group("fn").strip()
        return f"{fn}(**{args})"

    return _CALL_TOOL_RE.sub(_sub, source)


# ---------------------------------------------------------------------------
# 4. Add @observe() decorator to every function
# ---------------------------------------------------------------------------


_ENUM_LIKE_BASES = frozenset({
    "Enum",
    "IntEnum",
    "StrEnum",
    "Flag",
    "IntFlag",
    "NamedTuple",
    "TypedDict",
})


def _add_observe_decorators(source: str) -> str:
    """Add ``@observe()`` above every ``def`` / ``async def`` that lacks it.

    Skips methods inside enum-like classes (``IntEnum``, ``Enum``, etc.)
    because their metaclass restricts name lookup in the class body,
    making module-level ``observe`` invisible during class construction.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.split("\n")
    func_lines: list[int] = []

    skip_classes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = ""
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name in _ENUM_LIKE_BASES:
                    skip_classes.add(id(node))
                    break

    def _visit(node: ast.AST, inside_skip: bool = False) -> None:
        if isinstance(node, ast.ClassDef):
            child_skip = inside_skip or id(node) in skip_classes
            for child in ast.iter_child_nodes(node):
                _visit(child, inside_skip=child_skip)
            return

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not inside_skip:
            already_has = any(
                (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "observe")
                or (isinstance(d, ast.Name) and d.id == "observe")
                or (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "observe")
                for d in node.decorator_list
            )
            if not already_has:
                func_lines.append(node.lineno)

        for child in ast.iter_child_nodes(node):
            _visit(child, inside_skip=inside_skip)

    _visit(tree)

    for lineno in sorted(func_lines, reverse=True):
        idx = lineno - 1
        existing_line = lines[idx]
        indent = len(existing_line) - len(existing_line.lstrip())
        decorator = " " * indent + "@observe()"
        lines.insert(idx, decorator)

    return "\n".join(lines)


def is_instrumented(source: str) -> bool:
    """Return True if the source already has overmind instrumentation.

    Requires both the import (with ``observe``) and at least one
    ``@observe()`` decorator to be present.
    """
    has_import = "observe" in source and ("from overmind import" in source or "import overmind" in source)
    has_decorator = "@observe()" in source
    return has_import and has_decorator


_OBSERVE_DECORATOR_RE = re.compile(r"^[ \t]*@observe\(\s*\)\s*\n", re.MULTILINE)
# Only match the exact import form we inject in :func:`instrument_source` so
# we don't clobber a user's pre-existing ``from overmind import ...`` line.
_OVERMIND_IMPORT_RE = re.compile(
    r"^[ \t]*from\s+overmind\s+import\s+[^\n]*\b(?:overmind_init|observe)\b"
    r"[^\n]*\n",
    re.MULTILINE,
)
_OVERMIND_INIT_RE = re.compile(r"^[ \t]*overmind_init\s*\(\s*\)\s*\n", re.MULTILINE)


def deinstrument_source(source: str) -> str:
    """Reverse the additions made by :func:`instrument_source`.

    Strips ``@observe()`` decorators, the ``overmind`` import line,
    and bare ``overmind_init()`` calls so the cleaned-up source matches
    what the user originally wrote (minus any pre-existing instrumentation
    they may have had, which is intentionally unsupported here).

    This is used by ``overmind optimize`` when committing optimized
    sources back to the user's original agent files.
    """
    source = _OBSERVE_DECORATOR_RE.sub("", source)
    source = _OVERMIND_IMPORT_RE.sub("", source)
    source = _OVERMIND_INIT_RE.sub("", source)
    return source


def instrument_directory(directory: str) -> int:
    """Apply ``instrument_source`` to every ``.py`` file under *directory*.

    Files that are already instrumented (per :func:`is_instrumented`) are
    skipped.  Returns the number of files modified.
    """
    from pathlib import Path

    root = Path(directory)
    count = 0
    for py_file in root.rglob("*.py"):
        try:
            source = py_file.read_text(encoding="utf-8")
        except Exception:
            continue
        if is_instrumented(source):
            continue
        new_source = instrument_source(source)
        if new_source != source:
            py_file.write_text(new_source, encoding="utf-8")
            count += 1
    return count
