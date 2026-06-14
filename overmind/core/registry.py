"""Agent registry — reads and writes ``.overmind/agents.toml``.

Each entry maps a short *name* to a Python *module:function* entrypoint.
The dotted module path is resolved to a file path relative to the project root
(the directory that **contains** ``.overmind/``). The CLI requires ``.overmind/`` to
exist before any command other than ``overmind init`` (:func:`require_overmind_initialized`).

Example ``.overmind/agents.toml``::

    # Overmind agent registry

    agents = [
        { name = "lead-qualification", entrypoint = "agents.agent1.sample_agent:run", id = "uuid-..." },
    ]
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import tomlkit
from tomlkit import inline_table

from overmind.core.constants import OVERMIND_DIR_NAME

# ---------------------------------------------------------------------------
# Project root (Overmind state directory only)
# ---------------------------------------------------------------------------


def _find_overmind_project_root_from_cwd() -> Path | None:
    """Return the directory that contains the Overmind state dir, or ``None``."""
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / OVERMIND_DIR_NAME).is_dir():
            return parent
    return None


def _find_project_root_from_cwd() -> Path:
    """Directory containing ``.overmind/`` (must already exist)."""
    found = _find_overmind_project_root_from_cwd()
    if found is not None:
        return found
    print(
        f"\n  Error: No `{OVERMIND_DIR_NAME}/` directory found in this directory or any parent.\n\n"
        "  Change to your project root and run:\n"
        "    overmind init\n\n"
        f"  That creates `{OVERMIND_DIR_NAME}/` and configures API keys and models.\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


def init_project_root() -> Path:
    """Directory where ``overmind init`` creates ``.overmind/`` (current working directory).

    Use this for ``init`` only — it does not require ``.overmind`` to exist yet.
    """
    return Path.cwd().resolve()


def project_root() -> Path:
    """Root of the current Overmind project (directory containing ``.overmind/``)."""
    return _find_project_root_from_cwd()


def require_overmind_initialized() -> None:
    """Exit unless an ancestor of cwd contains ``.overmind/``.

    Call from the CLI before any subcommand except ``overmind init``.
    """
    if _find_overmind_project_root_from_cwd() is None:
        print(
            f"\n  Error: No `{OVERMIND_DIR_NAME}/` directory found in this directory or any parent.\n\n"
            "  Change to your project root and run:\n"
            "    overmind init\n",
            file=sys.stderr,
        )
        raise SystemExit(1)


def project_root_from_agent_file(agent_path: str | Path) -> Path | None:
    """Find the project root by walking parents of the agent entry file.

    Used by the bundler when the agent lives deep under ``agents/...`` so that
    ``parent.parent`` of the entry file is *not* the repo root (e.g.
    ``original_agent/runner.py`` → ``agents/agent3``), which would break import
    resolution for ``agents.agent3...`` modules.
    """
    p = Path(agent_path).resolve()
    start = p.parent if p.is_file() else p
    for ancestor in [start, *start.parents]:
        if (ancestor / OVERMIND_DIR_NAME).is_dir():
            return ancestor
    return None


def _agents_registry_path() -> Path:
    """Registered agent names and entrypoints (``agents.toml`` under state dir)."""
    return project_root() / OVERMIND_DIR_NAME / "agents.toml"


# ---------------------------------------------------------------------------
# TOML parsing / serialization
# ---------------------------------------------------------------------------


def _str_val(v) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _raw_agents_to_entries(raw) -> list[dict[str, str]]:
    """Normalize TOML ``agents`` value into registry entries."""
    if raw is None:
        return []

    if isinstance(raw, list):
        out: list[dict[str, str]] = []
        for item in raw:
            row = dict(item) if hasattr(item, "keys") else {}
            name = _str_val(row.get("name"))
            ep = _str_val(row.get("entrypoint"))
            if name and ep:
                normalized = {"name": name, "entrypoint": ep}
                agent_id = _str_val(row.get("id"))
                if agent_id:
                    normalized["id"] = agent_id
                out.append(normalized)
        return out

    if isinstance(raw, dict):
        out = []
        for name, data in raw.items():
            if isinstance(data, dict):
                ep = _str_val(data.get("entrypoint"))
                aid = _str_val(data.get("id"))
            else:
                ep = ""
                aid = ""
            name_s = _str_val(name)
            if name_s and ep:
                normalized = {"name": name_s, "entrypoint": ep}
                if aid:
                    normalized["id"] = aid
                out.append(normalized)
        return out

    return []


def _entries_to_toml_array(entries: list[dict[str, str]]) -> tomlkit.items.Array:
    """Build a multiline TOML array of inline tables."""
    arr = tomlkit.array()
    arr.multiline(True)
    for e in sorted(entries, key=lambda x: x["name"].lower()):
        row = inline_table()
        row["name"] = e["name"]
        row["entrypoint"] = e["entrypoint"]
        eid = _str_val(e.get("id"))
        if eid:
            row["id"] = eid
        arr.append(row)
    return arr


def _empty_agents_array() -> tomlkit.items.Array:
    arr = tomlkit.array()
    arr.multiline(True)
    return arr


def _read_registry_entries() -> list[dict[str, str]]:
    path = _agents_registry_path()
    if not path.is_file():
        return []
    doc = tomlkit.loads(path.read_text(encoding="utf-8"))
    return _raw_agents_to_entries(doc.get("agents"))


def _write_registry_entries(entries: list[dict[str, str]]) -> None:
    path = _agents_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = tomlkit.document()
    doc.add(tomlkit.comment(" Overmind agent registry — use: overmind agent register / list / remove / update "))
    doc.add(tomlkit.nl())
    doc["agents"] = _entries_to_toml_array(entries) if entries else _empty_agents_array()
    path.write_text(tomlkit.dumps(doc), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entrypoint parsing and validation
# ---------------------------------------------------------------------------


def entrypoint_from_path(agent_path: str, fn_name: str) -> str:
    """Build a registry entrypoint (``module.path:function``) from a source file.

    The inverse of :func:`_module_to_file`: an agent file path (relative to the
    project root, or absolute under it) plus a function name becomes the dotted
    ``module:function`` string stored in ``agents.toml``. Directory segments that
    start with a dot cannot round-trip through dotted notation, so those fall
    back to the slash-path form that :func:`_module_to_file` also accepts.
    """
    p = Path(agent_path)
    try:
        root = project_root()
        if p.is_absolute():
            p = p.relative_to(root)
    except (ValueError, SystemExit):
        # Outside the project root, or no project root yet — keep the path as
        # given; the dotted/slash conversion below still produces a usable ref.
        pass
    if p.suffix.lower() in _SUPPORTED_EXTENSIONS:
        p = p.with_suffix("")
    parts = [seg for seg in p.parts if seg not in (".", "")]
    module = p.as_posix() if any(seg.startswith(".") for seg in parts) else ".".join(parts)
    fn = (fn_name or "").strip()
    return f"{module}:{fn}" if fn else module


def parse_entrypoint(entrypoint: str) -> tuple[str, str]:
    """Split 'module.path:function' into (module, function_name).

    Raises ValueError if the format is invalid.
    """
    if ":" not in entrypoint:
        raise ValueError(
            f"Invalid entrypoint '{entrypoint}'. "
            "Expected format: module.path:function_name  "
            "(e.g. agents.agent1.sample_agent:run)"
        )
    module, fn = entrypoint.rsplit(":", 1)
    module, fn = module.strip(), fn.strip()
    if not module or not fn:
        raise ValueError(f"Invalid entrypoint '{entrypoint}'. Both module path and function name must be non-empty.")
    return module, fn


_SUPPORTED_EXTENSIONS = (".py", ".js", ".ts", ".mjs", ".mts")


def _module_to_file(module: str, root: Path) -> Path:
    """Convert a module reference to an absolute file path.

    Accepts two formats:
    - Dotted Python module path: ``agents.my_agent.main``
    - Relative file path (with ``/``): ``.overmind/agents/gsec/.../file``

    The second form is used for generated wrappers whose directory name
    starts with a dot, which cannot survive a round-trip through dotted
    module notation.

    Tries ``.py`` first, then JS/TS extensions, so existing Python
    agents keep working unchanged.
    """
    if "/" in module or "\\" in module:
        base = root / Path(module)
    else:
        parts = module.split(".")
        base = root / Path(*parts)
    for ext in _SUPPORTED_EXTENSIONS:
        candidate = base.with_suffix(ext)
        if candidate.exists():
            return candidate
    return base.with_suffix(".py")


class EntrypointNotFoundError(Exception):
    """The entrypoint file exists but the expected function is missing.

    Raised by :func:`resolve_entrypoint` so callers can offer wrapper
    generation instead of hard-exiting.
    """

    def __init__(self, file_path: Path, fn_name: str) -> None:
        self.file_path = file_path
        self.fn_name = fn_name
        super().__init__(f"Function '{fn_name}' not found in '{file_path}'")


class EntrypointSignatureError(Exception):
    """The entrypoint function exists but doesn't match Overmind's contract.

    Covers cases like missing input arguments or missing return value.
    Raised so callers can offer wrapper generation instead of hard-exiting.
    """

    def __init__(self, file_path: Path, fn_name: str, reason: str) -> None:
        self.file_path = file_path
        self.fn_name = fn_name
        self.reason = reason
        super().__init__(f"Function '{fn_name}' in '{file_path}': {reason}")


def resolve_module_to_file(module: str) -> Path | None:
    """Resolve a bare module path (no ``:function`` part) to an existing file.

    Returns the absolute Path if a matching source file exists, otherwise None.
    Used when the user provides only a filename/module without specifying a
    function name.
    """
    root = project_root()
    file_path = _module_to_file(module, root)
    return file_path if file_path.exists() else None


def resolve_entrypoint_file(entrypoint: str) -> tuple[Path, str]:
    """Resolve the entrypoint to a file path without checking the function.

    Returns (resolved_file_path, function_name).

    Raises:
        ValueError: invalid entrypoint format or module file not found.
    """
    module, fn = parse_entrypoint(entrypoint)

    root = project_root()
    file_path = _module_to_file(module, root)

    if not file_path.exists():
        raise ValueError(
            f"Module '{module}' resolves to '{file_path}', "
            "which does not exist.\n"
            "  Check that the module path is correct and the file is present."
        )

    return file_path, fn


def resolve_entrypoint(entrypoint: str) -> tuple[Path, str]:
    """Validate and resolve an entrypoint string.

    Returns (resolved_file_path, function_name).

    Raises:
        ValueError: invalid entrypoint format or module file not found.
        EntrypointNotFoundError: file exists but the function is missing.
    """
    module, fn = parse_entrypoint(entrypoint)

    root = project_root()
    file_path = _module_to_file(module, root)

    if not file_path.exists():
        raise ValueError(
            f"Module '{module}' resolves to '{file_path}', "
            "which does not exist.\n"
            "  Check that the module path is correct and the file is present."
        )

    code = file_path.read_text(encoding="utf-8")
    ext = file_path.suffix.lower()

    if ext == ".py":
        found = f"def {fn}(" in code or f"def {fn} (" in code
    else:
        found = bool(
            re.search(
                rf"(?:function\s+{re.escape(fn)}\s*\(|"
                rf"(?:const|let|var)\s+{re.escape(fn)}\s*=|"
                rf"exports\.{re.escape(fn)}\s*=|"
                rf"module\.exports\s*=)",
                code,
            )
        )

    if not found:
        raise EntrypointNotFoundError(file_path, fn)

    if ext == ".py":
        _validate_python_entrypoint(code, fn, file_path)
    else:
        _validate_js_entrypoint(code, fn, file_path)

    return file_path, fn


def validate_entrypoint(entrypoint: str) -> tuple[Path, str]:
    """Validate entrypoint string, resolve it, and verify the file + function exist.

    Returns (resolved_file_path, function_name).
    Prints a clear error and raises SystemExit(1) on any failure.
    """
    try:
        return resolve_entrypoint(entrypoint)
    except ValueError as exc:
        print(f"\n  Error: {exc}\n", file=sys.stderr)
        raise SystemExit(1) from exc
    except EntrypointNotFoundError as exc:
        ext = exc.file_path.suffix.lower()
        fn = exc.fn_name
        hint = f"'def {fn}(input)'" if ext == ".py" else f"'function {fn}(input)' or 'module.exports = {{ {fn} }}'"
        print(
            f"\n  Error: Function '{fn}' not found in '{exc.file_path}'.\n"
            f"  Make sure your agent file defines {hint}.\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    except EntrypointSignatureError as exc:
        print(
            f"\n  Error: Entrypoint '{exc.fn_name}' in '{exc.file_path}' {exc.reason}.\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def _validate_python_entrypoint(code: str, fn: str, file_path: Path) -> None:
    """Ensure the Python entrypoint accepts input arguments and returns a value."""
    try:
        tree = ast.parse(code, filename=str(file_path))
    except SyntaxError:
        return

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != fn:
            continue

        args = node.args
        param_count = len(args.args) + len(args.posonlyargs) + len(args.kwonlyargs)
        has_var = args.vararg is not None or args.kwarg is not None
        if param_count == 0 and not has_var:
            raise EntrypointSignatureError(file_path, fn, "takes no input arguments")

        has_meaningful_return = False
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Return)
                and child.value is not None
                and not (isinstance(child.value, ast.Constant) and child.value.value is None)
            ):
                has_meaningful_return = True
                break

        if not has_meaningful_return:
            raise EntrypointSignatureError(file_path, fn, "does not return a value")

        break


def _validate_js_entrypoint(code: str, fn: str, file_path: Path) -> None:
    """Best-effort validation that a JS/TS entrypoint accepts args and returns a value."""
    fn_pattern = (
        rf"(?:function\s+{re.escape(fn)}\s*\(\s*\)|"
        rf"(?:const|let|var)\s+{re.escape(fn)}\s*=\s*(?:async\s*)?\(\s*\)\s*=>)"
    )
    if re.search(fn_pattern, code):
        raise EntrypointSignatureError(file_path, fn, "takes no input arguments")

    fn_body_match = re.search(
        rf"(?:function\s+{re.escape(fn)}\s*\([^)]*\)\s*\{{|"
        rf"(?:const|let|var)\s+{re.escape(fn)}\s*=\s*(?:async\s*)?\([^)]*\)\s*=>\s*\{{)",
        code,
    )
    if fn_body_match:
        start = fn_body_match.end()
        depth, i = 1, start
        while i < len(code) and depth > 0:
            if code[i] == "{":
                depth += 1
            elif code[i] == "}":
                depth -= 1
            i += 1
        body = code[start:i]
        if "return " not in body and "return\n" not in body:
            raise EntrypointSignatureError(file_path, fn, "does not return a value")


# ---------------------------------------------------------------------------
# Registry CRUD
# ---------------------------------------------------------------------------


def load_registry() -> dict[str, dict[str, str]]:
    """Load all registered agents from ``.overmind/agents.toml``.

    Returns a dict of::

        {
            "lead-qualification": {
                "entrypoint": "agents.agent1.sample_agent:run",
                "file_path": "/abs/path/agents/agent1/sample_agent.py",
                "fn_name": "run",
                "id": "uuid-..." or "",
            },
            ...
        }
    """
    root = project_root()
    entries = _read_registry_entries()

    result: dict[str, dict[str, str]] = {}
    for row in entries:
        name = row["name"]
        ep = row["entrypoint"]
        try:
            module, fn = parse_entrypoint(ep)
            file_path = str(_module_to_file(module, root))
        except ValueError:
            file_path = ""
            fn = ""
        result[name] = {
            "entrypoint": ep,
            "file_path": file_path,
            "fn_name": fn,
            "id": _str_val(row.get("id")),
        }
    return result


def resolve_agent(name: str) -> tuple[str, str]:
    """Resolve a registered agent name to (file_path, fn_name).

    Exits with a clear, actionable error if the name is not registered.
    """
    from rich.console import Console

    console = Console()
    registry = load_registry()

    if name not in registry:
        console.print(f"\n  [bold red]Error:[/bold red] '[bold]{name}[/bold]' is not a registered agent.\n")
        console.print(
            f"  Register it first:\n"
            f"    [bold]overmind agent register {name} <module:function>[/bold]\n\n"
            "  To see all registered agents:\n"
            "    [bold]overmind agent list[/bold]\n"
        )
        raise SystemExit(1)

    entry = registry[name]
    file_path = entry["file_path"]
    fn_name = entry["fn_name"]

    if not file_path or not Path(file_path).exists():
        console.print(
            f"\n  [bold red]Error:[/bold red] "
            f"Agent '[bold]{name}[/bold]' is registered but its file was not found:\n"
            f"  [dim]{file_path}[/dim]\n\n"
            "  Update the entrypoint:\n"
            f"    [bold]overmind agent update {name} <module:function>[/bold]\n"
        )
        raise SystemExit(1)

    return file_path, fn_name


def get_agent_id(name: str) -> str | None:
    """Return stored ``id`` for *name*, or ``None`` if absent."""
    entries = _read_registry_entries()
    for row in entries:
        if row.get("name") == name:
            agent_id = _str_val(row.get("id"))
            return agent_id or None
    return None


def save_agent(name: str, entrypoint: str, id: str | None = None) -> None:
    """Add or overwrite an agent in ``.overmind/agents.toml``.

    Preserves existing ``id`` for *name*.
    """
    entries = _read_registry_entries()
    existing = next((e for e in entries if e["name"] == name), None)
    preserved_id = _str_val((existing or {}).get("id"))
    final_id = _str_val(id) if id is not None else preserved_id
    filtered = [e for e in entries if e["name"] != name]
    row = {"name": name, "entrypoint": entrypoint}
    if final_id:
        row["id"] = final_id
    filtered.append(row)
    _write_registry_entries(filtered)


def register_agents(
    agents: list[dict[str, str]],
    *,
    prune: bool = False,
) -> list[dict[str, str]]:
    """Write many agents into ``.overmind/agents.toml`` in one pass (bulk register).

    Each item is ``{"name", "entrypoint", "id"?}``. This is the multi-agent
    counterpart of :func:`save_agent`, used when the server has *extracted* every
    agent in a codebase and the local registry should reflect all of them at
    once.

    Existing entries are merged by name: an incoming entry overwrites that name's
    entrypoint (and ``id`` when provided), names already present but absent from
    *agents* are preserved, and an incoming entry without an ``id`` keeps any id
    already on file. Pass ``prune=True`` to drop local entries not in *agents*.
    Returns the entries written.
    """
    by_name = {e["name"]: dict(e) for e in _read_registry_entries()}

    written: set[str] = set()
    for item in agents:
        name = _str_val(item.get("name"))
        entrypoint = _str_val(item.get("entrypoint"))
        if not name or not entrypoint:
            continue
        row = by_name.get(name, {})
        row["name"] = name
        row["entrypoint"] = entrypoint
        incoming_id = _str_val(item.get("id"))
        if incoming_id:
            row["id"] = incoming_id
        by_name[name] = row
        written.add(name)

    if prune:
        by_name = {name: row for name, row in by_name.items() if name in written}

    entries = list(by_name.values())
    _write_registry_entries(entries)
    return entries


def set_agent_id(name: str, id: str | None) -> None:
    """Set or clear stored ``id`` for *name* in ``agents.toml``."""
    entries = _read_registry_entries()
    updated = False
    for row in entries:
        if row.get("name") != name:
            continue
        if id:
            row["id"] = _str_val(id)
        else:
            row.pop("id", None)
        updated = True
        break
    if updated:
        _write_registry_entries(entries)


def remove_agent(name: str) -> None:
    """Remove an agent from ``.overmind/agents.toml``. Raises KeyError if not found."""
    entries = _read_registry_entries()
    filtered = [e for e in entries if e["name"] != name]
    if len(filtered) == len(entries):
        raise KeyError(name)

    _write_registry_entries(filtered)
