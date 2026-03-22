"""Agent registry — reads and writes ``.overclaw/agents.toml``.

Each entry maps a short *name* to a Python *module:function* entrypoint.
The dotted module path is resolved to a file path relative to the project root
(the directory that **contains** ``.overclaw/``). The CLI requires ``.overclaw/`` to
exist before any command other than ``overclaw init`` (:func:`require_overclaw_initialized`).

Example ``.overclaw/agents.toml``::

    # OverClaw agent registry

    agents = [
        { name = "lead-qualification", entrypoint = "agents.agent1.sample_agent:run", id = "uuid-..." },
    ]
"""

from __future__ import annotations

import sys
from pathlib import Path

import tomlkit
from tomlkit import inline_table

from overclaw.core.constants import OVERCLAW_DIR_NAME


# ---------------------------------------------------------------------------
# Project root (OverClaw state directory only)
# ---------------------------------------------------------------------------


def _find_overclaw_project_root_from_cwd() -> Path | None:
    """Return the directory that contains the OverClaw state dir, or ``None``."""
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / OVERCLAW_DIR_NAME).is_dir():
            return parent
    return None


def _find_project_root_from_cwd() -> Path:
    """Directory containing ``.overclaw/`` (must already exist)."""
    found = _find_overclaw_project_root_from_cwd()
    if found is not None:
        return found
    print(
        f"\n  Error: No `{OVERCLAW_DIR_NAME}/` directory found in this directory or any parent.\n\n"
        "  Change to your project root and run:\n"
        "    overclaw init\n\n"
        f"  That creates `{OVERCLAW_DIR_NAME}/` and configures API keys and models.\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


def init_project_root() -> Path:
    """Directory where ``overclaw init`` creates ``.overclaw/`` (current working directory).

    Use this for ``init`` only — it does not require ``.overclaw`` to exist yet.
    """
    return Path.cwd().resolve()


def project_root() -> Path:
    """Root of the current OverClaw project (directory containing ``.overclaw/``)."""
    return _find_project_root_from_cwd()


def require_overclaw_initialized() -> None:
    """Exit unless an ancestor of cwd contains ``.overclaw/``.

    Call from the CLI before any subcommand except ``overclaw init``.
    """
    if _find_overclaw_project_root_from_cwd() is None:
        print(
            f"\n  Error: No `{OVERCLAW_DIR_NAME}/` directory found in this directory or any parent.\n\n"
            "  Change to your project root and run:\n"
            "    overclaw init\n",
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
        if (ancestor / OVERCLAW_DIR_NAME).is_dir():
            return ancestor
    return None


def _agents_registry_path() -> Path:
    """Registered agent names and entrypoints (``agents.toml`` under state dir)."""
    return project_root() / OVERCLAW_DIR_NAME / "agents.toml"


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
    doc.add(
        tomlkit.comment(
            " OverClaw agent registry — use: overclaw agent register / list / remove / update "
        )
    )
    doc.add(tomlkit.nl())
    doc["agents"] = (
        _entries_to_toml_array(entries) if entries else _empty_agents_array()
    )
    path.write_text(tomlkit.dumps(doc), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entrypoint parsing and validation
# ---------------------------------------------------------------------------


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
        raise ValueError(
            f"Invalid entrypoint '{entrypoint}'. "
            "Both module path and function name must be non-empty."
        )
    return module, fn


def _module_to_file(module: str, root: Path) -> Path:
    """Convert a dotted module path to an absolute .py file path."""
    parts = module.split(".")
    return (root / Path(*parts)).with_suffix(".py")


def validate_entrypoint(entrypoint: str) -> tuple[Path, str]:
    """Validate entrypoint string, resolve it, and verify the file + function exist.

    Returns (resolved_file_path, function_name).
    Prints a clear error and raises SystemExit(1) on any failure.
    """
    try:
        module, fn = parse_entrypoint(entrypoint)
    except ValueError as exc:
        print(f"\n  Error: {exc}\n", file=sys.stderr)
        raise SystemExit(1) from exc

    root = project_root()
    file_path = _module_to_file(module, root)

    if not file_path.exists():
        print(
            f"\n  Error: Module '{module}' resolves to '{file_path}', "
            "which does not exist.\n"
            "  Check that the module path is correct and the file is present.\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

    code = file_path.read_text(encoding="utf-8")
    if f"def {fn}(" not in code and f"def {fn} (" not in code:
        print(
            f"\n  Error: Function '{fn}' not found in '{file_path}'.\n"
            f"  Make sure your agent file defines 'def {fn}(input)'.\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

    return file_path, fn


# ---------------------------------------------------------------------------
# Registry CRUD
# ---------------------------------------------------------------------------


def load_registry() -> dict[str, dict[str, str]]:
    """Load all registered agents from ``.overclaw/agents.toml``.

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
        console.print(
            f"\n  [bold red]Error:[/bold red] "
            f"'[bold]{name}[/bold]' is not a registered agent.\n"
        )
        console.print(
            f"  Register it first:\n"
            f"    [bold]overclaw agent register {name} <module:function>[/bold]\n\n"
            "  To see all registered agents:\n"
            "    [bold]overclaw agent list[/bold]\n"
        )
        raise SystemExit(1)

    entry = registry[name]
    file_path = entry["file_path"]
    fn_name = entry["fn_name"]

    if not Path(file_path).exists():
        console.print(
            f"\n  [bold red]Error:[/bold red] "
            f"Agent '[bold]{name}[/bold]' is registered but its file was not found:\n"
            f"  [dim]{file_path}[/dim]\n\n"
            "  Update the entrypoint:\n"
            f"    [bold]overclaw agent update {name} <module:function>[/bold]\n"
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
    """Add or overwrite an agent in ``.overclaw/agents.toml``.

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
    """Remove an agent from ``.overclaw/agents.toml``. Raises KeyError if not found."""
    entries = _read_registry_entries()
    filtered = [e for e in entries if e["name"] != name]
    if len(filtered) == len(entries):
        raise KeyError(name)

    _write_registry_entries(filtered)
