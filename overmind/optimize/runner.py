"""Language-agnostic agent runner with automatic environment provisioning.

Replaces the in-process ``importlib``-based execution with subprocess
isolation.  Each agent runs in its own interpreter (Python venv or
Node.js) so dependency conflicts are impossible and crash safety is
guaranteed by process boundaries.

Supported languages
-------------------
- **Python** — detected via ``*.py`` entry file.  Dependencies resolved
  from ``requirements.txt`` or ``pyproject.toml``.  Uses ``uv`` when
  available (10-100× faster), falls back to stdlib ``venv`` + ``pip``.
- **JavaScript / TypeScript** — detected via ``*.js`` / ``*.ts`` /
  ``*.mjs`` / ``*.mts`` entry file.  Dependencies from ``package.json``,
  installed with ``npm``.  TypeScript executed via ``npx tsx``.

I/O contract
------------
The agent entry file must expose a callable that accepts JSON on
**stdin** and writes JSON to **stdout**.  A thin wrapper script is
generated automatically so existing agents that define a plain Python
function (``def run(input: dict) -> dict``) or a Node ``module.exports``
function keep working without any modifications.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from overmind.optimize.shadow_runtime import ShadowConfig

logger = logging.getLogger("overmind.optimize.runner")


# ---------------------------------------------------------------------------
# Distributed tracing helpers
# ---------------------------------------------------------------------------


def _current_traceparent() -> str | None:
    """Return the current OTel span serialised as a W3C ``traceparent`` value.

    Used to inject trace context into subprocess environments so every
    per-case agent evaluation spans are linked back to the optimizer's
    parent span, forming a single unified trace instead of many orphan traces.

    Format: ``00-{32-hex trace_id}-{16-hex span_id}-{flags}``
    Returns ``None`` when there is no active, valid span.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx is None or not ctx.is_valid:
            return None
        flags = "01" if (ctx.trace_flags & 0x01) else "00"
        return f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-{flags}"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


class Language(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"

    @classmethod
    def from_path(cls, path: str | Path) -> Language:
        ext = Path(path).suffix.lower()
        _MAP = {
            ".py": cls.PYTHON,
            ".js": cls.JAVASCRIPT,
            ".mjs": cls.JAVASCRIPT,
            ".ts": cls.TYPESCRIPT,
            ".mts": cls.TYPESCRIPT,
        }
        lang = _MAP.get(ext)
        if lang is None:
            raise ValueError(f"Unsupported agent file extension '{ext}'. Supported: {', '.join(_MAP)}")
        return lang


# ---------------------------------------------------------------------------
# Runner output
# ---------------------------------------------------------------------------


@dataclass
class RunOutput:
    """Result of a single agent invocation."""

    success: bool
    data: Any = None
    error: str = ""
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


# ---------------------------------------------------------------------------
# Runner configuration
# ---------------------------------------------------------------------------


@dataclass
class RunnerConfig:
    timeout: int = 120
    extra_env: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Import-to-package-name mapping
# ---------------------------------------------------------------------------

_IMPORT_TO_PYPI: dict[str, str] = {
    "dotenv": "python-dotenv",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "gi": "PyGObject",
    "attr": "attrs",
    "serial": "pyserial",
    "usb": "pyusb",
    "wx": "wxPython",
    "Crypto": "pycryptodome",
    "jose": "python-jose",
    "magic": "python-magic",
    "dateutil": "python-dateutil",
    "lxml": "lxml",
    "skimage": "scikit-image",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "Bio": "biopython",
    "Levenshtein": "python-Levenshtein",
    "jwt": "PyJWT",
    "git": "GitPython",
    "github": "PyGithub",
    "telegram": "python-telegram-bot",
    "flask_cors": "Flask-Cors",
    "flask_sqlalchemy": "Flask-SQLAlchemy",
}

_PYTHON_STDLIB: frozenset[str] = (
    getattr(sys, "stdlib_module_names", frozenset())
    | frozenset(sys.builtin_module_names)
    | frozenset({
        "pkg_resources",
        "setuptools",
        "pip",
        "_thread",
    })
)


# ---------------------------------------------------------------------------
# Dependency detection
# ---------------------------------------------------------------------------


def has_dep_manifest(agent_dir: Path, language: Language) -> bool:
    """Return True if the agent directory has a dependency manifest file."""
    if language == Language.PYTHON:
        return any((agent_dir / f).is_file() for f in ("requirements.txt", "pyproject.toml", "setup.py"))
    return (agent_dir / "package.json").is_file()


def detect_external_imports(agent_dir: Path, entry_file: str, language: Language) -> list[str]:
    """Scan the entry file for non-stdlib, non-relative imports.

    Returns a de-duped list of top-level package names that appear to be
    external dependencies.  Filters out:
    - Python stdlib modules
    - Sibling ``.py`` files in the agent directory
    - Directories/packages in the project root (e.g. ``overmind``,
      ``examples``, ``tests``) — these are project-local, not PyPI deps
    - The ``overmind`` package itself (always available at runtime)
    """
    entry_path = agent_dir / entry_file
    if not entry_path.is_file():
        return []

    code = entry_path.read_text(encoding="utf-8")
    raw_imports = extract_imports(code, language)

    if language == Language.PYTHON:
        local_modules = {p.stem for p in agent_dir.rglob("*.py") if p.stem != "__init__"}

        project_root = _find_project_root(agent_dir)
        if project_root:
            for child in project_root.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    local_modules.add(child.name)
                elif child.is_file() and child.suffix == ".py":
                    local_modules.add(child.stem)

        local_modules.add("overmind")

        return [m for m in raw_imports if m not in _PYTHON_STDLIB and m not in local_modules]

    if language in (Language.JAVASCRIPT, Language.TYPESCRIPT):
        return [m for m in raw_imports if m not in (".", "..")]

    return raw_imports


def _find_project_root(start: Path) -> Path | None:
    """Walk up from *start* to find the directory containing ``.overmind/``."""
    current = start.resolve()
    for ancestor in [current, *current.parents]:
        if (ancestor / ".overmind").is_dir():
            return ancestor
    return None


def imports_to_package_names(imports: list[str], language: Language) -> list[str]:
    """Map import names to likely package manager names (PyPI / npm)."""
    if language == Language.PYTHON:
        return [_IMPORT_TO_PYPI.get(m, m) for m in imports]
    return list(imports)


def generate_requirements_txt(packages: list[str]) -> str:
    """Generate a requirements.txt content string (unpinned)."""
    return "\n".join(sorted(set(packages))) + "\n"


def generate_package_json(packages: list[str], agent_name: str = "agent") -> str:
    """Generate a minimal package.json content string."""
    pkg = {
        "name": agent_name,
        "version": "1.0.0",
        "private": True,
        "dependencies": {p: "*" for p in sorted(set(packages))},
    }
    return json.dumps(pkg, indent=2) + "\n"


class MissingDependenciesError(Exception):
    """Raised when an agent has external imports but no dependency manifest."""

    def __init__(
        self,
        agent_dir: Path,
        language: Language,
        imports: list[str],
    ) -> None:
        self.agent_dir = agent_dir
        self.language = language
        self.imports = imports
        manifest = "requirements.txt / pyproject.toml" if language == Language.PYTHON else "package.json"
        super().__init__(
            f"Agent in {agent_dir} imports {len(imports)} external package(s) "
            f"({', '.join(imports[:5])}{'…' if len(imports) > 5 else ''}) "
            f"but has no {manifest}. "
            f"Run 'overmind setup' to configure dependencies, or create "
            f"the file manually."
        )


# ---------------------------------------------------------------------------
# Deps-hash helpers
# ---------------------------------------------------------------------------

_PYTHON_DEP_FILES = ("requirements.txt", "pyproject.toml", "setup.py", "setup.cfg")
_JS_DEP_FILES = ("package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml")


def _hash_dep_files(agent_dir: Path, filenames: tuple[str, ...]) -> str:
    """SHA-256 over the concatenated contents of dependency files."""
    h = hashlib.sha256()
    for name in sorted(filenames):
        p = agent_dir / name
        if p.is_file():
            h.update(name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _read_cached_hash(marker: Path) -> str:
    if marker.is_file():
        return marker.read_text().strip()
    return ""


def _write_cached_hash(marker: Path, digest: str) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(digest)


# ---------------------------------------------------------------------------
# Agent-based environment provisioning
# ---------------------------------------------------------------------------

_ENV_SETUP_PROMPT = """\
You are setting up a development environment for a {language} project.

Project directory: {agent_dir}

Files in this directory:
{file_listing}

{manifest_contents}

Your task:
1. Determine which package manager / tool this project uses (look at
   lockfiles, pyproject.toml sections like [tool.poetry], package.json
   scripts, etc.).
2. Create an isolated environment:
   - For Python: create a virtualenv at `.venv` using the project's
     native tooling.  If the project uses poetry, run `poetry install`.
     If it uses uv, run `uv sync`.  If it uses pip, create a venv and
     `pip install -r requirements.txt` or `pip install .`.
   - For JavaScript/TypeScript: run the appropriate install command
     (npm install, yarn install, pnpm install, bun install) based on
     the lockfile present.
3. Verify the environment was created:
   - Python: confirm `.venv/bin/python` (or `.venv/Scripts/python.exe`
     on Windows) exists.
   - JS/TS: confirm `node_modules/` exists.

Constraints:
- Do NOT modify any source code files (*.py, *.js, *.ts, etc.).
- Do NOT run the agent itself.
- Only install dependencies and set up the environment.
- If a command fails, read the error, diagnose, and try an alternative
  approach.
- When done, print EXACTLY: ENV_SETUP_COMPLETE
"""

_ENV_SETUP_MODEL_ENV = "ENV_SETUP_MODEL"
_ENV_SETUP_DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"


def _get_env_setup_model() -> str | None:
    """Return the model to use for agent-based env setup, or None to skip."""
    explicit = os.environ.get(_ENV_SETUP_MODEL_ENV, "").strip()
    if explicit:
        return explicit

    analyzer = os.environ.get("ANALYZER_MODEL", "").strip()
    if analyzer:
        return analyzer

    try:
        import litellm  # noqa: F401

        return _ENV_SETUP_DEFAULT_MODEL
    except ImportError:
        return None


def _gather_project_context(agent_dir: Path) -> tuple[str, str]:
    """Build a file listing and manifest contents summary for the prompt."""
    lines: list[str] = []
    for child in sorted(agent_dir.iterdir()):
        if child.name.startswith(".") and child.name != ".python-version":
            continue
        kind = "dir/" if child.is_dir() else ""
        size = ""
        if child.is_file():
            size = f"  ({child.stat().st_size} bytes)"
        lines.append(f"  {child.name}{kind}{size}")
    file_listing = "\n".join(lines) if lines else "  (empty directory)"

    manifests: list[str] = []
    manifest_files = [
        "pyproject.toml",
        "requirements.txt",
        "setup.py",
        "setup.cfg",
        "Pipfile",
        "package.json",
        ".python-version",
        "Makefile",
    ]
    for name in manifest_files:
        p = agent_dir / name
        if p.is_file():
            content = p.read_text(encoding="utf-8", errors="replace")
            if len(content) > 3000:
                content = content[:3000] + "\n... (truncated)"
            manifests.append(f"--- {name} ---\n{content}")

    lockfiles = [
        "poetry.lock",
        "uv.lock",
        "pdm.lock",
        "Pipfile.lock",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "bun.lockb",
    ]
    for name in lockfiles:
        if (agent_dir / name).is_file():
            manifests.append(f"--- {name} --- (present, not shown)")

    manifest_contents = "\n\n".join(manifests) if manifests else "(no manifest files found)"
    return file_listing, manifest_contents


def _provision_with_agent(agent_dir: Path, language: Language) -> bool:
    """Use the coding agent to set up the environment.

    Returns True if the agent successfully provisioned the environment,
    False if it failed or was unavailable (caller should fall back to
    hardcoded logic).
    """
    model = _get_env_setup_model()
    if not model:
        logger.debug("No model available for agent-based env setup, skipping")
        return False

    try:
        from overmind.coding_agent.agent import run as run_coding_agent
    except ImportError:
        logger.debug("Coding agent not importable, skipping agent-based env setup")
        return False

    file_listing, manifest_contents = _gather_project_context(agent_dir)

    lang_label = "Python" if language == Language.PYTHON else "JavaScript/TypeScript"
    instruction = _ENV_SETUP_PROMPT.format(
        language=lang_label,
        agent_dir=str(agent_dir),
        file_listing=file_listing,
        manifest_contents=manifest_contents,
    )

    logger.info(
        "Using coding agent (%s) to provision %s environment in %s …",
        model,
        lang_label,
        agent_dir,
    )

    try:
        run_coding_agent(
            instruction=instruction,
            model=model,
            cwd=str(agent_dir),
            max_steps=15,
        )
    except Exception as exc:
        logger.warning("Agent-based env setup failed: %s", exc, exc_info=True)
        return False

    if language == Language.PYTHON:
        venv_py = _venv_python(agent_dir / ".venv")
        if venv_py.is_file():
            logger.info("Agent-based env setup succeeded (Python venv ready)")
            return True
        logger.warning("Agent ran but .venv/bin/python not found — falling back")
        return False

    if (agent_dir / "node_modules").is_dir():
        logger.info("Agent-based env setup succeeded (node_modules ready)")
        return True
    logger.warning("Agent ran but node_modules/ not found — falling back")
    return False


# ---------------------------------------------------------------------------
# Overmind SDK injection
# ---------------------------------------------------------------------------

_OVERMIND_PACKAGE = "overmind>=0.1.39"


def _ensure_overmind_sdk(venv_dir: Path, agent_dir: Path) -> None:
    """Ensure the pinned overmind is installed in the agent's venv."""
    py = _venv_python(venv_dir)
    if not py.is_file():
        return

    logger.info("Installing %s into agent venv …", _OVERMIND_PACKAGE)
    use_uv = bool(shutil.which("uv"))
    if use_uv:
        subprocess.run(
            ["uv", "pip", "install", "--python", str(py), _OVERMIND_PACKAGE],
            cwd=str(agent_dir),
            capture_output=True,
            check=False,
        )
    else:
        pip = _venv_pip(venv_dir)
        subprocess.run(
            [str(pip), "install", _OVERMIND_PACKAGE],
            cwd=str(agent_dir),
            capture_output=True,
            check=False,
        )


# ---------------------------------------------------------------------------
# Python environment provisioning (hardcoded fallback)
# ---------------------------------------------------------------------------


def _is_windows() -> bool:
    return platform.system() == "Windows"


def _venv_python(venv_dir: Path) -> Path:
    if _is_windows():
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_pip(venv_dir: Path) -> Path:
    if _is_windows():
        return venv_dir / "Scripts" / "pip.exe"
    return venv_dir / "bin" / "pip"


def _provision_python(agent_dir: Path) -> Path:
    """Ensure a venv exists with the agent's deps installed.

    Strategy:
    1. If deps haven't changed (hash match), skip entirely.
    2. Try the coding agent — it reads the project files and runs
       the right tool (poetry, uv, pip, pdm, etc.) automatically.
    3. If the agent isn't available or fails, fall back to hardcoded
       uv/pip logic that handles the most common cases.

    Returns the path to the venv's Python interpreter.  When no
    dependency files exist, returns the system interpreter that runs
    overmind itself (backward-compatible zero-isolation mode).
    """
    has_requirements = (agent_dir / "requirements.txt").is_file()
    has_pyproject = (agent_dir / "pyproject.toml").is_file()
    has_setup_py = (agent_dir / "setup.py").is_file()

    if not (has_requirements or has_pyproject or has_setup_py):
        return Path(sys.executable)

    venv_dir = agent_dir / ".venv"
    marker = venv_dir / ".overmind_deps_hash"
    current_hash = _hash_dep_files(agent_dir, _PYTHON_DEP_FILES)

    if venv_dir.exists() and _read_cached_hash(marker) == current_hash:
        py = _venv_python(venv_dir)
        if py.is_file():
            _ensure_overmind_sdk(venv_dir, agent_dir)
            logger.debug("Python venv up-to-date for %s", agent_dir)
            return py

    # --- Try agent-based provisioning first ---
    if _provision_with_agent(agent_dir, Language.PYTHON):
        py = _venv_python(venv_dir)
        if py.is_file():
            _ensure_overmind_sdk(venv_dir, agent_dir)
            _write_cached_hash(marker, current_hash)
            return py

    # --- Fallback: hardcoded uv / pip logic ---
    logger.info("Provisioning Python environment for %s (fallback) …", agent_dir)

    use_uv = bool(shutil.which("uv"))

    if use_uv:
        if has_pyproject:
            subprocess.run(
                ["uv", "sync", "--no-dev"],
                cwd=str(agent_dir),
                check=True,
                capture_output=True,
            )
        else:
            if not venv_dir.exists():
                subprocess.run(
                    ["uv", "venv", str(venv_dir)],
                    cwd=str(agent_dir),
                    check=True,
                    capture_output=True,
                )
            pip_args = ["uv", "pip", "install", "--python", str(_venv_python(venv_dir))]
            if has_requirements:
                pip_args += ["-r", "requirements.txt"]
            elif has_setup_py:
                pip_args += ["."]
            subprocess.run(
                pip_args,
                cwd=str(agent_dir),
                check=True,
                capture_output=True,
            )
    else:
        if not venv_dir.exists():
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                cwd=str(agent_dir),
                check=True,
                capture_output=True,
            )
        pip = str(_venv_pip(venv_dir))
        if has_requirements:
            subprocess.run(
                [pip, "install", "-r", "requirements.txt"],
                cwd=str(agent_dir),
                check=True,
                capture_output=True,
            )
        elif has_pyproject or has_setup_py:
            subprocess.run(
                [pip, "install", "."],
                cwd=str(agent_dir),
                check=True,
                capture_output=True,
            )

    _ensure_overmind_sdk(venv_dir, agent_dir)
    _write_cached_hash(marker, current_hash)
    logger.info("Python environment ready for %s", agent_dir)
    return _venv_python(venv_dir)


# ---------------------------------------------------------------------------
# JS/TS environment provisioning
# ---------------------------------------------------------------------------


def _provision_js(agent_dir: Path) -> None:
    """Install JS/TS dependencies if ``package.json`` exists and deps are stale.

    Strategy mirrors Python: try agent first, fallback to ``npm install``.
    """
    pkg_json = agent_dir / "package.json"
    if not pkg_json.is_file():
        return

    marker = agent_dir / "node_modules" / ".overmind_deps_hash"
    current_hash = _hash_dep_files(agent_dir, _JS_DEP_FILES)

    if (agent_dir / "node_modules").is_dir() and _read_cached_hash(marker) == current_hash:
        logger.debug("node_modules up-to-date for %s", agent_dir)
        return

    # --- Try agent-based provisioning first ---
    if _provision_with_agent(agent_dir, Language.JAVASCRIPT):
        _write_cached_hash(marker, current_hash)
        return

    # --- Fallback: npm install ---
    logger.info("Provisioning JS environment for %s (fallback) …", agent_dir)
    subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=str(agent_dir),
        check=True,
        capture_output=True,
    )
    _write_cached_hash(marker, current_hash)
    logger.info("JS environment ready for %s", agent_dir)


# ---------------------------------------------------------------------------
# Wrapper script generation
# ---------------------------------------------------------------------------

_PYTHON_WRAPPER = """\
{shadow_bootstrap}
import json, sys, os, io, asyncio, inspect, importlib.util, traceback
_cwd = os.getcwd()
if _cwd not in sys.path:
    sys.path.insert(0, _cwd)
if os.environ.get("OVERMIND_API_KEY"):
    try:
        from overmind import init as overmind_init
        overmind_init()
    except (ImportError, SystemExit, RuntimeError):
        # ImportError  — overmind SDK not installed in this environment.
        # SystemExit / RuntimeError — init failure; swallow so the agent can still run.
        pass

def _ocl_force_flush():
    try:
        from opentelemetry import trace as _otel_trace
        _provider = _otel_trace.get_tracer_provider()
        if hasattr(_provider, "force_flush"):
            _provider.force_flush(timeout_millis=3000)
    except Exception:
        pass

# Preflight import so module-level failures surface with a clear marker
# (argparse-on-sys.argv crashes, load_dotenv misses, broken imports, …)
try:
    spec = importlib.util.spec_from_file_location("_agent", {entry_path!r})
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
except SystemExit as _ocl_sysexit:
    sys.stderr.write(
        "__OVERMIND_IMPORT_ERROR__\\n"
        "Agent module called sys.exit({{0}}) at import time (typical for "
        "module-level argparse.parse_args). Move CLI parsing behind "
        "`if __name__ == '__main__':`.\\n".format(_ocl_sysexit.code)
    )
    raise
except Exception as _ocl_imp_exc:
    sys.stderr.write("__OVERMIND_IMPORT_ERROR__\\n")
    traceback.print_exc()
    raise
try:
    fn = getattr(mod, {fn_name!r})
except AttributeError:
    sys.stderr.write(
        "__OVERMIND_ENTRYPOINT_MISSING__\\n"
        "Module loaded but function {fn_name!r} is not defined.\\n"
    )
    raise
data = json.loads(sys.stdin.read())

sig = inspect.signature(fn)
params = list(sig.parameters.values())
_param_names = [p.name for p in params if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)]
_single_dict_param = (
    len(params) == 1
    and params[0].annotation in (dict, inspect.Parameter.empty)
)

# Schema-aware dispatch: prefer kwargs when the dict's keys are a subset of
# the function's parameter names.  Gracefully falls back to positional
# single-dict if a TypeError surfaces — this recovers from ambiguous cases
# without the optimiser having to re-guess.
def _ocl_invoke(fn, data):
    if not isinstance(data, dict):
        return fn(data)
    if _single_dict_param:
        return fn(data)
    if params and set(data.keys()).issubset(set(_param_names)):
        try:
            return fn(**data)
        except TypeError:
            pass
    try:
        return fn(**data)
    except TypeError:
        return fn(data)

_real_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    result = _ocl_invoke(fn, data)
    if inspect.isawaitable(result):
        result = asyncio.run(result)
finally:
    _agent_prints = sys.stdout.getvalue()
    sys.stdout = _real_stdout
if _agent_prints:
    sys.stderr.write(_agent_prints)
_ocl_force_flush()
_MARKER = "\\n__OVERMIND_RESULT__\\n"
sys.stdout.write(_MARKER)
if isinstance(result, str):
    sys.stdout.write(result)
else:
    json.dump(result, sys.stdout, default=str)
sys.stdout.write(_MARKER)
"""

_JS_WRAPPER = """\
const {{ readFileSync }} = require("fs");
const mod = require({entry_path});
const fn = typeof mod === "function" ? mod : (mod.default || mod[{fn_name}]);
const data = JSON.parse(readFileSync("/dev/stdin", "utf8"));
const _origWrite = process.stdout.write.bind(process.stdout);
const _buf = [];
process.stdout.write = (chunk, enc, cb) => {{ _buf.push(chunk); if (cb) cb(); return true; }};
const _call = (fn.length > 1 && typeof data === "object" && data !== null && !Array.isArray(data))
  ? fn(...Object.values(data))
  : fn(data);
Promise.resolve(_call).then(result => {{
  process.stdout.write = _origWrite;
  if (_buf.length) process.stderr.write(_buf.join(""));
  const MARKER = "\\n__OVERMIND_RESULT__\\n";
  _origWrite(MARKER);
  _origWrite(typeof result === "string" ? result : JSON.stringify(result));
  _origWrite(MARKER);
}}).catch(err => {{
  process.stdout.write = _origWrite;
  process.stderr.write(err.stack || String(err));
  process.exit(1);
}});
"""

_TS_WRAPPER = """\
import {{ readFileSync }} from "fs";
import * as mod from {entry_path};
const fn = typeof (mod as any).default === "function"
  ? (mod as any).default
  : (mod as any)[{fn_name}] || mod;
const data = JSON.parse(readFileSync("/dev/stdin", "utf8"));
const _origWrite = process.stdout.write.bind(process.stdout);
const _buf: string[] = [];
process.stdout.write = ((chunk: any, enc?: any, cb?: any) => {{ _buf.push(chunk); if (cb) cb(); return true; }}) as any;
const _call = (fn.length > 1 && typeof data === "object" && data !== null && !Array.isArray(data))
  ? fn(...Object.values(data))
  : fn(data);
Promise.resolve(_call).then((result: any) => {{
  process.stdout.write = _origWrite;
  if (_buf.length) process.stderr.write(_buf.join(""));
  const MARKER = "\\n__OVERMIND_RESULT__\\n";
  _origWrite(MARKER);
  _origWrite(typeof result === "string" ? result : JSON.stringify(result));
  _origWrite(MARKER);
}}).catch((err: any) => {{
  process.stdout.write = _origWrite;
  process.stderr.write(err.stack || String(err));
  process.exit(1);
}});
"""


def _generate_wrapper(
    language: Language,
    entry_path: str,
    fn_name: str,
    agent_dir: Path,
    shadow_config: ShadowConfig | None = None,
) -> Path:
    """Create a thin wrapper that reads stdin JSON, calls the agent, writes stdout JSON.

    When *shadow_config* is given and enabled, the generated wrapper is
    prepended with the Overmind shadow bootstrap (see
    :mod:`overmind.optimize.shadow_runtime`) which intercepts LLM, HTTP,
    and browser calls for cassette-based record/replay and simulation.
    Otherwise the wrapper gets the minimal sys.argv guard that is already
    enough to fix ~90% of "module-level side effects crash on import"
    failures.

    Returns the path to the generated wrapper file.
    """
    from overmind.optimize.shadow_runtime import bootstrap_source

    wrapper_dir = agent_dir / ".overmind_runners"
    wrapper_dir.mkdir(parents=True, exist_ok=True)

    if language == Language.PYTHON:
        bootstrap = bootstrap_source(shadow_config)
        code = _PYTHON_WRAPPER.format(
            shadow_bootstrap=bootstrap,
            entry_path=entry_path,
            fn_name=fn_name,
        )
        if shadow_config and shadow_config.enabled:
            wrapper_name = "_run_agent_shadow.py"
        elif shadow_config and shadow_config.cassette_path:
            wrapper_name = "_run_agent_record.py"
        else:
            wrapper_name = "_run_agent.py"
        wrapper_path = wrapper_dir / wrapper_name
        wrapper_path.write_text(code)
    elif language == Language.JAVASCRIPT:
        entry_for_require = os.path.relpath(entry_path, str(wrapper_dir))
        if not entry_for_require.startswith("."):
            entry_for_require = "./" + entry_for_require
        code = _JS_WRAPPER.format(
            entry_path=json.dumps(entry_for_require),
            fn_name=json.dumps(fn_name),
        )
        wrapper_path = wrapper_dir / "_run_agent.js"
        wrapper_path.write_text(code)
    else:
        entry_for_import = os.path.relpath(entry_path, str(wrapper_dir))
        if not entry_for_import.startswith("."):
            entry_for_import = "./" + entry_for_import
        code = _TS_WRAPPER.format(
            entry_path=json.dumps(entry_for_import),
            fn_name=json.dumps(fn_name),
        )
        wrapper_path = wrapper_dir / "_run_agent.ts"
        wrapper_path.write_text(code)

    return wrapper_path


# ---------------------------------------------------------------------------
# AgentRunner
# ---------------------------------------------------------------------------


class AgentRunner:
    """Language-agnostic, process-isolated agent executor.

    Typical usage::

        runner = AgentRunner(
            agent_dir="/path/to/agent",
            entry_file="main.py",
            entrypoint_fn="run",
        )
        runner.ensure_environment()   # one-time: install deps
        output = runner.run({"query": "hello"})
        if output.success:
            print(output.data)
    """

    def __init__(
        self,
        agent_dir: str | Path,
        entry_file: str,
        entrypoint_fn: str,
        config: RunnerConfig | None = None,
        env_dir: str | Path | None = None,
    ) -> None:
        self.agent_dir = Path(agent_dir).resolve()
        self.entry_file = entry_file
        self.entrypoint_fn = entrypoint_fn
        self.config = config or RunnerConfig()
        self.language = Language.from_path(entry_file)
        self.env_dir = Path(env_dir).resolve() if env_dir else self.agent_dir
        self._env_provisioned = False
        self._python_path: Path | None = None
        self._wrapper_path: Path | None = None

    # ------------------------------------------------------------------
    # Environment provisioning
    # ------------------------------------------------------------------

    def ensure_environment(self) -> None:
        """Detect deps and install into an isolated environment.

        Safe to call multiple times — uses hash-based caching to skip
        redundant installs.  Uses ``env_dir`` (the original agent
        directory) for dependency manifest lookup and venv provisioning,
        so optimized code running from a different folder still finds
        the correct environment.

        Raises :class:`MissingDependenciesError` if the agent imports
        external packages but has no dependency manifest.  This guides
        the user to run ``overmind setup`` (interactive) or create a
        manifest manually.
        """
        if self._env_provisioned:
            return

        logger.info(
            "AgentRunner.ensure_environment language=%s env_dir=%s entry=%s",
            self.language.value,
            self.env_dir,
            self.entry_file,
        )

        if not has_dep_manifest(self.env_dir, self.language):
            ext_imports = detect_external_imports(self.env_dir, self.entry_file, self.language)
            if ext_imports:
                logger.error(
                    "Missing dependency manifest for %s; detected external imports: %s",
                    self.env_dir,
                    ext_imports,
                )
                raise MissingDependenciesError(self.env_dir, self.language, ext_imports)

        if self.language == Language.PYTHON:
            self._python_path = _provision_python(self.env_dir)
            logger.info(
                "AgentRunner.ensure_environment python interpreter resolved: %s",
                self._python_path,
            )
        else:
            _provision_js(self.env_dir)

        self._env_provisioned = True

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        input_data: Any,
        timeout: int | None = None,
        trace_file: str | Path | None = None,
        shadow_config: ShadowConfig | None = None,
    ) -> RunOutput:
        """Execute the agent in a subprocess. Returns structured output.

        If *trace_file* is given, ``OVERMIND_TRACE_FILE`` is set in the
        child environment so the overmind writes spans there.

        When *shadow_config* is provided and enabled, the subprocess is
        launched with the Overmind shadow bootstrap active so external
        calls (LLM, HTTP, browser) are intercepted and either replayed
        from a cassette or simulated.  LLM calls with novel prompts still
        hit the real model so optimisation signal remains meaningful.
        """
        effective_timeout = timeout or self.config.timeout
        entry_abs = str(self.agent_dir / self.entry_file)

        wrapper = self._get_wrapper(entry_abs, shadow_config=shadow_config)
        cmd = self._build_command(wrapper)
        env = self._build_env(trace_file=trace_file, shadow_config=shadow_config)

        input_json = json.dumps(input_data, default=str)

        logger.debug(
            "AgentRunner.run spawning subprocess cmd=%s cwd=%s timeout=%ds trace_file=%s input_bytes=%d",
            cmd,
            self.agent_dir,
            effective_timeout,
            trace_file,
            len(input_json),
        )

        try:
            proc = subprocess.run(
                cmd,
                input=input_json,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                cwd=str(self.agent_dir),
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            logger.warning(
                "AgentRunner.run subprocess timeout after %ds cmd=%s",
                effective_timeout,
                cmd,
            )
            partial_stderr = ""
            if exc.stderr:
                partial_stderr = exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode(errors="replace")
            partial_stdout = ""
            if exc.stdout:
                partial_stdout = exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode(errors="replace")
            return RunOutput(
                success=False,
                error=f"Agent timed out after {effective_timeout}s",
                stdout=partial_stdout[-4000:],
                stderr=partial_stderr[-4000:],
                returncode=-1,
            )
        except FileNotFoundError as exc:
            logger.error("AgentRunner.run interpreter not found cmd=%s err=%s", cmd, exc)
            return RunOutput(
                success=False,
                error=f"Interpreter not found: {exc}",
                returncode=-1,
            )

        logger.debug(
            "AgentRunner.run subprocess exited rc=%d stdout_bytes=%d stderr_bytes=%d",
            proc.returncode,
            len(proc.stdout or ""),
            len(proc.stderr or ""),
        )

        if proc.returncode != 0:
            logger.warning(
                "AgentRunner.run subprocess non-zero exit rc=%d stderr_tail=%s",
                proc.returncode,
                (proc.stderr or "")[-500:],
            )
            return RunOutput(
                success=False,
                error=proc.stderr[-4000:] if proc.stderr else f"Exit code {proc.returncode}",
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
            )

        result_payload = _extract_marked_result(proc.stdout)
        if result_payload is None:
            result_payload = proc.stdout.strip()

        if not result_payload:
            return RunOutput(
                success=False,
                error="Agent produced no output on stdout",
                stdout=proc.stdout[-2000:],
                stderr=proc.stderr[-2000:],
                returncode=proc.returncode,
            )

        parsed = _try_parse_json(result_payload)
        data = parsed if parsed is not None else result_payload

        return RunOutput(
            success=True,
            data=data,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=0,
        )

    # ------------------------------------------------------------------
    # Validation (callable check without full execution)
    # ------------------------------------------------------------------

    def validate_entrypoint(self, code: str | None = None) -> bool:
        """Check that the entry file defines the expected function.

        For Python, uses AST.  For JS/TS, uses a lightweight regex
        check.  When *code* is supplied it is checked instead of
        reading from disk.
        """
        if code is None:
            entry_abs = self.agent_dir / self.entry_file
            if not entry_abs.is_file():
                return False
            code = entry_abs.read_text(encoding="utf-8")

        if self.language == Language.PYTHON:
            return _validate_python_entrypoint(code, self.entrypoint_fn)
        return _validate_js_entrypoint(code, self.entrypoint_fn)

    def validate_syntax(self, code: str | None = None) -> bool:
        """Check whether the entry file (or *code*) is syntactically valid."""
        if code is None:
            entry_abs = self.agent_dir / self.entry_file
            if not entry_abs.is_file():
                return False
            code = entry_abs.read_text(encoding="utf-8")

        if self.language == Language.PYTHON:
            return _validate_python_syntax(code)
        return _validate_js_syntax(code, self.agent_dir)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_wrapper(
        self,
        entry_abs: str,
        shadow_config: ShadowConfig | None = None,
    ) -> Path:
        # Three distinct wrapper flavours — cached separately so one doesn't
        # overwrite another when the same runner is invoked with different
        # shadow configs (e.g. subprocess=record-only, shadow=full):
        #   - "shadow": full intercept bootstrap (HTTP, browser, LLM)
        #   - "record": LLM-only intercept for cassette capture
        #   - "real":   minimal argv guard, no intercepts
        if shadow_config and shadow_config.enabled:
            cache_key = "_shadow"
        elif shadow_config and shadow_config.cassette_path:
            cache_key = "_record"
        else:
            cache_key = "_real"
        cached: Path | None = getattr(self, f"_wrapper_path{cache_key}", None)
        if cached is not None and cached.exists():
            return cached
        wrapper = _generate_wrapper(
            self.language,
            entry_abs,
            self.entrypoint_fn,
            self.agent_dir,
            shadow_config=shadow_config,
        )
        setattr(self, f"_wrapper_path{cache_key}", wrapper)
        self._wrapper_path = wrapper
        return wrapper

    def _build_command(self, wrapper: Path) -> list[str]:
        if self.language == Language.PYTHON:
            python = str(self._python_path or sys.executable)
            return [python, str(wrapper)]
        elif self.language == Language.JAVASCRIPT:
            return ["node", str(wrapper)]
        else:
            return ["npx", "tsx", str(wrapper)]

    def _build_env(
        self,
        trace_file: str | Path | None = None,
        shadow_config: ShadowConfig | None = None,
    ) -> dict[str, str]:
        env = dict(os.environ)
        env["TERM"] = "dumb"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        if self.language != Language.PYTHON:
            env["NODE_NO_WARNINGS"] = "1"
        if trace_file is not None:
            env["OVERMIND_TRACE_FILE"] = str(trace_file)
            env.pop("OVERMIND_API_KEY", None)
        if shadow_config is not None:
            env.update(shadow_config.env())
        env.update(self.config.extra_env)

        # Propagate the active OTel span to the child process using the W3C
        # Trace Context header so every agent evaluation run is a child span
        # of the optimizer's current span (distributed tracing across
        # subprocess boundaries).  Always overwrite any inherited TRACEPARENT
        # from the parent process env — it would point to a stale span.
        traceparent = _current_traceparent()
        if traceparent:
            env["TRACEPARENT"] = traceparent
        else:
            env.pop("TRACEPARENT", None)

        return env

    def cleanup(self) -> None:
        """Remove generated wrapper scripts."""
        runner_dir = self.agent_dir / ".overmind_runners"
        if runner_dir.is_dir():
            shutil.rmtree(runner_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# JSON extraction from stdout
# ---------------------------------------------------------------------------

_RESULT_MARKER = "\n__OVERMIND_RESULT__\n"


def _extract_marked_result(stdout: str) -> str | None:
    """Extract the payload between ``__OVERMIND_RESULT__`` markers.

    Returns the raw string between markers, or *None* if markers are absent.
    """
    idx = stdout.find(_RESULT_MARKER)
    if idx == -1:
        return None
    start = idx + len(_RESULT_MARKER)
    end = stdout.find(_RESULT_MARKER, start)
    if end == -1:
        return stdout[start:].strip() or None
    payload = stdout[start:end]
    return payload if payload else None


def _try_parse_json(text: str) -> Any:
    """Attempt to parse *text* as JSON. Returns the parsed value or None."""
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    for start_char, end_char in [("{", "}"), ("[", "]")]:
        last_end = text.rfind(end_char)
        if last_end == -1:
            continue
        first_start = text.rfind(start_char, 0, last_end + 1)
        while first_start >= 0:
            candidate = text[first_start : last_end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                first_start = text.rfind(start_char, 0, first_start)

    return None


# ---------------------------------------------------------------------------
# Syntax & entrypoint validation helpers
# ---------------------------------------------------------------------------


def _validate_python_syntax(code: str) -> bool:
    try:
        compile(code, "<agent>", "exec")
        return True
    except SyntaxError:
        return False


def _validate_python_entrypoint(code: str, fn_name: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
            return True
    return False


def _validate_js_syntax(code: str, agent_dir: Path) -> bool:
    """Use ``node --check`` for JS syntax validation.

    Returns True if node is unavailable (optimistic fallback).
    """
    if not shutil.which("node"):
        return True
    with tempfile.NamedTemporaryFile(suffix=".js", dir=str(agent_dir), mode="w", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        result = subprocess.run(
            ["node", "--check", tmp],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return True
    finally:
        Path(tmp).unlink(missing_ok=True)


def _validate_js_entrypoint(code: str, fn_name: str) -> bool:
    """Lightweight regex check for JS/TS function or export."""
    patterns = [
        rf"\bfunction\s+{re.escape(fn_name)}\s*\(",
        rf"\bconst\s+{re.escape(fn_name)}\s*=",
        rf"\blet\s+{re.escape(fn_name)}\s*=",
        rf"\bvar\s+{re.escape(fn_name)}\s*=",
        rf"exports\.{re.escape(fn_name)}\s*=",
        rf"export\s+(default\s+)?function\s+{re.escape(fn_name)}\b",
        rf"export\s+\{{[^}}]*\b{re.escape(fn_name)}\b",
        rf"export\s+(const|let|var)\s+{re.escape(fn_name)}\b",
        r"module\.exports\s*=",
    ]
    for pat in patterns:
        if re.search(pat, code):
            return True
    return False


# ---------------------------------------------------------------------------
# Import extraction (multi-language)
# ---------------------------------------------------------------------------


def extract_imports(code: str, language: Language) -> list[str]:
    """Extract top-level import module names from *code*."""
    if language == Language.PYTHON:
        return _extract_python_imports(code)
    return _extract_js_imports(code)


def _extract_python_imports(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    modules: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module.split(".")[0])
    return list(dict.fromkeys(modules))


def _extract_js_imports(code: str) -> list[str]:
    modules: list[str] = []
    for m in re.finditer(r"""(?:import\s+.*?\s+from\s+|require\s*\(\s*)['"]([^'"]+)['"]""", code):
        mod = m.group(1)
        if not mod.startswith("."):
            modules.append(mod.split("/")[0])
    return list(dict.fromkeys(modules))
