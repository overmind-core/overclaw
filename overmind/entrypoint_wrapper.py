"""Auto-generate an Overmind-compatible entrypoint wrapper via the coding agent.

When a user registers an agent that doesn't expose a simple
``def run(input: dict) -> dict`` function (e.g. a Google ADK agent, a
LangChain graph, or a CrewAI crew), this module uses the coding agent to
read the agent's source code, detect the framework and execution pattern,
and generate a thin wrapper file that conforms to Overmind's I/O contract.

The generated file lives inside the instrumented copy at
``.overmind/agents/<name>/instrumented/`` — alongside the copied agent
source so that all imports resolve correctly.  The user's original code
is never modified.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

WRAPPER_FILENAME = "_overmind_entrypoint.py"

_DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"

_WRAPPER_PROMPT = """\
You are generating an Overmind entrypoint wrapper for an AI agent.

Overmind optimizes agents by running them against evaluation datasets.
It needs a single Python function with this exact contract:

    def run(input_data: dict) -> dict:
        # Execute the agent with the given input
        # Return a dict with the agent's output

The function:
- Receives a dict (parsed from JSON) as input
- Must return a dict (serialized to JSON) as output
- May use asyncio.run() internally if the agent is async
- Should handle all framework setup (sessions, runners, etc.) internally

AGENT SOURCE DIRECTORY: {agent_dir}

Here is the agent's code and surrounding files:

{code_context}

═══════════════════════════════════════════════════════════════════════
CRITICAL RULE — THE WRAPPER MUST BE TRIVIALLY SIMPLE
═══════════════════════════════════════════════════════════════════════

The wrapper you generate MUST be a thin bridge that:
  1. Imports the existing agent object/function from the agent's own modules.
  2. Calls it using the framework's standard runner/executor.
  3. Returns the result as a dict.

That's it. The wrapper should be ~20-40 lines at most.

GOOD wrapper (just imports + calls):
    from my_agent.agent import root_agent
    from framework import Runner
    def run(input_data: dict) -> dict:
        result = Runner(agent=root_agent).run(input_data["query"])
        return {{"response": result}}

BAD wrapper (re-implements agent logic):
    def run(input_data: dict) -> dict:
        # Setting up tools manually
        # Re-implementing tool call logic
        # Parsing tool outputs
        # Building prompts from scratch
        # Any agent-specific business logic

If generating a correct wrapper would require:
  - Re-implementing tool calls, agent logic, or prompt construction
  - Copying substantial code from the agent into the wrapper
  - Any domain-specific logic beyond framework boilerplate

Then DO NOT generate the wrapper. Instead, create the file with ONLY
this content:

    # OVERMIND_WRAPPER_REFUSED
    #
    # This agent's code cannot be wrapped with a simple bridge.
    # The entrypoint function needs agent-specific logic that belongs
    # in the agent code itself, not in an auto-generated wrapper.
    #
    # To make your agent work with Overmind, add a function to your
    # agent code with this signature:
    #
    #     def run(input_data: dict) -> dict:
    #         # call your agent here
    #         return {{"response": ...}}
    #
    # Then register it:
    #     overmind agent register <name> your_module:run

═══════════════════════════════════════════════════════════════════════

YOUR TASK:
1. Read the agent code and identify the framework (Google ADK, LangChain,
   CrewAI, AutoGen, LangGraph, Claude Agent SDK, or custom).
2. Understand how the agent is executed (look at test files, examples,
   or main blocks for patterns).
3. Decide: can the agent be invoked with a trivial import-and-call wrapper?
   - YES → generate the wrapper at {wrapper_path}
   - NO  → write the refusal comment file at {wrapper_path}
4. If generating:
   - Import the agent from its module using absolute imports based on
     the agent source directory structure. The agent source directory
     will be on sys.path at runtime.
   - Define exactly: def run(input_data: dict) -> dict
   - The function should extract a query/prompt from input_data
     (use input_data.get("query", "") as the primary input key)
   - Execute the agent using the framework's standard runner
   - Return {{"response": <agent_output_text>}} at minimum
   - Handle async execution with asyncio.run() if needed
   - No logging, no retries, no error wrapping.

IMPORTANT:
- The file MUST be at exactly: {wrapper_path}
- The function MUST be named: run
- Do NOT modify any existing files
- Do NOT install packages or run the agent
- Use absolute imports that work when the agent source directory is on
  sys.path (e.g. ``from llm_red_team_agent.agent import root_agent``)
- Do NOT manipulate ``sys.path`` yourself.  Overmind prepends a bootstrap
  header to this wrapper that places the agent source directory
  ({agent_dir}) at ``sys.path[0]`` at runtime, so bare imports of modules
  that live directly inside that directory (e.g. ``from news_monitor
  import extract_latest_article``) will resolve.
"""

WRAPPER_REFUSED_MARKER = "# OVERMIND_WRAPPER_REFUSED"

_SYS_PATH_BOOTSTRAP_BEGIN = "# --- OVERMIND_SYS_PATH_BOOTSTRAP (auto-generated, do not edit) ---"
_SYS_PATH_BOOTSTRAP_END = "# --- END OVERMIND_SYS_PATH_BOOTSTRAP ---"


def _instrumented_agent_dir_relpath(agent_dir: Path, agent_name: str) -> Path:
    """Where the agent source sits inside the instrumented copy, relative to the wrapper.

    Mirrors the layout produced by
    :func:`overmind.commands.agent_env.instrument_agent_files`:
    the copy boundary is ``project_root_from_agent_file`` (or, fallback,
    the agent file's parent).  The wrapper lives at the top of the
    instrumented dir, so the relative path is simply ``agent_dir``
    relative to that copy root.
    """
    from overmind.core.registry import project_root_from_agent_file

    original_root = project_root_from_agent_file(agent_dir) or agent_dir
    try:
        return agent_dir.relative_to(original_root)
    except ValueError:
        return Path(".")


def _render_sys_path_bootstrap(agent_relpath: Path) -> str:
    """Return a small deterministic header that puts the agent dir on sys.path.

    The header is prepended to every generated wrapper so that bare imports
    like ``from news_monitor import ...`` resolve at runtime, matching the
    promise made to the coding agent in ``_WRAPPER_PROMPT``.
    """
    parts = agent_relpath.parts if agent_relpath != Path(".") else ()
    path_parts = [repr(p) for p in parts] or ["'.'"]
    join_args = ", ".join([
        "os.path.dirname(os.path.abspath(__file__))",
        *path_parts,
    ])
    return (
        f"{_SYS_PATH_BOOTSTRAP_BEGIN}\n"
        "import os\n"
        "import sys\n"
        f"_OVERMIND_AGENT_DIR = os.path.normpath(os.path.join({join_args}))\n"
        "if _OVERMIND_AGENT_DIR not in sys.path:\n"
        "    sys.path.insert(0, _OVERMIND_AGENT_DIR)\n"
        f"{_SYS_PATH_BOOTSTRAP_END}\n\n"
    )


def _prepend_sys_path_bootstrap(wp: Path, agent_relpath: Path) -> None:
    """Inject the sys.path bootstrap at the top of the generated wrapper.

    Python requires ``from __future__`` imports to appear before any other
    code (with the narrow exception of the module docstring, comments, and
    blank lines).  The coding agent that generates wrappers often emits
    ``from __future__ import annotations`` right under the module header,
    so we must hoist it above our bootstrap — otherwise we produce code
    that fails with ``SyntaxError: from __future__ imports must occur at
    the beginning of the file``.

    Idempotent: if the marker is already present, we rewrite the block so
    re-generation stays in sync with the current instrumented layout.
    """
    content = wp.read_text(encoding="utf-8")

    if _SYS_PATH_BOOTSTRAP_BEGIN in content:
        before, _, rest = content.partition(_SYS_PATH_BOOTSTRAP_BEGIN)
        _, _, after = rest.partition(_SYS_PATH_BOOTSTRAP_END + "\n")
        # The header itself ends in ``\n\n`` — strip the leading blank so we
        # don't accumulate blank lines across repeated injections.
        after = after.removeprefix("\n")
        content = before + after

    header = _render_sys_path_bootstrap(agent_relpath)

    shebang = ""
    if content.startswith("#!"):
        shebang, _, content = content.partition("\n")
        shebang += "\n"

    # Pull out module docstring + future imports that MUST stay at the top,
    # leaving everything else in ``body``.  We keep a very simple tokenizer
    # here — the wrapper files we generate are short and syntactically
    # plain, so a line-oriented scan is good enough.
    preamble_lines: list[str] = []
    body_lines: list[str] = []
    in_docstring = False
    docstring_quote = ""
    seen_code = False

    for line in content.splitlines(keepends=True):
        stripped = line.strip()
        if not seen_code:
            if stripped == "" or stripped.startswith("#"):
                preamble_lines.append(line)
                continue
            if not in_docstring and (stripped.startswith(('"""', "'''"))):
                quote = stripped[:3]
                preamble_lines.append(line)
                # Single-line docstring (opened and closed on same line)
                rest = stripped[3:]
                if rest.count(quote) >= 1:
                    continue
                in_docstring = True
                docstring_quote = quote
                continue
            if in_docstring:
                preamble_lines.append(line)
                if docstring_quote in stripped:
                    in_docstring = False
                continue
            if stripped.startswith("from __future__"):
                preamble_lines.append(line)
                continue
            seen_code = True
        body_lines.append(line)

    preamble = "".join(preamble_lines)
    body = "".join(body_lines)

    wp.write_text(shebang + preamble + header + body, encoding="utf-8")


def _get_model() -> str | None:
    """Return the model to use for wrapper generation, or None if unavailable."""
    for env_var in ("ANALYZER_MODEL", "ENV_SETUP_MODEL"):
        val = os.environ.get(env_var, "").strip()
        if val:
            return val

    try:
        import litellm  # noqa: F401

        return _DEFAULT_MODEL
    except ImportError:
        return None


def _gather_code_context(agent_dir: Path, max_chars: int = 30_000) -> str:
    """Build a code-context string from the agent directory for the prompt."""
    sections: list[str] = []
    total = 0

    priority_patterns = [
        "agent.py",
        "agents.py",
        "main.py",
        "app.py",
        "run.py",
        "crew.py",
        "graph.py",
        "chain.py",
        "workflow.py",
    ]
    secondary_patterns = [
        "*.py",
    ]
    test_patterns = [
        "test_*.py",
        "*_test.py",
        "tests/*.py",
        "tests/**/*.py",
        "examples/*.py",
        "example*.py",
    ]

    seen: set[Path] = set()

    def _add_file(p: Path) -> bool:
        nonlocal total
        if p in seen or not p.is_file():
            return False
        seen.add(p)
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        if total + len(content) > max_chars:
            return False
        rel = p.relative_to(agent_dir) if p.is_relative_to(agent_dir) else p
        sections.append(f"--- {rel} ---\n{content}")
        total += len(content)
        return True

    for pattern in priority_patterns:
        for p in sorted(agent_dir.rglob(pattern)):
            if p.name == WRAPPER_FILENAME:
                continue
            _add_file(p)

    for pattern in test_patterns:
        for p in sorted(agent_dir.rglob(pattern)):
            if p.name == WRAPPER_FILENAME:
                continue
            _add_file(p)

    for pattern in secondary_patterns:
        for p in sorted(agent_dir.rglob(pattern)):
            if p.name.startswith(".") or p.name == WRAPPER_FILENAME:
                continue
            if "__pycache__" in str(p) or ".venv" in str(p):
                continue
            _add_file(p)

    config_files = [
        "pyproject.toml",
        "requirements.txt",
        "setup.py",
        "package.json",
        "README.md",
    ]
    for name in config_files:
        p = agent_dir / name
        if p.is_file():
            _add_file(p)

    return "\n\n".join(sections) if sections else "(no Python files found)"


def wrapper_dir(agent_name: str) -> Path:
    """Return the directory where the wrapper is generated.

    This is ``.overmind/agents/<name>/instrumented/`` — the same directory
    that holds the copied agent source, so imports inside the wrapper
    resolve exactly as they would in the original project.
    """
    from overmind.core.paths import agent_instrumented_dir

    return agent_instrumented_dir(agent_name)


def wrapper_path(agent_name: str) -> Path:
    """Full path to the generated wrapper file."""
    return wrapper_dir(agent_name) / WRAPPER_FILENAME


def wrapper_entrypoint(agent_name: str, fn_name: str = "run") -> str:
    """Return the entrypoint string for the wrapper.

    Uses a slash-based relative path (not dotted module notation) because
    the ``.overmind`` directory name starts with a dot which cannot
    survive a dotted-module round-trip.

    Example return value::

        .overmind/agents/gsec/instrumented/_overmind_entrypoint:run
    """
    from overmind.core.registry import project_root

    wp = wrapper_path(agent_name)
    rel = wp.relative_to(project_root())
    module_ref = str(rel.with_suffix(""))
    return f"{module_ref}:{fn_name}"


def generate_entrypoint_wrapper(
    agent_dir: str | Path,
    agent_name: str,
) -> Path | str | None:
    """Use the coding agent to generate an entrypoint wrapper.

    The wrapper is written to
    ``.overmind/agents/<name>/instrumented/_overmind_entrypoint.py`` —
    alongside the instrumented copy of the agent source so that all
    imports resolve correctly.  The user's original code is never modified.

    Returns the path to the generated wrapper file, ``"refused"`` if the
    coding agent determined the wrapper would be too complex, or ``None``
    if generation failed or the coding agent is unavailable.
    """
    model = _get_model()
    if not model:
        logger.debug("No model available for wrapper generation")
        return None

    try:
        from overmind.coding_agent.agent import run as run_coding_agent
    except ImportError:
        logger.debug("Coding agent not importable")
        return None

    agent_dir = Path(agent_dir).resolve()
    dest_dir = wrapper_dir(agent_name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    wp = dest_dir / WRAPPER_FILENAME

    code_context = _gather_code_context(agent_dir)

    instruction = _WRAPPER_PROMPT.format(
        agent_dir=str(agent_dir),
        wrapper_path=str(wp),
        code_context=code_context,
    )

    logger.info(f"Using coding agent ({model}) to generate entrypoint wrapper …")

    try:
        run_coding_agent(
            instruction=instruction,
            model=model,
            cwd=str(dest_dir),
            max_steps=15,
        )
    except Exception as exc:
        logger.warning(f"Wrapper generation failed: {exc}")
        return None

    if not wp.is_file():
        logger.warning(f"Coding agent ran but wrapper file not found at {wp}")
        return None

    content = wp.read_text(encoding="utf-8")
    if content.strip().startswith(WRAPPER_REFUSED_MARKER):
        logger.info("Coding agent declined to generate wrapper — agent needs manual restructuring")
        wp.unlink(missing_ok=True)
        return "refused"

    # Guarantee that bare imports from the agent source directory resolve at
    # runtime.  The coding agent is told this happens automatically (see
    # ``_WRAPPER_PROMPT``); we make that promise true by prepending a
    # deterministic ``sys.path`` bootstrap computed from the layout produced
    # by ``instrument_agent_files``.
    agent_relpath = _instrumented_agent_dir_relpath(agent_dir, agent_name)
    _prepend_sys_path_bootstrap(wp, agent_relpath)

    logger.info(f"Entrypoint wrapper generated at {wp}")
    return wp
