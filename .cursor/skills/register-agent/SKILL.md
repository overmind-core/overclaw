______________________________________________________________________

## name: register-agent description: Register an agent with the Overmind registry without interactive CLI prompts. Use when the user wants to register an agent, run `overmind agent register`, set up agent credentials, configure an LLM provider for an agent, or add a new agent to an Overmind project. Discovers the entrypoint function, derives the module path, runs registration, then asks the user to fill in their credentials. disable-model-invocation: true

# Register an Overmind Agent

Registers an agent in `.overmind/agents.toml` without interactive CLI prompts.
Runs registration immediately, then asks the user to fill in their credentials at the end.

## Workflow

### Step 1 — Collect inputs

Ask (use `AskQuestion` for multiple-choice, plain conversation for free-form):

1. **Agent file path** — relative to the project root (e.g. `examples/hotel/agent.py`)
1. **Agent name (slug)** — default to the parent folder name; confirm before proceeding

### Step 2 — Discover the entrypoint function

Read the agent file. Find the entrypoint in priority order:

1. `def run(`
1. `def run_agent(`
1. `def agent(`
1. Any top-level `def` that returns `dict` or `str`

If multiple candidates exist, ask the user to pick one.

**Derive the module path** from the file path — strip the extension, replace `/` with `.`:

| File path                          | Module path                     |
| ---------------------------------- | ------------------------------- |
| `examples/hotel/agent.py`          | `examples.hotel.agent`          |
| `new_examples/langextract/test.py` | `new_examples.langextract.test` |
| `agents/support/bot.py`            | `agents.support.bot`            |

> Exception: paths containing a directory starting with `.` (e.g. `.overmind/`) must use the slash form — Python can't import dotted names starting with a dot.

Construct the entrypoint string: `<module_path>:<function_name>`
Example: `examples.hotel.agent:run`

### Step 3 — Scan for env vars

Scan the file for these patterns:

- `os.environ.get("KEY")` / `os.environ.get("KEY", "default")`
- `os.getenv("KEY")` / `os.getenv("KEY", "default")`
- `os.environ["KEY"]`

Exclude system vars: `PATH HOME USER LOGNAME SHELL TERM LANG PWD TMPDIR TMP TEMP`.

Note any literal default values from the code — use them as placeholder text in the `.env` file.

### Step 4 — Ask for provider (for .env scaffolding only)

Use `AskQuestion`:

> "Which LLM provider does this agent use?"
> Options: OpenAI | Anthropic | Other (OpenAI-compatible) | No LLM / configure manually

Determine the required key(s):

- **OpenAI** → `OPENAI_API_KEY`
- **Anthropic** → `ANTHROPIC_API_KEY`
- **Other** → `OPENAI_BASE_URL` + `OPENAI_API_KEY`
- **No LLM / manually** → no provider keys

### Step 5 — Run registration

Write `_register_runner.py` in the project root:

```python
import sys
from rich.console import Console

import overmind
from overmind.core.paths import load_overmind_dotenv

load_overmind_dotenv()
overmind.init()

from overmind.commands.agent_env import instrument_agent_files
from overmind.core.registry import resolve_entrypoint, save_agent, load_registry

console = Console()

AGENT_NAME = "<name>"
ENTRYPOINT = "<module:function>"
AGENT_FILE = "<relative/path/to/agent.py>"


# Guard: already registered?
registry = load_registry()
if AGENT_NAME in registry:
    current = registry[AGENT_NAME]["entrypoint"].strip()
    if current == ENTRYPOINT.strip():
        console.print(
            f"  [dim]'{AGENT_NAME}' is already registered with this entrypoint — nothing to do.[/dim]"
        )
        sys.exit(0)
    console.print(
        f"  [bold red]Error:[/bold red] '{AGENT_NAME}' is already registered with a "
        f"different entrypoint: [dim]{current}[/dim]\n\n"
        f"  To change it:  [bold]overmind agent update {AGENT_NAME} {ENTRYPOINT}[/bold]"
    )
    sys.exit(1)

# 1. Copy source tree into .overmind/agents/<name>/instrumented/
instrument_agent_files(AGENT_FILE, AGENT_NAME, console)

# 2. Validate the entrypoint function exists and has the right signature
try:
    file_path, fn = resolve_entrypoint(ENTRYPOINT)
    console.print(
        f"  [bold green]✓[/bold green]  Entrypoint validated: [bold]{fn}[/bold] in {file_path}"
    )
except Exception as exc:
    console.print(f"\n  [bold red]✗  Entrypoint error:[/bold red] {exc}")
    sys.exit(1)

# 3. Write to registry
save_agent(AGENT_NAME, ENTRYPOINT)
console.print(
    f"\n  [bold green]✓[/bold green]  Agent '[bold]{AGENT_NAME}[/bold]' registered.\n"
    f"  [dim]Entrypoint:[/dim] {ENTRYPOINT}\n"
    f"  [dim]File:[/dim]      {AGENT_FILE}\n"
)
```

Run it from the **project root** (the directory containing `.overmind/`). Never `cd` to a parent directory:

```bash
python _register_runner.py
```

After success, delete `_register_runner.py`.

### Step 6 — Create the .env file with placeholders

After registration succeeds, create `.overmind/agents/<name>/.env` with placeholders for the credentials identified in Step 4:

```
# Overmind agent env — <name>

OPENAI_API_KEY=<your-openai-api-key-here>
```

For "Other": include both `OPENAI_BASE_URL=<your-base-url-here>` and `OPENAI_API_KEY=<your-key-here>`.

For each additional env var discovered in Step 3 that isn't already covered, add a placeholder line:

```
SOME_OTHER_KEY=<value-here>
```

If "No LLM / manually" was chosen and no env vars were discovered, skip creating the file.

### Step 7 — Summarize

Tell the user:

- Agent name and entrypoint that was registered
- If a `.env` was created: tell them to open `.overmind/agents/<name>/.env` and fill in the placeholder value(s) before running the agent.
- Next step: run `/generate-dataset` with agent name `<name>`.

Do **not** mention runner scripts, file cleanup, registry internals, or any implementation details.

## Fallback: if overmind internals can't be imported

If the runner fails with `ImportError` on `overmind.commands` or `overmind.core`, activate the project venv first:

```bash
source .venv/bin/activate && python _register_runner.py
```

If the project uses `uv`, run from the **project root** (do NOT `cd` to a parent directory or pass `--project` to `uv run`):

```bash
uv run python _register_runner.py
```

## Common issues

| Problem                                                  | Fix                                                                                                  |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `Module 'x.y.z' resolves to '...', which does not exist` | Module path is wrong — check slashes vs dots; try the `/`-based form if the path has unusual dirs    |
| `EntrypointNotFoundError`                                | Function name not found in the file — re-read and confirm the spelling                               |
| `Agent already registered` (different entrypoint)        | Use `overmind agent update <name> <entrypoint>`                                                      |
| `EntrypointSignatureError`                               | The function's signature is missing required `dict`/`str` params — offer to generate an auto-wrapper |
| `ImportError: No module named overmind`                  | Activate the project venv                                                                            |
| User's agent uses no LLM directly                        | Choose "No LLM / manually" in Step 4 — skip the `.env` unless env vars were discovered in Step 3     |
