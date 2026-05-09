______________________________________________________________________

## name: overmind-register-agent description: Register an agent with the Overmind registry without interactive CLI prompts. Use when the user wants to register an agent, run `overmind agent register`, set up agent credentials, configure an LLM provider for an agent, or add a new agent to an Overmind project. Discovers the entrypoint function, derives the module path, runs registration, then asks the user to fill in their credentials. disable-model-invocation: true

# Register an Overmind Agent

Registers an agent in `.overmind/agents.toml` without interactive CLI prompts.
Runs registration immediately, then asks the user to fill in their credentials at the end.

## Workflow

Copy this checklist into your response and check off each step as you complete it:

```
Registration Progress:
- [ ] Step 1: Collect agent file path and name
- [ ] Step 2: Discover and validate entrypoint function
- [ ] Step 3: Scan for env vars
- [ ] Step 4: Detect model usage and confirm
- [ ] Step 5: Ask for LLM provider
- [ ] Step 6: Run registration
- [ ] Step 7: Create .env file with placeholders
- [ ] Step 8: Summarize
```

### Step 1 — Collect inputs

Ask (use `AskQuestion` for multiple-choice, plain conversation for free-form):

1. **Agent file path** — relative to the project root (e.g. `examples/hotel/agent.py`)
1. **Agent name (slug)** — default to the parent folder name; confirm before proceeding

### Step 2 — Discover and validate the entrypoint function

Read the agent file. Find the entrypoint in priority order:

1. `def run(`
1. `def run_agent(`
1. `def agent(`
1. Any top-level `def` that returns `dict` or `str`

If multiple candidates exist, ask the user to pick one.

**If no entrypoint function is found**, ask the user (use `AskQuestion`):

> "No entrypoint function was found in `<file>`. Would you like me to scaffold a `run` function for you?"
> Options: Yes, scaffold one for me | No, I'll add it manually

- If **No**: tell the user the function should accept at least one input parameter and return a `dict` or `str`, then stop and ask them to re-run the skill after adding it.

- If **Yes**: scaffold a `run` function at the bottom of the file. The scaffolded function must:

  - Be named `run`
  - Accept typed input parameters that make sense for the agent's apparent purpose (infer from the file's other code, imports, constants, and comments)
  - Have a docstring explaining what it does and what each parameter means
  - Return a `dict` with named output keys that reflect what the agent produces
  - Include a short inline comment on each parameter and return field

  Also ensure the **top of the file** loads the agent's `.env`. If `dotenv` imports are not already present, add these lines at the top (after any existing imports):

  ```python
  from dotenv import load_dotenv
  from pathlib import Path

  load_dotenv(Path(__file__).parent / ".env", override=True)
  ```

  If `load_dotenv` is already called anywhere in the file, do not add it again.

  After writing it, **show the scaffolded function in full to the user** and explain it in plain English:

  > "Here's the entrypoint I created:
  >
  > ```python
  > <full scaffolded function>
  > ```
  >
  > **What it does:** \<1–2 sentence plain-English description of the function's purpose>
  > **Inputs:** <list each parameter and what it represents>
  > **Output:** <describe what the returned dict contains>
  >
  > Please review this. If the parameter names, types, or return shape don't match how you plan to call the agent, edit `<file>` now before continuing. When you're ready, reply to proceed."

  Wait for the user to confirm before continuing. Do not proceed to Step 3 until they explicitly say the function looks correct.

**Do not modify the file for any other reason.** Only scaffold when the user explicitly requests it via the question above.

**Once a candidate is found, analyze it — do not modify it:**

1. **Accepts inputs?** — Does the function have at least one parameter (excluding `self`)? If not, stop and tell the user:

   > "The function `<name>` takes no parameters. It needs at least one input parameter so Overmind can pass data to it. Please update `<file>` and re-run this skill."

1. **Returns a value?** — Does the function have a return type annotation of `dict`, `str`, `list`, or similar (not `None`)? Or does the body contain a `return` statement with a non-`None` value? If the function clearly returns nothing, stop and tell the user:

   > "The function `<name>` does not appear to return a value. It should return a `dict` or `str`. Please update `<file>` and re-run this skill."

If both checks pass, proceed — record the exact parameter names and return type for use in later steps.

**Derive the module path** from the file path — strip the extension, replace `/` with `.`:

| File path                 | Module path            |
| ------------------------- | ---------------------- |
| `examples/hotel/agent.py` | `examples.hotel.agent` |
| `examples/support/bot.py` | `examples.support.bot` |
| `agents/myagent/main.py`  | `agents.myagent.main`  |

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

### Step 4 — Detect model usage and confirm

Scan the agent file for any hardcoded model names. Look for patterns like:

- `model="..."` / `model='...'`
- `"model": "..."` in dicts
- Common model name strings: `gpt-4`, `gpt-3.5`, `claude-3`, `claude-opus`, `mistral`, `llama`, etc.

**If a model name is found**, ask the user (use `AskQuestion`):

> "Your agent uses `<detected-model>`. Do you want to keep using this model?"
> Options: Yes, keep `<detected-model>` | No, I want to use a different model

- If **No**: ask them to type the model name they want to use, then update the model string in the agent file before continuing.

**If no model name is detected**, skip this step silently.

### Step 5 — Ask for provider (for .env scaffolding only)

Use `AskQuestion`:

> "Which LLM provider does this agent use?"
> Options: OpenAI | Anthropic | Other (OpenAI-compatible) | No LLM / configure manually

Determine the required key(s):

- **OpenAI** → `OPENAI_API_KEY`
- **Anthropic** → `ANTHROPIC_API_KEY`
- **Other** → `OPENAI_BASE_URL` + `OPENAI_API_KEY`
- **No LLM / manually** → no provider keys

### Step 5b — Final confirmation before registering (only when entrypoint was scaffolded)

**Skip this step if the entrypoint already existed in the file.** Only run it when Step 2 scaffolded the function.

Tell the user:

> "Before I register the agent, please open `<file>` and check that the entrypoint looks correct:
>
> - Parameter names and types match how you intend to call the agent
> - The return dict keys match what the agent actually produces
> - The `load_dotenv` line at the top will find the right `.env`
>
> Take a moment to edit the file if anything needs adjusting. Reply when you're ready to register."

Wait for explicit confirmation. Do not proceed to Step 6 until the user confirms they have reviewed the file and are happy with it.

### Step 6 — Run registration

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

### Step 7 — Create the .env file with placeholders

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

### Step 8 — Summarize

Tell the user:

- Agent name and entrypoint that was registered
- If a `.env` was created: tell them to open `.overmind/agents/<name>/.env` and fill in the placeholder value(s) before running the agent.
- Next step: run `/overmind-generate-dataset` with agent name `<name>`. Mention that if they already have example inputs/outputs for this agent, they can provide a seed dataset file path when running that skill.

Do **not** mention runner scripts, file cleanup, registry internals, or any implementation details.

## Fallback: if overmind internals can't be imported

If the runner fails with `ImportError` on `overmind.commands` or `overmind.core`, overmind is not installed in the active Python environment. Tell the user to install it first, then re-run:

```bash
pip install overmind
python _register_runner.py
```

If the project uses `uv`, run from the **project root** (do NOT `cd` to a parent directory or pass `--project` to `uv run`):

```bash
uv add overmind && uv run python _register_runner.py
```

## Common issues

| Problem                                                  | Fix                                                                                                  |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `Module 'x.y.z' resolves to '...', which does not exist` | Module path is wrong — check slashes vs dots; try the `/`-based form if the path has unusual dirs    |
| `EntrypointNotFoundError`                                | Function name not found in the file — re-read and confirm the spelling                               |
| `Agent already registered` (different entrypoint)        | Use `overmind agent update <name> <entrypoint>`                                                      |
| `EntrypointSignatureError`                               | The function's signature is missing required `dict`/`str` params — offer to generate an auto-wrapper |
| `ImportError: No module named overmind`                  | Install overmind first: `pip install overmind`, then re-run                                          |
| User's agent uses no LLM directly                        | Choose "No LLM / manually" in Step 4 — skip the `.env` unless env vars were discovered in Step 3     |
