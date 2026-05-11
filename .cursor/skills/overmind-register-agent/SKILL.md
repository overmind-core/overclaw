---
name: overmind-register-agent
description: "Create or register an Overmind-compatible agent entrypoint without interactive CLI prompts. Use when the user wants Overmind to run and evaluate an agent, create a separate callable interaction file for an existing agent, register an agent, run overmind agent register, set up agent credentials, configure an LLM provider, add a new agent to an Overmind project, or fix a failed/partial agent registration."
metadata:
  version: "2.0"
  product: "Overmind"
---

# Create and Register an Overmind Agent Entrypoint

Registers an agent in `.overmind/agents.toml` without interactive CLI prompts. Creates a **separate Overmind entrypoint file** — a thin interaction harness that wraps the native agent — rather than registering the native agent file directly.

## Operating principles

- **Codebase-derived artifacts**: Inspect the agent source, adjacent modules, configuration, README files, examples, tests, and existing invocation paths before asking the user for anything.
- **Project-root discipline**: Run all commands from the directory containing `.overmind/`. Do not run from a parent directory.
- **No secret inspection**: Never ask the user to paste API keys into chat. Never print or inspect secret values. Create placeholder entries and tell the user where to fill them in.
- **Separate entrypoint file**: Always create or maintain a distinct entrypoint file for Overmind-agent interaction. Do not register the native agent implementation file directly.
- **Interaction harness, not agent logic**: The entrypoint imports and invokes the native agent, maps dataset inputs to the agent's native call, and normalizes outputs for evaluation. It must not contain optimizable behavior.
- **Entrypoint is fixed and invisible to optimization**: It exists only to let Overmind invoke the agent. Never treat it as logic to optimize.
- **Snapshot safety**: The entrypoint and every local file it imports must live under the project root and be included in the instrumented snapshot.
- **Re-instrument on entrypoint changes**: If the entrypoint changes after registration, refresh the instrumented copy even when the agent name and callable string are unchanged.
- **Minimal edits**: Only modify Overmind registration artifacts, the separate entrypoint file, and the agent-specific `.env` placeholder file.
- **Ask only for blockers**: Prefer autonomous codebase analysis over questions. Ask only when the codebase cannot resolve a material ambiguity.

## Workflow

```
Registration Progress:
- [ ] Step 1: Collect agent path and name
- [ ] Step 2: Ask analyzer model + run credential preflight
- [ ] Step 3: Build codebase context and understand native interface
- [ ] Step 4: Create or validate separate Overmind entrypoint file
- [ ] Step 5: Scan for env vars + ask LLM provider
- [ ] Step 6: Run registration via CLI
- [ ] Step 7: Validate local imports and instrumentation snapshot
- [ ] Step 8: Create .env placeholder file
- [ ] Step 9: Validate with sample data + summarize
```

### Step 1 — Collect agent path and name

Derive from the codebase where possible. Ask only for:

1. **Agent file path** — relative to project root (e.g. `examples/hotel/agent.py`).
1. **Agent name (slug)** — default to the parent folder name; confirm before proceeding.
1. **Entrypoint choice** — ask whether the user already has an Overmind-compatible entrypoint they want to point Overmind at, or whether they want one created:
   - If they have one: ask for the project-relative path and callable if known. Validate it satisfies the entrypoint contract (importable, non-interactive, returns serializable top-level fields) before proceeding to Step 6.
   - If they want one created: continue with codebase inspection below.

### Step 2 — Ask analyzer model + credential preflight

**Collect analyzer choices via `AskQuestion`:**

> "Which provider should Overmind use for its analyzer (failure diagnosis)?"
> Options: Anthropic | OpenAI | Other OpenAI-compatible | Keep existing environment configuration

> "Which analyzer model?"
> Options: `anthropic/claude-sonnet-4-20250514` | `openai/gpt-4o` | Custom model string | Keep existing `ANALYZER_MODEL`

If the user chooses a custom model, collect the model string in a follow-up free-text question.

**Run credential preflight** — bootstrap `.overmind/.env` and check required keys are set without printing values:

```bash
python - <<'PY'
import os, pathlib

env_path = pathlib.Path(".overmind/.env")
if not env_path.exists():
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("# Overmind environment\n")

text = env_path.read_text()
keys = {ln.split("=", 1)[0].strip() for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#") and "=" in ln}

def add_placeholder(key):
    global text
    if not text.endswith("\n"):
        text += "\n"
    text += f"{key}=<set-me>\n"

if "OVERMIND_API_KEY" not in keys:
    add_placeholder("OVERMIND_API_KEY")

selected_provider = "<anthropic|openai|openai-compatible|keep-existing>"
selected_model    = "<anthropic/claude-sonnet-4-20250514|openai/gpt-4o|custom|keep-existing>"

if selected_provider == "anthropic" and "ANTHROPIC_API_KEY" not in keys:
    add_placeholder("ANTHROPIC_API_KEY")
elif selected_provider in {"openai", "openai-compatible"} and "OPENAI_API_KEY" not in keys:
    add_placeholder("OPENAI_API_KEY")
if selected_provider == "openai-compatible" and "OPENAI_BASE_URL" not in keys:
    add_placeholder("OPENAI_BASE_URL")

env_path.write_text(text)

# Write chosen model unless keeping existing
if selected_model not in {"keep-existing"}:
    lines = env_path.read_text().splitlines()
    updated, found = [], False
    for ln in lines:
        if ln.strip().startswith("ANALYZER_MODEL="):
            updated.append(f"ANALYZER_MODEL={selected_model}")
            found = True
        else:
            updated.append(ln)
    if not found:
        updated.append(f"ANALYZER_MODEL={selected_model}")
    env_path.write_text("\n".join(updated).rstrip() + "\n")

# Check readiness (no values printed)
required = ["OVERMIND_API_KEY"]
if selected_provider == "anthropic":
    required.append("ANTHROPIC_API_KEY")
elif selected_provider in {"openai", "openai-compatible"}:
    required.append("OPENAI_API_KEY")

def is_configured(name):
    if os.getenv(name):
        return True
    for line in env_path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        if k.strip() == name and v.strip() and v.strip() != "<set-me>":
            return True
    return False

missing = [k for k in required if not is_configured(k)]
print("configured" if not missing else "missing:" + ",".join(missing))
PY
```

If any key is missing, stop and tell the user to fill in `.overmind/.env` (which was just created), then confirm before continuing.

### Step 3 — Build codebase context and understand native interface

Read:

- The target agent file and adjacent modules.
- CLI commands, app routes, framework runners, or tests that invoke the agent.
- Example inputs/outputs in README files, examples, fixtures, or notebooks.
- Existing Overmind artifacts, if any.
- Environment variable names needed to run the agent.

Identify how the agent is invoked today. Look for public functions (`run`, `run_agent`, `agent`, `main`, `invoke`, `predict`, `respond`, `__call__`), classes with invocation methods, CLI entrypoints, and route handlers. Distinguish the **native interface** (how it runs today) from the **Overmind entrypoint** (the separate harness).

### Step 4 — Create or validate the separate Overmind entrypoint file

The entrypoint file must:

- Be importable from Python.
- Accept explicit, serializable inputs representable in a dataset.
- Return a serializable, evaluator-compatible result.
- Not require interactive input, start a server/UI loop, or run more than once per call.

The entrypoint should expose a single function (`run` or `run_agent`) that:

- Accepts explicit keyword arguments matching dataset input fields.
- Converts them into the native agent's expected input shape.
- Calls the native agent exactly once.
- Normalizes the result into evaluator-compatible **top-level fields** (`text`, `enum`, `number`, `boolean`).
- Loads its `.env` at the top if not already loaded elsewhere:

  ```python
  from dotenv import load_dotenv
  from pathlib import Path
  load_dotenv(Path(__file__).parent / ".env", override=True)
  ```

For native list or nested outputs, normalize into top-level fields (JSON text representation, item count, extracted key fields, validity booleans). Never require the evaluator to score nested keys or list items directly.

After creating the harness, **show it in full to the user** and explain inputs, outputs, and what it does in plain English. Wait for confirmation before continuing.

**Derive the module path**: strip `.py`, replace `/` with `.` (e.g. `examples/hotel/overmind_ep.py` → `examples.hotel.overmind_ep`). Exception: paths with segments starting with `.` must use the slash form. Construct the entrypoint string: `<module_path>:<function_name>`.

**Smoke-check the callable** before registering:

```bash
python - <<'PY'
import importlib, inspect
module = importlib.import_module("<module_path>")
fn = getattr(module, "<callable>")
print("callable" if callable(fn) else "not-callable")
print("async" if inspect.iscoroutinefunction(fn) else "sync")
print(inspect.signature(fn))
PY
```

If import or signature checks fail, repair the entrypoint before continuing. If the entrypoint is async (`async def`), Overmind and downstream smoke tests will wrap it in `asyncio.run` automatically — no harness change is needed, but make sure the agent does not also start its own event loop (e.g. `asyncio.run(...)` inside the entrypoint body), which would fail with `RuntimeError: asyncio.run() cannot be called from a running event loop`.

### Step 5 — Scan for env vars + ask LLM provider

Scan the native agent source and the entrypoint file for `os.environ.get`, `os.getenv`, `os.environ["KEY"]`. Exclude system vars (`PATH HOME USER LOGNAME SHELL TERM LANG PWD TMPDIR TMP TEMP`). Note defaults as placeholder hints.

Use `AskQuestion`:

> "Which LLM provider does this agent use?"
> Options: OpenAI | Anthropic | Other (OpenAI-compatible) | No LLM / configure manually

Provider → required key(s):
- **OpenAI** → `OPENAI_API_KEY`
- **Anthropic** → `ANTHROPIC_API_KEY`
- **Other** → `OPENAI_BASE_URL` + `OPENAI_API_KEY`
- **No LLM** → no provider keys (unless env vars were discovered)

### Step 6 — Run registration via CLI

Check if already registered:

```bash
python - <<'PY'
import pathlib, tomllib
p = pathlib.Path(".overmind/agents.toml")
data = tomllib.loads(p.read_text()) if p.exists() else {}
agents = data.get("agents", {}) if isinstance(data, dict) else {}
print("exists" if "<agent-name>" in agents else "missing")
PY
```

If already registered with the same entrypoint, check whether the entrypoint file changed and refresh instrumentation if needed. If registered with a **different** entrypoint, stop and tell the user to use `overmind agent update <name> <entrypoint>`.

Run registration non-interactively from the project root:

```bash
overmind agent register --help   # inspect supported flags first
overmind agent register "<agent-name>" "<module_path>:<callable>"
```

Pass both parameters explicitly. Do not run interactive prompt flows. If the CLI uses different flag names, map the same values and document the exact command executed.

If registration fails with `ImportError` or `ModuleNotFoundError`:

```bash
# With pip
pip install overmind && overmind agent register "<agent-name>" "<module_path>:<callable>"

# With uv (from project root only — do NOT cd to a parent directory)
uv add overmind && uv run overmind agent register "<agent-name>" "<module_path>:<callable>"
```

Verify registration succeeded:

```bash
python - <<'PY'
import pathlib, tomllib
p = pathlib.Path(".overmind/agents.toml")
data = tomllib.loads(p.read_text())
print(data.get("agents", {}).get("<agent-name>", {}).get("entrypoint", "missing"))
PY
```

Confirm the stored entrypoint matches the expected callable string.

### Step 7 — Validate local imports and instrumentation snapshot

For every local import in the entrypoint file:

- Resolve the imported file or package.
- Confirm it lives under the project root.
- Confirm it will be included in the instrumented snapshot (`.overmind/agents/<name>/instrumented/`).
- If any local import lives outside the project root, stop. Adjust the project structure so all dependencies are inside the registered snapshot.

If the entrypoint changed after a previous registration, refresh the instrumented copy by re-running registration with the same name and callable.

### Step 8 — Create .env placeholder file

Create `.overmind/agents/<name>/.env`:

```
# Overmind agent env — <name>

OPENAI_API_KEY=<your-openai-api-key-here>
# ANTHROPIC_API_KEY=<your-anthropic-api-key-here>
```

For "Other": include both `OPENAI_BASE_URL=<your-base-url-here>` and `OPENAI_API_KEY=<your-key-here>`.

For each additional env var discovered in Step 5, add a placeholder line. Preserve existing non-placeholder lines. Do not overwrite real-looking secret values. Skip the file if "No LLM" was chosen and no env vars were discovered.

### Step 9 — Validate with sample data + summarize

**If the user has sample input data**, run:

```bash
overmind agent validate "<agent-name>" --data "<sample-data-path>"
```

Use a JSON object whose keys exactly match the entrypoint parameter names. This catches parameter-name mismatches before dataset generation. If this fails with a signature error, repair the entrypoint or sample keys before proceeding.

If no sample data is available, skip validation and note that runtime validation requires real credentials and a sample input.

**Tell the user:**

- The agent name and entrypoint (file + callable) registered.
- Which native agent interface the harness invokes.
- Which analyzer model/provider was configured in `.overmind/.env`.
- Whether instrumentation was refreshed.
- Whether an `.env` placeholder file was created or updated; that they must fill in placeholders before running.
- Whether `overmind agent validate` was run or skipped.
- Next steps, in order:
  1. `/overmind-generate-spec-and-dataset` — creates the policy, eval spec, and dataset in one pass so the input/output schemas always agree. Mention the user can pass a seed dataset path here.
  2. `/overmind-preflight` — runs the agent end-to-end against a 2-row slice and autonomously fixes every plumbing issue (eval-spec / dataset / instrumentation) before optimization.
  3. `/overmind-optimise-agent` — runs the iterative optimization loop. It will refuse to start until preflight is green.

## Common issues

| Problem | Fix |
|---|---|
| `Module 'x.y.z' resolves to '...', which does not exist` | Module path is wrong — check slashes vs dots; try slash form for unusual dirs |
| `EntrypointNotFoundError` | Function name not found in the file — re-read and confirm the spelling |
| `Agent already registered` (different entrypoint) | Use `overmind agent update <name> <entrypoint>` |
| `EntrypointSignatureError` | Function missing required params — offer to repair the harness |
| `ImportError: No module named overmind` | Run `pip install overmind` or `uv add overmind`, then re-run |
| Agent has no importable callable | Create a separate Overmind entrypoint harness |
| Agent starts a server or UI loop | Create a harness that calls the underlying one-shot inference function |
| Agent returns custom objects | Normalize into top-level `text`, `enum`, `number`, or `boolean` fields |
| Entrypoint changed but registration exists | Refresh instrumentation by re-running registration |
| Entrypoint imports files outside project root | Move the harness or adjust project structure |
| `OVERMIND_API_KEY` or `ANALYZER_MODEL` missing | Fill in `.overmind/.env`; confirm before continuing |
| No direct LLM usage | Choose "No LLM / manually"; skip provider placeholders unless env vars were discovered |
