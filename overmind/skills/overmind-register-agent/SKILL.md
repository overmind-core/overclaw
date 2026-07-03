______________________________________________________________________

## name: overmind-register-agent description: "Create or register an Overmind-compatible agent entrypoint: deterministic CLI checks and env gates, with the coding agent doing harness synthesis, stub JSON, and file IO. Use when the user wants Overmind to run and evaluate an agent, register an agent, configure providers, or fix failed registration." metadata: version: "2.4" product: "Overmind"

# Create and Register an Overmind Agent Entrypoint
> Note; before trying to find agents locally, first try to fetch from remote and see if the agents are already extracted from the codebase.

OVERMIND_API_KEY=os.environ.get('OVERMIND_API_KEY')
OVERMIND_API_URL=os.environ.get('OVERMIND_API_URL', 'https://api.overmindlab.ai')
OVERMIND_PROJECT_ID=os.environ.get('OVERMIND_PROJECT_ID')

try to get the agents information for this project from this url with the specified header
curl -H "Authorization: ${OVERMIND_API_KEY}" "${OVERMIND_API_URL}/api/agents/?ordering=name&project=${OVERMIND_PROJECT_ID}"
if agents are available save them and move onto the next step otherwise continue


Use this skill to create a separate entrypoint file that Overmind can call to run and evaluate an agent, then register that entrypoint in `.overmind/agents.toml`. The entrypoint file is the stable interaction contract between Overmind and the agent under test. It is not the native agent implementation, and it is not optimization material.

**Split responsibility:** Use **deterministic** steps for anything a script can decide (project root, registry checks, `overmind agent register` argv, `configured_in_file` parity with `overmind init`, validate CLI). Use the **coding agent** for everything that needs reasoning (reading the codebase, drafting the harness, building minimal JSON stubs from the locked schema, reconciling dataset vs code). Do not paste secrets into chat.

**Setup coverage (no silent skips):** Every topic in **Inputs** must be **resolved once per run** — either via `AskQuestion` / chat, or via a **one-line explicit confirmation** when the user already answered in the same conversation before this skill run (quote their choice, ask “still correct?”). Never proceed without resolving each topic.

> always export the environment variable OVERMIND_API_KEY if present in the `.env`

## Operating principles

- **Mandatory topic coverage**: Cover every item in **Inputs** in order (entrypoint choice, agent path, name, dataset presence, analyzer provider, model, agent LLM provider, **mandatory keys pause**, credentials verified). Do not infer away a user choice; use in-thread short-circuit only as above.
- **Codebase-derived artifacts**: After collecting user choices, derive the Overmind entrypoint from the agentic codebase context. Inspect the agent source, adjacent modules, configuration, README files, examples, tests, and existing invocation paths.
- **Project-root discipline**: Run all commands from the project root, defined as the directory containing `.overmind/`. Do not run from a parent directory.
- **No secret inspection**: Never ask the user to paste API keys into chat. Never print or inspect secret values. Create placeholder entries and tell the user where to fill them in.
- **Separate entrypoint file**: Always create or maintain a distinct entrypoint file for Overmind-agent interaction. Do not register the native agent implementation file directly.
- **Interaction harness, not agent logic**: The entrypoint file should be a thin interaction harness that imports and invokes the native agent, maps dataset inputs into the agent’s native call, and normalizes outputs for evaluation. It must not contain optimizable behavior.
- **Cold-start aware harness:** Overmind runs evaluations in isolated processes. When wiring the native agent, avoid rebuilding the full orchestrator stack, clients, or tool registries on every harness call; pay fixed construction cost once per interpreter process and reset only per-conversation state each call. This is harness plumbing, not new business logic.
- **Entrypoint is fixed and invisible to optimization**: The entrypoint file exists only to let Overmind invoke the agent. It must never be treated as agent logic to optimize.
- **Snapshot safety**: The entrypoint file and every local file it imports must live under the project root and be included in the instrumented snapshot. Do not import local files from outside the project root.
- **Re-instrument on entrypoint changes**: If the entrypoint file changes after registration, refresh the instrumented copy even when the agent name and callable string are unchanged.
- **Minimal edits**: Only modify Overmind registration artifacts and the separate entrypoint file.
- **Clean temporary files**: If you create helper files to execute registration logic, delete them after success or after a terminal failure.
- **Environment**: Overmind loads a **single** env file — `.overmind/.env` — at the project root. There is no per-agent `.env`; provider keys, the analyzer model, and any agent runtime variables live in the project file. (Historically `.overmind/agents/<name>/.env` was loaded with `override=True`, which caused placeholders to silently win over real project values — that path has been removed.)

## Inputs

**Execution order:** For anything that touches `.overmind/.env`, follow **Workflow** (analyzer provider → model → **command block 4** bootstrap → **mandatory keys pause** → **command block 5** verify → `configured`) **before** `overmind agent register`. You may still collect other answers (entrypoint choice, paths) earlier in the thread, but do not register until this gate has passed.

The coding agent infers technical facts from the repo; the user still **confirms** intent for each bullet below (or confirms prior message).

- **Entrypoint choice**: Before creating or updating registration artifacts, ask whether the user already has an Overmind-compatible entrypoint they want to point Overmind at, or whether they want the agent to create a compatible entrypoint.
  - If the user has an entrypoint, ask for the project-relative path and callable if known. Validate it before registering.
  - If the user wants a compatible entrypoint created, inspect the native agent implementation and create a separate thin Overmind interaction harness.
- **Agent file path**: A path relative to the project root, such as `examples/hotel/agent.py`.
- **Agent name**: A slug. Suggest the parent folder name of the agent file, but **always** confirm with the user before proceeding.
- **Native invocation shape**: Infer from code, tests, examples, CLI definitions, app routes, or README usage. If multiple incompatible invocation paths remain after inspection, ask the user which contract to target before writing the harness.
- **Dataset file (AskQuestion, required)**: Ask whether the user has a representative dataset or examples file (JSON, JSONL, or CSV).
  - If **yes**: ask for the project-relative path, read it, and reconcile field names with the codebase-derived interface before locking the entrypoint schema (same rules as “Read dataset and reconcile schema”).
  - If **no**: derive the schema from the codebase, get explicit user approval on the locked schema, then the coding agent **writes** a minimal JSON file (see **Validate stub**, below) so `overmind agent validate --data` still runs deterministically.
- **Analyzer provider (AskQuestion, required)**: First ask only for the **provider** — Anthropic, OpenAI, Other OpenAI-compatible, or Keep existing environment configuration. Do not show model IDs in this step.
- **Analyzer model (AskQuestion, required)**: Second step: based on the chosen provider, ask which **model** to persist as `ANALYZER_MODEL` (multiple-choice from a short curated list for that provider, e.g. Anthropic: `anthropic/claude-sonnet-4-20250514`; OpenAI: `openai/gpt-4o`; OpenAI-compatible: custom LiteLLM id via follow-up free text only if needed). If the user chose **Keep existing**, ask them to confirm that `ANALYZER_MODEL` is already set correctly in `.overmind/.env` or the host environment (yes/no); if not, return to provider+model selection.
- **LLM provider for placeholders**: Ask whether the agent under test uses OpenAI, Anthropic, another OpenAI-compatible provider, or no directly configured LLM.
- **Mandatory keys pause (AskQuestion, strict — never skip)**: Immediately after **command block 4** has created/updated `.overmind/.env` with placeholders, **stop all forward progress** (no harness work, no `overmind agent register`) until this step completes.
  - Show the **absolute path** to `.overmind/.env` and list **by variable name only** what the user must fill: always `OVERMIND_API_KEY`, plus the analyzer provider key line(s) present in the file (`ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY`, and `OPENAI_BASE_URL` when OpenAI-compatible). Remind them `ANALYZER_MODEL` is already set unless they chose keep-existing and still need to fix it.
  - **Required `AskQuestion` prompt text** (adapt only for host UI limits; meaning must be preserved):
    *Fill in your Overmind API key and your analyzer model provider key(s) in `.overmind/.env` (edit the file on disk). Do not paste secrets in this chat. When you have saved real values, click **Yes — continue**.*
  - **Options**: **Yes — continue** | **No — which keys are still missing?**
  - On **No**: run **command block 5** (verify) and tell them only the `missing:` names from the script output; then show the same `AskQuestion` again.
  - On **Yes**: run **command block 5**. If output is not exactly `configured`, do **not** continue — explain which names still fail (from `missing:` only), send the user back to edit the file, and **repeat the same `AskQuestion`** until block 5 prints `configured`.
- **Analyzer provider key readiness**: Covered by command block 5 after the pause; same `key_ok` rules as `overmind init` (`overmind/commands/init_cmd.py`).

Use the host agent’s normal user-question mechanism. If no structured question tool exists, ask plainly in chat.

## Required command blocks (non-interactive)

Use these command blocks to keep registration deterministic. Do not rely on interactive prompts.

1. **Project root preflight**

   - Required: run from the directory that contains `.overmind/`.
   - Command:
     - `test -d .overmind`
   - If this fails, stop and resolve the correct project root before continuing.

1. **Registry preflight**

   - Required inputs: `<agent-name>`.
   - Command:
     - `python - <<'PY'`
       `import pathlib, tomllib`
       `p = pathlib.Path(".overmind/agents.toml")`
       `data = tomllib.loads(p.read_text()) if p.exists() else {}`
       `agents = data.get("agents", {}) if isinstance(data, dict) else {}`
       `print("exists" if "<agent-name>" in agents else "missing")`
       `PY`
   - Use this to detect whether registration already exists before writing.

1. **Init configuration gate (AskQuestion required, two steps)**

   - Required outcome: collect **analyzer provider** first, then **analyzer model** from that provider, before registration.
   - Step A — Provider (multiple choice): Anthropic, OpenAI, Other OpenAI-compatible, Keep existing environment configuration.
   - Step B — Model (multiple choice, conditional on Step A): offer provider-appropriate LiteLLM model ids; include custom model string only when needed (OpenAI-compatible or user requests). If Step A is Keep existing, confirm `ANALYZER_MODEL` is already correct; if not, re-run Step A–B.

1. **Bootstrap `.overmind/.env` (write placeholders only)**

   - Run **once** after analyzer provider and model are known. Creates the file and **all required placeholder lines** (`OVERMIND_API_KEY=<set-me>`, provider keys, `ANALYZER_MODEL` from the user’s model choice). Does **not** require real secrets yet. Success line: `bootstrap_complete`.
   - Command:
     - `python - <<'PY'`
       `import pathlib`
       `env_path = pathlib.Path(".overmind/.env")`
       `env_path.parent.mkdir(parents=True, exist_ok=True)`
       `text = env_path.read_text() if env_path.is_file() else "# Overmind environment\n"`
       `keys = {ln.split("=", 1)[0].strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#") and "=" in ln}`
       `if "OVERMIND_API_KEY" not in keys:`
       `    if text and not text.endswith("\n"): text += "\n"`
       `    text += "OVERMIND_API_KEY=<set-me>\n"`
       `if "ANALYZER_MODEL" not in keys:`
       `    if text and not text.endswith("\n"): text += "\n"`
       `    text += "ANALYZER_MODEL=<set-me>\n"`
       `selected_provider = "<analyzer-provider-choice>"  # anthropic | openai | openai-compatible | keep-existing`
       `selected_model = "<analyzer-model-choice>"  # keep-existing | anthropic/claude-sonnet-4-20250514 | openai/gpt-4o | <custom-model>`
       `if selected_provider == "anthropic" and "ANTHROPIC_API_KEY" not in keys:`
       `    if text and not text.endswith("\n"): text += "\n"`
       `    text += "ANTHROPIC_API_KEY=<set-me>\n"`
       `elif selected_provider in {"openai", "openai-compatible"} and "OPENAI_API_KEY" not in keys:`
       `    if text and not text.endswith("\n"): text += "\n"`
       `    text += "OPENAI_API_KEY=<set-me>\n"`
       `if selected_provider == "openai-compatible" and "OPENAI_BASE_URL" not in keys:`
       `    if text and not text.endswith("\n"): text += "\n"`
       `    text += "OPENAI_BASE_URL=<set-me>\n"`
       `env_path.write_text(text)`
       `if selected_model != "keep-existing":`
       `    lines = env_path.read_text().splitlines()`
       `    updated, found = [], False`
       `    for ln in lines:`
       `        if ln.strip().startswith("ANALYZER_MODEL="):`
       `            updated.append(f"ANALYZER_MODEL={selected_model}")`
       `            found = True`
       `        else:`
       `            updated.append(ln)`
       `    if not found:`
       `        updated.append(f"ANALYZER_MODEL={selected_model}")`
       `    env_path.write_text("\n".join(updated).rstrip() + "\n")`
       `print("bootstrap_complete")`
       `PY`
   - Never print secret values.

1. **Verify `.overmind/.env` keys (same semantics as `overmind init` `key_ok`)**

   - Run **after** the user has confirmed the mandatory keys pause (workflow). Prints `configured` or `missing:name,name,...` (names only). Uses `os.environ` merged with file values the same way as the prior combined script.
   - Command:
     - `python - <<'PY'`
       `import os, pathlib, re`
       `def key_ok(raw):`
       `    v = (raw or "").strip()`
       `    if not v or v == "<set-me>":`
       `        return False`
       `    if re.fullmatch(r"your[-_]?key[-_]?here|changeme|xxx+", v, re.IGNORECASE):`
       `        return False`
       `    return True`
       `def configured_in_file(name, env_path):`
       `    if not env_path.is_file():`
       `        return False`
       `    for line in env_path.read_text().splitlines():`
       `        s = line.strip()`
       `        if not s or s.startswith("#") or "=" not in s:`
       `            continue`
       `        k, v = s.split("=", 1)`
       `        if k.strip() == name:`
       `            return key_ok(v)`
       `    return False`
       `env_path = pathlib.Path(".overmind/.env")`
       `selected_provider = "<analyzer-provider-choice>"`
       `selected_model = "<analyzer-model-choice>"`
       `required = ["OVERMIND_API_KEY"]`
       `if selected_provider == "anthropic":`
       `    required.append("ANTHROPIC_API_KEY")`
       `elif selected_provider in {"openai", "openai-compatible"}:`
       `    required.append("OPENAI_API_KEY")`
       `required_nonsecret = []`
       `if selected_model == "keep-existing":`
       `    required_nonsecret.append("ANALYZER_MODEL")`
       `if selected_provider == "openai-compatible":`
       `    required_nonsecret.append("OPENAI_BASE_URL")`
       `missing = [k for k in required if not (os.getenv(k) or configured_in_file(k, env_path))]`
       `missing += [k for k in required_nonsecret if not (os.getenv(k) or configured_in_file(k, env_path))]`
       `print("configured" if not missing else "missing:" + ",".join(missing))`
       `PY`
   - Do not register until this prints `configured`. If `missing:...`, return the user to the mandatory pause step — never ask them to paste values in chat.

1. **Deterministic registration command**

   - Required inputs: `<agent-name>`, `<module_path>:<callable>`.
   - First inspect help and run the non-interactive form supported by the installed Overmind version.
   - Commands:
     - `overmind agent register --help`
     - Preferred shape (when supported):
       - `overmind agent register --non-interactive "<agent-name>" "<module_path>:<callable>"`
     - Fallback (older builds without the flag): `OVERMIND_NONINTERACTIVE=1 overmind agent register "<agent-name>" "<module_path>:<callable>"`
   - Rules:
     - Pass both required parameters explicitly in the command invocation.
     - Always pass `--non-interactive` (or set `OVERMIND_NONINTERACTIVE=1`) so the CLI never opens `/dev/tty`; this is required in sandboxed/CI shells where the arrow-key menu would crash with `OSError: Device not configured`.
     - Provider keys live in `.overmind/.env` (configured by `overmind init`); register does not prompt for or write per-agent env files.
     - If the installed CLI requires different flag names, map to the same required values and document the exact command executed in the user update.

1. **Instrumentation refresh**

   - Required whenever the entrypoint file changed, even when agent name and callable are unchanged.
   - Command shape:
     - Use the project-local Overmind registration refresh path that updates `.overmind/agents/<agent-name>/...` snapshot artifacts.
   - If the CLI exposes an explicit refresh command, run that command non-interactively. Otherwise rerun registration with the same agent name and callable using the non-interactive form so instrumentation is rebuilt.

1. **Entrypoint smoke-check command**

   - Required inputs: `<module_path>`, `<callable>`.
   - Command:
     - `python - <<'PY'`
       `import importlib, inspect`
       `module = importlib.import_module("<module_path>")`
       `fn = getattr(module, "<callable>")`
       `print("callable" if callable(fn) else "not-callable")`
       `print(inspect.signature(fn))`
       `PY`
   - If import or signature checks fail, stop and repair the entrypoint before finalizing registration.

1. **Post-registration verification**

   - Required inputs: `<agent-name>`.
   - Command:
     - `python - <<'PY'`
       `import pathlib, tomllib`
       `p = pathlib.Path(".overmind/agents.toml")`
       `data = tomllib.loads(p.read_text())`
       `agents = data.get("agents", {})`
       `print(agents.get("<agent-name>", {}).get("entrypoint", "missing"))`
       `PY`
   - Confirm the stored entrypoint matches the expected callable string.

1. **Optional runtime validation with sample input**

- Required inputs: `<agent-name>`, `<sample-data-path>` (only when the user confirmed they have a dataset or sample JSON file during Inputs).
- Command:
  - `overmind agent validate "<agent-name>" --data "<sample-data-path>"`
- Sample data shape:
  - JSON object whose keys match the entrypoint parameter names.
- If the user had **no** dataset file, skip this step unless they provide optional sample JSON later.
- If this fails with a signature error, repair either the sample keys or the entrypoint signature before proceeding to spec generation.

## Workflow

### Locate and verify the project

Confirm the working directory is the project root. It should contain `.overmind/`, or enough project files to identify where `.overmind/` should be created or read. If the root is ambiguous, ask the user to identify the project path before taking action.

After this project-root step passes, `.overmind/.env` is first **materialized with placeholders** in **command block 4** (after provider/model choices), then the **mandatory keys pause** runs before any registration.

### Init analyzer provider and model (AskQuestion, two steps)

Before registration, run **two** multiple-choice steps unless the user already chose provider+model in the **same conversation** — then quote their answers and ask “Still use these?” once.

1. **Provider**: Anthropic, OpenAI, Other OpenAI-compatible, Keep existing environment configuration.
1. **Model**: Offer models appropriate to the provider chosen in step 1. If Keep existing, you will confirm `ANALYZER_MODEL` passes **command block 5** after the mandatory keys pause and bootstrap; if not, return to step 1.

Persist the chosen model in `.overmind/.env` as `ANALYZER_MODEL=<chosen-model>` unless the user chose Keep existing and confirmed the existing value.

### Bootstrap `.overmind/.env` and mandatory keys pause (strict)

Order is fixed:

1. Run **command block 4** (bootstrap) so `.overmind/.env` exists and contains `OVERMIND_API_KEY=<set-me>`, the chosen `ANALYZER_MODEL` (or placeholder if not yet chosen), and provider placeholder lines matching the selected analyzer provider.
1. **Immediately** run the **Mandatory keys pause** from **Inputs** (the strict `AskQuestion` with *Fill in your Overmind API key and your analyzer model provider key(s)…* and **Yes — continue**). Do not interleave other work before the user has clicked **Yes — continue** at least once.
1. Run **command block 5** (verify). Loop pause + verify until the script prints `configured`.
1. Only then continue with entrypoint harness work and later `overmind agent register`.

### Choose the entrypoint path

Ask the user whether they want to point Overmind at an existing Overmind-compatible entrypoint or have the agent create one.

If the user provides an existing entrypoint, validate that it:

- Lives under the project root.
- Is importable from Python.
- Exposes a non-interactive callable.
- Accepts serializable inputs.
- Returns evaluator-compatible, serializable output.
- Does not start a server, UI loop, or long-running process.

If the existing entrypoint is not compatible, explain the incompatibility and ask whether to repair it or create a new separate Overmind entrypoint.

If the user wants the agent to create a compatible entrypoint, continue with codebase inspection and create a separate interaction harness as described below.

### Build codebase context

Before designing the entrypoint, build a focused context bundle from the codebase:

- The target agent file or likely agent files.
- Adjacent modules imported by the agent.
- CLI commands, app routes, framework runners, or tests that invoke the agent.
- Example inputs and outputs in README files, examples, fixtures, tests, or notebooks.
- Existing Overmind artifacts, if any.
- Environment variable names needed to run the agent.

Use this codebase context as the source of truth for the entrypoint contract. Do not rely on user elicitation for facts that are present in code.

### Understand the native agent interface

Read the agent file and identify how the agent is actually invoked today. Look for:

- Public functions such as `run`, `run_agent`, `agent`, `main`, `invoke`, `predict`, `generate`, `respond`, or `__call__`.
- Classes that encapsulate the agent and expose an invocation method.
- CLI entrypoints, framework runners, or route handlers that call the agent.
- Input parsing logic that reveals the fields Overmind should provide.
- Return values or response construction logic that reveal the output fields Overmind should evaluate.

Distinguish between the **native agent interface** and the **Overmind entrypoint**. The native interface is how the code currently runs. The Overmind entrypoint is the separate interaction file and callable Overmind will invoke.

### Read dataset and reconcile schema (when user has a file)

During Inputs you already asked whether a dataset file exists.

- If **yes**: read the file, extract input and output field names, cross-reference with the codebase-derived interface, resolve conflicts, and lock the schema before writing the entrypoint.
- If **no**: derive the proposed input/output schema from the codebase only, present it to the user for **explicit approval**, then lock the schema. Do not invent fields without user sign-off when no dataset is present.

Do not write the entrypoint file until the locked schema is agreed.

### Create the separate Overmind entrypoint file

Create or maintain a separate entrypoint file dedicated to Overmind-agent interaction using the locked schema. When a dataset was used, the function's keyword argument names must exactly match the locked input field names and return keys must match locked outputs. When no dataset was used, the locked schema is the user-approved codebase-only schema. Prefer a stable name that clearly marks it as an evaluation harness, such as an Overmind-specific entrypoint module near the agent source.

The entrypoint file must satisfy the Overmind entrypoint contract:

- It can be imported from Python.
- It accepts explicit, serializable inputs that can be represented in a dataset.
- It returns a serializable, evaluator-compatible result.
- It does not require interactive input.
- It does not start a long-running server or UI loop.
- It performs one agent run per call.

The entrypoint file should:

- Expose a single top-level function, preferably named `run` or `run_agent`.
- Accept explicit keyword arguments matching the evaluation dataset inputs.
- Convert those arguments into the native agent’s expected input shape.
- Call the native agent exactly once.
- Normalize the result into evaluator-compatible top-level fields.
- Preserve the agent’s behavior and avoid refactoring internal logic.
- Avoid hiding errors that evaluation should surface.
- Be documented or named as an evaluation harness so future artifact generation excludes it from optimization.

If the native agent only runs through a CLI, the entrypoint file should import and call the same underlying function the CLI uses. Use subprocess invocation only as a last resort, and only when outputs can be captured reliably.

If multiple entrypoint designs are plausible, prefer the one supported by tests, examples, docs, or actual production invocation paths. Ask the user only if the codebase supports multiple materially different contracts with no clear canonical path.

### Normalize outputs for evaluation

The entrypoint file must return an evaluation-friendly shape. Overmind evaluation scores top-level output fields. It should not rely on nested dictionaries or raw lists being scored correctly.

Normalize native outputs into a top-level dictionary whose values can be evaluated as:

- `text`
- `enum`
- `number`
- `boolean`

For native list or nested outputs, expose top-level evaluation fields such as a JSON text representation, item count, extracted key fields, validity booleans, or summary text. Preserve the raw native output only as a diagnostic field if useful, and do not make diagnostic fields the primary scoring target.

Never design the entrypoint so the evaluator must score nested keys or list items directly.

### Discover or derive the callable string

After creating the separate Overmind entrypoint file, identify the callable in this priority order:

- `run`
- `run_agent`
- A single explicitly documented top-level evaluation function

If multiple candidate callables remain inside the entrypoint file, simplify the file so it exposes one canonical evaluation callable.

Derive the module path from the relative file path by removing `.py` and replacing path separators with dots. For example, `examples/hotel/agent.py` becomes `examples.hotel.agent`. If any path segment starts with `.`, keep the slash-style path because dotted Python imports cannot start with a dot.

Construct the entrypoint as `<module_path>:<function_name>`. This is the value Overmind will use to import and invoke the agent.

### Scan environment requirements

Scan the native agent source and the separate entrypoint file for environment variable lookups through common Python patterns such as `os.environ.get`, `os.getenv`, and direct `os.environ[...]` access. Exclude common system variables such as `PATH`, `HOME`, `USER`, `LOGNAME`, `SHELL`, `TERM`, `LANG`, `PWD`, `TMPDIR`, `TMP`, and `TEMP`.

Record any defaults found in the code as placeholder hints, but do not expose existing secret values.

### Credential readiness gate

Before running registration, ensure `.overmind/.env` exists and **command block 5** has already printed `configured` after the mandatory keys pause. Then double-check:

- `OVERMIND_API_KEY` passes **`key_ok`** (verified by block 5).
- `ANALYZER_MODEL` passes **`key_ok`** in `.overmind/.env` or host environment.
- Analyzer-provider keys required for the selected provider pass **`key_ok`**.

Do not ask the user to paste any key in chat. If any required value fails `key_ok`, stop and instruct the user to set it locally, then re-run the command block 5 script after they confirm.

### Determine provider placeholders

Create provider placeholders based on the user’s answer:

- **OpenAI**: Add `OPENAI_API_KEY`.
- **Anthropic**: Add `ANTHROPIC_API_KEY`.
- **Other OpenAI-compatible provider**: Add `OPENAI_BASE_URL` and `OPENAI_API_KEY`.
- **No LLM or manual configuration**: Do not add provider keys unless environment variables were discovered from code.

Include discovered non-system environment variables that are not already covered by the provider selection.

### Run registration

Register the separate Overmind entrypoint file by invoking Overmind’s project-local registration internals or CLI from the project root. The registration operation must:

- Load Overmind environment configuration.
- Initialize Overmind.
- Check whether the agent name already exists in the registry.
- If the existing entrypoint matches, still check whether the entrypoint file or its local imports changed after the previous instrumentation and refresh the instrumented snapshot when needed.
- If the existing entrypoint differs, stop and tell the user to choose whether to update the registration.
- Validate that the entrypoint resolves, the function exists, and the function can be called non-interactively with serializable inputs.
- Save the agent name and entrypoint to `.overmind/agents.toml`.
- Instrument or copy the entrypoint file and all local files it imports into the expected `.overmind/agents/<name>/` location if the Overmind project requires that step.

Always refresh instrumentation after changing the entrypoint file, even if registration already exists. A stale instrumented copy can cause Overmind to run old entrypoint logic.

If direct imports fail because Overmind is not importable, retry from the project’s virtual environment. If the project uses `uv`, use the project-local `uv` execution path from the project root. Do not pass a parent project path or change into a parent directory.

### Validate local imports and instrumentation snapshot

Before finalizing registration, inspect imports in the entrypoint file. For every local import:

- Resolve the imported file or package.
- Confirm it lives under the project root.
- Confirm it will be included in the instrumented snapshot.
- If it is a sibling module, ensure registration/instrumentation includes the sibling source tree.
- If it lives outside the project root, do not rely on that import. Move the interaction harness or adjust project structure so the dependency is inside the registered project snapshot.

Do not create an entrypoint file that imports local code unavailable to Overmind at runtime.

### Agent runtime environment

Agent runtime variables (provider API keys, base URLs, etc.) live in **the project `.overmind/.env`** alongside the Overmind/analyzer keys configured by `overmind init`. There is no per-agent `.env`; do not create or write to `.overmind/agents/<name>/.env`. If the harness needs a new key the user has not configured, ask them to add it to `.overmind/.env` before running `overmind agent validate`.

### Smoke-check the entrypoint contract

Before finalizing registration, run a lightweight non-destructive check of the entrypoint contract when feasible:

- Import the entrypoint.
- Inspect its signature.
- Confirm required parameters can be represented as dataset input fields.
- If safe sample inputs are available, call the entrypoint once and confirm the result is serializable.

If a real call would trigger external API cost or side effects, do not call it without user approval. In that case, validate by import and signature inspection only, then report that runtime validation still requires real credentials and sample input.

### Validate with `overmind agent validate` (always)

After registration and instrumentation refresh, always run the CLI validate step once:

- If the user **provided** a dataset or examples file:
  `overmind agent validate "<agent-name>" --data "<user-path>"`
  (directory of JSON cases is allowed if the CLI supports it.)

- If the user had **no** dataset file: the coding agent **writes** a temporary JSON file under the project root (e.g. `_register_validate_stub.json`) containing **one object** whose keys are exactly the entrypoint keyword parameters, using **type-appropriate safe dummy values** derived from the locked schema (empty string, `0`, `false`, short enum literal, etc.). Then run:
  `overmind agent validate "<agent-name>" --data "<path-to-that-json>"`
  Delete the stub file after success or after a terminal failure (unless the user asks to keep it).

If validation fails, read the **innermost** error (imports, kwargs, tracebacks) before attributing failure to API keys.

If a real validate call would intentionally hit paid external APIs and the user has not approved that spend, still run validate when the stub avoids real calls; if the harness always hits the network, obtain one-line user approval or stop with a clear reason after import/signature checks.

### Final validation

Before responding, verify:

- The registry contains the agent name.
- The saved entrypoint matches the discovered entrypoint.
- The separate entrypoint file is callable by Overmind.
- The entrypoint file is identified as a fixed harness that must be ignored by Overmind optimization.
- Analyzer provider and model were collected in **two** AskQuestion steps (or keep-existing with explicit confirmation).
- **Command block 4** ran so `.overmind/.env` was created/updated with placeholders, then the **mandatory keys pause** (`AskQuestion`: *Fill in your Overmind API key and your analyzer model provider key(s)…* / **Yes — continue**) ran **without skipping**, and **command block 5** printed `configured` before registration.
- The instrumented snapshot has been refreshed after any entrypoint file change.
- Every local import used by the entrypoint file is inside the project root and included in instrumentation.
- Temporary helper files have been removed.
- If a dataset file was used: it was read and field names reconciled before the entrypoint was written; signature and return keys match the locked schema.
- If no dataset file: the user explicitly approved the codebase-derived schema before the entrypoint was written.
- `overmind agent validate --data` was run successfully (user dataset path **or** generated stub path).

## User-facing summary

Tell the user:

- The agent name that was registered.
- The separate Overmind entrypoint file and callable that were created or updated.
- Which native agent interface the entrypoint file invokes.
- Whether instrumentation was refreshed.
- Which analyzer model/provider was selected for init.
- That `.overmind/.env` was bootstrapped with placeholders, they completed the **mandatory keys** step (Overmind API key + provider key(s)), and verification passed before registration.
- Whether `overmind agent validate --data` used the user’s dataset file or a **temporary stub JSON** the agent generated from the locked schema (stub path deleted afterward unless the user opted in to keep it).
- Which dataset file was used (if any) and whether any field name conflicts between the dataset and the codebase were found and resolved.
- That all agent runtime keys live in `.overmind/.env` (there is no per-agent `.env`); they should fill in any remaining placeholders there before running the agent.
- The next recommended step: run `/overmind-generate-spec-and-dataset` for the agent.

Do not mention temporary helper files, registry internals, or implementation details unless the user asks.

## Common issues

- **Module path resolves to the wrong file**: Re-check slash-to-dot conversion and path segments starting with `.`.
- **Entrypoint not found**: Re-read the file and confirm the function name with the user.
- **Agent has no importable callable**: Create a separate Overmind entrypoint file that invokes the native path.
- **Agent starts a server or UI loop**: Do not register that directly; create a separate entrypoint file that invokes the underlying one-shot inference call instead.
- **Agent returns custom objects**: Normalize the entrypoint output into top-level text, enum, number, or boolean fields.
- **Entrypoint changed but registration already exists**: Refresh instrumentation; do not assume the instrumented copy is current.
- **Entrypoint imports sibling files that are missing at runtime**: Ensure all local imports live under the project root and are included in the instrumented snapshot.
- **Agent already registered with a different entrypoint**: Stop and ask before updating the registration.
- **Entrypoint signature error**: Explain which required parameters are missing or incompatible, and offer to repair the separate entrypoint file.
- **Overmind import failure**: Activate the project virtual environment or run through the project’s package manager from the project root.
- **No direct LLM usage**: Skip provider placeholders unless the code explicitly reads environment variables.
- **Dataset field not found in codebase**: Ask the user whether it is a naming mismatch or a parameter the native agent does not yet expose. Do not write the entrypoint until the conflict is resolved.
- **Codebase field not found in dataset**: Check whether the native code provides a default value. If a default exists, mark the field optional. If not, flag it to the user rather than silently omitting it from the entrypoint signature.
- **Validate fails on stub only**: Prefer fixing the harness signature vs. editing many callers; ensure stub keys match **keyword** parameters exactly.
