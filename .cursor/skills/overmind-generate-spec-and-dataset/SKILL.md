______________________________________________________________________

## name: overmind-generate-spec-and-dataset description: "Generate the policy, eval spec, and evaluation dataset for an Overmind agent in one pass. Use when the user wants to author or rebuild eval criteria for an agent, fix a broken eval spec (wrong input_schema, missing output fields, bad weights), produce or augment a dataset, or prepare everything needed before running `overmind optimize`. Combines policy elicitation, spec construction, and dataset generation so the artifacts always agree on the same input/output schema." metadata: version: "1.3" product: "Overmind"

# Generate the Policy, Eval Spec, and Dataset

Builds the three canonical artifacts that drive optimization, in a single
ordered pass so the input/output schemas always agree:

1. `.overmind/agents/<name>/setup_spec/policies.md` — domain knowledge, constraints, edge cases.
1. `.overmind/agents/<name>/setup_spec/eval_spec.json` — scoring spec (input_schema, output_fields, weights, tools, embedded policy).
1. `.overmind/agents/<name>/setup_spec/dataset.json` — synthetic + seed test cases that conform to the eval spec.

This skill replaces the two earlier ones (`overmind-generate-policy-and-eval` and `overmind-generate-dataset`). Doing both in one pass eliminates the most common failure mode of the old flow: a dataset that was generated against one input/output shape and an eval spec that scores a different shape.

> always export the environment variable OVERMIND_API_KEY if present in the `.env`

After this skill finishes, run `/overmind-optimize-agent` or `overmind optimize <agent>` to start optimization.

## When to use this skill

- After `/overmind-register-agent`, to create or rebuild `policies.md`, `eval_spec.json`, and `dataset.json` so schemas agree.
- When the eval spec is broken (weights, `input_schema`, output types, scope) or the dataset does not match the entrypoint.
- When preparing for optimization and setup artifacts are missing or stale.

## When not to use this skill

- The agent is not registered or the Overmind harness does not exist — use `/overmind-register-agent` first.
- The user only wants the CLI optimization loop and artifacts already exist and validate — use `/overmind-optimize-agent` (do not use optimize alone to “fix” registration).

## Example (abbreviated)

User: *“Generate eval spec + dataset for `hotel-agent`.”* You resolve the registered entrypoint, lock I/O keys from AST + low-confidence paths where needed, write preview `eval_spec` / policy, get save approval, write `dataset` to `_preview_dataset.json`, ask replace/append/backup for any existing `dataset.json`, smoke-check, then point to `/overmind-optimize-agent`.

## Operating principles

- **Codebase is the source of truth**: every input field, output key, and tool comes from the registered Overmind entrypoint and the modules it imports. Do not invent fields.
- **Entrypoint contract is fixed**: The registered Overmind entrypoint harness must stay **out of** `optimizable_paths` and **in** `read_only_paths` (the accept step enforces a byte-equality diff). **`/overmind-optimize-agent` compatibility:** If an existing eval spec already includes the harness path under `optimizable_paths` (legacy misconfiguration), optimization may still touch it only when Overmind’s scope and candidate prompts allow; after such a run, **repair** the spec here so the harness returns to `read_only_paths` / `fixed_elements`.
- **Evaluator-compatible types only**: output `type` must be one of `text`, `enum`, `number`, `boolean`. Never `string`, `object`, `array`, `dict`, `list`, `json`.
- **Top-level scoring only**: nested dicts and list outputs are normalized in the entrypoint into top-level fields before reaching the evaluator.
- **Schema agreement is mandatory**: every dataset row's `input` keys must equal the eval spec's `input_schema` keys; every `expected_output` key must appear in `output_fields`.
- **Deterministic weights**: weights sum exactly to `total_points = 100`.
- **Approval before overwrite**: if `setup_spec/` already exists, show a concrete diff summary and ask before replacing.
- **No secret inspection**: never echo, log, or infer API key values. Provider keys are configured by `/overmind-register-agent` or the project environment.
- **Mandatory setup (no silent skips)**: Step 0 must record policy/dataset intent. The coding agent may **compress** questions when the user already answered in-thread — one-line reconfirm, then proceed.
- **Preview files over giant chat pastes**: Prefer writing preview artifacts to disk and summarizing in chat (deterministic paths, IDE-openable). Full paste is optional when the user requests it.
- **No silent dropping**: never silently drop input fields, output fields, sibling packages, seed cases, or existing artifact logic. Preserve, repair, or explicitly report every dropped item.
- **Smoke testing here is non-blocking but owned by this skill**: this skill may run light invocation/schema smoke checks against up to three dataset cases. Do not run full semantic evaluation here. If a smoke check reaches external APIs and fails due to credentials, auth, network, or provider configuration, classify it as an environment issue and keep structurally valid artifacts.
- **Backend sync is mandatory (no silent skips)**: every artifact this skill writes — `eval_spec.json`, `policies.md`, and `dataset.json` — must also be pushed to the Overmind backend via `overmind.storage.get_storage()` (`save_spec` / `save_policy` / `save_dataset`) in the same step that writes the local file. The optimize loop (`/overmind-optimize-agent`) and the UI both read from the backend record, so a local-only artifact is **not** considered "saved". If `OVERMIND_API_KEY` is not configured, stop and tell the user to configure it in `.overmind/.env` before continuing — do not write the local files only and claim success.
- **Scope is derived from local code AND user-approved (no silent defaults)**: `scope.optimizable_paths` and `scope.read_only_paths` are the single highest-leverage decision in this skill — they decide what the optimizer is allowed to touch, and a bad split (zero optimizable files, harness in editable scope, missing sibling packages) silently produces severe regressions in `/overmind-optimize-agent`. **Do not** ship a template scope or copy the example block verbatim. Derive both lists from **your own local reading of the codebase** in Step 1 — specifically the entrypoint's import graph, the registered agent package, sibling packages, and any runtime fixture/data files the bundle needs. Then **print the proposed split to the user with file counts and sample paths**, and **block on `AskQuestion`** until the user explicitly approves or edits it. See **Step 4b — Confirm scope from local code (mandatory)** below.
- **No transient errors in ground truth (env-health gate + quarantine)**: if the skill captures `expected_output` by running the registered entrypoint live on each synthetic input (the only way to get faithful targets), it MUST first verify the environment is healthy and MUST reject any captured output whose error matches a known transient pattern (`AuthenticationError`, `RateLimitError`, `APIConnectionError`, `APITimeoutError`, empty/garbage `Unknown model: ''`). The previous run of this skill poisoned 7+ rows by silently capturing `AuthenticationError: api_key not set` as the "expected" outcome for valid happy-path inputs, capping the baseline at 83.5 and actively misleading the optimizer toward edge-case short-circuit "fixes". See **Step 8b — Live-capture safety (env-health + transient-error quarantine)** for the required gate.

## Workflow

### Mandatory elicitation (never skip — run first)

Before Step 3 (policy generation), ask **in order** (use `AskQuestion` when available):

1. **Pre-existing policy**: Do you already have a policy document (markdown or text)? Options: *Yes* / *No*. If *Yes*, ask for the **project-relative path**, read it, and carry it into Step 3 as the starting policy text (merge/improve against code as today).
1. **Pre-existing dataset or seed file**: Do you already have a dataset, seed JSON/JSONL, or examples file to inform generation? Options: *Yes* / *No*. If *Yes*, ask for the **project-relative path** and use it when generating the dataset (Step 7–8) after the eval spec exists.

Do not infer “no” from silence or empty directories — ask explicitly.

```
Spec + Dataset Progress:
- [ ] Step 0: Mandatory elicitation (policy path? dataset path?)
- [ ] Step 1: Resolve agent + read entrypoint
- [ ] Step 2: Confirm canonical input / output keys
- [ ] Step 3: Generate policy from code, existing doc, or targeted elicitation
- [ ] Step 4: Build eval_spec deterministically (in memory)
- [ ] Step 4b: Print local-code-derived scope split; block on user approval
- [ ] Step 5: Write preview files + summary; optional full paste; save vs edit AskQuestion
- [ ] Step 6: Save policy.md + eval_spec.json **and push spec + policy to the Overmind backend**
- [ ] Step 7: Decide on seed data (ask before generating)
- [ ] Step 8: Generate dataset to **preview** file; enforce schema agreement
- [ ] Step 8b: If live-capturing `expected_output`, run env-health gate + transient-error quarantine
- [ ] Step 9: Promote preview → `dataset.json` (replace / append / backup) **and push dataset to the Overmind backend (`make_active=True`)**
- [ ] Step 10: Smoke check, summarize, confirm backend sync, and recommend optimization
```

### Step 1 — Resolve the agent

Read `.overmind/agents.toml`, find the entry, derive `(file_path, fn_name)` from the entrypoint string. Read the entrypoint file.

If the agent is not registered, stop and recommend `/overmind-register-agent`.

### Step 2 — Determine canonical input + output keys

**Input keys** — start from the entrypoint signature. Use `ast.parse` and walk the `FunctionDef` (or `AsyncFunctionDef`) for *exact* parameter names, type annotations, and defaults. Exclude `self`, `*args`, `**kwargs`. Do not delegate this to an LLM — the analyzer routinely collapses dict-typed parameters into a single opaque field.

If the signature exposes a single dict-like payload, pydantic model, dataclass, typed dict, or other structured input object, decompose it only when there is concrete evidence from type definitions, seed data, fixtures, tests, examples, serializers, or user confirmation. Otherwise keep the real entrypoint parameter and mark the schema as low-confidence rather than inventing fields.

**Output keys** — union the keys across every `return {...}` literal in the function body. Mark a key `optional: true` if it appears in some but not all returns. For `-> str` return types or non-dict returns, set the single output to `result` of type `text`. If outputs are built via `model_dump()`, serializers, helper functions, or variables (no literal dict in the entrypoint body), treat output-key inference as **low confidence**: inspect those code paths, serializers, and tests, or ask the user to confirm keys — do not invent fields.

**Tools** — scan for `@tool`, `Tool(`, `FunctionTool(`, `tools=[...]`, OpenAI/Anthropic tool dicts in the entrypoint and modules it imports. Record name, description, parameter schema.

Also collect:

- Module docstring or `AGENT_DESCRIPTION` constant → `agent_description`.
- Sibling local packages (top-level imports that resolve to directories next to the entrypoint inside the project root).

Confirm the analysis to the user in a compact table before continuing:

```
Agent:        examples/lead_qualifier/agent.py
Entrypoint:   run(query, company_name)
Inputs:       query (str), company_name (str)
Outputs:      qualification (enum), score (number), reasoning (text), is_enterprise (boolean)
Tools:        search_company, lookup_revenue
Sibling pkgs: prompts, tools
```

For **sibling packages**, prefer **one** structured question listing every package with options *Optimizable / Context only / Exclude* (table or multi-part `AskQuestion`) instead of one question per package when there are several; only split per-package if the user asks to refine one row.

### Step 3 — Generate the policy

Default to code-derived policy generation. Build the policy from:

- Agent purpose from docs, prompts, names, and invocation paths.
- Domain rules from branch logic, validators, tests, tool descriptions, prompts, and constants.
- Hard constraints and unacceptable outcomes from tests, error handling, safety checks, and prompt instructions.
- Edge cases from tests, fixtures, examples, and defensive code.
- Terminology, thresholds, and categories from schemas, enums, constants, and docs.
- Required tool ordering from orchestration logic and tool dependencies.
- Output style and quality expectations from prompts, response serializers, examples, and tests.

Use **only** Step 0 for “do you have an existing policy / dataset path?”. If Step 0 recorded a policy path, read that file and merge into Step 3. If Step 0 said no policy file, proceed with code-derived policy only — **do not** ask again for the same path question. Same rule for dataset paths: Step 0 owns that answer for the whole run.

If the codebase lacks enough signal for material domain rules, mark those sections as low-confidence instead of inventing rules. Use interactive elicitation only for blockers or low-confidence areas that materially affect scoring:

1. *Purpose*: one sentence describing the agent's job.
1. *Domain rules*: real-world business rules the agent must follow.
1. *Hard constraints*: outcomes that are unacceptable even if the agent technically succeeds.
1. *Edge cases*: tricky inputs and their correct handling.
1. *Terminology*: key terms, categories, or thresholds.
1. *Tool ordering*: required orderings between tools.
1. *Quality expectations*: style or format requirements for free-text fields.

Call `overmind.setup.policy_generator.generate_policy_from_code` if available. Otherwise synthesize the policy directly from the inspected codebase context.

### Step 4 — Build the eval spec deterministically

Construct the spec dict directly. **Do not** trust an LLM to allocate weights.

```python
spec = {
    "agent_description": <description>,
    "agent_path": <abs path>,
    "entrypoint_fn": <fn_name>,
    "input_schema": {
        param: {"type": <inferred>, "description": "..."}
        for param in canonical_input_keys
    },
    "output_fields": {
        field: {
            "type":       <"text"|"enum"|"number"|"boolean">,  # never "string"
            "description":"...",
            "values":     [...],          # enum only, non-empty
            "range":      [lo, hi],       # number only
            "optional":   <bool>,
            "weight":     <int>,
            "importance": <"critical"|"important"|"minor">,
            "eval_mode":  "similarity",   # text only — "similarity" for important, "non_empty" for minor
        }
        for field in canonical_output_keys
    },
    "structure_weight": 20,
    "total_points":     100,
    "tool_config":      {"expected_tools": [...], "dependencies": [...], "param_constraints": {...}},
    "tool_usage_weight":10,                # only if tools exist
    "llm_judge_weight": 10,                # only if any text field is critical/important OR a policy exists
    "consistency_rules":[  # [] is fine; each entry must be a dict, never a prose string
        # {"field_a": "<output_fields key>", "field_b": "<output_fields key>",
        #  "type": "correlation"|"ordering",     # optional
        #  "operator": "<="|"<"|">="|">",         # optional, ordering only
        #  "penalty": <number>}                   # optional
    ],
    "scope":            {"optimizable_paths": [...], "read_only_paths": [...]},
    "optimizable_elements": [...],
    "fixed_elements": [...],
    "policy":           <structured policy dict from Step 3>,
}
```

**Weight allocation** (must sum exactly to `total_points = 100`):

```
remaining = 100 - structure_weight - tool_usage_weight - llm_judge_weight   # e.g. 60
mult = {"critical": 3, "important": 2, "minor": 1}
raw  = {f: mult[importance[f]] for f in output_fields}
total_raw = sum(raw.values())
for f in output_fields:
    weight[f] = round(raw[f] / total_raw * remaining)
weight[first] += remaining - sum(weight.values())
```

**Scope construction** — two lists only. Every entry must be a **`Path.glob`-compatible pattern** (literal file path OR glob), **never a bare directory name**. `BundleFactory.from_entry_point` calls `Path(root).glob(pattern)` and keeps only entries where `is_file()` is true; a bare directory like `"langextract"` matches the directory entry itself (not a file) and silently expands to **zero files**, leaving `optimizable_files` empty. The optimizer then has nothing it can edit and degenerates into editing the harness through indirect prompting — see the `Path-expansion preflight` gate below.

```
optimizable_paths = [
    "<package>/**/*.py",          # the whole agent package — preferred default
    # OR narrower:
    # "<package>/prompts/**/*.py",
    # "<package>/agent.py",
]
read_only_paths   = [
    <entrypoint_rel_path>,                # registered harness, never editable
    # Plus: fixture data, runtime adapters, README, docs, pyproject.toml,
    # eval templates, JSON schemas — anything the bundle needs at runtime
    # but candidates must not edit. The accept step enforces a byte-equality
    # diff against these.
]
```

**Never** write `optimizable_paths: ["my_package"]`. Write `optimizable_paths: ["my_package/**/*.py"]` instead. The same rule applies to `read_only_paths` — `"README.md"` (a real file) is fine, `"docs"` (a directory) is not.

Project-level drops (test directories, build artefacts, vendored code, infra) do NOT go in the spec. The hard-coded skip list already handles `.git`, `.venv`, `__pycache__`, `node_modules`, `.pytest_cache`, `dist`, `build`, etc. For project-specific drops, add a `.overmindignore` file at the project root (gitignore-style globs). NEVER drop a sibling package the entrypoint imports — it would silently break candidate worktrees with `ModuleNotFoundError`.

For each sibling package, append to the right scope list per the user's answer in Step 2.

If the user wants to score only a subset of outputs, keep unscored outputs visible as diagnostic or skipped fields instead of silently removing them. Do not assign scoring weights to fields the evaluator cannot score.

**Validation gates** — assert before showing the user:

- **Path-expansion preflight (mandatory)**: every entry in `scope.optimizable_paths` and `scope.read_only_paths` must expand (`Path(root).glob(pattern)`, keeping only `is_file()==True` matches) to **at least one real file**. Overmind's canonical validator only checks shape (list-of-strings), so a directory literal like `"my_package"` passes shape but resolves to zero editable files at runtime, leaving the optimizer with nothing to do and triggering harness-poisoning regressions. Run this gate **before** the canonical preflight so the user gets a fixable error ("did you mean `my_package/**/*.py`?") instead of a silent zero-optimizable bundle later.

  ```python
  from pathlib import Path
  root = Path(spec["agent_path"]).parent  # or project root
  while root.parent != root and not (root / ".overmind").is_dir():
      root = root.parent  # walk up to the directory containing .overmind/
  def _expansion(patterns):
      zero = []
      for pat in patterns or []:
          if not any(p.is_file() for p in root.glob(pat)) and not (root / pat).is_file():
              zero.append(pat)
      return zero
  for key in ("optimizable_paths", "read_only_paths"):
      missing = _expansion(spec.get("scope", {}).get(key, []))
      if missing:
          raise SystemExit(
              f"scope.{key} expands to 0 files for: {missing}. "
              f"Each entry must be a real file path or a Path.glob pattern that "
              f"matches at least one file. Bare directory names (e.g. 'my_package') "
              f"do NOT walk the tree — use 'my_package/**/*.py' instead."
          )
  ```

- **Canonical spec-shape preflight (mandatory, runs after path-expansion)**: import the same validator the `overmind optimize` / `overmind optimize-step` CLI runs and call it against the in-memory spec dict. Treat `SpecValidationError` as a hard stop — the message carries a JSON path (`consistency_rules[0]`, `output_fields.<name>.weight`, `scope.optimizable_paths`, etc.) — show it verbatim, fix the offending field, then re-run this gate. Do not write the Step 5 preview file until this passes.

  ```python
  from overmind.optimize.config import validate_eval_spec, SpecValidationError
  try:
      validate_eval_spec(spec)
  except SpecValidationError as exc:
      raise SystemExit(f"eval_spec preflight failed: {exc}")
  ```

  Notes for common shapes the validator pins (so the authoring step doesn't trip them):
  - `consistency_rules` must be `[{field_a, field_b, type?, operator?, penalty?}]`. **Never a list of prose strings.** Natural-language assertions belong in `policies.md`. Empty list `[]` is the correct default when no output-vs-output covariation is meaningful.
  - `scope.*_paths` must be `list[str]`.
  - `output_fields[name].weight` must be numeric; `values` must be a list when present.

- Every `input_schema` key is a real entrypoint parameter, a decomposed typed-dict/model/dataclass field, or a user-confirmed input.
- Every `output_fields` key appears in at least one normalized entrypoint `return` statement or was user-confirmed.
- No `output_fields.type` is `string` / `object` / `array` / `list` / `dict` / `json`.
- No scored output field depends on nested paths or list item indexing.
- No scored output field has zero weight unless it is intentionally marked skipped or diagnostic.
- Every enum field has non-empty `values`; every number field has `range = [lo, hi]`.
- The weight sum check passes (exactly 100).
- `policy["domain_rules"]` is non-empty, or low-confidence areas are explicitly marked because the codebase lacks domain signal.
- Every detected sibling local package appears in exactly one scope category.
- The registered entrypoint is **absent** from `optimizable_paths` and **present** in `read_only_paths` (and ideally `fixed_elements` for documentation).

### Step 4b — Confirm scope from local code (mandatory, no silent defaults)

**Why this gate exists.** Scope decides what the optimizer is *allowed* to edit. Get it wrong and one of two failure modes ships:

1. **Zero optimizable files** — patterns expand to nothing (e.g. `"my_package"` without `/**/*.py`), the analyzer has no editable surface, and the optimizer starts attacking the read-only harness via indirect prompting. Baselines drop by tens of points.
2. **Harness in editable scope** — the registered entrypoint ends up in `optimizable_paths`, candidates start mutating the I/O contract, every iteration fails the accept-step byte-equality diff, and the run wastes provider spend.

Neither failure is caught by Overmind's CLI validator (shape-only) or by the path-expansion preflight alone (a harness-in-editable spec passes both gates). Only the user can confirm the *intent* of the split. So:

**What you (the coding agent) must do:**

1. **Derive both lists from local code context** built in Step 1 — never from a template. Specifically:
   - `optimizable_paths`: every Python module in the registered agent package(s) that the entrypoint's import graph reaches, expressed as glob patterns (`"<pkg>/**/*.py"`, or narrower if the user wants surgical edits like `"<pkg>/prompting.py"`).
   - `read_only_paths`: the registered entrypoint file (always), plus every runtime-needed but not-editable file you found while reading the entrypoint — fixture data, runtime adapters, README, `pyproject.toml`, JSON schemas, eval templates, prompt fixtures.
   - Sibling packages: assigned per the user's Step 2 answer (do not re-ask here).
2. **Expand both lists locally** so the printout shows real file counts (not just patterns):

   ```python
   from pathlib import Path
   root = Path(spec["agent_path"])
   while root.parent != root and not (root / ".overmind").is_dir():
       root = root.parent

   def _expand(patterns):
       seen = []
       for pat in patterns:
           hits = sorted({str(p.relative_to(root)) for p in root.glob(pat) if p.is_file()})
           if not hits and (root / pat).is_file():
               hits = [pat]
           seen.append((pat, hits))
       return seen

   opt = _expand(spec["scope"]["optimizable_paths"])
   ro  = _expand(spec["scope"]["read_only_paths"])
   ```
3. **Print the split in chat** with this exact shape so the user can scan it without opening files. Sample at most 5 files per pattern; collapse the rest with `(+N more)`.

   ```text
   Proposed optimization scope (derived from local code context — please review)

   optimizable_paths  →  N files the optimizer MAY edit
     • "<pkg>/**/*.py"           [42 files]  e.g. <pkg>/agent.py, <pkg>/prompts.py, <pkg>/resolver.py (+39 more)
     • "<other-pkg>/agent.py"    [1 file]

   read_only_paths    →  M files the optimizer MUST NOT edit (byte-equality enforced)
     • scripts/<entrypoint>.py   [1 file]   ← registered harness, never editable
     • README.md                 [1 file]
     • pyproject.toml            [1 file]
     • <pkg>/fixtures/*.json     [3 files]  e.g. <pkg>/fixtures/seed.json, ... (+1 more)

   Bundle estimate: <total_chars> chars across <total_files> files
     (Overmind truncates analyzer prompts at ~60k chars; if the bundle is much
      larger, the analyzer will only see the first ~60k of file content.)
   ```

   If `total_chars` exceeds ~120k, append a one-line recommendation: "The bundle is `<X>×` over Overmind's analyzer budget — consider narrowing `optimizable_paths` to the prompt/policy files the LLM-judge actually cares about so the analyzer gets full context."

4. **Block on `AskQuestion`** with these exact options — never auto-continue:

   - **Approve scope as shown** → proceed to Step 5.
   - **Narrow `optimizable_paths`** → ask "Which subset?" (free-text or numbered list of current patterns), rebuild the printout, re-ask.
   - **Add to `read_only_paths`** → ask "Which file(s) or pattern(s)?", rebuild, re-ask.
   - **Move a file between lists** → ask "Which file, and which direction?", rebuild, re-ask.
   - **Edit and explain (free-text)** → take the user's edit, rebuild, re-ask.

5. **Re-run both preflights after every edit** (path-expansion + canonical) before re-printing. Never show the user a scope that fails either gate.

6. **Do not write the preview files in Step 5 until this gate has been approved.** If the user abandons the chat at this point, the spec is not saved — that is the intended fail-safe.

**In-thread shortcut:** If the user's original invocation already named a specific scope ("only optimize the prompt files", "make everything in `src/` editable except the entrypoint"), echo the choice once ("Using the scope you specified — optimizable: …, read-only: …"), still **print the full split with file counts**, and still ask the `AskQuestion` so they can confirm the resolution matches their intent. The print step is non-negotiable; only the *re-derivation* is optional.

### Step 5 — Preview artifacts, summarize, approve (no save until confirmed)

After building `policies.md` content and `eval_spec` dict in memory (Steps 3–4):

1. **Write preview files** (coding agent, deterministic paths under the agent’s `setup_spec/`):

   - `.overmind/agents/<agent-name>/setup_spec/_preview_policies.md`
   - `.overmind/agents/<agent-name>/setup_spec/_preview_eval_spec.json`
     Use `json.dumps(spec, indent=2)` for the JSON file. These files are **not** the canonical artifacts until Step 6 copies or replaces them.

1. **In chat, post a compact summary** (always): one-line purpose, list of `input_schema` keys, list of `output_fields` keys with weights summing to 100, optimizable vs excluded scope highlights, and **absolute paths** to both preview files so the user can open them in the editor.

1. **Optional full content**: Only if the user asks for in-chat review, paste full markdown / JSON (may split across messages). Default is **preview files + summary** to avoid token limits and log leakage.

1. **`AskQuestion`**: **Save and continue** | **Edit policy** | **Edit eval spec** | **Edit both**. On edits, revise in memory, **overwrite the two preview files**, refresh the summary, ask again. **Do not** write `policies.md` or `eval_spec.json` until the user picks **Save and continue**. If the user picks **Edit eval spec** and the edit touches `scope.optimizable_paths` or `scope.read_only_paths`, **re-run Step 4b** (re-derive from local code, re-print the split, re-confirm) before returning to this gate — never silently mutate scope at the all-up approval stage.

### Step 6 — Save policy and spec (local + backend)

Write canonical `policies.md` and `eval_spec.json`, **then immediately push them to the Overmind backend** via `overmind.storage.get_storage()`, then delete the preview files (unless the user asked to keep them).

`save_spec` upserts the `Agent` record (creating it if needed and capturing the assigned UUID into `storage.agent_id` — persist this for Step 9 and for `/overmind-optimize-agent`). `save_policy` patches `policy_markdown` and `policy_data` on the same agent. Both calls are mandatory; treat an exception as a hard failure (report it to the user, do not pretend the artifacts are saved).

```python
import json
from pathlib import Path

import overmind
from overmind.core.paths import load_overmind_dotenv
from overmind.optimize.config import validate_eval_spec, SpecValidationError
from overmind.storage import configure_storage, get_storage, StorageNotConfiguredError

load_overmind_dotenv()
overmind.init()

# Belt + braces: re-run the canonical validator in case `spec` mutated
# between Step 4 approval and this save. Same call the CLI subprocess runs.
try:
    validate_eval_spec(spec)
except SpecValidationError as exc:
    raise SystemExit(f"eval_spec preflight failed before save: {exc}")

base = Path(".overmind/agents") / agent_name / "setup_spec"
base.mkdir(parents=True, exist_ok=True)
(base / "policies.md").write_text(policy_md.rstrip() + "\n")
(base / "eval_spec.json").write_text(json.dumps(spec, indent=2))

configure_storage(agent_path=spec["agent_path"], agent_name=agent_name)
try:
    storage = get_storage()
except StorageNotConfiguredError as exc:
    raise SystemExit(
        f"Overmind backend not configured ({exc}). Set OVERMIND_API_KEY "
        "in .overmind/.env, then re-run this step. Local files are "
        "written but not synced."
    )

storage.save_spec(spec)
storage.save_policy(policy_md, spec.get("policy"))
agent_id = storage.get_agent_id()
print(f"Backend sync ok — agent_id={agent_id}")

for name in ("_preview_policies.md", "_preview_eval_spec.json"):
    p = base / name
    if p.is_file():
        p.unlink()
```

Record the returned `agent_id` (e.g. write it into `.overmind/agents/<agent-name>/.overmind_agent_id` or pass it through your in-memory state) so Step 9 can reuse the same backend record for the dataset upload.

### Step 7 — Decide on seed data

If **Step 0** already collected a seed/dataset path, use it here and skip re-asking for the path (still confirm case counts and personas below).

Otherwise `AskQuestion`:

> "Do you have existing example inputs/outputs (real production traces, golden examples)?"
> Options: *Yes — I'll provide a path* | *No — synthetic generation only*

If *No*, warn that synthetic-only datasets miss real distribution and adversarial edge cases. Require the user to explicitly confirm before continuing without seed data.

If *Yes*, ask for the path. Read and validate against the canonical input/output keys before merging.

Also ask:

- *Number of cases* (default 20)
- *Number of personas* (default 5) — diverse + adversarial intents

Explain that red-teamers/personas are generation perspectives, such as novice user, power user, edge-case tester, adversarial attacker, or domain expert. More personas usually means broader and harder coverage.

Preserve seed cases unless they are malformed. If a seed case is malformed but the intended mapping is clear from codebase context, repair it and record the repair. If it cannot be safely repaired, exclude it and report why.

### Step 8 — Generate the dataset

Before generation, create a compact coverage plan:

- Detected input fields.
- Detected normalized expected-output fields.
- The separate entrypoint file the dataset targets.
- Number of cases and personas.
- Persona mix.
- Edge cases to include.
- Seed coverage gaps.

Use this model-selection priority (**never** silently default to `openai/gpt-4o` or any vendor model in code):

1. `SYNTHETIC_DATAGEN_MODEL` in the process environment or `.overmind/.env` — value must be a **non-empty** LiteLLM model id after trim.
1. If unset, stop dataset generation and obtain a model id from the user (`AskQuestion` or chat), then export it for the runner process or append `SYNTHETIC_DATAGEN_MODEL=<id>` to `.overmind/.env` without clobbering unrelated keys.
1. Do **not** infer a model from “which API key exists” alone; that hides misconfiguration.

The runner below must call `generate_diverse_synthetic_data` only with a model string that came from step 1 or 2. If still unset, `raise SystemExit("SYNTHETIC_DATAGEN_MODEL is not set — set it or pass a user-chosen LiteLLM model id before running.")`.

Write `_datagen_runner.py` in the **project root**:

```python
import json, os
from pathlib import Path
from rich.console import Console

import overmind
from overmind.core.paths import load_overmind_dotenv

load_overmind_dotenv()
overmind.init()

from overmind.optimize.data import generate_diverse_synthetic_data

console      = Console()
AGENT_NAME   = "<name>"
NUM_SAMPLES  = <N>
NUM_PERSONAS = <R>
MODEL        = os.getenv("SYNTHETIC_DATAGEN_MODEL", "").strip()
if not MODEL:
    raise SystemExit(
        "SYNTHETIC_DATAGEN_MODEL is not set — set it in .overmind/.env, export it for this process, or pass a user-chosen LiteLLM model id before running."
    )

CANONICAL_INPUT_KEYS  = frozenset(<exact param names>)
CANONICAL_OUTPUT_KEYS = frozenset(<exact output keys>)        # or None for plain-text

AGENT_DESCRIPTION = """<from step 2>"""
AGENT_CODE        = """<full source of the entrypoint file>"""
EVAL_SPEC         = <eval_spec dict from step 4>
SEED_CASES        = []                                           # populate from seed file when given

cases = generate_diverse_synthetic_data(
    agent_description=AGENT_DESCRIPTION,
    model=MODEL,
    num_samples=NUM_SAMPLES,
    num_personas=NUM_PERSONAS,
    agent_code=AGENT_CODE,
    eval_spec=EVAL_SPEC,
    existing_cases=SEED_CASES or None,
    console=console,
)


def enforce_schema(rows, ikeys, okeys):
    clean, dropped = [], []
    for i, c in enumerate(rows):
        inp, out = c.get("input"), c.get("expected_output")
        if not isinstance(inp, dict) or set(inp.keys()) != ikeys:
            dropped.append((i, "input keys mismatch"))
            continue
        if okeys is not None and isinstance(out, dict) and set(out.keys()) != okeys:
            dropped.append((i, "output keys mismatch"))
            continue
        clean.append(c)
    return clean, dropped


cases, dropped = enforce_schema(cases, CANONICAL_INPUT_KEYS, CANONICAL_OUTPUT_KEYS)
if len(dropped) > 0.2 * (len(cases) + len(dropped)):
    raise SystemExit(f"More than 20% of cases dropped — regenerate with stricter prompts. dropped={dropped[:5]}")

preview = Path(f".overmind/agents/{AGENT_NAME}/setup_spec/_preview_dataset.json")
preview.parent.mkdir(parents=True, exist_ok=True)
preview.write_text(json.dumps(cases, indent=2))
print(f"Preview: {len(cases)} cases -> {preview}; dropped {len(dropped)}")
```

Run from the project root, then delete the runner file:

```bash
python _datagen_runner.py
rm _datagen_runner.py
```

If `generate_diverse_synthetic_data` import fails, install overmind (`pip install overmind` / `uv add overmind`) and re-run. Direct `litellm` fallback is acceptable only when overmind is genuinely unavailable.

### Step 8b — Live-capture safety (env-health + transient-error quarantine)

`generate_diverse_synthetic_data` returns cases whose `expected_output` is **hallucinated by the synthetic LLM** — it never invokes the registered agent. That's cheap but produces drift between the dataset's `expected_output` and what the real agent actually returns, which hurts both baseline scoring and analyzer diagnosis.

If you want **faithful** targets, the only correct option is to **run the registered entrypoint live on each synthetic input** and use that output as `expected_output`. This is high-leverage but environment-sensitive. If you take this path, the runner MUST implement the two gates below — without them, transient infrastructure failures (missing API key on first call, rate limits, 429 spikes, model_id typos) get silently captured as ground truth and quietly cap your baseline forever.

**Gate 1 — Pre-flight env-health check (mandatory, runs once before any case):**

```python
import os, sys
import openai  # or the appropriate provider SDK

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit(
        "FATAL: OPENAI_API_KEY is not set after load_overmind_dotenv(). "
        "Refusing to live-capture expected_output — every case would silently "
        "return AuthenticationError and poison the dataset."
    )

try:
    openai.OpenAI().models.retrieve(EXPECTED_MODEL)  # one canary call
except openai.AuthenticationError as exc:
    sys.exit(f"FATAL: OPENAI_API_KEY present but rejected: {exc}")
except openai.NotFoundError:
    sys.exit(f"FATAL: '{EXPECTED_MODEL}' not visible to this key.")
print("env health: ok")
```

Repeat for each provider you'll actually call. **Do not** skip this gate "because the key looked set" — `.env` precedence bugs (per-agent `.env` overriding the project `.env` with a `<set-me>` placeholder) are common and silent.

**Gate 2 — Transient-error quarantine (mandatory, runs for every captured case):**

```python
TRANSIENT_ERROR_PREFIXES = (
    "AuthenticationError",   # missing or invalid API key
    "RateLimitError",        # 429 — temporary, not a real test
    "APIConnectionError",    # network blip
    "APITimeoutError",       # slow path, not a behavioural target
    "InternalServerError",   # 5xx — provider, not agent
)
TRANSIENT_VALUE_ERROR_SUBSTRINGS = (
    "Unknown model: ''",     # empty model_id leaked from env-loading bug
)

def is_transient(err: str) -> bool:
    if not err:
        return False
    if any(err.startswith(p) for p in TRANSIENT_ERROR_PREFIXES):
        return True
    return any(s in err for s in TRANSIENT_VALUE_ERROR_SUBSTRINGS)

quarantined = []
for case in synthetic_cases:
    live = run(**case["input"])
    err = (live.get("error") or "")
    if is_transient(err):
        quarantined.append({"input": case["input"], "error": err[:120]})
        continue                       # do NOT add to the dataset
    case["expected_output"] = live
    final.append(case)

if quarantined:
    print(f"[quarantine] dropped {len(quarantined)} cases with transient errors:")
    for q in quarantined[:5]:
        print(f"  - {q['error']!r}")
    if len(quarantined) > 0.1 * len(synthetic_cases):
        raise SystemExit(
            f"More than 10% of cases produced transient errors "
            f"({len(quarantined)}/{len(synthetic_cases)}). "
            f"Fix the environment and re-run — do NOT proceed with a sparse dataset."
        )
```

**Intentional negative tests are still allowed.** If the user wants cases with deliberately bogus `model_id` (e.g. `"openai/this-model-does-not-exist"`), the agent will legitimately return `NotFoundError: ...` or `InferenceConfigError: ...`. Those are **not** in the transient list — they belong in the dataset because the agent's behaviour on bad model_ids IS a tested contract. The quarantine list deliberately covers only **infrastructure** failure modes, not **semantic** ones.

**Symptoms this gate catches that would otherwise cap your baseline:**

- A `.env` precedence bug where a per-agent `.env` shadows the project `OPENAI_API_KEY` with a `<set-me>` placeholder — every live call returns `AuthenticationError`, every case gets the same poisoned `expected_output`, baseline can never exceed (correctly-passing-cases / total) × 100.
- A rate-limit spike during generation — 30% of cases get `RateLimitError` frozen as expected.
- A model_id typo in the runner — every case returns the same `NotFoundError`, baseline scores near zero, analyzer hallucinates "fix" that bypasses the LLM.

If you skipped this gate and shipped a poisoned dataset (the typical symptom is "baseline scores 70-85 instead of 95+, with `Success` and `Error` dimensions dramatically underweight"), the precision-repair recipe lives in this skill's commit history — re-run the entrypoint on every case, replace `expected_output` only where the dataset disagrees with reality, and re-sync. Do **not** rebuild the whole dataset — that throws away clean rows and re-rolls the dice on case mix.

### Step 9 — Promote preview to `dataset.json` (local + backend)

The canonical dataset path is `.overmind/agents/<name>/setup_spec/dataset.json`. Step 8 must **only** write `_preview_dataset.json` until this step decides fate.

- If **`dataset.json` does not exist**: rename or copy `_preview_dataset.json` → `dataset.json` (then remove the preview file).
- If **`dataset.json` already exists**, ask once:

> "A dataset already exists. *Replace* / *Append* / *Save backup, then replace*"

- *Replace*: move aside or delete the old file, then promote the preview to `dataset.json`, then delete `_preview_dataset.json`.
- *Append*: load existing `dataset.json`, merge new cases after the existing ones, re-run the schema enforcement on the combined list, write the result to `dataset.json`, then delete `_preview_dataset.json`.
- *Save backup, then replace*: copy current `dataset.json` to `dataset.backup.json` (or timestamped), then same as *Replace*.

If the user aborts, delete `_preview_dataset.json` only after they confirm they do not need the preview; leave `dataset.json` unchanged.

**Push the final list to the Overmind backend** as soon as `dataset.json` is settled (i.e. on every Replace / Append / fresh promote — *not* on user-aborted runs that leave `dataset.json` unchanged). Use the same `agent_id` captured in Step 6 so the dataset attaches to the right `Agent` record, and set `make_active=True` so the optimize loop picks it up:

```python
import json
from pathlib import Path

import overmind
from overmind.core.paths import load_overmind_dotenv
from overmind.storage import configure_storage, get_storage

load_overmind_dotenv()
overmind.init()

ds_path = Path(".overmind/agents") / agent_name / "setup_spec" / "dataset.json"
datapoints = json.loads(ds_path.read_text())

configure_storage(
    agent_path=spec["agent_path"],
    agent_name=agent_name,
    agent_id=agent_id_from_step_6,  # required so dataset attaches to the same Agent row
)
storage = get_storage()

meta = storage.save_dataset(
    datapoints,
    source="synthetic" if not seed_path_from_step_0 else "mixed",
    generator_model=os.environ.get("SYNTHETIC_DATAGEN_MODEL", ""),
    metadata={
        "num_cases": len(datapoints),
        "num_personas": NUM_PERSONAS,
        "seed_path": seed_path_from_step_0 or None,
    },
    make_active=True,
)
if not meta:
    raise SystemExit(
        "Dataset upload to Overmind backend failed. Check OVERMIND_API_KEY "
        "and re-run this step. Local dataset.json is fine; the backend "
        "record is missing."
    )
print(
    f"Backend sync ok — dataset_id={meta['id']} version={meta['version']} cases={meta['num_datapoints']}"
)
```

A failed `save_dataset` (returns `None`) is a hard failure: the optimize loop will not see the new cases. Stop, report the error, and ask the user to fix the API configuration before continuing.

After the final `dataset.json` is in place, run a light smoke check against up to three cases. Call the entrypoint exactly once per case and store the result before inspecting it. For async entrypoints, run through the host language's async event loop.

A smoke-check pass means the entrypoint returns a non-null result without raising an exception. Do not require semantic correctness during this smoke check; semantic scoring belongs to optimization.

Classify smoke-check failures:

- **Schema failure**: unexpected keyword argument, missing required argument, wrong input shape, serialization mismatch, or output shape incompatible with `output_fields`.
- **Environment failure**: missing API key, auth failure, network failure, model/provider error, or external service outage.
- **Runtime failure**: agent code raises an internal error unrelated to schema or environment.

If failures are schema-related, tell the user which field or shape likely needs repair before optimization. If failures are environment-related, keep the dataset and tell the user which configuration is needed before a full optimization run.

### Step 10 — Summarize

Tell the user:

- Full paths to `policies.md`, `eval_spec.json`, `dataset.json`.
- Field counts, weight totals, policy stats.
- Number of seed cases preserved, generated cases added, and cases repaired or dropped by schema enforcement.
- Smoke-check status and failure classification, if any.
- Scope summary, including confirmation that every sibling package was classified.
- Confirmation that the entrypoint file is excluded from optimization scope.
- **Backend sync status**: the resolved `agent_id`, `dataset_id`, `dataset_version`, and that `policy_markdown` was patched onto the Agent record. If any push failed, surface the failure here — do not bury it.
- **Next step**: run `/overmind-optimize-agent` or `overmind optimize <agent>`.

## Repair mode

When the user points the skill at an agent that already has `setup_spec/`:

1. Read existing `eval_spec.json` and `policies.md`.
1. Run static analysis on the entrypoint (Step 2).
1. Diff against the existing artifacts: collapsed input schema, missing output keys, missing diagnostic fields, weight sum ≠ 100, zero-weight scored fields, empty policy lists, low-confidence policy areas, mismatched enum values vs code, missing number ranges, `string` type instead of `text`, nested/list scored fields, missing sibling package scope, and entrypoint accidentally in `optimizable_paths`.
1. Show the diff side-by-side. `AskQuestion`: *Apply all fixes* / *Pick which to apply* / *Abort*.
1. Re-run validation gates, save, then run the light smoke check when a dataset exists.

The diff must be concrete, showing current and proposed values rather than vague statements.

## Common issues

- **Agent not in registry**: Register the agent first.
- **Overmind imports fail**: Activate the project virtual environment or use the project package manager from the project root.
- **Overmind data generator not importable**: Activate the project virtual environment, install Overmind, or use the direct LLM fallback.
- **`SYNTHETIC_DATAGEN_MODEL` unset**: The datagen runner exits without a default vendor model. Set `SYNTHETIC_DATAGEN_MODEL` in `.overmind/.env` (or export for the process) or obtain a LiteLLM id from the user before running `_datagen_runner.py`.
- **Model auth error**: Ask the user to configure the relevant provider key in the project or agent environment file.
- **Input schema collapsed to one object**: Decompose using typed dicts, pydantic models, dataclasses, seed data, examples, or user-confirmed fields.
- **Output fields missing**: Union dictionary keys across all normalized return branches.
- **Weights sum to 99 or 101**: Apply the rounding residual to a valid scored field.
- **Policy is empty or generic**: Re-read prompts, tests, validators, examples, and docs; ask targeted questions only for genuinely missing domain rules.
- **Sibling package excluded accidentally**: Ask whether it is optimizable, context-only, or excluded, and place it in exactly one scope category.
- **Entrypoint appears in optimizable scope**: Remove it immediately and place it in excluded or fixed scope. Optimize native agent behavior files instead.
- **Output field type is `string`**: Replace it with `text`.
- **`consistency_rules[i]: each rule must be an object with field_a/field_b keys`**: You authored natural-language prose where the schema wants structured cross-field invariants. Move the prose into `policies.md`. Either leave `consistency_rules: []`, or convert pairs of output fields into entries like `{"field_a": "<key>", "field_b": "<key>", "type": "correlation"}` (both keys must already exist in `output_fields`). Re-run the Step 4 canonical spec-shape preflight and re-sync.
- **Native output is a list or nested object**: Repair the separate entrypoint file to normalize outputs into top-level evaluator-compatible fields before generating the eval spec or dataset.
- **Many generated cases dropped**: Tighten the schema prompt, reduce batch size, generate per persona, or add seed data.
- **Smoke check unexpected keyword error**: The dataset input field names do not match the Overmind entrypoint signature; repair the dataset schema or repair the separate entrypoint file.
- **Smoke check API/auth failure**: The artifacts may still be structurally valid; configure credentials before optimization.
- **Backend sync failure (`StorageNotConfiguredError`)**: `OVERMIND_API_KEY` is missing from `.overmind/.env` and the process environment. Set it, then re-run Step 6 / Step 9.
- **`save_dataset` returns `None`**: Backend rejected the upload. Inspect the surfaced error (most often the agent record was never created in Step 6, or the project token does not own the project). Re-run Step 6 first, then retry Step 9.

## What this skill must NOT do

- Never write artifacts outside `.overmind/agents/<name>/setup_spec/`.
- Never invent enum values or output keys not present in the code.
- Never silently drop output fields, sibling packages, or seed cases.
- Never produce a spec where weights don't sum to `total_points`.
- Never use `string` as an output type.
- Never put the registered entrypoint in `optimizable_paths`.
- Never run full semantic evaluation. Only run light smoke checks for invocation/schema compatibility, and classify external API/provider failures as environment issues.
- Never finish the skill with the local files written but the backend record missing. Backend sync (`save_spec` + `save_policy` + `save_dataset`) is part of "saved", not an optional follow-up.
