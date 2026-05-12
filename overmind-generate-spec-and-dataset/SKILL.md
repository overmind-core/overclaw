---
name: overmind-generate-spec-and-dataset
description: "Generate the policy, eval spec, and evaluation dataset for an Overmind agent in one pass. Use when the user wants to author or rebuild eval criteria for an agent, fix a broken eval spec (wrong input_schema, missing output fields, bad weights), produce or augment a dataset, or prepare everything needed before running `overmind optimize`. Combines policy elicitation, spec construction, and dataset generation so the artifacts always agree on the same input/output schema."
metadata:
  version: "1.2"
  product: "Overmind"
---

# Generate the Policy, Eval Spec, and Dataset

Builds the three canonical artifacts that drive optimization, in a single
ordered pass so the input/output schemas always agree:

1. `.overmind/agents/<name>/setup_spec/policies.md` — domain knowledge, constraints, edge cases.
2. `.overmind/agents/<name>/setup_spec/eval_spec.json` — scoring spec (input_schema, output_fields, weights, tools, embedded policy).
3. `.overmind/agents/<name>/setup_spec/dataset.json` — synthetic + seed test cases that conform to the eval spec.

This skill replaces the two earlier ones (`overmind-generate-policy-and-eval` and `overmind-generate-dataset`). Doing both in one pass eliminates the most common failure mode of the old flow: a dataset that was generated against one input/output shape and an eval spec that scores a different shape.

After this skill finishes, run `/overmind-optimise-agent` or `overmind optimize <agent>` to start optimization.

## Operating principles

- **Codebase is the source of truth**: every input field, output key, and tool comes from the registered Overmind entrypoint and the modules it imports. Do not invent fields.
- **Entrypoint contract is fixed**: the registered Overmind entrypoint is a fixed harness — never put it in `optimizable_paths`; always include it in `exclude_paths` or `fixed_elements`.
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

## Workflow

### Mandatory elicitation (never skip — run first)

Before Step 3 (policy generation), ask **in order** (use `AskQuestion` when available):

1. **Pre-existing policy**: Do you already have a policy document (markdown or text)? Options: *Yes* / *No*. If *Yes*, ask for the **project-relative path**, read it, and carry it into Step 3 as the starting policy text (merge/improve against code as today).
2. **Pre-existing dataset or seed file**: Do you already have a dataset, seed JSON/JSONL, or examples file to inform generation? Options: *Yes* / *No*. If *Yes*, ask for the **project-relative path** and use it when generating the dataset (Step 7–8) after the eval spec exists.

Do not infer “no” from silence or empty directories — ask explicitly.

```
Spec + Dataset Progress:
- [ ] Step 0: Mandatory elicitation (policy path? dataset path?)
- [ ] Step 1: Resolve agent + read entrypoint
- [ ] Step 2: Confirm canonical input / output keys
- [ ] Step 3: Generate policy from code, existing doc, or targeted elicitation
- [ ] Step 4: Build eval_spec deterministically (in memory)
- [ ] Step 5: Write preview files + summary; optional full paste; save vs edit AskQuestion
- [ ] Step 6: Save policy.md + eval_spec.json
- [ ] Step 7: Decide on seed data (ask before generating)
- [ ] Step 8: Generate dataset; enforce schema agreement
- [ ] Step 9: Save dataset.json (ask before overwriting)
- [ ] Step 10: Smoke check, summarize, and recommend optimization
```

### Step 1 — Resolve the agent

Read `.overmind/agents.toml`, find the entry, derive `(file_path, fn_name)` from the entrypoint string. Read the entrypoint file.

If the agent is not registered, stop and recommend `/overmind-register-agent`.

### Step 2 — Determine canonical input + output keys

**Input keys** — start from the entrypoint signature. Use `ast.parse` and walk the `FunctionDef` (or `AsyncFunctionDef`) for *exact* parameter names, type annotations, and defaults. Exclude `self`, `*args`, `**kwargs`. Do not delegate this to an LLM — the analyzer routinely collapses dict-typed parameters into a single opaque field.

If the signature exposes a single dict-like payload, pydantic model, dataclass, typed dict, or other structured input object, decompose it only when there is concrete evidence from type definitions, seed data, fixtures, tests, examples, serializers, or user confirmation. Otherwise keep the real entrypoint parameter and mark the schema as low-confidence rather than inventing fields.

**Output keys** — union the keys across every `return {...}` literal in the function body. Mark a key `optional: true` if it appears in some but not all returns. For `-> str` return types or non-dict returns, set the single output to `result` of type `text`.

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

For each sibling package, ask via `AskQuestion`: *Optimizable / Context only / Exclude*. Never silently exclude a sibling package.

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
2. *Domain rules*: real-world business rules the agent must follow.
3. *Hard constraints*: outcomes that are unacceptable even if the agent technically succeeds.
4. *Edge cases*: tricky inputs and their correct handling.
5. *Terminology*: key terms, categories, or thresholds.
6. *Tool ordering*: required orderings between tools.
7. *Quality expectations*: style or format requirements for free-text fields.

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
    "consistency_rules":[...],
    "scope":            {"optimizable_paths": [...], "context_paths": [...], "exclude_paths": [...]},
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

**Scope construction**:

```
optimizable_paths = [<native agent files imported by entrypoint>]
context_paths     = []                    # README, docs, pyproject.toml if present
exclude_paths     = [
    <entrypoint_rel_path>,                # entrypoint is ALWAYS excluded
    ".overmind/**", ".venv/**", ".github/**",
    "tests/**", "benchmarks/**", "scripts/**",
    "**/__pycache__/**", "**/*.egg-info/**",
    "uv.lock", "poetry.lock", "Dockerfile",
]
```

For each sibling package, append to the right scope list per the user's answer in Step 2.

If the user wants to score only a subset of outputs, keep unscored outputs visible as diagnostic or skipped fields instead of silently removing them. Do not assign scoring weights to fields the evaluator cannot score.

**Validation gates** — assert before showing the user:

- Every `input_schema` key is a real entrypoint parameter, a decomposed typed-dict/model/dataclass field, or a user-confirmed input.
- Every `output_fields` key appears in at least one normalized entrypoint `return` statement or was user-confirmed.
- No `output_fields.type` is `string` / `object` / `array` / `list` / `dict` / `json`.
- No scored output field depends on nested paths or list item indexing.
- No scored output field has zero weight unless it is intentionally marked skipped or diagnostic.
- Every enum field has non-empty `values`; every number field has `range = [lo, hi]`.
- The weight sum check passes (exactly 100).
- `policy["domain_rules"]` is non-empty, or low-confidence areas are explicitly marked because the codebase lacks domain signal.
- Every detected sibling local package appears in exactly one scope category.
- The registered entrypoint is **absent** from `optimizable_paths` and **present** in `exclude_paths` or `fixed_elements`.

### Step 5 — Preview artifacts, summarize, approve (no save until confirmed)

After building `policies.md` content and `eval_spec` dict in memory (Steps 3–4):

1. **Write preview files** (coding agent, deterministic paths under the agent’s `setup_spec/`):
   - `.overmind/agents/<agent-name>/setup_spec/_preview_policies.md`
   - `.overmind/agents/<agent-name>/setup_spec/_preview_eval_spec.json`  
   Use `json.dumps(spec, indent=2)` for the JSON file. These files are **not** the canonical artifacts until Step 6 copies or replaces them.

2. **In chat, post a compact summary** (always): one-line purpose, list of `input_schema` keys, list of `output_fields` keys with weights summing to 100, optimizable vs excluded scope highlights, and **absolute paths** to both preview files so the user can open them in the editor.

3. **Optional full content**: Only if the user asks for in-chat review, paste full markdown / JSON (may split across messages). Default is **preview files + summary** to avoid token limits and log leakage.

4. **`AskQuestion`**: **Save and continue** | **Edit policy** | **Edit eval spec** | **Edit both**. On edits, revise in memory, **overwrite the two preview files**, refresh the summary, ask again. **Do not** write `policies.md` or `eval_spec.json` until the user picks **Save and continue**.

### Step 6 — Save policy and spec

Write canonical `policies.md` and `eval_spec.json`, then **delete** `_preview_policies.md` and `_preview_eval_spec.json` if they exist (unless the user asked to keep them).

```python
base = Path(".overmind/agents") / agent_name / "setup_spec"
base.mkdir(parents=True, exist_ok=True)
(base / "policies.md").write_text(policy_md.rstrip() + "\n")
(base / "eval_spec.json").write_text(json.dumps(spec, indent=2))
for name in ("_preview_policies.md", "_preview_eval_spec.json"):
    p = base / name
    if p.is_file():
        p.unlink()
```

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

Use this model-selection priority:

1. `SYNTHETIC_DATAGEN_MODEL` from the project environment if configured.
2. A provider implied by available non-secret environment variable names.
3. A user-selected model when no provider is clear.

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
MODEL        = os.getenv("SYNTHETIC_DATAGEN_MODEL", "openai/gpt-4o")

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

out_path = Path(f".overmind/agents/{AGENT_NAME}/setup_spec/dataset.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(cases, indent=2))
print(f"Saved {len(cases)} cases to {out_path}; dropped {len(dropped)}")
```

Run from the project root, then delete the runner file:

```bash
python _datagen_runner.py
rm _datagen_runner.py
```

If `generate_diverse_synthetic_data` import fails, install overmind (`pip install overmind` / `uv add overmind`) and re-run. Direct `litellm` fallback is acceptable only when overmind is genuinely unavailable.

### Step 9 — Save the dataset

If `dataset.json` already exists, ask:

> "A dataset already exists. *Replace* / *Append* / *Save backup, then replace*"

For *Append*, merge the new cases after the existing ones, then re-run the schema enforcement on the combined list.

After saving, run a light smoke check against up to three cases. Call the entrypoint exactly once per case and store the result before inspecting it. For async entrypoints, run through the host language's async event loop.

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
- **Next step**: run `/overmind-optimise-agent` or `overmind optimize <agent>`.

## Repair mode

When the user points the skill at an agent that already has `setup_spec/`:

1. Read existing `eval_spec.json` and `policies.md`.
2. Run static analysis on the entrypoint (Step 2).
3. Diff against the existing artifacts: collapsed input schema, missing output keys, missing diagnostic fields, weight sum ≠ 100, zero-weight scored fields, empty policy lists, low-confidence policy areas, mismatched enum values vs code, missing number ranges, `string` type instead of `text`, nested/list scored fields, missing sibling package scope, and entrypoint accidentally in `optimizable_paths`.
4. Show the diff side-by-side. `AskQuestion`: *Apply all fixes* / *Pick which to apply* / *Abort*.
5. Re-run validation gates, save, then run the light smoke check when a dataset exists.

The diff must be concrete, showing current and proposed values rather than vague statements.

## Common issues

- **Agent not in registry**: Register the agent first.
- **Overmind imports fail**: Activate the project virtual environment or use the project package manager from the project root.
- **Overmind data generator not importable**: Activate the project virtual environment, install Overmind, or use the direct LLM fallback.
- **Model auth error**: Ask the user to configure the relevant provider key in the project or agent environment file.
- **Input schema collapsed to one object**: Decompose using typed dicts, pydantic models, dataclasses, seed data, examples, or user-confirmed fields.
- **Output fields missing**: Union dictionary keys across all normalized return branches.
- **Weights sum to 99 or 101**: Apply the rounding residual to a valid scored field.
- **Policy is empty or generic**: Re-read prompts, tests, validators, examples, and docs; ask targeted questions only for genuinely missing domain rules.
- **Sibling package excluded accidentally**: Ask whether it is optimizable, context-only, or excluded, and place it in exactly one scope category.
- **Entrypoint appears in optimizable scope**: Remove it immediately and place it in excluded or fixed scope. Optimize native agent behavior files instead.
- **Output field type is `string`**: Replace it with `text`.
- **Native output is a list or nested object**: Repair the separate entrypoint file to normalize outputs into top-level evaluator-compatible fields before generating the eval spec or dataset.
- **Many generated cases dropped**: Tighten the schema prompt, reduce batch size, generate per persona, or add seed data.
- **Smoke check unexpected keyword error**: The dataset input field names do not match the Overmind entrypoint signature; repair the dataset schema or repair the separate entrypoint file.
- **Smoke check API/auth failure**: The artifacts may still be structurally valid; configure credentials before optimization.

## What this skill must NOT do

- Never write artifacts outside `.overmind/agents/<name>/setup_spec/`.
- Never invent enum values or output keys not present in the code.
- Never silently drop output fields, sibling packages, or seed cases.
- Never produce a spec where weights don't sum to `total_points`.
- Never use `string` as an output type.
- Never put the registered entrypoint in `optimizable_paths`.
- Never run full semantic evaluation. Only run light smoke checks for invocation/schema compatibility, and classify external API/provider failures as environment issues.
