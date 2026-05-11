---
name: overmind-generate-policy-and-eval
description: "Generate, repair, or improve the policy file and eval spec for an Overmind agent. Use when the user wants to create a policy.md / eval_spec.json, fix a broken eval spec (wrong input_schema, missing output fields, bad weights), or rebuild policies before running overmind optimize. Analyzes the agent entrypoint, decomposes inputs and outputs correctly, asks clarifying questions about domain rules and edge cases, and writes the canonical artifacts under .overmind/agents/<name>/setup_spec/."
metadata:
  version: "2.0"
  product: "Overmind"
---

# Generate Policy and Eval Spec for an Overmind Agent

Builds two canonical artifacts that drive `overmind optimize`:

1. `.overmind/agents/<agent-name>/setup_spec/eval_spec.json` — the scoring spec (input/output schema, weights, tool config, consistency rules, embedded policy).
1. `.overmind/agents/<agent-name>/setup_spec/policies.md` — human-readable domain knowledge and behavior policy.

The skill **always validates** what it produces and **repairs** common breakages from prior `overmind setup` runs (collapsed input schemas, missing output fields, weights that don't sum to 100, empty default policies).

## Operating principles

- **Codebase-derived artifacts**: Generate policy and eval artifacts from the agentic codebase context. The codebase, registered entrypoint, adjacent modules, tests, examples, README files, existing artifacts, and invocation paths are the source of truth.
- **Canonical paths only**: Never write policy or eval artifacts outside `.overmind/agents/<agent-name>/setup_spec/`.
- **Entrypoint contract is ground truth**: Analyze the separate Overmind entrypoint file registered in `.overmind/agents.toml`. The entrypoint signature and normalized top-level return shape define the evaluation contract unless the entrypoint must be repaired.
- **Never optimize the entrypoint**: The registered entrypoint file is an invocation harness, not agent behavior. It must be **absent from `optimizable_paths`** and **present in excluded or fixed scope**. This is a hard rule with no exceptions.
- **Evaluator-compatible types only**: Output field `type` values must be only `text`, `enum`, `number`, or `boolean`. **Never use `string`** (use `text` instead). Never use `object`, `array`, `list`, `dict`, or `json` as scoring field types.
- **Top-level scoring only**: The evaluator scores top-level output fields. Nested dictionaries and list items must be normalized by the entrypoint into top-level fields before they can be scored.
- **No silent dropping**: Never silently drop input fields, output fields, tools, or sibling local packages.
- **Deterministic weights**: The eval spec must sum exactly to `total_points = 100`.
- **Approval before overwrite**: If artifacts already exist, show a concrete diff summary and ask before replacing them.
- **Smoke tests are non-blocking**: A failed smoke test should explain the mismatch but should not automatically rewrite the spec unless the user approves.

## When this skill is needed

- "Generate a policy and eval spec for `<agent-name>`"
- "Fix my eval_spec — input_schema only has `input_data` / weights are wrong"
- "Rebuild policies for `<agent>` before optimize"
- "I have an agent file but no setup_spec yet"
- The user shows an `eval_spec.json` whose `input_schema` is `{"input_data": {"type": "object"}}` or whose `output_fields` is missing keys returned by the agent
- The user shows a `policies.md` that is just the auto-generated stub

## Workflow

Copy this checklist into your response and check off each step as you complete it:

```
Policy & Eval Spec Progress:
- [ ] Step 1: Read and analyze the agent
- [ ] Step 2: Confirm analysis with user
- [ ] Step 3: Elicit the policy
- [ ] Step 4: Generate policy and eval spec (held in memory — do not save yet)
- [ ] Step 5: Show content and get user approval
- [ ] Step 6: Save artifacts to disk
- [ ] Step 7: Smoke test (non-blocking)
- [ ] Step 8: Summarize
```

### Step 1 — Read and analyze the agent

The user provides an **agent name** (the slug used during `overmind agent register`). Everything else is resolved from the registry and the entrypoint file.

Collect in order, asking only what isn't obvious from context:

| # | Field | How to get it |
|---|---|---|
| 1 | **Agent name (slug)** | From the user's request. |
| 2 | **Agent file path** | Look up from `.overmind/agents.toml`: find the entry where `name` matches, take the `entrypoint` (e.g. `examples.myagent.overmind_ep:run`), split on `:`, convert the module part to a path (`examples/myagent/overmind_ep.py`). If not registered, tell the user to run `/overmind-register-agent` first. |
| 3 | **Entrypoint function** | The function name from the entrypoint string (after `:`). |
| 4 | **Mode** | `AskQuestion`: *fresh generation*, *repair an existing spec*, or *improve an existing policy doc*. |
| 5 | **Policy source** | `AskQuestion`: *interactive elicitation* (recommended), *auto-infer from code only*, or *I have a policy doc — point me at it*. |
| 6 | **Existing policy file** | If the user says "yes" to option 3 above, ask for the path. Read that file, verify it's relevant to the registered agent/entrypoint, and treat it as the source for `policies.md`. Preserve useful language; repair inconsistencies against the codebase-derived contract. If it conflicts with the entrypoint or code, show a concise conflict summary and ask whether to adapt, keep as-is, or generate fresh from code. |
| 7 | **Existing artifacts** | Read any current `setup_spec/eval_spec.json` and `setup_spec/policies.md`. Diff against what we'd generate. |

Read the **Overmind entrypoint file** (the separate harness). From it, extract **statically** (do not rely on the LLM analyzer alone — it collapses dict params and misses output keys):

1. **Entrypoint signature** — name, every parameter, default values, type annotations. Use `ast.parse` / `ast.FunctionDef` walk.
1. **Return shape** — collect every `return {...}` literal in the function body. Union the keys across branches; mark a key `optional: true` if it appears in some but not all returns. For non-dict returns, set output type to `text`.
1. **Tool definitions** — look for `@tool`, `Tool(`, `FunctionTool(`, `tools=[...]`, OpenAI/Anthropic tool dicts. Record name, description, parameter schema.
1. **Module docstring + `AGENT_DESCRIPTION` constant** — use as `agent_description`.
1. **Imports** — note any local sibling modules so the spec's `scope.optimizable_paths` covers the right files.
1. **Sibling local packages** — for every top-level `import X` / `from X import ...` in the entrypoint file, check whether `X` resolves to a directory sitting next to the entrypoint inside the project root. Collect them as `sibling_local_packages = [<path>, ...]`.

Also read the native agent files imported by the entrypoint for behavioral context. Use the entrypoint signature for `input_schema` and the entrypoint-normalized top-level output for `output_fields`.

**Verify the entrypoint is live-importable** before building the spec:

```bash
python - <<'PY'
import importlib, inspect
module = importlib.import_module("<module_path>")
fn = getattr(module, "<callable>")
print(inspect.signature(fn))
PY
```

If this fails, stop and recommend repairing the entrypoint with `/overmind-register-agent` before generating eval artifacts.

**Do not** trust the LLM analyzer's `input_schema` if it returns a single field whose name matches the entrypoint's only parameter and whose type is `object`/`dict` — that means the analyzer collapsed a dict-of-fields into an opaque blob. In that case, decompose using:

- The user's seed data (if a `dataset.json` exists, take the keys of `cases[0]["input"]`).
- Direct user prompts (one question per top-level field).

### Step 2 — Confirm the analysis with the user

Print a compact summary table:

```
Agent:        examples/hotel/overmind_ep.py
Entrypoint:   run(query, context, ...)
Inputs:       query (string), context (dict), ...
Outputs:      answer (text), confidence (number), ... (N fields)
Tools:        get_data, get_news, ... (N tools)
```

Then `AskQuestion`:

- *"The detected output has N fields. Score all of them, or only a subset?"* → if subset, ask for the list.
- *"`decision` looks like an enum. Valid values are [Buy, Hold, Sell] — correct?"*
- *"Are there fields I missed?"* (free text)

**Scope confirmation (mandatory whenever `sibling_local_packages` is non-empty):**

For each sibling local package, `AskQuestion` with options:

1. *Optimizable* — package is part of the agent; the optimizer may edit it.
1. *Context only* — optimizer can read but must not edit.
1. *Exclude* — treat as a vendored third-party copy and ignore entirely.

Record the answer on `static_analysis.scope_decisions[<pkg>]`. Never silently exclude a sibling package.

### Step 3 — Elicit the policy

Branch on the user's policy-source choice.

**3a. Interactive elicitation** (preferred — produces the strongest policy)

Ask each of the following as a separate question. Skip questions where the answer is obvious from code, but always ask the domain questions — they aren't in the code:

1. *Purpose*: "In one sentence, what is this agent's job?"
1. *Domain rules*: "What real-world business rules must the agent follow?"
1. *Hard constraints*: "What outcomes are unacceptable, even if the agent technically succeeds?"
1. *Edge cases*: "Tricky inputs and the correct handling for each."
1. *Terminology*: "Key terms, categories, or thresholds the agent needs to know."
1. *Tool ordering*: "Are there required orderings between tools?"
1. *Quality expectations*: "Style/format requirements for free-text output fields."

**3b. Auto-infer from code** — call `overmind.setup.policy_generator.generate_policy_from_code` if available; otherwise use a direct LLM call with the full codebase context bundle.

**3c. Improve existing doc** — read the file, call `overmind.setup.policy_generator.improve_existing_policy` if available, then show the diff and ask the user which version to keep.

### Step 4 — Generate the full policy and eval spec

Using everything gathered in Steps 1–3, produce the complete artifacts now. **Do not save anything to disk yet** — hold both artifacts in memory until the user approves them in Step 5.

Construct the spec dict directly (do **not** trust the LLM to allocate weights — do it deterministically):

```python
spec = {
    "agent_description": static_analysis["description"],
    "agent_path": str(Path(agent_path).resolve()),
    "entrypoint_fn": entrypoint_fn,
    "input_schema": {
        # one entry per *parameter* of the entrypoint, NOT a single "input_data" blob.
        param: {"type": <inferred>, "description": "..."}
        for param in static_analysis["params"]
    },
    "output_fields": {
        # one entry per key in the union of return dicts
        # IMPORTANT: type must be "text", "enum", "number", or "boolean" — NEVER "string"
        field: {
            "type": <"enum"|"number"|"text"|"boolean">,  # never "string"
            "description": "...",
            "values": [...],           # enum only
            "range": [lo, hi],         # number only
            "optional": <bool>,
            "weight": <int>,
            "importance": <"critical"|"important"|"minor">,
            "eval_mode": "similarity",  # text only — "similarity" for important, "non_empty" for minor
        }
        for field in static_analysis["output_keys"]
    },
    "structure_weight": 20,
    "total_points": 100,
    "tool_config": {
        "expected_tools": [...],
        "dependencies": [...],
        "param_constraints": {...},
    },
    "tool_usage_weight": 10,            # only if tools exist
    "llm_judge_weight": 10,             # only if any text field is critical/important OR a policy exists
    "consistency_rules": [...],
    "scope": build_scope(static_analysis),
    "optimizable_elements": [...],
    "fixed_elements": [...],
    "policy": <structured policy dict from Step 3>,
}
```

**Scope construction algorithm:**

```
optimizable_paths = []          # NEVER add the entrypoint file here
context_paths     = []
exclude_paths     = [
    "<entrypoint_rel_path>",    # entrypoint is ALWAYS excluded from optimization
    ".overmind/**", ".venv/**", ".github/**",
    "tests/**", "benchmarks/**", "scripts/**",
    "**/__pycache__/**", "**/*.egg-info/**",
    "uv.lock", "poetry.lock", "Dockerfile"
]

# Native agent files the entrypoint invokes — these ARE optimizable
for native_file in static_analysis["native_agent_files"]:
    optimizable_paths.append(native_file)

for pkg, decision in static_analysis["scope_decisions"].items():
    glob = f"{pkg}/**/*.py"
    if   decision == "optimizable": optimizable_paths.append(glob)
    elif decision == "context":     context_paths.append(glob)
    elif decision == "exclude":     exclude_paths.append(f"{pkg}/**")

for ctx in ("README.md", "docs/**/*.md", "pyproject.toml"):
    if Path(project_root / ctx.split("/")[0]).exists():
        context_paths.append(ctx)
```

**Weight allocation algorithm** (must sum to exactly `total_points = 100`):

```
remaining = 100 - structure_weight - tool_usage_weight - llm_judge_weight   # e.g. 60
mult = {"critical": 3, "important": 2, "minor": 1}
raw  = {f: mult[importance[f]] for f in output_fields}
total_raw = sum(raw.values())
for f in output_fields:
    weight[f] = round(raw[f] / total_raw * remaining)
# fix rounding by adding the residual to the first field
weight[first] += remaining - sum(weight.values())
assert structure_weight + tool_usage_weight + llm_judge_weight + sum(weight.values()) == 100
```

**Validation gates** (assert before showing to user):

- Every key of `input_schema` is a real parameter of the entrypoint.
- Every key of `output_fields` appears in at least one `return` statement.
- No `output_fields` type is `string`, `object`, `array`, `list`, `dict`, or `json` — use `text`, `enum`, `number`, or `boolean` only.
- No scored output field has `weight == 0` unless `importance == "minor"` and the user opted to skip it.
- For every enum field, `values` is non-empty.
- For every number field, `range` has two numeric entries.
- The weight sum check passes (exactly 100).
- `policy["domain_rules"]` is a non-empty list (silent empty policies are the #1 cause of useless optimize runs).
- For every sibling local package, exactly one scope category references it.
- The registered entrypoint file is **absent** from `optimizable_paths` and **present** in `exclude_paths` or `fixed_elements`.

### Step 5 — Show generated content, get user approval, iterate

Show the **actual generated content** — not a skeleton — to the user:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POLICY  (policies.md)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Purpose: <one-sentence purpose>

Domain rules:
  • <rule 1>
  ...

Hard constraints:
  • <constraint 1>
  ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVAL SPEC  (eval_spec.json)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
input_schema:
  <param1>  (<type>)  — <description>
  ...

output_fields:
  <field1>  <type>  importance=<critical|important|minor>  weight=<N>
  ...
  ── weights: fields <N> + structure <N> + tools <N> + llm_judge <N> = 100

scope:
  optimizable: <paths>
  context:     <paths>
  excluded:    <paths incl. entrypoint>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Then `AskQuestion`:

> "Does this look right? Approve to save, or tell me what to change."
> Options:
>
> - Approve and save
> - Change the policy
> - Change the eval spec (fields / weights / enums / scope)
> - Change both

If the user requests changes, apply them, regenerate the affected artifact(s), show the updated content again, and ask for approval again. Repeat until the user explicitly approves.

**Do not write any file to disk until the user selects "Approve and save."**

### Step 6 — Save

Once approved, write both files:

```python
from pathlib import Path
import json

base = Path(".overmind/agents") / agent_name / "setup_spec"
base.mkdir(parents=True, exist_ok=True)
(base / "eval_spec.json").write_text(json.dumps(spec, indent=2))
(base / "policies.md").write_text(policy_md.rstrip() + "\n")
```

**Never** write to a different location — the optimizer only reads from this canonical path.

The repo ships helpers under `overmind.setup.*`. Try them first; fall back to direct LLM calls when imports fail:

```python
try:
    from overmind.setup.agent_analyzer import analyze_agent
    from overmind.setup.policy_generator import (
        elicit_policy,
        generate_policy_from_code,
        improve_existing_policy,
        generate_policy_from_document,
    )
    from overmind.setup.spec_generator import generate_spec_from_proposal, save_spec
    from overmind.setup.policy_generator import save_policy

    HAS_OVERMIND = True
except Exception:
    HAS_OVERMIND = False
```

If `HAS_OVERMIND` is `False`, tell the user to install it first (`pip install overmind`) before proceeding to direct LLM fallback.

When `HAS_OVERMIND` is True, prefer `analyze_agent(...)` for the LLM analysis pass, then **post-process** its output with the static checks from Step 1 before passing to `generate_spec_from_proposal`. Common breakages:

| Symptom | Cause | Fix |
|---|---|---|
| `input_schema` has one entry typed `object` | LLM collapsed a single dict-typed parameter | Replace with the decomposed schema built statically |
| `output_fields` missing keys from `return {...}` | LLM only captured "important-looking" keys | Union all return-dict keys yourself, then re-call |
| All weights `0` or `None` | `proposed_criteria.fields` was empty | Build the spec dict directly using the weight algorithm in Step 4 |
| `policy` block missing rules | LLM response fenced under wrong tag | Re-run with a stricter prompt asking for a JSON block; or parse manually |
| `consistency_rules` empty for an enum/number agent | Auto-generator only fires on naming patterns | Ask the user: "Should `<number_field>` correlate with `<enum_field>`?" and append manually |

When `HAS_OVERMIND` is False, drop a `_policy_eval_runner.py` in the project root that performs Steps 1–4 with `litellm` directly. Run from the **project root** — never `cd` to a parent directory. Delete the runner on success.

### Step 7 — Smoke test (non-blocking)

If `setup_spec/dataset.json` already exists, run the agent against `cases[0]["input"]` once to confirm the new `input_schema` matches the function signature. Use a subprocess so a hung agent can't block the chat:

```python
import subprocess, sys, json, textwrap
from pathlib import Path
from dotenv import load_dotenv

agent_env = Path(".overmind/agents") / agent_name / ".env"
if agent_env.exists():
    load_dotenv(agent_env, override=True)

case = json.loads(Path(base / "dataset.json").read_text())[0]
input_kwargs = (
    case["input"]
    if isinstance(case.get("input"), dict)
    else {"input_data": case["input"]}
)
script = textwrap.dedent(f"""
    import asyncio, inspect, json, sys
    from pathlib import Path
    from dotenv import load_dotenv
    _env = Path(".overmind/agents/{agent_name}/.env")
    if _env.exists():
        load_dotenv(_env, override=True)
    sys.path.insert(0, {repr(str(Path(agent_path).parent))})
    from {Path(agent_path).stem} import {entrypoint_fn} as fn
    call = fn(**{input_kwargs!r})
    result = asyncio.run(call) if inspect.iscoroutine(call) else call
    print(json.dumps({{"ok": True, "out_keys": list(result.keys()) if isinstance(result, dict) else []}}))
""")
res = subprocess.run(
    [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
)
```

On failure, print the error and tell the user which `input_schema` field name is likely wrong. **Do not** rewrite the spec automatically — let the user decide.

Note: call the agent function exactly once and store the result before inspecting it. Do not call the function twice inside the same expression, as agents may call LLMs or external APIs with side effects.

### Step 8 — Summarize

End the session with:

- Full path to `eval_spec.json` and `policies.md`.
- Field counts, weight totals, policy stats.
- Scope summary (optimizable / context / excluded paths).
- Confirmation that the entrypoint file is excluded from optimization scope.
- Smoke-test result.
- **Next command** (conditional):
  - If `.overmind/agents/<agent-name>/setup_spec/dataset.json` exists: `overmind optimize <agent-name>`
  - If the dataset does not exist yet: run `/overmind-generate-dataset` first, then `overmind optimize <agent-name>`

## Repair mode (existing broken artifacts)

When the user points the skill at an agent that already has a `setup_spec/` directory:

1. Read `eval_spec.json` and `policies.md`.
1. Run static analysis on the entrypoint (Step 1 of the main workflow).
1. Diff: list every field that is wrong (collapsed input, missing output keys, weight sum ≠ 100, empty policy lists, mismatched enum values vs code, `string` type instead of `text`, entrypoint in optimizable scope).
1. Show the diff to the user — **concrete, side-by-side** current vs proposed values, not vague "this looks wrong". `AskQuestion`: *"Apply all fixes / pick which to apply / abort"*.
1. Apply selected fixes, re-run all validation gates, re-save.

## Common issues

| Problem | Fix |
|---|---|
| `input_schema` collapsed to one `object` field | Decompose using seed data, typed dicts, or user-confirmed fields |
| `output_fields` missing keys | Union dictionary keys across all return branches |
| Weights sum to 99 or 101 | Apply the rounding residual to a valid scored field |
| Policy is empty | Re-run interactive elicitation (Step 3a) |
| Output field type is `string` | Replace with `text` — the evaluator does not score `string` |
| Entrypoint appears in `optimizable_paths` | Remove immediately; place in `exclude_paths` or `fixed_elements` |
| Sibling package excluded accidentally | Ask the user: optimizable, context-only, or excluded? |
| Native output is a list or nested object | Repair the entrypoint file to normalize outputs into top-level evaluator-compatible fields |
| Smoke test unexpected keyword error | Rename the input schema field, repair the dataset, or repair the entrypoint signature |
| Overmind setup imports fail | Activate the project virtual environment or use the project's package manager from the project root |

## What the skill must NOT do

- Never write the eval_spec or policy outside `.overmind/agents/<name>/setup_spec/`.
- Never invent enum values that aren't in the code or confirmed by the user.
- Never silently drop output fields.
- Never produce a spec where the weights don't sum to `total_points`.
- Never ship a policy with all domain-rule lists empty without a visible warning.
- Never use `string` as an output field type — always use `text`.
- Never place the registered entrypoint file in `optimizable_paths`.
- Never silently drop a sibling local package from scope.
