---
name: overmind-generate-spec-and-dataset
description: "Generate the policy, eval spec, and evaluation dataset for an Overmind agent in one pass. Use when the user wants to author or rebuild eval criteria for an agent, fix a broken eval spec (wrong input_schema, missing output fields, bad weights), produce or augment a dataset, or prepare everything needed before running `overmind preflight` and `overmind optimize`. Combines policy elicitation, spec construction, and dataset generation so the artifacts always agree on the same input/output schema."
metadata:
  version: "1.0"
  product: "Overmind"
---

# Generate the Policy, Eval Spec, and Dataset

Builds the three canonical artifacts that drive optimization, in a single
ordered pass so the input/output schemas always agree:

1. `.overmind/agents/<name>/setup_spec/policies.md` — domain knowledge, constraints, edge cases.
2. `.overmind/agents/<name>/setup_spec/eval_spec.json` — scoring spec (input_schema, output_fields, weights, tools, embedded policy).
3. `.overmind/agents/<name>/setup_spec/dataset.json` — synthetic + seed test cases that conform to the eval spec.

This skill replaces the two earlier ones (`overmind-generate-policy-and-eval` and `overmind-generate-dataset`). Doing both in one pass eliminates the most common failure mode of the old flow: a dataset that was generated against one input/output shape and an eval spec that scores a different shape.

After this skill finishes, run `/overmind-preflight` to validate the wiring before optimization.

## Operating principles

- **Codebase is the source of truth**: every input field, output key, and tool comes from the registered Overmind entrypoint and the modules it imports. Do not invent fields.
- **Entrypoint contract is fixed**: the registered Overmind entrypoint is a fixed harness — never put it in `optimizable_paths`; always include it in `exclude_paths` or `fixed_elements`.
- **Evaluator-compatible types only**: output `type` must be one of `text`, `enum`, `number`, `boolean`. Never `string`, `object`, `array`, `dict`, `list`, `json`.
- **Top-level scoring only**: nested dicts and list outputs are normalized in the entrypoint into top-level fields before reaching the evaluator.
- **Schema agreement is mandatory**: every dataset row's `input` keys must equal the eval spec's `input_schema` keys; every `expected_output` key must appear in `output_fields`.
- **Deterministic weights**: weights sum exactly to `total_points = 100`.
- **Approval before overwrite**: if `setup_spec/` already exists, show a concrete diff summary and ask before replacing.
- **No secret inspection**: never echo or log API key values. Provider keys are configured by `/overmind-register-agent` and `/overmind-preflight`.
- **Smoke testing here is non-blocking**: this skill does light schema validation only. The full pipeline check (does the agent run? do the metrics score? are the weights consistent?) is `overmind preflight`'s job — *that* skill autonomously fixes plumbing issues.

## Workflow

```
Spec + Dataset Progress:
- [ ] Step 1: Resolve agent + read entrypoint
- [ ] Step 2: Confirm canonical input / output keys
- [ ] Step 3: Elicit policy (interactive or from existing doc)
- [ ] Step 4: Build eval_spec deterministically (in memory)
- [ ] Step 5: Show policy + spec, get approval
- [ ] Step 6: Save policy.md + eval_spec.json
- [ ] Step 7: Decide on seed data (ask before generating)
- [ ] Step 8: Generate dataset; enforce schema agreement
- [ ] Step 9: Save dataset.json (ask before overwriting)
- [ ] Step 10: Summarize; recommend `/overmind-preflight`
```

### Step 1 — Resolve the agent

Read `.overmind/agents.toml`, find the entry, derive `(file_path, fn_name)` from the entrypoint string. Read the entrypoint file.

If the agent is not registered, stop and recommend `/overmind-register-agent`.

### Step 2 — Determine canonical input + output keys

**Input keys** — from the entrypoint signature only. Use `ast.parse` and walk the `FunctionDef` (or `AsyncFunctionDef`) for *exact* parameter names, type annotations, and defaults. Exclude `self`, `*args`, `**kwargs`. Do not delegate this to an LLM — the analyzer routinely collapses dict-typed parameters into a single opaque field.

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

### Step 3 — Elicit the policy

`AskQuestion`: *interactive elicitation* (recommended) | *auto-infer from code only* | *I have a policy doc — point me at it*.

For interactive elicitation, ask each as a separate question and skip the ones the code obviously answers (purpose, etc.):

1. *Purpose*: one sentence describing the agent's job.
2. *Domain rules*: real-world business rules the agent must follow.
3. *Hard constraints*: outcomes that are unacceptable even if the agent technically succeeds.
4. *Edge cases*: tricky inputs and their correct handling.
5. *Terminology*: key terms / categories / thresholds.
6. *Tool ordering*: required orderings between tools.
7. *Quality expectations*: style/format requirements for free-text fields.

For *auto-infer*, call `overmind.setup.policy_generator.generate_policy_from_code` if available. For *existing doc*, read it, call `improve_existing_policy`, show the diff, ask which version to keep.

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

**Validation gates** — assert before showing the user:

- Every `input_schema` key is a real entrypoint parameter.
- Every `output_fields` key appears in at least one `return` statement.
- No `output_fields.type` is `string` / `object` / `array` / `list` / `dict` / `json`.
- Every enum field has non-empty `values`; every number field has `range = [lo, hi]`.
- The weight sum check passes (exactly 100).
- `policy["domain_rules"]` is non-empty.
- The registered entrypoint is **absent** from `optimizable_paths` and **present** in `exclude_paths` or `fixed_elements`.

### Step 5 — Show generated content and get approval

Show the actual content (not a skeleton):

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POLICY  (policies.md)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Purpose: <one-sentence purpose>
Domain rules: …
Hard constraints: …

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVAL SPEC  (eval_spec.json)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
input_schema:  <param>(<type>) — <desc>
output_fields: <field> <type> importance=<...> weight=<N>
weights:       fields=<N> + structure=<N> + tools=<N> + judge=<N> = 100
scope:
  optimizable: …
  context:     …
  excluded:    … (incl. entrypoint)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

`AskQuestion`: *Approve and save* / *Change the policy* / *Change the eval spec* / *Change both*.

Iterate until the user explicitly approves. **Do not write any file until approved.**

### Step 6 — Save policy and spec

```python
base = Path(".overmind/agents") / agent_name / "setup_spec"
base.mkdir(parents=True, exist_ok=True)
(base / "policies.md").write_text(policy_md.rstrip() + "\n")
(base / "eval_spec.json").write_text(json.dumps(spec, indent=2))
```

### Step 7 — Decide on seed data

`AskQuestion`:

> "Do you have existing example inputs/outputs (real production traces, golden examples)?"
> Options: *Yes — I'll provide a path* | *No — synthetic generation only*

If *No*, warn that synthetic-only datasets miss real distribution and adversarial edge cases. Require the user to explicitly confirm before continuing without seed data.

If *Yes*, ask for the path. Read and validate against the canonical input/output keys before merging.

Also ask:
- *Number of cases* (default 20)
- *Number of personas* (default 5) — diverse + adversarial intents

### Step 8 — Generate the dataset

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

### Step 10 — Summarize

Tell the user:

- Full paths to `policies.md`, `eval_spec.json`, `dataset.json`.
- Field counts, weight totals, policy stats.
- Number of seed vs generated cases; how many were dropped by schema enforcement.
- Confirmation that the entrypoint file is excluded from optimization scope.
- **Next step**: run `/overmind-preflight` for `<agent>` to validate the wiring and let it autonomously fix any remaining plumbing issues. After that returns green, run `/overmind-optimise-agent`.

## Repair mode

When the user points the skill at an agent that already has `setup_spec/`:

1. Read existing `eval_spec.json` and `policies.md`.
2. Run static analysis on the entrypoint (Step 2).
3. Diff against the existing artifacts: collapsed input schema, missing output keys, weight sum ≠ 100, empty policy lists, mismatched enum values vs code, `string` type instead of `text`, entrypoint accidentally in `optimizable_paths`.
4. Show the diff side-by-side. `AskQuestion`: *Apply all fixes* / *Pick which to apply* / *Abort*.
5. Re-run validation gates, save, then suggest `/overmind-preflight`.

## What this skill must NOT do

- Never write artifacts outside `.overmind/agents/<name>/setup_spec/`.
- Never invent enum values or output keys not present in the code.
- Never silently drop output fields, sibling packages, or seed cases.
- Never produce a spec where weights don't sum to `total_points`.
- Never use `string` as an output type.
- Never put the registered entrypoint in `optimizable_paths`.
- Never run the agent against external APIs (that's `/overmind-preflight`'s job — it has the autonomous repair loop).
