______________________________________________________________________

## name: overmind-generate-policy-and-eval description: Interactively generate (or repair) the policy file and eval spec for an Overmind/Overclaw agent. Use when the user wants to create a policy.md / eval_spec.json, fix a broken eval spec (wrong input_schema, missing output fields, bad weights), or rebuild policies before running `overmind optimize`. The skill analyzes the agent entrypoint, decomposes inputs and outputs correctly, asks the user clarifying questions about domain rules and edge cases, and writes the canonical artifacts under `.overmind/agents/<name>/setup_spec/`. disable-model-invocation: true

# Generate Policy and Eval Spec for an Overmind Agent

Builds two canonical artifacts that drive `overmind optimize`:

1. `.overmind/agents/<agent-name>/setup_spec/eval_spec.json` — the scoring spec (input/output schema, weights, tool config, consistency rules, embedded policy).
1. `.overmind/agents/<agent-name>/setup_spec/policies.md` — human-readable domain knowledge and behavior policy.

The skill **always validates** what it produces and **repairs** common breakages from prior `overmind setup` runs (collapsed input schemas, missing output fields, weights that don't sum to 100, empty default policies).

## When this skill is needed

Trigger on any of:

- "Generate a policy and eval spec for `<agent-name>`"
- "Fix my eval_spec — input_schema only has `input_data` / weights are wrong"
- "Rebuild policies for `<agent>` before optimize"
- "I have an agent file but no setup_spec yet"
- The user shows an `eval_spec.json` whose `input_schema` is `{"input_data": {"type": "object"}}` or whose `output_fields` is missing keys returned by the agent
- The user shows a `policies.md` that is just the auto-generated stub

## Inputs the skill needs

The user provides an **agent name** (the slug used during `overmind agent register`, e.g. `my-agent`). Everything else is resolved from the registry and the agent file.

Collect, in order, asking only what isn't obvious from context. Use the `AskQuestion` tool for multiple-choice prompts; use plain conversation for free-form answers.

| #   | Field                   | How to get it                                                                                                                                                                                                                                                                                  |
| --- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Agent name (slug)**   | From the user's request.                                                                                                                                                                                                                                                                       |
| 2   | **Agent file path**     | Look up from `.overmind/agents.toml`: find the entry where `name` matches, take the `entrypoint` (e.g. `examples.myagent.agent:run`), split on `:`, convert the module part to a path (`examples/myagent/agent.py`). If not registered, tell the user to run `/overmind-register-agent` first. |
| 3   | **Entrypoint function** | The function name from the entrypoint string (after `:`).                                                                                                                                                                                                                                      |
| 4   | **Mode**                | `AskQuestion`: *fresh generation*, *repair an existing spec*, or *improve an existing policy doc*.                                                                                                                                                                                             |
| 5   | **Policy source**       | `AskQuestion`: *interactive elicitation* (recommended), *auto-infer from code only*, or *I have a markdown doc — point me at it*.                                                                                                                                                              |
| 6   | **Existing artifacts**  | Read any current `setup_spec/eval_spec.json` and `setup_spec/policies.md`. Diff against what we'd generate.                                                                                                                                                                                    |

## Workflow

### Step 1 — Read and analyze the agent

Read the entrypoint file. From it, extract **statically** (do not rely on the LLM analyzer alone — it collapses dict params and misses output keys):

1. **Entrypoint signature** — name, every parameter, default values, type annotations.
   Use `ast.parse` / `ast.FunctionDef` walk.
1. **Return shape** — collect every `return {...}` literal in the function body. Union the keys across branches; mark a key `optional: true` if it appears in some but not all returns. For non-dict returns, set output type to `text`.
1. **Tool definitions** — look for `@tool`, `Tool(`, `FunctionTool(`, `tools=[...]`, OpenAI/Anthropic tool dicts. Record name, description, parameter schema.
1. **Module docstring + `AGENT_DESCRIPTION` constant** — use as `agent_description`.
1. **Imports** — note any local sibling modules so the spec's `scope.optimizable_paths` covers the right files.
1. **Sibling local packages** — for every top-level `import X` / `from X import ...` in the entrypoint file, check whether `X` resolves to a directory sitting next to the entrypoint inside the project root. Such packages are **the agent**, not third-party dependencies — even if they have their own `pyproject.toml`, `LICENSE`, `.egg-info/`, or `tests/`. Collect them as `sibling_local_packages = [<path>, ...]`.

Output of this step is a `static_analysis` dict you carry through the rest of the skill. **Do not** trust the LLM analyzer's `input_schema` if it returns a single field whose name matches the entrypoint's *only* parameter and whose type is `object`/`dict` — that means the analyzer flattened a dict-of-fields into an opaque blob. In that case, decompose using either:

- The user's seed data (if a `dataset.json` exists, take the keys of `cases[0]["input"]`).
- Direct user prompts (one question per top-level field).

### Step 2 — Confirm the analysis with the user

Print a compact summary table:

```
Agent:        new_examples/TradingAgents/tradingagents/agent_entrypoint.py
Entrypoint:   run(ticker, date, llm_provider=None, ...)
Inputs:       ticker (string), date (string), llm_provider (string|null), ...
Outputs:      ticker, date, decision (enum), market_report (text), ... (12 fields)
Tools:        get_stock_data, get_indicators, get_news, ... (9 tools)
```

Then `AskQuestion`:

- *"The detected output has 12 fields. Score all of them, or only a subset?"* → if subset, ask for the list.
- *"`decision` looks like an enum. Valid values are [Buy, Overweight, Hold, Underweight, Sell] — correct?"*
- *"Are there fields I missed?"* (free text)

**Scope confirmation (mandatory whenever `static_analysis.sibling_local_packages` is non-empty):**

For each sibling local package detected in Step 1.6, `AskQuestion` with options:

1. *Optimizable* — package is part of the agent; the optimizer may edit it. (default for sibling packages imported by the entrypoint)
1. *Context only* — optimizer can read but must not edit (use this for stable internal libraries the user doesn't want rewritten).
1. *Exclude* — treat as a true vendored third-party copy and ignore entirely.

Phrase the prompt concretely, e.g.:

> *"`<pkg>/` is a sibling Python package that `<entrypoint_file>` imports. It contains part of the agent's logic. Should the optimizer be allowed to edit it, treat it as read-only context, or ignore it as an external library?"*

Record the answer on `static_analysis.scope_decisions[<pkg>]`. Step 4 builds `scope.optimizable_paths` / `context_paths` / `exclude_paths` from this map — never silently exclude a sibling package because it "looks like a library".

### Step 3 — Elicit the policy

Branch on the user's policy-source choice from Step 1.

**3a. Interactive elicitation** (preferred — produces the strongest policy)

Ask each of the following as a separate question. Skip questions where the answer is obvious from code (e.g. enum values) but always ask the *domain* questions — they aren't in the code.

1. *Purpose*: "In one sentence, what is this agent's job?"
1. *Domain rules*: "What real-world business rules must the agent follow? (e.g. 'Refunds over $500 require manager approval', 'Cold leads never get a demo offer')"
1. *Hard constraints*: "What outcomes are unacceptable, even if the agent technically succeeds? (e.g. 'Never recommend an out-of-stock product')"
1. *Edge cases*: "Tricky inputs and the correct handling for each."
1. *Terminology*: "Key terms, categories, or thresholds the agent needs to know (e.g. 'Hot lead = visited pricing page in last 7 days')."
1. *Tool ordering*: "Are there required orderings between tools? (e.g. 'fetch data before computing indicators')"
1. *Quality expectations*: "Style/format requirements for free-text output fields."

Free-text answers are fine — the skill restructures them.

**3b. Auto-infer from code** — call `overmind.setup.policy_generator.generate_policy_from_code` if available; otherwise use the fallback prompt in [Step 6](#step-6--fallbacks-when-overmind-helpers-fail).

**3c. Improve existing doc** — read the file. Call `overmind.setup.policy_generator.improve_existing_policy` if available, then show the diff and ask the user which version to keep.

### Step 4 — Generate the full policy and eval spec

Using everything gathered in Steps 1–3, produce the complete artifacts now:

1. Build the full `policy` dict from the elicited answers (Step 3).
1. Build the full `spec` dict using the algorithm below.
1. Render `policies.md` as a human-readable markdown document from the `policy` dict.

Do **not** save anything to disk yet — hold both artifacts in memory until the user approves them in Step 5.

Construct the spec dict directly (do **not** trust the LLM to allocate weights — do it deterministically):

```python
spec = {
    "agent_description": static_analysis["description"],
    "agent_path": str(Path(agent_path).resolve()),
    "entrypoint_fn": entrypoint_fn,
    "input_schema": {
        # one entry per *parameter* of the entrypoint, NOT a single "input_data" blob.
        # If the entrypoint takes a single typed dict, decompose its keys.
        param: {"type": <inferred>, "description": "..."}
        for param in static_analysis["params"]
    },
    "output_fields": {
        # one entry per key in the union of return dicts
        field: {
            "type": <"enum"|"number"|"text"|"boolean">,
            "description": "...",
            "values": [...],          # enum only
            "range": [lo, hi],         # number only
            "optional": <bool>,
            "weight": <int>,
            "importance": <"critical"|"important"|"minor">,
            "eval_mode": "similarity",  # text only — use "similarity" for important text, "non_empty" for minor
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
    "tool_usage_weight": 10,           # only if tools exist
    "llm_judge_weight": 10,            # only if any text field is critical/important OR a policy exists
    "consistency_rules": [...],
    "scope": build_scope(static_analysis),  # see "Scope construction" below
    "optimizable_elements": [...],
    "fixed_elements": [...],
    "policy": <structured policy dict from Step 3>,
}
```

**Scope construction algorithm:**

```
optimizable_paths = [entrypoint_rel]
context_paths     = []
exclude_paths     = [".overmind/**", ".venv/**", ".github/**",
                     "tests/**", "benchmarks/**", "scripts/**",
                     "**/__pycache__/**", "**/*.egg-info/**",
                     "uv.lock", "poetry.lock", "Dockerfile"]

for pkg, decision in static_analysis["scope_decisions"].items():
    glob = f"{pkg}/**/*.py"
    if   decision == "optimizable": optimizable_paths.append(glob)
    elif decision == "context":     context_paths.append(glob)
    elif decision == "exclude":     exclude_paths.append(f"{pkg}/**")

# README/docs/pyproject as read-only context unless they're the agent itself
for ctx in ("README.md", "docs/**/*.md", "pyproject.toml"):
    if Path(project_root / ctx.split("/")[0]).exists():
        context_paths.append(ctx)
```

The skill MUST produce this scope deterministically from the user's Step 2 answers. Do not pass through the LLM analyzer's `scope` block when sibling packages exist — it predates this rule and may exclude them.

**Weight allocation algorithm** (must sum to exactly `total_points = 100`):

```
remaining = 100 - structure_weight - tool_usage_weight - llm_judge_weight   # may be 60
mult = {"critical": 3, "important": 2, "minor": 1}
raw  = {f: mult[importance[f]] for f in output_fields}
total_raw = sum(raw.values())
for f in output_fields:
    weight[f] = round(raw[f] / total_raw * remaining)
# fix rounding by adding the residual to the first field
weight[first] += remaining - sum(weight.values())
assert structure_weight + tool_usage_weight + llm_judge_weight + sum(weight.values()) == 100
```

**Validation gates** (assert before showing to user — surface any failures immediately):

- Every key of `input_schema` is a real parameter of the entrypoint.
- Every key of `output_fields` appears in at least one `return` statement.
- No `output_fields` key has `weight == 0` unless `importance == "minor"` and the user opted to skip it.
- For every enum field, `values` is non-empty.
- For every number field, `range` has two numeric entries.
- The sum check above passes.
- If `policy` is present, `policy["domain_rules"]` is a non-empty list (otherwise prompt the user — silent empty policies are the #1 cause of useless optimize runs).
- For every sibling local package in `static_analysis.sibling_local_packages`, exactly one of `optimizable_paths` / `context_paths` / `exclude_paths` references it. Never let a sibling package be invisible to all three.

### Step 5 — Show generated content, get user approval, iterate

Show the **actual generated content** — not a skeleton — to the user. Display both artifacts in full (or truncated with `...` only for very long field lists):

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POLICY  (policies.md)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Purpose: <one-sentence purpose>

Domain rules:
  • <rule 1>
  • <rule 2>
  ...

Hard constraints:
  • <constraint 1>
  ...

Edge cases:
  • <case 1>
  ...

Terminology:
  • <term>: <definition>
  ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVAL SPEC  (eval_spec.json)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
input_schema:
  <param1>  (<type>)  — <description>
  <param2>  (<type>)  — <description>
  ...

output_fields:
  <field1>  <type>  importance=<critical|important|minor>  weight=<N>
  <field2>  ...
  ...
  ── weights: fields <N> + structure <N> + tools <N> + llm_judge <N> = 100

scope:
  optimizable: <paths>
  context:     <paths>
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

**If the user requests changes:**

1. Ask in plain conversation: *"What specifically should change?"*
1. Apply the changes, regenerate the affected artifact(s) (loop back to Step 4 for a full rebuild, or patch in-place for minor edits).
1. Show the updated content again using the same display format above.
1. Ask for approval again.
1. Repeat until the user explicitly approves.

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

The repo ships helpers under `overmind.setup.*`. Try them first; fall back to direct LLM calls when imports fail or output is malformed.

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

If `HAS_OVERMIND` is `False`, overmind is not installed. Tell the user to install it first (`pip install overmind`) and re-run before proceeding to the fallback runner.

When `HAS_OVERMIND` is True, prefer `analyze_agent(...)` for the LLM analysis pass, then **post-process** its output with the static checks from Step 1 before passing to `generate_spec_from_proposal`. The most common breakage is:

| Symptom                                                            | Cause                                                                                                      | Fix                                                                                                                                         |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `input_schema` has one entry typed `object`                        | LLM saw a single dict-typed parameter and didn't decompose                                                 | Replace with the decomposed schema you built statically.                                                                                    |
| `output_fields` missing keys present in the agent's `return {...}` | LLM only captured "important-looking" keys                                                                 | Union all return-dict keys yourself, classify type, then re-call `generate_spec_from_proposal` with the corrected `output_schema`.          |
| All weights `0` or `None`                                          | `proposed_criteria.fields` was empty                                                                       | Skip `generate_spec_from_proposal` and build the spec dict directly using the algorithm in Step 4.                                          |
| `policy` block missing rules but `policies.md` saved               | LLM response fenced under wrong tag (parsing in `_extract_markdown_and_json` couldn't find the JSON block) | Re-run policy generation with a stricter prompt asking for a `\`\`\`json\` fenced block at the end; or parse it manually from the markdown. |
| `consistency_rules` empty for an enum/number agent                 | Auto-generator only fires on naming patterns (min/max etc.)                                                | Ask the user: "Should `<number_field>` correlate with `<enum_field>`?" and append the rule manually.                                        |

When `HAS_OVERMIND` is False (e.g. running outside the project venv), drop a `_policy_eval_runner.py` in the project root that performs Steps 1–4 with `litellm` directly. Run it from the **project root** (the directory containing `.overmind/`) — never `cd` to a parent directory or pass `--project` to `uv run`:

```bash
python _policy_eval_runner.py
```

Delete the runner on success.

### Step 7 — Smoke test (non-blocking)

If `setup_spec/dataset.json` already exists, run the agent against `cases[0]["input"]` once to confirm the new `input_schema` matches the function signature. Use a subprocess so a hung agent can't block the chat:

```python
import subprocess, sys, json, textwrap

case = json.loads(Path(base / "dataset.json").read_text())[0]
input_kwargs = (
    case["input"]
    if isinstance(case.get("input"), dict)
    else {"input_data": case["input"]}
)
script = textwrap.dedent(f"""
    import json, sys
    sys.path.insert(0, {repr(str(Path(agent_path).parent))})
    from {Path(agent_path).stem} import {entrypoint_fn} as fn
    print(json.dumps({{"ok": True, "out_keys": list(fn(**{input_kwargs!r}).keys() if isinstance(fn(**{input_kwargs!r}), dict) else [])}}))
""")
res = subprocess.run(
    [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
)
```

On failure, print the error and tell the user which `input_schema` field name is likely wrong. **Do not** rewrite the spec automatically — let the user decide whether to fix the agent or the schema.

### Step 8 — Summarize

End the session with:

- Full path to `eval_spec.json` and `policies.md`.
- Field counts, weight totals, policy stats.
- Smoke-test result.
- Next command: `overmind optimize <agent-name>`.

## Repair mode (existing broken artifacts)

When the user points the skill at an agent that already has a `setup_spec/` directory:

1. Read `eval_spec.json` and `policies.md`.
1. Run static analysis on the agent (Step 1).
1. Diff: list every field that is wrong (collapsed input, missing output keys, weight sum ≠ 100, empty policy lists, mismatched enum values vs code).
1. Show the diff to the user. `AskQuestion`: *"Apply all fixes / pick which to apply / abort"*.
1. Apply selected fixes, re-run validation, re-save.

The diff format must be concrete — show the *current* vs *proposed* value side by side, not a vague "this looks wrong".

## Common issues

| Problem                                                               | Fix                                                                                                                                                                                                                                               |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ImportError: overmind.setup`                                         | overmind is not installed — run `pip install overmind` and re-run.                                                                                                                                                                                |
| Policy generation returns empty `domain_rules: []`                    | The LLM produced markdown but no JSON block — re-prompt with: *"Append a `\`\`\`json\` block at the end with keys: domain_rules, domain_edge_cases, terminology, output_constraints, tool_requirements, decision_mapping, quality_expectations."* |
| `input_schema` has one `object`-typed entry matching a dict parameter | Decompose: read seed data keys or ask the user one question per sub-field.                                                                                                                                                                        |
| Weights sum to 99 or 101                                              | Apply the rounding-residual fix in Step 4.                                                                                                                                                                                                        |
| Output field present in `return` but absent from spec                 | Add it; re-allocate weights.                                                                                                                                                                                                                      |
| Smoke test: `TypeError: unexpected keyword argument`                  | Field name in `input_schema` doesn't match a parameter. Use the entrypoint signature as ground truth and rename.                                                                                                                                  |
| User has a long policy doc but no structured policy block             | Use `generate_policy_from_document` (when overmind is importable) or wrap it in a single `domain_rules: [<full text>]` entry as a temporary stop-gap and warn the user.                                                                           |
| Agent's only entrypoint is async                                      | Wrap the smoke-test call in `asyncio.run(fn(**kwargs))`. The eval spec itself doesn't change.                                                                                                                                                     |

## What the skill must NOT do

- Never write the eval_spec or policy outside `.overmind/agents/<name>/setup_spec/`.
- Never invent enum values that aren't in the code or confirmed by the user.
- Never silently drop output fields. If the user wants to score only a subset, record that explicitly via `importance: "minor"` + `eval_mode: "skip"`.
- Never produce a spec where the weights don't sum to `total_points`.
- Never ship a policy with all domain-rule lists empty without a visible warning.
- Never silently drop a sibling local package from scope. If the entrypoint imports a directory that lives next to it inside the project root, the user must be asked explicitly whether to make it optimizable, context-only, or excluded.
