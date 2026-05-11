---
name: overmind-generate-dataset
description: "Generate, augment, validate, and smoke-test synthetic evaluation datasets for Overmind agents. Use when the user wants to generate a dataset, create test cases, add red-team examples, augment seed data, validate dataset schema consistency, or prepare evaluation data before running policy/eval generation or optimization."
metadata:
  version: "2.0"
  product: "Overmind"
---

# Generate Dataset for Overmind Agents

Generates a synthetic JSON test dataset for a registered Overmind agent by analyzing its entrypoint file and running LLM-powered persona-driven generation — the same pipeline used by `overmind setup`.

The dataset must target the Overmind-compatible entrypoint (the separate interaction harness), because those inputs are what Overmind will pass when it runs and evaluates the agent. The dataset must be saved to the canonical path:

- `.overmind/agents/<agent-name>/setup_spec/dataset.json`

## Operating principles

- **Codebase-derived dataset**: Generate dataset cases from the registered entrypoint, native agent files, schemas, prompts, tests, examples, docs, tools, validators, and existing artifacts.
- **Canonical path only**: Save datasets to `.overmind/agents/<agent-name>/setup_spec/dataset.json`.
- **Registration first**: Resolve the agent from `.overmind/agents.toml`. If not registered, tell the user to run `/overmind-register-agent` first.
- **Entrypoint-first schema**: Use the registered separate entrypoint file signature as the canonical input schema. Do not generate inputs for an internal native function.
- **Normalized output targets**: Expected outputs must target the entrypoint's top-level output fields, not nested native structures or raw lists.
- **Entrypoint is not optimization material**: Use it to understand how to call the agent, but treat it as a fixed interaction harness.
- **Schema consistency**: Every case must have the same top-level input shape and compatible expected-output shape.
- **Seed data first**: Synthetic generation without seed data may miss real production distribution, nuanced domain labels, and hard edge cases. Always ask whether seed data is available and require an explicit choice before proceeding without it.
- **Ask before overwriting**: If a dataset already exists, ask whether to replace, append, or backup before writing. Do not silently overwrite existing curated data.
- **Non-blocking smoke tests**: Save the dataset even if live external calls fail, but clearly distinguish schema failures from auth/API failures.
- **No secret inspection**: Do not inspect or print API key values.
- **Ask only for blockers**: Do not ask the user to describe behavior, schemas, or examples when the codebase contains enough context.

## Workflow

Copy this checklist into your response and check off each step as you complete it:

```
Dataset Generation Progress:
- [ ] Step 1: Resolve the agent file
- [ ] Step 2: Collect parameters
- [ ] Step 3: Build eval spec
- [ ] Step 4: Generate the dataset
- [ ] Step 5: Handle seed data (if provided)
- [ ] Step 6: Validate schema consistency
- [ ] Step 7: Smoke-test the agent
- [ ] Step 8: Summarize
```

### Step 1 — Resolve the agent file

The user provides an **agent name** (the slug used during `overmind agent register`, e.g. `my-agent`). They may **optionally** also provide a **seed dataset file** (a path to an existing JSON file with example inputs/outputs). Let the user know upfront:

> "If you already have example inputs/outputs for this agent, you can pass a seed dataset file path and I'll use those as a starting point."

Look up from `.overmind/agents.toml`:

- Find the entry where `name` matches the agent name.
- Take the `entrypoint` value (e.g. `examples.myagent.overmind_ep:run`).
- Derive the file path: split on `:`, take the module part, replace `.` with `/`, append `.py`.

If the agent name is not in `.overmind/agents.toml`, tell the user to register it first with `/overmind-register-agent`.

Read the resolved **Overmind entrypoint file** (the separate harness, not the native agent). Extract:

- **Entrypoint function**: the function name from the entrypoint string (after `:`).
- **Agent description**: from a module docstring, `AGENT_DESCRIPTION` variable, or comments near the function.
- **Canonical input parameter names**: read the function signature directly — **exact** parameter names in order, excluding `self`. Every generated `input` dict must have exactly these keys, no more, no less.
- **Input types**: from type annotations on each parameter (`str`, `int`, `dict`, `list`, etc.). If no annotation, infer from usage or default to `string`.
- **Canonical output keys**: read the return type annotation or `return` statements. Extract the exact field names the function produces:
  - `-> dict` with a `TypedDict` or typed `return {...}` literal → use those exact keys
  - `-> str` → single key `"result"` of type `text`
  - If the return type is unclear, read all `return` statements and union the keys across them
  - Every generated `expected_output` dict must have exactly these keys.

Also read the native agent files imported by the separate entrypoint for behavioral context, but keep dataset input fields aligned to the entrypoint signature and expected outputs aligned to the entrypoint's normalized top-level fields.

**Write down both the canonical input keys and canonical output keys explicitly before moving to Step 2.** Both will be used to validate every generated case.

If seed samples exist (user-provided JSON), verify:

- Their `input` keys match the canonical input keys.
- Their `expected_output` keys match the canonical output keys.

Drop any seed case that fails either check and warn the user.

### Step 2 — Collect parameters

The agent name is already known from Step 1. Collect:

1. **Seed data decision (required, before generating anything)** — Use `AskQuestion`:
   > "Do you have existing example inputs/outputs for this agent?"
   > Options: Yes, I'll provide a seed file path | No, proceed with synthetic generation

   If **No**: explain the trade-off — *"Synthetic generation is available but may miss real production distribution, nuanced domain labels, and hard edge cases. A seed dataset from real usage is strongly preferred."* Require the user to explicitly choose to proceed. Do not silently start generation.

   If **Yes**: ask for the project-relative path or attachment. Read and validate the seed file before continuing.

1. **Number of test cases** — total datapoints to generate (default: 20)
1. **Number of red-teamers** — how many distinct adversarial/diverse personas (default: 5). Explain: *"Each red-teamer is a persona — novice user, power user, edge-case tester, adversarial attacker, domain expert, etc. More = broader and harder coverage."*

### Step 3 — Build eval spec

Construct the `eval_spec` dict from what you detected.

**Critical rules**:

- Keys of `input_schema` must be **exactly** the canonical input parameter names extracted from the function signature in Step 1.
- Keys of `output_fields` must be **exactly** the canonical output keys extracted from the return type/statements in Step 1.

```python
eval_spec = {
    "agent_description": "<detected or inferred description>",
    "input_schema": {
        # One entry per entrypoint parameter — key = exact parameter name
        "param_name": {
            "type": "string",  # string | number | boolean | enum | dict | list
            "description": "...",
            "values": ["a", "b"],  # only for enum type
        },
    },
    "output_fields": {
        "field_name": {
            "type": "string",
            "description": "...",
            "weight": 10,
            "importance": "important",  # important | critical | minor
        },
        # one entry per expected output key
    },
}
```

For plain-text or list outputs, use a single `output_fields` entry with `type: "text"`.

Before continuing, write both canonical key sets as Python constants in the runner script (Step 4):

```python
CANONICAL_INPUT_KEYS = frozenset(<list of exact parameter names>)
CANONICAL_OUTPUT_KEYS = frozenset(<list of exact output field names>)  # None if output is plain text/list
```

### Step 4 — Generate the dataset

Write a runner script `_datagen_runner.py` in the project root:

```python
import json, os
from pathlib import Path
from rich.console import Console

import overmind
from overmind.core.paths import load_overmind_dotenv
load_overmind_dotenv()
overmind.init()

from overmind.optimize.data import generate_diverse_synthetic_data

console = Console()

AGENT_NAME = "<name>"
NUM_SAMPLES = <N>
NUM_PERSONAS = <R>   # red-teamers
MODEL = os.getenv("SYNTHETIC_DATAGEN_MODEL", "openai/gpt-4o")

# Exact parameter names from the entrypoint function signature — ground truth for input keys
CANONICAL_INPUT_KEYS = frozenset(<list of exact parameter names>)

# Exact field names from the return type/statements — ground truth for output keys (None if plain text)
CANONICAL_OUTPUT_KEYS = frozenset(<list of exact output field names>)  # or None

AGENT_DESCRIPTION = """<description from step 1>"""

AGENT_CODE = """<full source of the entrypoint file>"""

EVAL_SPEC = <eval_spec dict from step 3>

SEED_CASES = []  # fill from seed samples if available

cases = generate_diverse_synthetic_data(
    agent_description=AGENT_DESCRIPTION,
    model=MODEL,
    num_samples=NUM_SAMPLES,
    num_personas=NUM_PERSONAS,
    agent_code=AGENT_CODE,
    eval_spec=EVAL_SPEC,
    existing_cases=SEED_CASES if SEED_CASES else None,
    console=console,
)

out_path = Path(f".overmind/agents/{AGENT_NAME}/setup_spec/dataset.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(cases, indent=2))
console.print(f"[bold green]✓[/bold green]  Saved [bold]{len(cases)}[/bold] cases → {out_path}")
```

Run it from the **project root** (the directory containing `.overmind/`). Never `cd` to a parent directory or pass `--project` to `uv run`:

```bash
python _datagen_runner.py
```

After success, delete `_datagen_runner.py`.

**Model selection priority:**

1. `SYNTHETIC_DATAGEN_MODEL` env var (from `.overmind/.env` or `.env` in the project)
1. `OPENAI_API_KEY` present → default to `openai/gpt-4o`
1. If neither, ask the user: *"Which model should I use for data generation? (e.g. openai/gpt-4o, anthropic/claude-opus-4-5)"*

**Fallback — if `generate_diverse_synthetic_data` cannot be imported:**

If the import fails, overmind is not installed. Tell the user to `pip install overmind` and re-run. If overmind is installed but the import still fails, fall back to direct LLM calls:

```python
import os, json
import litellm  # or openai

MODEL = os.getenv("SYNTHETIC_DATAGEN_MODEL", "gpt-4o")

PROMPT = f"""
You are generating a synthetic test dataset for an AI agent.

Agent description: {AGENT_DESCRIPTION}

Agent source code:
```python
{AGENT_CODE}
```

Input schema: {json.dumps(EVAL_SPEC['input_schema'], indent=2)}
Output schema: {json.dumps(EVAL_SPEC['output_fields'], indent=2)}

Generate {NUM_SAMPLES} diverse test cases covering these {NUM_PERSONAS} personas:

1. Novice user — basic, possibly incomplete inputs
1. Power user — complex, well-formed inputs
1. Edge case tester — boundary values, empty fields, unusual combos
1. Adversarial user — misleading, contradictory, or injection-style inputs
1. Domain expert — nuanced, technically precise scenarios
   (repeat or mix personas if more than 5 are requested)

Return ONLY a JSON array. Each item: {{"input": {{...}}, "expected_output": {{...}}}}
"""

response = litellm.completion(
    model=MODEL,
    messages=[{"role": "user", "content": PROMPT}],
    temperature=0.8,
)
content = response.choices[0].message.content
start, end = content.find("["), content.rfind("]") + 1
cases = json.loads(content[start:end])
```

### Step 5 — Handle seed data (when provided)

If the user has seed examples:

1. Save them to `_seed.json` (or read from existing path).
1. Use `existing_cases=seed_cases` in the runner — the generator avoids duplicating them.
1. Merge seed + generated: `combined = seed_cases + generated_cases`, then save.

To also run coverage analysis on seed data before augmenting:

```python
from overmind.optimize.data_analyzer import analyze_seed_coverage

coverage = analyze_seed_coverage(
    cases=seed_cases,
    eval_spec=EVAL_SPEC,
    policy_context=None,
    agent_description=AGENT_DESCRIPTION,
    model=MODEL,
    console=console,
)
gaps = coverage.get("coverage_gaps", [])
# pass gaps= to generate_diverse_synthetic_data for targeted augmentation
```

### Step 6 — Validate schema consistency

Before saving, enforce that every datapoint's `input` keys **exactly match** `CANONICAL_INPUT_KEYS` and every `expected_output` dict's keys **exactly match** `CANONICAL_OUTPUT_KEYS`. Both are derived from the entrypoint source in Step 1 — never use the first generated case as a reference.

```python
def _enforce_schema_consistency(
    cases: list[dict],
    canonical_input_keys: frozenset[str],
    canonical_output_keys: frozenset[str] | None,
) -> list[dict]:
    if not cases:
        return cases

    clean, dropped = [], []
    for i, case in enumerate(cases):
        inp = case.get("input")
        out = case.get("expected_output")

        if inp is None or out is None:
            dropped.append((i, "missing input or expected_output"))
            continue

        if isinstance(inp, dict):
            if set(inp.keys()) != canonical_input_keys:
                dropped.append(
                    (i, f"input keys {set(inp.keys())} != {canonical_input_keys}")
                )
                continue

        if canonical_output_keys is not None and isinstance(out, dict):
            if set(out.keys()) != canonical_output_keys:
                dropped.append(
                    (i, f"output keys {set(out.keys())} != {canonical_output_keys}")
                )
                continue

        clean.append(case)

    if dropped:
        console.print(
            f"[yellow]⚠  Dropped {len(dropped)} case(s) with schema mismatches "
            f"(kept {len(clean)}):[/yellow]"
        )
        for idx, reason in dropped[:5]:
            console.print(f"  case {idx}: {reason}")
    return clean


cases = _enforce_schema_consistency(cases, CANONICAL_INPUT_KEYS, CANONICAL_OUTPUT_KEYS)
```

If more than 20% of cases are dropped, regenerate with tighter schema instructions rather than accepting a thin dataset.

**Before saving, ask if a dataset already exists:**

If `.overmind/agents/<name>/setup_spec/dataset.json` already exists, `AskQuestion`:

> "A dataset already exists for this agent. What would you like to do?"
> Options: Replace it | Append to it | Save a timestamped backup, then replace

### Step 7 — Smoke-test the agent

After saving `dataset.json`, verify the agent actually runs on a sample of the generated inputs:

```python
import asyncio, importlib.util, inspect, traceback, sys
from pathlib import Path
from dotenv import load_dotenv


def _call_entrypoint(fn, inp):
    """Invoke fn with inp, transparently awaiting if the entrypoint is async."""
    call = fn(**inp) if isinstance(inp, dict) else fn(inp)
    if inspect.iscoroutine(call):
        return asyncio.run(call)
    return call


def _smoke_test(
    agent_file: str, entrypoint_fn: str, sample_cases: list[dict], agent_name: str
) -> None:
    """Run the entrypoint against up to 3 generated cases and report pass/fail."""
    agent_env = Path(".overmind/agents") / agent_name / ".env"
    if agent_env.exists():
        load_dotenv(agent_env, override=True)
        console.print(f"  [dim]Loaded env from {agent_env}[/dim]")
    else:
        console.print(
            f"  [yellow]⚠  No .env found at {agent_env} — agent may fail if it needs API keys[/yellow]"
        )

    spec = importlib.util.spec_from_file_location("_agent_under_test", agent_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, entrypoint_fn)
    if inspect.iscoroutinefunction(fn):
        console.print(f"  [dim]Detected async entrypoint — wrapping calls in asyncio.run[/dim]")

    passed, failed = 0, []
    for i, case in enumerate(sample_cases[:3]):
        inp = case["input"]
        try:
            result = _call_entrypoint(fn, inp)
            assert result is not None, "returned None"
            passed += 1
        except Exception as exc:
            failed.append((i, str(exc)))

    if failed:
        console.print(
            f"[bold red]✗  Smoke test: {len(failed)}/3 case(s) failed[/bold red]"
        )
        for idx, err in failed:
            console.print(f"  case {idx}: {err}")
        console.print(
            "  [dim]The dataset is saved but these inputs may not match the "
            "entrypoint signature. Review the input_schema and re-generate if needed.[/dim]"
        )
    else:
        console.print(
            f"[bold green]✓[/bold green]  Smoke test passed ({passed}/3 cases ran successfully)"
        )


_smoke_test("<path/to/overmind_entrypoint.py>", "<entrypoint_fn_name>", cases, AGENT_NAME)
```

Classify failures:

- **Schema failure**: Unexpected keyword argument, missing required argument, wrong input shape. Tell the user which field or shape needs repair.
- **Environment failure**: Missing API key, auth failure, network failure. Keep the dataset and tell the user which configuration is needed before a full run.
- **Runtime failure**: Internal agent error unrelated to schema. Report the traceback.

The smoke test is **non-blocking** — a failure prints a warning but still saves the dataset.

### Step 8 — Summarize

**Validation checklist** (verify before responding):

- [ ] Agent was resolved from `.overmind/agents.toml`.
- [ ] Dataset path is `.overmind/agents/<agent-name>/setup_spec/dataset.json`.
- [ ] Seed data decision was explicit (not skipped silently).
- [ ] Final case count matches the requested count or the shortfall is explained.
- [ ] Dropped case count is recorded.
- [ ] Smoke-test status is recorded.
- [ ] Existing dataset was not overwritten without user approval.
- [ ] The response does not suggest the entrypoint should be optimized.

Tell the user:

- Full path to the saved `dataset.json`
- Number of cases saved (after schema filtering) and how many were dropped
- Number of seed cases preserved vs generated cases added
- Smoke test result (✓ pass / ✗ fail with error details)
- Next step: run `/overmind-generate-policy-and-eval` with agent name `<agent-name>`

## Common issues

| Problem                                              | Fix                                                                                       |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `generate_diverse_synthetic_data` not found          | overmind not installed — run `pip install overmind`, then re-run; use direct LLM fallback if needed |
| Model auth error                                     | Check `.overmind/.env` or `.env` for `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`              |
| 0 cases generated                                    | Increase temperature or reduce `NUM_SAMPLES` per run; retry                              |
| Input schema missing fields                          | Re-read the entrypoint and check `*args`/`**kwargs` usage                                |
| >20% cases dropped by schema filter                  | Tighten the `eval_spec` and regenerate; the LLM is producing inconsistent keys           |
| Smoke test: `TypeError: unexpected keyword argument` | The detected `input_schema` has extra or wrong parameter names — fix and regenerate      |
| Smoke test: API / auth errors from the agent         | Expected if the agent calls external APIs; configure credentials before full evaluation  |
| Smoke test: result is a `<coroutine>` / never awaited warning | Entrypoint is async but the smoke test called it synchronously — use the `_call_entrypoint` helper which detects coroutines via `inspect.iscoroutine` and wraps in `asyncio.run` |
| Smoke test: `RuntimeError: asyncio.run() cannot be called from a running event loop` | The entrypoint already calls `asyncio.run` internally — refactor the harness to expose an `async def run` instead and let the smoke test wrap it |
| Entrypoint not a separate harness                    | Re-run `/overmind-register-agent` to create a proper separate entrypoint file            |
| Expected outputs target nested/list fields           | Repair the separate entrypoint so native outputs are normalized to top-level fields      |
