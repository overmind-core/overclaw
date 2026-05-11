---
name: overmind-generate-dataset
description: Generate synthetic test datasets for overclaw/overmind agents from an entrypoint file. Use when the user wants to generate a dataset, create test data, or build evaluation data for their agent. Auto-detects the entrypoint function, then prompts the user for number of datapoints and red-teamers (adversarial personas) before generating.
disable-model-invocation: true
---

# Generate Dataset for Overclaw Agents

Generates a synthetic JSON test dataset for any agent by analyzing its entrypoint file and running LLM-powered persona-driven generation — the same pipeline used by `overmind setup`.

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

The user provides an **agent name** (the slug used during `overmind agent register`, e.g. `my-agent`). They may **optionally** also provide a **seed dataset file** (a path to an existing JSON file with example inputs/outputs). Let the user know upfront they can supply one — e.g.:

> "If you already have example inputs/outputs for this agent, you can pass a seed dataset file path and I'll use those as a starting point."

If the user provides a seed file path at this point, note it and skip the seed-samples question in Step 2.

Look up the file path from the registry at `.overmind/agents.toml`:

- Find the entry where `name` matches the agent name.
- Take the `entrypoint` value (e.g. `examples/myagent/agent.py:run`).
- Derive the file path: split on `:`, take the module part, replace `.` with `/`, append `.py`.
  - `examples.myagent.agent` → `examples/myagent/agent.py`

If the agent name is not in `.overmind/agents.toml`, tell the user to register it first with `/overmind-register-agent`.

Read the resolved file. Extract:

- **Entrypoint function**: the function name from the entrypoint string (after `:`).
- **Agent description**: from a module docstring, `AGENT_DESCRIPTION` variable, or comments near the function.
- **Canonical input parameter names**: read the function signature directly and list the **exact** parameter names in order, excluding `self`. These are the ground truth — every generated `input` dict must have exactly these keys, no more, no less.
- **Input types**: from type annotations on each parameter (`str`, `int`, `dict`, `list`, etc.). If a parameter has no annotation, infer from usage in the function body or default to `string`.
- **Canonical output keys**: read the return type annotation or `return` statements in the function body. Extract the exact field names the function produces:
  - `-> dict` with a `TypedDict` or typed `return {...}` literal → use those exact keys
  - `-> list[dict]` → extract the keys each dict item contains from return statements or docstrings
  - `-> str` or `-> list[str]` → single key `"result"` of type `text`
  - If the return type is unclear, read all `return` statements and union the keys across them
  - These are the ground truth — every generated `expected_output` dict must have exactly these keys

Example (generic):

```
entrypoint: run(query, context)
canonical_input_keys: ["query", "context"]
input_schema: { query: string, context: dict }
canonical_output_keys: ["answer", "confidence"]
output_schema: dict with keys answer (str), confidence (float)
```

**Write down both the canonical input keys and canonical output keys explicitly before moving to Step 2.** Both will be used to validate every generated case.

If seed samples exist (user-provided JSON), read them and verify:

- Their `input` keys match the canonical input keys
- Their `expected_output` keys match the canonical output keys

Drop any seed case that fails either check and warn the user.

### Step 2 — Collect parameters interactively

The agent name is already known from Step 1. Use the `AskQuestion` tool or natural conversation to collect:

1. **Number of test cases** — total datapoints to generate (default: 20)
1. **Number of red-teamers** — how many distinct adversarial/diverse personas (default: 5). Tell the user: *"Each red-teamer is a persona — novice user, power user, edge-case tester, adversarial attacker, domain expert, etc. More = broader and harder coverage."*
1. **Output format** — what does the agent return? Options: JSON object / plain text / markdown / list of items / other. Only ask if the return type isn't clear from the code.
1. **Seed samples** — any existing example inputs/outputs? If yes, collect or locate them.

### Step 3 — Build eval spec

Construct the `eval_spec` dict from what you detected.

**Critical rules**:

- The keys of `input_schema` must be **exactly** the canonical input parameter names extracted from the function signature in Step 1 — same names, same count, no extras, no omissions. This is what guarantees every generated `input` dict can be passed directly as `fn(**input)` without a `TypeError`.
- The keys of `output_fields` must be **exactly** the canonical output keys extracted from the return type/statements in Step 1 — same names, same count. This is what guarantees every generated `expected_output` dict reflects what the function actually returns.

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

Both are used to validate every case after generation.

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

AGENT_CODE = """<full source of the agent file>"""

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

Before saving, enforce that every datapoint's `input` keys **exactly match** `CANONICAL_INPUT_KEYS` and every `expected_output` dict's keys **exactly match** `CANONICAL_OUTPUT_KEYS`. Both are derived from the entrypoint source in Step 1 — never use the first generated case as a reference, as it could itself be wrong.

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

        # Drop cases missing required top-level keys
        if inp is None or out is None:
            dropped.append((i, "missing input or expected_output"))
            continue

        # Enforce exact match against entrypoint parameter names
        if isinstance(inp, dict):
            if set(inp.keys()) != canonical_input_keys:
                dropped.append(
                    (i, f"input keys {set(inp.keys())} != {canonical_input_keys}")
                )
                continue

        # Enforce exact match against canonical output keys (skip if output is plain text)
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
        for idx, reason in dropped[:5]:  # show first 5 reasons
            console.print(f"  case {idx}: {reason}")
    return clean


cases = _enforce_schema_consistency(cases, CANONICAL_INPUT_KEYS, CANONICAL_OUTPUT_KEYS)
```

If more than 20% of cases are dropped, regenerate that batch rather than accepting a thin dataset. Re-run with a tighter prompt or reduce `NUM_SAMPLES` per persona shard.

### Step 7 — Smoke-test the agent with generated data

After saving `dataset.json`, verify the agent actually runs on a sample of the generated inputs. Add this block to the runner script:

```python
import importlib.util, traceback, sys
from pathlib import Path
from dotenv import load_dotenv


def _smoke_test(
    agent_file: str, entrypoint_fn: str, sample_cases: list[dict], agent_name: str
) -> None:
    """Run the entrypoint against up to 3 generated cases and report pass/fail."""
    # Load the agent's .env before importing — the agent may need API keys at import time
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

    passed, failed = 0, []
    for i, case in enumerate(sample_cases[:3]):
        inp = case["input"]
        try:
            result = fn(**inp) if isinstance(inp, dict) else fn(inp)
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


_smoke_test("<path/to/agent_file.py>", "<entrypoint_fn_name>", cases, AGENT_NAME)
```

The smoke test is **non-blocking** — a failure prints a warning but still saves the dataset. The user should fix the input schema mismatch and re-run if smoke tests fail.

**What counts as a pass**: the entrypoint returns a non-None value without raising an exception. The agent's external API calls (LLM, HTTP, etc.) are expected to be live; if you want dry-run tests, mock the external calls before importing.

### Step 8 — Summarize

After the script runs, tell the user:

- Full path to the saved `dataset.json`
- Number of cases saved (after schema filtering) and how many were dropped
- Smoke test result (✓ pass / ✗ fail with error details)
- Next step: run `/overmind-generate-policy-and-eval` with agent name `<agent-name>`.

## Model selection

Priority order:

1. `SYNTHETIC_DATAGEN_MODEL` env var (from `.overmind/.env` or `.env` in the project)
1. `OPENAI_API_KEY` → default to `openai/gpt-4o`
1. If neither, ask the user: *"Which model should I use for data generation? (e.g. openai/gpt-4o, anthropic/claude-opus-4-5)"*

## Output format

Dataset is saved to `.overmind/agents/<agent-name>/dataset.json` as a JSON array:

```json
[
  {
    "input": { "param1": "value", "param2": "value" },
    "expected_output": { "field1": "value" }
  }
]
```

If the agent returns plain text, `expected_output` is a string.

## Fallback: direct LLM generation

See [references/fallback.md](references/fallback.md) for the direct LLM fallback script.

## Common issues

See [references/common-issues.md](references/common-issues.md) for the full troubleshooting table.
