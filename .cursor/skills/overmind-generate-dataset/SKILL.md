______________________________________________________________________

## name: overmind-generate-dataset description: Generate synthetic test datasets for overclaw/overmind agents from an entrypoint file. Use when the user wants to generate a dataset, create test data, or build evaluation data for their agent. Auto-detects the entrypoint function, then prompts the user for number of datapoints and red-teamers (adversarial personas) before generating. disable-model-invocation: true

# Generate Dataset for Overclaw Agents

Generates a synthetic JSON test dataset for any agent by analyzing its entrypoint file and running LLM-powered persona-driven generation — the same pipeline used by `overmind setup`.

## Workflow

### Step 1 — Resolve the agent file

The user provides an **agent name** (the slug used during `overmind agent register`, e.g. `langextract`).

Look up the file path from the registry at `.overmind/agents.toml`:

- Find the entry where `name` matches the agent name.
- Take the `entrypoint` value (e.g. `new_examples.langextract.test:run_agent`).
- Derive the file path: split on `:`, take the module part, replace `.` with `/`, append `.py`.
  - `new_examples.langextract.test` → `new_examples/langextract/test.py`

If the agent name is not in `.overmind/agents.toml`, tell the user to register it first with `/overmind-register-agent`.

Read the resolved file. Extract:

- **Entrypoint function**: the function name from the entrypoint string (after `:`).
- **Agent description**: from a module docstring, `AGENT_DESCRIPTION` variable, or comments near the function.
- **Input schema**: from the function signature — each parameter becomes a schema field.
- **Output schema**: from the return type annotation or a sample return value in the code.

Example for agent name `langextract` → file `new_examples/langextract/test.py`:

```
entrypoint: run_agent(instruction, text, example)
input_schema: { instruction: string, text: string, example: dict }
output_schema: list[dict] with keys extraction_class, extraction_text
```

If seed samples exist (user-provided JSON), read them too.

### Step 2 — Collect parameters interactively

The agent name is already known from Step 1. Use the `AskQuestion` tool or natural conversation to collect:

1. **Number of test cases** — total datapoints to generate (default: 20)
1. **Number of red-teamers** — how many distinct adversarial/diverse personas (default: 5). Tell the user: *"Each red-teamer is a persona — novice user, power user, edge-case tester, adversarial attacker, domain expert, etc. More = broader and harder coverage."*
1. **Output format** — what does the agent return? Options: JSON object / plain text / markdown / list of items / other. Only ask if the return type isn't clear from the code.
1. **Seed samples** — any existing example inputs/outputs? If yes, collect or locate them.

### Step 3 — Build eval spec

Construct the `eval_spec` dict from what you detected:

```python
eval_spec = {
    "agent_description": "<detected or inferred description>",
    "input_schema": {
        "param_name": {
            "type": "string",  # string | number | boolean | enum | dict | list
            "description": "...",
            "values": ["a", "b"],  # only for enum type
        },
        # one entry per entrypoint parameter
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

### Step 4 — Generate the dataset

Write a runner script `_datagen_runner.py` in the project root:

```python
import json, os
from pathlib import Path
from rich.console import Console

import overmind
overmind.init()

from overmind.optimize.data import generate_diverse_synthetic_data

console = Console()

AGENT_NAME = "<name>"
NUM_SAMPLES = <N>
NUM_PERSONAS = <R>   # red-teamers
MODEL = os.getenv("SYNTHETIC_DATAGEN_MODEL", "openai/gpt-4o")

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

Before saving, enforce that every datapoint shares the same top-level `input` keys and the same top-level `expected_output` keys. Do this inside the runner script immediately after generation:

```python
def _enforce_schema_consistency(cases: list[dict]) -> list[dict]:
    if not cases:
        return cases

    # Determine canonical key sets from the first valid case
    ref_input_keys = (
        set(cases[0]["input"].keys())
        if isinstance(cases[0].get("input"), dict)
        else None
    )
    ref_output_keys = (
        set(cases[0]["expected_output"].keys())
        if isinstance(cases[0].get("expected_output"), dict)
        else None
    )

    clean, dropped = [], []
    for i, case in enumerate(cases):
        inp = case.get("input")
        out = case.get("expected_output")

        # Drop cases missing required top-level keys
        if inp is None or out is None:
            dropped.append(i)
            continue

        # Enforce identical input key set
        if ref_input_keys is not None and isinstance(inp, dict):
            if set(inp.keys()) != ref_input_keys:
                dropped.append(i)
                continue

        # Enforce identical output key set
        if ref_output_keys is not None and isinstance(out, dict):
            if set(out.keys()) != ref_output_keys:
                dropped.append(i)
                continue

        clean.append(case)

    if dropped:
        console.print(
            f"[yellow]⚠  Dropped {len(dropped)} case(s) with inconsistent schema "
            f"(kept {len(clean)}).[/yellow]"
        )
    return clean


cases = _enforce_schema_consistency(cases)
```

If more than 20% of cases are dropped, regenerate that batch rather than accepting a thin dataset. Re-run with a tighter prompt or reduce `NUM_SAMPLES` per persona shard.

### Step 7 — Smoke-test the agent with generated data

After saving `dataset.json`, verify the agent actually runs on a sample of the generated inputs. Add this block to the runner script:

```python
import importlib.util, traceback, sys


def _smoke_test(agent_file: str, entrypoint_fn: str, sample_cases: list[dict]) -> None:
    """Run the entrypoint against up to 3 generated cases and report pass/fail."""
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


_smoke_test("<path/to/agent_file.py>", "<entrypoint_fn_name>", cases)
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

If `from overmind.optimize.data import generate_diverse_synthetic_data` fails (overmind not installed or wrong env), fall back to direct LLM calls:

````python
import os, json
import litellm  # or openai

MODEL = os.getenv("SYNTHETIC_DATAGEN_MODEL", "gpt-4o")

PROMPT = f"""
You are generating a synthetic test dataset for an AI agent.

Agent description: {AGENT_DESCRIPTION}

Agent source code:
```python
{AGENT_CODE}
````

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

## Common issues

| Problem | Fix |
|---------|-----|
| `generate_diverse_synthetic_data` not found | Activate the project virtualenv: `source .venv/bin/activate` |
| Model auth error | Check `.overmind/.env` or `.env` for `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` |
| 0 cases generated | Increase temperature or reduce `NUM_SAMPLES` per run; retry |
| Input schema missing fields | Re-read the entrypoint and check `*args`/`**kwargs` usage |
| >20% cases dropped by schema filter | Tighten the `eval_spec` and regenerate; the LLM is producing inconsistent keys |
| Smoke test: `TypeError: unexpected keyword argument` | The detected `input_schema` has extra or wrong parameter names — fix and regenerate |
| Smoke test: API / auth errors from the agent | Expected if the agent calls external APIs; mock them or ignore and focus on schema errors |
```
