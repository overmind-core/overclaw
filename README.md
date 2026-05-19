<img width="3000" height="1000" alt="X Company Banner Black" src="https://github.com/user-attachments/assets/4a5caceb-49e8-4b8e-a6aa-511222a94381" />

# Overmind

An open-source optimizer for LLM agents. Point Overmind at your existing Python agent, give it a policy and a few test cases, and it iteratively rewrites prompts, tool descriptions, model choices, and pipeline logic to improve measured performance.

**Documentation:** [Overmind guide](https://docs.overmindlab.ai/guides/overmind_optimizer/)

**Overmind:** [Overmind Console](https://console.overmindlab.ai/)

## What it does

Overmind runs your agent against a test dataset, traces every LLM call and tool invocation, scores the outputs, and uses a strong reasoning model to generate concrete improvements. Changes that raise the score are kept; the rest are reverted. After several iterations you get a measurably better agent without manual prompt tweaking.

What makes Overmind different is **policy-driven optimization**. It goes beyond tracing by building deep context about your codebase and the behavior your agent is expected to follow. You define the decision rules, constraints, and expectations your agent must follow, and those policies guide every stage: evaluation criteria, test data synthesis, optimization diagnosis, and scoring.

### What gets optimized

- **System prompts** — more precise instructions, output format enforcement
- **Tool descriptions** — clearer parameters, better usage guidance
- **Model selection** — find the right quality/cost tradeoff
- **Agent logic** — tool-call ordering, iteration limits, output parsing
- **Policy compliance** — alignment with your domain rules and constraints

## Skills Quickstart

Use these from Cursor, Codex, or Claude Code.

The skills are the recommended way to use Overmind because they keep the workflow inside your normal coding environment: they scaffold the entrypoint, check the repo, bootstrap `.overmind/.env`, generate the policy/spec/dataset in the right order, and run optimization.

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/) or `pipx`, and API keys for at least one LLM provider.

```bash
uv tool install overmind
# or
pipx install overmind

cd your-agent-project/
overmind init
```

This creates `.overmind/` in your project root and prompts for API keys and default models. Safe to re-run anytime.

### Available skills

| Skill                                 | Path                                                          | What it does                                                                                                              |
| ------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `/overmind-register-agent`            | `overmind/skills/overmind-register-agent/SKILL.md`            | Creates or verifies the entrypoint harness, registers the agent, smoke-tests invocation, and bootstraps `.overmind/.env`. |
| `/overmind-generate-spec-and-dataset` | `overmind/skills/overmind-generate-spec-and-dataset/SKILL.md` | Generates `policies.md`, `eval_spec.json`, and `dataset.json` in one ordered pass so schemas stay aligned.                |
| `/overmind-optimize-agent`            | `overmind/skills/overmind-optimize-agent/SKILL.md`            | Runs the optimization loop from your coding environment, either via the CLI or host-driven `optimize-step`.               |

### How to use the skills

Run the skills from your coding-agent chat in this order:

```text
/overmind-register-agent path/to/your/agent.py
/overmind-generate-spec-and-dataset <agent-name>
/overmind-optimize-agent <agent-name>
```

`overmind init` is the only terminal step. It creates `.overmind/.env`, configures provider keys, and sets default models.

`/overmind-register-agent` inspects your repo, creates a thin entrypoint harness if needed, registers the agent, and runs smoke tests to confirm Overmind can invoke it reliably.

`/overmind-generate-spec-and-dataset` generates the behavioral policy, evaluation spec, and dataset together. This keeps the policy, scoring fields, and expected outputs aligned.

`/overmind-optimize-agent` runs the full optimization loop end to end from your coding environment.

<br />

______________________________________________________________________

<p align="center">
  <strong>The optimization loop</strong>
</p>

______________________________________________________________________

## How it Works

<p align="center">
  <code>Register</code> → <code>Generate policy</code> → <code>Build dataset</code> → <code>Optimize</code> → <code>Review report</code>
</p>

### 1. Initialize (`overmind init`)

Configure API keys and default models. Writes `.overmind/.env` in the current directory. Safe to re-run. This is the only terminal step — everything else runs through Agent Skills in your coding environment.

### 2. Register your agent (`/overmind-register-agent`)

Run `/overmind-register-agent path/to/your/agent.py` in your Cursor or Claude Code chat. The skill inspects your repo, creates a thin entrypoint harness if needed, registers the agent in `.overmind/agents.toml`, and runs smoke tests to confirm Overmind can invoke it reliably.

Your entrypoint function receives an input dict and must return a dict:

```python
def run(input_data: dict) -> dict:
    return {"response": result}
```

For framework-based agents, create a small wrapper that exposes this `dict → dict` contract.

### 3. Generate policy, spec, and dataset (`/overmind-generate-spec-and-dataset`)

Run `/overmind-generate-spec-and-dataset <agent-name>` in chat. The skill generates the behavioral policy, evaluation spec, and dataset in one ordered pass so their schemas stay aligned:

| Phase                   | What happens                                                                                                                                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Agent analysis**      | An LLM reads your agent code to detect the input/output schema, tools, and decision logic.                                                                                                                    |
| **Policy generation**   | If you have an existing policy, the skill analyzes it against the code and suggests improvements. Otherwise a policy is inferred automatically. You can refine it in a conversational loop before confirming. |
| **Dataset**             | Overmind uses your existing test data or generates diverse synthetic cases from the policy and agent description.                                                                                             |
| **Evaluation criteria** | Scoring rules are proposed for each output field. Policy constraints inform stricter scoring where relevant.                                                                                                  |

This produces two artifacts in `.overmind/agents/<name>/setup_spec/`:

- **eval_spec.json** — machine-readable evaluation spec used at runtime
- **policies.md** — human-readable policy document you maintain

Both are editable after generation. A preview is shown in chat before anything is saved.

### 4. Optimize (`/overmind-optimize-agent`)

Run `/overmind-optimize-agent <agent-name>` in chat. The skill drives the full optimization loop end to end. You can adjust settings before it starts or accept the defaults.

| Setting                      | Description                                                                                                                                                       |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Analyzer model**           | The strong model that diagnoses failures and generates code fixes.                                                                                                |
| **LLM-as-Judge**             | Optional semantic scoring alongside mechanical matching.                                                                                                          |
| **Iterations**               | Number of optimize, evaluate, accept/revert rounds. Default: 5.                                                                                                   |
| **Candidates per iteration** | How many variant fixes to generate per round. Each biases edits toward a different area, such as tool descriptions, core logic, input handling, or system prompt. |
| **Parallel execution**       | Run agent evaluations across multiple workers.                                                                                                                    |

#### What happens each iteration

1. **Run** the agent on every test case and collect traces and outputs.
1. **Score** outputs against the eval spec across weighted output fields.
1. **Diagnose** — the analyzer receives traces, scores, policy, and code. It identifies failure patterns and root causes.
1. **Generate** N candidate fixes, each targeting a different area of the code. If N≥3, the last candidate uses a separate diagnosis for diversity.
1. **Validate** — syntax checks, interface checks, and a smoke test on a small case subset.
1. **Evaluate** — surviving candidates are scored on the full dataset.
1. **Accept or revert** — the best candidate is kept only if it improves the score without regressing too many individual cases.

Advanced settings include regression thresholds, train/holdout splits to detect overfitting, early stopping patience, and diagnosis visibility controls.

### Multi-file agents

By default Overmind optimizes the single registered entry file. For agents split across multiple modules, it automatically resolves local imports, extracts individual functions and classes, and applies targeted edits back to the original files so your project structure stays intact.

## Agent policies

Policies are the foundation of meaningful optimization. They tell the optimizer what the agent should do, not just how it currently scores, preventing improvements that raise numbers but violate business rules.

A `policies.md` looks like this:

```markdown
# Agent Policy: Lead Qualification

## Purpose
Qualifies inbound sales leads by analyzing company data and inquiry content.

## Decision Rules
1. If the inquiry mentions "enterprise" or "custom pricing", classify as hot
2. Companies with 500+ employees get a minimum lead score of 60

## Constraints
- Never disqualify without checking company size
- Score and category must be consistent (hot = 70+, warm = 40-69, cold = <40)

## Priority Order
1. Accuracy of category classification
2. Score calibration
3. Reasoning quality

## Edge Cases
| Scenario | Expected Behaviour |
|---|---|
| Missing company name | Default to cold, note in reasoning |
| Competitor inquiry | Classify as cold, recommend nurture |

## Quality Expectations
- Reasoning should reference specific data points from the input
- Scores should be calibrated: hot leads 70-100, warm 40-69, cold 0-39
```

Policies feed into diagnosis prompts, code generation constraints, synthetic data generation, and LLM-as-Judge scoring so every stage of the pipeline respects your domain rules.

## Using your own data

Data files are JSON arrays where each element has an `input` and `expected_output`:

```json
[
  {
    "input": { "company_name": "Acme Corp", "inquiry": "Need enterprise pricing" },
    "expected_output": { "category": "hot", "lead_score": 85 }
  }
]
```

Place data files in your agent directory under `data/` and Overmind will detect them during setup. If you do not have data, Overmind generates realistic synthetic test cases using the policy and agent description.

<br />

______________________________________________________________________

<p align="center">
  <strong>Artifacts, traces, reports, and CLI reference</strong>
</p>

______________________________________________________________________

## Output

After optimization, results are saved under `.overmind/agents/<name>/`:

| Path                        | Description                                              |
| --------------------------- | -------------------------------------------------------- |
| `setup_spec/policies.md`    | Agent policy document                                    |
| `setup_spec/eval_spec.json` | Evaluation criteria with embedded policy                 |
| `setup_spec/dataset.json`   | Test dataset used for optimization                       |
| `experiments/best_agent.py` | The highest-scoring agent version for single-file agents |
| `experiments/best_agent/`   | All optimized files for multi-file agents                |
| `experiments/results.tsv`   | Score history for every iteration                        |
| `experiments/traces/`       | Detailed JSON traces of every agent run                  |
| `experiments/report.md`     | Summary report with scores and diffs                     |

Other paths under `.overmind/` do not all exist until you run the skills.

| Path                                            | Required?           | Notes                                                                                                                                      |
| ----------------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `agents.toml`                                   | Yes                 | Registry of agent names and `module:fn` entrypoints. Written by `/overmind-register-agent`.                                                |
| `.env`                                          | Optional            | API keys and model defaults from `overmind init`.                                                                                          |
| `agents/<name>/instrumented/`                   | Regenerated         | Full mirror of the project root minus skips like `.git` and `venv`. Put `.overmind` next to a small project root so this tree stays small. |
| `agents/<name>/run_state.json`                  | Written by optimize | Regression cases and run history across sessions.                                                                                          |
| `logs/overmind.log`                             | Auto                | Rotating CLI log from `setup_logging`.                                                                                                     |
| `agents/<name>/instrumented/.overmind_runners/` | Ephemeral           | Generated subprocess wrappers such as `_run_agent.py`; removed when the runner calls `cleanup()`; safe to delete manually.                 |

All provider keys and model defaults live in `.overmind/.env`. Per-agent `.env` files are not supported.

### Bundle scope and caps

For large repositories, the optimizer resolves a bounded import closure, defaulting to 24 files and 60k characters, and skips common paths such as `tests/`, `docs/`, and `.overmind/` using built-in rules plus optional `.overmindignore` and `.gitignore`.

After `/overmind-generate-spec-and-dataset` runs, `eval_spec.json` includes a `scope` block with two path lists, both relative to the project root:

- `optimizable_paths` — files the optimizer may edit.
- `read_only_paths` — files materialized into the bundle but enforced not-editable at accept time.

Project-level drops go in `.overmindignore`, not the spec. Inspect what will load without running an LLM:

```bash
overmind doctor my-agent
```

## CLI reference

The only terminal command in the normal workflow is `overmind init`. The rest of the workflow runs through Agent Skills in Cursor or Claude Code.

```text
overmind init                                        Configure API keys and models
overmind doctor <name>                               Diagnose bundle scope and eval spec (read-only)
```

Run `overmind <command> --help` for full flag documentation.
