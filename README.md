# Overmind

Automatically optimize your AI agent's prompts, tool definitions, model selection, and pipeline logic through structured experimentation.

**Documentation:** [Overmind guide](https://docs.overmindlab.ai/guides/overmind_optimizer/)

## What it does

Overmind runs your agent against a test dataset, traces every LLM call and
tool invocation, scores the outputs, and uses a strong reasoning model to
generate concrete improvements. Changes that raise the score are kept; the rest
are reverted. After several rounds you get a measurably better agent — without
manual prompt tweaking.

What makes Overmind different is **policy-driven optimization**. You define the
decision rules, constraints, and expectations your agent must follow, and those
policies guide every stage: evaluation criteria, test data synthesis,
optimization diagnosis, and scoring.

### What gets optimized

- **System prompts** — more precise instructions, output format enforcement
- **Tool descriptions** — clearer parameters, better usage guidance
- **Model selection** — find the right quality/cost tradeoff
- **Agent logic** — tool-call ordering, iteration limits, output parsing
- **Policy compliance** — alignment with your domain rules and constraints

## Get started

This walks you through the full workflow — from installation to your first
optimized agent. The whole process takes about 10 minutes.

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), and API keys
for at least one LLM provider (OpenAI, Anthropic).

### 1. Install

```bash
uv tool install overmind

# or dev install
git clone https://github.com/overmind-core/overmind
cd overmind
uv tool install -e .
```

> Using `uv run` instead? All commands work as `uv run overmind <command>`
> after `uv sync`.

### 2. Initialize the project

```bash
cd your-agent-project/
overmind init
```

This creates `.overmind/` in your project root and prompts for API keys and
default models. Safe to re-run anytime.

### 3. Register your agent

```bash
overmind agent register my-agent agents.my_agent:run
```

Point Overmind at the Python function it should call. The function receives an
input dict and must return a dict:

```python
def run(input_data: dict) -> dict:
    # your agent logic
    return {"response": result}
```

**Framework-based agents** (Google ADK, LangChain, CrewAI, etc.) often don't
expose a simple callable. Overmind detects this and offers to auto-generate an
entrypoint wrapper — no manual boilerplate needed. During registration it will
also collect any API keys your agent needs.

### 4. Validate the entrypoint (optional)

```bash
overmind agent validate my-agent --data tests/sample.json
```

Runs the first case from your test data through the agent to make sure the
entrypoint works end-to-end before investing time in setup. For the bundled
examples, paths look like `examples/<folder>/data/seed.json` — see
[`examples/README.md`](examples/README.md). **Exception:** `contract_extractor`
must be registered and run from `examples/contract_extractor/` (see that
folder’s README).

### 5. Set up evaluation criteria

```bash
overmind setup my-agent
# or with seed data (JSON file or directory of *.json):
overmind setup my-agent --data data/seed.json
# or with an existing policy document:
overmind setup my-agent --policy docs/my_policy.md
# or non-interactive:
overmind setup my-agent --fast
```

An interactive flow that analyzes your code, defines policies, builds (or
imports) a test dataset, and generates scoring criteria.

### 6. Optimize

```bash
overmind optimize my-agent
```

Iteratively runs your agent, scores outputs, diagnoses failures, and generates
code improvements. Changes that raise the score are kept; the rest are reverted.

## Cursor Agent Skills

If you use Cursor, this repository includes **Agent Skills** under
[`.cursor/skills/`](.cursor/skills/) — structured playbooks the Cursor agent
follows to drive Overmind workflows without you typing CLI commands manually.

### Available skills

| Skill (slash command)                | Path                                                                                                      | What it does                                                                                                                                    |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `/overmind-register-agent`           | [`overmind-register-agent/SKILL.md`](.cursor/skills/overmind-register-agent/SKILL.md)                     | Registers an agent in `.overmind/agents.toml` — discovers the entrypoint, derives `module:function`, runs registration, and scaffolds env vars. |
| `/overmind-generate-dataset`         | [`overmind-generate-dataset/SKILL.md`](.cursor/skills/overmind-generate-dataset/SKILL.md)                 | Generates a synthetic `dataset.json` via persona-driven LLM generation, with schema validation and smoke-testing.                               |
| `/overmind-generate-policy-and-eval` | [`overmind-generate-policy-and-eval/SKILL.md`](.cursor/skills/overmind-generate-policy-and-eval/SKILL.md) | Creates or repairs `setup_spec/policies.md` and `setup_spec/eval_spec.json` (input/output schemas, weights, domain rules).                      |
| `/overmind-optimize-agent`           | [`overmind-optimize-agent/SKILL.md`](.cursor/skills/overmind-optimize-agent/SKILL.md)                     | Runs the optimization loop with the Cursor agent applying candidate edits in parallel git worktrees via `overmind optimize-step`.               |

### How to use the skills

The skills replace the interactive `overmind` CLI flow. Run them in order inside Cursor chat:

**1. Initialize** (one-time, run in the terminal — not a skill):

```bash
overmind init
```

This sets up `.overmind/` and writes your API keys to `.overmind/.env`.

**2. Register your agent** — type in Cursor chat:

```
/overmind-register-agent path/to/your/agent.py
```

The agent will ask for the agent name, detect the entrypoint function, run
registration, and create a `.env` placeholder file for your credentials.

**3. Generate a dataset** — type in Cursor chat:

```
/overmind-generate-dataset <agent-name>
```

Generates `dataset.json` under `.overmind/agents/<name>/setup_spec/`. Skip
this step if you already have your own test data.

**4. Generate the policy and eval spec** — type in Cursor chat:

```
/overmind-generate-policy-and-eval <agent-name>
```

Produces `policies.md` and `eval_spec.json` through an interactive elicitation
of domain rules, constraints, and scoring criteria.

**5. Optimize** — either run the CLI directly or use the host-driven skill:

```bash
# CLI (built-in optimizer)
overmind optimize <agent-name>

# or via Cursor chat (host-agent-driven loop)
# /overmind-optimize-agent <agent-name>
```

> Skills assume commands are run from the directory that contains the relevant
> `.overmind/` folder. All API keys and default models must be configured via
> `overmind init` before invoking any skill.

## How it works

### 1. Initialize (`overmind init`)

Configure API keys and default models. Writes `.overmind/.env` in the current
directory. Safe to re-run.

### 2. Register your agent (`overmind agent register`)

Point Overmind at the Python function it should call for each test case:

```bash
overmind agent register <name> <module:function>
```

The module path is resolved relative to the project root. Your function
receives an input dict and must return a dict.

Other registry commands:

| Command                                        | Description                                        |
| ---------------------------------------------- | -------------------------------------------------- |
| `overmind agent list`                          | List all registered agents                         |
| `overmind agent show <name>`                   | Show registration details and pipeline status      |
| `overmind agent update <name> <mod:fn>`        | Update the entrypoint (e.g. after renaming a file) |
| `overmind agent remove <name>`                 | Remove from registry and instrumented copy         |
| `overmind agent validate <name> --data <path>` | Run the first test case to verify the entrypoint   |

### 3. Setup (`overmind setup`)

An interactive flow that prepares everything the optimizer needs:

| Phase                   | What happens                                                                                                                                                                                                                                 |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Agent analysis**      | An LLM reads your agent code to detect the input/output schema, tools, and decision logic.                                                                                                                                                   |
| **Policy generation**   | If you pass `--policy`, your document is analyzed against the code and improvements are suggested. Otherwise, a policy is inferred from the code automatically. You can refine either version in a conversational loop until you approve it. |
| **Dataset**             | Overmind either uses your existing test data or generates diverse synthetic cases based on the policy and agent description.                                                                                                                 |
| **Evaluation criteria** | Scoring rules are proposed for each output field. Policy constraints inform stricter scoring where relevant. You can accept, refine, or edit manually.                                                                                       |

Setup produces two artifacts in `.overmind/agents/<name>/setup_spec/`:

- **eval_spec.json** — machine-readable evaluation spec (used at runtime)
- **policies.md** — human-readable policy document you maintain

Both are editable after generation.

| Flag            | Description                                                                                      |
| --------------- | ------------------------------------------------------------------------------------------------ |
| `--fast`        | Skip all prompts. Requires `ANALYZER_MODEL` and `SYNTHETIC_DATAGEN_MODEL` in `.env`.             |
| `--data PATH`   | JSON seed dataset file or directory of `*.json` files (optional; wizard can pick data instead).  |
| `--policy PATH` | Provide an existing policy document. Overmind analyzes it against agent code and suggests edits. |

### 4. Optimize (`overmind optimize`)

The iterative optimization loop. You configure a few settings interactively
(or use `--fast` for defaults):

| Setting                      | Description                                                                                                                                                                                                |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Analyzer model**           | The strong model that diagnoses failures and generates code fixes.                                                                                                                                         |
| **LLM-as-Judge**             | Optional semantic scoring alongside mechanical matching (adds ~10% eval cost).                                                                                                                             |
| **Iterations**               | Number of optimize → evaluate → accept/revert rounds (default: 5).                                                                                                                                         |
| **Candidates per iteration** | How many variant fixes to generate per round (best-of-N). Each biases edits toward a different area — tool descriptions, core logic, input handling, system prompt. Higher N improves odds but costs more. |
| **Parallel execution**       | Run agent evaluations across multiple workers.                                                                                                                                                             |

#### What happens each iteration

1. **Run** the agent on every test case and collect traces + outputs.
1. **Score** outputs against the eval spec (0–100 across dynamic dimensions).
1. **Diagnose** — the analyzer receives traces, scores, policy, and code. It identifies failure patterns and root causes.
1. **Generate** N candidate fixes, each targeting a different area of the code. If N≥3, the last candidate uses a separate diagnosis for diversity.
1. **Validate** — syntax checks, interface checks, and a smoke test on a small case subset.
1. **Evaluate** — surviving candidates are scored on the full dataset.
1. **Accept or revert** — the best candidate is kept only if it improves the score without regressing too many individual cases.

Advanced settings (available during interactive config) include regression
thresholds, train/holdout splits to detect overfitting, early stopping
patience, and diagnosis visibility controls.

| Flag     | Description                                                           |
| -------- | --------------------------------------------------------------------- |
| `--fast` | Skip all prompts. Requires `ANALYZER_MODEL` in `.env`. Uses defaults. |

### Multi-file agents

By default Overmind optimizes the single registered entry file. For agents
split across multiple modules, it automatically resolves local imports,
extracts individual functions and classes, and applies targeted edits back to
the original files — so your project structure stays intact.

## Agent policies

Policies are the foundation of meaningful optimization. They tell the optimizer
*what* the agent should do, not just how it currently scores — preventing
improvements that raise numbers but violate business rules.

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
| Scenario             | Expected Behaviour                    |
|----------------------|---------------------------------------|
| Missing company name | Default to cold, note in reasoning    |
| Competitor inquiry   | Classify as cold, recommend nurture   |

## Quality Expectations
- Reasoning should reference specific data points from the input
- Scores should be calibrated: hot leads 70-100, warm 40-69, cold 0-39
```

Policies feed into diagnosis prompts, code generation constraints, synthetic
data generation, and LLM-as-Judge scoring — so every stage of the pipeline
respects your domain rules.

## Using your own data

Data files are JSON arrays where each element has an `input` and
`expected_output`:

```json
[
  {
    "input": { "company_name": "Acme Corp", "inquiry": "Need enterprise pricing" },
    "expected_output": { "category": "hot", "lead_score": 85 }
  }
]
```

Place data files in your agent directory under `data/` and Overmind will
detect them during setup. If you don't have data, Overmind generates realistic
synthetic test cases using the policy and agent description.

## Output

After optimization, results are saved under `.overmind/agents/<name>/`:

| Path                        | Description                                  |
| --------------------------- | -------------------------------------------- |
| `setup_spec/policies.md`    | Agent policy document                        |
| `setup_spec/eval_spec.json` | Evaluation criteria with embedded policy     |
| `setup_spec/dataset.json`   | Test dataset used for optimization           |
| `experiments/best_agent.py` | The highest-scoring agent version            |
| `experiments/best_agent/`   | All optimized files (multi-file agents only) |
| `experiments/results.tsv`   | Score history for every iteration            |
| `experiments/traces/`       | Detailed JSON traces of every agent run      |
| `experiments/report.md`     | Summary report with scores and diffs         |

Other paths under `.overmind/` (not all exist until you run commands):

| Path                                            | Required?              | Notes                                                                                                                                                                                                    |
| ----------------------------------------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `agents.toml`                                   | Yes for `overmind` CLI | Registry of agent names and `module:fn` entrypoints.                                                                                                                                                     |
| `.env`                                          | Optional               | API keys and model defaults from `overmind init`.                                                                                                                                                        |
| `agents/<name>/.env`                            | Optional               | Per-agent overrides (written by setup when you save keys).                                                                                                                                               |
| `agents/<name>/instrumented/`                   | Regenerated            | Full **mirror of the project root** (everything under the directory that contains `.overmind/`, minus skips like `.git`, `venv`). Put `.overmind` next to a small project root so this tree stays small. |
| `agents/<name>/run_state.json`                  | Written by optimize    | Regression cases and run history across sessions.                                                                                                                                                        |
| `logs/overmind.log`                             | Auto                   | Rotating CLI log from `setup_logging`.                                                                                                                                                                   |
| `agents/<name>/instrumented/.overmind_runners/` | Ephemeral              | Generated subprocess wrappers (`_run_agent.py`, etc.); removed when the runner calls `cleanup()`; safe to delete manually.                                                                               |

### Bundle scope and caps

For large repositories, the optimizer resolves a **bounded** import closure (defaults: 24 files, 60k characters) and skips common paths (`tests/`, `docs/`, `.overmind/`, etc.) using built-in rules plus optional `.overmindignore` / `.gitignore`.

After `overmind setup`, `eval_spec.json` may include a `scope` block (`optimizable_paths`, `context_paths`, `exclude_paths` as globs relative to the project root). Inspect what will load without running an LLM:

```bash
overmind doctor my-agent
```

One-off overrides:

```bash
overmind optimize my-agent --scope "myagent/prompts/**/*.py" --max-files 32 --max-chars 80000
overmind setup my-agent --scope "agents/core/*.py"   # hints for the analyzer
```

## CLI reference

```
overmind init                                        Configure API keys and models
overmind agent register <name> <mod:fn>              Register an agent
overmind agent list                                  List registered agents
overmind agent show <name>                           Show agent status
overmind agent update <name> <mod:fn>                Update entrypoint
overmind agent remove <name>                         Remove from registry
overmind agent validate <name> --data <path>         Run first test case to verify entrypoint
overmind setup <name> [--fast] [--data PATH] [--policy PATH]  Analyze agent, build eval spec
overmind optimize <name> [--fast] [--scope GLOB] [--max-files N] [--max-chars N]  Run optimization loop
overmind doctor <name>                               Diagnose bundle scope and eval spec (read-only)
overmind sync [name]                                 Sync local setup artifacts to Overmind
overmind sync-optimize [name]                        Sync local optimize artifacts to Overmind
```

Run `overmind <command> --help` for full documentation on any command.
