---
name: overmind-optimise-agent
description: "Drive the full overmind optimize loop from the host coding agent (Cursor / Codex / Claude Code) instead of the in-process Python coder. Use when the user wants to optimize a registered Overmind agent, run iterative improvement, generate N candidate fixes per iteration in parallel, evaluate them, and keep the best — with early stopping."
metadata:
  version: "2.0"
  product: "Overmind"
---

# Optimise an Overmind Agent (host-agent driven)

Drives the same loop as `overmind optimize <agent>`, but with **you (the host coding agent)** doing the per-candidate code edits in parallel worktrees instead of the in-process `overmind/coding_agent/` loop.

The skill is built on top of `overmind optimize-step`, a JSON-in/JSON-out CLI that owns the heavy lifting (config persistence, baseline eval, diagnosis, candidate eval, acceptance gates, report rendering). You own loop control, parallel candidate fan-out, and early-stopping decisions.

## When this skill is needed

- *"Optimise `<agent-name>`"* / *"run the optimizer on `<agent-name>`"*
- *"Improve my agent through iterations using parallel candidates"*
- *"I want Cursor / Codex / Claude to be the coder during optimization, not the built-in one"*
- *"Run 5 iterations with 3 candidates each on `<agent-name>` and stop early if it stalls"*

## Prerequisites

1. The agent is registered in `.overmind/agents.toml` (`/overmind-register-agent`).
1. `setup_spec/eval_spec.json` and `setup_spec/dataset.json` exist under `.overmind/agents/<name>/` (`/overmind-generate-spec-and-dataset`).
1. Provider API keys are set in `.overmind/.env` or `.overmind/agents/<name>/.env`.

`overmind preflight` is **recommended but optional**. Running `/overmind-preflight` first catches and autonomously fixes most plumbing issues (missing env vars, weight drift, output schema mismatch, broken metrics, instrumentation gaps, harness bugs) so optimize can focus on *agent quality* instead of debugging infrastructure mid-loop. If the user wants to skip preflight, optimize will just run; any plumbing failure surfaces as an iteration error and the user can re-try after fixing it (or run preflight then).

```bash
overmind preflight status <agent-name>
# -> {"status": "green", ...}                  ← good to go
# -> {"status": "blocked_secrets", ...}        ← suggest /overmind-preflight
# -> {"status": "error", "error": "no_preflight_report"}  ← suggest /overmind-preflight or proceed if the user prefers
```

If preflight has run and is non-green, surface the message to the user but let them decide whether to fix it first or proceed.

Note: in this repo the `.overmind/` state directory may live at the project root **or** inside a sub-project (e.g. `new_examples/langextract/.overmind/`). Run all commands from the directory that contains the relevant `.overmind/`.

## Workflow

Copy this checklist into your response and check off each step as you complete it:

```
Optimization Progress:
- [ ] Step 1: Resolve agent + check prerequisites
- [ ] Step 2: Collect configuration
- [ ] Step 3: Initialize
- [ ] Step 4: Run baseline eval (classify zero baselines before proceeding)
- [ ] Step 5: Optimization loop (per iteration: diagnose → edit candidates → evaluate → accept/reject)
- [ ] Step 6: Render report
- [ ] Step 7: Summarize to user
```

### Step 1 — Resolve agent + check prerequisites

Look up the agent in `.overmind/agents.toml`. Confirm `setup_spec/eval_spec.json` and `setup_spec/dataset.json` exist. If either is missing, stop and refer the user to `/overmind-generate-spec-and-dataset`.

Then check the preflight status (advisory only):

```bash
overmind preflight status <agent-name>
```

| Response | Action |
|---|---|
| `status: green` or `green_with_quality_notes` | continue |
| `error: no_preflight_report` | mention `/overmind-preflight` is available but proceed if the user wants to skip it |
| `status: blocked_secrets` / `blocked_no_convergence` | surface `message` and `missing_secrets`, suggest `/overmind-preflight`, then ask the user whether to fix first or proceed anyway |

### Step 2 — Collect configuration via `AskQuestion`

First, ask:

> "How would you like to configure the optimizer?"
> Options: Quick setup (use all defaults) | Adjust parameters

If the user chooses **Quick setup**, apply every default below and skip straight to Step 3.

If the user chooses **Adjust parameters**, ask the **core** questions in one batch first; only ask the **advanced** ones if the user opts in. Build a `settings` dict from the answers.

#### Core (ask when adjusting parameters)

| Field | Default | Question |
|---|---|---|
| `iterations` | 5 | "How many optimization iterations?" |
| `candidates_per_iteration` | 3 | "Candidates per iteration (best-of-N parallel)?" |
| `early_stopping_patience` | 3 | "Stop after N iterations with no improvement? (0 = disabled)" |
| `analyzer_model` | `$ANALYZER_MODEL`, else `anthropic/claude-sonnet-4-20250514` | "Which model should diagnose failures and design candidate plans?" |
| `enable_judge` (yes/no) | no | "Enable LLM-as-Judge scoring? (~10% extra eval cost)" |

If `enable_judge: yes`, set `llm_judge_model = analyzer_model` (or ask for a specific judge model).

#### Advanced (only if user opts in)

`runs_per_eval` (1), `regression_threshold` (0.35), `holdout_ratio` (0.2), `holdout_enforcement` (true), `diagnosis_case_fraction` (0.7), `cross_run_persistence` (true), `failure_clustering` (true), `adaptive_focus` (true), `max_workers` (5), `smoke_test_cases` (2), `codegen_max_steps` (50), `model_backtesting` (false).

### Step 3 — Initialise

```bash
echo '<settings JSON>' | overmind optimize-step init <agent-name>
# add --overwrite if a prior skill_state.json exists and the user agreed
```

Parse the JSON envelope. On `status: error, error: state_already_exists`, ask the user whether to start fresh and re-run with `--overwrite`. On `missing_eval_spec` / `missing_dataset`, stop and refer them to `/overmind-generate-spec-and-dataset`. The response includes a `preflight` field summarising the latest preflight status (or `null` if it never ran) — surface this to the user but do not block on it.

Capture `STATE_PATH` from the response — every subsequent step uses `--state $STATE_PATH`.

### Step 4 — Baseline (with zero-baseline classification)

```bash
overmind optimize-step baseline --state $STATE_PATH
```

Returns `{baseline_score, train_size, holdout_size, working_path, ...}`. Tell the user the baseline.

**If the baseline score is exactly 0, stop and investigate before proceeding.** Do not assume optimization should continue from zero — this usually signals a setup problem, not a performance problem.

Use a focused investigation subagent when the baseline artifacts or codebase are large. The investigation subagent should inspect the baseline artifacts, eval spec, dataset, registered entrypoint, and score reports, then return a concise classification.

Classify the zero baseline as one of:

| Classification | Meaning | What to do |
|---|---|---|
| **Setup failure** | Agent can't import/run — credentials missing, entrypoint broken, imports fail | Fix registration or credentials; do not optimize |
| **Scoring failure** | Agent runs but the eval spec can't score its outputs — field mismatches, wrong types, `string` instead of `text` | Fix the eval spec; do not optimize |
| **Dataset mismatch** | Dataset inputs don't match the registered callable, or expected outputs don't align with evaluator fields | Fix the dataset; do not optimize |
| **Genuine performance failure** | Agent runs and scores correctly, but fails every case | Proceed with optimization |
| **Inconclusive** | Not enough evidence to classify | Investigate further before proceeding |

Proceed to optimization **only** if the zero baseline is classified as genuine performance failure, or the user explicitly asks to optimize anyway despite the risk.

### Step 5 — Optimization loop

For `i` in `1..iterations`:

#### 5a. Diagnose + materialise N worktrees

```bash
overmind optimize-step diagnose --state $STATE_PATH --iteration $i
```

Returns:

```json
{
  "status": "ok",
  "candidates": [
    {
      "candidate_id": "c0",
      "worktree": "<absolute path>",
      "prompt_path": "<worktree>/PROMPT.md",
      "plan_path": "<worktree>/plan.json",
      "entry_file": "agent.py",
      "entry_path": "<worktree>/agent.py",
      "method": "plan(tool_description)",
      "focus_area": "tool_description"
    }
  ]
}
```

Each worktree is a proper **git worktree** (created via `git worktree add --detach`) populated with the current best agent files plus a `PROMPT.md` with full edit instructions for that candidate.

If the envelope has `status: "warn"` and a `diagnose_warning` block, **stop the loop and report to the user**. This means the analyzer LLM call failed (most often a missing API key for the analyzer model). Do not silently fall back to manual edits — surface `diagnose_warning.last_error` and `diagnose_warning.hint`, ask the user to fix env / model config, then re-run.

#### 5b. Spawn N parallel sub-coding-agents

Detect the host once, at skill start, and use the right spawn method below. **Always** background the spawn and wait on all agents before evaluating.

**Cursor (preferred):**

```
For each candidate, call the Task tool with:
  subagent_type: "best-of-n-runner"
  description: "Apply candidate <candidate_id> edits"
  prompt: |
    You are an expert coding agent improving an AI agent codebase.

    Working directory: <worktree>

    1. Read PROMPT.md in <worktree> — it contains full instructions, the
       diagnosis, and the specific edits to make.
    2. Read plan.json in <worktree> for the structured diagnosis with focus
       area, root cause, and suggested changes.
    3. Apply the edits to the source files IN PLACE inside <worktree>.
       Do NOT create copies of files. Do NOT move files outside <worktree>.

    Critical rules (violations cause automatic test rejection):
    - Do NOT hardcode values, responses, or answers for specific inputs from
      the diagnosis or test results.
    - Do NOT add if/elif/match branches that pattern-match on specific field
      values or example data.
    - Do NOT add lookup tables keyed by example input values.
    - Prefer general improvements: better prompt wording, smarter parsing,
      cleaner logic, new helper functions.
    - Read before editing. Check callers/callees when modifying a function.
    - Use grep/glob to locate code you are unsure about.
    - Prefer find-and-replace edits over full-file rewrites.
    - Re-read a file after a non-trivial edit to verify correctness.
    - Do NOT add comments narrating your changes.

    When finished, print OPTIMIZE_DONE.
  run_in_background: true
```

Collect the returned task IDs and wait for all to complete.

**Codex CLI:**

```bash
PIDS=()
for c in "${CANDIDATES[@]}"; do
  WT="$(echo "$c" | jq -r .worktree)"
  ( cd "$WT" && codex exec --json -p "$(cat PROMPT.md)" > done.json 2> codex.err ) &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
```

**Claude Code:**

```bash
PIDS=()
for c in "${CANDIDATES[@]}"; do
  WT="$(echo "$c" | jq -r .worktree)"
  ( cd "$WT" && claude -p "$(cat PROMPT.md)" > done.json 2> claude.err ) &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
```

If the host cannot be detected, fall back to **sequential** edits: switch to each worktree and apply edits yourself one at a time.

#### 5c. Evaluate each candidate

```bash
for c in $CANDIDATES_JSON; do
  overmind optimize-step evaluate \
    --state $STATE_PATH \
    --iteration $i \
    --candidate-id $(echo "$c" | jq -r .candidate_id) \
    --candidate-dir $(echo "$c" | jq -r .worktree)
done
```

Each evaluate writes `score.json` into the worktree. Build a `candidate_results.json` array. Use **absolute paths** for every path field to avoid resolution errors across host environments:

```json
[
  {
    "candidate_id": "c0",
    "candidate_dir": "/abs/path/to/.overmind/agents/<name>/experiments/iter_001_c0",
    "entry_path": "/abs/path/to/.overmind/agents/<name>/experiments/iter_001_c0/agent.py",
    "score_path": "/abs/path/to/.overmind/agents/<name>/experiments/iter_001_c0/score.json"
  }
]
```

#### 5d. Acceptance + early-stopping

```bash
overmind optimize-step accept \
  --state $STATE_PATH \
  --iteration $i \
  --candidate-results candidate_results.json
```

Returns `{decision: "accept" | "reject" | "all_crashed", winner, all_scores, best_score, stall_count, early_stop}`.

If `early_stop: true`, **break the loop** and tell the user "Early stopping fired after $stall_count stalls."

### Step 6 — Render the report

```bash
overmind optimize-step report --state $STATE_PATH
```

Returns `{report_path, best_score, baseline_score, iterations_completed, early_stopping_triggered}`.

### Step 7 — Summarise to the user

Tell them:

- Baseline → final score (delta).
- Iterations completed; whether early-stopping fired.
- Path to `report.md` and the best-agent working file.
- The N candidate worktrees per iteration are under `experiments/iter_NNN_cI/` for inspection.

## Progress updates

Give the user a concise update at each of these milestones (do not wait until the end):

- Prerequisites checked.
- Settings initialized and `STATE_PATH` captured.
- Baseline score computed.
- Zero-baseline investigation result, if applicable.
- Candidate worktrees materialized for each iteration.
- Candidate edits completed.
- Candidate scores and acceptance decision.
- Early stopping triggered, if applicable.
- Final report rendered.

## Candidate edit guardrails

Reject or repair candidate work before evaluation when it violates these hard rules:

- It edits generated `.overmind` state or setup artifacts during optimization.
- It hardcodes exact dataset inputs, expected answers, IDs, or diagnosis examples.
- It adds lookup tables keyed by example values.
- It adds brittle `if`, `elif`, `match`, or regex branches that exist only to match known test examples.
- It deletes core agent behavior rather than improving it.
- It moves files out of the worktree.
- It modifies provider secrets or prints secret values.
- It edits the registered Overmind entrypoint file in a way that breaks the callable contract.

Prefer to let evaluation catch quality regressions, but do not evaluate candidates that violate hardcoding, state-mutation, secret-handling, or worktree-boundary rules.

## Using subagents

Use subagents whenever they improve reliability or parallelism:

- **Candidate subagents**: Spawn one sub-coding-agent per candidate worktree when the host supports background tasks.
- **Investigation subagents**: Spawn a focused codebase/debugging subagent for zero baselines, confusing evaluator failures, or analyzer warnings that require artifact inspection.
- **Review subagents**: Spawn a review subagent when candidate patches are large or touch shared behavior before evaluation.

Do not spawn subagents that edit the same worktree concurrently.

## Useful inspection commands

```bash
overmind optimize-step status --state $STATE_PATH
# -> {status: ok, state: {...}, early_stop: bool}
```

## Handling common failures

| Problem | Fix |
|---|---|
| State already exists | Ask whether to resume or start fresh. Use overwrite only with explicit approval. |
| Missing eval spec | Stop and run or recommend `/overmind-generate-spec-and-dataset` |
| Missing dataset | Stop and run or recommend `/overmind-generate-spec-and-dataset` |
| Preflight missing or not green | Mention `/overmind-preflight` to the user; proceed if they want to skip it |
| Zero baseline | Classify as setup / scoring / dataset / genuine failure before proceeding (see Step 4) |
| Analyzer warning | Stop and report the warning's last error and hint; fix env / model config, then re-run |
| Candidate worktree missing | Mark that candidate failed and continue evaluating the others |
| All candidates crash | Report the iteration result, then follow the accept-step state about whether to continue |
| No improvement for patience window | Stop early when the accept step reports early stopping |

## Build status

The current implementation is a working MVP with **simplified acceptance gates**: the highest-scoring candidate wins iff it strictly beats the current best. The following are **not yet ported** from the in-process `Optimizer.run()`:

- Cross-run regression suite checks (`_check_regression_suite`)
- Holdout enforcement / blended scoring (`_rollback_to_best_snapshot`)
- Complexity penalty + max code growth ratio
- Smoke test (catastrophic-failure quick filter)
- Re-eval for close calls (`reeval_margin`)
- ApiReporter / OTLP per-iteration spans

The diagnosis side **does** carry over: failure clustering, adaptive focus weights, cross-run failed/successful change history, and the same `analyzer_model` prompts as the original loop.

## Quick smoke-test recipe

```bash
cd <project root containing .overmind/>

echo '{"iterations": 2, "candidates_per_iteration": 2,
       "early_stopping_patience": 1,
       "analyzer_model": "anthropic/claude-sonnet-4-20250514"}' \
  | overmind optimize-step init <agent-name> --overwrite

STATE=.overmind/agents/<agent-name>/experiments/skill_state.json

overmind optimize-step status   --state $STATE
overmind optimize-step baseline --state $STATE
overmind optimize-step diagnose --state $STATE --iteration 1
# ... edit the c0/c1 worktrees yourself or fan out sub-agents ...
overmind optimize-step evaluate --state $STATE --iteration 1 --candidate-id c0 \
    --candidate-dir .overmind/agents/<agent-name>/experiments/iter_001_c0
overmind optimize-step evaluate --state $STATE --iteration 1 --candidate-id c1 \
    --candidate-dir .overmind/agents/<agent-name>/experiments/iter_001_c1
overmind optimize-step accept   --state $STATE --iteration 1 \
    --candidate-results candidate_results.json
overmind optimize-step report   --state $STATE
```
