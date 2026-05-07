______________________________________________________________________

## name: optimise-agent description: Drive the full `overmind optimize` loop from the host coding agent (Cursor / Codex / Claude Code) instead of the in-process Python coder. Use when the user wants to optimize a registered Overmind agent, run iterative improvement, generate N candidate fixes per iteration in parallel, evaluate them, and keep the best — with early stopping. The skill collects all configuration via `AskQuestion`, then drives the optimization loop by calling the `overmind optimize-step` JSON CLI between phases. Heavy lifting (subprocess-isolated agent runs, scoring, regression gating) stays in Python; per-candidate code edits are delegated to parallel sub-coding-agents in git worktrees. disable-model-invocation: true

# Optimise an Overmind Agent (host-agent driven)

Drives the same loop as `overmind optimize <agent>`, but with **you (the host coding agent)** doing the per-candidate code edits in parallel worktrees instead of the in-process `overmind/coding_agent/` loop.

The skill is built on top of `overmind optimize-step`, a JSON-in/JSON-out CLI that owns the heavy lifting (config persistence, baseline eval, diagnosis, candidate eval, acceptance gates, report rendering). You own loop control, parallel candidate fan-out, and early-stopping decisions.

## When this skill is needed

- *"Optimise `<agent-name>`"* / *"run the optimizer on `<agent-name>`"*
- *"Improve my agent through iterations using parallel candidates"*
- *"I want Cursor / Codex / Claude to be the coder during optimization, not the built-in one"*
- *"Run 5 iterations with 3 candidates each on `<agent-name>` and stop early if it stalls"*

## Prerequisites

1. The agent is registered in `.overmind/agents.toml` (`/register-agent` skill or `overmind agent register`).
1. `setup_spec/eval_spec.json` and `setup_spec/dataset.json` exist under `.overmind/agents/<name>/`. If not, run `/generate-policy-and-eval` and `/generate-dataset` first.
1. Provider API keys are set in `.overmind/.env` or `.overmind/agents/<name>/.env`.

If any prerequisite is missing, **stop** and tell the user which one to satisfy. Do not attempt to proceed.

Note: in this repo the `.overmind/` state directory may live at the project root **or** inside a sub-project (e.g. `new_examples/langextract/.overmind/`). Run all commands from the directory that contains the relevant `.overmind/`.

## Workflow

### Step 1 — Resolve agent + check prerequisites

Look up the agent in `.overmind/agents.toml`. Confirm `setup_spec/eval_spec.json` and `setup_spec/dataset.json` exist. Stop with a clear message if anything is missing.

### Step 2 — Collect configuration via `AskQuestion`

Use `AskQuestion`. Ask the **core** questions in one batch first; only ask the **advanced** ones if the user opts in. Build a `settings` dict from the answers.

#### Core (always ask)

| Field                      | Default                                                      | Question                                                           |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------ |
| `iterations`               | 5                                                            | "How many optimization iterations?"                                |
| `candidates_per_iteration` | 3                                                            | "Candidates per iteration (best-of-N parallel)?"                   |
| `early_stopping_patience`  | 3                                                            | "Stop after N iterations with no improvement? (0 = disabled)"      |
| `analyzer_model`           | `$ANALYZER_MODEL`, else `anthropic/claude-sonnet-4-20250514` | "Which model should diagnose failures and design candidate plans?" |
| `enable_judge` (yes/no)    | no                                                           | "Enable LLM-as-Judge scoring? (~10% extra eval cost)"              |

If `enable_judge: yes`, set `llm_judge_model = analyzer_model` (or ask for a specific judge model).

#### Advanced (only if user opts in)

`runs_per_eval` (1), `regression_threshold` (0.35), `holdout_ratio` (0.2), `holdout_enforcement` (true), `diagnosis_case_fraction` (0.7), `cross_run_persistence` (true), `failure_clustering` (true), `adaptive_focus` (true), `max_workers` (5), `smoke_test_cases` (2), `codegen_max_steps` (50), `model_backtesting` (false).

### Step 3 — Initialise

```bash
echo '<settings JSON>' | overmind optimize-step init <agent-name>
# add --overwrite if a prior skill_state.json exists and the user agreed
```

Parse the JSON envelope. On `status: error, error: state_already_exists`, ask the user whether to start fresh and re-run with `--overwrite`. On `missing_eval_spec` / `missing_dataset`, stop and refer them to the appropriate skill.

Capture `STATE_PATH` from the response — every subsequent step uses `--state $STATE_PATH`.

### Step 4 — Baseline

```bash
overmind optimize-step baseline --state $STATE_PATH
```

Returns `{baseline_score, train_size, holdout_size, working_path, ...}`. Tell the user the baseline.

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
      "entry_file": "test.py",
      "entry_path": "<worktree>/test.py",
      "method": "plan(tool_description)",
      "focus_area": "tool_description"
    },
    ...
  ]
}
```

Each worktree is already populated with the current best agent files plus a `PROMPT.md` containing the full edit instructions for that candidate.

If the envelope has `status: "warn"` and a `diagnose_warning` block, **stop the loop and report to the user**. This means the analyzer LLM call failed (most often a missing `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` for the analyzer model) and `n_candidates` collapsed to a single empty placeholder. Do not silently fall back to manual edits — surface `diagnose_warning.last_error` and `diagnose_warning.hint`, ask the user to fix env / model config, then re-run.

#### 5b. Spawn N parallel sub-coding-agents

Detect the host once, at skill start, and use the right spawn method below. **Always** background the spawn and `wait` on all PIDs before evaluating.

**Cursor (preferred):**

```
For each candidate, call the Task tool with:
  subagent_type: "best-of-n-runner"
  description: "Apply candidate <candidate_id> edits"
  prompt: "Read the file PROMPT.md in <worktree> and follow it exactly. Edit files in this worktree. Stop when done."
  run_in_background: true
```

Collect the returned task IDs and wait for all to complete (poll with `AwaitShell`/`AwaitTask` until each terminates).

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

If you cannot detect the host, fall back to **sequential** edits: for each candidate, switch to its worktree and apply edits yourself one at a time. This is slower but always works.

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

Each evaluate writes `score.json` into the worktree. Build a `candidate_results.json` array listing each candidate's `candidate_id`, `candidate_dir`, `entry_path`, and `score_path`.

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
- The N candidate worktrees per iteration are under `experiments/iter_NNN_cI/` if they want to inspect them.

## Useful inspection commands

```bash
overmind optimize-step status --state $STATE_PATH
# -> {status: ok, state: {...}, early_stop: bool}
```

## Build status

The current implementation is a working MVP with **simplified acceptance gates**: the highest-scoring candidate wins iff it strictly beats the current best. The following are **not yet ported** from the in-process `Optimizer.run()` (they will be added in a follow-up):

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
