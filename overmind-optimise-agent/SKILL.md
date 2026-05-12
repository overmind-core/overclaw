---
name: overmind-optimise-agent
description: "Drive the full host-agent controlled overmind optimize loop for a registered Overmind agent. Use when the user wants to optimise or optimize an agent, run iterative improvement, fan out multiple candidate edits in parallel worktrees, evaluate candidates with overmind optimize-step, accept the best improvement, stop early on stalls, and render an optimization report."
metadata:
  version: "2.5"
  product: "Overmind"
---

# Optimise an Overmind Agent

Use this skill to drive the `overmind optimize-step` JSON CLI from a host coding agent such as Cursor, Codex, Claude Code, or another code-editing agent. Overmind owns state, baseline evaluation, diagnosis, worktree materialisation, candidate evaluation, acceptance gates, and report rendering. The host coding agent owns loop control, **multi-agent fan-out** (several subagents across iterations and candidates), parallel candidate fan-out, per-candidate code edits, and early stopping.

**Audience:** Fan-out, worktree editing, and `optimize-step` orchestration below are instructions for **you, the coding agent executing this skill** — not steps you assign to the human user unless the host truly cannot spawn subagents or run background work (then say so explicitly).

This skill optimizes the agent files selected by the existing Overmind eval spec and optimizer scope. It should not add extra setup restrictions that prevent `overmind optimize-step` from running.

## When to use this skill

- The agent is registered, `eval_spec.json` and `dataset.json` exist, and the user wants iterative improvement via `overmind optimize-step`.
- You need a host-controlled loop: baseline → diagnose → candidate worktrees → evaluate → accept → report.

## When not to use this skill

- Registration or the Overmind harness is missing — use `/overmind-register-agent`.
- Eval spec or dataset is missing or structurally wrong — use `/overmind-generate-spec-and-dataset` first.
- Do **not** use this skill alone to “repair” broken registration or spec unless `optimize-step` itself reports the failure and the fix is clearly inside the optimization CLI’s remit.

## Example (abbreviated)

User: *“Optimize `hotel-agent` with defaults.”* You confirm the configuration branch, run `ANALYZER_MODEL` preflight, `init` with heredoc JSON, `baseline`, then each iteration: `diagnose` → parallel subagents per candidate worktree (or sequential fallback) → `evaluate` each candidate → `accept` with `candidate_results.json` → early-stop check → `report`.

## Operating principles

- **Codebase-derived optimization**: Use the registered agent, eval spec, dataset, policy, diagnosis output, worktree prompts, tests, examples, and codebase context as the source of truth. Do not rely on broad user elicitation for optimization strategy.
- **Prerequisites first**: Stop if the agent is not registered, `eval_spec.json` is missing, `dataset.json` is missing, or required provider configuration is absent.
- **Correct project root**: Run all Overmind commands from the directory that contains the relevant `.overmind/`. Some repositories contain nested projects with their own `.overmind/`.
- **Respect optimizer scope**: Candidate edits should target files selected by the eval spec and candidate prompt. Do not override the project’s configured optimization scope with additional skill-level assumptions.
- **Preserve invocation stability (default)**: Treat the registered Overmind **harness entrypoint** as **fixed** — same default as `/overmind-register-agent` and `/overmind-generate-spec-and-dataset`. **Legacy compatibility:** If the **existing** `eval_spec.json` already includes that file in optimizer scope, **do not block** `optimize-step`; treat it as inherited project configuration. Candidate edits there must **preserve importability, signature, and output contract**. After such a run, recommend `/overmind-generate-spec-and-dataset` (repair mode) to move the harness back to `exclude_paths` / `fixed_elements` unless the team intentionally optimizes it.
- **No hardcoding**: Candidate edits must not special-case dataset examples, diagnosis examples, expected answers, field values, or test-case IDs.
- **Parallel isolation**: Each candidate edits only inside its own git worktree. Never edit the main working tree during candidate generation.
- **Evaluation owns truth**: Do not manually choose a winner. Evaluate all candidates through `overmind optimize-step evaluate`, then accept through `overmind optimize-step accept`.
- **Investigate zero baselines**: A baseline score of 0 may indicate a broken entrypoint, invalid dataset, unscorable eval spec, provider failure, or genuine total task failure. Investigate before running candidate optimization.
- **Use subagents when useful**: Use parallel sub-coding-agents for candidate edits when the host supports them. Also use focused investigation subagents when baseline failures, confusing score reports, or large codebase context would benefit from isolated analysis.
- **Multi-agent iteration fan-out (default when the host supports it)**: Do not drive the entire multi-iteration optimize loop alone in one bloated session. Spin out **several coding subagents** (for example 2–4) across the run: after `diagnose` for iteration `i`, delegate **each candidate worktree’s editing** to its own subagent (see **Spawn candidate coding agents**). Prefer a **fresh subagent** (or round-robin across a small pool) for each iteration’s edit leg so prior-iteration transcripts do not accumulate in one context. **One** coordinator must still run `overmind optimize-step` **in order** on the same `STATE_PATH` — iterations are sequential in the state file; parallelism is on **candidate branches** and on **which subagent** owns the editing leg.
- **Host capability fallbacks**: Prefer multi-agent fan-out when the host supports parallel subagents or background tasks. **If the host cannot spawn subagents or parallel tasks**, run the same loop **sequentially** in this session: one candidate worktree after another, then evaluate each; state aloud that parallel fan-out was unavailable. **Never** fake parallelism by editing two worktrees from one agent turn. **If background shell is the only option**, use distinct processes with isolated cwd per worktree, still one coordinator for `STATE_PATH` CLI order.
- **Surface analyzer failures**: If diagnosis returns a warning because analyzer generation failed, stop and report the warning. Do not silently proceed with manual placeholder edits.
- **Mandatory configuration branch (no silent defaults)**: Before `overmind optimize-step init`, the user must explicitly choose **Set optimization parameters** or **Run with defaults** — unless they already stated that exact choice in the **same message** that invoked this skill, in which case **echo the choice once** (“Using run-with-defaults from your message”) and continue. Never invent the branch from context alone.
- **Entrypoint cold-start**: Overmind evaluates agents in isolated processes. The host should ensure the registered entrypoint performs expensive construction once per interpreter process and keeps per-call work limited to mapping inputs, invoking the agent, and normalizing outputs. Do not rely on increasing smoke-test case counts to fix baseline cold-start; smoke tests filter candidates after a non-zero baseline exists.

## Prerequisites

Before starting, verify:

- `.overmind/agents.toml` exists in the active project root.
- The requested agent is registered.
- `.overmind/agents/<agent-name>/setup_spec/eval_spec.json` exists.
- `.overmind/agents/<agent-name>/setup_spec/dataset.json` exists.
- Provider configuration needed for evaluation and analyzer models is available in `.overmind/.env`, `.overmind/agents/<agent-name>/.env`, or the host environment. Per-agent `.env` overrides `.overmind/.env` for duplicate keys when both are loaded.
- Git is available and the project can create detached worktrees.

If any prerequisite is missing, stop and tell the user which setup skill or configuration step to run.

## Configuration

### Required first question (explicit branch)

Before initializing optimization, obtain exactly one of:

- **Set optimization parameters** — Collect **every** core field in the table below (use `AskQuestion` / chat). For any field the user defers, use that row’s default. Then optionally ask whether to adjust **advanced** settings.
- **Run with defaults** — Apply all defaults from the **Core** and **Advanced** tables without per-field prompts.

**In-thread shortcut:** If the user’s invoke message already contains a clear sentence such as “use defaults for optimize” or “set iterations to 8, defaults otherwise”, treat that as the branch + overrides after one-line confirmation.

### Deterministic preflight before `optimize-step init`

Run from project root **before** piping settings JSON into `init` (coding agent executes; fail fast with a clear stderr message):

```bash
python - <<'PY'
import os, pathlib, re, sys
def key_ok(v):
    v = (v or "").strip()
    if not v or v == "<set-me>": return False
    return not re.fullmatch(r"your[-_]?key[-_]?here|changeme|xxx+", v, re.I)
def amodel():
    v = os.getenv("ANALYZER_MODEL", "")
    if key_ok(v): return v
    p = pathlib.Path(".overmind/.env")
    if p.is_file():
        for ln in p.read_text().splitlines():
            s = ln.strip()
            if s.startswith("ANALYZER_MODEL="):
                return s.split("=",1)[1].strip()
    return ""
if not key_ok(amodel()):
    sys.exit("ANALYZER_MODEL missing or placeholder — set it in .overmind/.env or the environment before optimize-step init.")
print("ok")
PY
```

When the user chose **Run with defaults**, this script **must** print `ok` before `init`; if it exits non-zero, stop and tell them to fix `ANALYZER_MODEL` (do not rely on silent fallbacks). When the user chose **Set optimization parameters**, still run this script before `init` so a bad env fails fast even if they typed a model id in chat.

### Core settings

When the user chose **Set optimization parameters**, ask for **all** of the following (defaults shown — use them only when the user defers that specific field):

| Field | Default | Description |
| --- | --- | --- |
| `iterations` | `5` | Number of optimization iterations. |
| `candidates_per_iteration` | `3` | Parallel best-of-N candidates per iteration. |
| `parallel` | `true` | Run candidate / eval work in parallel when supported. |
| `max_workers` | `5` | Max parallel subprocess workers (meaningful when `parallel` is true). |
| `early_stopping_patience` | `3` | Stop after N stalled iterations. Use `0` to disable early stopping. |
| `analyzer_model` | `$ANALYZER_MODEL` or `claude-sonnet-4-20250514` | Model for diagnosing failures and generating plans. |
| `llm_judge_model` | *(empty)* | **Omit or empty** = no LLM judge. Set to a LiteLLM model id (often same as `analyzer_model`) to enable judge scoring. |

When the user chose **Run with defaults**, set at minimum: `iterations=5`, `candidates_per_iteration=3`, `parallel=true`, `max_workers=5`, `early_stopping_patience=3`, omit or clear `llm_judge_model`, and set `analyzer_model` from a **real** `ANALYZER_MODEL` env / `.overmind/.env` value (the preflight script above must pass). Apply all **Advanced settings** defaults below without prompting.

### Advanced settings

Ask whether to configure advanced settings **only** when the user chose **Set optimization parameters** and core fields are collected. If the user declines advanced configuration, or when the user chose **Run with defaults**, use the defaults:

| Field | Default | Description |
| --- | --- | --- |
| `runs_per_eval` | `1` | How many times to run each candidate full eval; the optimizer can take a median across runs for stability. Raising this reduces noisy candidate scores but does not replace correct entrypoint one-time initialization; baseline scoring behavior depends on the installed Overmind version. |
| `regression_threshold` | `0.35` | Minimum score delta required to accept a candidate. |
| `holdout_ratio` | `0.2` | Fraction of dataset reserved as holdout. |
| `holdout_enforcement` | `true` | Enforce holdout scoring. |
| `diagnosis_case_fraction` | `0.7` | Fraction of failing cases sent to the analyzer. |
| `cross_run_persistence` | `true` | Persist fix/failure history across iterations. |
| `failure_clustering` | `true` | Group similar failures before diagnosis. |
| `adaptive_focus` | `true` | Adjust focus weights based on failure patterns. |
| `smoke_test_cases` | `2` | Cases used for catastrophic-failure quick filter. |
| `codegen_max_steps` | `50` | Max edit steps per candidate sub-agent. |
| `model_backtesting` | `false` | Enable model backtesting mode. |
| `backtest_models` | `[]` | **If and only if** `model_backtesting` is `true`, this list **must be non-empty** — ask the user for one or more LiteLLM model ids before `init`. If `model_backtesting` is `false`, keep `[]` and do not ask. |

If advanced settings are already present in an existing state file or prompt, preserve them unless the user explicitly changes them. If starting fresh, use the defaults above unless the user specifies otherwise.

### Canonical `optimize-step init` JSON (subset)

The coding agent should build stdin JSON **only** from keys on Overmind’s `Config` (`unknown keys are dropped`). Example shape (values illustrative):

```json
{
  "iterations": 5,
  "candidates_per_iteration": 3,
  "parallel": true,
  "max_workers": 5,
  "early_stopping_patience": 3,
  "analyzer_model": "anthropic/claude-sonnet-4-20250514",
  "llm_judge_model": "",
  "runs_per_eval": 1,
  "regression_threshold": 0.35,
  "holdout_ratio": 0.2,
  "holdout_enforcement": true,
  "diagnosis_case_fraction": 0.7,
  "cross_run_persistence": true,
  "failure_clustering": true,
  "adaptive_focus": true,
  "smoke_test_cases": 2,
  "codegen_max_steps": 50,
  "model_backtesting": false,
  "backtest_models": []
}
```

- Leave `llm_judge_model` empty or omit to **disable** the judge; set to a model id to **enable** it.
- When `model_backtesting` is `true`, `backtest_models` must contain at least one model string or backtesting will not run.

## Entrypoint cold-start and evaluation stability

These rules are invariant-focused so they apply to any language or agent framework; implementers map them to local factories, modules, or dependency-injection style.

**Why it matters:** Harnesses run the agent under **process isolation**. Rebuilding the full agent stack, clients, tool registries, or large assets on **every** evaluation call repeats fixed cost and can make the first wall-clock window compete with model latency, producing empty or inconsistent outputs and misleadingly low early scores.

**Core rule:** **Construct once per interpreter process; invoke many times.** Anything that loads models, registers tools, opens pools, builds orchestration graphs, reads large assets, or walks heavy import graphs belongs in **initialization**, not in the per-invocation body of the function the harness calls for each case.

**Patterns (names only):**

- **Module-scoped initialization:** Perform expensive setup once after imports resolve, before the first request is handled; subsequent calls only pass inputs through the already-built object.
- **Lazy first-use initialization:** If configuration is not ready at import time, defer construction until the first real call, then **reuse** that result for all later calls in the same process. Document that the first call may be slower.
- **Async entrypoints:** If the harness wraps a short-lived event loop per call, keep **synchronous** construction on the one-time path; restrict the per-call path to async work that must run per request.

**Process constraints to respect:**

- **Subprocess isolation:** Each process has its own memory; globals do not survive across subprocesses. That is expected.
- **Parallelism:** If multiple harness invocations can run concurrently **within one process**, document thread-safety for any shared singleton or restrict parallelism.
- **Per-session state:** Reset conversation-scoped fields on each harness input; do **not** rebuild the entire stack each time.

**Author checklist before optimizing:**

- One-time costs are separated from per-request costs.
- The public entry function only resolves shared resources, maps input, runs the agent, and normalizes output.
- Long-lived resources are not re-acquired on every call without teardown.
- Two consecutive harness calls with different inputs succeed without cross-talk.

**Anti-patterns:** Rebuilding the full agent or orchestrator on every harness call; heavy I/O or client setup inside the per-call path; relying on “the second run fixes it” instead of fixing initialization placement.

**Optional explicit warm-up:** Only when the ecosystem supports a dedicated warm-up phase. Prefer singleton or lazy first-use initialization as the default because it avoids depending on discarding an initial run.

## Workflow

## Required command sequence (non-interactive)

Use this exact command sequence. Do not skip required parameters.

1. **Init state**
   - Required parameters: `<agent-name>`, settings JSON on **stdin** (full `Config`; same keys as **Canonical `optimize-step init` JSON**).
   - Example — **heredoc** (replace agent name; keep JSON on stdin):

```bash
overmind optimize-step init "my-agent-name" <<'EOF'
{
  "iterations": 5,
  "candidates_per_iteration": 3,
  "parallel": true,
  "max_workers": 5,
  "early_stopping_patience": 3,
  "analyzer_model": "anthropic/claude-sonnet-4-20250514",
  "llm_judge_model": "",
  "runs_per_eval": 1,
  "regression_threshold": 0.35,
  "holdout_ratio": 0.2,
  "holdout_enforcement": true,
  "diagnosis_case_fraction": 0.7,
  "cross_run_persistence": true,
  "failure_clustering": true,
  "adaptive_focus": true,
  "smoke_test_cases": 2,
  "codegen_max_steps": 50,
  "model_backtesting": false,
  "backtest_models": []
}
EOF
```

   - Example — **file redirect** after writing JSON to disk:

```bash
overmind optimize-step init "my-agent-name" < /tmp/overmind_optimize_config.json
```

   - Parse response and persist `STATE_PATH`.

2. **Baseline**
   - Required parameters: `--state <STATE_PATH>`.
   - Command:
     - `overmind optimize-step baseline --state "<STATE_PATH>"`

3. **Per-iteration diagnosis**
   - Required parameters: `--state <STATE_PATH> --iteration <i>`.
   - Command:
     - `overmind optimize-step diagnose --state "<STATE_PATH>" --iteration "<i>"`

4. **Per-candidate evaluation**
   - Required parameters: `--state <STATE_PATH> --iteration <i> --candidate-id <candidate_id> --candidate-dir <worktree>`.
   - Command:
     - `overmind optimize-step evaluate --state "<STATE_PATH>" --iteration "<i>" --candidate-id "<candidate_id>" --candidate-dir "<worktree>"`

5. **Iteration accept/reject**
   - Required parameters: `--state <STATE_PATH> --iteration <i> --candidate-results <candidate_results_path>`.
   - Command:
     - `overmind optimize-step accept --state "<STATE_PATH>" --iteration "<i>" --candidate-results "<candidate_results_path>"`

6. **Final report**
   - Required parameters: `--state <STATE_PATH>`.
   - Command:
     - `overmind optimize-step report --state "<STATE_PATH>"`

Rules:
- Every command after init must use the same `STATE_PATH`.
- If a required parameter is missing, stop and repair inputs before continuing.
- Never use interactive CLI prompts for optimization steps.

### Resolve the project and agent

Find the project root that contains the relevant `.overmind/`. Read `.overmind/agents.toml`, resolve the requested agent, and identify the registered entrypoint.

Do not require the registered entrypoint to be a separate interaction file before optimizing. Use the existing registration and eval spec as the source of truth.

If the **harness entrypoint** appears in optimizer scope, follow **Preserve invocation stability (default)** in **Operating principles** (legacy compatibility + post-run spec repair).

### Check setup artifacts

Confirm that `setup_spec/eval_spec.json` and `setup_spec/dataset.json` exist for the agent.

Do not preemptively stop optimization because of output field types, nested outputs, or list-shaped outputs. If the eval spec appears incompatible with the evaluator, warn the user that scoring may be affected, then let `overmind optimize-step baseline` or `evaluate` produce the authoritative result.

### Initialize optimization state

Follow **Configuration** above: the user must have chosen **Set optimization parameters** or **Run with defaults** before this step.

Create a settings JSON object that includes every `Config` field you collected (defaults path = tables’ default columns). **Run the ANALYZER_MODEL preflight script** from **Configuration** immediately before `init`. Feed that JSON on stdin using the **heredoc or file redirect** patterns in **Init state** under **Required command sequence** (do not invoke `init` without stdin JSON).

If a prior skill state already exists, ask whether to resume or start fresh. Use overwrite only when the user explicitly agrees to discard the previous optimization state.

Parse the JSON envelope. If it reports missing eval spec, missing dataset, invalid output schema, missing provider configuration, or state conflicts, stop with a clear next action.

Record the returned `STATE_PATH`. Every later optimize-step command must use that state path.

### Run baseline

Run `overmind optimize-step baseline --state <STATE_PATH>`. Parse the JSON envelope and report the baseline score, training set size, holdout size, and working path when available.

If baseline evaluation fails because the entrypoint cannot be imported, outputs cannot be scored, or provider configuration is missing, stop and report the optimize-step error. Point to the appropriate setup repair step only after the CLI reports the concrete failure.

If the baseline score is exactly 0, investigate before proceeding. Do not assume optimization should continue from zero. Review the baseline output, score artifacts, evaluator messages, failed cases, entrypoint import/runtime errors, dataset shape, and eval spec field mappings. Classify the zero baseline as one of:

- **Setup failure**: The agent cannot run, credentials are missing, imports fail, or the entrypoint contract is broken.
- **Scoring failure**: The eval spec cannot score the returned outputs, fields are mismatched, or all fields are unscorable.
- **Dataset mismatch**: Dataset inputs do not match the registered callable, or expected outputs do not align with evaluator fields.
- **Genuine performance failure**: The agent runs and scores correctly, but fails every case.
- **Inconclusive**: There is not enough evidence to classify the zero.

Use a focused subagent when the baseline artifacts or codebase are large enough that investigation would distract from loop control. The investigation subagent should inspect the baseline artifacts, eval spec, dataset, registered entrypoint, and score reports, then return a concise classification and recommended next step.

Proceed to optimization only if the zero baseline is classified as genuine performance failure or the user explicitly asks to optimize anyway despite the risk. If the zero is a setup, scoring, or dataset mismatch, stop and recommend the appropriate setup repair skill or configuration fix.

### Iterate

Optimization **iterations** share one `STATE_PATH` and must stay **strictly ordered**: for each index `i`, complete `diagnose` → edit all candidates for `i` → `evaluate` → `accept` before starting `i+1`. **Never** run `diagnose` or `accept` for two different iteration indices concurrently against the same state file.

For each iteration from 1 through the configured iteration count, run diagnosis, spawn candidate edits, evaluate all candidates, accept or reject the best candidate, and check early stopping — using **you** (and subagents you spawn) as the implementers, not the human user.

**Multi-agent pattern (default when the host supports it):**

1. **Coordinator** (you or a lead subagent you designate) runs `optimize-step diagnose` for iteration `i` and records candidate descriptors.
2. Spawn **one subagent per candidate worktree** (up to `candidates_per_iteration`) so edits run in parallel — **required** when the host exposes parallel tasks or background agents.
3. Coordinator runs `evaluate` and `accept` for that iteration (or, if the host allows safe parallel shells only, delegate **per-candidate** `evaluate` to subagents, then coordinator assembles `candidate_results.json` and runs `accept` once).
4. Optionally **rotate** which subagent receives the next iteration’s edit workload so context stays fresh.

If the user wants maximum parallelism, increase **candidate** subagent count first. Use **parallel iteration coordinators** only for **separate** optimization runs (separate `STATE_PATH` or separate agents), never two iteration indices on the same state file.

### Diagnose and materialise candidate worktrees

Run `overmind optimize-step diagnose --state <STATE_PATH> --iteration <i>`.

The response should include candidate descriptors with a candidate ID, worktree path, prompt path, plan path, entry file metadata, focus area, and suggested edit method.

If the response has warning status and includes a diagnosis warning, stop the loop. Report the warning’s last error and hint. This usually indicates missing analyzer provider configuration or an invalid analyzer model. Do not manually proceed with placeholder edits.

Inspect each candidate prompt and plan enough to understand the intended edit. If a candidate targets the registered harness entrypoint, allow it only when the candidate prompt or optimizer scope clearly includes that file, and remind **that subagent** to preserve importability, signature compatibility, and output contract stability.

### Spawn candidate coding agents

Detect the host environment once at skill start. **Prefer spinning out multiple coding subagents** — at minimum **one subagent per candidate worktree** for the current iteration, up to `candidates_per_iteration`, whenever the host exposes parallel tasks, background agents, or a Task tool. Otherwise use the host’s CLI in background processes. If no parallel mechanism exists, perform candidates sequentially inside their own worktrees and **tell the user** that multi-agent fan-out was unavailable on this host.

Use subagents whenever they improve reliability or parallelism:

- **Candidate subagents**: When the host supports parallel work, treat **one sub-coding-agent per candidate worktree** as the default (not optional). Each subagent edits only its assigned worktree.
- **Investigation subagents**: Spawn a focused codebase/debugging subagent for zero baselines, confusing evaluator failures, or analyzer warnings that require artifact inspection.
- **Review subagents**: Spawn a review subagent when candidate patches are large or touch shared behavior before evaluation.

Do not spawn subagents that edit the same worktree concurrently. Each editing subagent must have exactly one candidate worktree.

For each candidate, instruct that subagent to:

- Work only inside the candidate worktree.
- Read `PROMPT.md` and `plan.json`.
- Apply edits in place only to files requested by the candidate prompt and optimizer scope.
- Avoid copying files, moving files outside the worktree, or editing `.overmind` state.
- Preserve the registered entrypoint contract if any candidate edit touches it.
- Never hardcode dataset examples, diagnosis examples, expected outputs, user-specific values, or exact field values from test cases.
- Prefer general improvements such as better prompt wording, stronger parsing, cleaner logic, improved tool use, more robust validation, or better helper functions.
- Read files before editing and inspect callers and callees before changing shared functions.
- Re-read files after non-trivial edits.
- Use the worktree’s git diff to verify the candidate patch.
- Finish with a clear completion marker or status.

Spawn all candidates for the iteration before waiting, up to the configured worker limit. Wait for all candidate agents to finish before evaluating.

If a candidate agent crashes or times out, still record that candidate and proceed to evaluation if the worktree exists. Evaluation should classify failures.

### Evaluate candidates

For every candidate descriptor returned by diagnosis, run `overmind optimize-step evaluate --state <STATE_PATH> --iteration <i> --candidate-id <candidate_id> --candidate-dir <worktree>`.

Each evaluation should write a candidate score artifact in the candidate worktree. Build a candidate results array containing candidate ID, candidate directory, entry path, and score path for every candidate that reached evaluation.

Use this concrete `candidate_results.json` shape for the accept step:

```json
[
  {
    "candidate_id": "c0",
    "candidate_dir": "/abs/path/to/.overmind/agents/<agent-name>/experiments/iter_001_c0",
    "entry_path": "/abs/path/to/.overmind/agents/<agent-name>/experiments/iter_001_c0/agent.py",
    "score_path": "/abs/path/to/.overmind/agents/<agent-name>/experiments/iter_001_c0/score.json"
  },
  {
    "candidate_id": "c1",
    "candidate_dir": "/abs/path/to/.overmind/agents/<agent-name>/experiments/iter_001_c1",
    "entry_path": "/abs/path/to/.overmind/agents/<agent-name>/experiments/iter_001_c1/agent.py",
    "score_path": "/abs/path/to/.overmind/agents/<agent-name>/experiments/iter_001_c1/score.json"
  }
]
```

Use absolute paths for every path field to avoid resolution errors across host environments.

Do not manually adjust scores. **Evaluate every candidate** that has a materialized worktree and does not violate **hard** rules in **Candidate edit guardrails** — including patches that look weak or unlikely to win. Only skip `evaluate` when the worktree is missing, the candidate was never materialized, or you aborted before edits because of a **hard** guardrail violation (hardcoding, `.overmind` state edits, secrets, worktree boundary breach).

### Accept, reject, or stop early

Run `overmind optimize-step accept --state <STATE_PATH> --iteration <i> --candidate-results <candidate_results_path>`.

Parse the decision. Possible outcomes include accept, reject, all crashed, best score, winner, all scores, stall count, and early stop.

If a candidate is accepted, the optimize-step CLI owns promoting that candidate into the current best state. Do not manually copy files from a worktree into the main project.

If all candidates crash, report that iteration result and continue only if the returned state indicates the loop can proceed.

If early stopping fires, break the loop and tell the user the stall count and iteration where it fired.

### Render report

After the loop ends, run `overmind optimize-step report --state <STATE_PATH>`.

Parse the report path, best score, baseline score, iterations completed, early stopping status, and best-agent working file when present.

## Candidate edit guardrails

Reject or repair candidate work **before** calling `evaluate` only when it violates **hard** safety rules (below). **Do not** withhold evaluation because the change seems low quality, stylistically poor, or probably worse than baseline — those still go through `overmind optimize-step evaluate` so scores stay honest.

Hard violations — **do not evaluate** until repaired or discarded as a failed attempt:

- It edits generated `.overmind` state or setup artifacts during optimization.
- It hardcodes exact dataset inputs, expected answers, IDs, or diagnosis examples.
- It adds lookup tables keyed by example values.
- It adds brittle `if`, `elif`, `match`, or regex branches that exist only to match known test examples.
- It deletes core agent behavior rather than improving it.
- It moves files out of the worktree.
- It modifies provider secrets or prints secret values.

Prefer to let evaluation catch **quality** regressions. For anything that is merely suboptimal but compliant, run `evaluate`.

## Handling common failures

- **State already exists**: Ask whether to resume or start fresh. Use overwrite only with explicit approval.
- **Missing eval spec or dataset**: Stop and run or recommend `/overmind-generate-spec-and-dataset` (or `overmind setup <agent>`).
- **Output schema may be incompatible**: Warn the user that scoring may be affected, then rely on optimize-step baseline or evaluation to confirm the actual failure.
- **Nested or list outputs**: Do not block up front. Let the evaluator determine whether the current eval spec can score them.
- **Analyzer warning**: Stop and report the warning’s last error and hint; usually provider configuration or model name is wrong.
- **Candidate worktree missing**: Mark that candidate failed and continue evaluating the others.
- **All candidates crash**: Report the iteration result, then follow the accept-step state about whether to continue.
- **No improvement for patience window**: Stop early when the accept step reports early stopping.

## User-facing updates

Give concise progress updates at these milestones:

- Prerequisites checked.
- Settings initialized and state path captured.
- Baseline score computed.
- Candidate worktrees materialized for each iteration.
- Zero-baseline investigation result, if applicable.
- Candidate edits completed.
- Candidate scores and acceptance decision computed.
- Early stopping triggered, if applicable.
- Final report rendered.

## Final summary

When optimization finishes, tell the user:

- Baseline score and final best score.
- Absolute and relative delta when available.
- Iterations completed.
- Whether early stopping fired.
- Winning candidate summary if available.
- Report path.
- Best-agent working file or best snapshot path.
- Candidate worktree location pattern for inspection.
- Any warnings encountered about entrypoint edits, output schema compatibility, or evaluator compatibility.

If optimization could not run, give the exact blocker and the setup skill or configuration change needed next.
