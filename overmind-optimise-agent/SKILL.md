---
name: overmind-optimise-agent
description: "Drive the full host-agent controlled overmind optimize loop for a registered Overmind agent. Use when the user wants to optimise or optimize an agent, run iterative improvement, fan out multiple candidate edits in parallel worktrees, evaluate candidates with overmind optimize-step, accept the best improvement, stop early on stalls, and render an optimization report."
metadata:
  version: "2.5"
  product: "Overmind"
---

# Optimise an Overmind Agent

Use this skill to drive the `overmind optimize-step` JSON CLI from a host coding agent such as Cursor, Codex, Claude Code, or another code-editing agent. Overmind owns state, baseline evaluation, diagnosis, worktree materialisation, candidate evaluation, acceptance gates, and report rendering. The host coding agent owns loop control, **multi-agent fan-out** (several subagents across iterations and candidates), parallel candidate fan-out, per-candidate code edits, and early stopping.

**Audience:** Fan-out, worktree editing, and `overmind optimize-step` orchestration below are instructions for **you, the coding agent executing this skill** — not steps you assign to the human user unless the host truly cannot spawn subagents or run background work (then say so explicitly).

This skill optimizes the agent files selected by the existing Overmind eval spec and optimizer scope. It should not add extra setup restrictions that prevent `overmind optimize-step` from running.

## When to use / when not to use

- **Use** only after registration, `eval_spec.json`, and `dataset.json` exist for the agent and the user wants iterative improvement via `overmind optimize-step`.
- **Do not use** to repair broken registration, missing harness, or absent/broken eval artifacts — use `/overmind-register-agent` or `/overmind-generate-spec-and-dataset` first unless `optimize-step` already ran and pointed at a specific optimizer-only failure.

### Example (short)

User: *“Optimize `hotel-agent` with defaults.”* You confirm the **Run with defaults** branch, run the `ANALYZER_MODEL` preflight, pipe the canonical JSON settings into `overmind optimize-step init "hotel-agent"`, capture `STATE_PATH`, run `baseline`, then loop `diagnose` → parallel candidate edits in worktrees → `evaluate` → `accept`, fanning out subagents when the host supports it, and finish with `report`.

## Operating principles

- **Codebase-derived optimization**: Use the registered agent, eval spec, dataset, policy, diagnosis output, worktree prompts, tests, examples, and codebase context as the source of truth. Do not rely on broad user elicitation for optimization strategy.
- **Prerequisites first**: Stop if the agent is not registered, `eval_spec.json` is missing, `dataset.json` is missing, or required provider configuration is absent.
- **Correct project root**: Run all Overmind commands from the directory that contains the relevant `.overmind/`. Some repositories contain nested projects with their own `.overmind/`.
- **Respect optimizer scope**: Candidate edits should target files selected by the eval spec and candidate prompt. Do not override the project’s configured optimization scope with additional skill-level assumptions.
- **Preserve invocation stability (default)**: In specs authored by the setup skills, the registered Overmind entrypoint is a **fixed harness** and should stay out of optimizer scope. **Legacy / existing specs:** if `eval_spec.json` already includes the entrypoint in optimizer scope, **do not block** — treat that as compatibility with an older configuration; candidate edits must still preserve importability, signature compatibility, and the output contract. Recommend `/overmind-generate-spec-and-dataset` repair afterward if the entrypoint was incorrectly scoped.
- **No hardcoding**: Candidate edits must not special-case dataset examples, diagnosis examples, expected answers, field values, or test-case IDs.
- **Parallel isolation**: Each candidate edits only inside its own git worktree. Never edit the main working tree during candidate generation.
- **Evaluation owns truth**: Do not manually choose a winner. Evaluate all candidates through `overmind optimize-step evaluate`, then accept through `overmind optimize-step accept`.
- **Investigate zero baselines**: A baseline score of 0 may indicate a broken entrypoint, invalid dataset, unscorable eval spec, provider failure, or genuine total task failure. Investigate before running candidate optimization.
- **Use subagents when useful**: Use parallel sub-coding-agents for candidate edits when the host supports them. Also use focused investigation subagents when baseline failures, confusing score reports, or large codebase context would benefit from isolated analysis.
- **Multi-agent iteration fan-out (default when the host supports it)**: Do not drive the entire multi-iteration optimize loop alone in one bloated session. Spin out **several coding subagents** (for example 2–4) across the run: after `overmind optimize-step diagnose` for iteration `i`, delegate **each candidate worktree’s editing** to its own subagent (see **Spawn candidate coding agents**). Prefer a **fresh subagent** (or round-robin across a small pool) for each iteration’s edit leg so prior-iteration transcripts do not accumulate in one context. **One** coordinator must still run `overmind optimize-step` **in order** on the same `STATE_PATH` — iterations are sequential in the state file; parallelism is on **candidate branches** and on **which subagent** owns the editing leg. If the host cannot spawn subagents, state that you are falling back to single-agent sequential mode.
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

## Host capability fallbacks

Subagents and background tasks are **preferred** when the host exposes them, but not all hosts do. Detect once at skill start:

- **Full capability** (Tasks / subagents + parallel shell): use the multi-agent iteration pattern and **one subagent per candidate worktree** as the default.
- **Parallel shell only** (no subagents): run candidate patch applications **sequentially**, each in its own worktree, optionally using background `&` only if you can still guarantee **one editor per worktree** and ordered `evaluate` / `accept`.
- **Single-threaded host**: run everything sequentially in one agent context; **tell the user once** that multi-agent fan-out was unavailable so they know throughput is limited.

Never parallelize two different **iteration indices** against the same `STATE_PATH` file.

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

### Required command sequence (non-interactive)

Use this exact command sequence. Do not skip required parameters.

1. **Init state**
   - Required parameters: `<agent-name>`, settings JSON on **stdin** (not argv).
   - Command (settings JSON must be piped or heredoc’d — do not rely on the agent “remembering” flags that do not exist):
     - `overmind optimize-step init "<agent-name>"` with stdin attached, for example:
     ```bash
     overmind optimize-step init "hotel-agent" <<'JSON'
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
     JSON
     ```
     Equivalent: `printf '%s' '<json-minified>' | overmind optimize-step init "hotel-agent"` (mind quoting in the shell).
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

Use the existing registration and eval spec as the source of truth. **Entrypoint in optimizer scope:** the **desired** configuration (per `/overmind-generate-spec-and-dataset`) keeps the registered Overmind entrypoint in `exclude_paths` / `fixed_elements`, not in `optimizable_paths`. If the **current** spec already includes the entrypoint as optimizable, treat that as **legacy compatibility** — do not refuse `optimize-step` — and warn once that candidates touching the entrypoint must preserve the callable contract; recommend running the spec/dataset repair skill after optimization if scope was wrong.

### Check setup artifacts

Confirm that `setup_spec/eval_spec.json` and `setup_spec/dataset.json` exist for the agent.

Do not preemptively stop optimization because of output field types, nested outputs, or list-shaped outputs. If the eval spec appears incompatible with the evaluator, warn the user that scoring may be affected, then let `overmind optimize-step baseline` or `evaluate` produce the authoritative result.

### Initialize optimization state

Follow **Configuration** above: the user must have chosen **Set optimization parameters** or **Run with defaults** before this step.

Create a settings JSON object that includes every `Config` field you collected (defaults path = tables’ default columns). **Run the ANALYZER_MODEL preflight script** from **Configuration** immediately before `init`. Then run `overmind optimize-step init <agent-name>` with the settings JSON on stdin (see **Required command sequence** heredoc example — stdin is mandatory; do not pass settings as undocumented CLI flags).

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

**Multi-agent pattern (default when the host supports it — see Host capability fallbacks):**

1. **Coordinator** (you or a lead subagent you designate) runs `optimize-step diagnose` for iteration `i` and records candidate descriptors.
2. Spawn **one subagent per candidate worktree** (up to `candidates_per_iteration`) so edits run in parallel — **required** when the host exposes parallel tasks or background agents; when it does not, follow **Host capability fallbacks**.
3. Coordinator runs `evaluate` and `accept` for that iteration (or, if the host allows safe parallel shells only, delegate **per-candidate** `evaluate` to workers, then coordinator assembles `candidate_results.json` and runs `accept` once).
4. Optionally **rotate** which subagent receives the next iteration’s edit workload so context stays fresh.

If the user wants maximum parallelism, increase **candidate** subagent count first. Use **parallel iteration coordinators** only for **separate** optimization runs (separate `STATE_PATH` or separate agents), never two iteration indices on the same state file.

### Diagnose and materialise candidate worktrees

Run `overmind optimize-step diagnose --state <STATE_PATH> --iteration <i>`.

The response should include candidate descriptors with a candidate ID, worktree path, prompt path, plan path, entry file metadata, focus area, and suggested edit method.

If the response has warning status and includes a diagnosis warning, stop the loop. Report the warning’s last error and hint. This usually indicates missing analyzer provider configuration or an invalid analyzer model. Do not manually proceed with placeholder edits.

Inspect each candidate prompt and plan enough to understand the intended edit. If a candidate targets the registered entrypoint, allow it only when the **existing** eval spec / candidate prompt places that file in scope; preserve importability, signature compatibility, and output contract stability. Prefer recommending spec repair to exclude the entrypoint when optimization finishes.

### Spawn candidate coding agents

Detect the host environment once at skill start; apply **Host capability fallbacks** when subagents are unavailable. **Prefer spinning out multiple coding subagents** — at minimum **one subagent per candidate worktree** for the current iteration, up to `candidates_per_iteration`, whenever the host exposes parallel tasks, background agents, or a Task tool. Otherwise use the host’s CLI in background processes **only** if worktree isolation is preserved. If no parallel mechanism exists, perform candidates sequentially inside their own worktrees and **tell the user** that multi-agent fan-out was unavailable on this host.

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

Do not manually adjust scores. Do not skip **valid** candidates for quality reasons. Skip **only** candidates disqualified under **Candidate edit guardrails** (missing worktree, terminal `evaluate` error, or hard safety violations before eval).

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

Reject or repair candidate work **before** calling `overmind optimize-step evaluate` **only** when it violates **hard safety** rules below (hardcoding, forbidden state edits, secrets, worktree boundaries). These are disqualifying defects — do not spend eval budget on them.

- It edits generated `.overmind` state or setup artifacts during optimization.
- It hardcodes exact dataset inputs, expected answers, IDs, or diagnosis examples.
- It adds lookup tables keyed by example values.
- It adds brittle `if`, `elif`, `match`, or regex branches that exist only to match known test examples.
- It deletes core agent behavior rather than improving it.
- It moves files out of the worktree.
- It modifies provider secrets or prints secret values.

**Evaluate normal quality issues:** For patches that are merely weak, stylistically poor, or likely low-scoring but **not** violating the list above, still run `evaluate` and let scores decide.

Prefer to let evaluation catch quality regressions, but **skip `evaluate` only** for hard-rule violations; **do not** skip weak-but-valid candidates.

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
