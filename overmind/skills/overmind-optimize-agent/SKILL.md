______________________________________________________________________

## name: overmind-optimize-agent description: "Optimize a registered Overmind agent. Use when the user wants to run iterative improvement on an agent. **Defaults to the native host coding agent driving an `overmind optimize-step` loop with subagent fan-out**; only switches to the autonomous `overmind optimize` CLI path when the user explicitly asks for it." metadata: version: "2.7" product: "Overmind"

# Optimize an Overmind Agent

Use this skill to optimize a registered Overmind agent end-to-end. **Two execution paths are supported, and the skill defaults to Path B (native coding agent) without asking.** Only switch to Path A when the user explicitly requests it (e.g. "use the overmind CLI", "run it in a terminal", "use Path A"):

- **Path B — Native coding agent (host-driven `overmind optimize-step` loop with subagent fan-out) — DEFAULT.** The host coding agent (Cursor / Codex / Claude Code) drives the loop step-by-step in-chat via the `overmind optimize-step` JSON CLI, fans out **one subagent per candidate worktree** for parallel edits, and orchestrates evaluation and acceptance. This is the default path: it gives per-candidate editorial control, custom diagnosis follow-ups, and external subagent parallelism beyond what the CLI offers.
- **Path A — Overmind CLI in a new IDE terminal (explicit opt-in only).** Launch `overmind optimize <agent-name> [--fast]` in a fresh **IDE-integrated terminal** (the host coding agent's terminal panel — e.g. Cursor's terminal pane — *not* a separate macOS / Linux desktop terminal window). Wrap the command in `script -q /dev/null <cmd>` (BSD/macOS) or `unbuffer <cmd>` (Linux + `expect`) so Rich is handed a pseudo-TTY and renders colours / progress bars correctly. **Then stop.** Overmind owns the entire loop end-to-end in that terminal (baseline, diagnosis, candidates, evaluate, accept, early stop, final report) and pushes its own live UI updates to the Overmind dashboard via OTLP. The host coding agent's responsibility ends at launch — **no REST polling, no Job UUID resolution, no monitor script, no progress chatter in this chat**. The user watches the IDE terminal directly; the dashboard updates itself. If the user later asks for a status snapshot, only then query the REST API. Only use this path when the user explicitly asks for it.

**Audience:** Path B's fan-out, worktree editing, and `optimize-step` orchestration below are instructions for **you, the coding agent executing this skill** — not steps you assign to the human user unless the host truly cannot spawn subagents or run background work (then say so explicitly). Path A's launch (terminal pop-up) is also yours to execute when the user opts in.

This skill optimizes the agent files selected by the existing Overmind eval spec and optimizer scope. It should not add extra setup restrictions that prevent `overmind optimize-step` from running.

> always export the environment variable OVERMIND_API_KEY if present in the `.env`

## Operating principles

- **Codebase-derived optimization**: Use the registered agent, eval spec, dataset, policy, diagnosis output, worktree prompts, tests, examples, and codebase context as the source of truth. Do not rely on broad user elicitation for optimization strategy.
- **Prerequisites first**: Stop if the agent is not registered, `eval_spec.json` is missing, `dataset.json` is missing, or required provider configuration is absent.
- **Correct project root**: Run all Overmind commands from the directory that contains the relevant `.overmind/`. Some repositories contain nested projects with their own `.overmind/`.
- **Respect optimizer scope**: Candidate edits should target files selected by the eval spec and candidate prompt. Do not override the project’s configured optimization scope with additional skill-level assumptions.
- **Preserve invocation stability**: Avoid editing the registered entrypoint unless the candidate prompt and optimizer scope explicitly require it. If it is edited, keep the callable contract stable.
- **No hardcoding**: Candidate edits must not special-case dataset examples, diagnosis examples, expected answers, field values, or test-case IDs.
- **Parallel isolation**: Each candidate edits only inside its own git worktree. Never edit the main working tree during candidate generation.
- **Evaluation owns truth**: Do not manually choose a winner. Evaluate all candidates through `overmind optimize-step evaluate`, then accept through `overmind optimize-step accept`.
- **Investigate zero baselines**: A baseline score of 0 may indicate a broken entrypoint, invalid dataset, unscorable eval spec, provider failure, or genuine total task failure. Investigate before running candidate optimization.
- **Use subagents when useful**: Use parallel sub-coding-agents for candidate edits when the host supports them. Also use focused investigation subagents when baseline failures, confusing score reports, or large codebase context would benefit from isolated analysis.
- **Loop host defaults to Path B (native coding agent) — do not ask**: Path B runs by default with **no `AskQuestion` prompt**. Only switch to **Path A (Overmind CLI in a new IDE terminal)** when the user's invoke message explicitly names it (e.g. "use the overmind CLI", "run optimize in a terminal", "use Path A", "open a terminal in the IDE and run optimize"). When the user explicitly opts into Path A, echo the choice once ("Switching to Path A — Overmind CLI in a new IDE terminal from your message") and continue. Never ask the user to pick between the two paths.
- **Multi-agent iteration fan-out (Path B only)**: When running Path B (the default), do not drive the entire multi-iteration optimize loop alone in one bloated session. Spin out **several coding subagents** (for example 2–4) across the run: after `diagnose` for iteration `i`, delegate **each candidate worktree’s editing** to its own subagent (see **Spawn candidate coding agents**). Prefer a **fresh subagent** (or round-robin across a small pool) for each iteration’s edit leg so prior-iteration transcripts do not accumulate in one context. **One** coordinator must still run `overmind optimize-step` **in order** on the same `STATE_PATH` — iterations are sequential in the state file; parallelism is on **candidate branches** and on **which subagent** owns the editing leg. If the host cannot spawn subagents while running Path B, state that you are falling back to single-agent sequential mode (or recommend switching to Path A).
- **Surface analyzer failures**: If diagnosis returns a warning because analyzer generation failed, stop and report the warning. Do not silently proceed with manual placeholder edits.
- **Mandatory configuration branch (no silent defaults)**: Before `overmind optimize-step init`, the user must explicitly choose **Set optimization parameters** or **Run with defaults** — unless they already stated that exact choice in the **same message** that invoked this skill, in which case **echo the choice once** (“Using run-with-defaults from your message”) and continue. Never invent the branch from context alone.
- **Entrypoint cold-start**: Overmind evaluates agents in isolated processes. The host should ensure the registered entrypoint performs expensive construction once per interpreter process and keeps per-call work limited to mapping inputs, invoking the agent, and normalizing outputs. Do not rely on increasing smoke-test case counts to fix baseline cold-start; smoke tests filter candidates after a non-zero baseline exists.

## Prerequisites

Before starting, verify:

- `.overmind/agents.toml` exists in the active project root.
- The requested agent is registered.
- `.overmind/agents/<agent-name>/setup_spec/eval_spec.json` exists.
- `.overmind/agents/<agent-name>/setup_spec/dataset.json` exists.
- Provider configuration needed for evaluation and analyzer models is available in `.overmind/.env` or the host environment. There is **no** per-agent `.env`; all keys (Overmind, analyzer provider, agent runtime providers) come from the single project-level file or the shell.
- `OVERMIND_API_KEY` is configured so the agent / policy / eval-spec / dataset can be synced before `optimize-step init`. The project is auto-resolved from the key.
- Git is available and the project can create detached worktrees.
- **`scope.optimizable_paths` expands to ≥1 real file** (defense in depth — the spec-generation skill already runs this gate, but Overmind's CLI validator does not). Run the check below before touching `optimize` / `optimize-step`. A bare directory entry (`"my_package"`) silently expands to zero files inside `BundleFactory.from_entry_point`, leaving the analyzer with nothing to edit; it then mistakes the read-only harness for fair game and emits severe regressions. If this check fails, **stop** and tell the user to fix `optimizable_paths` (almost always: change `"my_package"` → `"my_package/**/*.py"`), re-save the spec, and re-run this skill.

  ```python
  import json, pathlib, sys
  spec = json.loads(pathlib.Path(".overmind/agents/<agent>/setup_spec/eval_spec.json").read_text())
  root = pathlib.Path(".").resolve()
  for key in ("optimizable_paths", "read_only_paths"):
      missing = [
          pat for pat in spec.get("scope", {}).get(key, [])
          if not any(p.is_file() for p in root.glob(pat)) and not (root / pat).is_file()
      ]
      if missing:
          sys.exit(f"scope.{key} resolves to 0 files for: {missing} — fix the spec before optimizing.")
  ```

- **Cross-run analyzer memory is fresh** (mandatory whenever `dataset.json` or `eval_spec.json` was edited since the last optimize). Overmind's analyzer reads `run_state.json` for the same agent and uses three fields to bias diagnosis and codegen: `failure_registry.clusters` (open failure patterns), `cumulative_failed_attempts[*].suggestions` (verbatim suggestion strings the analyzer surfaces back into prompts), and `component_failure_weights` (which component bucket — `agent_logic`, `system_prompt`, etc. — gets 100% focus). **None of these are invalidated when the dataset changes.** If the dataset was repaired (e.g. transient errors removed), stale clusters and stale suggestions will pull every diagnosis back to the *old* failure pattern and produce candidates that catastrophically regress (e.g. iteration 1 score drop of −50 or more with `Success: 0.0/N` and `Extractions Json: 0.0/N`). Before optimizing, run the check below. If any of the three fields is non-empty AND the dataset's modification time is newer than `run_state.json`'s, **stop**, back up `run_state.json`, then clear those three fields and `regression_cases`, and `run_history`. Do not skip — the analyzer will silently use stale state otherwise.

  ```python
  import json, pathlib, shutil, sys
  from datetime import datetime
  agent_dir = pathlib.Path(".overmind/agents/<agent>")
  rs_path = agent_dir / "run_state.json"
  ds_path = agent_dir / "setup_spec" / "dataset.json"
  if rs_path.exists() and ds_path.exists() and ds_path.stat().st_mtime > rs_path.stat().st_mtime:
      rs = json.loads(rs_path.read_text())
      stale = (
          rs.get("failure_registry", {}).get("clusters")
          or rs.get("cumulative_failed_attempts")
          or rs.get("component_failure_weights")
      )
      if stale:
          shutil.copy2(rs_path, rs_path.with_suffix(f".json.bak.{datetime.now():%Y%m%d-%H%M%S}"))
          rs["failure_registry"] = {"clusters": {}}
          rs["cumulative_failed_attempts"] = []
          rs["cumulative_successful_changes"] = []
          rs["component_failure_weights"] = {}
          rs["regression_cases"] = []
          rs["run_history"] = []
          rs_path.write_text(json.dumps(rs, indent=2))
          print(f"cleared stale analyzer memory (dataset newer than run_state).")
  ```

  Also clear `.overmind/agents/<agent>/experiments/` if any prior optimize attempt failed mid-run — it may contain orphaned candidate worktrees and intermediate state that confuse `optimize-step` resumption.

If any prerequisite is missing, stop and tell the user which setup skill or configuration step to run.

## Configuration

### Loop host (default: Path B — do not ask)

The skill **defaults to Path B (native coding agent driving `overmind optimize-step` with subagent fan-out)**. **Do not ask the user which path to run.** There are still two paths available, but Path B is selected unless the user explicitly opts into Path A in their invoke message.

| Option                                                          | What runs                                                                                                                                                                                                                                                                                                                                  | What the user sees                                                                                                              | When it is selected                                                                                                                          |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **B. Native coding agent (host-driven `optimize-step` + subagents) — DEFAULT** | The host coding agent runs `overmind optimize-step init → baseline → (diagnose → spawn N candidate subagents → evaluate → accept)* → report` in-chat, fanning out one coding subagent per candidate worktree each iteration.                                                                                                              | In-chat tool calls and assistant messages summarizing each step.                                                                | **Default** — always selected unless the user explicitly opts into Path A. Gives per-candidate editorial control via subagents, custom diagnosis follow-ups, and external parallelism. |
| **A. Overmind CLI in a new IDE terminal (explicit opt-in)**     | A fresh **IDE-integrated terminal** (host's terminal pane — e.g. Cursor's terminal panel) pops open and runs `script -q /dev/null overmind optimize <agent> [--fast]` (the `script` wrapper gives Rich a pseudo-TTY so colours and progress bars render). Overmind owns the loop end-to-end inside that terminal. **Host coding agent does nothing else** — no REST polling, no Job UUID resolution, no monitor script. The Overmind dashboard updates itself via OTLP. | **Live Rich UI in the IDE terminal pane** — progress bars, candidate score tables, accept/reject animations, final report rendering. Dashboard also updates live. | Only when the user's invoke message explicitly names this path (e.g. "use the overmind CLI", "run optimize in a terminal", "use Path A"). |

**Explicit opt-in for Path A:** If the user's invoke message explicitly names Path A ("use the overmind CLI", "run optimize in a terminal", "open a terminal in the IDE and run optimize", "use Path A"), echo the switch once ("Switching to Path A — Overmind CLI in a new IDE terminal from your message") and continue. Otherwise, silently proceed with Path B — do **not** ask the user to confirm, and do **not** call `AskQuestion` for this branch.

Once the path is settled:

- **Path B (default)** → proceed to **Required first question** below (parameters branch), then jump to **Workflow → Path B**.
- **Path A (explicit opt-in only)** → proceed to **Required first question** below (parameters branch), then jump to **Workflow → Path A**.

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

| Field                      | Default                                         | Description                                                                                                           |
| -------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `iterations`               | `5`                                             | Number of optimization iterations.                                                                                    |
| `candidates_per_iteration` | `3`                                             | Parallel best-of-N candidates per iteration.                                                                          |
| `parallel`                 | `true`                                          | Run candidate / eval work in parallel when supported.                                                                 |
| `max_workers`              | `5`                                             | Max parallel subprocess workers (meaningful when `parallel` is true).                                                 |
| `early_stopping_patience`  | `3`                                             | Stop after N stalled iterations. Use `0` to disable early stopping.                                                   |
| `analyzer_model`           | `$ANALYZER_MODEL` or `claude-sonnet-4-20250514` | Model for diagnosing failures and generating plans.                                                                   |
| `llm_judge_model`          | *(empty)*                                       | **Omit or empty** = no LLM judge. Set to a LiteLLM model id (often same as `analyzer_model`) to enable judge scoring. |

When the user chose **Run with defaults**, set at minimum: `iterations=5`, `candidates_per_iteration=3`, `parallel=true`, `max_workers=5`, `early_stopping_patience=3`, omit or clear `llm_judge_model`, and set `analyzer_model` from a **real** `ANALYZER_MODEL` env / `.overmind/.env` value (the preflight script above must pass). Apply all **Advanced settings** defaults below without prompting.

### Advanced settings

Ask whether to configure advanced settings **only** when the user chose **Set optimization parameters** and core fields are collected. If the user declines advanced configuration, or when the user chose **Run with defaults**, use the defaults:

| Field                     | Default | Description                                                                                                                                                                                                                                                                                  |
| ------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `runs_per_eval`           | `1`     | How many times to run each candidate full eval; the optimizer can take a median across runs for stability. Raising this reduces noisy candidate scores but does not replace correct entrypoint one-time initialization; baseline scoring behavior depends on the installed Overmind version. |
| `regression_threshold`    | `0.35`  | Minimum score delta required to accept a candidate.                                                                                                                                                                                                                                          |
| `holdout_ratio`           | `0.2`   | Fraction of dataset reserved as holdout.                                                                                                                                                                                                                                                     |
| `holdout_enforcement`     | `true`  | Enforce holdout scoring.                                                                                                                                                                                                                                                                     |
| `diagnosis_case_fraction` | `0.7`   | Fraction of failing cases sent to the analyzer.                                                                                                                                                                                                                                              |
| `cross_run_persistence`   | `true`  | Persist fix/failure history across iterations.                                                                                                                                                                                                                                               |
| `failure_clustering`      | `true`  | Group similar failures before diagnosis.                                                                                                                                                                                                                                                     |
| `adaptive_focus`          | `true`  | Adjust focus weights based on failure patterns.                                                                                                                                                                                                                                              |
| `smoke_test_cases`        | `2`     | Cases used for catastrophic-failure quick filter.                                                                                                                                                                                                                                            |
| `codegen_max_steps`       | `50`    | Max edit steps per candidate sub-agent.                                                                                                                                                                                                                                                      |
| `model_backtesting`       | `false` | Enable model backtesting mode.                                                                                                                                                                                                                                                               |
| `backtest_models`         | `[]`    | **If and only if** `model_backtesting` is `true`, this list **must be non-empty** — ask the user for one or more LiteLLM model ids before `init`. If `model_backtesting` is `false`, keep `[]` and do not ask.                                                                               |

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

The workflow branches on the loop-host selection from **Configuration → Loop host (default: Path B — do not ask)**. **Run Path B by default**; only run Path A when the user explicitly opted into it in their invoke message. Run **Path A** *or* **Path B**, never both.

### Path A — Overmind CLI in a new IDE terminal (explicit opt-in only)

Use this path **only** when the user's invoke message explicitly named Path A ("use the overmind CLI", "run optimize in a terminal", "use Path A", "open a terminal in the IDE and run optimize"). Do **not** default into this path. The host coding agent's job is **only** to do the three things below, then **stop and hand off to the user**.

**Hard rules for Path A:**

- Do **not** poll the Overmind REST API for Job / JobIteration status.
- Do **not** resolve the Job UUID.
- Do **not** spawn a monitor script or background poller.
- Do **not** narrate progress in this chat while the run is going.
- Do **not** use a separate desktop terminal window (no `osascript`, no `open -a Terminal`, no `iTerm` AppleScript).
- Do **not** redirect stdout/stderr to a log file — that strips the TTY and Rich falls back to plain non-interactive output.

The user watches the IDE terminal directly; the Overmind dashboard updates itself live via OTLP. If the user explicitly asks for a status snapshot later, only then query the REST API.

#### Step 1 — Sync setup artifacts to the backend

Run the **Sync setup artifacts** Python snippet from earlier in this skill. Required so the `Job` row created by the optimizer attaches to a fully-populated `Agent` record. Fail fast on any exception.

#### Step 2 — Preflight `ANALYZER_MODEL`

Run the Python preflight snippet from **Deterministic preflight before `optimize-step init`** above. It must print `ok`. If it fails, stop and tell the user to fix `ANALYZER_MODEL` in `.overmind/.env`.

#### Step 3 — Open an IDE terminal and run `overmind optimize`, then stop

Launch via the host's Shell tool in **background mode** (e.g. Cursor's Shell tool with `block_until_ms: 0`) so the terminal entry persists in the IDE terminal pane and the user can click it to watch live progress. Wrap the command in `script` (BSD/macOS) or `unbuffer` (Linux) so Rich gets a pseudo-TTY and progress bars / colours render.

- **macOS / BSD** (default for this repo):

  ```bash
  cd <project-root> && \
    source .venv/bin/activate && \
    script -q /dev/null overmind optimize "<agent-name>" --fast
  ```

- **Linux with `expect` available**:

  ```bash
  cd <project-root> && \
    source .venv/bin/activate && \
    unbuffer overmind optimize "<agent-name>" --fast
  ```

- **Fallback when no TTY emulator is available**: run `overmind optimize "<agent-name>" --fast` directly — the UI degrades to plain text but the run still works. Tell the user this happened.

Command variants:

- Drop `--fast` for the full defaults branch (LLM judge / backtesting available).
- When the user picked **Set optimization parameters**, pass overrides as CLI flags (`--iterations`, `--candidates-per-iteration`, `--analyzer-model`, `--llm-judge-model`, `--early-stopping-patience`, etc.) — verify with `overmind optimize --help`. Advanced settings without flags flow through `.overmind/.env`.

After launching, post **one** short message in chat: *"Optimizer running in the new IDE terminal — click it to watch live progress. The Overmind dashboard will also update in real time. I'll wait here; ping me when you want a status check or once it finishes."* Then **stop**.

**Troubleshooting Path A** (only if the user reports a problem):

- **IDE terminal shows plain text without colours or progress bars** → the `script` / `unbuffer` wrapper isn't taking effect. Verify `which script` (macOS ships it at `/usr/bin/script`) or install `expect` (`brew install expect` / `apt install expect`) and rerun.
- **CLI exits immediately with an entrypoint / dataset / analyzer error** → read the IDE terminal output, fix the underlying setup issue (entrypoint import, dataset shape, analyzer key), and relaunch step 3.
- **The user reports a stuck `running` Job in the dashboard hours after the terminal exited** → only then query `PATCH /api/jobs/{id}/ {"status":"failed"}` to clear it. Never sweep proactively.

### Path B — Host-driven `overmind optimize-step` loop (default)

This is the **default path** — run it whenever the user did **not** explicitly opt into Path A. Follow the exact non-interactive command sequence below. Do not skip required parameters.

1. **Init state**

   - Required parameters: `<agent-name>`, settings JSON on stdin.
   - Command:
     - `overmind optimize-step init "<agent-name>"`
   - Parse response and persist `STATE_PATH`.

1. **Baseline**

   - Required parameters: `--state <STATE_PATH>`.
   - Command:
     - `overmind optimize-step baseline --state "<STATE_PATH>"`

1. **Per-iteration diagnosis**

   - Required parameters: `--state <STATE_PATH> --iteration <i>`.
   - Command:
     - `overmind optimize-step diagnose --state "<STATE_PATH>" --iteration "<i>"`

1. **Per-candidate evaluation**

   - Required parameters: `--state <STATE_PATH> --iteration <i> --candidate-id <candidate_id> --candidate-dir <worktree>`.
   - Command:
     - `overmind optimize-step evaluate --state "<STATE_PATH>" --iteration "<i>" --candidate-id "<candidate_id>" --candidate-dir "<worktree>"`

1. **Iteration accept/reject**

   - Required parameters: `--state <STATE_PATH> --iteration <i> --candidate-results <candidate_results_path>`.
   - Command:
     - `overmind optimize-step accept --state "<STATE_PATH>" --iteration "<i>" --candidate-results "<candidate_results_path>"`

1. **Final report**

   - Required parameters: `--state <STATE_PATH>`.
   - Command:
     - `overmind optimize-step report --state "<STATE_PATH>"`

Rules:

- Every command after init must use the same `STATE_PATH`.
- If a required parameter is missing, stop and repair inputs before continuing.
- Never use interactive CLI prompts for optimization steps.

The remaining sections in this file (**Resolve the project and agent**, **Check setup artifacts**, **Diagnose and materialise candidate worktrees**, **Spawn candidate coding agents**, **Evaluate candidates**, **Accept**, **Early stopping**, **Report**) apply to **Path B only**. **Path A delegates all of these to the `overmind optimize` process** — do not run them manually on top of an active Path A run.

### Resolve the project and agent

Find the project root that contains the relevant `.overmind/`. Read `.overmind/agents.toml`, resolve the requested agent, and identify the registered entrypoint.

Do not require the registered entrypoint to be a separate interaction file before optimizing. Use the existing registration and eval spec as the source of truth.

If the entrypoint is included in optimizer scope, do not block. Treat it as part of the configured project behavior and remind candidate agents to preserve the registered callable contract.

### Check setup artifacts

Confirm that `setup_spec/eval_spec.json` and `setup_spec/dataset.json` exist for the agent.

Do not preemptively stop optimization because of output field types, nested outputs, or list-shaped outputs. If the eval spec appears incompatible with the evaluator, warn the user that scoring may be affected, then let `overmind optimize-step baseline` or `evaluate` produce the authoritative result.

### Sync setup artifacts to the Overmind backend (required, before `init`)

The optimize loop and the Overmind UI both read agent / policy / eval-spec / dataset state from the backend, not from `.overmind/agents/<name>/setup_spec/`. Before `overmind optimize-step init`, push the local artifacts so the resulting `Job` row attaches to a fully-populated `Agent` record — regardless of whether the artifacts were authored by `/overmind-generate-spec-and-dataset` (which already syncs) or by hand.

Run from the project root (coding agent executes; fail fast on any exception):

```python
import json
from pathlib import Path

import overmind
from overmind.core.paths import load_overmind_dotenv
from overmind.storage import configure_storage, get_storage, StorageNotConfiguredError

load_overmind_dotenv()
overmind.init()

agent_name = "<agent-name>"
base = Path(".overmind/agents") / agent_name / "setup_spec"
spec = json.loads((base / "eval_spec.json").read_text())
policy_md = (
    (base / "policies.md").read_text() if (base / "policies.md").is_file() else ""
)
datapoints = json.loads((base / "dataset.json").read_text())

configure_storage(agent_path=spec["agent_path"], agent_name=agent_name)
try:
    storage = get_storage()
except StorageNotConfiguredError as exc:
    raise SystemExit(
        f"Overmind backend not configured ({exc}). Set OVERMIND_API_KEY "
        "in .overmind/.env before running /overmind-optimize-agent."
    )

storage.save_spec(spec)
if policy_md:
    storage.save_policy(policy_md, spec.get("policy"))
ds_meta = storage.save_dataset(
    datapoints,
    source="local",
    metadata={"num_cases": len(datapoints), "synced_by": "overmind-optimize-agent"},
    make_active=True,
)
if not ds_meta:
    raise SystemExit(
        "Dataset upload failed — optimize would run against a stale or "
        "missing backend dataset. Fix the API configuration and re-run."
    )
print(
    f"Backend sync ok — agent_id={storage.get_agent_id()} "
    f"dataset_id={ds_meta['id']} version={ds_meta['version']} cases={ds_meta['num_datapoints']}"
)

# Persist the backend-assigned ``agent_id`` back into
# ``.overmind/agents.toml`` so ``optimize-step init`` (and its
# downstream terminal Job PATCH) can resolve it without another
# round-trip. ``save_agent`` preserves the existing entrypoint
# string and only patches the ``id`` column when it was empty.
agent_id = storage.get_agent_id()
if agent_id:
    from overmind.core.registry import _read_registry_entries, save_agent
    existing = next(
        (e for e in _read_registry_entries() if e.get("name") == agent_name),
        None,
    )
    entrypoint = (existing or {}).get("entrypoint") or ""
    if entrypoint and not (existing or {}).get("id"):
        save_agent(agent_name, entrypoint, id=agent_id)
        print(f"Persisted agent_id={agent_id} into .overmind/agents.toml")
```

If this push fails, stop the skill and report the concrete error (most often a missing or invalid `OVERMIND_API_KEY`). Do **not** proceed to `init` against a half-synced backend — the UI will show a `Job` with no spec / dataset and the optimize loop's scores will not surface against the right `Agent`.

### Backend Job lifecycle parity (Path B — automatic)

Path B emits the same OTel telemetry as Path A on a per-iteration / per-candidate basis (OTLP attribute set: `OPTIMIZE_ITERATION`, `OPTIMIZE_CANDIDATE_*`, `OPTIMIZE_ITERATION_DECISION`, `OPTIMIZE_STALL_COUNT`, `OPTIMIZE_ITERATION_AGENT_CODE`, `OPTIMIZE_ITERATION_SUGGESTIONS`, …). The Overmind UI receives these live during the run — the Job header, current iteration, baseline / best score and the per-iteration cards all update without any extra host code.

The two **terminal** writes that flip the Job out of `running` are also wired into Path B now:

- ``overmind optimize-step report`` calls ``ApiReporter.on_complete(best_score, baseline_score, report_markdown, best_agent_code)`` after rendering ``report.md``. Job status → ``completed``; the rendered ``report.md`` and best-agent code land on the Job row.
- The ``optimize-step`` CLI dispatcher catches ``BaseException`` (including ``KeyboardInterrupt``) and calls ``ApiReporter.on_failed(reason)``. Job status → ``failed``; the failure reason is appended to ``Job.report_markdown`` and ``Job.logs``.

Both PATCHes target the Job by ``(agent_id, job_id)``. ``job_id`` is minted by ``init`` and persisted in ``skill_state.json``; ``agent_id`` is resolved by ``init`` from (in order) ``.overmind/agents.toml`` → in-process storage → a fresh ``save_spec`` round-trip against the backend (and persisted back to ``agents.toml`` on success).

**Host coding agent does not need to call any REST endpoint or storage method manually** — both the sync block above and ``overmind optimize-step`` handle it. If the user reports a Job stuck on ``running`` after a clean run, check that:

1. ``skill_state.json`` carries a non-empty ``job_id`` and ``config.agent_id``.
2. The ``overmind optimize-step report`` step actually ran (its envelope returned ``status: ok``).
3. ``OVERMIND_API_KEY`` is set; without it the PATCH is silently skipped and only the OTLP path can update the UI.

If a step crashed mid-iteration and the Job is stuck, re-running ``overmind optimize-step report --state <STATE_PATH>`` will fire the terminal ``on_complete`` PATCH (provided ``best_score`` is set in the state). For an actually-failed run that should not be marked completed, PATCH the Job to ``failed`` directly via the REST API.

### Initialize optimization state

Follow **Configuration** above: the user must have chosen **Set optimization parameters** or **Run with defaults** before this step.

Create a settings JSON object that includes every `Config` field you collected (defaults path = tables’ default columns). **Run the ANALYZER_MODEL preflight script** from **Configuration** immediately before `init`. Then run `overmind optimize-step init <agent-name>` with the settings JSON on stdin.

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
1. Spawn **one subagent per candidate worktree** (up to `candidates_per_iteration`) so edits run in parallel — **required** when the host exposes parallel tasks or background agents.
1. Coordinator runs `evaluate` and `accept` for that iteration (or, if the host allows safe parallel shells only, delegate **per-candidate** `evaluate` to subagents, then coordinator assembles `candidate_results.json` and runs `accept` once).
1. Optionally **rotate** which subagent receives the next iteration’s edit workload so context stays fresh.

If the user wants maximum parallelism, increase **candidate** subagent count first. Use **parallel iteration coordinators** only for **separate** optimization runs (separate `STATE_PATH` or separate agents), never two iteration indices on the same state file.

### Diagnose and materialise candidate worktrees

Run `overmind optimize-step diagnose --state <STATE_PATH> --iteration <i>`.

The response should include candidate descriptors with a candidate ID, worktree path, prompt path, plan path, entry file metadata, focus area, and suggested edit method.

If the response has warning status and includes a diagnosis warning, stop the loop. Report the warning’s last error and hint. This usually indicates missing analyzer provider configuration or an invalid analyzer model. Do not manually proceed with placeholder edits.

Inspect each candidate prompt and plan enough to understand the intended edit. If a candidate targets the registered entrypoint, allow it only when the candidate prompt or optimizer scope clearly includes that file, and remind the coding agent to preserve importability, signature compatibility, and output contract stability.

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

Do not manually adjust scores. Do not skip weak candidates unless their worktree is missing or the evaluate command reports a terminal error.

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

Reject or repair candidate work before evaluation when it violates hard safety rules:

- It edits generated `.overmind` state or setup artifacts during optimization.
- It hardcodes exact dataset inputs, expected answers, IDs, or diagnosis examples.
- It adds lookup tables keyed by example values.
- It adds brittle `if`, `elif`, `match`, or regex branches that exist only to match known test examples.
- It deletes core agent behavior rather than improving it.
- It moves files out of the worktree.
- It modifies provider secrets or prints secret values.

Prefer to let evaluation catch quality regressions, but do not evaluate candidates that violate hardcoding, state-mutation, secret-handling, or worktree-boundary rules.

## Handling common failures

- **State already exists**: Ask whether to resume or start fresh. Use overwrite only with explicit approval.
- **Missing eval spec or dataset**: Stop and run or recommend `/overmind-generate-spec-and-dataset` (or `overmind setup <agent>`).
- **Backend sync failure before `init`**: `OVERMIND_API_KEY` is missing or invalid. Fix `.overmind/.env` and re-run the sync block; do not skip it.
- **Output schema may be incompatible**: Warn the user that scoring may be affected, then rely on optimize-step baseline or evaluation to confirm the actual failure.
- **Nested or list outputs**: Do not block up front. Let the evaluator determine whether the current eval spec can score them.
- **Analyzer warning**: Stop and report the warning’s last error and hint; usually provider configuration or model name is wrong.
- **Candidate worktree missing**: Mark that candidate failed and continue evaluating the others.
- **All candidates crash**: Report the iteration result, then follow the accept-step state about whether to continue.
- **No improvement for patience window**: Stop early when the accept step reports early stopping.

## User-facing updates

Give concise progress updates at these milestones:

- Prerequisites checked.
- Setup artifacts synced to the Overmind backend (`agent_id`, `dataset_id`, `dataset_version`).
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
