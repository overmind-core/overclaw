---
name: overmind
description: Operate the Overmind platform through its MCP server — add tracing to a Python AI/LLM project, inspect telemetry (traces, sessions, agent health, failures, context graph), upload and clean datasets, author evaluators / eval sets / eval runs, fine-tune models, and launch optimizer experiments. Use when the user mentions Overmind, adding telemetry/observability/tracing, traces landing, finetuning or training a model, uploading datasets, eval runs, evaluators, dataset quality, prompt/agent optimization, or when Overmind MCP tools are available.
---

# Overmind platform MCP

Overmind is an agent observability and optimization platform. It ingests traces
from LLM agents, turns them into datasets, grades them with evaluators, and
uses those datasets to fine-tune models and optimize agent prompts/code.

This skill covers the common Overmind workflows: instrumenting applications
with tracing, inspecting telemetry, datasets, evals, fine-tuning, and
optimizer experiments.

## Core principles

Follow these for ALL Overmind MCP work:

1. **MCP only.** All platform work goes through the Overmind MCP server. Do
   not curl REST endpoints, do not invent base URLs, and do not hardcode
   hosts. The server is already configured (plugin, or `overmind init`) and
   scoped to one project via the API key in its headers. Call the named
   tools; inspect each tool's schema for arguments. If tools are missing,
   tell the user to run `overmind init` (or re-check MCP config /
   `OVERMIND_API_KEY`). Do not paste a URL or ask them to paste the raw key
   into chat.
   Read-only REST fallback, only when no MCP server is configured (e.g. a
   machine where `overmind init` was never run still needs to instrument):
   equivalent reads exist at `GET /api/behaviours/…` and
   `GET /api/task-executions/…` with the project API key in an `X-Api-Key`
   header. Prefer MCP whenever it is configured; never use REST for writes.
1. **Reference file per use case.** Check the relevant reference below before
   implementing. This file holds conventions that apply everywhere; the
   workflow lives in the reference.
1. **Names, not ids.** Tools take human names resolved against the project.
   Get them from the matching `list_*` tool first. UUIDs work as a fallback.
   Never paste raw UUIDs to the user when a name/slug exists.
1. **Intent gates every dataset workflow.** Eval runs and optimizer
   experiments need **eval** intent; fine-tuning needs **ft** + model
   surface. Read the intent section below before creating or picking a
   dataset.
1. **Errors are values; mutations run immediately.** Every tool returns
   `{"error": "..."}` instead of raising — follow the `hint` when present.
   There is no confirmation gate, so verify arguments (and ask the user when
   destructive) before create/delete/cancel.
1. **Verify with a real trace.** Instrumentation isn't done when the code
   compiles — it's done when you have fetched the trace you just sent via
   MCP (`list_traces` → `get_trace`) and it carries everything the baseline
   in [references/instrumentation.md](references/instrumentation.md) requires.
   For trajectory instrumentation the execution row is the gate, not just
   raw spans: fetch it (`list_task_executions` → `get_task_execution`) and
   confirm `binding_source` is `anchor_join`/`declared`/`structural` (not
   `unbound`), check `attribution_verdict` / `binding_confidence` (never an
   `unbound_*` verdict or `bound_low_conf`), `user_intent` is right, and
   `success_score`/`session_score` populated; pull `behaviour_coverage` to
   confirm every step evaluator got evidence.
1. **One agent at a time.** Instrumentation tasks map to agents. Resolve the
   agent's identity and capability card from `get_agent` (the `id` UUID
   verbatim) and scope changes to that agent's files
   ([references/instrumentation.md](references/instrumentation.md) Step 0 /
   5b). For repo-wide tasks, run the systematic one-at-a-time pass (Step 5c)
   — never a giant all-agents-at-once edit.
1. **Trajectory-aware instrumentation.** The platform treats **task
   executions** as the primary observability rows: spans bind to Behaviour
   anchors by code identity (`code.namespace` + `code.function.name`) and git
   sha (`vcs.ref.head.revision`), and a runtime envelope scores each
   execution. Resolve the agent via `get_agent` (the capability card now also
   carries trajectory paths), pull `get_instrumentation_context` to see which
   anchors are still `remaining` vs already `instrumented`, instrument the
   task entry points + remaining anchors, and emit the envelope — `intent` at
   turn boundaries, `checkpoint` at milestones, `expect` per task contract,
   `eval_context` facts, `end_conversation` at completion. Verify with
   `list_task_executions` / `get_task_execution` (`binding_source`,
   `user_intent`, `success_score`) + `behaviour_coverage` before moving to
   the next agent.
1. **Declare tasks, don't guess the binding.** The trace binds to the right
   task/trajectory by contract, not guesswork: resolve the agent with
   `get_agent` (copy the `id` verbatim), map it to its tasks with
   `list_behaviours`, and declare the behaviour key with
   `@overmind.task("<behaviour key>")` on the task entry point (or the
   context-manager form) and `name=` on separating anchors. A declared key
   binds even when the git sha is missing or unanalyzed; without one the
   server falls back to structural matching, which can stay `unbound`.

## Use-case references

- Instrumenting an application (greenfield or alongside existing telemetry):
  [references/instrumentation.md](references/instrumentation.md)
  (`get_instrumentation_context` shows which behaviour anchors are `remaining`)
- Inspecting traces, sessions, agent health, failures, the context graph, and
  connectors (including the post-setup verification loop):
  [references/telemetry.md](references/telemetry.md)
  (`list_task_executions` / `get_task_execution` / `behaviour_coverage` /
  `behaviour_deviations` / `list_behaviours`)
- Uploading / building datasets (from traces, failures, or an attached file)
  and cleaning them in the workshop:
  [references/datasets.md](references/datasets.md)
- Authoring evaluators, grouping them into eval sets, running and comparing
  eval runs:
  [references/evals.md](references/evals.md)
- Fine-tuning a model (prerequisites, recommended-model sweep, deploy, swap PR):
  [references/finetuning.md](references/finetuning.md)
- Optimizer experiments (prompt/agent search via the local executioner):
  [references/optimizer.md](references/optimizer.md)

## Conventions (read before any workflow)

- **List first.** `list_datasets`, `list_agents`, `list_eval_sets`,
  `list_evaluators`, `list_eval_runs`, `list_finetune_jobs`,
  `list_deployed_models`, `list_optimizer_experiments`, `list_traces`,
  `list_sessions`, `list_behaviours`, `list_task_executions`. Then pass
  `dataset_name`, `eval_set_name`,
  `evaluator_names`, `agent_name_or_slug`, `eval_run_name`.
- **Async jobs.** Launch tools return `job_status: {kind, id}` with kind one
  of `eval_run`, `finetune_job`, `optimizer_experiment`. Poll with
  `job_status(kind, id)` until terminal (eval runs: completed / failed /
  cancelled; finetune: succeeded / failed / cancelled). `wait_for_job` is
  built for the Console chat's resume machinery — from a coding agent, poll
  `job_status` instead.
- Chat-UI-only helpers (`propose_plan`, `suggest_navigation`) are not exposed
  on MCP.

## Dataset types (intent) — read first, it gates every workflow

Every dataset has an immutable `intent`, assigned once at ingestion
(`list_datasets` shows it):

- **`eval`** ("Eval") — rows shaped `{input, expected_output?, extra}`.
- **`ft`** ("Train") — rows shaped for SFT: `input = {messages, tools?}`.
- **`unstructured`** ("Raw") — verbatim rows, no structural guarantees.

What each workflow accepts:

- **Eval runs** (`create_eval_run`) require an **eval**-intent dataset.
- **Optimizer experiments** (`create_optimizer_experiment`) require an
  **eval**-intent dataset.
- **Fine-tuning** (`create_finetune_job`) requires a **ft** (Train) intent
  training dataset — and additionally a `model`-surface one (LLM-in → LLM-out
  rows, not agent-level rows) — plus a separate **eval** dataset for
  in-training judge evals.

Intent never mutates; converting writes a NEW dataset (`derived_from` points
back). If a tool rejects a dataset for intent, re-ingest or pick another —
don't retry the same one. `analyze_dataset_file` infers intent from content;
`create_dataset_from_file` accepts an explicit `intent` override
(`eval | ft | unstructured`). Trace-built datasets get intent and surface
assigned at import. A dataset being read by a running job is frozen until the
job ends.

### Don't confuse runtime `intent()` with dataset intent

`overmind.intent("…")` at runtime declares what the *user* asked for on a
trace and grounds the judge's scoring of that execution — it is unrelated to
dataset intent. Dataset `intent` (`eval | ft | unstructured`) is an immutable
property assigned at ingestion that gates which workflows may use the dataset
(above). Sharing the word "intent" is the only link: calling `intent()` in
code does not change a dataset's intent, and a dataset's `eval` intent does
not count as a runtime intent on a trace.

## How the workflows chain

Typical loop, always via MCP:

1. **See what's happening** — [telemetry.md](references/telemetry.md)
   (`agent_health` → `agent_failures` → `list_traces` / `get_trace`, or the
   task-execution rows via `list_task_executions` / `get_task_execution`).
   Add
   tracing first if nothing is landing:
   [instrumentation.md](references/instrumentation.md) — resolve each
   agent's identity with `get_agent`, see which anchors are uninstrumented
   via `get_instrumentation_context`, and instrument one at a time.
1. **Turn traces into data** — [datasets.md](references/datasets.md)
   (`create_dataset_from_failures` or `create_dataset_from_file`).
1. **Clean it** — workshop in [datasets.md](references/datasets.md).
1. **Grade it** — [evals.md](references/evals.md) (baseline `create_eval_run`
   on an **eval**-intent dataset).
1. **Improve** — [finetuning.md](references/finetuning.md) (**ft** dataset;
   recommended-model sweep) or [optimizer.md](references/optimizer.md)
   (**eval** dataset + connected local executioner).
1. **Prove it** — `compare_eval_runs` new vs baseline
   ([evals.md](references/evals.md)).
1. **Ship** — `create_model_swap_pr` or `create_optimizer_pr`.
