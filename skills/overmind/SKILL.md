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
1. **Fast path, then ratchet.** Instrumentation runs `scan` → MCP
   `plan_instrumentation` → parallel subagent fan-out per placement file →
   local `check` → smoke run → MCP `verify_instrumentation_spans`, targeting
   under 10 minutes for Tier 0 + Tier 1. Tier 2 evidence gaps close
   afterward via the punch-list ratchet loop against real traffic. Full
   detail: [references/instrumentation.md](references/instrumentation.md).
1. **Verify before calling it done.** The static gate is
   `overmind instrumentation check --plan-file <path> [--root <path>] [--format json]`
   (deterministic, no network). The pre-traffic gate is MCP
   `verify_instrumentation_spans(spans)` against a smoke-run JSONL — every
   task's `binding_source` must be `"declared"`.
   Against real traffic, retain the exact trace id and call the read-only
   MCP `verify_instrumentation_trace(agent, plan_id, trace_id)`; use
   `list_traces` → `get_trace` only for the complementary raw-span audit,
   never an unrelated newest trace.
1. **Declare tasks, don't guess the binding.** Use exactly one task root per
   trace. The plan's task root is required; zero or multiple task roots fail
   verification. `@overmind.task("key")` for fixed tasks; for a shared-entry
   dynamic route, `@overmind.task(key_from=selector)` — the selector runs
   before span creation and must return one registered, non-empty key from
   the plan's known key set. Shared helpers and ordinary useful spans are
   nested workflow/tool/function spans, never independent task roots or
   Behaviour anchors unless the plan explicitly identifies them. Identity
   boundaries use `overmind.capability(name=..., id=...)`; `name=` on
   `workflow`/`tool`/`retrieval`/`function` anchors stable separating
   symbols. The context-manager form of `task()` is only for a fixed-key
   dynamic boundary with explicit `entrypoint=` metadata; it is not
   equivalent to the decorator for code identity or I/O capture. A declared
   key is strong evidence, but revision mismatch and unknown anchors remain
   verification failures; without one the server falls back to structural
   matching, which can stay `unbound`. Decorators capture I/O by default;
   use `capture_io=False` only for an explicit no-payload requirement.

## Instrumentation fast path (digest)

The complete command sequence; details in references/instrumentation.md —
read it once, not per step.

1. `list_behaviours` per capability. Populated registry → placements come
   from `get_instrumentation_context`; empty → steps 2-3.
2. `overmind instrumentation scan --root . --out candidates.json`
3. MCP `plan_instrumentation(candidates)` — send candidates.json content
   verbatim. Returns placements (with `required_identity`) + `ambiguous` +
   `dropped` (report dropped).
4. One subagent per placement file, all at once: apply
   `required_task_decorator` at `target.qualname`, add `target.import_line`,
   wire `required_identity` (init with agent_id/agent_name, or
   `overmind.capability(id=...)`). Lead handles `ambiguous` (key_from).
5. `overmind instrumentation check --plan-file plan.json`
6. Smoke scripts per task from `smoke_hint`; run with `OVERMIND_SMOKE=1
   OVERMIND_TRACE_FILE=spans.jsonl` (in-repo paths; no API key needed;
   never the real app).
7. `overmind instrumentation verify --spans-file spans.jsonl` — posts the
   spans to the server binder for you (never inline a large span array into
   a tool call). Gate: every task `binding_source == "declared"`; exit 0 is
   the pass signal. Tier 2 items go to the punch list, not this pass.
8. Report the per-stage timing table.

## Use-case references

- Instrumenting an application (greenfield or alongside existing telemetry):
  [references/instrumentation.md](references/instrumentation.md)
  (fast-path: `scan` → `plan_instrumentation` → subagent fan-out → `check` →
  smoke → `verify_instrumentation_spans`; ratchet loop against real traffic
  after)
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
   [instrumentation.md](references/instrumentation.md) — run the fast path
   (`scan` → `plan_instrumentation` → subagent fan-out → `check` → smoke →
   `verify_instrumentation_spans`), then ratchet Tier 2 evidence against
   real traffic.
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
