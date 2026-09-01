---
name: overmind
description: Operate the Overmind platform through its MCP server — add tracing to a Python AI/LLM project, inspect telemetry (traces, sessions, agent health, failures, context graph), upload and clean datasets, author evaluators / eval sets / eval runs, fine-tune models, and launch optimizer experiments. Use when the user mentions Overmind, adding telemetry/observability/tracing, traces landing, finetuning or training a model, uploading datasets, eval runs, evaluators, dataset quality, prompt/agent optimization, or when Overmind MCP tools are available.
---

# Overmind platform MCP

Overmind is an agent observability and optimization platform. It ingests traces
from LLM agents, turns them into datasets, grades them with evaluators, and
uses those datasets to fine-tune models and optimize agent prompts/code.

This skill covers the common Overmind workflows: resolving agents,
telemetry (instrument and inspect traces), datasets, evals, fine-tuning,
and optimizer experiments.

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
1. **Reference file per use case.** Check the relevant reference below before
   implementing. This file holds conventions that apply everywhere; the
   workflow lives in the reference.
1. **Names, not ids.** Tools take human names resolved against the project.
   Get them from the matching `list_*` tool first. UUIDs work as a fallback.
   Never paste raw UUIDs to the user when a name/slug exists. Exception:
   stamp `list_capabilities` / `get_capability` `id` (bare UUID) into the SDK, and
   pass a READY deployed-model UUID to `update_capability(..., active_model=)`
   (empty clears it). See [references/capabilities.md](references/capabilities.md).
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
   in [references/telemetry.md](references/telemetry.md#what-a-good-trace-carries) requires.
1. **One agent at a time.** Instrumentation tasks map to agents. Resolve the
   agent's identity and capability card from `get_capability` — top-level `id`
   (bare UUID, stamp verbatim), `active_model`, `source_repo`, capped
   `flow` (`flow_truncated` when clipped) — and scope changes to that
   agent's files ([references/capabilities.md](references/capabilities.md),
   [references/telemetry.md](references/telemetry.md) Step 0 /
   5b). For repo-wide tasks, run the systematic one-at-a-time pass (Step 5c)
   — never a giant all-agents-at-once edit.

## Use-case references

- Resolving / updating agents, prompts, eval spec, and GitHub analyze:
  [references/capabilities.md](references/capabilities.md)
- Telemetry (add tracing, inspect traces / sessions / health, connectors):
  [references/telemetry.md](references/telemetry.md)
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

- **List first.** `list_datasets`, `list_capabilities`, `list_eval_sets`,
  `list_evaluators`, `list_eval_runs`, `list_finetune_jobs`,
  `list_deployed_models`, `list_optimizer_experiments`, `list_traces`,
  `list_sessions`. Then pass `dataset_name`, `eval_set_name`,
  `evaluator_names`, `capability_name_or_slug`, `eval_run_name`.
  `list_capabilities` / `get_capability` return top-level `id` (bare UUID) and
  `active_model`; `get_capability` also `source_repo` and a capped `flow`
  (`flow_truncated` when clipped).
- **Async jobs.** Launch tools return `job_status: {kind, id}`. Kinds:
  `eval_run`, `finetune_job`, `optimizer_experiment`, plus poll-only
  `model_pr`, `dataset_analysis`, `agent_discovery`. Poll with
  `job_status(kind, id)` until terminal (eval runs: completed / failed /
  cancelled; finetune: succeeded / failed / cancelled; model_pr: open /
  failed; dataset_analysis: finished / error / cancelled; agent_discovery:
  ready / error). `wait_for_job` is built for the Console chat's resume
  machinery — from a coding agent, poll `job_status` instead.
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

Intent never mutates. Convert intent or surface by creating a NEW dataset
(`source_dataset` / `derived_from`) via `create_dataset_from_file`. If a
tool rejects a dataset for intent, pick another or re-ingest; don't retry
the same one. Train+eval split (`split_eval_fraction` + `split_method`) and
`surface` (`agent` | `model`) are available on
`create_dataset_from_file` / `create_dataset_from_traces`.
`analyze_dataset_file` infers intent from content. A dataset being read by
a running job is frozen until the job ends.

## How the workflows chain

Typical loop, always via MCP:

1. **See what's happening** — [telemetry.md](references/telemetry.md)
   (`capability_health` → `capability_failures` → `list_traces` / `get_trace`).
   Resolve / retarget agents via [capabilities.md](references/capabilities.md)
   (`list_capabilities` / `get_capability` / `update_capability`;
   `analyze_github_repo` if none exist). If nothing is landing, add tracing
   in the same file — stamp each agent's `id` and instrument one at a time.
1. **Turn traces into data** — [datasets.md](references/datasets.md)
   (`create_dataset_from_failures` or `create_dataset_from_file`).
1. **Clean it** — workshop in [datasets.md](references/datasets.md).
1. **Grade it** — [evals.md](references/evals.md) when you want an
   eval-vs-eval comparison you drive yourself. Finetune and optimizer runs
   create their own incumbent / experiment baselines automatically — do not
   spend a manual `create_eval_run` just to give them a comparison point.
1. **Improve** — [finetuning.md](references/finetuning.md) (**ft** dataset;
   recommended-model sweep) or [optimizer.md](references/optimizer.md)
   (**eval** dataset + connected local executioner).
1. **Prove it** — `compare_eval_runs` new vs the automatic baseline
   ([evals.md](references/evals.md)).
1. **Ship** — `create_model_swap_pr` or `create_optimizer_pr`.
