---
name: overmind
description: Operate the Overmind platform through its MCP server — fine-tune models, upload and clean datasets, create evaluators / eval sets / eval runs, and launch optimizer experiments. Use when the user mentions Overmind, finetuning or training a model, uploading datasets, eval runs, evaluators, dataset quality, prompt/agent optimization, or when Overmind MCP tools are available.
---

# Overmind platform MCP

Overmind is an agent observability and optimization platform: it ingests traces
from LLM agents, turns them into datasets, grades them with evaluators, and
uses them to fine-tune models and optimize agent code.

This plugin registers a remote MCP server named `overmind` at
`https://api.overmindlab.ai/api/mcp/` (Streamable HTTP), authenticated with a
project API key sent as `X-Api-Key` (from the `OVERMIND_API_KEY` env var). The
key is pinned to ONE project; every tool call is scoped to it.

## Setup

1. Create a project API key in Console → Projects → API keys.
1. Install this plugin (or add the MCP server manually).
1. Set **OVERMIND_API_KEY** in the plugin Configure panel (or your agent's MCP
   headers). Never paste the raw key into chat.

Local / self-hosted: point the MCP URL at `{API_BASE}/api/mcp/`.

## Conventions (read before any workflow)

- **Names, not ids.** Tools take human names — `dataset_name`, `eval_set_name`,
  `evaluator_names`, `agent_name_or_slug`, `eval_run_name` — resolved against
  the project. Get them from the matching `list_*` tool first
  (`list_datasets`, `list_agents`, `list_eval_sets`, `list_evaluators`,
  `list_eval_runs`, `list_finetune_jobs`, `list_deployed_models`,
  `list_optimizer_experiments`). UUIDs work as a fallback.
- **Errors are values.** Every tool returns `{"error": "..."}` instead of
  raising — check for it and follow the `hint` field when present.
- **Mutations run immediately.** Over MCP there is no confirmation gate, so
  verify arguments (and ask the user when destructive) before calling
  create/delete/cancel tools.
- **Async jobs.** Launch tools return
  `job_status: {kind, id}` with kind one of `eval_run`, `finetune_job`,
  `optimizer_experiment`. Poll with `job_status(kind, id)` until terminal
  (eval runs: completed/failed/cancelled; finetune: succeeded/failed/cancelled).
  `wait_for_job` exists but is built for the Console chat's resume machinery —
  from a coding agent, poll `job_status` instead.
- Chat-UI-only helpers (`propose_plan`, `suggest_navigation`) are not exposed
  on MCP.
- For read-only diagnosis start with `agent_health`, `agent_failures`,
  `list_traces` / `get_trace`, and the `graph_*` tools.

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

## Uploading datasets

### From a local file

1. Stage the file (CSV/TSV/JSON/JSONL) via REST — there is no MCP upload tool:
   `POST {API_BASE}/api/brain/chat/attachments/` with `X-Api-Key`, multipart
   fields `project` (project UUID) and `file`. The response `id` is the
   `attachment_id`.
1. `analyze_dataset_file(attachment_id, intent?)` — inferred intent, field
   mapping, per-intent viability, before→after preview. Nothing persisted.
1. Review the proposal with the user; pass `intent` explicitly if the
   inference is wrong.
1. `create_dataset_from_file(attachment_id, dataset_name?, agent_name?, intent?)` — same ingest path as the upload wizard; returns the created
   dataset's name, intent, and `num_datapoints`.

### From traces / failures

- Failures → dataset in one step: `create_dataset_from_failures( agent_name_or_slug, dataset_name, since_days?, limit?)`. Prefer this over
  manually plumbing trace ids.
- Specific or recent traces: `create_dataset_from_traces(agent_name_or_slug, trace_ids?)` — or omit `trace_ids` and pass `last_n` for the N most recent.
- Append to an existing mutable dataset: `add_traces_to_dataset(dataset_name, trace_ids)`.
- Remove rows: `remove_datapoints_from_dataset(dataset_name, datapoint_ids)`
  (destructive).

### Cleaning / editing (the workshop)

1. `open_workshop(dataset_name)` once (starts the first analysis; no-op after),
   or `analyze_workshop_dataset` to force a fresh analysis.
1. `workshop_state(dataset_name)` — score, verdict, failing checks, open
   insights, any staged change. Start here for dataset-quality questions.
1. For each insight: `workshop_insight_detail(insight)` →
   `apply_insight_fix(insight)` (STAGES a change) →
   `workshop_staged_diff(dataset_name)` → `approve_staged_change` to commit or
   `discard_staged_change` to undo.
1. Direct edits (each lands as a commit): `workshop_edit_row(dataset_name, dp_id, field, value)`, `workshop_delete_rows(dataset_name, dp_ids)`,
   `workshop_batch_edit(dataset_name, operations, message?)` (staged, needs
   approval), and `preview_find_replace` → `workshop_find_replace`.
1. Inspect: `workshop_export_rows` (rows at HEAD or a sha),
   `workshop_columns` (per-column stats), `workshop_commits` /
   `workshop_commit_diff` (history), `workshop_compare_datasets` (two datasets,
   same intent). Undo history with `restore_dataset_commit(dataset_name, sha)`.
1. Failing checks you accept: `waive_dataset_check(dataset_name, check_key)` /
   `unwaive_dataset_check`.

## Evals

### 1. Author evaluators

- `list_evaluators` first — reuse before creating.
- Whole suite for an agent: `generate_evaluators(agent_name_or_slug)` — authors
  grounded judges from the agent's codebase and merges them into its Default
  eval set. Runs in the background.
- Fit check for a dataset: `dataset_eval_capabilities(dataset_name)` —
  applicable kinds/scopes + recommended evaluators. Suggestions without
  persisting: `generate_dataset_evals(dataset_name)`.
- One judge: draft with `generate_evaluator_prompt(description, agent_name_or_slug?, applicable_role?)` or `compile_rubric(rubric_md, score_type?)` (both spend LLM credits, persist nothing), then save with
  `create_judge_evaluator(name, evaluation_prompt, score_type [numeric|boolean|categorical], categories? [required for categorical], agent_name_or_slug?, applicable_roles?)`.
- Verify what a judge reads: `preview_evaluator_prompt(evaluator, trajectory?, structured?, expected?)`.

### 2. Group into eval sets

1. `create_eval_set(agent_name_or_slug, name, activate?)`.
1. `add_evaluators_to_eval_set(eval_set_name, evaluator_names, role?)` —
   `role="generative"` grades eval-run outputs; `role="trace_scoring"` on the
   agent's ACTIVE set binds them as live scorers on incoming traces.
1. `activate_eval_set(eval_set_name)` — makes it the agent's active set (the
   default for eval runs, finetune jobs, and optimizer experiments).

### 3. Run

`create_eval_run(name, dataset_name, eval_set_name? | evaluator_names?, max_items?)` — creates AND launches. The dataset must be **eval** intent.
Evaluator precedence: `evaluator_names` > `eval_set_name` > the dataset
agent's active set (error if none). Over MCP the run scores the dataset's
captured rows as a single baseline variant.

### 4. Monitor and analyze

- Poll `job_status(kind="eval_run", id=<run name or uuid>)`;
  `cancel_eval_run(eval_run_name)` to stop.
- `get_eval_run(eval_run_name)` — status + aggregated summary.
- `eval_run_comparison(eval_run_name)` — variants, progress, per-evaluator
  rollup.
- `compare_eval_runs(eval_run_name, baseline_run_name)` — authoritative
  run-vs-run per-evaluator deltas.
- `eval_run_trend(eval_run_name)`, `evaluator_score_history( agent_name_or_slug)`, `eval_score_trends(days?, agent?, evaluator?)` — time
  series.
- Eval-variant models: `model_catalog(search?)`, `list_model_refs`,
  `create_model_ref(model_id, provider, label?, base_url?, api_key_ref?)`.

## Fine-tuning (training)

Run a baseline eval on your eval dataset BEFORE training so you have a
comparison point (see Evals above).

1. `list_datasets` — pick a **ft**-intent, model-surface training dataset, or
   build one from traces (see Uploading datasets).
1. `finetune_prerequisites(dataset_name, agent_name_or_slug?)` — ALWAYS call
   before launching. Returns `ready`/`missing` (train-intent check, surface
   check, validation, agent, default eval dataset, default eval set),
   train/eval `overlap_count`, `recommendations` (model + hyperparams +
   cost/time), and the trainable `catalog`. Fix everything in `missing` first;
   `base_model` MUST be a catalog id from this result — never invent one.
1. Optional preflight: `validate_finetune_dataset(dataset_name, ...)` for the
   full format report, `finetune_dataset_overlap(dataset_name, eval_dataset_name)` for contamination, `estimate_finetune_cost(dataset_name, base_model, n_epochs?, use_lora?)` to confirm cost with the user.
1. `create_finetune_job(dataset_name, base_model, name?, agent_name_or_slug?, eval_dataset_name?, eval_set_name?, hyperparameters?, model_tier?)` —
   creates AND queues. Eval dataset/set and hyperparameters default from the
   prerequisites report when omitted. Returns `finetune_job_id`.
1. Monitor: `job_status(kind="finetune_job", id=finetune_job_id)`,
   `finetune_job_events(finetune_job)` for the event log,
   `finetune_loss_curves(finetune_job)` for train/eval loss, token accuracy,
   progress percent and ETA. `cancel_finetune_job(finetune_job_id)` to stop;
   `retry_finetune_job(finetune_job)` re-queues a failed/cancelled job.
1. On success (status `succeeded`): `list_deployed_models` — the trained model
   registers for inference. `deploy_model(deployed_model_id)` (re-)registers
   and retries FAILED deployments; `retry_model_deploy` /
   `deployed_model_checkpoints` / `deployed_model_metrics` /
   `deployed_model_activity` / `deployed_model_live` cover deployment health.
   `undeploy_model` clears routing (weights kept).
1. Smoke-test: `run_inference(model=<ft-… serving id or DeployedModel UUID>, messages=[...])` against a READY deployment.
1. Verify quality: `create_eval_run` on the eval dataset, then
   `compare_eval_runs` against the pre-training baseline.
1. Ship: `create_model_swap_pr(finetune_job, pin?)` — opens a GitHub PR
   pointing the agent's code at the fine-tuned model (needs a SUCCEEDED job, a
   deployed model, and a linked GitHub repo; `analyze_github_repo` /
   `analyze_github_repo_url` link one). Runs in the background — follow with
   `finetune_job_events`.

## Optimizer (prompt/agent optimization)

Experiments execute on the user's machine through a LOCAL executioner CLI —
nothing runs until one is connected.

1. `optimizer_prerequisites(agent_name_or_slug)` — ALWAYS call first. Returns
   usable **eval**-intent datasets, the agent's active eval set,
   `executioner_connected`, and (when disconnected)
   `executioner_start_command`.
1. If disconnected, show the user `executioner_start_command` in a code block
   (run from the agent's repo), then re-check with `optimizer_connection`.
1. `create_optimizer_experiment(agent_name_or_slug, dataset_name, eval_set_name?, num_iterations? [2–5, default 5], num_candidates_per_iteration? [2–3, default 3], max_iterations_without_improvement? [default 3])` — eval set defaults to
   the agent's active set. Returns `experiment_id`.
1. Monitor: `job_status(kind="optimizer_experiment", id=experiment_id)` for
   iteration progress; `optimizer_iterations(experiment_id)` for per-round
   scores and candidates; `optimizer_candidate_detail(candidate_id)` for a
   candidate's scores, eval-run linkage, and the iteration's winning patch.
   `cancel_optimizer_experiment(experiment_id)` to stop.
1. Land it: `create_optimizer_pr(experiment_id)` — opens a GitHub PR with the
   winning diff. Requires a COMPLETED optimize-mode experiment whose best
   score beat baseline and a linked GitHub repo; no executioner is needed just
   to open the PR.

## How the workflows chain

Traces → dataset (`create_dataset_from_failures`) → clean it (workshop) →
evaluators + eval set → baseline `create_eval_run` → then either
`create_finetune_job` (ft dataset) or `create_optimizer_experiment` (eval
dataset) → `compare_eval_runs` new vs baseline → ship via
`create_model_swap_pr` or `create_optimizer_pr`.

## Related

- Telemetry / REST ingest and verification:
  [overmind-agent-telemetry](../overmind-agent-telemetry/SKILL.md)
- Docs index: https://docs.overmindlab.ai/llms.txt
