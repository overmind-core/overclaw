# Fine-tuning (training)

SFT one or more catalog base models on an **ft** / model-surface dataset,
keep a held-out **eval** dataset for judges, then deploy and prove the
winner beats the agent's production incumbent.

Training dataset: **ft** intent AND `model` surface (LLM-in → LLM-out rows,
not agent-level rows). Eval dataset: **eval** intent. See
[SKILL.md](../SKILL.md#dataset-types-intent--read-first-it-gates-every-workflow)
and [datasets.md](datasets.md).

Do **not** create a manual pre-training baseline eval run. The platform
scores the agent's production incumbent automatically and shares one
baseline EvalRun across a `group_id` sweep — that is what the monitor
delta and post-training comparison already use.

All of this is Overmind MCP. Poll `job_status` per job (see conventions in
[SKILL.md](../SKILL.md)). Inspect each tool's schema for arguments.

A training **run** is a sweep: one `create_finetune_job` per selected catalog
model, sharing a `group_id`. Cap the sweep at **4** models (wizard parity).
Do not invent a fifth.

## Workflow

```
- [ ] 1. list_datasets — ft + model-surface train set, plus a separate eval-intent set
- [ ] 2. finetune_prerequisites — ALWAYS; fix everything in missing first
- [ ] 3. Present recommended models (prerequisites.recommendations / finetune_recommendation);
         selected:true is the top pick only — ask how many to train (cap 4)
- [ ] 4. Optional: estimate_finetune_cost per pick; confirm total spend
- [ ] 5. create_finetune_job once per selected model, same group_id (one training run)
- [ ] 6. Poll each job_status; finetune_loss_curves / finetune_job_events until terminal
- [ ] 7. list_deployed_models — deploy is automatic on succeeded; retry only if failed
- [ ] 8. run_inference smoke-test winners (wait for ready; cold start is not failure)
- [ ] 9. create_eval_run then compare_eval_runs vs the automatic incumbent baseline
- [ ] 10. Ship: create_model_swap_pr (alias once per agent) or pin=true / Console Make live
```

## Steps

1. `list_datasets` — pick a **ft**-intent, model-surface training dataset, or
   build one from traces / a file. `list_finetune_jobs` /
   `list_finetune_base_models` show what's already been tried.
1. `finetune_prerequisites(dataset_name, agent_name_or_slug?)` — **ALWAYS**
   call before launching. Returns `ready` / `missing` (train-intent check,
   surface check, validation, agent, default eval dataset, default eval set),
   train/eval `overlap_count`, `recommendations` (ranked models + hyperparams
   - cost/time), and the trainable `catalog`. Fix everything in `missing`
     first. Do not launch while `ready` is false.
1. **Recommended models — show these before creating anything.** The
   prerequisites `recommendations` array is a ranking: `model` (catalog id),
   `display_name`, `tier`, `grade`, `confidence`, `cost_usd`, `time_human`,
   `hyperparams`, `selected`. Exactly one row has `selected: true` — that is
   the recommender's top pick, not a multi-model default sweep. For the full
   ranking plus benchmark evidence, also call
   `finetune_recommendation(dataset_name, agent_name_or_slug?)`. Present
   **only** ids from `recommendations` / `catalog` — never invent OpenAI
   API ids or Llama-2 names. Ask the user how many to train (one or
   several). Cap at 4. If they don't narrow it, launch the top pick
   (`selected: true`) unless they approve a wider top-N.
1. Optional preflight: `validate_finetune_dataset(dataset_name, ...)` for the
   full format report; `finetune_dataset_overlap(dataset_name, eval_dataset_name)` for contamination;
   `estimate_finetune_cost(dataset_name, base_model, n_epochs?, use_lora?)`
   **per selected model** so the user sees sweep cost;
   `agent_base_model_throughput` for tokens/sec on the agent's current base
   model (speed comparison later).
1. **Launch a sweep, not one arbitrary model.** `create_finetune_job` trains
   one catalog id per call. For multiple picks, call it once per
   `base_model`, reusing `group_id` from the first result on the rest so
   they share one training run (wizard parity). Same `dataset_name` / eval
   dataset / eval set on every call; omit hyperparameters unless
   overriding — they stamp from the recommender per model. Each call
   returns `finetune_job_id` + `group_id`. A lone pick is just one call.
   **Caveat:** `group_id` is accepted by the handler but **not declared** in
   the tool schema — pass it verbatim from the first job's result; do not
   expect it in the schema listing.
1. Monitor: `job_status(kind="finetune_job", id=finetune_job_id)` **per
   job**; `finetune_job_events(finetune_job)` for the event log,
   `finetune_loss_curves(finetune_job)` for train/eval loss, token accuracy,
   progress percent and ETA. `list_finetune_jobs` to see the whole run.
   `cancel_finetune_job(finetune_job_id)` to stop one job;
   `retry_finetune_job(finetune_job)` re-queues a failed/cancelled job.
   Terminal statuses: `succeeded` / `failed` / `cancelled`.
1. On success: deployment is **automatic** — `list_deployed_models` to find
   the new serving ids. `deploy_model` / `retry_model_deploy` are the retry
   path for `failed` / `deleted` deployments, not a required post-success
   step. `deployed_model_checkpoints` /
   `deployed_model_metrics` / `deployed_model_activity` /
   `deployed_model_live` cover deployment health. `undeploy_model` clears
   routing (weights kept).
1. Smoke-test: `run_inference(model=<ft-… serving id or DeployedModel UUID>, messages=[...])`
   against a **ready** deployment. Inference returns **503** until status is
   `ready`; a dormant model pays a cold start on the first call — a slow
   first response is not a failure. Do not invent a completion.
1. Verify quality: `create_eval_run` on the eval dataset, then
   `compare_eval_runs` against the automatic incumbent baseline
   ([evals.md](evals.md)). Pick the winning job of the sweep from that
   comparison.
1. Ship: `create_model_swap_pr(finetune_job, pin?)` — opens a GitHub PR
   pointing the agent's code at the **winning** fine-tuned model (needs a
   SUCCEEDED job, a deployed model, and a linked GitHub repo;
   `analyze_github_repo` / `analyze_github_repo_url` link one —
   [agents.md](agents.md)). The **alias
   PR is once per agent**: after one merged PR, calling without `pin` is
   refused for later jobs. For a second or later fine-tune, either
   `pin=true` (writes the concrete `ft-…` id) or tell the user to promote
   it in the Console ("Make live") or `update_agent(agent_name_or_slug, active_model=<READY deployed-model UUID>)` ([agents.md](agents.md)).
   Runs in the background —
   follow with `finetune_job_events`. `job_status(kind="model_pr", id=<finetune_job uuid>)` tracks the swap PR.
