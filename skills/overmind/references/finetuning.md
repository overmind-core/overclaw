# Fine-tuning (training)

SFT one or more catalog base models on an **ft** / model-surface dataset,
keep a held-out **eval** dataset for judges, then deploy and prove the
winner beats a pre-training baseline.

Training dataset: **ft** intent AND `model` surface (LLM-in → LLM-out rows,
not agent-level rows). Eval dataset: **eval** intent. See
[SKILL.md](../SKILL.md#dataset-types-intent--read-first-it-gates-every-workflow)
and [datasets.md](datasets.md).

Run a baseline eval on the eval dataset **before** training so you have a
comparison point — [evals.md](evals.md).

All of this is Overmind MCP. Poll `job_status` per job (see conventions in
[SKILL.md](../SKILL.md)). Inspect each tool's schema for arguments.

A training **run** is a sweep: one `create_finetune_job` per selected catalog
model, sharing a `group_id`. Do not default to a single model.

## Workflow

```
- [ ] 1. list_datasets — ft + model-surface train set, plus a separate eval-intent set
- [ ] 2. Baseline create_eval_run on the eval dataset (comparison point)
- [ ] 3. finetune_prerequisites — ALWAYS; fix everything in missing first
- [ ] 4. Present recommended models (prerequisites.recommendations / finetune_recommendation);
         user picks one or more catalog ids — never invent ids
- [ ] 5. Optional: estimate_finetune_cost per pick; confirm total spend
- [ ] 6. create_finetune_job once per selected model, same group_id (one training run)
- [ ] 7. Poll each job_status; finetune_loss_curves / finetune_job_events until terminal
- [ ] 8. list_deployed_models / deploy_model if needed; run_inference smoke-test winners
- [ ] 9. create_eval_run then compare_eval_runs vs the pre-training baseline
- [ ] 10. create_model_swap_pr on the winning job (SUCCEEDED + deploy + linked GitHub repo)
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
   prerequisites `recommendations` array is the list to present: `model`
   (catalog id), `display_name`, `tier`, `grade`, `confidence`, `cost_usd`,
   `time_human`, `hyperparams`, `selected`. Rows with `selected: true` are
   the recommender's default sweep. For the full ranking plus benchmark
   evidence, also call `finetune_recommendation(dataset_name, agent_name_or_slug?)`. Present **only** ids from `recommendations` /
   `catalog` — never invent OpenAI API ids or Llama-2 names. Confirm the
   picks with the user (one model or several). Default, if they don't
   narrow it: launch every `selected: true` row (or the top recommendations
   they approve), not a single arbitrary model.
1. Optional preflight: `validate_finetune_dataset(dataset_name, ...)` for the
   full format report; `finetune_dataset_overlap(dataset_name, eval_dataset_name)` for contamination;
   `estimate_finetune_cost(dataset_name, base_model, n_epochs?, use_lora?)`
   **per selected model** so the user sees sweep cost;
   `agent_base_model_throughput` for tokens/sec on the agent's current base
   model (speed comparison later).
1. **Launch a sweep, not one model.** `create_finetune_job` trains one
   catalog id per call. For multiple picks, call it once per `base_model`,
   reusing `group_id` from the first result on the rest so they share one
   training run (wizard parity). Same `dataset_name` / eval dataset / eval
   set on every call; omit hyperparameters unless overriding — they stamp
   from the recommender per model. Each call returns `finetune_job_id` +
   `group_id`. A lone pick is just one call.
1. Monitor: `job_status(kind="finetune_job", id=finetune_job_id)` **per
   job**; `finetune_job_events(finetune_job)` for the event log,
   `finetune_loss_curves(finetune_job)` for train/eval loss, token accuracy,
   progress percent and ETA. `list_finetune_jobs` to see the whole run.
   `cancel_finetune_job(finetune_job_id)` to stop one job;
   `retry_finetune_job(finetune_job)` re-queues a failed/cancelled job.
   Terminal statuses: `succeeded` / `failed` / `cancelled`.
1. On success: `list_deployed_models` — each trained model registers for
   inference. `deploy_model(deployed_model_id)` (re-)registers and retries
   FAILED deployments; `retry_model_deploy` / `deployed_model_checkpoints` /
   `deployed_model_metrics` / `deployed_model_activity` /
   `deployed_model_live` cover deployment health. `undeploy_model` clears
   routing (weights kept).
1. Smoke-test: `run_inference(model=<ft-… serving id or DeployedModel UUID>, messages=[...])` against a READY deployment. Do not invent a completion.
1. Verify quality: `create_eval_run` on the eval dataset, then
   `compare_eval_runs` against the pre-training baseline
   ([evals.md](evals.md)). Pick the winning job of the sweep from that
   comparison.
1. Ship: `create_model_swap_pr(finetune_job, pin?)` — opens a GitHub PR
   pointing the agent's code at the **winning** fine-tuned model (needs a
   SUCCEEDED job, a deployed model, and a linked GitHub repo;
   `analyze_github_repo` / `analyze_github_repo_url` link one). `pin=true`
   writes the concrete `ft-…` id instead of the permanent alias. Runs in
   the background — follow with `finetune_job_events`.
