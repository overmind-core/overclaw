# Fine-tuning (training)

SFT a catalog base model on an **ft** / model-surface dataset, keep a
held-out **eval** dataset for judges, then deploy and prove the new model
beats a pre-training baseline.

Training dataset: **ft** intent AND `model` surface (LLM-in → LLM-out rows,
not agent-level rows). Eval dataset: **eval** intent. See
[SKILL.md](../SKILL.md#dataset-types-intent--read-first-it-gates-every-workflow)
and [datasets.md](datasets.md).

Run a baseline eval on the eval dataset **before** training so you have a
comparison point — [evals.md](evals.md).

All of this is Overmind MCP. Poll `job_status` for the job (see conventions
in [SKILL.md](../SKILL.md)). Inspect each tool's schema for arguments.

## Workflow

```
- [ ] 1. list_datasets — ft + model-surface train set, plus a separate eval-intent set
- [ ] 2. Baseline create_eval_run on the eval dataset (comparison point)
- [ ] 3. finetune_prerequisites — ALWAYS; fix missing; pick base_model from catalog only
- [ ] 4. Optional: validate_finetune_dataset / overlap / estimate_finetune_cost (confirm cost)
- [ ] 5. create_finetune_job — creates AND queues
- [ ] 6. Poll job_status; finetune_loss_curves / finetune_job_events until succeeded
- [ ] 7. list_deployed_models / deploy_model if needed; run_inference smoke-test
- [ ] 8. create_eval_run then compare_eval_runs vs the pre-training baseline
- [ ] 9. create_model_swap_pr to ship (needs SUCCEEDED job, deploy, linked GitHub repo)
```

## Steps

1. `list_datasets` — pick a **ft**-intent, model-surface training dataset, or
   build one from traces / a file. `list_finetune_jobs` /
   `list_finetune_base_models` show what's already been tried.
1. `finetune_prerequisites(dataset_name, agent_name_or_slug?)` — **ALWAYS**
   call before launching. Returns `ready` / `missing` (train-intent check,
   surface check, validation, agent, default eval dataset, default eval set),
   train/eval `overlap_count`, `recommendations` (model + hyperparams +
   cost/time), and the trainable `catalog`. Fix everything in `missing`
   first. Present model picks **only** from `recommendations` / `catalog` —
   never invent OpenAI API ids or Llama-2 names. `base_model` MUST be a
   catalog id from this result.
1. Optional preflight (the prerequisites report already covers the common
   case): `validate_finetune_dataset(dataset_name, ...)` for the full format
   report (errors, warnings, token stats, split feasibility);
   `finetune_dataset_overlap(dataset_name, eval_dataset_name)` for
   contamination; `finetune_recommendation` for ranked models + evidence;
   `estimate_finetune_cost(dataset_name, base_model, n_epochs?, use_lora?)`
   to confirm cost with the user before spending;
   `agent_base_model_throughput` for tokens/sec on the agent's current base
   model (speed comparison later).
1. `create_finetune_job(dataset_name, base_model, name?, agent_name_or_slug?,
   eval_dataset_name?, eval_set_name?, hyperparameters?, model_tier?,
   use_case?)` — creates AND queues. Eval dataset/set and hyperparameters
   default from the prerequisites report when omitted. Returns
   `finetune_job_id`.
1. Monitor: `job_status(kind="finetune_job", id=finetune_job_id)`,
   `finetune_job_events(finetune_job)` for the event log,
   `finetune_loss_curves(finetune_job)` for train/eval loss, token accuracy,
   progress percent and ETA. `cancel_finetune_job(finetune_job_id)` to stop;
   `retry_finetune_job(finetune_job)` re-queues a failed/cancelled job.
   Terminal statuses: `succeeded` / `failed` / `cancelled`.
1. On success: `list_deployed_models` — the trained model registers for
   inference. `deploy_model(deployed_model_id)` (re-)registers and retries
   FAILED deployments; `retry_model_deploy` / `deployed_model_checkpoints` /
   `deployed_model_metrics` / `deployed_model_activity` /
   `deployed_model_live` cover deployment health. `undeploy_model` clears
   routing (weights kept).
1. Smoke-test: `run_inference(model=<ft-… serving id or DeployedModel UUID>,
   messages=[...])` against a READY deployment. Do not invent a completion.
1. Verify quality: `create_eval_run` on the eval dataset, then
   `compare_eval_runs` against the pre-training baseline
   ([evals.md](evals.md)).
1. Ship: `create_model_swap_pr(finetune_job, pin?)` — opens a GitHub PR
   pointing the agent's code at the fine-tuned model (needs a SUCCEEDED job,
   a deployed model, and a linked GitHub repo; `analyze_github_repo` /
   `analyze_github_repo_url` link one). `pin=true` writes the concrete
   `ft-…` id instead of the permanent alias. Runs in the background — follow
   with `finetune_job_events`.
