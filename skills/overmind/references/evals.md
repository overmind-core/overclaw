# Evals — evaluators, eval sets, eval runs

An **evaluator** is a rubric (code check or LLM judge). An **eval set** is a
named grouping of evaluators on an agent. An **eval run** scores an
**eval**-intent dataset with those evaluators.

The dataset **must** be eval intent (`list_datasets`). Eval runs and
optimizer experiments reject `ft` / `unstructured` datasets — see
[SKILL.md](../SKILL.md#dataset-types-intent--read-first-it-gates-every-workflow)
and [datasets.md](datasets.md) if you need to ingest or convert.

All of this is Overmind MCP. Poll `job_status` for long runs (see
conventions in [SKILL.md](../SKILL.md)). Inspect each tool's schema for
arguments.

## Workflow

```
- [ ] 1. list_datasets — confirm the dataset is eval intent
- [ ] 2. list_evaluators / list_eval_sets — reuse before creating
         (Default set may already exist from the post-scan preload)
- [ ] 2b. Task-scoped judge? behaviour_coverage → get_behaviour_authoring_context first
- [ ] 3. Author: generate_evaluators (agent suite) OR create_judge_evaluator (one judge)
- [ ] 4. create_eval_set → add_evaluators_to_eval_set → activate_eval_set
- [ ] 5. create_eval_run (creates AND launches) — MCP is the launch path
- [ ] 6. Poll job_status until completed / failed / cancelled
- [ ] 7. get_eval_run / eval_run_comparison; compare_eval_runs against a baseline when one exists
```

## 1. Author evaluators

- `list_evaluators(agent?, limit?)` first — reuse before creating.
- `list_eval_sets(agent?)` before authoring a suite — the Console runs the
  same authoring pass as `generate_evaluators` automatically after a code
  scan, so the agent's Default eval set may already exist.
- Whole suite for an agent: `generate_evaluators(capability_name_or_slug)` —
  authors grounded judges from the agent's codebase and merges them into its
  Default eval set. Runs in the background.
- Suggestions for a dataset without persisting: `generate_dataset_evals(dataset_name)`
  (spends LLM credits).
- One judge: draft with
  `generate_evaluator_prompt(description, capability_name_or_slug?, applicable_role?)` or `compile_rubric(rubric_md, score_type?)` (both spend
  LLM credits, persist nothing), then save with
  `create_judge_evaluator(name, evaluation_prompt, score_type [numeric|boolean|categorical], task?, behaviour_role? [outcome|step], anchor_segment?, eval_set_name?, eval_set_role?, categories? [required for categorical], capability_name_or_slug?, applicable_roles?, judge_model?)`.
  Edit an existing judge in place (same id/version):
  `update_judge_evaluator(evaluator, evaluation_prompt, score_type, …)`.
- Verify what a judge reads:
  `preview_evaluator_prompt(evaluator, trajectory?, structured?, expected?)`.

### Task-scoped judges (outcome and step)

A judge can bind to one task (behaviour) instead of the whole agent. Do this
before you call `create_judge_evaluator`:

1. `behaviour_coverage(capability_name_or_slug)` — per task, the outcome and
   step judges that exist and the contract segments still uncovered. Author
   against the gaps.
1. `list_behaviour_evaluators(task, capability_name_or_slug?)` — the suite
   already bound to that task. Reuse before adding another judge.
1. `get_behaviour_authoring_context(task, capability_name_or_slug?)` —
   **required** for a step judge. It returns the contract: `anchor_sequence`,
   named `steps` with the anchor each step judge binds to, `tool_set`, and
   `terminal`.

Then pass `task=<behaviour key>`, `behaviour_role="outcome"` (grades the
deliverable) or `"step"` (grades one segment), and — for `step` — an
`anchor_segment` copied from the authoring context. Never guess an
`anchor_segment`: a segment that is not in the contract silently never
matches, and the judge skips every unit.

Task keys and executions: [behaviours.md](behaviours.md).

## 2. Group into eval sets

1. `list_eval_sets(agent?, limit?)` — member counts and each member's
   `id` / name / role / enabled.
1. `create_eval_set(capability_name_or_slug, name, description?, activate?)`.
1. `add_evaluators_to_eval_set(eval_set_name, evaluator_names, role?)` —
   `role="generative"` grades eval-run outputs; `role="trace_scoring"` on the
   agent's ACTIVE set binds them as live scorers on incoming traces.
1. `update_eval_set_member(eval_set_name, member_id, enabled?, role?, order?)`
   to enable/disable; `remove_eval_set_member(eval_set_name, member_id)` to
   drop a member (`member_id` from `list_eval_sets`).
1. `activate_eval_set(eval_set_name)` — makes it the agent's active set (the
   default for eval runs, finetune jobs, and optimizer experiments).

## 3. Run

The Console has **no eval-run launcher** — MCP (or Brain chat) is the launch
path.

`create_eval_run(name, dataset_name, eval_set_name? | evaluator_names?, max_items?, variants_input?)` — creates AND launches. The dataset **must** be eval intent.
Evaluator precedence: `evaluator_names` > `eval_set_name` > the dataset
agent's active set (error if none). Omit `variants_input` for a single
captured-traces baseline; pass it through for multi-variant (same shape as
`POST /eval-runs/`). `max_items` caps datapoints (useful for a cheap smoke
run).

To re-run an existing run (wipe samples/scores and queue again), use
`relaunch_eval_run(eval_run_name)` — not `create_eval_run`.

Finetune and optimizer loops create their own incumbent / experiment
baselines — a manual `create_eval_run` beforehand is only for eval-vs-eval
comparisons you drive yourself. See [finetuning.md](finetuning.md) and
[optimizer.md](optimizer.md).

## 4. Monitor and analyze

- Poll `job_status(kind="eval_run", id=<run name or uuid>)`;
  `cancel_eval_run(eval_run_name)` to stop.
- `list_eval_samples(eval_run_name)` / `get_eval_sample(sample_id)` — bounded
  datapoint/trajectory drill-in.
- `get_eval_run(eval_run_name)` — status + aggregated summary.
- `eval_run_comparison(eval_run_name)` — variants, progress, per-evaluator
  rollup with deltas/ranking.
- `compare_eval_runs(eval_run_name, baseline_run_name)` — **authoritative**
  run-vs-run per-evaluator deltas. Use this after a finetune or optimizer
  loop, not a hand-rolled table from two `get_eval_run` payloads.
- `eval_run_trend(eval_run_name)`,
  `evaluator_score_history(capability_name_or_slug)`,
  `eval_score_trends(days?, agent?, evaluator?)` — time series.
- Human labels vs judges: `list_eval_annotations` /
  `create_eval_annotation` / `get_eval_annotation`. Calling
  `create_eval_annotation` with `eval_run_name` alone returns candidate
  `sample_id`s.
- Eval-variant models: `model_catalog(search?)`, `list_model_refs`,
  `create_model_ref(model_id, provider, label?, base_url?, api_key_ref?)`.
