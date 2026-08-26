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
- [ ] 2. list_evaluators — reuse before creating
- [ ] 3. Author: generate_evaluators (agent suite) OR create_judge_evaluator (one judge)
- [ ] 4. create_eval_set → add_evaluators_to_eval_set → activate_eval_set
- [ ] 5. create_eval_run (creates AND launches)
- [ ] 6. Poll job_status until completed / failed / cancelled
- [ ] 7. get_eval_run / eval_run_comparison; compare_eval_runs against a baseline when one exists
```

## 1. Author evaluators

- `list_evaluators(agent?, limit?)` first — reuse before creating.
- Whole suite for an agent: `generate_evaluators(agent_name_or_slug)` —
  authors grounded judges from the agent's codebase and merges them into its
  Default eval set. Runs in the background.
- Fit check for a dataset: `dataset_eval_capabilities(dataset_name)` —
  applicable kinds/scopes, recommended evaluators, semantic domain/task.
  Suggestions without persisting: `generate_dataset_evals(dataset_name)`
  (spends LLM credits).
- One judge: draft with
  `generate_evaluator_prompt(description, agent_name_or_slug?,
  applicable_role?)` or `compile_rubric(rubric_md, score_type?)` (both spend
  LLM credits, persist nothing), then save with
  `create_judge_evaluator(name, evaluation_prompt, score_type
  [numeric|boolean|categorical], categories? [required for categorical],
  agent_name_or_slug?, applicable_roles?, judge_model?)`.
- Verify what a judge reads (before persisting):
  `preview_evaluator_prompt(evaluator, trajectory?, structured?, expected?)`.
  `trajectory` is an **eval-authoring preview flag** (whether the preview
  includes the sample's transcript) — unrelated to runtime traces; it only
  controls how much context the preview shows while you author.

## Runtime eval envelope as an evaluation input

Traces instrumented with the runtime envelope
([instrumentation.md](instrumentation.md) Tier 2 API reference) feed evaluation directly:

- **Deterministic verdicts.** `expect(...)` declarations check mechanically —
  `contains` / `regex` / `schema` are verified without a judge; `constraint`
  (natural language) becomes an LLM-judge checklist item. Each declaration is
  addressed by its stable `id` (auto-derived from kind+spec, so re-running the
  code reuses the same expectation).
- **Predicates for authored evaluators.** `checkpoint_reached` /
  `expectation_declared` let an evaluator ask "did the run reach checkpoint X /
  declare expectation Y?" instead of re-parsing the transcript.
- **Gate semantics.** `expect(..., gate=True)` failures are hard fails: they
  cap the execution's score at 0. Reserve gates for invariants a run must
  never break; use plain expectations for quality checks.
- **Gap-finding.** `behaviour_coverage(agent)` reports per-behaviour /
  per-step eval coverage and gaps — use it to find where an agent lacks an
  authored evaluator instead of guessing from trace volume.

## 2. Group into eval sets

1. `list_eval_sets(agent?, limit?)` — member counts and each member's name +
   role.
1. `create_eval_set(agent_name_or_slug, name, description?, activate?)`.
1. `add_evaluators_to_eval_set(eval_set_name, evaluator_names, role?)` —
   `role="generative"` grades eval-run outputs; `role="trace_scoring"` on the
   agent's ACTIVE set binds them as live scorers on incoming traces.
1. `activate_eval_set(eval_set_name)` — makes it the agent's active set (the
   default for eval runs, finetune jobs, and optimizer experiments).

## 3. Run

`create_eval_run(name, dataset_name, eval_set_name? | evaluator_names?,
max_items?)` — creates AND launches. The dataset **must** be eval intent.
Evaluator precedence: `evaluator_names` > `eval_set_name` > the dataset
agent's active set (error if none). Over MCP the run scores the dataset's
captured rows as a single baseline variant. `max_items` caps datapoints
(useful for a cheap smoke run).

Always run a baseline on the eval dataset **before** a finetune or optimizer
loop so you have a comparison point afterwards.

## 4. Monitor and analyze

- Poll `job_status(kind="eval_run", id=<run name or uuid>)`;
  `cancel_eval_run(eval_run_name)` to stop.
- `get_eval_run(eval_run_name)` — status + aggregated summary.
- `eval_run_comparison(eval_run_name)` — variants, progress, per-evaluator
  rollup with deltas/ranking.
- `compare_eval_runs(eval_run_name, baseline_run_name)` — **authoritative**
  run-vs-run per-evaluator deltas. Use this after a finetune or optimizer
  loop, not a hand-rolled table from two `get_eval_run` payloads.
- `eval_run_trend(eval_run_name)`,
  `evaluator_score_history(agent_name_or_slug)`,
  `eval_score_trends(days?, agent?, evaluator?)` — time series.
- Human labels vs judges: `list_eval_annotations` /
  `create_eval_annotation` / `get_eval_annotation`. Calling
  `create_eval_annotation` with `eval_run_name` alone returns candidate
  `sample_id`s.
- Eval-variant models: `model_catalog(search?)`, `list_model_refs`,
  `create_model_ref(model_id, provider, label?, base_url?, api_key_ref?)`.
