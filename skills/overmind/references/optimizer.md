# Optimizer — prompt / agent optimization

Experiments **execute on the user's machine** through a LOCAL executioner CLI
(`overmind optimise`). Nothing runs until one is connected. This is prompt
and code search against an **eval** dataset — not weight training (that's
[finetuning.md](finetuning.md)).

The dataset **must** be eval intent. See
[SKILL.md](../SKILL.md#dataset-types-intent--read-first-it-gates-every-workflow)
and [datasets.md](datasets.md).

Do **not** create a manual baseline eval run before the experiment. The
experiment generates and scores its own baseline iteration (`order=0`) —
that is what the improvement delta and the PR gate use.

All of this is Overmind MCP. Poll `job_status` for the experiment (see
conventions in [SKILL.md](../SKILL.md)). Inspect each tool's schema for
arguments.

**Modes.** `create_optimizer_experiment` accepts `mode`
(`optimize` | `model_comparison` | `hybrid`), `model_ids` (1–5 slugs for
comparison/hybrid), and `openrouter_key_source` (`platform` | `local`).
Default is harness-only `optimize`. `create_optimizer_pr` remains
optimize-mode only.

## Workflow

```
- [ ] 1. optimizer_prerequisites — ALWAYS; note eval datasets, active eval set, executioner
- [ ] 2. If disconnected: show executioner_start_command, then optimizer_connection
- [ ] 3. Confirm eval-intent dataset + active eval set
- [ ] 4. create_optimizer_experiment — creates AND launches (pass mode / model_ids for comparison or hybrid)
- [ ] 5. Poll job_status; get_optimizer_experiment / optimizer_iterations / optimizer_candidate_detail
- [ ] 6. create_optimizer_pr when best score beat the experiment's own baseline
         (needs linked GitHub repo; optimize-mode only)
```

## Steps

1. `list_optimizer_experiments(agent?, limit?)` — what's already running or
   finished. `list_backtest_runs` covers model-replay jobs (alternative
   models, not prompt search).
1. `optimizer_prerequisites(capability_name_or_slug)` — **ALWAYS** call first.
   Returns usable **eval**-intent datasets, the agent's active eval set,
   `executioner_connected`, and (when disconnected)
   `executioner_start_command`.
1. If disconnected, show the user `executioner_start_command` in a code block
   (run from the agent's repo), then re-check with `optimizer_connection`.
   Do not invent the command — use the one from prerequisites.
1. Confirm the dataset is **eval** intent (`list_datasets`) and that the eval
   set you want is active (`list_eval_sets` / `activate_eval_set`).
1. `create_optimizer_experiment(capability_name_or_slug, dataset_name, eval_set_name?, mode?, model_ids?, openrouter_key_source?, num_iterations? [2–5, default 5], num_candidates_per_iteration? [2–3, default 3], max_iterations_without_improvement? [default 3])` — eval set defaults to
   the agent's active set. Returns `experiment_id`. Default `mode` is
   `optimize`; comparison/hybrid need `model_ids`.
1. Monitor: `job_status(kind="optimizer_experiment", id=experiment_id)` for
   iteration progress; `get_optimizer_experiment(experiment_id)` for mode,
   model list, `state`, scores, `failure_reason`, entrypoint;
   `optimizer_iterations(experiment_id)` for per-round scores and every
   candidate (index, score, baseline flag);
   `optimizer_candidate_detail(candidate_id)` for a candidate's scores,
   eval-run linkage, and the iteration's winning patch.
   `cancel_optimizer_experiment(experiment_id)` to stop.
1. Land it: `create_optimizer_pr(experiment_id)` — opens a GitHub PR with
   the winning diff. Requires a COMPLETED optimize-mode experiment whose
   best score beat the experiment's own baseline and a linked GitHub repo
   (`analyze_github_repo` / `analyze_github_repo_url` if none is linked —
   [capabilities.md](capabilities.md)).
   An experiment that never ran (no executioner) has nothing to ship; no
   executioner is needed just to open the PR once a winning diff exists.
