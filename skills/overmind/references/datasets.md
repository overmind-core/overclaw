# Datasets — upload, build from traces, clean

`list_datasets` first — name, source, **intent**, datapoint count. Pick or
create with the right intent before any eval / finetune / optimizer step.
Intent is immutable and gates every downstream workflow; see
[SKILL.md](../SKILL.md#dataset-types-intent--read-first-it-gates-every-workflow).

All of this is Overmind MCP. Do not POST files to a REST attachments
endpoint. Inspect each tool's schema for arguments.

## Workflow

```
- [ ] 1. list_datasets — name, source, intent, datapoint count
- [ ] 2. Build with the right intent:
         failures → create_dataset_from_failures
         OR traces → create_dataset_from_traces (last_n, don't copy ids off a page)
         OR file → analyze_dataset_file then create_dataset_from_file
- [ ] 3. Confirm intent matches the downstream workflow (eval vs ft vs unstructured)
- [ ] 4. open_workshop (once) / workshop_state — start here for quality
         (open_workshop / re-analysis spend credits; surface the verdict)
- [ ] 5. For each insight: workshop_insight_detail → apply_insight_fix →
         staged_diff → approve or discard
         (one staged op at a time; if the fix went stale, re-apply — don't approve)
- [ ] 6. Direct edits if needed; waive checks you accept
- [ ] 7. Before Train / Optimise: workshop_state verdict must not be
         not_ready / needs_work (Console rail disables those buttons otherwise)
```

## What only the Console can do

Do not promise these over MCP — they exist only in the Console ingest /
workshop UI:

- **Per-row exclusions** before commit (measurement-based scan exclusions).
- The async ingest wizard (scan → plan → commit as a job). MCP split /
  re-ingest goes through `DatasetSerializer` / `create_dataset_from_traces`
  on already-accepted rows.

## From traces / failures

Preferred when telemetry already exists (see [telemetry.md](telemetry.md)).

- Failures → dataset in one step:
  `create_dataset_from_failures(agent_name_or_slug, dataset_name, since_days?, limit?)`. Prefer this over manually plumbing trace ids. Do **not** copy
  ids from `agent_failures` into `create_dataset_from_traces`.
- Specific or recent traces:
  `create_dataset_from_traces(agent_name_or_slug, trace_ids?, last_n?, dataset_name?, surface?, split_eval_fraction?, split_method?, eval_dataset_name?)`.
  Omit `trace_ids` and pass `last_n` for the N most recent (default 20, max
  1000\) — better than copying ids off a `list_traces` page. `trace_ids`
  accept bare OTel hex or `traces:<hex>` refs from `agent_failures`.
  `surface` is `agent` | `model`. Set `split_eval_fraction` (0.01–0.9) plus
  optional `split_method` (`shuffle` | `tail`) to create train+eval
  datasets in one call.
- Append to an existing mutable dataset:
  `add_traces_to_dataset(dataset_name, trace_ids)`.
- Remove rows:
  `remove_datapoints_from_dataset(dataset_name, datapoint_ids)` (destructive
  — confirm with the user).

Trace-built datasets get intent and surface assigned at import. A dataset
being read by a running job is frozen until the job ends.

## From a file the user attached

There is no separate upload tool. The user attaches a CSV / TSV / JSON /
JSONL; you get an `attachment_id`. Then:

1. `analyze_dataset_file(attachment_id, intent?)` — inferred intent, field
   mapping, per-intent viability, before→after preview. **Nothing is
   persisted.** Same engine as the dataset upload wizard.
1. Review the proposal with the user. Pass `intent` explicitly
   (`eval | ft | unstructured`) if inference is wrong.
1. `create_dataset_from_file(attachment_id?, dataset_name?, agent_name?, intent?, surface?, source_dataset?, split_eval_fraction?, split_method?, eval_dataset_name?)` — ingest. Returns the created dataset's name, intent,
   surface, and `num_datapoints`. Bind `agent_name` when you know which
   agent owns it. `surface` is `agent` | `model`. Set
   `split_eval_fraction` to create an **ft** train dataset plus an **eval**
   holdout — rows must be conversation-shaped (`messages`) or pass `intent`
   if they aren't. Re-ingest an
   existing dataset with `source_dataset` (name) plus a target `intent`
   (or a split). `attachment_id` is not required when `source_dataset` is
   set.

If no file is attached and you are not re-ingesting, ask the user to
attach one.

Intent never mutates on the same row. Convert intent or surface by
creating a **new** dataset via `source_dataset`. If a tool rejects a
dataset for intent, pick another or re-ingest; don't retry the same one.

## Cleaning / editing (the workshop)

Treat this as a git-like loop: analyze → inspect insights → stage a fix →
diff → approve (commit) or discard. A dataset frozen by a running job will
reject writes. `open_workshop` and forced re-analysis spend credits.

Gates the Console enforces (and you should too):

- **One staged op at a time** per dataset — finish or discard before the
  next apply / batch edit.
- A staged fix goes **stale** if the dataset moved — re-apply, don't
  approve the stale op.
- A `not_ready` / `needs_work` verdict disables Train and Optimise in the
  Console rail — surface the verdict from `workshop_state` before launching
  either.

1. `open_workshop(dataset_name)` once (starts the first analysis; no-op
   after), or `analyze_workshop_dataset` to force a fresh analysis.
   `refresh_dataset_context` rebuilds cached semantic context (domain, task,
   evaluator fit) in the background.
1. `workshop_state(dataset_name)` — score, verdict, failing checks, open
   insights, any staged change. **Start here** for dataset-quality questions.
   `list_insights(status?, severity?)` for a project-wide triage
   (`status=open`).
1. For each insight: `workshop_insight_detail(insight)` →
   `apply_insight_fix(insight)` (STAGES a change, does not commit) →
   `workshop_staged_diff(dataset_name)` → `approve_staged_change` to commit
   or `discard_staged_change` to undo.
1. Direct edits (each lands as a commit unless noted):
   `workshop_edit_row(dataset_name, dp_id, field, value)`,
   `workshop_delete_rows(dataset_name, dp_ids)`,
   `workshop_batch_edit(dataset_name, operations, message?)` (staged, needs
   approval), and `preview_find_replace` → `workshop_find_replace`.
1. Inspect: `workshop_export_rows(dataset_name, limit?, sha?, fmt?)` —
   `fmt` is `json` (default structured rows), `jsonl`, or `csv`; cap is
   5000 (`truncated=true` when clipped). Also `workshop_columns`
   (per-column stats), `workshop_commits` / `workshop_commit_diff`
   (history), `workshop_compare_datasets` (two datasets, same intent).
   Undo history with `restore_dataset_commit(dataset_name, sha)`.
1. Failing checks you accept: `waive_dataset_check(dataset_name, check_key)`
   / `unwaive_dataset_check`. `check_key` comes from `workshop_state`.
