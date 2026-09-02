# Tasks — behaviour registry, task executions, coverage

Overmind models production work as **Agent > Capabilities > Tasks**. A
**behaviour** (a "task") is a contract minted from the codebase scan. A
**task execution** is one carved unit of a real trace — a run or a turn —
bound to that contract and scored.

This is the layer between an agent and its raw traces. When quality drops,
read the failing units here before you walk spans in
[telemetry.md](telemetry.md).

All of this is Overmind MCP. Inspect each tool's schema for arguments.

## Workflow

```
- [ ] 1. list_behaviours — the task map for the agent (key, grain, entry_anchor, status)
- [ ] 2. list_task_executions — bound units, newest first; filter by status / binding_source
- [ ] 3. get_task_execution — route, per-step results, user intent, session rationale
- [ ] 4. behaviour_coverage — which tasks still have no outcome / step judge
- [ ] 5. get_behaviour_authoring_context — the contract to write a judge against
- [ ] 6. list_behaviour_evaluators — the judges already bound to one task
```

## The registry

`list_behaviours(capability_name_or_slug?, status?, limit?)` — rows carry
`key` (stable across rescans), `display_name`, `capability`, `entry_anchor`
(`module.qualname` of the entry symbol), `status` (`active` | `retired`),
`grain` (`run` | `turn`), and `graph_ref`.

Behaviours come from the scan, not from instrumented code. If a task is
missing, the repo scan is stale — re-run analyze
([capabilities.md](capabilities.md)), do not try to create one.

The `key` is the same slug the SDK stamps: `overmind.task(key, unit="turn")`
and the LangGraph `bind(behaviours={...})` map both take it. See
[telemetry.md](telemetry.md#step-5--carve-phases-into-turn-units).

## Executions

`list_task_executions(capability_name_or_slug?, task?, trace_id?, conversation_id?, binding_source?, status?, limit?)`
— one row per carved unit, newest first:

- `status` — `completed` | `error` | `interrupted`. `interrupted` means a
  rootless trace: the run was killed, or is still in flight past the settle
  window.
- `binding_source` — `anchor_join` (matched on the code anchor), `declared`
  (the SDK stamped the key), or `unbound`.
- `success_score`, `session_score`, `terminal_kind`, `route_flags`,
  `duration_ms`, `trace_id`, `conversation_id`, `graph_ref`.

**`unbound` is the first thing to check when scores look empty.** An unbound
unit was never joined to a contract, so its step judges never ran. The fix is
instrumentation — decorate the anchor, or stamp the key with
`overmind.task(...)` ([telemetry.md](telemetry.md)) — not a re-score.

`get_task_execution(task_execution_id)` — the UUID from
`list_task_executions`. Adds `unit_span_id`, `observed_route`,
`step_results`, `user_intent`, and `session_rationale`. Drill the raw spans
with `get_trace(trace_id)`.

## Coverage and authoring

- `behaviour_coverage(capability_name_or_slug)` — per task, which outcome and
  step judges exist and which contract segments are still uncovered. Start
  here when the ask is "what is not evaluated".
- `get_behaviour_authoring_context(task, capability_name_or_slug?)` — the
  contract to write against: `anchor_sequence`, named `steps` (each with the
  anchor a step judge binds to), `tool_set`, `terminal`. Call it **before**
  `create_judge_evaluator` for any step-scoped judge — the `anchor_segment`
  argument must come from this payload, not from a guess.
- `list_behaviour_evaluators(task, capability_name_or_slug?)` — the compiled
  suite already bound to one task (step + outcome). Reuse before authoring.

Authoring itself lives in [evals.md](evals.md).
