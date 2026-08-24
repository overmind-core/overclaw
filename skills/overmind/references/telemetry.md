# Telemetry — traces, sessions, health

Start here for "is my agent working?", "are traces landing?", "what failed?",
cost/latency, or anything about production traffic. Do not fetch traces over
REST — use the Overmind MCP tools named below. Inspect each tool's schema
for arguments.

Adding tracing *code* (SDK init, decorators, flush) is
[instrumentation.md](instrumentation.md). Once traces should be landing,
verify attribution with `verify_instrumentation_trace` and inspect raw spans
with `list_traces` / `get_trace`.

Conventions (names not ids, errors-as-values, mutations) live in
[SKILL.md](../SKILL.md).

## Workflow

```
- [ ] 1. list_agents — get the slug / display name
- [ ] 2. agent_health — start here for "how is this agent doing"
- [ ] 3. agent_failures if quality is down (do not copy ids into create_dataset_from_traces)
- [ ] 4. list_traces → get_trace to drill in (use summary for totals, don't sum pages)
- [ ] 5. list_sessions / get_session when the app is multi-turn
- [ ] 6. graph_* when the question is lineage / similarity, not a simple list
- [ ] 7. After instrumenting: run the path, then list_traces (newest) → get_trace and audit
```

## Orientation

1. `list_agents` — names, slugs, model, status. Use slug or display name in
   every later call.
1. `agent_health(days?, agent?)` — **start here** for "how is this agent
   doing". Returns offline eval-run rollups in `scores` AND live production
   trace scoring in `live_trace_scores` (what the traces UI shows). Each
   block has per-evaluator count, avg, pass rate, failures, and deltas vs the
   previous window (worst first), plus trace volume / error rate / latency
   (avg, p95). Prefer `live_trace_scores` for production quality when present.
1. `get_agent(agent_name_or_slug)` / `agent_prompts` / `agent_eval_spec` —
   identity, versioned prompts, eval contract (input schema, output fields,
   tools, weights, what is optimizable).
1. `get_instrumentation_context(agent)` — anchor-level "where to instrument"
   bundle: behaviours with anchors ranked `entry → discriminating → supplementary`, each anchor carrying `instrumented`, `import_line`,
   `verification_hint`; plus `remaining` (uninstrumented anchors) and
   `indistinguishable_pairs`. Ask for this before instrumenting an agent
   (see [instrumentation.md](instrumentation.md) Step 5a).
1. `behaviour_coverage(agent)` — per-behaviour and per-step eval coverage and
   gaps.
1. `behaviour_deviations(project, agent?)` — deviation clusters across
   executions.
1. `list_behaviours(project, agent?, status?)` — behaviours with
   `execution_count`, `avg_success_score`, `tool_inventory`.
1. `list_task_executions(project?, agent?, behaviour?, binding_source?)` /
   `get_task_execution(id)` — execution rows and detail (see Task
   executions).
1. `cost_rollup(since?)` — inference spend per served model.
1. `tool_stats(tool)` / `tool_error_trends(days?)` — tool-call volume, error
   rate, naming drift vs capability cards.
1. `contract_drift(agent, since?)` — declared schema vs what traces actually
   wrote.
1. `evaluator_stats(evaluator?, since?)` — noisy judges (abstain + error).
1. `eval_score_trends(days?, agent?, evaluator?)` — daily pass_rate / avg
   over a window.

## Failures

`agent_failures(agent, since_days?, limit?)` — one-call digest of recent
traces with at least one failing score: failed evaluator names + reasoning,
tools the trace called, violated schema_field refs. Use this instead of
walking traces by hand.

When the user wants a dataset from those failures, jump to
`create_dataset_from_failures` (see [datasets.md](datasets.md)) — do **not**
copy trace ids from this result into `create_dataset_from_traces`.

## Traces

- `list_traces(agent?, limit?, offset?)` — production root spans. Compact
  rows: `trace_id`, name, agent, status, duration, `total_tokens`,
  `total_cost`, model, live `trace_scores`, `n_scored`, `any_failed`,
  `graph_ref`. The `summary` object aggregates the **full** filtered set
  (counts, errors, sum/avg tokens and cost, duration stats) regardless of
  the page returned — answer totals/averages from `summary`; do not paginate
  or sum rows yourself. Paginate (`limit` + `offset`, `has_more`) only when
  the user needs the per-trace rows. Default page is small (~20); raise
  `limit` (max 5000) rather than stopping after one page.
- `get_trace(trace_id)` — one trace in detail: headline usage
  (`total_tokens` / `total_cost` / model, same as the Observability table),
  root span, live `trace_scores` with rationale, child spans (name, type,
  status, duration). Use the top-level usage fields; child span rows omit
  attribute-level token keys. Multi-invocation traces include
  `scoring_mode='multi_entry'` and per entry_point scores.
- `assign_traces_to_agent` — move mis-attributed traces onto the right agent.

### Verification after instrumenting

Instrumentation isn't done when the code compiles — it's done when you have
fetched the trace you just sent and it carries everything the baseline in
[instrumentation.md](instrumentation.md#what-a-good-trace-carries) requires.

1. Run the instrumented path end-to-end so a real trace is sent (flush
   short-lived processes first: `overmind.force_flush_traces()`).
1. Call `verify_instrumentation_trace` with the placement `plan_id` and exact
   `trace_id`. A strict pass is the attribution gate.
1. `list_traces` → `get_trace` on that exact `trace_id`. When the task
   names one agent in a multi-agent repo, filter `list_traces(agent=...)` to
   that agent's UUID — never the repo-wide newest trace, which may belong to
   a sibling agent.
1. Audit: `agent_id` (the agent's UUID, verbatim) and `agent_name` set and
   constant, `model` + `total_tokens` + `total_cost` populated, `conversation`
   / session set for multi-turn apps (`list_sessions`), span types varied (not
   everything `llm_call`), inputs/outputs on the entry point and key steps
   (`overmind.input.data` / `overmind.output.data` on span attributes), no
   secrets in payloads. If the trace's agent UUID differs from the card's,
   the identity stamp is wrong — fix it before anything else.
1. For trajectory instrumentation, verify at the **execution** level too —
   raw span checks are not the completion gate (`list_task_executions` →
   `get_task_execution` → `binding_source` is `anchor_join`/`declared`, not
   `unbound`; `user_intent` correct; `success_score`/`session_score`
   populated; `behaviour_coverage` shows every step evaluator got evidence).
1. Fix every gap, re-run, re-fetch until it clears. Empty `list_traces`
   means ingest failed — see troubleshooting in
   [instrumentation.md](instrumentation.md). Do not poll REST.

## Task executions (trajectory)

Task executions are the primary observability rows for trajectory
instrumentation — one per task run, bound to a Behaviour anchor by code
identity (`code.namespace` + `code.function.name`) and git sha
(`vcs.ref.head.revision`). Binding is `declared` when the SDK stamped
`overmind.behaviour.key` (`@overmind.task`), else `structural` / `anchor_join`
server-side matching, or `unbound` when nothing matched.

- `list_behaviours(agent)` — map the agent to its tasks: behaviour key,
  entry anchor, anchor sequence, terminal, execution/unbound counts.
- `list_task_executions(project?, agent?, behaviour?, binding_source?)` —
  execution rows incl. `binding_source`
  (`declared | structural | anchor_join | unbound`), `binding_confidence`
  (0-1), `attribution_verdict` (`bound_declared | bound_structurally | bound_low_conf | unbound_ambiguous | unbound_declared_key_unknown | unbound_no_evidence`), `success_score` (this execution), `session_score`
  (conversation-level — identical on every execution sharing the
  conversation), `terminal_kind`.
- `get_task_execution(id)` — one execution in detail: `observed_route`,
  `step_results`, `user_intent`, and `binding_provenance` (`rung`,
  `confidence`, `margin`, `version_mismatch`, `evidence`).

Verification flow after instrumenting:

1. `list_task_executions(agent=...)` — narrow with an optional
   `binding_source` filter.
1. `get_task_execution(id)` — check `binding_source` is `anchor_join` /
   `declared` / `structural` and `attribution_verdict` is `bound_declared` /
   `bound_structurally` (an `unbound_*` verdict or `bound_low_conf` means
   the identity never matched — fix, don't move on), `binding_confidence` is
   high, and `binding_provenance` (`rung`, `confidence`, `margin`,
   `version_mismatch`) looks right; `user_intent` correct, `success_score` /
   `session_score` populated.
1. `behaviour_coverage(agent)` — confirm every step evaluator got evidence;
   `behaviour_deviations(project)` clusters where executions drift.
1. `list_behaviours(agent)` / `get_instrumentation_context(agent)` — the
   task ↔ key map and which anchors are still `remaining` vs `instrumented`.

## Sessions (multi-turn)

Traces group by `conversation.id`:

- `list_sessions(agent_name_or_slug?, limit?)` — trace/span counts, tokens,
  cost, activity window.
- `get_session(session, limit?)` — aggregates plus member traces (newest
  first). Drill any trace with `get_trace`. Raise `limit` when the session
  has more traces than the default page.

## Context graph

Use when the question is lineage / similarity / "what produced this", not a
simple list:

- `graph_search(query, kind?, where?, limit?)` — semantic search over the
  project graph (trace summaries, score reasoning, …). Empty when embeddings
  are unavailable.
- `graph_node(ref)` — one node by `source_ref` (e.g. `agents:<id>`,
  `traces:<trace_id>`) plus 1-hop edges.
- `graph_walk(start_ref, edge_kinds, depth?, target_kind?, direction?, target_where?)` — follow edges up to 3 hops. Example: from a trace,
  `edge_kinds=['score_for']`, `direction='in'`, `target_kind='score'` lists
  attached scores; then `edge_kinds=['violates']` (out) for fields a score
  broke.
- `graph_lineage(start_ref, edge_kinds?, max_depth?)` — bidirectional BFS up
  to 8 hops, the mixed-direction spine a single `graph_walk` can't express
  (failing score → trace → datapoint → dataset → training run → model).
- `graph_trend(kind, bucket?, where?, since?)` — time-bucketed counts (e.g.
  failing scores per week: `kind='score'`, `where={"passed": false}`).
- `backfill_context_graph` — rebuild graph nodes/edges when search/lineage
  looks empty after a data import.

## External trace sources (connectors)

When traces live in Langfuse (etc.) rather than Overmind's SDK:

1. `list_connectors` — existing credentials and sync status.
1. `create_connector(connector_type, ...)` — currently `langfuse`. Omit
   secrets from the call so the config form collects them; never invent keys.
1. `configure_connector_sync` → `start_connector_setup` (first sync, optional
   auto_sync) or `trigger_connector_sync` for an on-demand pull.
1. `discover_connector_agents` → `set_connector_agent_mapping` so imported
   spans land on the right Overmind agents.
1. `connector_preview` / `connector_fetch_import` for a bounded import check.
