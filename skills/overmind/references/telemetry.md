# Telemetry — instrument, inspect, health

Add tracing so spans land, and inspect production traffic (health, traces,
sessions, graph, connectors). The two are one loop: wire the SDK, then
verify with a real trace. Do not fetch traces over REST — use the Overmind
MCP tools named below. Inspect each tool's schema for arguments.

The SDK surface is the `overmind` Python package (`init`, decorators,
`force_flush_traces`). This file may lag the package — prefer the code in
the installed SDK if they disagree.

Agent card, `update_agent`, and GitHub analyze: [agents.md](agents.md).
Conventions (names not ids, errors-as-values, mutations) live in
[SKILL.md](../SKILL.md).

## Workflow

```
- [ ] 0. Resolve the agent's identity — `get_agent` via MCP; copy top-level `id` (bare UUID) verbatim ([agents.md](agents.md))
- [ ] 1. agent_health — start here for "how is this agent doing"
- [ ] 2. If nothing is landing: detect existing OTel, install the SDK, init, decorate (below)
- [ ] 3. agent_failures if quality is down (do not copy ids into create_dataset_from_traces)
- [ ] 4. list_traces → get_trace to drill in (use summary for totals, don't sum pages)
- [ ] 5. list_sessions / get_session when the app is multi-turn
- [ ] 6. graph_* when the question is lineage / similarity, not a simple list
- [ ] 7. After instrumenting: run the path, flush, then list_traces (newest) → get_trace and audit
```

## Orientation

1. `list_agents` — `id` (bare UUID), names, slugs, model, `active_model`,
   status. Use slug or display name in every later call. Card, prompts,
   `update_agent`, and GitHub analyze: [agents.md](agents.md).
1. `agent_health(days?, agent?)` — **start here** for "how is this agent
   doing". Returns offline eval-run rollups in `scores` AND live production
   trace scoring in `live_trace_scores` (what the traces UI shows). Each
   block has per-evaluator count, avg, pass rate, failures, and deltas vs the
   previous window (worst first), plus trace volume / error rate / latency
   (avg, p95). Prefer `live_trace_scores` for production quality when present.
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

- `list_traces(agent?, search?, status?, model?, min_duration_ms?, max_duration_ms?, start_after?, start_before?, session?, all_spans?, ordering?, limit?, offset?)` — production spans. Default is **root-only**;
  `all_spans=true` includes children (adds `span_id` on rows). Compact
  rows: `trace_id`, name, agent, status, duration, `total_tokens`,
  `total_cost`, model, live `trace_scores`, `n_scored`, `any_failed`,
  `graph_ref`. The `summary` object aggregates the **full** filtered set
  (counts, errors, sum/avg tokens and cost, duration stats) regardless of
  the page returned — answer totals/averages from `summary`; do not paginate
  or sum rows yourself. Paginate (`limit` + `offset`, `has_more`) only when
  the user needs the per-trace rows. Default page is small (~20); raise
  `limit` (max 5000) rather than stopping after one page.
- `get_trace(trace_id, span_id?)` — one trace in detail: headline usage
  (`total_tokens` / `total_cost` / model, same as the Observability table),
  root span, live `trace_scores` with rationale, child spans with
  `attributes` / `events` / `status_message` (capped; `truncated=true`
  when clipped). Pass `span_id` to fetch one span fully. Multi-invocation
  traces include `scoring_mode='multi_entry'` and per entry_point scores.
- `assign_traces_to_agent(agent_name_or_slug, trace_ids)` — re-attribute
  every span of those traces onto the agent (bare hex or `traces:<hex>`).
  Typical use: connector-imported traces that landed agentless. See
  [agents.md](agents.md).

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

1. `list_connectors` — existing credentials and sync status (`is_active`,
   `auto_sync_enabled`, `sync_error`, backfill progress). Pass
   `include_inactive` to see disconnected ones.
1. `create_connector(connector_type, ...)` — `connector_type` is the
   provider slug from the tool enum. Omit secrets so the config form
   collects them; never invent keys. Disconnect and patch (name / URL /
   keys / auto_sync) are Console-only.
1. `configure_connector_sync` → `start_connector_setup` (first sync, optional
   auto_sync) or `trigger_connector_sync` for an on-demand pull.
1. `discover_connector_agents` → `set_connector_agent_mapping` so imported
   spans land on the right Overmind agents.
1. `list_connector_sync_runs` for recent sync history.
1. `connector_preview` / `connector_fetch_import` for a bounded import check.

## What a good trace carries

Audit every integration — new or existing — against this baseline before
calling it done. Fetch a real trace with `list_traces` → `get_trace`; do
not ask the user to describe the Console.

| Requirement             | How                                                                                                                                                                                                                                | Why                                                                                                                                                                                                                                                     |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Agent identity          | `agent_id=` **and** `agent_name=` in `init()` (or `set_agent_id()` + `set_agent_name()`). Get BOTH from `get_agent` via MCP — copy top-level `id` verbatim into `agent_id=` (bare UUID; never invent, truncate, or reformat it)    | The server resolves `overmind.agent.id` by direct UUID lookup (drift-proof); `agent_name` is display + fallback. Name-only stamps risk a slug mismatch with the code-scan agent, which mints a duplicate agent whose live trace scoring silently no-ops |
| Per-agent scoping       | When your task names ONE agent in a multi-agent repo, stamp THAT agent's identity only — never the repo's, never a generic name                                                                                                    | Traces group under the right agent in the Console; sibling agents' spans are untouched                                                                                                                                                                  |
| Model + token usage     | Automatic via provider auto-instrumentation (Step 4); raw-OTel spans should carry `gen_ai.request.model` / `gen_ai.usage.*`                                                                                                        | Cost is computed server-side from these                                                                                                                                                                                                                 |
| Inputs and outputs      | The decorators (Step 6) capture call args and return values automatically; make sure the entry point and key steps are decorated so the trace shows what the agent saw and produced                                                | A trace without I/O can't be debugged or turned into eval data                                                                                                                                                                                          |
| Sensitive data excluded | Not for agents — trace normally and mask credential fields (API keys, tokens, passwords) before they reach decorated functions. `capture="none"` (timing/status, no values) is a manual escape hatch for human implementation only | Inputs/outputs are stored verbatim                                                                                                                                                                                                                      |
| Session grouping        | `set_conversation_id(...)` per conversation/thread (stamped as `conversation.id`) whenever the app has multi-turn interactions                                                                                                     | Groups traces into Sessions                                                                                                                                                                                                                             |
| User attribution        | `set_user(user_id, email=...)` where the app has accounts                                                                                                                                                                          | Per-user filtering and cost attribution                                                                                                                                                                                                                 |
| Span hierarchy + types  | One `@entry_point` at the top; `@workflow` / `@tool` / `@retrieval` for the steps under it, with descriptive names                                                                                                                 | Shows which step failed or was slow, instead of one flat LLM call                                                                                                                                                                                       |

## Step 0 — Resolve the agent's identity

The authoritative source for agent identity is `get_agent` via MCP (see
[agents.md](agents.md)): it returns a top-level `id` (bare UUID),
`active_model`, `source_repo`, and a capped `flow` (`flow_truncated` when
clipped) — the capability card with `agent_path`, `modes[*].entrypoint_fn`,
system prompt, and tool surface. Call it with the agent's name/slug before
writing any instrumentation. Copy `id` verbatim — never invent, shorten,
re-format, or "fix" it, and never substitute another agent's id. If `id` is
missing or does not look like a UUID, STOP and report instead of guessing:
a wrong id silently attributes every trace to the wrong agent. When
`flow_truncated` is true, still use the paths you got — don't invent files
that weren't in the card. If the project has no agents yet,
`analyze_github_repo` / `analyze_github_repo_url` discovers them
([agents.md](agents.md)).

## Step 1 — Detect existing telemetry

Grep the project before writing any code. The result decides Step 3.

```bash
rg -n "set_tracer_provider|TracerProvider|opentelemetry|traceloop|Traceloop|langsmith|OTEL_EXPORTER" --glob '!**/.venv/**'
```

- **No matches** → greenfield path (Step 3a). `overmind.init()` creates and
  installs the provider.
- **A `TracerProvider` is already set** (OTel directly, Traceloop/OpenLLMetry,
  LangSmith's OTel bridge, etc.) → fan-out path (Step 3b). OpenTelemetry only
  honours the **first** `set_tracer_provider()` call and ignores later ones with
  a warning, so calling `overmind.init()` on top of an existing provider would
  silently attach nothing. Instead, add Overmind's exporter to the provider the
  project already owns.

## Step 2 — Install and configure

```bash
uv add overmind        # or: pip install overmind
```

Required environment variable (project API key — same one the MCP server
uses). Ask the user to set it in their shell or `.env`. Never ask them to
paste the key into chat.

```bash
export OVERMIND_API_KEY=<your-api-key>
```

Optional identity/config (all have env-var equivalents read by `init()`):

| Env var                 | Purpose                                          |
| ----------------------- | ------------------------------------------------ |
| `OVERMIND_SERVICE_NAME` | Service name on the traces                       |
| `OVERMIND_AGENT_NAME`   | Human-readable agent name                        |
| `OVERMIND_AGENT_ID`     | Agent UUID (preferred over name once registered) |
| `OVERMIND_ENVIRONMENT`  | e.g. `production` (default `development`)        |
| `OVERMIND_API_URL`      | Override the trace endpoint base URL             |

## Step 3a — Greenfield init

Call once at process start, before the traced code runs:

```python
import overmind

overmind.init(
    service_name="my-agent",
    agent_id="<agent-uuid>",  # copy verbatim from get_agent — never invent
    agent_name="My Agent",  # this agent's constant display name
    providers="auto",  # instrument every installed provider SDK
)
```

`providers="auto"` detects the installed target libraries (openai, anthropic,
google, agno, langchain) and enables every one whose instrumentor is also
present, logging the resolved list. Name providers explicitly
(`providers=["openai"]`) to pin the set; `providers=[]` enables all known;
omitting `providers` enables none. `init()` is graceful: without
`OVERMIND_API_KEY` it logs, returns `False`, and every decorator/helper
becomes a no-op — safe to ship in apps where Overmind is optional. Set
`OVERMIND_STRICT_MODE=true` to make a missing key raise. `init(debug=True)`
prints the endpoint, resolved identity, enabled instrumentors, and export
mode.

The core package is tracing-only. Add extras when needed:
`overmind[langchain]` (LangChain/LangGraph auto-instrumentation),
`overmind[inference]` (token-cost enrichment via litellm),
`overmind[cli]` (the `overmind` command), `overmind[tracing-full]`
(requests/httpx/logging spans).

## Step 3b — Fan-out onto an existing telemetry provider

Keep the project's current telemetry untouched and add a second exporter that
ships the same spans to Overmind. This works because a `TracerProvider` can
hold many span processors — each exports independently.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

import overmind
from overmind.tracing import (
    enable_tracing,
    get_api_settings,
    set_agent_id,
    set_agent_name,
)

api_key, base_url = get_api_settings()  # reads OVERMIND_API_KEY / OVERMIND_API_URL

provider = trace.get_tracer_provider()
if not isinstance(provider, TracerProvider):
    # Nothing real was installed yet — let Overmind own the pipeline instead.
    overmind.init(
        service_name="my-agent",
        agent_id="<agent-uuid>",  # copy verbatim from get_agent
        agent_name="My Agent",
        providers="auto",
    )
else:
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=f"{base_url}/api/v1/traces",
                headers={"X-Api-Key": api_key},
            )
        )
    )
    # Auto-instrument the LLM SDKs against the existing provider.
    enable_tracing("auto")
    # Fan-out path: identity is NOT stamped by an on-start processor, so stamp it
    # explicitly on the spans you decorate (Step 6) and via the context helpers.
    set_agent_id("<agent-uuid>")  # verbatim from get_agent
    set_agent_name("My Agent")
```

Notes:

- The project's existing backend keeps receiving spans; Overmind gets a copy.
- Overmind's server reads canonical `genai.*` usage attributes. Spans from
  third-party auto-instrumentors that only emit OTel `gen_ai.*` keys are
  bridged automatically **only when Overmind owns the provider**. On the
  fan-out path, prefer Overmind's own auto-instrumentation (`enable_tracing`)
  or the decorators in Step 6 so token/cost rollups populate.

## Step 4 — Bracket the run

Every agent execution needs exactly one run boundary. `overmind.run(...)` is
the one scope that covers it — capability identity, the entry-point run span,
intent, conversation id, error status, and a flush on exit:

```python
with overmind.run(
    "triage-run", intent=request["question"], conversation_id=ticket_id
) as run:
    answer = agent.invoke(request)
    run.deliver(answer)  # terminal deliverable, auto-grounded
```

As a decorator (sync or async) every parameter except `name` also accepts a
callable receiving the wrapped call's arguments, and the run span carries the
function's code identity — one decoration also satisfies an entry-point
scan-contract anchor:

```python
class Agent:
    @overmind.run(
        intent=lambda self, *a, **k: self.task,
        conversation_id=lambda self, *a, **k: self.task_id,
    )
    async def run(self): ...
```

Spans created **outside** any run boundary that would start their own trace
are suppressed as orphan fragments (the SDK warns once). If a trace is
missing, the fix is this bracket — not `init(export_orphan_spans=True)`.

## Step 5 — Carve phases into turn units

`task(key, unit="turn")` makes a phase an independently scored unit. The key
is a `Behaviour.slug` from the project's scanned task map. Re-entering the
same key re-uses the still-open turn span, so a re-entrant phase (tool loop,
debate rounds) lands in one unit:

```python
with overmind.task("investment-debate", unit="turn"):
    ...  # spans here nest under the debate's turn unit
```

Rules that matter:

- `deliver()` runs **inside the unit that produced the deliverable**.
- Internal fan-out/retries/loop bodies must **not** declare `unit`.
- Multi-capability agents scope identity with `overmind.capability(name)`;
  entering a different capability mid-trace is a handoff and opens a new unit
  automatically.

For LangGraph agents, `overmind.integrations.langgraph.bind()` does this
declaratively — call it on the `StateGraph` after `add_node()`, before
`compile()`:

```python
from overmind.integrations import langgraph as overmind_langgraph

overmind_langgraph.bind(
    workflow,
    # Default key per node: slugified node name. Override where the task map
    # groups nodes differently; None opts a node out.
    behaviours={
        "Bull Researcher": "investment-debate",
        "Bear Researcher": "investment-debate",
    },
    deliver="Portfolio Manager",  # this node's return value is the deliverable
)
app = workflow.compile()
```

## Step 6 — Decorate anchors and add custom spans

**Every function the scanned task map anchors on must be decorated** — an
undecorated anchor emits no `code.namespace`/`code.function.name`, so its step
judges silently skip it. Decorators (sync and async) — use the type that
matches the code:

```python
@overmind.entry_point()  # top-level request handler
def run(payload: dict) -> dict: ...


@overmind.workflow()  # multi-step orchestration
def pipeline(): ...


@overmind.tool()  # a tool/function the agent can call
def search(query: str) -> list[dict]: ...


@overmind.retrieval()  # RAG / vector lookup
def fetch_docs(q: str): ...


@overmind.observe()  # any other traced function
def score(x): ...
```

All decorators accept `capture=` (`"auto"` scrubbed args/result, `"none"`,
`"messages"`), `ignore=` (argument names never captured), `capability=`, and
`format_input=`/`format_output=` hooks. Captured payloads are scrubbed
automatically: secret-named keys redacted, base64/data-URL blobs replaced,
text kept in full.

Context manager and current-span helpers:

```python
with overmind.start_span("rerank", span_type=overmind.SpanType.FUNCTION) as span:
    overmind.set_tag("candidate_count", len(candidates))

overmind.set_user("user-123", email="a@b.com")
overmind.set_conversation_id("conv-abc")  # groups spans into one session
overmind.set_agent_id("<agent-uuid>")  # verbatim from get_agent — never invented
overmind.set_agent_name("My Agent")  # keep constant for this agent

try:
    ...
except Exception as exc:
    overmind.capture_exception(exc)  # marks the span errored
    raise
```

`start_span` and the decorators use the ambient tracer, so they attach to
whichever provider is active — greenfield or fan-out.

**Minimum for a good trace.** The entry point alone is NOT enough. For every
function the agent's code path actually calls that is a meaningful step — a
tool the agent can invoke, a policy lookup, a retrieval, a scoring step —
decorate it with the matching type (`@tool`, `@retrieval`, `@function`,
`@workflow`, `@observe`). A trace whose spans are all `entry_point` is flat: it cannot
show which step failed or was slow. Rule of thumb: if the function has a name
a human would use to describe the agent's work ("search", "lookup_policy",
"rerank"), it should be a span.

## Step 6b — Instrumenting ONE agent in a multi-agent repo

Most instrumentation tasks name **one specific agent**. Everything in this
skill is scoped to that agent:

- **Identity.** Stamp exactly the `agent_id` and `agent_name` from `get_agent`
  (Step 0). The `agent_id` is a UUID — copy it verbatim into `init(agent_id=)`,
  `set_agent_id()`, or `OVERMIND_AGENT_ID`. Never invent, shorten, re-format,
  or substitute another agent's id.
- **Scope.** Only touch the named agent's files (`agent_path`,
  `modes[*].entrypoint_fn`, its own tools and prompt). Do not edit sibling
  agents, and do not re-decorate code they own. Shared infrastructure (a
  common LLM client, a shared `core/` module) is usually fine to instrument
  once — leave its own identity alone and let this agent's identity ride on
  the spans that pass through it.
- **One identity per agent.** Other agents in the repo each have their own
  stable `agent_id`/`agent_name`. Distinct names across agents are correct;
  only a SINGLE agent's name changing between runs is a bug (it forks the
  agent).
- **Shared process.** If several agents run in one process (e.g. a FastAPI
  app with per-agent routers), do not rely on one global identity. Resource
  attrs are process-global — the first `init()` in a shared process pins
  them, so spans from sibling agents misattribute to that first agent. Stamp
  each agent's identity at the start of its own request path: the identity
  setters (`set_agent_id` / `set_agent_name`) are scoped to the current
  task/context, so calling them in each handler keeps that agent's spans
  attributed to it. Span-level stamps win over the stale resource identity on
  the server.
- **Verify per agent.** In Step 7, fetch traces filtered to THIS agent's UUID
  (`list_traces` with the agent filter) and confirm they carry
  `overmind.agent.id` = the agent's UUID and its `agent_name`.

## Step 6c — Multi-agent repos: the systematic one-at-a-time pass

When the task covers every agent in a repo (or the repo as a whole), do NOT
try to instrument everything in one giant pass. Work ONE agent at a time,
end to end, in a strict loop — each pass has a small, focused context (one
agent's files + one UUID), a failure can't poison sibling agents, and every
agent ships *verified* instead of "hope it worked". A repo with 20 agents =
20 small successful passes, not one huge risky one.

The loop:

1. **Discover.** `list_agents` (MCP) — or the agents named in the task
   prompt. The work unit is N separate passes.
1. **Pick one agent.** Start with the first, and never start the next until
   the current one is done and verified.
1. **Fetch its card.** `get_agent` → top-level `id` (bare UUID) and `flow`
   (`flow_truncated` when clipped). Note `agent_path` /
   `modes[*].entrypoint_fn` — the exact files this agent owns.
1. **Instrument only that agent.** Follow Steps 0-6b scoped to THIS agent:
   touch only its files, stamp its UUID verbatim, leave sibling agents' code
   alone (shared infrastructure is fine to instrument once).
1. **Run + verify only that agent.** Run its entrypoint, flush, then fetch
   traces filtered to ITS UUID (Step 7). Audit against the baseline: the
   trace's `agent` equals this agent's UUID, `agent_name` constant, model +
   tokens + cost populated, inputs/outputs on the entry point, no secrets.
1. **Close it.** Fix gaps until this agent's trace clears. Then move to the
   next agent (back to step 2).

Only at the end, report each agent with its trace link.

## Step 7 — Flush, then run and audit (required)

`overmind.run(...)` flushes on exit. Code paths without it (short scripts,
signal handlers) need an explicit flush before the process exits:

```python
overmind.force_flush_traces()
```

Instrumentation isn't done when the code compiles. This is a loop you own as
the agent:

**a.** Run the instrumented path end-to-end so a real trace is sent.

**b.** Fetch it via Overmind MCP: `list_traces` (newest) → `get_trace` on
that `trace_id`. Do not curl REST. When your task names one agent in a
multi-agent repo, fetch filtered to that agent's UUID — never the repo-wide
newest trace, which may belong to a sibling agent.

**c.** Audit against the [baseline table](#what-a-good-trace-carries). On the
list row check `agent_id` (the agent's UUID, verbatim) and `agent_name`,
`model`, `total_tokens`, `total_cost`, and session grouping for multi-turn
apps; on the detail spans check `span_type` variety (not everything
`llm_call`, and not everything `entry_point`), inputs/outputs on the entry
point and key steps (`overmind.input.data` / `overmind.output.data` on span
attributes), and that no secrets appear in captured payloads. If the trace's
agent UUID differs from the card's, the identity stamp is wrong — fix it
before anything else.

**d.** Fix every gap, re-run, re-fetch. Repeat until the trace clears the
baseline. Then report what is traced.

If nothing shows up:

- Confirm `OVERMIND_API_KEY` is set in the running process.
- On the fan-out path, confirm the existing object really is an SDK
  `TracerProvider` (a no-op default won't accept processors) and that
  `force_flush_traces()` (or the app) ran long enough to export.
- Spans emitted outside a run boundary are dropped as orphans — add the
  `overmind.run(...)` bracket (Step 4).
- `init(debug=True)` prints the resolved endpoint, identity, instrumentors,
  and export mode; `OVERMIND_STRICT_MODE=true` makes missing keys and
  instrumentation packages raise instead of warn.
- Empty `list_traces` means ingest failed — fix instrumentation, don't poll
  REST.

## Common mistakes

| Mistake                                                              | Consequence                                              | Fix                                                                                                                                              |
| -------------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| No flush in scripts/serverless                                       | Traces silently never sent                               | `force_flush_traces()` before exit                                                                                                               |
| Init after LLM clients are created                                   | Provider calls not instrumented                          | Call `init()` at process start, before client construction                                                                                       |
| `overmind.init()` on top of an existing `TracerProvider`             | OTel keeps the first provider; Overmind attaches nothing | Fan-out path (Step 3b)                                                                                                                           |
| Agent name varies per run/env                                        | Each variant becomes a separate agent                    | Set `agent_id` (UUID) once and keep `agent_name` constant — distinct names across DIFFERENT agents are correct, drift on ONE agent is the bug    |
| Invented / mangled `agent_id` (UUID)                                 | Traces attribute to the wrong or a brand-new agent       | Copy the UUID verbatim from `get_agent` (Step 0); if it is missing or not a UUID, stop and report                                                |
| Several agents share one process and one global identity             | All spans land under one agent                           | Stamp each agent's `agent_id`/`agent_name` at its own entry point (Step 6b)                                                                      |
| Only auto-instrumentation, no decorators                             | Flat traces with no inputs/outputs and no step structure | Decorate the entry point and key steps (Step 6)                                                                                                  |
| Credentials (API keys, tokens, passwords) in decorated function args | Stored verbatim in the trace                             | Mask them before passing; `capture="none"` only as a manual, human-maintained escape hatch — never preemptively for data that might be sensitive |
| No `set_conversation_id` in a chat app                               | Sessions view stays empty                                | Stamp the thread/conversation id per request                                                                                                     |
