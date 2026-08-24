# Instrumenting an application with Overmind tracing

Wire an existing Python project into Overmind so every LLM call and traced
function is exported. Overmind is built on OpenTelemetry, so it can either
own the tracing pipeline or ride alongside a telemetry stack the project
already has.

The SDK surface is the `overmind` Python package (`init`, decorators,
`force_flush_traces`). This file may lag the package — prefer the code in
the installed SDK if they disagree.

Verify traces through Overmind MCP (`list_traces` / `get_trace`), not REST.
See [telemetry.md](telemetry.md).

## Workflow

```
- [ ] 0. Resolve the agent's identity — `get_agent` via MCP; copy its `id` (UUID) verbatim
- [ ] 0a. Treat the MCP placement plan as the source of required and known targets for this pass, not a closed-world scanner allowlist; run `overmind instrumentation check --plan-file <path> [--root <path>] [--format json]` before editing
- [ ] 1. Detect existing telemetry (OpenTelemetry, Traceloop, LangSmith, etc.)
- [ ] 2. Install the SDK and set env vars
- [ ] 3. Initialise — greenfield OR fan-out onto the existing provider, with the agent's identity (Step 3a/3b)
- [ ] 4. Auto-instrument the LLM providers in use
- [ ] 5. Add useful nested spans as appropriate; ordinary spans are not Behaviour anchors
- [ ] 5a. Declare exactly one task root per trace (`@overmind.task("<behaviour key>")` on the planned entry point); shared helpers are never independent tasks
- [ ] 5b. Scope everything to the ONE agent your task names — identity, files, verification
- [ ] 5c. Multi-agent repos: run the systematic one-at-a-time pass
- [ ] 6. Flush on shutdown, then run the app and audit traces via MCP
```

## What a good trace carries

Audit every integration — new or existing — against this baseline before
calling it done. Fetch a real trace with `list_traces` → `get_trace`
([telemetry.md](telemetry.md)); do not ask the user to describe the Console.

| Requirement             | How                                                                                                                                                                                                                                                                            | Why                                                                                                                                                                                                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Agent identity          | `agent_id=` **and** `agent_name=` in `init()` (or `set_agent_id()` + `set_agent_name()`). Get BOTH from `get_agent` via MCP — copy `agent_id` verbatim (it is a UUID; never invent, truncate, or reformat it)                                                                  | The server resolves `overmind.agent.id` by direct UUID lookup (drift-proof); `agent_name` is display + fallback. Name-only stamps risk a slug mismatch with the code-scan agent, which mints a duplicate agent whose live trace scoring silently no-ops |
| Per-agent scoping       | When your task names ONE agent in a multi-agent repo, stamp THAT agent's identity only — never the repo's, never a generic name                                                                                                                                                | Traces group under the right agent in the Console; sibling agents' spans are untouched                                                                                                                                                                  |
| Model + token usage     | Automatic via provider auto-instrumentation (Step 4); raw-OTel spans should carry `gen_ai.request.model` / `gen_ai.usage.*`                                                                                                                                                    | Cost is computed server-side from these                                                                                                                                                                                                                 |
| Inputs and outputs      | Decorators capture call args and return values by default, including `@overmind.task`; use `capture_io=False` only for an explicit no-payload requirement                                                                                                                        | A trace without I/O cannot be debugged or attributed reliably                                                                                                                                                                                           |
| Payload policy          | Do not censor, summarize, truncate, or redact ordinary prompts, context, tool data, or model outputs. Mask only a clearly identifiable credential field, or use `capture_io=False` when no payload is explicitly required                                                                 | Over-redaction destroys attribution evidence                                                                                                                                                                                                            |
| Session grouping        | `set_conversation_id(...)` per conversation/thread (stamped as `conversation.id`) whenever the app has multi-turn interactions; `@conversation` wraps a handler that owns a conversation | Groups traces into Sessions                                                                                                                                                                                                                             |
| User attribution        | `set_user(user_id, email=...)` where the app has accounts                                                                                                                                                                                                                      | Per-user filtering and cost attribution                                                                                                                                                                                                                 |
| Span hierarchy + types  | One `@overmind.task("<behaviour key>")` at the planned task root; `@workflow` / `@tool` / `@retrieval` / `@function` for useful nested steps, with descriptive names                                                                                                                      | Shows which step failed or was slow, instead of one flat LLM call                                                                                                                                                                                       |
| Behaviour anchor        | Only the MCP plan's task root and explicitly identified anchors are Behaviour anchors. Ordinary useful spans and shared helpers are nested telemetry, not independent task roots or anchors                                                                                  | Keeps attribution tied to the declared placement contract                                                                                                                                                                                             |
| Git sha                 | `vcs.ref.head.revision` auto-stamped at `init()` — detects `OVERMIND_GIT_SHA` (explicit override), then CI env vars (`GIT_SHA`, `GITHUB_SHA`, …), then `.git/HEAD`; silently omitted when undetectable                                                                         | Lets the server pin executions to the exact code revision                                                                                                                                                                                               |

## Step 0 — Resolve the agent's identity

The authoritative source for agent identity is `get_agent` via MCP (see
[telemetry.md](telemetry.md)): it returns the agent's `id` (a UUID) and its
`flow` — the capability card with `agent_path`, `modes[*].entrypoint_fn`,
system prompt, and tool surface. Call it with the agent's name/slug before
writing any instrumentation. Copy the returned `id` verbatim — never invent,
shorten, re-format, or "fix" it, and never substitute another agent's id. If
the id is missing or does not look like a UUID, STOP and report instead of
guessing: a wrong id silently attributes every trace to the wrong agent.

Then map the agent to its tasks with `list_behaviours(agent)`: behaviour
keys, entry anchors, anchor sequence, terminal, and execution/unbound
counts. The key for each task is what you declare in Step 5a.

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
    providers=["openai", "anthropic"],  # auto-instrument these SDKs; see Step 4
)
```

`providers=[]` (empty list) enables every supported provider;
omitting `providers` enables none.

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
        providers=["openai"],
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
    enable_tracing(["openai", "anthropic"])
    # Fan-out path: identity is NOT stamped by an on-start processor, so stamp it
    # explicitly on the spans you decorate (Step 5) and via the context helpers.
    set_agent_id("<agent-uuid>")  # verbatim from get_agent
    set_agent_name("My Agent")
```

Notes:

- The project's existing backend keeps receiving spans; Overmind gets a copy.
- Overmind's server reads canonical `genai.*` usage attributes. Spans from
  third-party auto-instrumentors that only emit OTel `gen_ai.*` keys are
  bridged automatically **only when Overmind owns the provider**. On the
  fan-out path, prefer Overmind's own auto-instrumentation (`enable_tracing`)
  or the decorators in Step 5 so token/cost rollups populate.

## Step 4 — Auto-instrument LLM providers

Supported providers: `openai`, `anthropic`, `google`, `agno`. Each needs the
matching instrumentation package installed (bundled with `overmind`). Pass them
to `init(providers=[...])` (greenfield) or `enable_tracing([...])` (fan-out).
Instrumentation is idempotent and safe to call more than once.

## Step 5 — Add custom spans

Decorators (sync and async) — use the type that matches the code:

```python
@overmind.task(
    "behaviour-key"
)  # one primary boundary; key from list_behaviours
def run(payload: dict) -> dict: ...


@overmind.workflow()  # multi-step orchestration
def pipeline(): ...


@overmind.tool()  # a tool/function the agent can call
def search(query: str) -> list[dict]: ...


@overmind.retrieval()  # RAG / vector lookup
def fetch_docs(q: str): ...


@overmind.function()  # any other traced function
def score(x): ...
```

`@overmind.task("<behaviour key>")` opens the task's `entry_point` unit span,
captures code identity and I/O by default, and stamps the declared key — copy the key from
`list_behaviours(agent)`. A dispatcher must choose the key before entering this
boundary. Nested work is a `workflow`, `function`, `retrieval`, or `tool` span;
the context-manager form is supported for a fixed-key dynamic boundary, but provide `entrypoint=<callable>` when possible and record
I/O explicitly; it cannot infer callable identity or final output on its own.
`name=` on decorators stamps `overmind.anchor.name`, a rename-proof semantic
anchor. Without it, the qualname (`code.namespace` + `code.function.name`) is
the diagnostic identity. Set `capture_io=False` only when the caller explicitly
requires no payload; do not use it to censor ordinary private or domain data.

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

**Minimum for a good trace.** Instrument the planned task root and the
placement plan's remaining anchors. Other useful model, tool, or helper spans
are optional nested telemetry; they are not Behaviour anchors unless the plan
explicitly identifies them. Do not add evaluator prompts, rubrics, judge
logic, or scoring metadata as part of instrumentation.

### Shared-entry dynamic routes

Keep fixed tasks as `@overmind.task("key")`. When several registered tasks
share one entry function, use `@overmind.task(key_from=selector)`. The selector
runs before span creation and must return a registered, non-empty key; a
selector error or invalid result produces no task span. This is a
decorator-only boundary, not a context-manager placement.

### Where to instrument — anchor priority

`get_instrumentation_context(agent)` returns a versioned placement plan: the
canonical file, qualname, AST insertion point, task key or selector, allowed
keys, analyzed revision, and placement constraints. It also returns the
Behaviour anchors — the `code.namespace` + `code.function.name` identity pairs
ranked `entry → discriminating → supplementary` — plus `remaining` and
`indistinguishable_pairs`.

The plan identifies required and known targets for this pass; it is not a
closed-world scanner allowlist. Use it to place the task root and check known
anchors without claiming that unlisted files or targets are forbidden.

Save the returned `plan_id`. After editing, run the local
`overmind instrumentation check --plan-file <path>` against that plan. After a
real run, call the read-only MCP `verify_instrumentation_trace` with the agent,
`plan_id`, and exact `trace_id`. It is the attribution gate for the declared
task root and plan-identified anchors.

Work `remaining` first, in priority order:

1. **entry** — the task's entry point(s); the spine of the execution row.
1. **discriminating** — steps that distinguish one execution/outcome from
   another (outcome-critical).
1. **supplementary** — supporting steps (nice-to-have structure).

Honour each anchor's `verification_hint` when instrumenting it, and resolve
`indistinguishable_pairs` — two anchors the trace cannot tell apart because
their spans stamp the same identity — by naming spans/functions so the
identities disambiguate.

### Declared keys vs the failsafe

A declared key is the strongest binding evidence: a known key on the unit span
binds even when the git sha is missing, but an unknown key falls through to
structural matching and is flagged `declared_key_unknown`. A revision mismatch,
unknown anchor, or missing evidence is still a verification failure; declaration
does not make stale code current.

Without a declared key the server structurally matches span identity against
the registry: scored matched/expected coverage-fraction, binds only when the
best beats the runner-up by ≥1.5×, and is file-path-joined (a bare `run` in
`entry.py` cannot suffix-collide with `app.b.run`). Ties and weak matches
stay `unbound_ambiguous`; a sole candidate still binds but with zero evidence
— flagged `bind_review` at confidence 0.0, never a silent overconfident bind.

So verification checks `attribution_verdict` / `binding_confidence`, plus
unknown anchors, version match, primary-task count, entry I/O, and final output;
`bound_structurally` means deterministic fallback evidence, not declared
instrumentation compliance.

## Step 5b — Instrumenting ONE agent in a multi-agent repo

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
- **Verify per agent.** In Step 6, fetch traces filtered to THIS agent's UUID
  (`list_traces` with the agent filter, see [telemetry.md](telemetry.md)) and
  confirm they carry `overmind.agent.id` = the agent's UUID and its
  `agent_name`.

## Step 5c — Multi-agent repos: the systematic one-at-a-time pass

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
1. **Fetch its card.** `get_agent` → its `id` (UUID) and capability card.
   Note `agent_path` / `modes[*].entrypoint_fn` — the exact files this agent
   owns.
1. **Instrument only that agent.** Follow Steps 0-5b scoped to THIS agent:
   touch only its files, stamp its UUID verbatim, leave sibling agents' code
   alone (shared infrastructure is fine to instrument once).
1. **Run + verify only that agent.** Run its entrypoint, flush, then fetch
   traces filtered to ITS UUID (Step 6). Audit against the baseline: the
   trace's `agent` equals this agent's UUID, `agent_name` constant, model +
   tokens + cost populated, inputs/outputs on the entry point, no secrets.
1. **Close it.** Fix gaps until this agent's trace clears. Then move to the
   next agent (back to step 2).

Only at the end, report each agent with its trace link.

## Step 6 — Flush on shutdown, then run and audit (required)

Batch export is async; flush before a short-lived process exits or spans are
lost:

```python
overmind.force_flush_traces()
```

Instrumentation isn't done when the code compiles. This is a loop you own as
the agent:

**a.** Run the instrumented path end-to-end so a real trace is sent.

**b.** Fetch the exact trace via Overmind MCP — [telemetry.md](telemetry.md):
retain the `trace_id` for the exercised route, use `list_traces` only to
identify that exact run (filtered to the agent UUID), then call `get_trace` on
that `trace_id`. Do not accept an unrelated newest trace or curl REST.

**b2.** Call the read-only MCP
`verify_instrumentation_trace(agent, plan_id, trace_id)` on that exact run. A
pass requires `bound_declared`, confidence `1.0`, the expected Behaviour
version and SHA, the expected entry anchor, a raw `entry_point` unit span, and
no route flags. A structural fallback is useful failsafe evidence but is not
strict instrumentation success. Fix any failed or unverifiable check before
moving on.

**c.** Audit the raw spans against the [baseline table](#what-a-good-trace-carries)
too — this is complementary to b2, not a replacement. On
the list row check `agent_id` (the agent's UUID, verbatim) and `agent_name`,
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
- Set `OVERMIND_STRICT_MODE=true` to make missing instrumentation packages
  raise instead of warn.
- Empty `list_traces` means ingest failed — fix instrumentation, don't poll
  REST.

## Common mistakes

| Mistake                                                              | Consequence                                              | Fix                                                                                                                                               |
| -------------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| No flush in scripts/serverless                                       | Traces silently never sent                               | `force_flush_traces()` before exit                                                                                                                |
| Init after LLM clients are created                                   | Provider calls not instrumented                          | Call `init()` at process start, before client construction                                                                                        |
| `overmind.init()` on top of an existing `TracerProvider`             | OTel keeps the first provider; Overmind attaches nothing | Fan-out path (Step 3b)                                                                                                                            |
| Agent name varies per run/env                                        | Each variant becomes a separate agent                    | Set `agent_id` (UUID) once and keep `agent_name` constant — distinct names across DIFFERENT agents are correct, drift on ONE agent is the bug     |
| Invented / mangled `agent_id` (UUID)                                 | Traces attribute to the wrong or a brand-new agent       | Copy the UUID verbatim from `get_agent` (Step 0); if it is missing or not a UUID, stop and report                                                 |
| Several agents share one process and one global identity             | All spans land under one agent                           | Stamp each agent's `agent_id`/`agent_name` at its own entry point (Step 5b)                                                                       |
| Only auto-instrumentation, no decorators                             | Flat traces with no inputs/outputs and no step structure | Decorate the entry point and key steps (Step 5)                                                                                                   |
| Credentials (API keys, tokens, passwords) in decorated function args | Stored verbatim in the trace                             | Mask the clearly identifiable credential field, or use `capture_io=False` when no payload is explicitly required; do not censor ordinary data merely because it might be sensitive |
| No `set_conversation_id` in a chat app                               | Sessions view stays empty                                | Stamp the thread/conversation id per request                                                                                                      |
