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
- [ ] 1. Detect existing telemetry (OpenTelemetry, Traceloop, LangSmith, etc.)
- [ ] 2. Install the SDK and set env vars
- [ ] 3. Initialise — greenfield OR fan-out onto the existing provider, with the agent's identity (Step 3a/3b)
- [ ] 4. Auto-instrument the LLM providers in use
- [ ] 5. Add custom spans where useful
- [ ] 5a. Emit the runtime eval envelope (intent / checkpoint / expect / eval_context / end_conversation)
- [ ] 5b. Scope everything to the ONE agent your task names — identity, files, verification
- [ ] 5c. Multi-agent repos: run the systematic one-at-a-time pass
- [ ] 6. Flush on shutdown, then run the app and audit traces via MCP
```

## What a good trace carries

Audit every integration — new or existing — against this baseline before
calling it done. Fetch a real trace with `list_traces` → `get_trace`
([telemetry.md](telemetry.md)); do not ask the user to describe the Console.

| Requirement             | How                                                                                                                                                                                                                                        | Why                                                               |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| Agent identity          | `agent_id=` **and** `agent_name=` in `init()` (or `set_agent_id()` + `set_agent_name()`). Get BOTH from `get_agent` via MCP — copy `agent_id` verbatim (it is a UUID; never invent, truncate, or reformat it)                                | The server resolves `overmind.agent.id` by direct UUID lookup (drift-proof); `agent_name` is display + fallback. Name-only stamps risk a slug mismatch with the code-scan agent, which mints a duplicate agent whose live trace scoring silently no-ops |
| Per-agent scoping       | When your task names ONE agent in a multi-agent repo, stamp THAT agent's identity only — never the repo's, never a generic name                                                                                                             | Traces group under the right agent in the Console; sibling agents' spans are untouched |
| Model + token usage     | Automatic via provider auto-instrumentation (Step 4); raw-OTel spans should carry `gen_ai.request.model` / `gen_ai.usage.*`                                                                                                                | Cost is computed server-side from these                           |
| Inputs and outputs      | The decorators (Step 5) capture call args and return values automatically; make sure the entry point and key steps are decorated so the trace shows what the agent saw and produced                                                        | A trace without I/O can't be debugged or turned into eval data    |
| Sensitive data excluded | Not for agents — trace normally and mask credential fields (API keys, tokens, passwords) before they reach decorated functions. `@observe_safe()` (traces timing/status, no values) is a manual escape hatch for human implementation only | Inputs/outputs are stored verbatim                                |
| Session grouping        | `set_conversation_id(...)` per conversation/thread (stamped as `conversation.id`) whenever the app has multi-turn interactions; `@conversation` wraps a handler that owns a conversation. Session grain, conversation-scope `expect` and `end_conversation()` all depend on it                                      | Groups traces into Sessions                                       |
| User attribution        | `set_user(user_id, email=...)` where the app has accounts                                                                                                                                                                                                                  | Per-user filtering and cost attribution                           |
| Span hierarchy + types  | One `@entry_point` at the top; `@workflow` / `@tool` / `@retrieval` for the steps under it, with descriptive names                                                                                                                                                          | Shows which step failed or was slow, instead of one flat LLM call |
| Behaviour anchor        | Every decorator auto-stamps `code.namespace` (`__module__`) + `code.function.name` (`__qualname__`) — the pair is the Behaviour Registry anchor the server binds spans to for task-execution scoring (`start_span` has no function to read, so no stamp)                     | Without the pair the server cannot bind spans to an anchor; executions land `unbound` |
| Git sha                 | `vcs.ref.head.revision` auto-stamped at `init()` — detects `OVERMIND_GIT_SHA` (explicit override), then CI env vars (`GIT_SHA`, `GITHUB_SHA`, …), then `.git/HEAD`; silently omitted when undetectable                                                                      | Lets the server pin executions to the exact code revision          |
| Runtime eval envelope   | The five `overmind.eval.*` span events (`intent`, `expectation`, `context`, `checkpoint`, `conversation_end`), each with `schema_version`=1 + JSON `payload` (Step 5a)                                                                                                       | The scoring inputs: declared intent grounds the judge, expectations become verdicts, checkpoints mark milestones |

## Step 0 — Resolve the agent's identity

The authoritative source for agent identity is `get_agent` via MCP (see
[telemetry.md](telemetry.md)): it returns the agent's `id` (a UUID) and its
`flow` — the capability card with `agent_path`, `modes[*].entrypoint_fn`,
system prompt, and tool surface. Call it with the agent's name/slug before
writing any instrumentation. Copy the returned `id` verbatim — never invent,
shorten, re-format, or "fix" it, and never substitute another agent's id. If
the id is missing or does not look like a UUID, STOP and report instead of
guessing: a wrong id silently attributes every trace to the wrong agent.

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
    agent_id="<agent-uuid>",        # copy verbatim from get_agent — never invent
    agent_name="My Agent",          # this agent's constant display name
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
from overmind.tracing import enable_tracing, get_api_settings, set_agent_id, set_agent_name

api_key, base_url = get_api_settings()  # reads OVERMIND_API_KEY / OVERMIND_API_URL

provider = trace.get_tracer_provider()
if not isinstance(provider, TracerProvider):
    # Nothing real was installed yet — let Overmind own the pipeline instead.
    overmind.init(
        service_name="my-agent",
        agent_id="<agent-uuid>",   # copy verbatim from get_agent
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
    set_agent_id("<agent-uuid>")    # verbatim from get_agent
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
@overmind.entry_point()  # top-level request handler
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

Context manager and current-span helpers:

```python
with overmind.start_span("rerank", span_type=overmind.SpanType.FUNCTION) as span:
    overmind.set_tag("candidate_count", len(candidates))

overmind.set_user("user-123", email="a@b.com")
overmind.set_conversation_id("conv-abc")  # groups spans into one session
overmind.set_agent_id("<agent-uuid>")     # verbatim from get_agent — never invented
overmind.set_agent_name("My Agent")       # keep constant for this agent

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
`@workflow`). A trace whose spans are all `entry_point` is flat: it cannot
show which step failed or was slow. Rule of thumb: if the function has a name
a human would use to describe the agent's work ("search", "lookup_policy",
"rerank"), it should be a span.

## Step 5a — Runtime envelope: declare intent, milestones, expectations

Decorators make a trace *visible*; the runtime envelope makes it *scorable*.
Each call emits a pinned `overmind.eval.*` span event (see the baseline
table). Exact signatures/semantics live in `overmind/evals.py` — read it if
in doubt. All five **no-op (debug log) when there is no recording span**, so
call them inside a decorated span:

```python
@overmind.entry_point()
def run(request: dict) -> dict:
    overmind.intent(request["user_message"])  # grounds the judge; omit -> server falls back to the first user message
    overmind.eval_context(user_tier="premium", retries=3)  # facts for the judge

    overmind.expect("contains", "USD", gate=True)   # hard fail: failure caps the execution score at 0
    overmind.expect("regex", r"\d{4}-\d{2}-\d{2}", id="date-format")
    overmind.expect("schema", {"type": "object", "required": ["amount"]}, scope="trace")
    overmind.expect("checkpoints", ["plan_formed", "payment_confirmed", "receipt_sent"])

    overmind.checkpoint("plan_formed")            # named milestone / turn boundary
    ...
    overmind.checkpoint("payment_confirmed")
    ...
    overmind.end_conversation()   # conversation-scope scoring; needs set_conversation_id / @conversation
```

Semantics:

- **`intent(text, *, source="declared")`** — declare what the user asked for
  this run; the platform grounds judge scoring in it. Declare it at every
  turn boundary in multi-turn agents. When undeclared, the server falls back
  to the first user message.
- **`checkpoint(name)`** — named trajectory milestone / turn boundary;
  `expect(..., kind="checkpoints")` can assert the expected ordered path.
- **`expect(kind, spec, *, id=None, scope="trace", gate=False)`** — runtime
  expectation. `kind` ∈ `contains | regex | schema | constraint |
  checkpoints`; `scope` ∈ `span | trace | conversation`. `id` auto-derives as
  a stable short hash of kind+spec when omitted (the platform dedupes /
  aggregates per expectation). `schema` takes a JSON schema object,
  `checkpoints` an ordered list of names, `constraint` natural-language text.
  `gate=True` makes a failure a hard fail that caps the execution's score at 0.
- **`eval_context(**facts)`** — runtime facts for the judge; values coerced
  like `set_tag`.
- **`end_conversation()`** — signal the conversation is complete; triggers
  conversation-scope scoring (requires a conversation id from
  `set_conversation_id` / `@conversation`).

### Where to instrument — anchor priority

`get_instrumentation_context(agent)` returns the agent's Behaviour anchors —
the `code.namespace` + `code.function.name` span-identity pairs the server
binds spans to — ranked `entry → discriminating → supplementary`. Each anchor
carries an `instrumented` bool, an `import_line`, a `verification_hint`
(tells you how to confirm the `code.namespace=…` / `code.function.name=…`
pair really arrives on the trace), and the context bundles `remaining` (the
anchors not yet instrumented) and `indistinguishable_pairs`.

Work `remaining` first, in priority order:

1. **entry** — the task's entry point(s); the spine of the execution row.
2. **discriminating** — steps that distinguish one execution/outcome from
   another (scoring-critical).
3. **supplementary** — supporting steps (nice-to-have structure).

Honour each anchor's `verification_hint` when instrumenting it, and resolve
`indistinguishable_pairs` — two anchors the trace cannot tell apart because
their spans stamp the same identity — by naming spans/functions so the
identities disambiguate.

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
2. **Pick one agent.** Start with the first, and never start the next until
   the current one is done and verified.
3. **Fetch its card.** `get_agent` → its `id` (UUID) and capability card.
   Note `agent_path` / `modes[*].entrypoint_fn` — the exact files this agent
   owns.
4. **Instrument only that agent.** Follow Steps 0-5b scoped to THIS agent:
   touch only its files, stamp its UUID verbatim, leave sibling agents' code
   alone (shared infrastructure is fine to instrument once).
5. **Run + verify only that agent.** Run its entrypoint, flush, then fetch
   traces filtered to ITS UUID (Step 6). Audit against the baseline: the
   trace's `agent` equals this agent's UUID, `agent_name` constant, model +
   tokens + cost populated, inputs/outputs on the entry point, no secrets.
6. **Close it.** Fix gaps until this agent's trace clears. Then move to the
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

**b.** Fetch it via Overmind MCP — [telemetry.md](telemetry.md):
`list_traces` (newest) → `get_trace` on that `trace_id`. Do not curl REST.
When your task names one agent in a multi-agent repo, fetch filtered to that
agent's UUID — never the repo-wide newest trace, which may belong to a
sibling agent.

**b2.** Task-execution-first audit — execution rows are the primary
observability row for trajectory instrumentation. `list_task_executions` on
that agent → `get_task_execution(id)` on the row:
`binding_source` must be `anchor_join` or `declared` (an `unbound` execution
means the code identity + git sha never matched the server's registry — fix
the identity or the code path, don't move on), `user_intent` must be the
declared intent (or the expected first-user-message fallback), and
`success_score` / `session_score` must be populated. Then pull
`behaviour_coverage` on the agent to confirm every step evaluator got
evidence and no `remaining` anchors are still silent.

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
| Invented / mangled `agent_id` (UUID)                                 | Traces attribute to the wrong or a brand-new agent       | Copy the UUID verbatim from `get_agent` (Step 0); if it is missing or not a UUID, stop and report                                                |
| Several agents share one process and one global identity             | All spans land under one agent                           | Stamp each agent's `agent_id`/`agent_name` at its own entry point (Step 5b)                                                                       |
| Only auto-instrumentation, no decorators                             | Flat traces with no inputs/outputs and no step structure | Decorate the entry point and key steps (Step 5)                                                                                                   |
| Credentials (API keys, tokens, passwords) in decorated function args | Stored verbatim in the trace                             | Mask them before passing; `@observe_safe()` only as a manual, human-maintained escape hatch — never preemptively for data that might be sensitive |
| No `set_conversation_id` in a chat app                               | Sessions view stays empty                                | Stamp the thread/conversation id per request                                                                                                      |
