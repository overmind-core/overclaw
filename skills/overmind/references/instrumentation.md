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
- [ ] 1. Detect existing telemetry (OpenTelemetry, Traceloop, LangSmith, etc.)
- [ ] 2. Install the SDK and set env vars
- [ ] 3. Initialise — greenfield OR fan-out onto the existing provider
- [ ] 4. Auto-instrument the LLM providers in use
- [ ] 5. Add custom spans where useful
- [ ] 6. Flush on shutdown, then run the app and audit traces via MCP
```

## What a good trace carries

Audit every integration — new or existing — against this baseline before
calling it done. Fetch a real trace with `list_traces` → `get_trace`
([telemetry.md](telemetry.md)); do not ask the user to describe the Console.

| Requirement             | How                                                                                                                                                                                                                                        | Why                                                               |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| Agent name              | `agent_name=` in `init()`, `OVERMIND_AGENT_NAME`, or `set_agent_name()` — pick one constant string; the server slugifies it into the agent's identity, so renaming forks a new agent                                                       | Traces group under one agent                                      |
| Model + token usage     | Automatic via provider auto-instrumentation (Step 4); raw-OTel spans should carry `gen_ai.request.model` / `gen_ai.usage.*`                                                                                                                | Cost is computed server-side from these                           |
| Inputs and outputs      | The decorators (Step 5) capture call args and return values automatically; make sure the entry point and key steps are decorated so the trace shows what the agent saw and produced                                                        | A trace without I/O can't be debugged or turned into eval data    |
| Sensitive data excluded | Not for agents — trace normally and mask credential fields (API keys, tokens, passwords) before they reach decorated functions. `@observe_safe()` (traces timing/status, no values) is a manual escape hatch for human implementation only | Inputs/outputs are stored verbatim                                |
| Session grouping        | `set_conversation_id(...)` per conversation/thread (stamped as `conversation.id`) whenever the app has multi-turn interactions                                                                                                             | Groups traces into Sessions                                       |
| User attribution        | `set_user(user_id, email=...)` where the app has accounts                                                                                                                                                                                  | Per-user filtering and cost attribution                           |
| Span hierarchy + types  | One `@entry_point` at the top; `@workflow` / `@tool` / `@retrieval` for the steps under it, with descriptive names                                                                                                                         | Shows which step failed or was slow, instead of one flat LLM call |

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
from overmind.tracing import enable_tracing, get_api_settings

api_key, base_url = get_api_settings()  # reads OVERMIND_API_KEY / OVERMIND_API_URL

provider = trace.get_tracer_provider()
if not isinstance(provider, TracerProvider):
    # Nothing real was installed yet — let Overmind own the pipeline instead.
    overmind.init(service_name="my-agent", providers=["openai"])
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
overmind.set_agent_name("support-bot")

try:
    ...
except Exception as exc:
    overmind.capture_exception(exc)  # marks the span errored
    raise
```

`start_span` and the decorators use the ambient tracer, so they attach to
whichever provider is active — greenfield or fan-out.

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

**c.** Audit against the [baseline table](#what-a-good-trace-carries). On the
list row check `agent_name`, `model`, `total_tokens`, `total_cost`, and
session grouping for multi-turn apps; on the detail spans check `span_type`
variety (not everything `llm_call`), inputs/outputs on the entry point and
key steps (`overmind.input.data` / `overmind.output.data` on span
attributes), and that no secrets appear in captured payloads.

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
| Agent name varies per run/env                                        | Each variant becomes a separate agent                    | One constant `agent_name`, set once                                                                                                               |
| Only auto-instrumentation, no decorators                             | Flat traces with no inputs/outputs and no step structure | Decorate the entry point and key steps (Step 5)                                                                                                   |
| Credentials (API keys, tokens, passwords) in decorated function args | Stored verbatim in the trace                             | Mask them before passing; `@observe_safe()` only as a manual, human-maintained escape hatch — never preemptively for data that might be sensitive |
| No `set_conversation_id` in a chat app                               | Sessions view stays empty                                | Stamp the thread/conversation id per request                                                                                                      |
