---
name: overmind-agent-telemetry
description: Add Overmind tracing to an existing AI/LLM project so its traces are sent to Overmind. Handles projects that already have OpenTelemetry or another telemetry library configured (fan-out) as well as greenfield projects. Use when the user asks to add Overmind telemetry/observability/tracing, send traces to Overmind, or instrument an existing agent with Overmind.
---

# Overmind Agent Telemetry

Wire an existing Python project into Overmind so every LLM call and traced
function is exported to Overmind (`https://api.overmindlab.ai/api/v1/traces`).
Overmind is built on OpenTelemetry, so it can either own the tracing pipeline
or ride alongside a telemetry stack the project already has.

## Workflow

```
- [ ] 1. Detect existing telemetry (OpenTelemetry, Traceloop, LangSmith, etc.)
- [ ] 2. Install the SDK and set env vars
- [ ] 3. Initialize — greenfield OR fan-out onto the existing provider
- [ ] 4. Auto-instrument the LLM providers in use
- [ ] 5. Add custom spans where useful
- [ ] 6. Flush on shutdown and verify traces land in the Console
```

## Step 1 — Detect existing telemetry

Grep the project before writing any code. The result decides Step 3.

```bash
rg -n "set_tracer_provider|TracerProvider|opentelemetry|traceloop|Traceloop|langsmith|OTEL_EXPORTER" --glob '!**/.venv/**'
```

- **No matches** → greenfield path (Step 3a). `overmind.init()` creates and
  installs the provider.
- **A `TracerProvider` is already set** (OTel directly, Traceloop/OpenLLMetry,
  LangSmith's OTel bridge, etc.) → fan-out path (Step 3b). OpenTelemetry only
  honors the **first** `set_tracer_provider()` call and ignores later ones with
  a warning, so calling `overmind.init()` on top of an existing provider would
  silently attach nothing. Instead, add Overmind's exporter to the provider the
  project already owns.

## Step 2 — Install and configure

```bash
uv add overmind        # or: pip install overmind
```

Required environment variable (create a key at
https://console.overmindlab.ai/projects):

```bash
export OVERMIND_API_KEY=<your-api-key>
```

Optional identity/config (all have env-var equivalents read by `init()`):

| Env var | Purpose |
| --- | --- |
| `OVERMIND_SERVICE_NAME` | Service name on the traces |
| `OVERMIND_AGENT_NAME` | Human-readable agent name |
| `OVERMIND_AGENT_ID` | Agent UUID (preferred over name once registered) |
| `OVERMIND_ENVIRONMENT` | e.g. `production` (default `development`) |
| `OVERMIND_API_URL` | Override the trace endpoint base URL |

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
@overmind.entry_point()   # top-level request handler
def run(payload: dict) -> dict: ...

@overmind.workflow()      # multi-step orchestration
def pipeline(): ...

@overmind.tool()          # a tool/function the agent can call
def search(query: str) -> list[dict]: ...

@overmind.retrieval()     # RAG / vector lookup
def fetch_docs(q: str): ...

@overmind.function()      # any other traced function
def score(x): ...
```

Context manager and current-span helpers:

```python
with overmind.start_span("rerank", span_type=overmind.SpanType.FUNCTION) as span:
    overmind.set_tag("candidate_count", len(candidates))

overmind.set_user("user-123", email="a@b.com")
overmind.set_conversation_id("conv-abc")   # groups spans into one session
overmind.set_agent_name("support-bot")

try:
    ...
except Exception as exc:
    overmind.capture_exception(exc)        # marks the span errored
    raise
```

`start_span` and the decorators use the ambient tracer, so they attach to
whichever provider is active — greenfield or fan-out.

## Step 6 — Flush on shutdown and verify

Batch export is async; flush before a short-lived process exits or spans are
lost:

```python
overmind.force_flush_traces()
```

Verify: run the app, then check traces appear in the
[Console](https://console.overmindlab.ai/). If nothing shows up:

- Confirm `OVERMIND_API_KEY` is set in the running process.
- On the fan-out path, confirm the existing object really is an SDK
  `TracerProvider` (a no-op default won't accept processors) and that
  `force_flush_traces()` (or the app) ran long enough to export.
- Set `OVERMIND_STRICT_MODE=true` to make missing instrumentation packages
  raise instead of warn.
