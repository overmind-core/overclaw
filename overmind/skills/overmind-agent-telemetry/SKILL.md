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
- [ ] 3. Initialise — greenfield OR fan-out onto the existing provider
- [ ] 4. Bracket the agent's entry point with overmind.run(...)
- [ ] 5. Carve phases into turn units; place deliver() in the producing unit
- [ ] 6. Decorate anchor functions; add custom spans where useful
- [ ] 7. Verify traces land in the Console
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
  honours the **first** `set_tracer_provider()` call and ignores later ones with
  a warning, so calling `overmind.init()` on top of an existing provider would
  silently attach nothing. Instead, add Overmind's exporter to the provider the
  project already owns.

## Step 2 — Install and configure

```bash
uv add overmind        # or: pip install overmind
```

The core package is tracing-only. Add extras when needed:
`overmind[langchain]` (LangChain/LangGraph auto-instrumentation),
`overmind[inference]` (token-cost enrichment via litellm),
`overmind[cli]` (the `overmind` command), `overmind[tracing-full]`
(requests/httpx/logging spans).

Required environment variable (create a key at
https://console.overmindlab.ai/projects):

```bash
export OVERMIND_API_KEY=<your-api-key>
```

Optional identity/config (all have env-var equivalents read by `init()`):

| Env var | Purpose |
| --- | --- |
| `OVERMIND_SERVICE_NAME` | Service name on the traces |
| `OVERMIND_AGENT_ID` | Capability UUID from the Console — the identifier; stable through renames |
| `OVERMIND_AGENT_NAME` | Optional display label (slug or name); advisory when an id is set |
| `OVERMIND_ENVIRONMENT` | e.g. `production` (default `development`) |
| `OVERMIND_API_URL` | Override the trace endpoint base URL |

## Step 3a — Greenfield init

Call once at process start, before the traced code runs:

```python
import overmind

overmind.init(
    service_name="my-agent",
    providers="auto",   # instrument every installed provider SDK
)
```

`providers="auto"` detects the installed target libraries (openai, anthropic,
google, agno, langchain) and enables every one whose instrumentor is also
present, logging the resolved list. Name providers explicitly
(`providers=["openai"]`) to pin the set; `providers=[]` enables all known;
omitting `providers` enables none.

`init()` is graceful: without `OVERMIND_API_KEY` it logs, returns `False`, and
every decorator/helper becomes a no-op — safe to ship in apps where Overmind
is optional. Set `OVERMIND_STRICT_MODE=true` to make a missing key raise.
`init(debug=True)` prints the endpoint, resolved identity, enabled
instrumentors, and export mode.

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
    overmind.init(service_name="my-agent", providers="auto")
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
```

Notes:
- The project's existing backend keeps receiving spans; Overmind gets a copy.
- Overmind's server reads canonical `genai.*` usage attributes. Spans from
  third-party auto-instrumentors that only emit OTel `gen_ai.*` keys are
  bridged automatically **only when Overmind owns the provider**. On the
  fan-out path, prefer Overmind's own auto-instrumentation (`enable_tracing`)
  or the decorators below so token/cost rollups populate.

## Step 4 — Bracket the run

Every agent execution needs exactly one run boundary. `overmind.run(...)` is
the one scope that covers it — capability identity, the entry-point run span,
intent, conversation id, error status, and a flush on exit:

```python
with overmind.run("triage-run", intent=request["question"], conversation_id=ticket_id) as run:
    answer = agent.invoke(request)
    run.deliver(answer)   # terminal deliverable, auto-grounded
```

As a decorator (sync or async) every parameter except `name` also accepts a
callable receiving the wrapped call's arguments, and the run span carries the
function's code identity — one decoration also satisfies an entry-point
scan-contract anchor:

```python
class Agent:
    @overmind.run(intent=lambda self, *a, **k: self.task,
                  conversation_id=lambda self, *a, **k: self.task_id)
    async def run(self):
        ...
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
- Multi-capability agents scope identity with
  `overmind.capability("slug-or-name", id=...)` — pin `id=` (the capability
  UUID) whenever you have it; entering a different capability mid-trace is a
  handoff and opens a new unit automatically.

For LangGraph agents, `overmind.integrations.langgraph.bind()` does this
declaratively — call it on the `StateGraph` after `add_node()`, before
`compile()`:

```python
from overmind.integrations import langgraph as overmind_langgraph

overmind_langgraph.bind(
    workflow,
    # Default key per node: slugified node name. Override where the task map
    # groups nodes differently; None opts a node out.
    behaviours={"Bull Researcher": "investment-debate", "Bear Researcher": "investment-debate"},
    deliver="Portfolio Manager",  # this node's return value is the deliverable
)
app = workflow.compile()
```

Full concepts guide: `docs/carving-runs-into-units.md` in the SDK repo.

## Step 6 — Decorate anchors and add custom spans

**Every function the scanned task map anchors on must be decorated** — an
undecorated anchor emits no `code.namespace`/`code.function.name`, so its step
judges silently skip it. Use the type that matches the code:

```python
@overmind.workflow()      # multi-step orchestration
def pipeline(): ...

@overmind.tool()          # a tool/function the agent can call
def search(query: str) -> list[dict]: ...

@overmind.retrieval()     # RAG / vector lookup
def fetch_docs(q: str): ...

@overmind.observe()       # any other traced function
def score(x): ...
```

All decorators accept `capture=` (`"auto"` scrubbed args/result, `"none"`,
`"messages"`), `ignore=` (argument names never captured), `capability=` /
`capability_id=` (prefer the id — it survives renames), and
`format_input=`/`format_output=` hooks. Captured payloads are scrubbed
automatically: secret-named keys redacted, base64/data-URL blobs replaced,
text kept in full.

Context manager and current-span helpers:

```python
with overmind.start_span("rerank", span_type=overmind.SpanType.FUNCTION) as span:
    overmind.set_tag("candidate_count", len(candidates))

overmind.set_user("user-123", email="a@b.com")
overmind.set_conversation_id("conv-abc")   # groups spans into one session
overmind.capture_exception(exc)            # marks the current span errored
```

`start_span` and the decorators use the ambient tracer, so they attach to
whichever provider is active — greenfield or fan-out.

## Step 7 — Verify

`overmind.run(...)` flushes on exit. Code paths without it (short scripts,
signal handlers) need an explicit flush before the process exits:

```python
overmind.force_flush_traces()
```

Run the app, then check traces appear in the
[Console](https://console.overmindlab.ai/). If nothing shows up, work through
the SDK's `docs/troubleshooting.md` checklist: no API key, wrong endpoint,
exit before batch export, provider not mounted, spans outside a run boundary,
or a pre-existing TracerProvider owning the pipeline. `init(debug=True)`
prints the resolved setup; `OVERMIND_STRICT_MODE=true` makes missing keys and
instrumentation packages raise instead of warn.
