# Overmind tracing — canonical attribute contract

This is the **source-of-truth contract** for every span attribute the Overmind
Python SDK emits. The server ingest (`overbae/api/otlp.py`), the JS SDK, and the
public API docs are reconciled against this file.

Rules of the road:

- Keys we own live under `overmind.*` (dotted segments) or `genai.*` / `tool.*`.
- Every key is defined once in [`overmind/attrs.py`](../overmind/attrs.py) — never
  inline a raw string.
- Token usage and cost use the **`genai.*`** keys (NOT the OTel semconv
  `gen_ai.*`). The SDK *also* emits the `gen_ai.*` semconv keys alongside them so
  OTel-native consumers and the optimiser's `trace_reader` keep working, and the
  on-end enrichment processor mirrors any `gen_ai.*` usage produced by
  third-party auto-instrumentors into these canonical keys.
- **Never zero-fill.** A token count / cost we don't have is omitted, so the
  server never records a misleading `0`.

Legend for **When present**: `always` = on every span of that kind; `if known` =
only when the value is available; `derived` = computed if the primary source is
absent.

---

## 1. Resource attributes (set once per process in `init()`)

| Key | Type | When present | Meaning |
|-----|------|-------------|---------|
| `service.name` | string | always | Service name (`service_name=` / `OVERMIND_SERVICE_NAME`). Server ignores for agent resolution. |
| `service.version` | string | always | `SERVICE_VERSION` env, else SDK version. |
| `deployment.environment` | string | always | `environment=` / `OVERMIND_ENVIRONMENT` (default `development`). |
| `overmind.sdk.name` | string | always | `"overmind-python"`. |
| `overmind.sdk.version` | string | always | SDK release (`overmind.__version__`). |
| `overmind.agent.id` | string (UUID) | if `init(agent_id=)` / `OVERMIND_AGENT_ID` | Agent PK. Server resolves this first (direct lookup). |
| `overmind.agent.name` | string | if `init(agent_name=)` / `OVERMIND_AGENT_NAME` | Agent display name; server slugifies for stable identity. |
| `overmind.project.id` | string (UUID) | if `init(project_id=)` / `OVERMIND_PROJECT_ID` | Project PK (session auth only; API tokens pin the project). |

Identity is *also* seeded into the OTel context, so the on-start processor stamps
the same `overmind.agent.id` / `overmind.agent.name` / `overmind.project.id` onto
**every span** (including spans from third-party auto-instrumentors).

---

## 2. Common attributes (every SDK span)

Emitted by `@observe` / `@entry_point` / `@workflow` / `@tool` / `@function` /
`@retrieval` and by `start_span` / `start_child_span`.

| Key | Type | When present | Meaning |
|-----|------|-------------|---------|
| `overmind.span.type` | string | always | One of `function`, `entry_point`, `workflow`, `tool_call`, `llm_call`, `retrieval`. |
| `overmind.status` | string | always | `success` \| `failed` \| `cancelled`. |
| `overmind.duration.seconds` | float | always | Wall-clock duration of the wrapped span. |
| `overmind.error.type` | string | on failure | Exception class name. |
| `overmind.error.message` | string | on failure | Scrubbed message (≤1024 chars). |
| `overmind.agent.id` / `overmind.agent.name` / `overmind.project.id` | string | if identity seeded | Copied from the OTel context (see §1). |
| `overmind.workflow.name` | string | if `set_workflow_name()` | Traceloop workflow label. |
| `conversation.id` | string | if `set_conversation_id()` | Groups traces into a session. |
| `inputs` | JSON string | `@observe`/`@tool`/`@retrieval` (not `observe_safe`) | Serialised positional + keyword args. |
| `outputs` | JSON string | `@observe`/`@tool`/`@retrieval` (not `observe_safe`) | Serialised return value. |

> Exception spans also call `record_exception()` and set the OTel status to
> `ERROR`.

---

## 3. LLM / generation span (`overmind.span.type = "llm_call"`)

Emitted by `overmind.utils.llm.llm_completion` and by the on-end enrichment
processor for auto-instrumentor spans. **These are the keys the server rolls up.**

| Key | Type | When present | Meaning |
|-----|------|-------------|---------|
| `genai.model` | string | if known | Requested model id. |
| `genai.response.model` | string | if known | Model the provider actually served. |
| `genai.provider` | string | if known | Provider (`openai`, `anthropic`, …). |
| `genai.prompt_tokens` | int | if known | Prompt/input tokens. **Rolled up.** |
| `genai.completion_tokens` | int | if known | Completion/output tokens. **Rolled up.** |
| `genai.total_tokens` | int | if known / derived | Total tokens (derived = prompt + completion). **Rolled up.** |
| `genai.cost` | float (USD) | if known / derived | Provider-reported cost, else computed from model pricing. **Rolled up.** |
| `genai.cache_read_tokens` | int | if known | Cache-read (prompt-cache) tokens. *(new — server does not roll up yet)* |
| `genai.elapsed_seconds` | float | always | Client-measured latency. |
| `genai.error` | string | on failure | Exception class name. |
| `genai.request.message_count` | int | always | Number of messages sent. |
| `genai.request.message_chars` | int | always | Total chars across message content. |
| `genai.request.tool_count` | int | always | Number of tool schemas provided. |
| `genai.request.kwargs` | string | if any | Comma-joined kwarg names (excl. `api_key`). |
| `genai.request.temperature` | float | if passed | Sampling temperature. *(new)* |
| `genai.request.max_tokens` | int | if passed | Max output tokens. *(new)* |
| `genai.request.top_p` | float | if passed | Nucleus-sampling top-p. *(new)* |
| `genai.response.message_chars` | int | if known | Chars in the response message. *(new)* |
| `genai.response.finish_reason` | string | if known | e.g. `stop`, `tool_calls`, `length`. *(new)* |
| `genai.streaming` | bool | streaming only | `True` when `stream=True`. *(new)* |
| `genai.time_to_first_token_seconds` | float | streaming only | Time to first streamed chunk (TTFT). *(new)* |

The server also accepts the `genai.usage.prompt_tokens` /
`genai.usage.completion_tokens` / `genai.usage.total_tokens` aliases for the
three token counts.

**OTel semconv mirror (also emitted, for OTel-native consumers — do NOT read
server-side):** `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.system`,
`gen_ai.usage.prompt_tokens`, `gen_ai.usage.completion_tokens`,
`gen_ai.usage.total_tokens`. Third-party instrumentors additionally emit
`gen_ai.usage.input_tokens` / `output_tokens` and `llm.usage.total_tokens`; the
enrichment processor recognises all of these when mirroring to `genai.*`.

---

## 4. Tool-call span (`overmind.span.type = "tool_call"`)

Emitted by the `@tool` decorator (plus all common attributes from §2).

| Key | Type | When present | Meaning |
|-----|------|-------------|---------|
| `tool.name` | string | always | Tool / function name (the span name). |
| `tool.arg_keys` | string[] | if any args | Argument names passed (keys only, not values). |
| `tool.error` | string | on failure | Exception class name. |

---

## 5. Retrieval / RAG span (`overmind.span.type = "retrieval"`)

Emitted by the `@retrieval` decorator (plus common attributes from §2). The
step-specific keys are set by the caller via `set_tag`.

| Key | Type | When present | Meaning |
|-----|------|-------------|---------|
| `overmind.retrieval.query_chars` | int | if tagged | Length of the retrieval query. *(new)* |
| `overmind.retrieval.result_count` | int | if tagged | Number of documents/chunks returned. *(new)* |

---

## 6. Eval envelope span events (`overmind.eval.*`) — wire contract v1

Runtime eval declarations, emitted by `overmind.expect()` / `eval_context()` /
`checkpoint()` / `end_conversation()` (`overmind/evals.py`) as **span events**
on the current span (standard OTel `add_event`; no-ops without a recording
span). The platform's evaluation layer parses `Span.events` against exactly
these names and payload shapes — **pinned, do not rename**.

Every envelope event carries the same two attributes:

| Attribute | Type | Meaning |
|-----------|------|---------|
| `overmind.eval.schema_version` | int | Envelope schema version. Currently `1`. |
| `overmind.eval.payload` | JSON string | Event payload; shape depends on the event name (below). |

Payload shapes (v1):

| Event name | Payload | Emitted by |
|------------|---------|------------|
| `overmind.eval.expectation` | `{"id": str, "kind": "contains"\|"regex"\|"schema"\|"constraint"\|"checkpoints", "spec": str or object, "scope": "span"\|"trace"\|"conversation", "gate": bool}` | `expect(kind, spec, *, id=None, scope="trace", gate=False)`. For `checkpoints`, `spec` is the ordered list of checkpoint names the run is expected to reach. `id` is auto-derived as a short stable hash of kind+spec when omitted. Bad `kind`/`scope` raise `ValueError`. |
| `overmind.eval.context` | `{"facts": {str: JSON scalar or small object}}` | `eval_context(**facts)`. Values are coerced the same way `set_tag` coerces attribute values (rich values become JSON strings). |
| `overmind.eval.checkpoint` | `{"name": str}` | `checkpoint(name)` — named trajectory milestone / turn boundary. |
| `overmind.eval.conversation_end` | `{}` | `end_conversation()` — triggers conversation-scope scoring. |

---

## 7. Server-ingest reconciliation (Phase 2 checklist)

The server **already reads** these keys today (`_build_span_usage` /
`_resolve_agent` / `_classify_span_type`):

- `genai.prompt_tokens`, `genai.completion_tokens`, `genai.total_tokens`
  (+ `genai.usage.*` aliases), `genai.cost`, `genai.model`
- `overmind.agent.id`, `overmind.agent.name`, `overmind.project.id`
- `overmind.span.type` (verbatim), `tool.name`

Keys that are **NEW** in this contract and the server does **not** roll up yet —
call out for Phase 2 server work if you want them surfaced:

- `genai.cache_read_tokens`
- `genai.request.temperature` / `genai.request.max_tokens` / `genai.request.top_p`
- `genai.response.message_chars` / `genai.response.finish_reason`
- `genai.streaming` / `genai.time_to_first_token_seconds`
- `overmind.retrieval.query_chars` / `overmind.retrieval.result_count`
- the `overmind.eval.*` span events (§6) — envelope parsing is the platform's
  Phase 1 ingest work

These are all additive and namespaced; ingesting spans that carry them is safe
today (unknown attributes are stored, just not aggregated).
