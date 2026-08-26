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

## Tier model

Instrumentation ships in tiers. Tiering is a scheduling device, not a quality
ceiling: the fast pass (below) ships Tier 0 + Tier 1 in one sitting; Tier 2
completes afterward through the punch-list loop. Full granularity — all three
tiers — is the end state for every task boundary.

- **Tier 0 — auto-capture.** `overmind.init(providers=[...])` at process
  start. This alone gets LLM call spans (`gen_ai.request.model`,
  `gen_ai.usage.*`), provenance auto-stamping (`tool` spans → `environment`,
  `llm` spans → `agent`), `unit_kind` auto-stamped on entry points (`"run"`),
  code identity (`code.namespace` / `code.function.name`), and git sha
  (`vcs.ref.head.revision`, auto-detected). Always pass `providers=` explicitly:
  `providers=[]` (empty list) turns on **every** supported provider
  (`openai`, `anthropic`, `google`, `agno`); omitting `providers` enables
  **none** of them.
- **Tier 1 — binding.** Declares which spans are the scoring skeleton:
  `@overmind.task(key)` or `@overmind.task(key_from=selector)` at each task's
  entry point (exactly one task root per trace — a task cannot nest inside
  another task), `overmind.capability(name=..., id=...)` at identity
  boundaries (a `with` block or decorator; entering a *different* capability
  mid-trace is a handoff and stamps the first span of the new scope
  `unit_kind="turn"`), and `name=` on `@overmind.workflow` / `@overmind.tool`
  / `@overmind.retrieval` / `@overmind.function` to anchor discriminating
  symbols with a rename-proof `overmind.anchor.name` (otherwise the qualname
  is the only diagnostic identity).
- **Tier 2 — evidence.** Makes a bound trace *scorable*:
  `overmind.deliver(payload, grounded_by=[...])` on the terminal deliverable,
  `overmind.intent(text)`, `overmind.expect(kind, spec, ...)`,
  `overmind.eval_context(**facts)`, `overmind.checkpoint(name)`,
  `overmind.end_conversation()`, `overmind.mark_unit(kind)`, and explicit
  `provenance=` on spans whose payload class isn't already inferred
  (`tool` → `environment`, `retrieval` → `environment`, `llm` → `agent`;
  everything else needs `provenance=` to be one of
  `user | agent | environment | harness`).

### Tier 1 API reference

```python
@overmind.task("behaviour-key")  # exactly one task root per trace
def run(payload: dict) -> dict: ...

@overmind.task(key_from=lambda payload: payload["route"])  # shared-entry dynamic dispatch
def dispatch(payload: dict) -> dict: ...

with overmind.capability(name="billing-agent"):  # or id=<uuid>; also usable as a decorator
    run(payload)

@overmind.workflow(name="checkout-pipeline")  # name= anchors a discriminating symbol
def pipeline(): ...
```

- `overmind.task(key=None, *, key_from=None, name=None, agent_id=None, project_id=None, entrypoint=None, capture_io=True)`
  — exactly one of `key` / `key_from` is required. `key_from` is a callable
  run against the decorated function's arguments before span creation; it
  must return one registered, non-empty key from the plan's known key set —
  an invalid result or a raised exception produces no task span. Decorator
  form (sync/async); a context-manager form exists for a fixed-key dynamic
  boundary and takes `entrypoint=<callable>` to stamp code identity, but has
  no callable to capture I/O from. Opens a `SpanType.ENTRY_POINT` unit span
  and stamps `overmind.behaviour.key`.
- `overmind.capability(name=None, *, id=None)` — context manager (sync/async)
  and decorator. Requires at least one of `name` / `id`. Attaches the
  identity to every span created inside; restores the outer identity on
  exit. Must be an identity the project already declared — `capability()`
  does not create one.
- `overmind.mark_unit(kind)` — `kind` ∈ `turn | run`. Entry points stamp
  `run` automatically; call this to mark a turn boundary the SDK doesn't
  already wrap.

### Tier 2 API reference

```python
overmind.intent(text, *, source="declared")
overmind.expect(kind, spec, *, id=None, scope="trace", gate=False)
overmind.eval_context(**facts)
overmind.checkpoint(name)
overmind.end_conversation()
overmind.deliver(payload, *, grounded_by=None, name="deliver", provenance="agent")
```

- `intent(text, source="declared")` — declares what the user asked for;
  grounds judge scoring. Undeclared falls back server-side to the first user
  message.
- `expect(kind, spec, id=None, scope="trace", gate=False)` — `kind` ∈
  `contains | regex | schema | constraint | checkpoints`; `scope` ∈
  `span | trace | conversation`. `id` defaults to a stable hash of
  `kind:spec`. `gate=True` caps the run's score at 0 on failure.
- `eval_context(**facts)` — runtime facts for the judge; values coerced like
  `set_tag`.
- `checkpoint(name)` — named trajectory milestone / turn boundary.
- `end_conversation()` — idempotent per active task boundary; triggers
  conversation-scope scoring (needs a conversation id from
  `set_conversation_id` / `@overmind.conversation`).
- `deliver(payload, grounded_by=None, name="deliver", provenance="agent")` —
  captures the terminal deliverable on its own child span
  (`overmind.delivery = true`). `grounded_by` is a list of the evidence
  spans the deliverable rests on — span_id hex strings or span handles (e.g.
  what `start_span` yields).
- All five of `intent`/`expect`/`eval_context`/`checkpoint`/`end_conversation`
  no-op (with a debug log) when there is no recording span — call them
  inside a decorated span.
- Only the primary task boundary owns the envelope and conversation
  completion; nested spans must not emit a second envelope.

## Route first

Call `list_behaviours` for each capability before anything else.

- **Registry populated** (behaviours exist — the repo was deep-scanned):
  skip the scan and `plan_instrumentation`. `get_instrumentation_context`
  per capability is the placement source; its `placements` follow the same
  schema. Everything else below still applies: parallel fan-out per file,
  static gate, smoke run, `verify_instrumentation_spans`. Never run the
  real app on this route either.
- **Registry empty** (no behaviours): run the full fast path below —
  scan → plan → fan-out → gates.

Constraints for both routes:
- Setup check: the Overmind MCP tools being visible IS the confirmation that
  setup is done. Do not probe the CLI (`overmind --version`, bare `overmind`)
  and do not read config outside the repository (`~/.config`, `/tmp`) — a
  sandboxed agent gets a permission error there, and that error does NOT mean
  the tooling is missing.
- Write every artifact (candidates.json, plan.json, smoke scripts,
  spans.jsonl) **inside the repository**. Sandboxed coding agents cannot
  write `/tmp` or read outside the project; an absolute path outside the
  repo fails the run.
- First pass is binding only: task roots, capability scopes, anchor names.
  Do not add `intent`/`expect`/`checkpoint`/`deliver`/`eval_context` until
  `verify_instrumentation_spans` reports every task `declared`. Tier 2 comes
  from the punch list, after the gate.
- Instrument every capability in one session — fan out across all of them
  at once, not one capability at a time.

## Fast-path workflow

Target under 10 minutes total. Record wall-clock per stage (scan / plan /
edit / validate) and report the table at the end.

1. **Scan.**
   ```bash
   uv run overmind instrumentation scan --root . --out candidates.json
   ```
   Pure-AST, no imports of user code, no network. Emits
   `{schema_version, repo_sha, frameworks_detected, files: [{path, symbols: [{qualname, kind, signature, docstring, decorators, lineno}]}]}`.

1. **Plan.** One MCP `plan_instrumentation(candidates)` call for the whole
   repo — no capability argument. Labels behaviours, mints
   Behaviour/BehaviourVersion server-side pre-traffic across every
   capability, and returns
   `{placements: [...], plans: [{capability, capability_id, plan_id, placement_count}, ...], ambiguous: [...], dropped: [{key, reason}, ...], minted: {...}}`.
   Each placement carries `placement_id`, `key`, `placement_mode`
   (`fixed | dynamic_key`), `allowed_keys`, `analyzed_sha`,
   `target: {file, qualname, module, import_line}`,
   `required_task_decorator`, `constraints`, `capability`, `capability_id`,
   `plan_id`, `why`, `tier`, `smoke_hint`. Read `dropped` and report it to the
   user — do not discard it silently.

1. **Parallel subagent fan-out.** One wave, covering every placement from
   that single plan call regardless of which capability it belongs to. Spawn
   one subagent per placement **file** — placements for the same file always
   go to the same subagent, so two edits never race on one file. Prompt each
   subagent with the placement JSON verbatim, plus this fixed instruction
   block:

   > Apply exactly the `required_task_decorator` at `target.qualname` in
   > `target.file`. Add `target.import_line` if it isn't already imported.
   > Keep local code style (quotes, import grouping, line length). Make no
   > other edits — no renames, no reformatting outside the touched lines, no
   > speculative anchors. Report the diff.

   The lead agent keeps the `ambiguous` list for itself: write `key_from`
   selectors for shared-entry dispatch, and make the dynamic-dispatch
   judgment calls a subagent can't (which registered key a route maps to).

1. **Static gate.**
   ```bash
   uv run overmind instrumentation check --plan-file plan.json [--root .] [--format text|json]
   ```
   Deterministic, no network. `--format json` for scripting.

1. **Smoke run.** Never run the real app; never hit real providers. Write a
   minimal script per task placement that calls the entry with synthetic
   args, using its `smoke_hint`. Two ways to execute:
   - Wire the script's path as `smoke_script` on the placement, then:
     ```bash
     uv run overmind instrumentation smoke --plan-file plan.json --out spans.jsonl [--root .]
     ```
     This iterates the plan's top-level `placements`, and for each one with a
     `smoke_script` runs it as a subprocess with `OVERMIND_SMOKE=1` and
     `OVERMIND_TRACE_FILE=<out>` set; a placement with only `smoke_hint` (no
     script yet) is echoed as a `TODO` and skipped.
   - Or run the scripts directly with the same two env vars:
     ```bash
     OVERMIND_SMOKE=1 OVERMIND_TRACE_FILE=spans.jsonl python my_smoke_script.py
     ```
   `OVERMIND_SMOKE=1` patches installed provider SDK clients (openai,
   anthropic, google.genai) one layer below the instrumentors with canned
   responses — instrumentor spans still fire, no network call is made.
   `OVERMIND_TRACE_FILE` makes `overmind.init()` write spans to that file
   (no API key required) instead of exporting over OTLP. `spans.jsonl` ends
   up as one JSON span per line.

1. **Verify.** MCP `verify_instrumentation_spans(spans)` with the JSONL
   content (parsed into a list of span dicts) from the smoke run — one call
   for every capability's spans together. Runs the real server binder as a
   dry run — zero ingestion. Returns
   `{tasks: [{capability, capability_id, behaviour_key, binding_source, binding_confidence, route_flags, unit_span_id, trace_id}], capabilities: [{capability, capability_id, grades: {task, units, tool_ops, provenance, observations, delivery}, punch_list: [{grade, instruction}]}], errors: []}`.
   Acceptance gate: every task's `binding_source == "declared"`. Act on
   `capabilities[].punch_list` items that are fixable now (Tier 1 gaps — a
   missing task root, a wrong key); park Tier 2 items (evidence gaps) for the
   ratchet loop.

1. **Report the timing table** (scan / plan / edit / validate wall-clock)
   alongside the pass/fail state of the static gate and the verify call.

## Ratchet loop (after real traffic)

Once the fast path ships and real traffic lands, close the remaining Tier 2
gaps:

1. Run the app, then fetch a real trace: `verify_instrumentation_trace`
   (agent/plan_id/trace_id) for declared-binding attribution, and
   `list_task_executions` for the resulting rows. See
   [telemetry.md](telemetry.md).
1. Each task execution carries an EvidenceProfile with six grades — `task`,
   `units`, `tool_ops`, `provenance`, `observations`, `delivery` — the same
   six the fast-path `verify_instrumentation_spans` call returns. Work the
   punch list until all six are green:

   | Grade          | Gap it flags                                       | Fix                                                                                |
   | -------------- | --------------------------------------------------- | ----------------------------------------------------------------------------------- |
   | `task`         | No declared task root, or more than one              | `@overmind.task("<key>")` on exactly the planned entry point                        |
   | `units`        | No turn/run boundaries                               | `mark_unit("turn")` at handoffs the SDK doesn't wrap, or an `overmind.capability()` handoff |
   | `tool_ops`     | Tool/retrieval spans missing or untyped              | `@overmind.tool()` / `@overmind.retrieval()` on the real call sites                 |
   | `provenance`   | Spans without an inferable or declared provenance    | `provenance="user\|agent\|environment\|harness"` on the span                        |
   | `observations` | No runtime evidence envelope                         | `intent()`, `expect()`, `eval_context()`, `checkpoint()`                            |
   | `delivery`     | No terminal deliverable captured                     | `deliver(payload, grounded_by=[...])` at the point of return                        |

1. Re-verify with `verify_instrumentation_trace` / `list_task_executions`
   after each fix; don't move on until the grade flips.

## Payload policy

Do not censor, summarize, truncate, or redact ordinary prompts, context,
tool data, or model outputs. Mask only a clearly identifiable credential
field, or use `capture_io=False` on a decorator when no payload is
explicitly required. Over-redaction destroys attribution evidence.

## Declared keys vs the structural failsafe

A declared key (`@overmind.task("key")` or a validated `key_from` result) is
the strongest binding evidence: a known key on the unit span binds even when
the git sha is missing. An unknown key falls through to structural matching
and is flagged `declared_key_unknown`. A revision mismatch, unknown anchor,
or missing evidence is still a verification failure regardless of
declaration.

Without a declared key the server structurally matches span identity against
the registry: scored matched/expected coverage-fraction, binds only when the
best beats the runner-up by ≥1.5×, and is file-path-joined (a bare `run` in
`entry.py` cannot suffix-collide with `app.b.run`). Ties and weak matches
stay `unbound_ambiguous`; a sole candidate still binds but with zero evidence
— flagged `bind_review` at confidence 0.0, never a silent overconfident bind.
`binding_source == "bound_structurally"` is useful failsafe evidence, but is
not strict instrumentation success — the fast-path gate requires `"declared"`.

## Existing-telemetry detection

Before greenfield-initializing, grep for a telemetry stack the project
already owns:

```bash
rg -n "set_tracer_provider|TracerProvider|opentelemetry|traceloop|Traceloop|langsmith|OTEL_EXPORTER" --glob '!**/.venv/**'
```

- **No matches** → `overmind.init()` creates and installs the provider.
- **A `TracerProvider` is already set** (OTel directly, Traceloop/OpenLLMetry,
  LangSmith's OTel bridge, etc.) → OpenTelemetry only honours the **first**
  `set_tracer_provider()` call and ignores later ones with a warning, so
  calling `overmind.init()` on top of an existing provider silently attaches
  nothing. Instead add Overmind's exporter to the provider the project
  already owns:

  ```python
  from opentelemetry import trace
  from opentelemetry.sdk.trace import TracerProvider
  from opentelemetry.sdk.trace.export import BatchSpanProcessor
  from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

  import overmind
  from overmind.tracing import enable_tracing, get_api_settings

  api_key, base_url = get_api_settings()  # reads OVERMIND_API_KEY / OVERMIND_API_URL
  provider = trace.get_tracer_provider()
  provider.add_span_processor(
      BatchSpanProcessor(
          OTLPSpanExporter(endpoint=f"{base_url}/api/v1/traces", headers={"X-Api-Key": api_key})
      )
  )
  enable_tracing(["openai", "anthropic"])  # or [] for every supported provider
  ```

  The project's existing backend keeps receiving spans; Overmind gets a
  copy. Overmind's server reads canonical `genai.*` usage attributes; spans
  from third-party auto-instrumentors that only emit OTel `gen_ai.*` keys
  are bridged automatically only when Overmind owns the provider — on the
  fan-out path, prefer `enable_tracing()` or the Tier 1/2 decorators so
  token/cost rollups populate.

## Install and env vars

```bash
uv add overmind        # or: pip install overmind
```

```bash
export OVERMIND_API_KEY=<your-api-key>   # never ask the user to paste it into chat
```

| Env var                  | Purpose                                                                     |
| ------------------------- | ---------------------------------------------------------------------------- |
| `OVERMIND_SERVICE_NAME`   | Service name on the traces                                                   |
| `OVERMIND_AGENT_NAME`     | Human-readable agent name                                                    |
| `OVERMIND_AGENT_ID`       | Agent UUID (preferred over name once registered)                             |
| `OVERMIND_ENVIRONMENT`    | e.g. `production` (default `development`)                                    |
| `OVERMIND_API_URL`        | Override the trace endpoint base URL                                         |
| `OVERMIND_TRACE_FILE`     | Write spans to this file instead of exporting over OTLP (no API key needed)  |
| `OVERMIND_SMOKE`          | `1` patches provider SDK clients with canned responses; no network call      |
| `OVERMIND_SMOKE_RESPONSE` | Canned text returned by smoke-patched provider calls                         |
| `OVERMIND_STRICT_MODE`    | `true` makes a missing instrumentation package raise instead of warn         |

## Flush on shutdown

Batch export is async; flush before a short-lived process exits or spans are
lost:

```python
overmind.force_flush_traces()
```

## Common mistakes

| Mistake                                                              | Consequence                                                | Fix                                                                                                                                                |
| --------------------------------------------------------------------| ------------------------------------------------------------| --------------------------------------------------------------------------------------------------------------------------------------------------|
| No flush in scripts/serverless                                       | Traces silently never sent                                  | `force_flush_traces()` before exit                                                                                                                 |
| Init after LLM clients are created                                   | Provider calls not instrumented                              | Call `init()` at process start, before client construction                                                                                         |
| `overmind.init()` on top of an existing `TracerProvider`              | OTel keeps the first provider; Overmind attaches nothing     | Fan-out (add a span processor to the existing provider)                                                                                            |
| Omitting `providers=` on `init()`                                    | No provider auto-instrumentation at all                      | Pass `providers=[...]` explicitly, or `providers=[]` for every supported provider                                                                  |
| A task nested inside another task                                    | Zero or multiple task roots on one trace fails verification  | Exactly one `@overmind.task` per trace; nested work is `workflow`/`tool`/`retrieval`/`function`                                                    |
| `key_from` selector returns an unregistered or empty key             | No task span is created                                      | Return one of the plan's `allowed_keys`                                                                                                            |
| Credentials (API keys, tokens, passwords) in decorated function args | Stored verbatim in the trace                                  | Mask the clearly identifiable credential field, or `capture_io=False` when no payload is explicitly required; do not censor ordinary data merely because it might be sensitive |
| No `set_conversation_id` in a chat app                                | Sessions view stays empty, `end_conversation()` has nothing to close | Stamp the thread/conversation id per request                                                                                                       |
| Smoke run against real providers                                     | Real API calls, real cost, non-deterministic spans            | Set `OVERMIND_SMOKE=1` before `overmind.init()` runs                                                                                               |
