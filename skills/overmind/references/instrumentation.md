# Instrumenting an application with Overmind tracing

Wire an existing Python project into Overmind so every LLM call and traced
function is exported. Overmind is built on OpenTelemetry, so it can either own
the tracing pipeline or ride alongside a telemetry stack the project already
has.

The SDK surface is the `overmind` Python package (`init`, `run`, `task`,
decorators, `force_flush_traces`). Signatures below are copied from the
installed SDK — prefer the code if they ever disagree.

Verify traces through Overmind MCP (`list_traces` / `get_trace`), not REST.
See [telemetry.md](telemetry.md).

## Tier model

Tiering is a scheduling device, not a quality ceiling: the fast path ships
Tier 0 + Tier 1 in one sitting; Tier 2 completes afterward through the
punch-list loop.

- **Tier 0 — auto-capture.** `overmind.init(providers=[...])` at process
  start. This alone gets LLM call spans (`gen_ai.request.model`,
  `gen_ai.usage.*`), provenance auto-stamping (`tool` spans →
  `environment`, `llm` spans → `agent`), `unit_kind` on run boundaries, code
  identity (`code.namespace` / `code.function.name`), and the git sha
  (`vcs.ref.head.revision`, auto-detected). `providers=[]` turns on **every**
  supported provider; `providers="auto"` turns on every provider whose
  library AND instrumentor are installed; omitting `providers` enables
  **none**.
- **Tier 1 — binding.** Declares the scoring skeleton: `overmind.run(...)`
  around the whole execution (the run boundary — without it a span that
  starts its own trace is suppressed as an orphan fragment),
  `overmind.task(key)` at each task's entry point, `capability_id` /
  `capability` at identity boundaries, and `name=` on
  `@overmind.tool` / `@overmind.workflow` / `@overmind.retrieval` to anchor
  discriminating symbols.
- **Tier 2 — evidence.** Makes a bound trace *scorable*:
  `overmind.deliver(payload, grounded_by=[...])`, `overmind.intent(text)`,
  `overmind.expect(kind, spec, ...)`, `overmind.eval_context(**facts)`,
  `overmind.checkpoint(name)`, `overmind.end_conversation()`, and explicit
  `provenance=` on spans whose payload class isn't already inferred
  (`tool` / `retrieval` → `environment`, `llm` → `agent`; everything else
  needs one of `user | agent | environment | harness`).

### Tier 0 + 1 API reference

```python
overmind.init(overmind_api_key=None, *, service_name=None, environment=None,
              providers=None, overmind_base_url=None, agent_id=None,
              agent_name=None, project_id=None, redact_keys=None,
              export_orphan_spans=False, debug=False) -> bool
overmind.run(name=None, *, capability=None, capability_id=None, intent=None,
             conversation_id=None, tags=None)
overmind.task(key, *, unit=None)
overmind.capability(name=None, *, id=None)
overmind.tool(name=None, **kwargs)       # also workflow / retrieval / entry_point
overmind.force_flush_traces(timeout_millis=1000)
```

- `init()` returns True when tracing is active. Without an API key it logs,
  no-ops every helper, and returns False — safe to call unconditionally. It
  must run **before** the traced entry executes and before the provider
  clients are constructed.
- `run()` is a context manager *and* a decorator, and is the one bracket every
  execution needs: capability identity, the entry-point run span
  (`unit="run"`), intent, conversation id, error status, and a flush on exit.
  As a decorator every parameter except `name` also accepts a callable
  receiving the wrapped call's arguments
  (`intent=lambda self, *a, **k: self.task`). The handle it yields exposes
  `deliver(payload, **kwargs)`.
- `task(key)` declares the `Behaviour.slug` this work belongs to — context
  manager or decorator. `key` is a plain string, so dynamic dispatch computes
  it first: `with overmind.task(route_for(request)):`. `unit="turn"`
  additionally makes the scope a scoring unit; re-entering the same key
  re-uses the still-open turn span, so a re-entrant phase lands in one unit.
  Internal fan-out, retries and loop bodies must **not** declare `unit`.
- Identity is `capability_id` — the capability UUID from the Console, the
  recommended identifier, resolved first server-side and stable through
  renames. `capability` (slug or display name) is safe to pass alongside but
  never load-bearing when the id is present. Three places accept it:
  `run(capability_id=...)`, `capability(name, id=...)` as a scope around the
  entry, and `init(agent_id=...)` / `OVERMIND_AGENT_ID` as the process-wide
  ambient default (`agent_id` is the older spelling of the same
  `overmind.agent.id` attribute). Entering a *different* capability mid-trace
  is a handoff: the first span of the new scope is stamped
  `unit_kind="turn"`. A capability must already be declared in the project —
  nothing is auto-created.
- `run()` with neither `capability` nor `capability_id` falls back to the
  `OVERMIND_AGENT_NAME` / `OVERMIND_AGENT_ID` env vars; passing either
  argument suppresses that fallback entirely.

### Tier 2 API reference

```python
overmind.intent(text, *, source="declared")
overmind.expect(kind, spec, *, id=None, scope="trace", gate=False)
overmind.eval_context(**facts)
overmind.checkpoint(name)
overmind.end_conversation()
overmind.deliver(payload, *, grounded_by=None, name="deliver", provenance="agent")
```

- `intent` declares what the user asked for and grounds judge scoring;
  undeclared, the server falls back to the first user message.
- `expect` — `kind` ∈ `contains | regex | schema | constraint | checkpoints`;
  `scope` ∈ `span | trace | conversation`; `gate=True` caps the run's score at
  0 on failure.
- `deliver` captures the terminal deliverable on its own child span
  (`overmind.delivery = true`). Call it **inside the unit that produced the
  deliverable**. `grounded_by` takes span_id hex strings or span handles;
  omitted, the SDK uses the environment-provenance spans it collected for the
  current trace.
- All of `intent` / `expect` / `eval_context` / `checkpoint` /
  `end_conversation` no-op (with a debug log) when no span is recording — call
  them inside a decorated span.

## Route first

Call `list_behaviours` for each capability before anything else.

- **Registry populated** (behaviours exist — the repo was deep-scanned): skip
  the scan and `plan_instrumentation`. `get_instrumentation_context` per
  capability is the placement source; its `placements` follow the same schema.
  Everything else below still applies: parallel fan-out per file, then
  `overmind instrumentation gate`. Smoke scaffolds are a `plan` side effect, so
  on this route write the smoke scripts yourself. Never run the real app on
  this route either.
- **Registry empty** (no behaviours), or neither tool exposed by the MCP
  server: run the full fast path below.

Constraints for both routes:

- Setup check: the Overmind MCP tools being visible IS the confirmation that
  setup is done. Do not probe the CLI (`overmind --version`, bare `overmind`)
  and do not read config outside the repository (`~/.config`, `/tmp`) — a
  sandboxed agent gets a permission error there, and that error does NOT mean
  the tooling is missing.
- A permission denial is never fatal: you do not need the denied file.
  Continue with the next step instead of stopping or apologising.
- Write every artifact (candidates.json, plan.json, smoke scripts,
  spans.jsonl) **inside the repository**. Sandboxed agents cannot write `/tmp`
  or read outside the project; an absolute path outside the repo fails the run.
- First pass is binding only: run brackets, task roots, capability identity,
  anchor names. Do not add `intent` / `expect` / `checkpoint` / `deliver` /
  `eval_context` until `verify_instrumentation_spans` reports every task
  bound. Tier 2 comes from the punch list, after the gate.
- Instrument every capability in one session — fan out across all of them at
  once, not one capability at a time.
- Trust the plan. A placement's `file`, `qualname` and `lineno` are
  scan-verified. Read the target function only — never re-read the module or
  re-explore the repo before editing.

Invoke the CLI as the installed binary (`overmind …` on PATH or
`.venv/bin/overmind …`), never through `uv run` / `poetry run`: those resync
the project environment first and can silently swap the installed SDK for
the version pinned in the repo's manifest.

## Fast-path workflow

Target under 10 minutes total. Record wall-clock per stage (scan / plan /
edit / validate) and report the table at the end.

1. **Scan + plan (one command).**

   ```bash
   overmind instrumentation plan --root . --out plan.json
   ```

   Runs the AST scan and posts it to the server's `plan_instrumentation` for
   you, writing `plan.json` and printing a summary. Never paste scan or plan
   JSON into an MCP tool call by hand. The standalone scan stays available for
   inspection:

   ```bash
   overmind instrumentation scan --root . --out candidates.json
   ```

   Pure-AST, no imports of user code, no network. Emits
   `{schema_version, repo_sha, frameworks_detected, files: [{path, symbols: [{qualname, kind, signature, docstring, decorators, lineno}]}], skipped}`.
   Tests, docs, examples and virtualenvs are skipped and counted in `skipped`.

   The planner labels behaviours, mints Behaviour/BehaviourVersion
   server-side pre-traffic across every capability, and returns
   `{placements: [...], plans: [{capability, capability_id, plan_id, placement_count}], ambiguous: [...], dropped: [{key, reason}], minted: {...}}`.
   Each placement carries `key`, `placement_mode` (`fixed | dynamic_key`),
   `allowed_keys`, `analyzed_sha`,
   `target: {file, qualname, module, import_line}`,
   `required_task_decorator`,
   `required_identity: {capability_id, capability_name, how}`, `capability`,
   `capability_id`, `plan_id`, `why`, `tier`, `smoke_hint`. Read `dropped` and
   report it to the user — do not discard it silently.

   The command also writes a `smoke_<key>.py` skeleton in the repo root for
   every placement with a `smoke_hint` and a `module`+`qualname` target, and
   wires it as that placement's `smoke_script`. An existing file of that name
   is never overwritten, so re-running `plan` keeps your filled-in scripts.
   The printed summary lists the scaffolds it created.

1. **Parallel subagent fan-out.** One wave, covering every placement from that
   single plan call regardless of which capability it belongs to. Spawn one
   subagent per placement **file** — placements for the same file always go to
   the same subagent, so two edits never race on one file. Dispatch every
   subagent in a single message; the edits are independent by construction.
   Prompt each subagent with the placement JSON verbatim, plus this fixed
   instruction block:

   > Apply exactly the `required_task_decorator` at `target.qualname` in
   > `target.file`. Add `target.import_line` if it isn't already imported.
   > Wire the placement's `required_identity`: the entry must run inside
   > `overmind.run(capability_id=...)` (or an `overmind.capability(id=...)`
   > scope), or the process must call
   > `overmind.init(providers=[], agent_id=<capability_id>)` before the entry
   > runs. Without identity on the spans the server cannot bind ANY task,
   > whatever the key says.
   > Keep local code style (quotes, import grouping, line length). Make no
   > other edits — no renames, no reformatting outside the touched lines, no
   > speculative anchors. Report the diff.

   The lead agent keeps the `ambiguous` list for itself: the dynamic-dispatch
   judgment calls a subagent can't make (which registered key a route maps
   to), written as a computed `with overmind.task(<key expression>)` boundary.

1. **Fill the smoke scaffolds.** Each generated `smoke_<key>.py` already
   imports the target module and calls the real decorated entry inside a
   try/except; only the `# TODO` args (and the constructor args for a method
   target) are missing. Fill those in. Do not rewrite the script, and do not
   author one from scratch unless `plan` skipped that placement (no `module`
   on the target).

1. **Gate (one command).**

   ```bash
   overmind instrumentation gate --plan-file plan.json [--root .] [--spans-file spans.jsonl] [--capability X]
   ```

   Runs check → smoke → verify in order, stopping at the first failing stage,
   and prints one summary:
   `{"check": {"ok", "failed"}, "smoke": {"ran", "failed"}, "verify": {"tasks", "unbound"}}`.
   Exit 0 is the pass signal; any unbound task fails the gate. Run the three
   stage commands below only to debug a failed gate.

1. **Static gate (debug).**

   ```bash
   overmind instrumentation check --plan-file plan.json [--root .] [--format text|json]
   ```

   Deterministic, no network. Validates the task boundary, its key, nesting
   (task boundaries may not nest), the import form, and each placement's
   `analyzed_sha` against the local git revision. `--format json` for
   scripting; exit 1 on any failure.

1. **Smoke run (debug).** Never run the real app; never hit real providers. Write a
   minimal script per task placement that calls the entry with synthetic args,
   using its `smoke_hint`. Two ways to execute:

   - Wire the script's path as `smoke_script` on the placement, then:

     ```bash
     overmind instrumentation smoke --plan-file plan.json --out spans.jsonl [--root .]
     ```

     This iterates the plan's top-level `placements` and runs each
     `smoke_script` as a subprocess with `OVERMIND_SMOKE=1` and
     `OVERMIND_TRACE_FILE=<out>` set; a placement with only `smoke_hint` (no
     script yet) is echoed as a `TODO` and skipped.

   - Or run the scripts directly with the same two env vars:

     ```bash
     OVERMIND_SMOKE=1 OVERMIND_TRACE_FILE=spans.jsonl python my_smoke_script.py
     ```

   `OVERMIND_SMOKE=1` patches the installed provider SDK clients (openai,
   anthropic, google.genai) one layer below the instrumentors with canned
   responses — instrumentor spans still fire, no network call is made.
   A smoke script must call the REAL decorated entry points — import the
   module and invoke the actual function with synthetic args. Never write
   stand-in functions or synthesize spans: fabricated spans carry no code
   identity and no git sha, so the anchor join has nothing to match and the
   unit parks `unbound` (`missing_sha`). A body that throws is fine — wrap
   the call in try/except; the span still exports and still binds.

   `OVERMIND_TRACE_FILE` makes `overmind.init()` write spans to that file (no
   API key required) instead of exporting over OTLP. These two env vars are
   the whole smoke contract — do not read the SDK source to verify how they
   interact; set them and run. `spans.jsonl` ends up as one JSON span per line.

1. **Verify (debug).**

   ```bash
   overmind instrumentation verify --spans-file spans.jsonl
   ```

   Posts the smoke spans to the server's `verify_instrumentation_spans` binder
   dry-run and prints the verdict; exit 0 means every task bound `declared`.
   Use the CLI — never paste a large span array inline into an MCP tool call
   (the MCP tool itself remains available for small span sets). One call
   covers every capability's spans together, and runs the real server binder
   as a dry run — zero ingestion. Returns
   `{tasks: [{behaviour_key, binding_source, binding_confidence, route_flags, unit_span_id, trace_id, capability, capability_id}], capabilities: [{capability, capability_id, grades, punch_list: [{grade, instruction}]}], errors: []}`.
   Acceptance gate: no task is `unbound`. Run-grain units bind
   `anchor_join` — the SDK never puts a behaviour key on a run boundary; the
   join is the entry/interior code identity (`@overmind.entry_point` and the
   observe-family decorators stamp it) matched against the contract anchors.
   Turn-grain units bind `declared` via `with overmind.task(key, unit="turn")`.
   Each task carries `spans_seen` — name, behaviour_key, unit_kind and
   qualname per member span, exactly as the binder read them. Diagnose an
   `unbound` verdict from that block; never dump span files locally.
   A task with
   `capability: null` means the spans carry no `overmind.agent.id` — fix the
   identity wiring (see the fan-out block); as a stopgap for a
   single-capability repo, re-run with `--capability <name-or-slug>` to force
   the fallback. Act on `capabilities[].punch_list` items that are fixable now
   (Tier 1 gaps — a missing task root, a wrong key); park Tier 2 items for the
   ratchet loop.

1. **Report the timing table** (scan / plan / edit / validate wall-clock)
   alongside the pass/fail state of the static gate and the verify call.

## Ratchet loop (after real traffic)

1. Run the app, then audit a real trace via `list_traces` → `get_trace` on
   that exact `trace_id` — never an unrelated newest trace. See
   [telemetry.md](telemetry.md).

1. `verify_instrumentation_spans` grades each capability on six axes and
   returns a punch list. Work it until all six are green:

   | Grade          | Gap it flags                                      | Fix                                                                    |
   | -------------- | ------------------------------------------------- | ---------------------------------------------------------------------- |
   | `task`         | No declared task root, or more than one           | `overmind.task("<key>")` on exactly the planned entry point            |
   | `units`        | No turn/run boundaries                            | `task(key, unit="turn")` per phase, or a `capability` handoff          |
   | `tool_ops`     | Tool/retrieval spans missing or untyped           | `@overmind.tool()` / `@overmind.retrieval()` on the real call sites    |
   | `provenance`   | Spans without an inferable or declared provenance | `provenance="user\|agent\|environment\|harness"` on the span           |
   | `observations` | No runtime evidence envelope                      | `intent()`, `expect()`, `eval_context()`, `checkpoint()`               |
   | `delivery`     | No terminal deliverable captured                  | `deliver(payload, grounded_by=[...])` inside the unit that produced it |

1. Re-verify after each fix; don't move on until the grade flips.

## Payload policy

Do not censor, summarize, truncate, or redact ordinary prompts, context, tool
data, or model outputs. Captured payloads are already scrubbed
(secret-named keys redacted, base64/data-URL blobs replaced, text kept in
full); mask a clearly identifiable credential field at the call site, or pass
`capture="none"` on a decorator when no payload is explicitly required.
Over-redaction destroys attribution evidence.

## Declared keys vs the structural failsafe

A declared key (`overmind.task("key")`) is the strongest binding evidence: a
known key on the unit span binds even when the git sha is missing. An unknown
key falls through to structural matching and is flagged
`declared_key_unknown`. A revision mismatch, unknown anchor, or missing
evidence is still a verification failure regardless of declaration.

Without a declared key the server structurally matches span identity against
the registry: scored matched/expected coverage-fraction, binds only when the
best beats the runner-up by ≥1.5×, and is file-path-joined (a bare `run` in
`entry.py` cannot suffix-collide with `app.b.run`). Ties and weak matches stay
`unbound_ambiguous`; a sole candidate still binds but with zero evidence —
flagged `bind_review` at confidence 0.0, never a silent overconfident bind.
`binding_source == "bound_structurally"` is useful failsafe evidence but is
not instrumentation success — the fast-path gate requires every unit bound (`anchor_join` or `declared`).

## Existing-telemetry detection

Before greenfield-initializing, grep for a telemetry stack the project already
owns:

```bash
rg -n "set_tracer_provider|TracerProvider|opentelemetry|traceloop|Traceloop|langsmith|OTEL_EXPORTER" --glob '!**/.venv/**'
```

- **No matches** → `overmind.init()` creates and installs the provider.
- **A `TracerProvider` is already set** → OpenTelemetry only honours the
  **first** `set_tracer_provider()` call, so `overmind.init()` on top of an
  existing provider silently attaches nothing. Add Overmind's exporter to the
  provider the project already owns instead — see
  [telemetry.md](telemetry.md) Step 3b for the fan-out snippet and its
  identity caveat.

## Install and env vars

```bash
uv add overmind        # or: pip install overmind
```

```bash
export OVERMIND_API_KEY=<your-api-key>   # never ask the user to paste it into chat
```

| Env var                   | Purpose                                                                     |
| ------------------------- | --------------------------------------------------------------------------- |
| `OVERMIND_SERVICE_NAME`   | Service name on the traces                                                  |
| `OVERMIND_AGENT_ID`       | Ambient capability UUID (`init(agent_id=)` default)                         |
| `OVERMIND_AGENT_NAME`     | Ambient capability name                                                     |
| `OVERMIND_ENVIRONMENT`    | e.g. `production` (default `development`)                                   |
| `OVERMIND_API_URL`        | Override the trace endpoint base URL                                        |
| `OVERMIND_TRACE_FILE`     | Write spans as JSONL to this file instead of OTLP (no API key needed)       |
| `OVERMIND_SMOKE`          | `1` patches provider SDK clients with canned responses; no network call     |
| `OVERMIND_SMOKE_RESPONSE` | Canned text returned by smoke-patched provider calls                        |
| `OVERMIND_STRICT_MODE`    | `true` makes a missing key or instrumentation package raise instead of warn |

## Common mistakes

| Mistake                                                  | Consequence                                                          | Fix                                                                       |
| -------------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| No run bracket                                           | Spans starting their own trace are dropped as orphans                | `overmind.run(...)` around the execution                                  |
| No flush in scripts/serverless                           | Traces silently never sent                                           | `force_flush_traces()` before exit (`run()` flushes on exit)              |
| Init after LLM clients are created                       | Provider calls not instrumented                                      | Call `init()` at process start, before client construction                |
| `overmind.init()` on top of an existing `TracerProvider` | OTel keeps the first provider; Overmind attaches nothing             | Fan out onto the existing provider                                        |
| Omitting `providers=` on `init()`                        | No provider auto-instrumentation at all                              | Pass `providers=[]` (all) or `providers="auto"` (all installed)           |
| A task boundary nested inside another task boundary      | Zero or multiple task roots on one trace fails verification          | One task root per trace; nested work is `workflow` / `tool` / `retrieval` |
| A computed task key outside the plan's `allowed_keys`    | The unit binds structurally at best                                  | Return one of the placement's `allowed_keys`                              |
| Identity only in `init()` in a multi-capability process  | Every span lands under the first capability                          | `run(capability_id=...)` per entry path                                   |
| No `conversation_id` in a chat app                       | Sessions view stays empty, `end_conversation()` has nothing to close | Pass the thread id per request                                            |
| Smoke run against real providers                         | Real API calls, real cost, non-deterministic spans                   | `OVERMIND_SMOKE=1` before `overmind.init()` runs                          |
