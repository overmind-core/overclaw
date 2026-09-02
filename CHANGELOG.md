# Changelog

Notable changes to the `overmind` package. The span-attribute wire contract is
pinned separately in [`docs/tracing-attributes.md`](docs/tracing-attributes.md);
entries here cover the SDK surface.

## Unreleased

### Added

- Single-owner span-stamping resolver: every unit-kind and behaviour-key
  decision is made in one on-start resolver, verified by an enumerated
  invariant suite (`tests/test_stamping_invariants.py`). One run boundary per
  trace; nested run declarations resolve to `turn`.
- `task(key, unit="turn")` turn units: one turn span per (trace, behaviour
  key), shared across re-entries so a phase's non-contiguous activity lands in
  one scoring unit; closes when the run-boundary span ends.
- `overmind.run(...)`: run-lifecycle bracket as context manager and decorator —
  capability identity, entry-point run span, intent, conversation id, tags,
  error status, flush on exit, and a handle that delivers the terminal payload.
- `overmind.integrations.langgraph.bind()`: declarative LangGraph node →
  behaviour-turn binding with per-node overrides, opt-outs, code-identity
  anchoring of function-backed nodes, and a `deliver=` node.
- `providers=["langchain"]`: LangChain + LangGraph span coverage through the
  OpenInference instrumentor (`overmind[langchain]` extra).
- `init(providers="auto")`: detect installed target libraries and enable every
  provider whose instrumentor is also present; resolved list logged at INFO.
- `init(debug=True)`: one-line setup summary (endpoint, identity, enabled
  instrumentors, export mode) plus DEBUG logging for the `overmind` logger.
- `py.typed`: the package ships type information; decorators preserve wrapped
  signatures.
- Docs: the pinned wire contract in [tracing-attributes](docs/tracing-attributes.md);
  integration guidance lives in the `overmind` skill's telemetry reference.

### Removed

- `observe_safe` — use `observe(capture="none")` (or `init(redact_keys=...)`).
- `function`, `conversation`, `get_tracer`, `set_agent_id`, `set_agent_name`,
  `set_project_id` are no longer exported; identity rides on
  `init(capability_id=...)` or a `capability(name, id=...)` scope.
- `start_child_span` is private; open child spans with `start_span`.
- Wire attributes renamed `overmind.agent.*` → `overmind.capability.*` with no
  aliases; `OVERMIND_AGENT_ID` / `OVERMIND_AGENT_NAME` are inert — use
  `OVERMIND_CAPABILITY_ID` / `OVERMIND_CAPABILITY_NAME`.

### Changed

- Packaging: the core `overmind` distribution is now tracing-only. The CLI
  moved to `overmind[cli]` (typer/rich/psutil), cost enrichment to
  `overmind[inference]` (litellm), and the HTTP-layer instrumentations to a
  published `overmind[tracing-full]` extra (previously an uninstallable uv
  dependency group). Missing extras degrade with an error or log line naming
  the extra to install.
- Orphan-span suppression: a `function` span that starts a new trace outside
  any run boundary (no parent, no unit declaration) is no longer exported —
  the platform quarantines such fragments as noise. A warning is logged once;
  `init(export_orphan_spans=True)` restores the old behaviour.
