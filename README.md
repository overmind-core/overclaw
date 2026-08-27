<img width="3000" height="1000" alt="X Company Banner Black" src="https://github.com/user-attachments/assets/4a5caceb-49e8-4b8e-a6aa-511222a94381" />

# Overmind

Overmind is two things in one package:

- **Tracing SDK** — drop-in observability for LLM agents. Decorate your code, get structured traces of every LLM call and tool invocation.
- **Optimiser client** — a thin, agent- and codebase-agnostic executioner. The optimisation loop (experiments → iterations → candidates → commands) is configured and driven **server-side** by the [Overmind Console](https://console.overmindlab.ai/). This repo just registers your machine, leases queued commands, runs them against your repo, and reports results back.

**Documentation:** [Overmind guide](https://docs.overmindlab.ai/guides/overmind_optimizer/)

**Console:** [console.overmindlab.ai](https://console.overmindlab.ai/)

## Install

The core package is the tracing SDK only:

```bash
pip install overmind          # tracing SDK (OTel core, thin)
```

Extras add the rest:

| Extra | Adds |
| --- | --- |
| `overmind[cli]` | The `overmind` command (optimiser loop, skills) — typer/rich/psutil |
| `overmind[inference]` | `genai.cost` enrichment via litellm's pricing tables |
| `overmind[langchain]` | LangChain + LangGraph auto-instrumentation |
| `overmind[tracing-full]` | HTTP-layer spans (requests / httpx / logging instrumentation) |

For the CLI as a standalone tool:

```bash
uv tool install "overmind[cli]"
# or
pipx install "overmind[cli]"
```

## Tracing

Wire up tracing once at process start, then annotate the functions you want traced:

```python
import overmind

# Reads OVERMIND_API_KEY from the environment. Without a key this logs once
# per process, returns False, and every decorator below becomes a no-op —
# safe to ship.
overmind.init(
    service_name="my-agent",
    agent_name="Support Triage",           # the capability declared in the Console
    providers="auto",                      # instrument every installed provider SDK
)                                          # (or name them: providers=["openai", "anthropic"])

@overmind.entry_point()                    # run root (overmind.unit_kind = "run")
def run(request: dict) -> dict:
    overmind.intent(request["question"])   # what the user asked for
    answer = think(request)
    overmind.deliver(answer)               # terminal deliverable, auto-grounded
    return answer

@overmind.tool(ignore=("session",))        # tool evidence; session never captured
def search(query: str, session) -> list[dict]:
    ...

@overmind.observe(type="llm", capture="messages")  # full chat evidence
def call_model(messages: list[dict]) -> dict:
    ...
```

That is the whole integration: no init guards (everything no-ops without a key), no hand-rolled scrubbing (captured payloads redact secret-named keys and base64 blobs automatically, text is kept in full), and no evidence bookkeeping (`deliver()` grounds itself in the environment-provenance spans of the run — pass `grounded_by=[...]` to override). On `KeyboardInterrupt`/cancellation the entry-point span flushes before re-raising, so interrupted runs still land.

Decorators: `entry_point`, `workflow`, `tool`, `retrieval`, and the general `observe` (sync and async). All accept `capture=` (`"auto"` scrubbed args/result, `"none"`, `"messages"`), `ignore=` (argument names never captured), `format_input=` / `format_output=` hooks for custom payload shapes, `provenance=`, `unit=`, and `capability=`. `start_span(...)` is the context-manager companion; `set_tag`, `set_user`, `set_conversation_id`, and `capture_exception` annotate the current span Sentry-style.

The span name may be a callable receiving the call's arguments — for polymorphic dispatchers, where one function executes named actions and each invocation must emit its own tool span (`tool.name` follows the resolved name):

```python
class Tools:
    @overmind.tool(name=lambda self, action, **params: action.name)
    def act(self, action, **params):   # executes navigate / extract / done / ...
        ...
```

Spans declare evidence provenance for the platform's evaluation judges: tool and retrieval spans are tagged `overmind.provenance = "environment"` and LLM spans `"agent"` automatically; pass `provenance=` (`user` / `agent` / `environment` / `harness`) to override. `@entry_point` spans are run roots (`overmind.unit_kind = "run"`) — one per trace: a run declared inside an open trace resolves to `turn`. `unit="turn"` marks an independently scorable decision cycle — each turn becomes one scored task execution. Internal fan-out or iteration spans (parallel sub-queries, retries, loop bodies) must not declare `unit`; handoffs stamp their own `turn` automatically. A `function` span that starts a trace outside any run boundary is an orphan fragment and is not exported by default (`init(export_orphan_spans=True)` overrides).

[`docs/carving-runs-into-units.md`](docs/carving-runs-into-units.md) is the integrator's guide to all of this — run vs. turn, deliver placement, handoffs, and the anchor-decoration rule, with a worked LangGraph example. The wire-level attribute contract is **pinned** in [`docs/tracing-attributes.md`](docs/tracing-attributes.md); nothing there is renamed. When traces don't show up, work through [`docs/troubleshooting.md`](docs/troubleshooting.md) — `init(debug=True)` prints the endpoint, identity, enabled instrumentors, and export mode.

Multi-capability agents scope identity with `overmind.capability` — a context manager or decorator that stamps `overmind.agent.name` / `.id` on every span created inside and restores the outer identity on exit (`capability="..."` on any decorator is shorthand for the name-only scope):

```python
with overmind.capability("DOM Element Locator", id="..."):  # id optional
    locate(prompt)  # every span here belongs to the locator capability
```

Entering a different capability mid-trace is a handoff: the first span of the new scope is stamped `overmind.unit_kind = "turn"`, so the platform scores it as a new unit against that capability's evals. Only declared identities are stamped — nothing is auto-created. `overmind.task("behaviour-slug")` optionally pins spans to a declared Behaviour the same way.

Single-capability agents with multiple phases (graph nodes, debate rounds) carve a run into units with `task(..., unit="turn")` — the scope opens one turn span per behaviour per trace, re-entering the same key re-uses it even when a phase's activity is non-contiguous, and the span closes when the run ends:

```python
with overmind.task("investment-debate", unit="turn"):
    ...  # spans here nest under the behaviour's turn span
```

`overmind.run(...)` brackets a whole agent run in one scope — capability identity (args, else `OVERMIND_AGENT_NAME` / `OVERMIND_AGENT_ID`), the entry-point run span, intent, conversation id, tags, error status, and a flush on exit. The yielded handle delivers the terminal payload; call it inside the unit that produced it:

```python
with overmind.run("trading-run", intent=f"Analyze {ticker}", conversation_id=f"{ticker}:{date}") as run:
    final_state = app.invoke(state)
    with overmind.task("portfolio-manager", unit="turn"):
        run.deliver(final_state["final_trade_decision"])
```

It is also a decorator (sync or async) for method entry points. Every parameter except `name` accepts a callable receiving the wrapped call's arguments, resolved per invocation, and the run-boundary span carries the function's `code.namespace` / `code.function.name` — one decoration covers both the run bracket and a scan-contract anchor. The return value is not auto-delivered; call `overmind.deliver()` inside the unit that produced it:

```python
class Agent:
    @overmind.run(intent=lambda self, *a, **k: self.task,
                  conversation_id=lambda self, *a, **k: self.task_id)
    async def run(self):
        ...
```

### LangChain / LangGraph

`pip install 'overmind[langchain]'`, then `providers=["langchain"]` mounts the OpenInference LangChain instrumentor (covers LangGraph): every chain, LLM and tool invocation gets a span with usable model/token/cost evidence. For the scoring semantics no instrumentor can know, `overmind.integrations.langgraph.bind` maps graph nodes to behaviour turn units — call it on the `StateGraph` after the `add_node` calls, before `compile()`:

```python
from overmind.integrations import langgraph as overmind_langgraph

overmind.init(providers=["openai", "langchain"], agent_name="Multi-Agent Trading Analysis")

workflow = build_state_graph()
overmind_langgraph.bind(
    workflow,
    # Default key per node: slugified node name ("Market Analyst" → "market-analyst").
    # Override where the scanned task map groups nodes differently; None opts a node out.
    behaviours={"Bull Researcher": "investment-debate", "Bear Researcher": "investment-debate", "Msg Clear Market": None},
    deliver="Portfolio Manager",  # optional: this node's completion delivers its return value
)
app = workflow.compile()
```

Each node invocation runs inside `task(key, unit="turn")` (re-entrant phases share one unit) and function-backed nodes carry their `code.namespace` / `code.function.name` identity for contract anchoring.

## Optimise

Set up and configure the experiment (agent, policy, dataset, iterations) in the [Console](https://console.overmindlab.ai/). Then, from the root of the repo you want optimised:

```bash
export OVERMIND_API_KEY=<your-api-key>
overmind optimise
```

This registers the current machine with the backend and loops forever: it leases queued commands from the experiment you configured in the Console, checks out the iteration's git branch (applying its candidate diff as a commit), runs the shell command against your repo, and reports the result back. Stop it any time with Ctrl-C; re-running is safe and idempotent per iteration branch.

### Options / environment variables

| Flag                    | Env var                        | Default                          | Description                                        |
| ----------------------- | ------------------------------- | --------------------------------- | --------------------------------------------------- |
| `--api-key`              | `OVERMIND_API_KEY`               | *(required)*                      | Sent as `X-Api-Key`.                                 |
| `--api-url`              | `OVERMIND_API_URL`               | `https://api.overmindlab.ai`      | Backend base URL.                                    |
| `--cwd`                  | `OVERMIND_CWD`                   | current directory                 | Repo root to run commands in.                        |
| `--poll-interval`        | `OPTIMIZER_POLL_INTERVAL`        | `5`                                | Idle poll seconds.                                   |
| `--heartbeat-interval`   | `OPTIMIZER_HEARTBEAT_INTERVAL`   | `60`                               | Idle "still alive" log interval, seconds.            |
| `--log-level`            | `OPTIMIZER_LOG_LEVEL`            | `INFO`                             | `DEBUG`/`INFO`/`WARNING`/`ERROR`.                    |

> [!WARNING]
> The Console can hand this client arbitrary shell to run (`shell=True`, guarded only by a per-command timeout). Only point it at a backend you trust.

## Skills

Use these from Cursor, Codex, or Claude Code to scaffold agents and configure telemetry without leaving your coding environment.

```bash
overmind skills list --verbose
overmind skills sync <skill-name>
```

| Skill                        | What it does                                                                     |
| ----------------------------- | --------------------------------------------------------------------------------- |
| `Overmind Register Agent`     | Create or register an Overmind agent entrypoint and bootstrap provider config.     |
| `Overmind Generate Agent`     | Build an agent from scratch using natural language.                               |
| `Overmind Telemetry`          | Configure Overmind tracing for your AI project.                                   |
| `Ponytail`                    | Review an optimisation report and adjust the policy, eval spec, or dataset.       |

## CLI reference

```text
overmind optimise [OPTIONS]         Register this machine and run the optimisation loop
overmind skills list [--verbose]    List installed/available skills
overmind skills sync <name>...      Sync one or more skills to the latest version
```

Run `overmind <command> --help` for full flag documentation.
