<img width="3000" height="1000" alt="X Company Banner Black" src="https://github.com/user-attachments/assets/4a5caceb-49e8-4b8e-a6aa-511222a94381" />

# Overmind

Overmind is two things in one package:

- **Tracing SDK** — drop-in observability for LLM agents. Decorate your code, get structured traces of every LLM call and tool invocation.
- **Optimiser client** — a thin, agent- and codebase-agnostic executioner. The optimisation loop (experiments → iterations → candidates → commands) is configured and driven **server-side** by the [Overmind Console](https://console.overmindlab.ai/). This repo just registers your machine, leases queued commands, runs them against your repo, and reports results back.

**Documentation:** [Overmind guide](https://docs.overmindlab.ai/guides/overmind_optimizer/)

**Console:** [console.overmindlab.ai](https://console.overmindlab.ai/)

## Install

```bash
uv tool install overmind
# or
pipx install overmind
```

## Tracing

Wire up tracing once at process start, then annotate the functions you want traced:

```python
import overmind

overmind.init(service_name="my-agent", providers=["openai", "anthropic"])  # reads OVERMIND_API_KEY from the environment

@overmind.entry_point()
def run(input_data: dict) -> dict:
    return {"response": handle(input_data)}

@overmind.tool()
def search(query: str) -> list[dict]:
    ...
```

Available decorators/helpers: `entry_point`, `workflow`, `tool`, `function`, plus `start_span` (context manager), `set_tag`, `set_user`, and `capture_exception` for Sentry-style annotations on the current span.

Spans declare evidence provenance for the platform's evaluation judges: tool and retrieval spans are tagged `overmind.provenance = "environment"` and LLM spans `"agent"` automatically; pass `provenance=` (`user` / `agent` / `environment` / `harness`) on any decorator or `start_span` to override. `@entry_point` spans are marked as run roots (`overmind.unit_kind = "run"`); `overmind.mark_unit("turn")` marks the span beginning a user-visible turn. Mark the final answer with `overmind.deliver(answer, grounded_by=[...])` — it captures the payload on a span with `overmind.delivery = true`, grounded in the named evidence spans. See [`docs/tracing-attributes.md`](docs/tracing-attributes.md) for the full attribute contract.

Multi-capability agents scope identity with `overmind.capability` — a context manager or decorator that stamps `overmind.agent.name` / `.id` on every span created inside and restores the outer identity on exit:

```python
with overmind.capability("DOM Element Locator", id="..."):  # id optional
    locate(prompt)  # every span here belongs to the locator capability
```

Entering a different capability mid-trace is a handoff: the first span of the new scope is stamped `overmind.unit_kind = "turn"`, so the platform scores it as a new unit against that capability's evals. Only declared identities are stamped — nothing is auto-created.

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
