<img width="3000" height="1000" alt="X Company Banner Black" src="https://github.com/user-attachments/assets/4a5caceb-49e8-4b8e-a6aa-511222a94381" />

# Overmind

Overmind is two things in one package:

- **Tracing SDK** — drop-in observability for LLM agents. Decorate your code, get structured traces of every LLM call and tool invocation.

**Documentation:** [Overmind guide](https://docs.overmindlab.ai/core/observability)

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

overmind.init(
    service_name="my-agent", providers=["openai", "anthropic"]
)  # reads OVERMIND_API_KEY from the environment


@overmind.entry_point()
def run(input_data: dict) -> dict:
    return {"response": handle(input_data)}


@overmind.tool()
def search(query: str) -> list[dict]: ...
```

Available decorators/helpers: `entry_point`, `workflow`, `tool`, `function`, plus `start_span` (context manager), `set_tag`, `set_user`, and `capture_exception` for Sentry-style annotations on the current span.

## Skills

Use these from Cursor, Codex, or Claude Code to scaffold agents and operate
Overmind without leaving your coding environment. Skills live at the repo-root
[`skills/`](./skills/) directory so agent installers can pick them up from this
repository (e.g. `npx skills add overmind-core/overmind`).

```bash
overmind skills list --verbose
overmind skills sync overmind
overmind init --ide codex
```

Codex setup writes `.codex/config.toml`, installs the skill under
`.agents/skills`, and requires `OVERMIND_API_KEY` in the environment. Codex
loads project configuration only for trusted repositories.

| Skill      | What it does                                                                                        |
| ---------- | --------------------------------------------------------------------------------------------------- |
| `Overmind` | Instrument tracing, inspect telemetry via MCP, upload datasets, run evals, fine-tune, and optimize. |

## CLI reference

```text
overmind optimise [OPTIONS]         Register this machine and run the optimisation loop
overmind init [OPTIONS]             Configure Overmind for Cursor, Claude Code, OpenCode or Codex
overmind skills list [--verbose]    List installed/available skills
overmind skills sync <name>...      Sync one or more skills to the latest version
```

Run `overmind <command> --help` for full flag documentation.
