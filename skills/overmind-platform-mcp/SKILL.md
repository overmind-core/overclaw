______________________________________________________________________

## name: overmind-platform-mcp description: Use the Overmind platform MCP to inspect and act on a project from a coding agent — agents, traces, datasets, evals, finetunes, deployments. Use when the user wants to query or change Overmind via MCP tools, or when Overmind MCP tools are available.

# Overmind platform MCP

The Overmind Cursor plugin registers a remote MCP server at
`https://api.overmindlab.ai/api/mcp/` (Streamable HTTP). Authenticate with a
project API key (`X-Api-Key`). The key is pinned to one project and has full
read and write access to that project's tools.

## Setup

1. Create a project API key in Console → Projects → API keys.
1. Install this plugin (or add the MCP server manually).
1. Set **OVERMIND_API_KEY** in the plugin Configure panel (or your agent's MCP
   headers). Never paste the raw key into chat.

Local / self-hosted: point the MCP URL at `{API_BASE}/api/mcp/` instead of
production.

## How to use

Prefer MCP tools over inventing REST calls when the tools cover the job.

Typical flows:

| Goal                      | Start with                                                             |
| ------------------------- | ---------------------------------------------------------------------- |
| See what's in the project | `list_agents`, `list_datasets`, `list_eval_runs`, `list_finetune_jobs` |
| Diagnose failures         | `agent_failures`, `graph_search`, `list_traces` / `get_trace`          |
| Launch training           | `finetune_prerequisites` then `create_finetune_job`                    |
| Run evals                 | `create_eval_run` (after listing datasets / eval sets)                 |

Do not call chat-UI-only helpers — they are not exposed on MCP
(`propose_plan`, `suggest_navigation`).

## Related

- Telemetry / REST ingest and verification:
  [overmind-agent-telemetry](../overmind-agent-telemetry/SKILL.md)
- Docs index: https://docs.overmindlab.ai/llms.txt
