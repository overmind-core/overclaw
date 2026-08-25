---
name: overmind-platform
description: Operate the Overmind platform via the overmind platform CLI — discover tools with list/describe, execute with call, and poll long jobs with job_status. Use when Overmind MCP tools are unavailable or you want a compact tool-discovery workflow without loading 145 MCP schemas into context.
---

# Overmind platform CLI

The `overmind platform` commands proxy the same server-side tool registry as the
Overmind MCP server. Use them from a shell when MCP is not configured, for
debugging, or when you want agents to discover tools on demand instead of
loading every schema into context.

**Auth:** `OVERMIND_API_KEY` (required) and optional `OVERMIND_API_URL` (defaults
to `https://api.overmindlab.ai`). The API key pins project scope — no
`--project` flag. Run commands from the project root (or set `OVERMIND_CWD`)
so the CLI loads `.env` automatically; shell exports always win over `.env`.
Optional `OVERMIND_ENV_FILE` points at a non-default env file.

## Workflow: list → describe → call

1. **Catalog** — compact names and one-line descriptions (no full schemas):

   ```bash
   overmind platform list
   overmind platform list --domain evals
   overmind platform list --json
   ```

   Domains are inferred from tool name prefixes: `evals`, `workshop`,
   `finetune`, `builds`, `capabilities`, `observability`, `optimizer`,
   `connectors`, `inference`, `graph`, or `other`.

2. **Schema** — read arguments before calling:

   ```bash
   overmind platform describe create_eval_run
   overmind platform describe create_eval_run --json
   ```

3. **Execute** — mutations run immediately (same as HTTP MCP today; no chat
   confirm card). Verify destructive args before calling.

   ```bash
   overmind platform call list_capabilities --args '{}'
   overmind platform call create_eval_run --args '{"name":"…","dataset_name":"…","eval_set_name":"…"}'
   overmind platform call create_connector --args-file connector.json --json
   ```

   Prefer `--args-file` for connector credentials instead of pasting secrets
   into the shell history.

## Long-running jobs

There is no `wait_for_job` in the CLI. Poll explicitly:

```bash
overmind platform call job_status --args '{"kind":"eval_run","id":"<uuid>"}' --json
```

Repeat until the job is `completed`, `failed`, or `cancelled`. Use the same
pattern for fine-tune jobs, optimizer experiments, and backtest runs (check
`describe job_status` for valid `kind` values).

## Errors and results

- HTTP or JSON-RPC failures print to stderr and exit non-zero.
- Tool handlers return `{"error": "…"}` inside the JSON payload instead of
  raising — inspect the printed JSON; follow any `hint` field.
- `--json` on `call` prints `isError`, `content`, and `structuredContent`.

## When to use MCP instead

If the host supports MCP and you want native tool-call UX, configure the
Overmind MCP server (`overmind init`) and use the `overmind` skill. The CLI
and MCP hit the same backend tools.

## Related skills

- Full platform workflows (datasets, evals, fine-tuning, optimizer):
  `overmind` skill and its reference files.
- Agent telemetry instrumentation: `overmind-agent-telemetry` skill.
