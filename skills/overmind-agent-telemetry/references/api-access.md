# Querying Overmind trace data via the REST API

Read traces back with the same project API key used for ingest. There is no
CLI for this — call the REST endpoints directly with `curl` or `requests`.

## Auth and base URL

```bash
BASE=https://api.overmindlab.ai          # override: OVERMIND_API_URL
-H "X-Api-Key: $OVERMIND_API_KEY"        # or: -H "Authorization: Bearer $OVERMIND_API_KEY"
```

A key can read traces from every project its owner is a member of — add
`?project=<uuid>` to pin a query to one project.

## Endpoints

### `GET /api/traces/` — list traces

One row per trace (root spans only). Add `all_spans=true` to list individual
spans instead. Paginated: `{count, next, previous, results: [...]}` with
`page` / `page_size` (default 25, max 100).

Useful filters (django-filter style, combinable):

| Param | Meaning |
| --- | --- |
| `ordering` | e.g. `-received_at` (newest ingested first), `-start_time_ns`, `total_cost` |
| `trace_id`, `span_id` | Exact match (32-hex OTel ids) |
| `project`, `agent`, `conversation` | Exact match by UUID |
| `search` | Matches name / service_name / trace_id / span_id |
| `has_error`, `has_model` | `true` / `false` |
| `span_type`, `service_name`, `name`, `operation`, `status_code` | Exact match |
| `received_at__gte` / `__lte` | ISO datetime — "traces since I started the app" |
| `min_duration_ms` / `max_duration_ms` | Duration window |
| `total_tokens__gte` / `__lte`, `total_cost__gte` / `__lte` | Usage thresholds |
| `session` | Session UUID drill-in |

List row fields: `trace_id`, `span_id`, `project`, `agent`, `agent_name`,
`span_type`, `operation`, `name`, `service_name`, `kind`, `status_code`,
`status_message`, `start_time_ns`, `end_time_ns`, `duration_ns`,
`total_tokens`, `total_cost`, `cache_read_tokens`, `model`, `conversation`,
`feedback_score`, `received_at`. Token/cost are trace-wide aggregates.

### `GET /api/traces/{trace_id}/` — full trace

`trace_id` is the 32-char hex OTel id. Returns
`{trace_id, root, span_count, spans: [...]}` with spans ordered by
`start_time_ns`. Each span adds `parent_span_id`, `is_root`,
`resource_attrs`, `attributes`, `events`, `links` — inputs/outputs live in
`attributes` (`overmind.input.data` / `overmind.output.data`), so this is the
payload to audit.

### `GET /api/traces/services/` — distinct service names visible to the caller

## Verification workflow (after instrumenting)

1. Run the instrumented app so a trace is sent (flush first in short-lived
   processes).
2. Fetch the newest trace:

```bash
curl -s -H "X-Api-Key: $OVERMIND_API_KEY" \
  "$BASE/api/traces/?ordering=-received_at&page_size=1"
```

3. Pull the full span tree with the returned `trace_id`:

```bash
curl -s -H "X-Api-Key: $OVERMIND_API_KEY" "$BASE/api/traces/<trace_id>/"
```

4. Audit against the baseline in
   [instrumentation.md](instrumentation.md#what-a-good-trace-carries):
   `agent_name` set and constant, `model` + `total_tokens` + `total_cost`
   populated, `conversation` set for multi-turn apps, span `attributes`
   carrying inputs/outputs on the entry point and key steps, varied
   `span_type`s, no secrets in payloads.
5. Fix gaps, re-run, re-fetch until it clears. An empty result set means
   ingest failed — see the troubleshooting list at the end of
   [instrumentation.md](instrumentation.md).
