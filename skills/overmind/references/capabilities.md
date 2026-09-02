# Agents — identity, prompts, GitHub analyze

Resolve agents with `list_capabilities` / `get_capability`. Stamp the returned `id`
into the SDK and inspect traces in [telemetry.md](telemetry.md).

All of this is Overmind MCP. Inspect each tool's schema for arguments.
`update_capability` and the GitHub analyze tools are gated (they mutate).

## Workflow

```
- [ ] 1. list_capabilities — id (bare UUID), slug, model, active_model
- [ ] 2. get_capability — card: id, active_model, source_repo, capped flow
- [ ] 3. capability_prompts / capability_eval_spec when you need prompt text or the eval contract
- [ ] 4. update_capability to retarget the live model (or patch description / display_name)
- [ ] 5. If no agents / no repo: analyze_github_repo or analyze_github_repo_url
- [ ] 6. assign_traces_to_capability for mis-attributed traces
```

## Card

`list_capabilities` and `get_capability` both return a top-level `id` (bare UUID) and
`active_model` (`{id, model_id, status}` or null). Use slug or display name
in later tool args — never paste ids to the user. Copy `id` verbatim into
`overmind.init(capability_id=)` / a `capability(..., id=)` scope / `OVERMIND_CAPABILITY_ID`.

`get_capability(capability_name_or_slug)` also returns:

- `source_repo` — linked GitHub repo (`name`, `repo_id`, branch, `head_sha`)
  or null
- `flow` — capability card (`source_path`, `modes[*].entrypoint_fn`, system
  prompt, tools, schemas). Capped; `flow_truncated` is true when clipped.
  Use the paths you got — don't invent files that weren't in the card.
- `source_path`, `active_eval_set`, `alias_pr`, `status`, `model`,
  `description`

`capability_prompts(capability_name_or_slug, limit?)` — versioned snapshots plus
flow-derived prompts; `system_prompt` is bounded (`truncated` when clipped).
`capability_eval_spec(capability_name_or_slug)` — input schema, output fields, tool
config, weights, consistency rules, optimizable / fixed elements.
`agent_base_model_throughput` — tokens/sec from the agent's own traces
([finetuning.md](finetuning.md)).

## Update

`update_capability(capability_name_or_slug, description?, name?, active_model?)`
— at least one field. `active_model` is a **READY** deployed-model UUID
from `list_deployed_models` (same project), or empty to clear. Refuse
non-READY deployments.

## GitHub analyze

One repo per project; the branch locks after linking. Clone + analysis
runs in the background — agents and prompts appear when it finishes.

- Connected account: `analyze_github_repo(full_name, branch?)` —
  `full_name` is `owner/repo`. Needs GitHub connected in Settings and the
  Overmind GitHub App granted on that repo.
- Public URL (no connection): `analyze_github_repo_url(url, branch?)`.
  Private repos need the connected-account path.

Returns `{queued, repo, branch, status}` — no job id. Wait by
`list_capabilities` until rows appear. If you already have
`get_capability.source_repo.repo_id`, poll
`job_status(kind="agent_discovery", id=<repo_id>)` until `ready` / `error`.
A second analyze call while in flight returns `queued: false`.

Needed before `create_model_swap_pr` / `create_optimizer_pr` if no repo
is linked.

## Traces

`assign_traces_to_capability(capability_name_or_slug, trace_ids)` — re-attributes
every span of those traces onto the agent. `trace_ids` are bare OTel hex
or `traces:<hex>` refs. Typical use: connector-imported traces that landed
agentless.
