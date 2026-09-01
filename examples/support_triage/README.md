# Support Ticket Triage

Routes inbound support tickets: classifies, prioritises, flags escalation,
drafts a first reply. Uses an internal KB, a customer lookup, and public docs.

**Stack:** Anthropic SDK (native tool use) + local JSON KB + EXA fallback.

## Seeded sub-optimalities

- No priority rubric in the prompt (model over-uses P1/P0).
- Identical one-liner descriptions on all three tools.
- Prefers web search over the internal KB.
- Uses Claude Sonnet for a classification task.
- Tone not calibrated to customer tier.
- JSON output parsed with best-effort fallback.

## Connect

Scan the repository from the Console (**Agent → Connect a repository**). The
scan mints the capability and its tasks; copy the **capability id** from the
capability page. `overmind optimise` runs optimisation experiments against it.

## Trace it

Run the agent directly to stream a full trace (tokens, cost, tool calls, the
run unit and its deliverable) to Overmind. `capability_id` is the only key
ingest binds by; the `capability` label is display-only.

```bash
export OVERMIND_API_KEY=ovr_...
export OVERMIND_CAPABILITY_ID=6f1c...   # the capability UUID from the Console
python agent.py
```

See [`docs/tracing-attributes.md`](../../docs/tracing-attributes.md) for the full
attribute contract.
