---
name: overmind-agent-telemetry
description: Work with Overmind, the agent observability and optimization platform — add tracing to a Python AI/LLM project, verify and audit instrumentation by fetching real traces, query traces/sessions programmatically via the REST API, and look up current Overmind documentation. Use when the user asks to add Overmind telemetry/observability/tracing, send traces to Overmind, check whether traces are landing, or fetch Overmind trace data.
---

# Overmind

This skill covers the common Overmind workflows: instrumenting applications
with tracing, verifying the traces that instrumentation produces, querying
trace data via the REST API, and accessing documentation.

## Core principles

Follow these for ALL Overmind work:

1. **Documentation first.** Never implement from memory — fetch the current
   docs before writing code (section 2 below). This file may be older than the
   platform.
2. **API for data access.** The same project API key that ingests traces reads
   them back. Query traces via the REST API (section 1), not by asking the
   user to describe what they see in the Console.
3. **Reference file per use case.** Check the relevant reference below before
   implementing.
4. **Verify with a real trace.** Instrumentation isn't done when the code
   compiles — it's done when you have fetched the trace you just sent and it
   carries everything the baseline requires.

## Use-case references

- Instrumenting an application (greenfield or alongside existing telemetry):
  [references/instrumentation.md](references/instrumentation.md)
- Querying traces/sessions via the REST API, including the post-setup
  verification loop: [references/api-access.md](references/api-access.md)

## 1. Overmind REST API

Base URL `https://api.overmindlab.ai` (override with `OVERMIND_API_URL`).
Authenticate with the project API key — the same one used for trace ingest:

```bash
curl -s -H "X-Api-Key: $OVERMIND_API_KEY" ...     # or: -H "Authorization: Bearer $OVERMIND_API_KEY"
```

Quick check that traces are landing (newest first):

```bash
curl -s -H "X-Api-Key: $OVERMIND_API_KEY" \
  "https://api.overmindlab.ai/api/traces/?ordering=-received_at&page_size=3"
```

Endpoints, filters, response fields and the full verification workflow:
[references/api-access.md](references/api-access.md).

Keys are created at https://console.overmindlab.ai/projects. Ask the user to
set `OVERMIND_API_KEY` in their shell or a `.env` file — never ask them to
paste the key into chat.

## 2. Overmind documentation

All docs are fetchable as plain text/markdown. Prefer your native web-fetch
tool; `curl` examples are illustrative. Always follow redirects (`curl -sL`) —
doc paths get reorganized and old paths redirect.

### 2a. Documentation index (llms.txt)

```bash
curl -sL https://docs.overmindlab.ai/llms.txt        # index of every page
curl -sL https://docs.overmindlab.ai/llms-full.txt   # all pages in one blob
```

### 2b. Fetch individual pages as markdown

Append `.md` to any page path:

```bash
curl -sL https://docs.overmindlab.ai/core/observability.md   # tracing/instrumentation — the key page
curl -sL https://docs.overmindlab.ai/quickstart.md
```

### Documentation workflow

1. Fetch **llms.txt** to orient — scan for the relevant page.
2. Fetch that page as **.md** and implement from it, not from memory.
3. For tracing work, `core/observability.md` is the page to re-check before
   and after implementing.
