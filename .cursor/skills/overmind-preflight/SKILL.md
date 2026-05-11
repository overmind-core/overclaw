---
name: overmind-preflight
description: "Validate the agent + eval spec + dataset pipeline before `overmind optimize`. Use when the user wants to run preflight, smoke-test the eval pipeline, fix a stale preflight, resolve missing API credentials before optimization, or repair entrypoint / eval-spec / dataset / instrumentation issues that would otherwise break optimize. Runs the agent against a tiny dataset slice, classifies failures into deterministic kinds, and autonomously patches every plumbing issue — entrypoint harness bugs, weights, schema mismatches, broken metrics, invalid rows, instrumentation — only stopping to ask the user when a credential is missing."
metadata:
  version: "1.0"
  product: "Overmind"
---

# Run Overmind Preflight

`overmind preflight` is the validation gate between dataset generation and optimization. It runs the registered agent against a 2-row dataset slice, classifies every failure into a deterministic *kind*, and autonomously fixes every plumbing issue it can — eval-spec weight drift, output-schema mismatches, dataset rows that violate the input schema, broken metric configs, missing instrumentation, **and the registered Overmind entrypoint harness itself** (the thin wrapper file `/overmind-register-agent` writes; the native agent code it imports is left untouched and remains optimize's domain).

The only failure mode that requires a human is a missing credential. Everything else is patched in place, snapshotted to disk, and recorded in `preflight.log` for review.

After this skill finishes successfully, `overmind optimize` is guaranteed to start without infrastructure errors. If it doesn't pass, **do not** run optimize — fix the blocker first.

## Operating principles

- **Run from the project root** (the directory that contains `.overmind/`). Never `cd` into a parent.
- **JSON-in/JSON-out CLI**: every `overmind preflight` subcommand emits a single JSON envelope on stdout. Parse it; do not regex-scrape.
- **Autonomous repair, not autonomous secrets**: the loop will silently patch eval-spec, dataset, instrumented files, and the registered entrypoint harness file. It will *never* edit the native agent code that the harness imports (that's optimize's job) and *never* invent or assume a credential — credentials always trigger a question to the user.
- **Entrypoint repair is bounded**: at most `state.max_entrypoint_repairs` (default 2) LLM-driven harness rewrites per preflight run, snapshotted and reverted if they touch any file other than the harness, leave the file empty, or produce no change. If the model isn't available (no `ANALYZER_MODEL` / no provider key), the runner falls back to dropping the missing fields from the eval spec instead.
- **Snapshot before every patch**: `.overmind/agents/<name>/preflight/snapshots/iter_<N>/` always holds the previous version of every file the loop touched. Tell the user where to find it.
- **No secret inspection**: never echo, log, or copy a secret value into chat. Pipe values to `--with-secrets-stdin` or to `set-secret --key NAME` via stdin.
- **Idempotent**: re-running preflight on a green pipeline produces no patches and bumps no iterations. If the user re-runs without changing anything, simply re-report the existing state.
- **Only re-runs after edits**: `overmind optimize` checks the hashes recorded in `preflight.json` against the live artifacts. If anything drifts, optimize refuses with `preflight is stale` — re-run this skill.

## Workflow

```
Preflight Progress:
- [ ] Step 1: Verify prerequisites (registration + spec + dataset)
- [ ] Step 2: Scan for missing credentials
- [ ] Step 3: Ask the user for any missing credentials (single batched question)
- [ ] Step 4: Run the convergence loop
- [ ] Step 5: Interpret the report and tell the user what changed
- [ ] Step 6: If blocked, surface the blocker; otherwise hand off to /overmind-optimise-agent
```

### Step 1 — Verify prerequisites

The skill assumes the agent is already registered and that
`setup_spec/eval_spec.json` + `setup_spec/dataset.json` already exist.

```bash
python - <<'PY'
import json, pathlib, tomllib

agent = "<agent-name>"
root = pathlib.Path(".overmind")
toml = root / "agents.toml"
spec = root / "agents" / agent / "setup_spec" / "eval_spec.json"
data = root / "agents" / agent / "setup_spec" / "dataset.json"

ok = []
miss = []
if toml.is_file() and agent in tomllib.loads(toml.read_text()).get("agents", {}):
    ok.append("registered")
else:
    miss.append("registration")
if spec.is_file():
    ok.append("eval_spec")
else:
    miss.append("eval_spec")
if data.is_file():
    ok.append("dataset")
else:
    miss.append("dataset")

print(json.dumps({"present": ok, "missing": miss}))
PY
```

If `registration` is missing → run `/overmind-register-agent` first.
If `eval_spec` or `dataset` is missing → run `/overmind-generate-spec-and-dataset` first.

### Step 2 — Scan for missing credentials

```bash
overmind preflight scan <agent-name>
```

Returns:

```json
{
  "agent_name": "lead_qualifier",
  "env_path":   ".overmind/agents/lead_qualifier/.env",
  "discovered_env_vars": {"EXA_API_KEY": null, "LEAD_QUALIFIER_MODEL": "gpt-4o"},
  "providers_detected": ["openai"],
  "required_keys":      ["OPENAI_API_KEY", "EXA_API_KEY", "LEAD_QUALIFIER_MODEL"],
  "missing":            ["OPENAI_API_KEY"],
  "supplied":           ["EXA_API_KEY", "LEAD_QUALIFIER_MODEL"],
  "status": "ok"
}
```

Show the user a short summary: providers detected, every required key, and which keys are still missing. Do not show any *values*.

### Step 3 — Ask the user for missing credentials (only if any)

If `missing` is non-empty, ask **once** in a single `AskQuestion` batch — one free-text question per missing key. Phrase the prompt:

> "Overmind needs `<KEY>` to call the provider during preflight. Paste the value (it will be saved to `.overmind/agents/<name>/.env` with `0600` permissions and never logged)."

After collecting the answers, persist each one without putting the value on a command line:

```bash
echo -n "<value>" | overmind preflight set-secret <agent-name> --key OPENAI_API_KEY
```

The CLI returns:

```json
{"status": "ok", "key": "OPENAI_API_KEY", "env_path": "...", "validated": true}
```

If `validated` is `false`, surface the `validate_error` field — usually the user pasted the wrong key or the provider rejected it. Ask once for a corrected value, retry, then proceed even if validation can't be performed (some providers don't support cheap probes).

If the user prefers a single round-trip, you can pipe a JSON object on stdin to `preflight run --with-secrets-stdin`:

```bash
echo '{"OPENAI_API_KEY": "<value>", "EXA_API_KEY": "<value>"}' \
  | overmind preflight run <agent-name> --with-secrets-stdin
```

This persists every key first, then runs the convergence loop in one call.

### Step 4 — Run the convergence loop

```bash
overmind preflight run <agent-name>
```

Useful flags (rarely needed):

- `--max-iters N` (default 5) — convergence budget. Larger isn't always better; if 5 iters can't fix it, manual investigation usually beats raising the budget.
- `--max-rows N` (default 2) — dataset slice size. Increase to 3–5 only if the agent's failure modes are highly input-dependent; otherwise the wall-clock cost compounds.
- `--timeout 120` — per-case subprocess timeout in seconds.

The CLI returns a `PreflightReport` envelope:

```json
{
  "status":         "green" | "green_with_quality_notes" | "blocked_secrets" | "blocked_no_convergence",
  "agent_name":     "...",
  "iterations":     2,
  "baseline_score": 0.42,
  "span_count":     17,
  "cases_run":      2,
  "cases_succeeded":2,
  "cases_failed":   0,
  "patches_applied":[
    {"iteration": 1, "kind": "invalid_weights", "file": ".../eval_spec.json", "diff_summary": "Renormalised 4 field weights to fit total_points=100."}
  ],
  "issues_remaining":[],
  "missing_secrets": [],
  "hashes":         {"entrypoint": "sha256:...", "eval_spec": "sha256:...", "dataset": "sha256:...", "instrumented": "sha256:...", "env_keys": "sha256:..."},
  "snapshots_dir":  ".overmind/agents/<name>/preflight/snapshots",
  "log_path":       ".overmind/agents/<name>/preflight/preflight.log",
  "message":        "Pipeline is healthy and ready for overmind optimize."
}
```

Exit codes:

| Code | Meaning |
|------|---------|
| 0    | `status` is `green` or `green_with_quality_notes` |
| 1    | The CLI itself errored (envelope has `status:"error"`) |
| 2    | A run completed but `status` is not green (blocked_secrets / blocked_no_convergence) |

### Step 5 — Interpret the report

Tell the user:

- **Status** and the human-readable `message`.
- **What changed**: if `patches_applied` is non-empty, list each `kind`, the file, and the `diff_summary`. Mention that the previous versions are in `snapshots_dir` and full audit lines are in `log_path`. The user did not ask for these mutations — be transparent about them.
- **Quality notes** (`green_with_quality_notes`): the pipeline runs but the baseline score is low or some cases crash inside the agent. That is by design — `overmind optimize` exists to improve score and fix runtime crashes in agent code. Tell the user the next step is still optimization.

### Step 6 — Branch on status

| Status | Action |
|---|---|
| `green` | Hand off: "Run `/overmind-optimise-agent` for `<agent>` to start the optimization loop." |
| `green_with_quality_notes` | Same hand-off, but warn: "Baseline is low / some agent-side crashes were observed. Optimize will tackle those." |
| `blocked_secrets` | Surface `missing_secrets`. Ask the user for each via `AskQuestion`, persist with `preflight set-secret …`, then re-run `preflight run`. Loop at most 2 times before escalating. |
| `blocked_no_convergence` | The fix loop ran out of budget. Read `log_path`, summarise the residual `issues_remaining` for the user, and ask whether to raise `--max-iters`, manually edit the spec/dataset, or open the snapshots dir. **Do not** suggest `OVERMIND_SKIP_PREFLIGHT=1` — that's an emergency-only escape hatch. |

## Useful inspection commands

```bash
overmind preflight status <agent-name>     # print the persisted report
overmind preflight reset  <agent-name>     # delete preflight state to force a re-run
```

`reset` is useful when the user explicitly wants to wipe past audit history.
Do not call it implicitly — snapshots and the log are the only record of what
the autofix loop changed.

## What the skill must NOT do

- Never put a secret value on a command line. Pipe via stdin to `set-secret` or `run --with-secrets-stdin`.
- Never call `overmind optimize` or `overmind optimize-step init` while the report is non-green or stale — both will refuse, but it wastes the user's time and pollutes their state.
- Never edit `eval_spec.json` / `dataset.json` / instrumented files manually. Let the autofix loop own those mutations so snapshots stay accurate.
- Never invent a missing credential by looking at related env vars — always ask the user.

## Common issues

| Problem | Fix |
|---|---|
| `status:"error", error:"agent_not_registered"` | Run `/overmind-register-agent`. |
| `status:"error", error:"missing_eval_spec"` / `missing_dataset` | Run `/overmind-generate-spec-and-dataset`. |
| `blocked_secrets` after a fresh scan | Provider rejected the credential. Ask the user once for a corrected value; if it still fails, ask whether to fall back to a different provider/model. |
| `blocked_no_convergence` with all `issues_remaining` of kind `runtime_crash` | The agent code itself crashes — that is exactly the situation `overmind optimize` is built to fix. Surface the issue, ask the user whether to optimize anyway (it is safe — preflight gate will accept `green_with_quality_notes`). |
| `blocked_no_convergence` with `dep_missing` repeating | The instrumented copy's `requirements.txt` is incomplete; manually add the package and re-run, or run `/overmind-register-agent` to refresh the instrumented snapshot. |
| Optimize complains that preflight is stale | The user changed `eval_spec.json` / `dataset.json` / the entrypoint after preflight passed. Re-run this skill. |
