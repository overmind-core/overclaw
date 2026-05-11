---
name: overmind-preflight
description: "Validate the agent + eval spec + dataset pipeline before `overmind optimize`. Use when the user wants to run preflight, smoke-test the eval pipeline, resolve missing API credentials before optimization, or repair eval-spec / dataset / instrumentation issues that would otherwise break optimize. Runs the agent end-to-end against a small dataset slice, detects mis-performance (agent crashes, degenerate output, broken schema/spec), auto-fixes what it can, and reports clearly."
metadata:
  version: "2.0"
  product: "Overmind"
---

# Run Overmind Preflight

`overmind preflight` validates the full agent → eval pipeline before you run `overmind optimize`. It runs the registered agent against 2 dataset rows end-to-end, detects any mis-performance or plumbing issues, auto-fixes what it can deterministically, and reports the result.

## What it checks and fixes

| Problem | Preflight action |
|---|---|
| Missing API credentials | Hard block — surfaces the key name, asks the user once |
| Agent crashes (entrypoint broken, runtime error) | Reports the error clearly; quality note for optimize |
| Agent returns identical output for every input (degenerate) | Quality note — flags it, leaves fix to optimize |
| eval_spec weights don't sum to total_points | Auto-fixed: renormalises weights |
| eval_spec scores a field the agent never returns | Auto-fixed: drops the missing field |
| eval_spec field type unrecognised by scorer | Auto-fixed: coerces to `"text"` |
| Dataset rows violating input_schema | Auto-fixed: invalid rows dropped |
| Missing Python dependency in instrumented copy | Auto-fixed: added to requirements.txt |
| No `@observe()` spans captured | Auto-fixed: re-runs `instrument_directory` |

Preflight does **two passes at most**: one to detect and fix, one to verify the fixes held. No convergence loop, no LLM calls, no snapshot files.

`overmind optimize` does **not** require a green preflight — it just prints an advisory if the report is missing or non-green. Running preflight is strongly recommended so you catch plumbing issues before they burn optimization budget.

## Operating principles

- **Run from the project root** (the directory that contains `.overmind/`).
- **JSON-in/JSON-out CLI**: every `overmind preflight` subcommand emits a single JSON envelope on stdout. Parse it; do not regex-scrape.
- **Never put a secret value on a command line.** Pipe values to `--with-secrets-stdin` or to `set-secret` via stdin.
- **Re-run after meaningful edits**: if the user changes the eval spec, dataset, or entrypoint after preflight passed, re-run this skill so the next optimize starts from a validated state.

## Workflow

```
Preflight Progress:
- [ ] Step 1: Verify prerequisites (registration + spec + dataset)
- [ ] Step 2: Scan for missing credentials
- [ ] Step 3: Ask the user for any missing credentials (single batched question)
- [ ] Step 4: Run preflight
- [ ] Step 5: Interpret the report and tell the user what changed
- [ ] Step 6: Hand off or surface the blocker
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

ok, miss = [], []
if toml.is_file() and agent in tomllib.loads(toml.read_text()).get("agents", {}):
    ok.append("registered")
else:
    miss.append("registration")
ok.append("eval_spec") if spec.is_file() else miss.append("eval_spec")
ok.append("dataset") if data.is_file() else miss.append("dataset")

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
  "required_keys":      ["OPENAI_API_KEY", "EXA_API_KEY"],
  "missing":            ["OPENAI_API_KEY"],
  "supplied":           ["EXA_API_KEY"],
  "status": "ok"
}
```

Show the user: providers detected, every required key, which are missing. Do **not** show any values.

### Step 3 — Ask for missing credentials (only if any)

If `missing` is non-empty, ask **once** in a single `AskQuestion` batch — one free-text question per missing key:

> "Overmind needs `<KEY>` to call the provider during preflight. Paste the value (it will be saved to `.overmind/agents/<name>/.env` with `0600` permissions and never logged)."

Persist each answer without putting the value on a command line:

```bash
echo -n "<value>" | overmind preflight set-secret <agent-name> --key OPENAI_API_KEY
```

The CLI returns `{"status": "ok", "key": "OPENAI_API_KEY", "validated": true}`.
If `validated` is `false`, surface the `validate_error` and ask once for a corrected value.

Alternatively, pass all secrets in one go:

```bash
echo '{"OPENAI_API_KEY": "<value>", "EXA_API_KEY": "<value>"}' \
  | overmind preflight run <agent-name> --with-secrets-stdin
```

### Step 4 — Run preflight

```bash
overmind preflight run <agent-name>
```

Optional flags (rarely needed):
- `--max-rows N` (default 2) — number of dataset rows to run.
- `--timeout 120` — per-case subprocess timeout in seconds.

Returns a `PreflightReport` envelope:

```json
{
  "status":          "green" | "green_with_quality_notes" | "blocked_secrets" | "blocked_no_convergence",
  "agent_name":      "...",
  "iterations":      1,
  "baseline_score":  0.42,
  "cases_run":       2,
  "cases_succeeded": 2,
  "cases_failed":    0,
  "patches_applied": [
    {"kind": "invalid_weights", "file": "...eval_spec.json", "diff_summary": "Renormalised 4 field weights."}
  ],
  "issues_remaining": [],
  "missing_secrets":  [],
  "log_path":         ".overmind/agents/<name>/preflight/preflight.log",
  "message":          "Pipeline is healthy and ready for overmind optimize."
}
```

Exit codes:

| Code | Meaning |
|------|---------|
| 0    | `status` is `green` or `green_with_quality_notes` |
| 1    | CLI error (envelope has `status:"error"`) |
| 2    | Run completed but status is not green |

### Step 5 — Interpret the report

Tell the user:

- **Status** and the human-readable `message`.
- **What changed**: if `patches_applied` is non-empty, list each `kind`, the file, and the `diff_summary`. Be transparent — the user did not ask for these mutations.
- **Quality notes** (`green_with_quality_notes`): the pipeline runs but the agent crashes on some inputs, returns degenerate output, or scores very low. These are quality issues — exactly what `overmind optimize` is built to fix. The next step is still optimization.

### Step 6 — Branch on status

| Status | Action |
|---|---|
| `green` | Hand off: "Run `/overmind-optimise-agent` for `<agent>` to start the optimization loop." |
| `green_with_quality_notes` | Same hand-off, but note: "Some quality issues were observed (crashes / degenerate output / low score). Optimize will tackle those." |
| `blocked_secrets` | Surface `missing_secrets`. Ask the user for each via `AskQuestion`, persist with `set-secret`, re-run `preflight run`. |
| `blocked_no_convergence` | Auto-fixes couldn't resolve all issues. Read `issues_remaining` from the report, describe each to the user, and ask whether to manually fix the spec/dataset or proceed straight to optimize anyway. |

## Useful inspection commands

```bash
overmind preflight status <agent-name>     # print the persisted report
overmind preflight reset  <agent-name>     # delete preflight state to force a re-run
```

## What the skill must NOT do

- Never put a secret value on a command line. Pipe via stdin.
- Never call `overmind optimize` from inside this skill — that's `/overmind-optimise-agent`'s job.
- Never manually edit `eval_spec.json` / `dataset.json`. Let the preflight runner own those mutations so the log stays accurate.
- Never invent a missing credential — always ask the user.

## Common issues

| Problem | Fix |
|---|---|
| `status:"error", error:"agent_not_registered"` | Run `/overmind-register-agent`. |
| `status:"error"` with `missing_eval_spec` / `missing_dataset` | Run `/overmind-generate-spec-and-dataset`. |
| `blocked_secrets` after scan | Credential rejected. Ask the user once for a corrected value; if it still fails, ask whether to fall back to a different model/provider. |
| `blocked_no_convergence` with `runtime_crash` or `degenerate_output` in `issues_remaining` | Agent code issue — exactly the situation `overmind optimize` is built for. Surface the issue and offer to hand off. |
| Optimize fails on what looks like plumbing | User likely changed eval_spec/dataset/entrypoint after preflight passed. Re-run this skill. |
