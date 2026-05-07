# Overmind Example Agents

A curated set of **deliberately sub-optimal** agents across different industries, agent patterns, and tool mixes. Each one is intended as a showcase for running`overmind setup` + `overmind optimize` and watching scores improve.

Every agent exposes a single entrypoint:

```python
def run(input_data: dict) -> dict: ...
```

so you can register it with:

```bash
overmind agent register agent:run
```

## The portfolio

Each example’s code lives under `examples/<id>/` in this repo (for example `examples/lead_qualifier/`). The table below uses short labels; scores use the directory `<id>` (`lead_qualifier`, `support_triage`, …).

| #   | Example            | Industry         | Pattern                                            | Models             | External APIs    |
| --- | ------------------ | ---------------- | -------------------------------------------------- | ------------------ | ---------------- |
| 1   | Lead qualifier     | Sales / SaaS     | Classifier + enrichment                            | OpenAI             | EXA + local CSV  |
| 2   | Support triage     | Customer Support | Router + response drafter                          | Anthropic          | Local KB + EXA   |
| 3   | Contract extractor | Legal            | Long-context extraction                            | OpenAI             | *none*           |
| 4   | Clinical coder     | Healthcare / RCM | Classifier + 4 lookups                             | OpenAI             | local JSON       |
| 5   | AP invoice         | Finance / AP     | Decisioning + policy + fraud signals               | OpenAI             | local JSON       |
| 6   | On-call triage     | DevOps / SRE     | **Multi-agent**: router → investigator → responder | OpenAI + Anthropic | local JSON       |
| 7   | Returns concierge  | E-commerce       | Policy-driven decisioning + customer copy          | OpenAI             | local JSON       |
| 8   | Research brief     | Marketing / SEO  | **Multi-agent**: researcher → outliner → editor    | OpenAI             | EXA + local JSON |

## Optimization runs (example results)

Scores are from a representative `overmind optimize` run (train-side summary in each agent’s `experiments/report.md` under `.overmind`, depending on where you ran it). **Final score** is the reported best average (same as the **Best** column in that report).

| Example            | Baseline | Final score | Improvement |
| ------------------ | -------- | ----------- | ----------- |
| lead_qualifier     | 52.5     | 73.9        | +21.4       |
| support_triage     | 57.9     | 64.7        | +6.8        |
| contract_extractor | 74.3     | 94.1        | +19.8       |
| clinical_coder     | 38.7     | 46.2        | +7.5        |
| ap_invoice         | 36.5     | 57.0        | +20.5       |
| oncall_triage      | 49.5     | 70.8        | +21.3       |
| returns_concierge  | 47.4     | 65.5        | +18.1       |
| research_brief     | 59.5     | 74.0        | +14.5       |

**Contract extractor** is different: register and run it only from the
`examples/contract_extractor/` directory (see that folder’s README). The other
examples assume `.overmind` at the **repository root** and entrypoints like
`examples.<folder>.agent:run`.

## Running

```bash
cd examples/lead_qualifier
overmind agent register lead-qualifier agent:run
overmind agent validate lead-qualifier --data data/seed.json
overmind setup lead-qualifier --data data/seed.json
overmind optimize lead-qualifier
```

Run **validate** before **setup** so you can smoke-test the entrypoint on seed data. Paths are relative to the repo root; if you work inside an example folder instead, adjust accordingly (e.g. from `examples/support_triage/` with the agent registered as `tria`: `overmind agent validate tria --data data/seed.json`).

(Agent **names** can be anything — they're just registry keys. The folder name must be a valid Python identifier, so we use `lead_qualifier` on disk and `lead-qualifier` as the pretty registry name.)

## Design notes

- No `policies.md` ships with these examples — `overmind setup` will infer one from the agent code, or you can drop in your own via `--policy`.
- Baselines are intentionally leaky (vague prompts, thin tool docstrings, wrong model tier, unbounded iteration) — but in realistic ways. They aren't straw-manned.
- Seed datasets (`data/seed.json`) are small (5–10 cases). Let`overmind setup` synthesise more from the policy.
