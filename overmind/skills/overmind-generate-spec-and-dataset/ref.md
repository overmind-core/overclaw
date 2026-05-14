```python
import json, os, pathlib

for line in pathlib.Path(".overmind/.env").read_text().splitlines():
    s = line.strip()
    if not s or s.startswith("#") or "=" not in s:
        continue
    k, v = s.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip())

from overmind.client import get_client

client = get_client()
if client is None:
    raise SystemExit("OVERMIND_API_KEY not configured")

agent = client.get_agent("airline-triage")
print(f"agent: name={agent.name!r} id={agent.id} slug={agent.slug}")

datapoints = json.loads(
    pathlib.Path(".overmind/agents/airline-triage/setup_spec/dataset.json").read_text()
)
print(f"loaded {len(datapoints)} datapoints")

result = client.upsert_dataset(
    agent_id=agent.id,
    datapoints=datapoints,
    source="synthetic",
    generator_model=os.getenv("ANALYZER_MODEL", ""),
    name="airline-triage initial dataset",
    make_active=True,
)
if result is None:
    raise SystemExit("upsert_dataset returned None (see logs)")

print(f"\nuploaded dataset:")
print(f"  id={result.get('id')}")
print(f"  version={result.get('version')}")
print(f"  active={result.get('is_active')}")
dps = result.get("datapoints") or []
print(f"  datapoints={len(dps)}")
```
