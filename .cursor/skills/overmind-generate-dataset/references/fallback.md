# Fallback: Direct LLM Generation

If `from overmind.optimize.data import generate_diverse_synthetic_data` fails, overmind is not installed. Tell the user to install it first (`pip install overmind`), then re-run. If overmind is installed but still can't be imported, fall back to direct LLM calls:

````python
import os, json
import litellm  # or openai

MODEL = os.getenv("SYNTHETIC_DATAGEN_MODEL", "gpt-4o")

PROMPT = f"""
You are generating a synthetic test dataset for an AI agent.

Agent description: {AGENT_DESCRIPTION}

Agent source code:
```python
{AGENT_CODE}
````

Input schema: {json.dumps(EVAL_SPEC['input_schema'], indent=2)}
Output schema: {json.dumps(EVAL_SPEC['output_fields'], indent=2)}

Generate {NUM_SAMPLES} diverse test cases covering these {NUM_PERSONAS} personas:

1. Novice user — basic, possibly incomplete inputs
1. Power user — complex, well-formed inputs
1. Edge case tester — boundary values, empty fields, unusual combos
1. Adversarial user — misleading, contradictory, or injection-style inputs
1. Domain expert — nuanced, technically precise scenarios
   (repeat or mix personas if more than 5 are requested)

Return ONLY a JSON array. Each item: {{"input": {{...}}, "expected_output": {{...}}}}
"""

response = litellm.completion(
model=MODEL,
messages=[{"role": "user", "content": PROMPT}],
temperature=0.8,
)
content = response.choices[0].message.content
start, end = content.find("["), content.rfind("]") + 1
cases = json.loads(content[start:end])

```
```
