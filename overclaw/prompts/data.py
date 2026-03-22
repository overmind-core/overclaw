"""Prompts for ``overclaw.optimize.data``."""

SYNTHETIC_DATA_LEGACY_PROMPT = (
    "You are a test-data engineer. Generate exactly {num_samples} "
    "diverse, realistic test cases for this AI agent:\n\n"
    "{agent_description}\n"
    "{code_section}{policy_section}"
    "Requirements:\n"
    '- Each test case MUST have an "input" dict and an "expected_output" dict.\n'
    '- The keys in "input" and "expected_output" must match the schemas used by the\n'
    "  agent (inspect the code carefully).\n"
    "- Include a mix of easy, medium, and hard difficulty.\n"
    "- Include 2–3 edge cases (ambiguous inputs, missing optional fields, etc.).\n"
    "- Expected outputs must reflect the *ideal correct* answer for each input.\n"
    "- For numeric scores, pick values that are realistic for the scenario.\n"
    "- For categorical fields, only use values the agent is expected to produce.\n"
    "- If agent policy rules or edge cases are provided above, generate specific "
    "test cases that exercise each rule and edge case. The expected_output for "
    "these cases MUST reflect the correct behaviour as defined by the policy.\n\n"
    "Return ONLY a JSON array — no markdown fences, no commentary.\n"
)

PERSONAS_GENERATION_PROMPT = """\
You are designing a comprehensive test suite for an AI agent.

<AgentDescription>
{agent_description}
</AgentDescription>
{code_section}
<InputSchema>
{input_schema_text}
</InputSchema>

<OutputSchema>
{output_schema_text}
</OutputSchema>
{domain_section}
Generate exactly {num_personas} distinct user personas who would realistically
interact with this agent. Each persona must include:
- name: a short label (2-4 words)
- role_and_background: their job, industry, and context
- skill_level: one of "novice", "intermediate", "expert"
- intent: one of "standard", "edge_case_probing", "adversarial", "exploratory"
- communication_style: how they phrase inputs (terse, verbose, ambiguous, precise)
- domain_behavior: specific patterns in how they'd construct inputs for THIS agent
  (reference actual input fields from the schema)
- typical_scenarios: 2-3 example scenarios this persona would bring

Diversity requirements:
- At least one novice and one expert
- At least one adversarial or edge-case-probing intent
- Cover different industries/contexts relevant to the agent's domain
- Vary communication styles

Respond ONLY with JSON (no markdown fences):
{{"personas": [
  {{
    "name": "...",
    "role_and_background": "...",
    "skill_level": "...",
    "intent": "...",
    "communication_style": "...",
    "domain_behavior": "...",
    "typical_scenarios": ["...", "..."]
  }}
]}}"""

BATCH_GENERATION_PROMPT = """\
You are generating test cases for an AI agent. You must produce cases that
match the exact schemas below.

<AgentDescription>
{agent_description}
</AgentDescription>
{code_section}
<InputSchema>
{input_schema_text}
</InputSchema>

<OutputSchema>
{output_schema_text}
</OutputSchema>
{policy_section}
You are generating cases from the perspective of this user persona:
<Persona>
{persona_block}
</Persona>
{gap_section}{existing_section}
Generate exactly {batch_size} test cases. For each case:
1. "input": a dict matching InputSchema — realistic values this persona would provide
2. "expected_output": a dict matching OutputSchema — the CORRECT output per the
   policy rules and decision logic
3. "_meta": {{"difficulty": "easy|medium|hard|edge_case", "persona": "{persona_name}"}}

CRITICAL RULES:
- expected_output enum fields MUST use ONLY the allowed values from OutputSchema
- expected_output number fields MUST be within the specified ranges
- expected_output MUST be consistent with the policy's decision mapping
- All required fields in both input and expected_output MUST be present
- For edge cases: test boundary conditions, missing optional fields, ambiguous inputs
- Each case must be DISTINCT from the others and from existing cases

Respond ONLY with a JSON object (no markdown fences):
{{"cases": [{{...}}, ...]}}"""
