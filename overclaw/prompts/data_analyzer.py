"""Prompts for ``overclaw.optimize.data_analyzer``."""

DATA_QUALITY_ANALYSIS_PROMPT = """\
You are a test data quality analyst. Analyze this dataset for an AI agent.

<AgentDescription>
{agent_description}
</AgentDescription>

<InputSchema>
{input_schema_text}
</InputSchema>

<OutputSchema>
{output_schema_text}
</OutputSchema>
{policy_section}
<Dataset>
{dataset_json}
</Dataset>

Analyze thoroughly:
1. **Coverage completeness**: Which input field values/combinations are
   well-represented? Which are missing or underrepresented?
2. **Difficulty distribution**: Classify each case as easy/medium/hard/edge_case.
   What's the distribution?
3. **Policy rule coverage**: For each domain rule and known edge case, is there
   at least one test case that exercises it? List uncovered rules.
4. **Enum field coverage**: For each enum output field, which values appear in
   expected_output? Which are missing?
5. **Number field boundary coverage**: For number fields with ranges, are
   boundary values (min, max, midpoint) tested?
6. **Potential quality issues**: Cases where expected_output seems inconsistent
   with the policy rules or with other fields.

Respond ONLY with JSON (no markdown fences):
{{
  "overall_quality_score": <1-10>,
  "case_count": <N>,
  "difficulty_distribution": {{"easy": <N>, "medium": <N>, "hard": <N>, "edge_case": <N>}},
  "coverage_gaps": [
    {{"area": "...", "description": "...", "severity": "high|medium|low"}}
  ],
  "uncovered_policy_rules": ["rule text..."],
  "uncovered_edge_cases": ["edge case..."],
  "uncovered_enum_values": {{"field": ["value1", "value2"]}},
  "quality_issues": [
    {{"case_index": <N>, "issue": "..."}}
  ],
  "augmentation_recommendation": "...",
  "suggested_additional_cases": <N>
}}"""
