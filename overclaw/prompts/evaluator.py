"""Prompts for ``overclaw.optimize.evaluator``."""

LLM_JUDGE_PROMPT = """\
You are an expert evaluator scoring an AI agent's output.

## Input
{input_json}

## Expected Output
{expected_json}

## Actual Output
{actual_json}

## Evaluation Criteria
{criteria}
{policy_rubric}
Score this output on the following dimensions. For each, give an integer 0–10:

1. **semantic_correctness**: How close is the actual output to the expected output \
in meaning? Consider whether the agent reached the right conclusion even if exact \
values differ slightly.

2. **internal_consistency**: Are all output fields logically consistent with each \
other? (e.g., a high score should align with a "hot" category, not "cold")

3. **reasoning_quality**: If there is a reasoning/explanation field, is it specific, \
grounded in the input data, and does it justify the other output fields?

4. **policy_compliance**: Did the output follow the agent's policy rules and \
constraints? (Score 10 if no policy is defined, or if all rules are satisfied. \
Deduct points for each rule violated.)

Return ONLY a JSON object:
{{"semantic_correctness": <0-10>, "internal_consistency": <0-10>, \
"reasoning_quality": <0-10>, "policy_compliance": <0-10>, \
"notes": "<one sentence>"}}
"""

LLM_JUDGE_BATCH_PROMPT = """\
You are an expert evaluator scoring multiple AI agent outputs independently.

## Evaluation Criteria
{criteria}
{policy_rubric}

## Cases to Score

{cases_block}

## Instructions

Score EACH case independently on four dimensions (integer 0–10 each):
1. **semantic_correctness**: How close is the actual output to the expected output in meaning?
2. **internal_consistency**: Are all output fields logically consistent with each other?
3. **reasoning_quality**: Is the reasoning specific, grounded in input data, and justified?
4. **policy_compliance**: Did the output follow policy rules? (10 if no policy or all satisfied.)

Return ONLY a JSON array with one object per case, in the same order:
[{{"case_id": <id>, "semantic_correctness": <0-10>, "internal_consistency": <0-10>, \
"reasoning_quality": <0-10>, "policy_compliance": <0-10>}}, ...]
"""
