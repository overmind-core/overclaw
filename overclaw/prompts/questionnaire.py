"""Prompts for ``overclaw.setup.questionnaire``."""

REFINEMENT_PROMPT = """\
You are refining evaluation criteria for an AI agent based on the user's \
domain knowledge and feedback.

## Original Agent Analysis
{analysis_json}

## Original Proposed Criteria
{criteria_json}

## User Feedback

What they want to change:
{feedback}

What a successful output looks like:
{expectations}

Critical mistakes to penalize:
{critical_mistakes}

Additional context:
{additional_context}

## Task
Produce a refined "proposed_criteria" JSON object that incorporates the \
user's feedback. The structure must match:

{{
  "structure_weight": <int 0-30>,
  "fields": {{
    "<field_name>": {{
      "importance": "critical" or "important" or "minor",
      "partial_credit": true or false,
      "tolerance": <int>,
      "eval_mode": "non_empty" or "skip"
    }}
  }}
}}

Rules:
- Keep the same field names from the original output_schema.
- For enum fields, include "partial_credit".
- For number fields, include "tolerance".
- For text fields, include "eval_mode".
- Adjust importance, tolerances, and settings based on what the user told you.
- If the user said a field doesn't matter, set importance to "minor".
- If the user emphasized a field, set importance to "critical".
- If the user wants strict scoring, reduce tolerances and disable partial credit.
- If the user wants lenient scoring, increase tolerances and enable partial credit.

Return ONLY the JSON object. No markdown fences, no commentary.
"""
