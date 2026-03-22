"""Prompts for ``overclaw.setup.agent_analyzer``."""

ANALYSIS_PROMPT = """\
You are analyzing a Python AI agent to understand its input/output contract, \
tool orchestration, and propose evaluation criteria.

Agent source code:
{agent_code_section}

Return a JSON object with this exact structure:
{{
  "description": "One paragraph describing what this agent does and its purpose",
  "input_schema": {{
    "field_name": {{"type": "string", "description": "what this field represents"}}
  }},
  "output_schema": {{
    "field_name": {{
      "type": "enum or number or text or boolean",
      "description": "what this field represents",
      "values": ["list", "of", "valid", "values"],
      "range": [0, 100]
    }}
  }},
  "proposed_criteria": {{
    "structure_weight": 20,
    "fields": {{
      "field_name": {{
        "importance": "critical or important or minor",
        "partial_credit": true,
        "tolerance": 10,
        "eval_mode": "non_empty or skip"
      }}
    }}
  }},
  "tool_analysis": {{
    "tools": {{
      "tool_name": {{
        "description_quality": "good or needs_improvement",
        "issues": ["list of specific issues with the tool definition"],
        "param_constraints": {{
          "param_name": ["list", "of", "valid", "values"]
        }}
      }}
    }},
    "dependencies": [
      {{
        "from_tool": "source_tool_name",
        "from_field": "output_field_name",
        "to_tool": "target_tool_name",
        "to_param": "parameter_name",
        "description": "why this dependency exists"
      }}
    ],
    "expected_tools": ["list of tools that should be called for a typical input"],
    "orchestration_issues": ["any issues with how tools are sequenced or called"]
  }},
  "consistency_rules": [
    {{
      "field_a": "first_field_name",
      "field_b": "second_field_name",
      "type": "correlation",
      "description": "how these fields should relate (e.g., high score = hot category)",
      "penalty": 3.0
    }}
  ],
  "tools_summary": "Brief description of what tools the agent uses and why",
  "decision_logic": "Brief description of the agent's decision-making process",
  "optimizable_elements": ["element1", "element2"],
  "fixed_elements": ["element1", "element2"]
}}

Rules for output_schema:
- Use "enum" for fields with a known set of valid string values. Include ALL valid \
values in "values".
- Use "number" for numeric fields. Include the expected range in "range".
- Use "text" for free-form string fields. Omit "values" and "range".
- Use "boolean" for true/false fields. Omit "values" and "range".

Rules for proposed_criteria:
- Set "importance" to "critical" for primary output fields, "important" for secondary, \
"minor" for supplementary.
- For enum fields: set "partial_credit" to true if a valid-but-wrong value still shows \
the agent is working.
- For number fields: set "tolerance" to a reasonable margin of error.
- For text fields: set "eval_mode" to "non_empty" if presence matters, "skip" if \
informational.

Rules for tool_analysis:
- Examine each tool's parameter definitions. If a parameter accepts enum-like values \
(e.g., company_size should be one of a fixed set), list them in param_constraints.
- Look for data dependencies between tools. If tool B needs output from tool A as an \
argument, list it in dependencies.
- Note if tool descriptions are vague or missing important constraints.
- List ALL tools that should be called for a typical input in expected_tools.

Rules for consistency_rules:
- Identify pairs of output fields that should logically correlate. For example, if \
there's a numeric score and a categorical field, a high score should align with the \
"best" category value. List the FIRST value in enum "values" as the highest/best.
- Set penalty proportional to how egregious the inconsistency would be.

Rules for optimizable_elements vs fixed_elements:
- optimizable_elements: Things the optimizer CAN change to improve performance.
  This MUST include:
  * System prompt / instructions
  * Tool definitions (descriptions, parameter schemas) — NOT their implementations
  * Input formatting functions (e.g., format_input)
  * Agent orchestration logic (the `{entrypoint_fn}()` entry function — tool call ordering, \
post-processing, retry logic, validation steps)
  * Model selection
- fixed_elements: Things that MUST NOT change because they are external \
dependencies or core infrastructure.
  * Tool IMPLEMENTATIONS (the actual Python functions that tools call)
  * Data sources / databases
  * Output parsing logic
  * Import structure and tracing integration
- The `{entrypoint_fn}()` entry function (or equivalent orchestration) should be listed in \
optimizable_elements with a note about what aspects can be changed (e.g., \
"{entrypoint_fn} — agent orchestration: tool call ordering, post-processing, validation").

Return ONLY the JSON object. No markdown fences, no commentary.
"""
