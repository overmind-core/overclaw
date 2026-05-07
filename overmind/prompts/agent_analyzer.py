"""Prompts for ``overmind.setup.agent_analyzer``."""

ANALYSIS_PROMPT = """\
You are analyzing a Python AI agent to understand its input/output contract, \
tool orchestration, and propose evaluation criteria.

Agent source code:
{agent_code_section}

The agent's entry function is `{entrypoint_fn}`. The input schema MUST be derived
from this function's **parameter list** — each parameter becomes a field in
`input_schema`. Do NOT infer inputs from the UI layer, internal helper functions,
or Streamlit widgets. Only the parameters of `{entrypoint_fn}()` matter.

CRITICAL: The runner calls the agent via `{entrypoint_fn}(**input_dict)` when
the function has multiple parameters. This means the `input_schema` field names
MUST exactly match the function's parameter names, and test-case inputs will
use these names as dict keys. Getting this wrong causes runtime crashes.

The `output_schema` describes what the agent RETURNS — this may be a structured
JSON object (dict with typed fields) or a plain string/markdown. Derive the
output structure from the function's return type and actual return statements
in the code.

Return a JSON object with this exact structure:
{{
  "description": "One paragraph describing what this agent does and its purpose",
  "input_schema": {{
    "field_name": {{"type": "string|number|boolean|object|array", "description": "what this parameter represents"}}
  }},
  "output_schema": {{
    "field_name": {{
      "type": "enum or number or text or boolean",
      "description": "what this field represents",
      "values": ["list", "of", "valid", "values"],
      "range": [0, 100],
      "optional": false
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
  "scope": {{
    "optimizable_paths": ["glob patterns relative to project root for files the optimizer may edit"],
    "context_paths": ["glob patterns for read-only context files (prompts, schemas) not in import closure"],
    "exclude_paths": ["glob patterns to skip entirely (tests, third-party vendored code, infra)"]
  }},
  "optimizable_elements": ["element1", "element2"],
  "fixed_elements": ["element1", "element2"]
}}

Rules for output_schema:
- Use "enum" for fields with a known set of valid string values. Include ALL valid \
values in "values".
- Use "number" for numeric fields. Include the expected range in "range".
- Use "text" for free-form string fields. Omit "values" and "range".
- Use "boolean" for true/false fields. Omit "values" and "range".
- Set "optional": true for any field that is only populated on SOME code paths \
(e.g. an "error_message" field that only appears when status="error", or success-only \
fields that are absent when the agent returns an error object). This is critical: if \
the agent returns a discriminated union keyed on a status/kind/type field, every \
field OTHER than the discriminator itself should be marked optional. Default is false \
(field is always present).

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

Rules for scope (critical for large repos):
- optimizable_paths: Files that materially affect LLM behaviour — system prompts, \
tool description/schema modules, orchestration around `{entrypoint_fn}()`, model routing. \
Use tight globs (e.g. ``myagent/prompts/**/*.py``). Aim for fewer than 25 patterns; prefer \
directories that hold prompts and agent config over the whole package tree.
- IMPORTANT — distinguish "the agent" from "third-party libraries":
  * If a top-level package directory sits next to `{entrypoint_fn}`'s file inside the project \
root and is imported by the entrypoint (e.g. `import myproj` where `myproj/` is a sibling \
folder), treat it as PART OF THE AGENT and include `myproj/**/*.py` in optimizable_paths. \
The presence of `pyproject.toml`, `setup.py`, `LICENSE`, or `.egg-info/` at the same level \
does NOT make it third-party — many agents are structured as installable local packages.
  * Only treat a directory as third-party / vendored (and exclude it) when it is clearly a \
copied dependency the user does not own — e.g. it lives under `vendor/`, `third_party/`, \
`site-packages/`, `node_modules/`, or carries a NOTICE/COPYING/upstream-readme indicating it \
is an unmodified upstream snapshot.
  * When in doubt about a sibling package, prefer INCLUDING it in optimizable_paths. The \
register-agent / generate-policy step will surface these to the user for confirmation; a \
silent exclude is worse than an over-broad include.
- context_paths: Important read-only context (eval templates, JSON schemas, README, \
pyproject.toml) the optimizer should see but must not edit. Omit if empty.
- exclude_paths: Tests, benchmarks, docs, examples, scripts, docker/k8s, web servers, \
database adapters, true third-party vendored trees, build artefacts (``*.egg-info``, \
``__pycache__``, ``uv.lock``, ``poetry.lock``). Be aggressive about *infra*, but never \
exclude a sibling package that the entrypoint imports.

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
