"""Prompts for ``overclaw.optimize.analyzer``.

Supports both single-file and multi-file (bundle) agent optimization.
When ``{agent_code_section}`` is present, it replaces the old
``{agent_code}`` placeholder with either a single code block or the
full virtual bundle with whole-file sections.
"""

DIAGNOSIS_SYSTEM_PROMPT = """\
You are an expert AI agent debugger. You analyze per-test-case \
performance and tool usage to produce precise, actionable diagnoses.

## Evaluation Criteria & Scoring Mechanics
{scoring_mechanics}

## Modifiable Elements
{optimizable_elements}

## Fixed Elements (DO NOT modify)
{fixed_elements}\
"""

FOCUS_LABELS = {
    "tool_description": "improving tool parameter descriptions and schemas",
    "agent_logic": (
        "modifying the {entrypoint_fn}() function (control flow, "
        "orchestration, validation, pre/post-processing, retry logic)"
    ),
    "format_input": "restructuring how input data is formatted for the LLM",
    "system_prompt": "refining system prompt instructions (but keep it concise)",
    "tool_implementation": (
        "improving tool execution logic, fixing tool functions in supporting "
        "modules, adding data processing helpers"
    ),
    "helper_module": (
        "adding or modifying utility functions in supporting files "
        "(data validation, parsing, transformation)"
    ),
    "error_handling": (
        "adding retry logic, fallback strategies, input validation, "
        "and graceful error recovery"
    ),
}

DIAGNOSIS_FOCUS_DIRECTIVE = (
    "\n\n**FOCUS CONSTRAINT**: Your primary change MUST target "
    "**{focus_area}** — specifically, {focus_desc}. "
    "You may include secondary changes to other targets, but the main "
    "improvement must come from {focus_area}."
)

CODEGEN_FOCUS_DIRECTIVE = (
    "\n\n**FOCUS PRIORITY**: Your primary change MUST target "
    "**{focus_area}** — specifically, {focus_desc}. "
    "Apply the diagnosis changes that target {focus_area} first, "
    "and include secondary changes only if they support the primary focus."
)

# ---------------------------------------------------------------------------
# Output format instructions injected depending on single/multi mode
# ---------------------------------------------------------------------------

_SINGLE_FILE_OUTPUT_INSTRUCTION = """\
Output the complete improved agent file inside a code fence:
```python
<entire agent file>
```"""

_BUNDLE_OUTPUT_INSTRUCTION = """\
For each file you modify, output the COMPLETE updated file using this exact format:

### FILE: <relative_path>
```python
<complete updated file contents>
```

Rules:
- Output ONLY files you actually changed. Do NOT output unchanged files.
- Do NOT modify files marked [READ-ONLY].
- Each file must be COMPLETE — include ALL imports, functions, classes,
  and constants, not just the parts you changed.
- Each file must be syntactically valid Python.
- Keep function/class signatures compatible unless the diagnosis
  explicitly says to change them.
- Preserve the original indentation style."""

# ---------------------------------------------------------------------------
# Agentic codegen instruction (used by the coding agent instead of the
# single-shot codegen prompt)
# ---------------------------------------------------------------------------

AGENTIC_CODEGEN_INSTRUCTION = """\
You are optimizing an AI agent's code based on analysis of its test performance.

## Diagnosis

The following diagnosis describes what is wrong and what changes to make:

```json
{diagnosis_json}
```

## Constraints

- The entry function `{entrypoint_fn}` MUST remain callable with its current signature.
- Do NOT change any evaluation infrastructure, only the agent's logic.
- Do NOT hardcode responses for specific test inputs or patterns.
- Do NOT add post-processing that recomputes numeric output fields using a
  formula (e.g., value = avg_price * sqft) and overwrites the LLM's output.
  The LLM often produces correct values through judgment (anchoring on
  assessed values, adjusting for conditions, etc.). A mechanical formula will
  destroy those correct outputs. Improve prompt instructions instead.
- Prefer structural improvements (new helpers, better data processing) over
  adding conditional branches.
{policy_constraints_section}

## File context

The entry file is: `{entry_file}`

### Files in this agent
{file_listing}

### Import relationships
{import_graph}

## Task

1. Start by reading the entry file and any supporting files relevant to the
   diagnosis. Understand the codebase architecture before making changes.
2. Apply the changes described in the diagnosis.
3. **Critically**: check whether your changes require corresponding updates in
   other files (updated imports, changed function signatures, new helpers, etc.).
4. If the diagnosis mentions tool implementation issues, find and fix the actual
   tool functions — they may be in supporting modules, not just the entry file.
5. If adding a helper function would be cleaner than inline logic, create it in
   the appropriate module.
6. Verify each edit is correct before moving on.
{focus_directive}"""

AGENTIC_CODEGEN_FOCUS = (
    "\n\n**FOCUS**: Your primary changes MUST target **{focus_area}** — "
    "specifically, {focus_desc}. Apply the diagnosis changes that target "
    "{focus_area} first, and include secondary changes only if they "
    "support the primary focus."
)

# ---------------------------------------------------------------------------
# Failure cluster and component targeting prompt sections
# ---------------------------------------------------------------------------

FAILURE_CLUSTERS_SECTION = """\

## Known Failure Clusters (prioritized by impact)

The system has identified recurring failure patterns across iterations. \
Focus your diagnosis on the highest-priority unresolved clusters.

{formatted_clusters}
"""

COMPONENT_IMPACT_SECTION = """\

## Component Impact Analysis (automated)

Based on failure pattern analysis, these are the estimated impact weights \
for each optimizable component:

{component_lines}

Prioritize your diagnosis toward the highest-impact components above.\
"""

MULTI_FILE_AWARENESS_SECTION = """\

## Multi-File Architecture

This agent spans multiple files. When diagnosing issues:
- Tool implementation bugs may live in supporting modules, not the entry file.
- Data processing issues may require changes to helper functions in other files.
- Consider whether the fix belongs in the entry file or a supporting module.
- Propose changes to the SPECIFIC file where the code lives.
- If a new utility function would improve the code, suggest which file it belongs in.
- When targeting **tool_implementation** or **helper_module**, always specify the \
exact file path in the ``files`` field of your change instructions.
"""

# ---------------------------------------------------------------------------
# Diagnosis prompt
# ---------------------------------------------------------------------------

DIAGNOSIS_PROMPT = """\
You are an expert AI agent debugger. Analyze the agent's per-test-case \
performance and tool usage to produce a precise diagnosis.

## Current Agent Code
{agent_code_section}

## Registered entry function

OverClaw invokes `{entrypoint_fn}(input)` from `{entry_file}` \
(input and return value are dicts). \
When proposing **agent_logic** changes, refer to `{entrypoint_fn}()` explicitly.

## Evaluation Criteria & Scoring Mechanics
{scoring_mechanics}

## Test Case Results (sorted worst → best)
{per_case_results}

## Tool Usage Analysis
{tool_usage_analysis}

## Agent Policy
{policy_context}

## Score Summary
- Average: {avg_score:.1f} / 100
- Weakest dimension: {weakest_dimension} ({weakest_dim_score:.1f} / {weakest_dim_max:.1f})

## Dimension Breakdown
{score_breakdown}

## Optimization History

### Successful changes (build on these):
{successful_changes}

If a successful change shows dimension losses, prioritize recovering those \
dimensions in this iteration without undoing the gains that justified acceptance.

### Failed attempts (DO NOT repeat these patterns):
{failed_attempts}

If a failed attempt shows dimension gains, the underlying approach had merit \
for those dimensions — try to preserve that directional improvement while \
avoiding the regressions that caused rejection. These are dimension-level \
trends indicating structural strengths, NOT signals to add case-specific rules.

## System Prompt Metrics

Current SYSTEM_PROMPT size: **{prompt_char_count}** characters, **{prompt_line_count}** lines.

## Critical Rules

1. **ANTI-OVERFITTING (MOST IMPORTANT)** — Your changes will be tested on cases \
you CANNOT see. The test cases below are only a SUBSET of the full evaluation set.
   - Do NOT hardcode responses for specific test inputs or patterns observed below.
   - Do NOT add hardcoded numeric thresholds, keyword lists, or regex patterns \
derived from the test data.
   - Post-processing for validation/normalization (enum enforcement, type coercion, \
empty-field defaults) is fine. Post-processing that SUBSTITUTES hardcoded decisions \
for the LLM's analysis (e.g., "if X then set field to Y") is overfitting.

2. **NO EVALUATION GAMING (CRITICAL)** — Do NOT game the evaluation system:
   - Do NOT fabricate or inject synthetic tool calls outside the agent's loop.
   - Do NOT inject synthetic "user" or "assistant" messages to trick scoring.
   - Do NOT add extra call_llm calls solely to "re-score" or "re-evaluate" \
after the main loop.
   - Do NOT pre-call tools before the LLM loop and stitch results into the \
conversation to bypass the agent's natural decision-making.
   However, you ARE encouraged to make genuine structural improvements:
   - Adding helper functions for data processing, validation, and transformation.
   - Improving tool implementation logic in supporting modules.
   - Adding pre-processing that enriches input data before the LLM sees it.
   - Adding post-processing that validates and normalizes output structure.
   - Improving error handling and retry logic.

3. **NO DETERMINISTIC OVERRIDE OF LLM OUTPUT (CRITICAL)** — Do NOT add \
post-processing that recomputes numeric output fields using a formula \
(e.g., computing estimated_value = avg_comp_price * sqft) and unconditionally \
overwriting the LLM's output. The LLM often produces correct values through \
judgment — anchoring on assessed values, adjusting for property condition, \
market context, etc. A mechanical formula will destroy those correct outputs. \
Instead, improve the **system prompt instructions** to guide the LLM toward \
better reasoning. Post-processing for **validation** (range checks, ordering \
enforcement, type coercion) is fine — post-processing that **substitutes a \
formula for the LLM's analysis** is harmful.

4. **PROMPT BLOAT** — The system prompt is already {prompt_char_count} chars. \
If the system prompt has grown significantly from the original, consider \
SIMPLIFYING it rather than piling on more rules. Prompt changes are fine when \
they add genuinely missing instructions, but avoid case-specific decision rules.

5. **CHANGE PRIORITY** — Consider changes across the full codebase. Combine \
changes across multiple targets when they reinforce each other:
   a. **Tool descriptions** — improve parameter descriptions, add constraints, \
clarify expected values.
   b. **Tool implementation** — fix bugs or improve logic in tool functions \
(these may live in supporting modules, not the entry file).
   c. **format_input** — restructure how input data is presented to the LLM.
   d. **System prompt** — add or clarify instructions that help the LLM make \
better decisions. Avoid case-specific rules.
   e. **Agent logic** — improve orchestration, add validation, error handling, \
helper functions for data processing. Keep changes purposeful and general.
   f. **Helper modules** — add or modify utility functions in supporting files.
   g. **Model** — only if the current model clearly lacks capability.
   Prefer structural improvements (new functions, better processing pipelines) \
over adding conditional branches.

6. **CONSERVATISM** — Suggest 1–4 targeted changes, not a complete rewrite.

7. **POLICY COMPLIANCE** — If an Agent Policy section is provided above, \
ensure proposed changes align with the stated decision rules and constraints. \
When diagnosing failures, check whether the agent violated policy rules — \
policy violations are high-priority fixes.

{model_change_rule}

## Your Task

Produce a JSON diagnosis:
```json
{{
  "root_cause": "<1-2 sentences: the primary reason for score loss>",
  "failure_patterns": [
    {{"pattern": "<description>", "affected_cases": <count>, "dimension": "<field>"}}
  ],
  "tool_issues": [
    {{"issue": "<description>", "severity": "high|medium|low", \
"fix": "<what to change>"}}
  ],
  "changes": [
    {{
      "target": "system_prompt|tool_description|format_input|agent_logic|tool_implementation|helper_module|error_handling|model",
      "action": "<specific instruction: what to add/remove/modify>",
      "rationale": "<why this will help>",
      "files": ["<relative path(s) of file(s) affected>"]
    }}
  ]
}}
```

Return ONLY the JSON inside a code fence. Be specific — each change instruction \
must be concrete enough that another developer could implement it without guessing.\
"""

# ---------------------------------------------------------------------------
# Code generation prompt
# ---------------------------------------------------------------------------

CODEGEN_PROMPT = """\
You are implementing specific changes to an AI agent based on a diagnosis.

## Current Agent Code
{agent_code_section}

## Registered entry function

The harness calls `{entrypoint_fn}(input)` from `{entry_file}` (dict in, dict out). \
Keep the entry function named `{entrypoint_fn}` unless the diagnosis explicitly \
requires renaming.

## Diagnosis & Change Instructions
{diagnosis_json}

## Modifiable Elements
{optimizable_elements}

## Fixed Elements (DO NOT modify)
{fixed_elements}

## Policy Constraints
{policy_constraints}

## Rules
- Implement the changes listed in the diagnosis. You may include small \
supporting changes (e.g., a prompt clarification that complements a tool \
description fix) if they naturally follow from the diagnosis.
- The code must be syntactically valid and maintain the same interface.
- Do NOT hardcode responses for specific test inputs.
- Do NOT add deterministic post-processing that overrides the LLM's judgment \
with hardcoded decision logic (e.g., "if field == X then set other_field = Y"). \
Validation and normalization (enum enforcement, type coercion, field presence) is fine.
- Do NOT add post-processing that recomputes numeric fields using a formula \
(e.g., value = avg_price * sqft) and overwrites the LLM's output. The LLM \
often produces correct values through judgment; a mechanical formula will \
destroy those correct outputs. Improve prompts/instructions instead.
- Do NOT introduce hardcoded numeric thresholds or keyword pattern lists \
derived from specific test cases. Improvements must generalize to unseen inputs.
- If a change targets tool_description, modify the TOOLS list (e.g., improve \
parameter descriptions, add enum values, clarify usage).
- If a change targets format_input, modify the format_input() function.
- If a change targets tool_implementation, fix/improve the actual tool \
functions — these may live in supporting modules.
- If a change targets agent_logic, improve orchestration, add validation, \
error handling, or helper functions for data processing.
- If a change targets helper_module, add or modify utility functions in \
the appropriate supporting file.
- **NO EVALUATION GAMING** — Do NOT fabricate synthetic tool calls, inject \
synthetic messages to trick scoring, or add extra call_llm calls solely \
to re-score. Genuine structural improvements (helper functions, better \
data processing, error handling) are encouraged.
- Keep the agent entry function named `{entrypoint_fn}` with a compatible signature \
(receives the input dict, returns a dict).
- Prefer structural improvements (new functions, better pipelines) over \
adding conditional branches.

{output_format_instruction}
"""

# ---------------------------------------------------------------------------
# Single-pass prompt
# ---------------------------------------------------------------------------

SINGLE_PASS_PROMPT = """\
You are an expert AI agent optimizer. Analyze performance and produce improved code.

## Current Agent Code
{agent_code_section}

## Evaluation Criteria & Scoring Mechanics
{scoring_mechanics}

## Test Case Results (sorted worst → best)
{per_case_results}

## Tool Usage Analysis
{tool_usage_analysis}

## Agent Policy
{policy_context}

## Score Summary
- Average: {avg_score:.1f} / 100
- Weakest dimension: {weakest_dimension} ({weakest_dim_score:.1f} / {weakest_dim_max:.1f})

## Dimension Breakdown
{score_breakdown}

## Optimization History

### Successful changes (build on these):
{successful_changes}

If a successful change shows dimension losses, prioritize recovering those \
dimensions in this iteration without undoing the gains that justified acceptance.

### Failed attempts (DO NOT repeat these patterns):
{failed_attempts}

If a failed attempt shows dimension gains, the underlying approach had merit \
for those dimensions — try to preserve that directional improvement while \
avoiding the regressions that caused rejection. These are dimension-level \
trends indicating structural strengths, NOT signals to add case-specific rules.

## Modifiable Elements
{optimizable_elements}

## Fixed Elements (DO NOT modify)
{fixed_elements}

## Agent entry point

OverClaw calls `{entrypoint_fn}(input)` from `{entry_file}` (input is a dict; return a dict). \
When changing orchestration, keep this function name and contract unless the diagnosis explicitly \
says otherwise.

## Critical Rules

1. **ANTI-OVERFITTING** — Do NOT hardcode responses for specific test inputs. \
Do NOT add post-processing that overrides the LLM's judgment with hardcoded values. \
Validation (format, types) is fine; decision overrides are NOT. \
You are seeing only a SUBSET of test cases — rules tailored to these specific \
cases WILL FAIL on unseen ones. Do NOT add keyword lists, regex patterns, or \
numeric thresholds derived from the test data. Prefer structural improvements \
(new functions, better processing) over conditional branches.
2. **NO DETERMINISTIC OVERRIDE** — Do NOT add post-processing that recomputes \
numeric output fields using a formula (e.g., value = avg * sqft) and overwrites \
the LLM's output. The LLM often produces correct values through judgment. \
Improve system prompt instructions to guide better LLM reasoning instead.
3. **NO EVALUATION GAMING** — Do NOT fabricate synthetic tool calls, inject \
synthetic messages to trick scoring, or add extra call_llm calls solely to \
re-score. Genuine structural improvements (helper functions, better data \
processing, error handling, tool implementation fixes) are encouraged.
4. **PROMPT BLOAT** — Do NOT keep adding rules to SYSTEM_PROMPT. Prefer changes \
to tool descriptions, tool implementations, format_input, and agent_logic over \
prompt expansion.
5. **FOCUS** — Concentrate on **{weakest_dimension}**.
6. **CONSERVATISM** — Make 1–4 targeted changes, at least one NOT targeting \
the system prompt. Prefer structural improvements over conditional branches.
7. **POLICY COMPLIANCE** — If an Agent Policy section is provided above, \
ensure changes align with stated decision rules and constraints.

{model_change_rule}

## Required Response Format

FIRST, analysis as JSON:
```json
{{
  "analysis": "<root cause>",
  "failure_patterns": ["<pattern 1>"],
  "suggestions": ["<change 1>", "<change 2>"]
}}
```

THEN, apply your changes:
{output_format_instruction}
"""
