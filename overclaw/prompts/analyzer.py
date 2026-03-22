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
        "modifying the {entrypoint_fn}() function (tool call ordering, "
        "post-processing, validation, retry logic)"
    ),
    "format_input": "restructuring how input data is formatted for the LLM",
    "system_prompt": "refining system prompt instructions (but keep it concise)",
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

2. **DO NOT TAMPER WITH THE AGENT PIPELINE (CRITICAL)** — The agent's existing \
call_llm → tool_calls → call_llm loop is the pipeline. You must NOT:
   - Inject, fabricate, or synthetically insert tool calls outside that loop.
   - Inject synthetic "user" or "assistant" messages into the conversation after \
the loop finishes (e.g., adding a "summary" user message to force re-scoring).
   - Add extra call_llm calls after the main loop to "re-evaluate" or "re-score".
   - Add code that tracks tool results in order to build post-hoc scoring prompts.
   - Pre-call tools before the LLM loop and stitch results into the conversation.
   - Add helper functions (e.g., _inject_tool_call, _get_called_tools) that \
manipulate the message list outside the normal LLM interaction loop.
   If the LLM isn't calling a tool or isn't scoring correctly, fix the \
**system prompt** or **tool descriptions** so it behaves correctly on its own. \
The LLM must make all decisions organically through its normal tool-calling loop.

3. **PROMPT BLOAT** — The system prompt is already {prompt_char_count} chars. \
If the system prompt has grown significantly from the original, consider \
SIMPLIFYING it rather than piling on more rules. Prompt changes are fine when \
they add genuinely missing instructions, but avoid case-specific decision rules.

4. **CHANGE PRIORITY** — Prefer changes in this order, but use your judgment — \
combining changes across multiple targets is fine when they reinforce each other:
   a. **Tool descriptions** — improve parameter descriptions, add constraints, \
clarify expected values.
   b. **format_input** — restructure how input data is presented to the LLM.
   c. **System prompt** — add or clarify instructions that help the LLM make \
better decisions. Avoid case-specific rules.
   d. **Agent logic** — ONLY for lightweight changes: enforce tool call ordering \
within the existing loop, add output validation/normalization. Do NOT add new \
helper functions, do NOT add code that manipulates the message list, do NOT add \
extra LLM calls.
   e. **Model** — only if the current model clearly lacks capability.

5. **CONSERVATISM** — Suggest 1–3 targeted changes, not a complete rewrite.

6. **POLICY COMPLIANCE** — If an Agent Policy section is provided above, \
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
      "target": "system_prompt|tool_description|format_input|agent_logic|model",
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
- Do NOT add deterministic post-processing that overrides the LLM's judgment. \
Post-processing should ONLY validate format (enum enforcement, type coercion, \
field presence) — NOT substitute hardcoded decisions for the LLM's analysis. \
For example, "if field == X then set other_field = Y" is an override and is forbidden.
- Do NOT introduce hardcoded numeric thresholds or keyword pattern lists \
derived from specific test cases. Improvements must generalize to unseen inputs.
- If a change targets tool_description, modify the TOOLS list (e.g., improve \
parameter descriptions, add enum values, clarify usage).
- If a change targets format_input, modify the format_input() function.
- If a change targets agent_logic, ONLY make lightweight changes within the \
existing `{entrypoint_fn}()` function: enforce tool call ordering within the \
existing loop, add output validation/normalization.
- **DO NOT TAMPER WITH THE AGENT PIPELINE** — The existing call_llm → tool_calls \
→ call_llm loop must stay intact. Specifically do NOT:
  - Add new helper functions that manipulate the message list.
  - Inject synthetic user/assistant/tool messages outside the normal loop.
  - Add extra call_llm calls after the main loop (e.g., "re-score" or "summarize").
  - Pre-call tools before the LLM loop and inject results into messages.
  - Track tool results in dicts/lists to build post-hoc prompts.
  If the LLM isn't calling tools or scoring correctly, fix the system prompt \
or tool descriptions — not the pipeline code.
- Keep the agent entry function named `{entrypoint_fn}` with a compatible signature \
(receives the input dict, returns a dict).

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
numeric thresholds derived from the test data.
2. **DO NOT TAMPER WITH THE AGENT PIPELINE** — The existing call_llm → tool_calls \
→ call_llm loop must stay intact. Do NOT inject synthetic messages (user, \
assistant, or tool) outside the normal loop. Do NOT add extra call_llm calls \
after the loop to "re-score" or "summarize". Do NOT pre-call tools or track tool \
results to build post-hoc prompts. Do NOT add helper functions that manipulate \
the message list. If the LLM isn't calling tools or scoring correctly, fix the \
system prompt or tool descriptions instead.
3. **PROMPT BLOAT** — Do NOT keep adding rules to SYSTEM_PROMPT. Prefer changes \
to tool descriptions, format_input, and agent_logic over prompt expansion.
4. **FOCUS** — Concentrate on **{weakest_dimension}**.
5. **CONSERVATISM** — Make 1–3 targeted changes, at least one NOT targeting \
the system prompt. Minimize new conditional branches in post-processing.
6. **POLICY COMPLIANCE** — If an Agent Policy section is provided above, \
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
