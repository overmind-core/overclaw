"""Prompts for ``overmind.setup.policy_generator``.

The five public prompts below all walk the LLM through the same canonical
two-layer policy format.  Rather than repeating the Markdown skeleton and the
JSON schema five times verbatim, we keep them once in module-level constants
and compose each prompt from those building blocks.

Constants
---------
:data:`_POLICY_MD_SKELETON_HEADERS`
    Headers-only Markdown skeleton used by every prompt that doesn't need
    inline guidance.
:data:`_POLICY_MD_SKELETON_DETAILED`
    The detailed skeleton with section-level guidance, used only by
    :data:`POLICY_GENERATION_PROMPT` which conducts the elicitation from
    scratch.
:data:`_POLICY_JSON_SCHEMA`
    Shared JSON schema block — appended at the end of every prompt.
"""

_POLICY_MD_SKELETON_HEADERS = """\
# Agent Policy: {agent_name}

## 1. Domain Knowledge

### 1.1 Purpose & Context
### 1.2 Domain Rules
### 1.3 Domain Edge Cases
### 1.4 Terminology & Definitions

## 2. Agent Behavior

### 2.1 Output Constraints
### 2.2 Tool Usage
### 2.3 Decision Mapping
### 2.4 Quality Expectations
"""


_POLICY_MD_SKELETON_DETAILED = """\
# Agent Policy: {agent_name}

## 1. Domain Knowledge

### 1.1 Purpose & Context
One paragraph describing the real-world domain this agent operates in, the \
decisions it makes, and who consumes its output.

### 1.2 Domain Rules
Numbered list of concrete business rules. These come primarily from the \
user's input — they are the ground truth. Each rule should be specific and \
testable. Include thresholds, heuristics, and conditional logic.

### 1.3 Domain Edge Cases
Markdown table with columns: Scenario | Correct Handling
Include the user's edge cases plus 2-3 inferred from the agent's logic.

### 1.4 Terminology & Definitions
Define key terms, categories, score ranges, or domain-specific vocabulary \
that the agent must understand.

## 2. Agent Behavior

### 2.1 Output Constraints
Bulleted list of hard schema constraints (field types, valid enums, ranges).

### 2.2 Tool Usage
Which tools must be called, expected ordering, and how outputs chain between \
tools. Note any data dependencies (e.g., Tool A's output feeds Tool B).

### 2.3 Decision Mapping
How domain rules map to output fields. E.g., score ranges → category values, \
signal combinations → recommended actions.

### 2.4 Quality Expectations
What "good" output looks like — internal consistency, reasoning depth, \
calibration guidance.
"""


_POLICY_JSON_SCHEMA = """\
{{
  "purpose": "<one sentence>",
  "domain_rules": [
    "<business rule 1>",
    "<business rule 2>"
  ],
  "domain_edge_cases": [
    {{"scenario": "<description>", "correct_handling": "<what should happen>"}}
  ],
  "terminology": {{
    "<term>": "<definition>"
  }},
  "output_constraints": [
    "<constraint 1>"
  ],
  "tool_requirements": [
    "<requirement 1>"
  ],
  "decision_mapping": [
    "<mapping rule 1>"
  ],
  "quality_expectations": [
    "<expectation 1>"
  ]
}}\
"""


_POLICY_JSON_SCHEMA_COMPACT = """\
{{
  "purpose": "<one sentence>",
  "domain_rules": ["<rule>"],
  "domain_edge_cases": [{{"scenario": "<desc>", "correct_handling": "<behaviour>"}}],
  "terminology": {{}},
  "output_constraints": ["<constraint>"],
  "tool_requirements": ["<requirement>"],
  "decision_mapping": ["<mapping>"],
  "quality_expectations": ["<expectation>"]
}}\
"""


_POLICY_JSON_SCHEMA_COMPACT_INFERRED = """\
{{
  "purpose": "<one sentence>",
  "domain_rules": ["<rule> (inferred)"],
  "domain_edge_cases": [{{"scenario": "<desc>", "correct_handling": "<behaviour> (inferred)"}}],
  "terminology": {{}},
  "output_constraints": ["<constraint>"],
  "tool_requirements": ["<requirement>"],
  "decision_mapping": ["<mapping>"],
  "quality_expectations": ["<expectation>"]
}}\
"""


POLICY_GENERATION_PROMPT = f"""\
You are an expert at defining evaluation policies for AI agents. Given the \
agent analysis and user-provided domain knowledge, produce a **two-layer** \
policy: domain knowledge (the real-world rules) and agent behaviour (how the \
agent should implement them).

## Agent Analysis
{{analysis_json}}

## User-Provided Domain Knowledge

### Business rules and decision logic:
{{decision_rules}}

### Domain-specific mistakes that are unacceptable:
{{hard_constraints}}

### Real-world edge cases and correct handling:
{{edge_cases}}

### Key terminology or definitions:
{{terminology}}

## Task

Produce TWO outputs, clearly separated.

**FIRST**, output a Markdown policy document inside a fenced block. Follow \
this exact structure:

```markdown
{_POLICY_MD_SKELETON_DETAILED}\
```

**SECOND**, output a JSON block with the machine-readable summary:

```json
{_POLICY_JSON_SCHEMA}
```

Rules:
- Domain rules must come from the user's input — they are ground truth, not \
inferred from code.
- Output constraints and tool requirements come from the agent analysis.
- Decision mapping bridges domain rules to agent output fields.
- Edge cases should cover the user's scenarios plus 2-3 inferred from code.
- Keep the Markdown document concise (300-600 words).
- The JSON must be parseable and consistent with the Markdown.
"""

POLICY_FROM_DOCUMENT_PROMPT = f"""\
You are an expert at structuring AI agent policies. The user has provided an \
existing policy or domain document. Restructure it into the canonical \
two-layer format: domain knowledge and agent behaviour.

## Agent Analysis
{{analysis_json}}

## User's Policy Document
{{user_document}}

## Task

Produce TWO outputs.

**FIRST**, a restructured Markdown policy following the two-layer format:

```markdown
{_POLICY_MD_SKELETON_HEADERS}\
```

**SECOND**, a JSON summary:

```json
{_POLICY_JSON_SCHEMA_COMPACT}
```

Rules:
- Extract real-world business rules, edge cases, and terminology into Section 1 \
(Domain Knowledge). This is the most important section — preserve all \
substantive rules from the original document.
- Extract schema constraints, tool-calling patterns, and quality heuristics \
into Section 2 (Agent Behavior) — infer these from the agent analysis if the \
document doesn't cover them.
- Keep the restructured document concise (300-600 words) while preserving all \
substantive rules from the original.
- Discard organizational boilerplate but keep every testable rule.
"""

POLICY_FROM_CODE_PROMPT = f"""\
You are an expert at inferring evaluation policies from AI agent code. The \
user has not provided explicit policies, so infer reasonable defaults from \
the agent's logic, tool definitions, and output schema.

## Agent Analysis
{{analysis_json}}

## Agent Code
{{agent_code_section}}

## Task

Produce TWO outputs: a Markdown policy and a JSON summary. Infer rules from:
- The system prompt instructions → domain rules and terminology
- Tool parameter constraints and descriptions → tool requirements
- Output field types and their relationships → output constraints and decision mapping
- Any validation or post-processing logic → quality expectations

```markdown
# Agent Policy: {{agent_name}}

## 1. Domain Knowledge

### 1.1 Purpose & Context
### 1.2 Domain Rules
Mark each rule with (inferred) so the user knows to verify.
### 1.3 Domain Edge Cases
Mark inferred cases with (inferred).
### 1.4 Terminology & Definitions

## 2. Agent Behavior

### 2.1 Output Constraints
### 2.2 Tool Usage
### 2.3 Decision Mapping
### 2.4 Quality Expectations
```

```json
{_POLICY_JSON_SCHEMA_COMPACT_INFERRED}
```

Be conservative — only include domain rules you can clearly derive from the \
code. Mark ALL inferred domain rules with "(inferred)" so the user knows to \
verify them. Agent behaviour rules derived from code don't need the marker. \
Keep the document to 200-400 words.
"""

POLICY_IMPROVE_PROMPT = f"""\
You are an expert at evaluating and improving AI agent policies. The user has \
provided an existing policy document. Compare it against the agent's actual \
code and analysis to identify gaps, inconsistencies, or improvements.

## Agent Analysis
{{analysis_json}}

## Agent Code
{{agent_code_section}}

## User's Existing Policy
{{existing_policy}}

## Task

Analyze the existing policy against the agent code and produce an IMPROVED \
version using the two-layer format.

Improvements should:
- Preserve all valid domain rules from the original
- Separate domain knowledge (Section 1) from agent behaviour (Section 2)
- Add domain rules visible in the system prompt that the policy doesn't mention
- Add output constraints visible in the code (valid enum values, score ranges)
- Add tool-chaining requirements from the agent's logic
- Add edge cases the code handles but the policy doesn't cover
- Remove or correct rules that contradict the code
- Improve vague rules to be more specific and testable

Produce THREE outputs.

**FIRST**, a brief summary of what changed and why (2-5 bullet points):

```changes
- <what changed and why>
```

**SECOND**, the improved Markdown policy inside a fenced block:

```markdown
{_POLICY_MD_SKELETON_HEADERS}\
```

**THIRD**, the JSON summary:

```json
{_POLICY_JSON_SCHEMA_COMPACT}
```

Rules:
- Be conservative — don't rewrite rules that are already good.
- Mark newly added rules with "(added)" so the user sees what's new.
- Domain rules from the user's original document are ground truth — don't \
remove them unless they directly contradict the code.
- Keep the document concise (300-600 words).
- If the existing policy is already excellent, say so and make minimal changes.
"""

POLICY_REFINE_PROMPT = f"""\
You are refining an existing agent policy based on user feedback.

## Agent Analysis
{{analysis_json}}

## Current Policy (Markdown)
{{current_md}}

## Current Policy (Structured)
{{current_data_json}}

## User Feedback
What they want to change:
{{feedback}}

Additional domain rules or edge cases:
{{additions}}

## Task

Produce the updated policy in TWO outputs, maintaining the two-layer format.

**FIRST**, the revised Markdown policy inside a fenced block:

```markdown
{_POLICY_MD_SKELETON_HEADERS}\
```

**SECOND**, the updated JSON summary:

```json
{_POLICY_JSON_SCHEMA_COMPACT}
```

Rules:
- Incorporate the user's feedback faithfully.
- Preserve existing rules that the user did not ask to change.
- Keep the document concise (300-600 words).
- User feedback about domain rules goes into Section 1.
- User feedback about agent behaviour goes into Section 2.
"""
