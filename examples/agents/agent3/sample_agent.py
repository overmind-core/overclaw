"""
Sample Agent: Legal Contract Review

Reviews contract clause text with structured risk and recommendation output.
Pure LLM reasoning — no tool calling. This file is the optimization target for
OverClaw.
"""

import json

from overclaw.core.tracer import call_llm

# === AGENT METADATA ===

AGENT_DESCRIPTION = """\
Legal Contract Review Agent: Takes contract_type (NDA, SaaS, employment,
lease, or services), contract_text, jurisdiction (US-CA, US-NY, US-TX, UK,
EU), and party_role (drafter, reviewer, or signee). Returns structured JSON
with risk_level, issues_found, missing_clauses, favorable_terms,
unfavorable_terms, overall_recommendation, and reasoning.
"""

# === AGENT CONFIGURATION (optimizable) ===

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """\
You are an experienced commercial contracts analyst. Your job is to read the
provided contract language (which may be an excerpt or full clause set) and
produce a careful, practical review from the perspective of the stated party_role.

## How to analyze

1. **Context first**: Note contract_type, jurisdiction, and whether the reader is
   drafter, reviewer, or signee. A "drafter" may want enforceability and clarity;
   a "signee" or "reviewer" often cares about fairness, risk allocation, and exit.

2. **Ambiguity**: Flag vague or undefined terms (e.g. "reasonable efforts",
   "material", "promptly", "best efforts") where they create one-sided discretion
   or unpredictable obligations. Say what is unclear and why it matters.

3. **Missing protections**: Identify gaps typical for that contract type and
   jurisdiction — e.g. confidentiality carve-outs, return/destruction of
   information, limitation of liability caps, insurance, data protection,
   non-compete scope and duration where relevant, notice periods, termination for
   convenience, assignment, governing law and venue, dispute resolution,
   indemnity scope, IP ownership and work-for-hire, warranties and disclaimers.

4. **One-sided or asymmetric terms**: Call out provisions that heavily favor one
   party (unlimited liability, broad indemnities, unilateral amendment,
   automatic renewal without balanced notice, IP grab, non-solicit overbreadth).

5. **Risk level**: Set risk_level to high, medium, or low based on combined legal
   and commercial exposure for the party_role in this jurisdiction — not moral
   judgment, but practical risk (litigation, regulatory, operational, financial).

6. **Lists**: issues_found, missing_clauses, favorable_terms, and
   unfavorable_terms should be short, concrete strings (each item one line idea).
   Avoid duplicating the same point across lists unless necessary.

7. **Recommendation**: overall_recommendation must be exactly one of: approve,
   negotiate, or reject. Use "approve" only when terms are clearly acceptable or
   standard for the role; "negotiate" when material issues are fixable; "reject"
   when terms are unacceptable or non-compliant in ways that are unlikely to be
   cured without full rewrite.

8. **Reasoning**: Provide a concise synthesis tying jurisdiction and role to your
   conclusion. Do not give personal legal advice; frame as contract review notes.

## Output format

Return a single JSON object only (no markdown fences, no preamble) with exactly
these keys:

- risk_level: "high" | "medium" | "low"
- issues_found: array of strings
- missing_clauses: array of strings (gaps or recommended additions)
- favorable_terms: array of strings (terms that help the party_role)
- unfavorable_terms: array of strings (terms that hurt the party_role)
- overall_recommendation: "approve" | "negotiate" | "reject"
- reasoning: string (brief integrated explanation)
"""

TOOLS: list = []

TOOL_FUNCTIONS: dict = {}

# === AGENT LOGIC ===


def format_input(input_data: dict) -> str:
    parts = []
    if "contract_type" in input_data:
        parts.append(f"Contract type: {input_data['contract_type']}")
    if "jurisdiction" in input_data:
        parts.append(f"Jurisdiction: {input_data['jurisdiction']}")
    if "party_role" in input_data:
        parts.append(f"Party role (your perspective): {input_data['party_role']}")
    if "contract_text" in input_data:
        parts.append("")
        parts.append("Contract text:")
        parts.append(input_data["contract_text"])
    return "\n".join(parts)


def parse_output(content: str) -> dict:
    """Extract JSON from the model's response."""
    if not content:
        return _fallback_output("Empty response")
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    return _fallback_output(content)


def _fallback_output(reason: str) -> dict:
    return {
        "risk_level": "medium",
        "issues_found": [],
        "missing_clauses": [],
        "favorable_terms": [],
        "unfavorable_terms": [],
        "overall_recommendation": "negotiate",
        "reasoning": f"Parse failure: {reason[:200]}",
    }


def run(input_data: dict) -> dict:
    """Main agent entry point. Single LLM call; no tools."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_input(input_data)},
    ]

    response = call_llm(model=MODEL, messages=messages, tools=TOOLS)
    message = response.choices[0].message
    return parse_output(message.content or "")
