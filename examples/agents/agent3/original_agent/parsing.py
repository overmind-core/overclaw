"""Format user input and parse model output."""

import json


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
