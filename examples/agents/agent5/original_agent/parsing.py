"""Format user input and parse model output."""

import json


def format_input(input_data: dict) -> str:
    parts = []
    if "ticket_subject" in input_data:
        parts.append(f"Subject: {input_data['ticket_subject']}")
    if "ticket_body" in input_data:
        parts.append(f"Body: {input_data['ticket_body']}")
    if "customer_tier" in input_data:
        parts.append(f"Customer tier: {input_data['customer_tier']}")
    if "product_area" in input_data:
        parts.append(f"Product area: {input_data['product_area']}")
    if "previous_tickets_count" in input_data:
        parts.append(
            f"Previous tickets (count): {input_data['previous_tickets_count']}"
        )
    return "\n".join(parts)


def parse_output(content: str) -> dict:
    """Extract JSON from the model's response."""
    if not content:
        return _fallback_output("Empty response")
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(content[start:end])
            if isinstance(parsed, dict):
                return _normalize_output(parsed)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return _fallback_output(content)


def _normalize_output(d: dict) -> dict:
    """Ensure required keys exist with sane types."""
    out = {
        "category": str(d.get("category", "how_to")),
        "priority": str(d.get("priority", "P3")),
        "assigned_team": str(d.get("assigned_team", "tier1_support")),
        "sentiment": str(d.get("sentiment", "neutral")),
        "estimated_resolution_time": str(
            d.get("estimated_resolution_time", "1-3 days")
        ),
        "auto_response_suggested": bool(d.get("auto_response_suggested", False)),
        "reasoning": str(d.get("reasoning", "")),
    }
    return out


def _fallback_output(reason: str) -> dict:
    return {
        "category": "how_to",
        "priority": "P3",
        "assigned_team": "tier1_support",
        "sentiment": "neutral",
        "estimated_resolution_time": "1-3 days",
        "auto_response_suggested": False,
        "reasoning": f"Parse failure: {reason[:200]}",
    }
