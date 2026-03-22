"""Format user input and parse model output."""

import json


def format_input(input_data: dict) -> str:
    parts = []
    if "address" in input_data:
        parts.append(f"Address: {input_data['address']}")
    if "property_type" in input_data:
        parts.append(f"Property type: {input_data['property_type']}")
    if "square_footage" in input_data:
        parts.append(f"Square footage: {input_data['square_footage']}")
    if "bedrooms" in input_data:
        parts.append(f"Bedrooms: {input_data['bedrooms']}")
    if "bathrooms" in input_data:
        parts.append(f"Bathrooms: {input_data['bathrooms']}")
    if "year_built" in input_data:
        parts.append(f"Year built: {input_data['year_built']}")
    if "lot_size_sqft" in input_data:
        parts.append(f"Lot size (sqft): {input_data['lot_size_sqft']}")
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
        "estimated_value": 0,
        "confidence_range_low": 0,
        "confidence_range_high": 0,
        "price_per_sqft": 0.0,
        "market_condition": "unknown",
        "comparable_properties_used": 0,
        "reasoning": f"Parse failure: {reason[:200]}",
    }
