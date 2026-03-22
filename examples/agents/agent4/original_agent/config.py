"""Optimizable agent configuration (model, prompts, tool schemas)."""

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """\
You are a real estate valuation assistant. Estimate fair market value using a
clear multi-step workflow:

1. Call `lookup_property` with the subject address to retrieve any on-record
   details (neighborhood, zip, prior assessments).
2. Call `get_comparable_sales` using the zip code and property type from the
   lookup (or from the user input if the lookup is partial), plus square
   footage when helpful.
3. Call `get_market_trends` for the property's zip code to understand
   appreciation, days on market, inventory, and whether the market is hot,
   balanced, or cool.

Synthesize the tool results into a coherent valuation. Return your final
answer as JSON only, with these exact fields:
- estimated_value (integer, USD)
- confidence_range_low (integer)
- confidence_range_high (integer)
- price_per_sqft (float)
- market_condition (one of: hot, balanced, cool)
- comparable_properties_used (integer, how many comps you relied on)
- reasoning (brief explanation citing tools)
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_property",
            "description": "Look up property details by address in the local property records database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {
                        "type": "string",
                        "description": "Full or partial street address including city, state, and ZIP if known.",
                    },
                },
                "required": ["address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_comparable_sales",
            "description": "Find recent comparable sales near the subject property.",
            "parameters": {
                "type": "object",
                "properties": {
                    "zip_code": {
                        "type": "string",
                        "description": "5-digit ZIP code.",
                    },
                    "property_type": {
                        "type": "string",
                        "description": "Property type: single_family, condo, townhouse, multi_family, or commercial.",
                    },
                    "square_footage": {
                        "type": "integer",
                        "description": "Subject living area in square feet (optional, improves matching).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of comps to return (default 10).",
                    },
                },
                "required": ["zip_code", "property_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_trends",
            "description": "Get market trend statistics for a ZIP code (appreciation, DOM, inventory).",
            "parameters": {
                "type": "object",
                "properties": {
                    "zip_code": {
                        "type": "string",
                        "description": "5-digit ZIP code.",
                    },
                },
                "required": ["zip_code"],
            },
        },
    },
]
