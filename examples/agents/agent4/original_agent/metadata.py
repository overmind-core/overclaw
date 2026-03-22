"""Agent metadata for registry and tooling."""

AGENT_DESCRIPTION = """\
Real Estate Property Valuation Agent: Takes property attributes (address,
property_type, square_footage, bedrooms, bathrooms, year_built,
lot_size_sqft) and returns a structured JSON valuation with estimated_value,
confidence_range_low/high, price_per_sqft, market_condition,
comparable_properties_used, and reasoning. Uses tools to look up the subject
property, comparable sales, and neighborhood market trends.
"""
