"""
Sample Agent: Real Estate Property Valuation

Multi-step tool calling: lookup property → comparable sales → market trends.
"""

import json
import re

from overclaw.core.tracer import call_llm, call_tool

# === AGENT METADATA ===

AGENT_DESCRIPTION = """\
Real Estate Property Valuation Agent: Takes property attributes (address,
property_type, square_footage, bedrooms, bathrooms, year_built,
lot_size_sqft) and returns a structured JSON valuation with estimated_value,
confidence_range_low/high, price_per_sqft, market_condition,
comparable_properties_used, and reasoning. Uses tools to look up the subject
property, comparable sales, and neighborhood market trends.
"""

# === AGENT CONFIGURATION (optimizable) ===

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

# === TOOL IMPLEMENTATIONS (fixed — do not optimize) ===


def _normalize_addr(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _extract_zip(address: str) -> str | None:
    m = re.search(r"\b(\d{5})(?:-\d{4})?\b", address)
    return m.group(1) if m else None


# ~12 properties across neighborhoods
PROPERTY_DB: dict[str, dict] = {
    "1245 n alpine dr beverly hills ca 90210": {
        "address": "1245 N Alpine Dr, Beverly Hills, CA 90210",
        "neighborhood": "Beverly Hills Flats",
        "zip_code": "90210",
        "property_type": "single_family",
        "square_footage": 6200,
        "bedrooms": 6,
        "bathrooms": 7,
        "year_built": 1992,
        "lot_size_sqft": 22000,
        "last_assessed_value": 4_850_000,
        "notes": "Gated estate; pool; city views.",
    },
    "420 ellis st unit 504 san francisco ca 94102": {
        "address": "420 Ellis St Unit 504, San Francisco, CA 94102",
        "neighborhood": "Tenderloin edge / Union Sq",
        "zip_code": "94102",
        "property_type": "condo",
        "square_footage": 720,
        "bedrooms": 1,
        "bathrooms": 1,
        "year_built": 2008,
        "lot_size_sqft": 0,
        "last_assessed_value": 685_000,
        "notes": "Starter condo; HOA $520/mo.",
    },
    "200 s michigan ave ste 1 chicago il 60601": {
        "address": "200 S Michigan Ave Ste 1, Chicago, IL 60601",
        "neighborhood": "Loop",
        "zip_code": "60601",
        "property_type": "commercial",
        "square_footage": 11500,
        "bedrooms": 0,
        "bathrooms": 4,
        "year_built": 1924,
        "lot_size_sqft": 8000,
        "last_assessed_value": 3_200_000,
        "notes": "Street-level retail + office; landmark building.",
    },
    "88 president st brooklyn ny 11201": {
        "address": "88 President St, Brooklyn, NY 11201",
        "neighborhood": "Brooklyn Heights",
        "zip_code": "11201",
        "property_type": "multi_family",
        "square_footage": 4800,
        "bedrooms": 12,
        "bathrooms": 8,
        "year_built": 1890,
        "lot_size_sqft": 3200,
        "last_assessed_value": 2_950_000,
        "notes": "Four-unit brownstone; rent roll on file.",
    },
    "77 oak ave cleveland oh 44113": {
        "address": "77 Oak Ave, Cleveland, OH 44113",
        "neighborhood": "Ohio City",
        "zip_code": "44113",
        "property_type": "single_family",
        "square_footage": 1850,
        "bedrooms": 3,
        "bathrooms": 2,
        "year_built": 1925,
        "lot_size_sqft": 5200,
        "last_assessed_value": 95_000,
        "notes": "Fixer-upper; roof and foundation need work.",
    },
    "55 rainey st austin tx 78701": {
        "address": "55 Rainey St, Austin, TX 78701",
        "neighborhood": "Rainey Street",
        "zip_code": "78701",
        "property_type": "single_family",
        "square_footage": 3100,
        "bedrooms": 4,
        "bathrooms": 4,
        "year_built": 2024,
        "lot_size_sqft": 4800,
        "last_assessed_value": 1_420_000,
        "notes": "New construction; smart home package.",
    },
    "9 route 16 whitefield nh 03561": {
        "address": "9 Route 16, Whitefield, NH 03561",
        "neighborhood": "Rural Whitefield",
        "zip_code": "03561",
        "property_type": "single_family",
        "square_footage": 1750,
        "bedrooms": 3,
        "bathrooms": 2,
        "year_built": 1978,
        "lot_size_sqft": 87120,
        "last_assessed_value": 265_000,
        "notes": "Rural; sparse comps within 15 miles.",
    },
    "2100 1st ave n seattle wa 98101": {
        "address": "2100 1st Ave N, Seattle, WA 98101",
        "neighborhood": "Belltown",
        "zip_code": "98101",
        "property_type": "townhouse",
        "square_footage": 2100,
        "bedrooms": 3,
        "bathrooms": 2,
        "year_built": 2016,
        "lot_size_sqft": 1800,
        "last_assessed_value": 1_180_000,
        "notes": "Urban townhouse; walk score high.",
    },
    "400 w high st columbus oh 43215": {
        "address": "400 W High St, Columbus, OH 43215",
        "neighborhood": "Downtown Columbus",
        "zip_code": "43215",
        "property_type": "condo",
        "square_footage": 1100,
        "bedrooms": 2,
        "bathrooms": 2,
        "year_built": 2005,
        "lot_size_sqft": 0,
        "last_assessed_value": 285_000,
        "notes": "Cool market; inventory elevated.",
    },
    "1900 brickell ave ph2 miami fl 33139": {
        "address": "1900 Brickell Ave PH2, Miami, FL 33139",
        "neighborhood": "Brickell",
        "zip_code": "33139",
        "property_type": "condo",
        "square_footage": 3400,
        "bedrooms": 3,
        "bathrooms": 3,
        "year_built": 2019,
        "lot_size_sqft": 0,
        "last_assessed_value": 3_400_000,
        "notes": "Penthouse; bay views.",
    },
    "500 nw burnside st portland or 97201": {
        "address": "500 NW Burnside St, Portland, OR 97201",
        "neighborhood": "Pearl District",
        "zip_code": "97201",
        "property_type": "townhouse",
        "square_footage": 1950,
        "bedrooms": 3,
        "bathrooms": 2,
        "year_built": 2012,
        "lot_size_sqft": 2200,
        "last_assessed_value": 825_000,
        "notes": "Market softening; negative YoY appreciation in trend data.",
    },
    "800 warehouse row dallas tx 75201": {
        "address": "800 Warehouse Row, Dallas, TX 75201",
        "neighborhood": "West End",
        "zip_code": "75201",
        "property_type": "single_family",
        "square_footage": 4200,
        "bedrooms": 4,
        "bathrooms": 3,
        "year_built": 1940,
        "lot_size_sqft": 6000,
        "last_assessed_value": 1_050_000,
        "notes": "Unusual: converted warehouse to residential; exposed brick.",
    },
}


# ~20 comparable sales
COMPARABLE_SALES: list[dict] = [
    {
        "zip_code": "90210",
        "property_type": "single_family",
        "address": "1190 Coldwater Canyon Dr",
        "square_footage": 5800,
        "sale_price": 4_200_000,
        "sale_date": "2024-11-02",
        "price_per_sqft": 724.14,
    },
    {
        "zip_code": "90210",
        "property_type": "single_family",
        "address": "1422 Summitridge Dr",
        "square_footage": 6500,
        "sale_price": 5_100_000,
        "sale_date": "2024-09-18",
        "price_per_sqft": 784.62,
    },
    {
        "zip_code": "94102",
        "property_type": "condo",
        "address": "450 Turk St #301",
        "square_footage": 690,
        "sale_price": 640_000,
        "sale_date": "2025-01-10",
        "price_per_sqft": 927.54,
    },
    {
        "zip_code": "94102",
        "property_type": "condo",
        "address": "188 Minna St #1201",
        "square_footage": 780,
        "sale_price": 715_000,
        "sale_date": "2024-12-05",
        "price_per_sqft": 916.67,
    },
    {
        "zip_code": "60601",
        "property_type": "commercial",
        "address": "180 N Wacker Dr",
        "square_footage": 10000,
        "sale_price": 2_800_000,
        "sale_date": "2024-10-22",
        "price_per_sqft": 280.0,
    },
    {
        "zip_code": "60601",
        "property_type": "commercial",
        "address": "233 S Wacker Dr Ste A",
        "square_footage": 12500,
        "sale_price": 3_400_000,
        "sale_date": "2024-08-30",
        "price_per_sqft": 272.0,
    },
    {
        "zip_code": "11201",
        "property_type": "multi_family",
        "address": "72 Hicks St",
        "square_footage": 4500,
        "sale_price": 2_650_000,
        "sale_date": "2024-11-28",
        "price_per_sqft": 588.89,
    },
    {
        "zip_code": "11201",
        "property_type": "multi_family",
        "address": "110 Columbia Hts",
        "square_footage": 5200,
        "sale_price": 3_050_000,
        "sale_date": "2024-07-14",
        "price_per_sqft": 586.54,
    },
    {
        "zip_code": "44113",
        "property_type": "single_family",
        "address": "61 W 41st St",
        "square_footage": 1700,
        "sale_price": 175_000,
        "sale_date": "2024-12-01",
        "price_per_sqft": 102.94,
    },
    {
        "zip_code": "44113",
        "property_type": "single_family",
        "address": "910 Starkweather Ave",
        "square_footage": 2100,
        "sale_price": 240_000,
        "sale_date": "2024-10-03",
        "price_per_sqft": 114.29,
    },
    {
        "zip_code": "78701",
        "property_type": "single_family",
        "address": "70 Rainey St",
        "square_footage": 3000,
        "sale_price": 1_350_000,
        "sale_date": "2025-02-01",
        "price_per_sqft": 450.0,
    },
    {
        "zip_code": "78701",
        "property_type": "townhouse",
        "address": "210 Lee Barton Dr",
        "square_footage": 2800,
        "sale_price": 1_200_000,
        "sale_date": "2024-11-15",
        "price_per_sqft": 428.57,
    },
    {
        "zip_code": "03561",
        "property_type": "single_family",
        "address": "22 Mountain View Rd",
        "square_footage": 1600,
        "sale_price": 248_000,
        "sale_date": "2024-09-01",
        "price_per_sqft": 155.0,
    },
    {
        "zip_code": "03561",
        "property_type": "single_family",
        "address": "4 Pine Tree Ln",
        "square_footage": 1900,
        "sale_price": 275_000,
        "sale_date": "2024-11-20",
        "price_per_sqft": 144.74,
    },
    {
        "zip_code": "98101",
        "property_type": "townhouse",
        "address": "2200 3rd Ave",
        "square_footage": 2000,
        "sale_price": 1_050_000,
        "sale_date": "2024-12-18",
        "price_per_sqft": 525.0,
    },
    {
        "zip_code": "98101",
        "property_type": "condo",
        "address": "2911 2nd Ave #400",
        "square_footage": 1300,
        "sale_price": 890_000,
        "sale_date": "2025-01-22",
        "price_per_sqft": 684.62,
    },
    {
        "zip_code": "43215",
        "property_type": "condo",
        "address": "120 W Nationwide Blvd #1502",
        "square_footage": 1050,
        "sale_price": 270_000,
        "sale_date": "2024-10-10",
        "price_per_sqft": 257.14,
    },
    {
        "zip_code": "33139",
        "property_type": "condo",
        "address": "200 Biscayne Blvd Way #4301",
        "square_footage": 3200,
        "sale_price": 3_100_000,
        "sale_date": "2024-12-12",
        "price_per_sqft": 968.75,
    },
    {
        "zip_code": "97201",
        "property_type": "townhouse",
        "address": "1230 NW Hoyt St",
        "square_footage": 1800,
        "sale_price": 780_000,
        "sale_date": "2024-11-05",
        "price_per_sqft": 433.33,
    },
    {
        "zip_code": "75201",
        "property_type": "single_family",
        "address": "600 Factory St",
        "square_footage": 3800,
        "sale_price": 980_000,
        "sale_date": "2024-08-18",
        "price_per_sqft": 257.89,
    },
]

# ~8 ZIP codes with rich trend data (others use DEFAULT_MARKET)
MARKET_TRENDS: dict[str, dict] = {
    "90210": {
        "zip_code": "90210",
        "yoy_appreciation_pct": 6.2,
        "median_days_on_market": 28,
        "months_of_inventory": 2.1,
        "market_condition": "hot",
    },
    "94102": {
        "zip_code": "94102",
        "yoy_appreciation_pct": 4.1,
        "median_days_on_market": 35,
        "months_of_inventory": 2.8,
        "market_condition": "hot",
    },
    "60601": {
        "zip_code": "60601",
        "yoy_appreciation_pct": 2.0,
        "median_days_on_market": 52,
        "months_of_inventory": 4.5,
        "market_condition": "balanced",
    },
    "11201": {
        "zip_code": "11201",
        "yoy_appreciation_pct": 3.5,
        "median_days_on_market": 45,
        "months_of_inventory": 3.9,
        "market_condition": "balanced",
    },
    "78701": {
        "zip_code": "78701",
        "yoy_appreciation_pct": 5.8,
        "median_days_on_market": 31,
        "months_of_inventory": 2.4,
        "market_condition": "hot",
    },
    "98101": {
        "zip_code": "98101",
        "yoy_appreciation_pct": 5.1,
        "median_days_on_market": 29,
        "months_of_inventory": 2.3,
        "market_condition": "hot",
    },
    "43215": {
        "zip_code": "43215",
        "yoy_appreciation_pct": -0.5,
        "median_days_on_market": 68,
        "months_of_inventory": 6.2,
        "market_condition": "cool",
    },
    "97201": {
        "zip_code": "97201",
        "yoy_appreciation_pct": -2.3,
        "median_days_on_market": 74,
        "months_of_inventory": 7.1,
        "market_condition": "cool",
    },
}

DEFAULT_MARKET = {
    "zip_code": None,
    "yoy_appreciation_pct": 1.5,
    "median_days_on_market": 55,
    "months_of_inventory": 4.8,
    "market_condition": "balanced",
}


def lookup_property(address: str) -> dict:
    norm = _normalize_addr(address)
    zip_guess = _extract_zip(address)
    # Exact / substring match on DB keys
    for key, rec in PROPERTY_DB.items():
        if key in norm or norm in key:
            return {"found": True, "record": rec}
    # Token overlap scoring
    best_key = None
    best_score = 0
    tokens = set(re.findall(r"[a-z0-9]+", norm))
    for key, rec in PROPERTY_DB.items():
        kt = set(re.findall(r"[a-z0-9]+", key))
        overlap = len(tokens & kt)
        if overlap > best_score:
            best_score = overlap
            best_key = key
    if best_key and best_score >= 3:
        return {"found": True, "record": PROPERTY_DB[best_key], "match": "fuzzy"}
    return {
        "found": False,
        "address_query": address,
        "zip_from_address": zip_guess,
        "hint": "No direct record; use user-provided attributes and comps for the ZIP.",
    }


def get_comparable_sales(
    zip_code: str,
    property_type: str,
    square_footage: int | None = None,
    limit: int = 10,
) -> dict:
    z = re.sub(r"\D", "", zip_code)[:5]
    pt = property_type.lower().strip()
    rows = [
        c for c in COMPARABLE_SALES if c["zip_code"] == z and c["property_type"] == pt
    ]
    if not rows:
        rows = [c for c in COMPARABLE_SALES if c["zip_code"] == z]
    if not rows:
        rows = [c for c in COMPARABLE_SALES if c["property_type"] == pt]
    if square_footage and rows:

        def sqft_dist(r: dict) -> float:
            return abs(r["square_footage"] - square_footage)

        rows = sorted(rows, key=sqft_dist)
    lim = max(1, min(limit, 20))
    return {
        "zip_code": z,
        "property_type": pt,
        "count": len(rows[:lim]),
        "comparables": rows[:lim],
    }


def get_market_trends(zip_code: str) -> dict:
    z = re.sub(r"\D", "", zip_code)[:5]
    base = MARKET_TRENDS.get(z, {**DEFAULT_MARKET})
    out = {**base, "zip_code": z}
    return out


TOOL_FUNCTIONS = {
    "lookup_property": lookup_property,
    "get_comparable_sales": get_comparable_sales,
    "get_market_trends": get_market_trends,
}

# === AGENT LOGIC ===


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


def run(input_data: dict) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_input(input_data)},
    ]
    response = call_llm(model=MODEL, messages=messages, tools=TOOLS)
    message = response.choices[0].message
    max_tool_rounds = 5
    for _ in range(max_tool_rounds):
        if not message.tool_calls:
            break
        assistant_msg = {"role": "assistant", "content": message.content}
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
        messages.append(assistant_msg)
        for tc in message.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}
            if fn_name in TOOL_FUNCTIONS:
                result = call_tool(fn_name, fn_args, TOOL_FUNCTIONS[fn_name])
            else:
                result = {"error": f"Unknown tool: {fn_name}"}
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)}
            )
        response = call_llm(model=MODEL, messages=messages, tools=TOOLS)
        message = response.choices[0].message
    return parse_output(message.content or "")
