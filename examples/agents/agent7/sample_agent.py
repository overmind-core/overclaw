"""
Sample Agent: Agriculture Crop Advisory

Uses soil, weather, and crop reference tools to recommend actions for a field.
"""

import json

from overclaw.core.tracer import call_llm, call_tool

# === AGENT METADATA ===

AGENT_DESCRIPTION = """\
Agriculture Crop Advisory Agent: Takes field context (field_id, location,
current_crop, soil_type, season, acres, irrigation_available) and returns
structured guidance: recommendation, suggested_crops, risk_factors,
expected_yield_rating, fertilizer_recommendation, timing_advice, reasoning.
Uses tools for soil data, regional weather, and crop growing requirements.
"""

# === AGENT CONFIGURATION (optimizable) ===

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """\
You are an agricultural crop advisory assistant. Use the tools to fetch soil
data for the field, weather for the region, and optimal growing conditions for
relevant crops. Combine tool results with the user's stated soil type, season,
acreage, and irrigation availability to give practical advice.

Return your final answer as JSON with exactly these fields:
- recommendation: one of plant, rotate, fertilize, irrigate, harvest, wait
- suggested_crops: array of up to 3 crop name strings, best options first
- risk_factors: array of short strings describing risks or concerns
- expected_yield_rating: one of excellent, good, fair, poor
- fertilizer_recommendation: a concise string (use organic options if context implies organic-only)
- timing_advice: a concise string on when to plant, irrigate, or act
- reasoning: brief explanation tying tools and inputs together
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_soil_data",
            "description": (
                "Return soil composition, pH, moisture, and macronutrients "
                "for a field or location."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field_id": {
                        "type": "string",
                        "description": "Field identifier",
                    },
                    "location": {
                        "type": "string",
                        "description": "Region, state, or area label",
                    },
                },
                "required": ["field_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": (
                "Return short-term weather forecast for a region: temperature, "
                "rain, humidity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "description": "Region or state to look up",
                    },
                },
                "required": ["region"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_crop_database",
            "description": (
                "Look up optimal growing conditions for a crop: temperature "
                "range, pH, water needs, preferred season, soil preference."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "crop_name": {
                        "type": "string",
                        "description": "Crop to look up (e.g., corn, wheat)",
                    },
                },
                "required": ["crop_name"],
            },
        },
    },
]

# === TOOL IMPLEMENTATIONS (fixed — do not optimize) ===

# ~10 fields: varying pH, N, P, K, moisture, composition notes
SOIL_DB: dict[str, dict] = {
    "F-IA-001": {
        "field_id": "F-IA-001",
        "location": "Iowa",
        "composition": "loam",
        "ph": 6.5,
        "moisture_pct": 42,
        "nutrients": {"nitrogen": 45, "phosphorus": 38, "potassium": 52},
        "organic_matter_pct": 4.2,
        "notes": "High organic matter; good structure.",
    },
    "F-TX-DROUGHT": {
        "field_id": "F-TX-DROUGHT",
        "location": "Texas Panhandle",
        "composition": "sandy loam",
        "ph": 7.1,
        "moisture_pct": 18,
        "nutrients": {"nitrogen": 22, "phosphorus": 28, "potassium": 35},
        "organic_matter_pct": 1.8,
        "notes": "Low moisture; drought stress likely without irrigation.",
    },
    "F-MN-DEPLETED": {
        "field_id": "F-MN-DEPLETED",
        "location": "Minnesota",
        "composition": "clay loam",
        "ph": 7.4,
        "moisture_pct": 35,
        "nutrients": {"nitrogen": 12, "phosphorus": 15, "potassium": 20},
        "organic_matter_pct": 2.1,
        "notes": "Heavy corn rotation; NPK depleted — rotation recommended.",
    },
    "F-OR-ORG": {
        "field_id": "F-OR-ORG",
        "location": "Oregon",
        "composition": "silt loam",
        "ph": 6.2,
        "moisture_pct": 48,
        "nutrients": {"nitrogen": 30, "phosphorus": 32, "potassium": 40},
        "organic_matter_pct": 5.5,
        "notes": "Certified organic field; no synthetic inputs on record.",
    },
    "F-AR-RICE": {
        "field_id": "F-AR-RICE",
        "location": "Arkansas Delta",
        "composition": "clay",
        "ph": 5.8,
        "moisture_pct": 55,
        "nutrients": {"nitrogen": 35, "phosphorus": 30, "potassium": 28},
        "organic_matter_pct": 3.0,
        "notes": "Poorly drained; suited to flooded rice culture.",
    },
    "F-CA-VINE": {
        "field_id": "F-CA-VINE",
        "location": "California Central Coast",
        "composition": "loam",
        "ph": 6.8,
        "moisture_pct": 28,
        "nutrients": {"nitrogen": 18, "phosphorus": 22, "potassium": 45},
        "organic_matter_pct": 2.5,
        "notes": "Gravelly subsoil; low vigor target for wine grapes.",
    },
    "F-KS-WHEAT": {
        "field_id": "F-KS-WHEAT",
        "location": "Kansas",
        "composition": "silt",
        "ph": 6.0,
        "moisture_pct": 38,
        "nutrients": {"nitrogen": 40, "phosphorus": 25, "potassium": 42},
        "organic_matter_pct": 2.8,
        "notes": "Winter wheat common; watch spring moisture.",
    },
    "F-GA-PEST": {
        "field_id": "F-GA-PEST",
        "location": "Georgia",
        "composition": "sandy",
        "ph": 5.5,
        "moisture_pct": 33,
        "nutrients": {"nitrogen": 28, "phosphorus": 20, "potassium": 25},
        "organic_matter_pct": 1.5,
        "notes": "Elevated pest pressure reported in adjacent fields.",
    },
    "F-HOBBY-02": {
        "field_id": "F-HOBBY-02",
        "location": "Vermont",
        "composition": "loam",
        "ph": 6.4,
        "moisture_pct": 45,
        "nutrients": {"nitrogen": 32, "phosphorus": 30, "potassium": 36},
        "organic_matter_pct": 4.8,
        "notes": "Small plot; high OM from compost history.",
    },
    "F-COMM-1000": {
        "field_id": "F-COMM-1000",
        "location": "Nebraska",
        "composition": "loam",
        "ph": 6.7,
        "moisture_pct": 40,
        "nutrients": {"nitrogen": 42, "phosphorus": 35, "potassium": 48},
        "organic_matter_pct": 3.5,
        "notes": "Large commercial block; grid soil sampling available.",
    },
}


def _normalize_key(s: str) -> str:
    return s.lower().strip()


def get_soil_data(field_id: str, location: str = "") -> dict:
    fid = field_id.strip()
    if fid in SOIL_DB:
        return SOIL_DB[fid]
    fid_lower = _normalize_key(fid)
    for k, v in SOIL_DB.items():
        if _normalize_key(k) == fid_lower:
            return v
    loc = _normalize_key(location)
    if loc:
        for v in SOIL_DB.values():
            vloc = _normalize_key(v.get("location", ""))
            if loc in vloc or vloc in loc:
                return v
    # Generic fallback by partial field id
    return {
        "field_id": field_id,
        "location": location or "unknown",
        "composition": "unknown",
        "ph": 6.5,
        "moisture_pct": 35,
        "nutrients": {"nitrogen": 30, "phosphorus": 28, "potassium": 32},
        "organic_matter_pct": 3.0,
        "notes": "No exact match; estimated regional average.",
    }


# ~8 regions: temp, rain, humidity (7-day style summary)
WEATHER_DB: dict[str, dict] = {
    "iowa": {
        "region": "Iowa",
        "temp_c_high_low": (18, 6),
        "rain_mm_next_7d": 25,
        "humidity_pct_avg": 62,
        "outlook": "Mild with periodic showers; favorable for spring work.",
    },
    "texas": {
        "region": "Texas Panhandle",
        "temp_c_high_low": (28, 12),
        "rain_mm_next_7d": 3,
        "humidity_pct_avg": 35,
        "outlook": "Hot and dry; irrigation critical for establishment.",
    },
    "minnesota": {
        "region": "Minnesota",
        "temp_c_high_low": (12, 2),
        "rain_mm_next_7d": 15,
        "humidity_pct_avg": 58,
        "outlook": "Cool; delay sensitive planting if frost risk remains.",
    },
    "oregon": {
        "region": "Oregon",
        "temp_c_high_low": (16, 8),
        "rain_mm_next_7d": 40,
        "humidity_pct_avg": 72,
        "outlook": "Wet spring; manage drainage and disease pressure.",
    },
    "arkansas": {
        "region": "Arkansas Delta",
        "temp_c_high_low": (24, 14),
        "rain_mm_next_7d": 30,
        "humidity_pct_avg": 68,
        "outlook": "Warm and humid; good for rice flooding windows.",
    },
    "california": {
        "region": "California Central Coast",
        "temp_c_high_low": (22, 10),
        "rain_mm_next_7d": 8,
        "humidity_pct_avg": 55,
        "outlook": "Mild; limited rain — deficit irrigation typical.",
    },
    "kansas": {
        "region": "Kansas",
        "temp_c_high_low": (14, 4),
        "rain_mm_next_7d": 12,
        "humidity_pct_avg": 52,
        "outlook": "Cool nights; suitable for winter wheat growth stages.",
    },
    "georgia": {
        "region": "Georgia",
        "temp_c_high_low": (26, 14),
        "rain_mm_next_7d": 18,
        "humidity_pct_avg": 70,
        "outlook": "Warm and humid; scout for pests after planting.",
    },
    "nebraska": {
        "region": "Nebraska",
        "temp_c_high_low": (20, 7),
        "rain_mm_next_7d": 20,
        "humidity_pct_avg": 55,
        "outlook": "Variable; suitable for large-scale row crop operations.",
    },
    "arid southwest": {
        "region": "Arid Southwest",
        "temp_c_high_low": (32, 14),
        "rain_mm_next_7d": 0,
        "humidity_pct_avg": 22,
        "outlook": "Extremely dry; water-limited; prioritize drought-tolerant crops.",
    },
}


def get_weather_forecast(region: str) -> dict:
    key = _normalize_key(region)
    for k, v in WEATHER_DB.items():
        if k in key or key in k:
            return v
        reg = _normalize_key(v.get("region", ""))
        if key in reg or reg in key:
            return v
    return {
        "region": region,
        "temp_c_high_low": (24, 10),
        "rain_mm_next_7d": 15,
        "humidity_pct_avg": 50,
        "outlook": "Regional average; verify with local station.",
    }


# ~12 crops: optimal temp, pH, water, season, soil
CROP_DB: dict[str, dict] = {
    "corn": {
        "optimal_temp_c": (18, 32),
        "ph_range": (6.0, 7.0),
        "water_need": "high",
        "preferred_season": "spring_summer",
        "soil_preference": "loam, well-drained",
    },
    "soybean": {
        "optimal_temp_c": (15, 30),
        "ph_range": (6.2, 7.0),
        "water_need": "medium",
        "preferred_season": "spring_summer",
        "soil_preference": "loam, clay loam",
    },
    "winter wheat": {
        "optimal_temp_c": (5, 24),
        "ph_range": (6.0, 7.5),
        "water_need": "medium",
        "preferred_season": "fall_winter_spring",
        "soil_preference": "silt, loam",
    },
    "rice": {
        "optimal_temp_c": (20, 35),
        "ph_range": (5.5, 6.5),
        "water_need": "very_high",
        "preferred_season": "spring_summer",
        "soil_preference": "clay, flooded",
    },
    "wine grapes": {
        "optimal_temp_c": (15, 28),
        "ph_range": (5.5, 7.0),
        "water_need": "low_to_medium",
        "preferred_season": "perennial",
        "soil_preference": "well-drained loam, hillsides",
    },
    "cotton": {
        "optimal_temp_c": (21, 35),
        "ph_range": (5.8, 8.0),
        "water_need": "medium_high",
        "preferred_season": "spring_summer",
        "soil_preference": "sandy loam to clay",
    },
    "barley": {
        "optimal_temp_c": (5, 22),
        "ph_range": (6.0, 7.5),
        "water_need": "medium",
        "preferred_season": "spring_fall",
        "soil_preference": "well-drained loam",
    },
    "tomatoes": {
        "optimal_temp_c": (18, 28),
        "ph_range": (6.0, 6.8),
        "water_need": "medium",
        "preferred_season": "spring_summer",
        "soil_preference": "loam, rich OM",
    },
    "bananas": {
        "optimal_temp_c": (24, 32),
        "ph_range": (5.5, 7.0),
        "water_need": "high",
        "preferred_season": "year_round_tropical",
        "soil_preference": "deep loam, good drainage",
    },
    "alfalfa": {
        "optimal_temp_c": (10, 28),
        "ph_range": (6.5, 7.5),
        "water_need": "medium_high",
        "preferred_season": "spring_summer",
        "soil_preference": "well-drained loam",
    },
    "potatoes": {
        "optimal_temp_c": (15, 22),
        "ph_range": (4.8, 6.5),
        "water_need": "medium",
        "preferred_season": "spring_summer",
        "soil_preference": "sandy loam, loose",
    },
    "sorghum": {
        "optimal_temp_c": (20, 38),
        "ph_range": (5.5, 8.5),
        "water_need": "low_to_medium",
        "preferred_season": "summer",
        "soil_preference": "sandy to clay; drought tolerant",
    },
}


def get_crop_database(crop_name: str) -> dict:
    key = _normalize_key(crop_name)
    if key in CROP_DB:
        return {"crop": crop_name, **CROP_DB[key]}
    for crop_key, data in CROP_DB.items():
        if crop_key in key or key in crop_key:
            return {"crop": crop_key, **data}
    return {
        "crop": crop_name,
        "error": "Crop not in database",
        "hint": "Try: corn, soybean, winter wheat, rice, wine grapes, sorghum, etc.",
    }


TOOL_FUNCTIONS = {
    "get_soil_data": get_soil_data,
    "get_weather_forecast": get_weather_forecast,
    "get_crop_database": get_crop_database,
}

# === AGENT LOGIC ===


def format_input(input_data: dict) -> str:
    parts = []
    if "field_id" in input_data:
        parts.append(f"Field ID: {input_data['field_id']}")
    if "location" in input_data:
        parts.append(f"Location: {input_data['location']}")
    if "current_crop" in input_data:
        parts.append(f"Current crop: {input_data['current_crop']}")
    if "soil_type" in input_data:
        parts.append(f"Soil type (reported): {input_data['soil_type']}")
    if "season" in input_data:
        parts.append(f"Season: {input_data['season']}")
    if "acres" in input_data:
        parts.append(f"Acres: {input_data['acres']}")
    if "irrigation_available" in input_data:
        parts.append(f"Irrigation available: {input_data['irrigation_available']}")
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
        "recommendation": "wait",
        "suggested_crops": [],
        "risk_factors": ["Could not parse model output"],
        "expected_yield_rating": "poor",
        "fertilizer_recommendation": "",
        "timing_advice": "Review inputs and retry.",
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
