"""
Sample Agent: Logistics Shipment Routing

Takes shipment parameters (origin, destination, weight, dimensions, type,
flags, dates) and produces a structured routing recommendation with carrier,
cost, delivery estimate, alternatives, risks, and reasoning.
"""

import json
import re
from datetime import datetime

from overclaw.core.tracer import call_llm, call_tool

# === AGENT METADATA ===

AGENT_DESCRIPTION = """\
Logistics Shipment Routing Agent: Takes origin_city, destination_city,
package_weight_kg, package_dimensions, shipment_type (standard/express/
overnight/freight), fragile, hazardous, requested_delivery_date and returns
JSON with recommended_route, carrier, estimated_cost, estimated_delivery_date,
alternative_options (carrier/cost/delivery_date), risk_factors, and reasoning.
Uses tools to query routes, warehouse inventory, and carrier availability.
"""

# === AGENT CONFIGURATION (optimizable) ===

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """\
You are a logistics routing assistant. Use the tools to find available routes,
check warehouse stock near the origin, and verify carrier availability and
rates for the shipment profile. Combine tool results with the user's constraints
(fragile, hazardous, weight, dimensions, shipment type, delivery date).

Return your final answer as JSON with these fields:
- recommended_route (string description of the chosen lane/service)
- carrier (string)
- estimated_cost (float, USD)
- estimated_delivery_date (string, ISO date YYYY-MM-DD preferred)
- alternative_options (list of objects, each with carrier, cost, delivery_date)
- risk_factors (list of strings)
- reasoning (brief explanation)
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_routes",
            "description": (
                "Find available shipping routes between origin and destination "
                "with distances, transit times, and baseline costs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin_city": {
                        "type": "string",
                        "description": "Origin city or metro area",
                    },
                    "destination_city": {
                        "type": "string",
                        "description": "Destination city or metro area",
                    },
                },
                "required": ["origin_city", "destination_city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_warehouse_inventory",
            "description": (
                "Check whether requested items are in stock at the warehouse "
                "nearest to the origin for fulfillment or consolidation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin_city": {
                        "type": "string",
                        "description": "Origin city to find nearest warehouse",
                    },
                    "item_category": {
                        "type": "string",
                        "description": (
                            "Item category or SKU hint (e.g. electronics, pharma_cold, "
                            "industrial_parts, artwork, hazmat_kit, envelope)"
                        ),
                    },
                },
                "required": ["origin_city", "item_category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_carrier_availability",
            "description": (
                "Check carrier availability, rates, and capability flags for a "
                "route on a given pickup or ship date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin_city": {"type": "string"},
                    "destination_city": {"type": "string"},
                    "ship_date": {
                        "type": "string",
                        "description": "Requested ship date YYYY-MM-DD",
                    },
                    "package_weight_kg": {
                        "type": "number",
                        "description": "Package weight in kilograms",
                    },
                    "fragile": {
                        "type": "boolean",
                        "description": "Whether contents are fragile",
                    },
                    "hazardous": {
                        "type": "boolean",
                        "description": "Whether contents are hazardous/DG",
                    },
                    "package_dimensions": {
                        "type": "string",
                        "description": "Dimensions e.g. 40x30x20cm for size/girth checks",
                    },
                },
                "required": [
                    "origin_city",
                    "destination_city",
                    "ship_date",
                    "package_weight_kg",
                    "fragile",
                    "hazardous",
                ],
            },
        },
    },
]

# === TOOL IMPLEMENTATIONS (fixed — do not optimize) ===

# route_id -> details (US-focused network; includes one international leg mock)
ROUTES_DB: dict[str, dict] = {
    "NYC-LAX": {
        "origin": "New York, NY",
        "destination": "Los Angeles, CA",
        "distance_miles": 2451,
        "transit_days_standard": 4,
        "transit_days_express": 2,
        "transit_days_overnight": 1,
        "baseline_cost_usd": 48.0,
        "lane_type": "cross_country",
    },
    "NYC-CHI": {
        "origin": "New York, NY",
        "destination": "Chicago, IL",
        "distance_miles": 790,
        "transit_days_standard": 2,
        "transit_days_express": 1,
        "transit_days_overnight": 1,
        "baseline_cost_usd": 22.0,
        "lane_type": "regional",
    },
    "BOS-CHI": {
        "origin": "Boston, MA",
        "destination": "Chicago, IL",
        "distance_miles": 983,
        "transit_days_standard": 2,
        "transit_days_express": 1,
        "transit_days_overnight": 1,
        "baseline_cost_usd": 24.0,
        "lane_type": "regional",
    },
    "CHI-LAX": {
        "origin": "Chicago, IL",
        "destination": "Los Angeles, CA",
        "distance_miles": 1745,
        "transit_days_standard": 3,
        "transit_days_express": 2,
        "transit_days_overnight": 1,
        "baseline_cost_usd": 38.0,
        "lane_type": "cross_country",
    },
    "CHI-DEN": {
        "origin": "Chicago, IL",
        "destination": "Denver, CO",
        "distance_miles": 1007,
        "transit_days_standard": 2,
        "transit_days_express": 1,
        "transit_days_overnight": 1,
        "baseline_cost_usd": 26.0,
        "lane_type": "mountain",
    },
    "CHI-DAL": {
        "origin": "Chicago, IL",
        "destination": "Dallas, TX",
        "distance_miles": 925,
        "transit_days_standard": 2,
        "transit_days_express": 1,
        "transit_days_overnight": 1,
        "baseline_cost_usd": 24.0,
        "lane_type": "regional",
    },
    "DAL-LAX": {
        "origin": "Dallas, TX",
        "destination": "Los Angeles, CA",
        "distance_miles": 1436,
        "transit_days_standard": 3,
        "transit_days_express": 2,
        "transit_days_overnight": 1,
        "baseline_cost_usd": 32.0,
        "lane_type": "regional",
    },
    "SEA-DEN": {
        "origin": "Seattle, WA",
        "destination": "Denver, CO",
        "distance_miles": 1314,
        "transit_days_standard": 3,
        "transit_days_express": 2,
        "transit_days_overnight": 1,
        "baseline_cost_usd": 28.0,
        "lane_type": "mountain",
    },
    "ATL-MIA": {
        "origin": "Atlanta, GA",
        "destination": "Miami, FL",
        "distance_miles": 662,
        "transit_days_standard": 2,
        "transit_days_express": 1,
        "transit_days_overnight": 1,
        "baseline_cost_usd": 18.0,
        "lane_type": "regional",
    },
    "BOS-PHL": {
        "origin": "Boston, MA",
        "destination": "Philadelphia, PA",
        "distance_miles": 308,
        "transit_days_standard": 1,
        "transit_days_express": 1,
        "transit_days_overnight": 1,
        "baseline_cost_usd": 14.0,
        "lane_type": "short_haul",
    },
    "SFO-LAX": {
        "origin": "San Francisco, CA",
        "destination": "Los Angeles, CA",
        "distance_miles": 383,
        "transit_days_standard": 1,
        "transit_days_express": 1,
        "transit_days_overnight": 1,
        "baseline_cost_usd": 12.0,
        "lane_type": "short_haul",
    },
    "PHX-ELP": {
        "origin": "Phoenix, AZ",
        "destination": "El Paso, TX",
        "distance_miles": 430,
        "transit_days_standard": 2,
        "transit_days_express": 1,
        "transit_days_overnight": 1,
        "baseline_cost_usd": 16.0,
        "lane_type": "southwest",
    },
    "MSP-FAR": {
        "origin": "Minneapolis, MN",
        "destination": "Fargo, ND",
        "distance_miles": 234,
        "transit_days_standard": 2,
        "transit_days_express": 1,
        "transit_days_overnight": 1,
        "baseline_cost_usd": 15.0,
        "lane_type": "rural_feeder",
    },
    "MEM-ORD": {
        "origin": "Memphis, TN",
        "destination": "Chicago, IL",
        "distance_miles": 532,
        "transit_days_standard": 2,
        "transit_days_express": 1,
        "transit_days_overnight": 1,
        "baseline_cost_usd": 17.0,
        "lane_type": "hub_spoke",
    },
    "LAX-JFK": {
        "origin": "Los Angeles, CA",
        "destination": "New York, NY",
        "distance_miles": 2451,
        "transit_days_standard": 4,
        "transit_days_express": 2,
        "transit_days_overnight": 1,
        "baseline_cost_usd": 48.0,
        "lane_type": "cross_country",
    },
    "MIA-LHR": {
        "origin": "Miami, FL",
        "destination": "London, UK",
        "distance_miles": 4430,
        "transit_days_standard": 5,
        "transit_days_express": 3,
        "transit_days_overnight": 2,
        "baseline_cost_usd": 120.0,
        "lane_type": "international_air",
    },
    "ORD-YYZ": {
        "origin": "Chicago, IL",
        "destination": "Toronto, ON",
        "distance_miles": 525,
        "transit_days_standard": 2,
        "transit_days_express": 1,
        "transit_days_overnight": 1,
        "baseline_cost_usd": 45.0,
        "lane_type": "international_ground",
    },
    "DEN-SLC": {
        "origin": "Denver, CO",
        "destination": "Salt Lake City, UT",
        "distance_miles": 371,
        "transit_days_standard": 1,
        "transit_days_express": 1,
        "transit_days_overnight": 1,
        "baseline_cost_usd": 13.0,
        "lane_type": "short_haul",
    },
}

# Multi-stop: synthetic legs (tool returns ordered stops)
MULTI_STOP_PRESETS: dict[str, list[str]] = {
    "NYC-CHI-LAX": ["NYC-CHI", "CHI-DAL", "DAL-LAX"],
}

WAREHOUSE_DB: dict[str, dict] = {
    "WH-EWR": {
        "code": "WH-EWR",
        "city": "Newark, NJ",
        "region": "Northeast",
        "inventory": [
            "electronics",
            "general_merch",
            "pharma_cold",
            "medical_supplies",
        ],
        "capacity_score": 0.92,
    },
    "WH-ORD": {
        "code": "WH-ORD",
        "city": "Chicago, IL",
        "region": "Midwest",
        "inventory": [
            "industrial_parts",
            "general_merch",
            "automotive",
            "hazmat_kit",
        ],
        "capacity_score": 0.88,
    },
    "WH-DFW": {
        "code": "WH-DFW",
        "city": "Dallas, TX",
        "region": "South Central",
        "inventory": ["general_merch", "electronics", "apparel", "pharma_cold"],
        "capacity_score": 0.9,
    },
    "WH-LAX": {
        "code": "WH-LAX",
        "city": "Los Angeles, CA",
        "region": "West",
        "inventory": ["electronics", "artwork_crate", "general_merch", "media"],
        "capacity_score": 0.85,
    },
    "WH-ATL": {
        "code": "WH-ATL",
        "city": "Atlanta, GA",
        "region": "Southeast",
        "inventory": ["general_merch", "food_nonperish", "apparel"],
        "capacity_score": 0.87,
    },
    "WH-SEA": {
        "code": "WH-SEA",
        "city": "Seattle, WA",
        "region": "Pacific NW",
        "inventory": ["electronics", "industrial_parts", "general_merch"],
        "capacity_score": 0.83,
    },
    "WH-MEM": {
        "code": "WH-MEM",
        "city": "Memphis, TN",
        "region": "Mid-South Hub",
        "inventory": ["medical_supplies", "general_merch", "sortation_bulk"],
        "capacity_score": 0.91,
    },
    "WH-MIA": {
        "code": "WH-MIA",
        "city": "Miami, FL",
        "region": "Southeast / Intl gateway",
        "inventory": ["pharma_cold", "general_merch", "intl_docs"],
        "capacity_score": 0.86,
    },
}

CITY_ALIASES: dict[str, str] = {
    "nyc": "New York, NY",
    "new york": "New York, NY",
    "la": "Los Angeles, CA",
    "los angeles": "Los Angeles, CA",
    "sf": "San Francisco, CA",
    "san francisco": "San Francisco, CA",
    "chi": "Chicago, IL",
    "chicago": "Chicago, IL",
    "dal": "Dallas, TX",
    "dallas": "Dallas, TX",
    "atl": "Atlanta, GA",
    "atlanta": "Atlanta, GA",
    "mia": "Miami, FL",
    "miami": "Miami, FL",
    "sea": "Seattle, WA",
    "seattle": "Seattle, WA",
    "den": "Denver, CO",
    "denver": "Denver, CO",
    "bos": "Boston, MA",
    "boston": "Boston, MA",
    "phl": "Philadelphia, PA",
    "philadelphia": "Philadelphia, PA",
    "phx": "Phoenix, AZ",
    "phoenix": "Phoenix, AZ",
    "elp": "El Paso, TX",
    "el paso": "El Paso, TX",
    "msp": "Minneapolis, MN",
    "minneapolis": "Minneapolis, MN",
    "far": "Fargo, ND",
    "fargo": "Fargo, ND",
    "mem": "Memphis, TN",
    "memphis": "Memphis, TN",
    "slc": "Salt Lake City, UT",
    "salt lake city": "Salt Lake City, UT",
    "lhr": "London, UK",
    "london": "London, UK",
    "yyz": "Toronto, ON",
    "toronto": "Toronto, ON",
}


def _normalize_city(name: str) -> str:
    key = name.strip().lower()
    if key in CITY_ALIASES:
        return CITY_ALIASES[key]
    # Title-case heuristic for "City, ST"
    return name.strip()


def _find_route_key(origin: str, dest: str) -> str | None:
    o = _normalize_city(origin)
    d = _normalize_city(dest)
    for rid, r in ROUTES_DB.items():
        if r["origin"] == o and r["destination"] == d:
            return rid
    for rid, r in ROUTES_DB.items():
        if r["origin"] == d and r["destination"] == o:
            return rid
    return None


def check_routes(origin_city: str, destination_city: str) -> dict:
    o = _normalize_city(origin_city)
    d = _normalize_city(destination_city)
    if o == d:
        return {
            "match": "local",
            "origin": o,
            "destination": d,
            "routes": [
                {
                    "route_id": "LOCAL",
                    "distance_miles": 0,
                    "transit_days_standard": 0,
                    "transit_days_express": 0,
                    "transit_days_overnight": 0,
                    "baseline_cost_usd": 8.0,
                    "lane_type": "local_courier",
                }
            ],
            "multi_stop_options": [],
        }

    direct = _find_route_key(o, d)
    if direct:
        r = ROUTES_DB[direct]
        return {
            "match": "direct",
            "route_id": direct,
            **r,
            "multi_stop_options": [],
        }

    # Hub path: try via Chicago for US domestic
    leg1 = _find_route_key(o, "Chicago, IL")
    leg2 = _find_route_key("Chicago, IL", d)
    if leg1 and leg2:
        total_mi = ROUTES_DB[leg1]["distance_miles"] + ROUTES_DB[leg2]["distance_miles"]
        return {
            "match": "hub_via_chicago",
            "origin": o,
            "destination": d,
            "legs": [leg1, leg2],
            "total_distance_miles": total_mi,
            "transit_days_standard": ROUTES_DB[leg1]["transit_days_standard"]
            + ROUTES_DB[leg2]["transit_days_standard"],
            "baseline_cost_usd": ROUTES_DB[leg1]["baseline_cost_usd"]
            + ROUTES_DB[leg2]["baseline_cost_usd"] * 0.85,
            "multi_stop_options": [
                {
                    "name": "NYC-CHI-LAX style",
                    "legs": MULTI_STOP_PRESETS.get("NYC-CHI-LAX", []),
                    "note": "Example consolidated multi-stop trunk for high volume",
                }
            ],
        }

    # International or unknown: return nearest conceptual route
    return {
        "match": "partial",
        "origin": o,
        "destination": d,
        "routes": [
            {
                "route_id": "MIA-LHR",
                **{k: v for k, v in ROUTES_DB["MIA-LHR"].items() if k != "origin"},
                "note": "Use international air template if overseas",
            }
        ],
        "multi_stop_options": [],
        "message": "No direct domestic lane in DB; returned international template or partial.",
    }


def _nearest_warehouse(city: str) -> tuple[str, dict]:
    c = _normalize_city(city)
    # Simple region map to warehouse (more specific cities first)
    region_map = [
        (["Miami"], "WH-MIA"),
        (["Memphis"], "WH-MEM"),
        (["New York", "Newark", "Boston", "Philadelphia"], "WH-EWR"),
        (["Chicago", "Minneapolis"], "WH-ORD"),
        (["Dallas", "El Paso", "Houston"], "WH-DFW"),
        (["Los Angeles", "Phoenix", "San Francisco"], "WH-LAX"),
        (["Atlanta"], "WH-ATL"),
        (["Seattle", "Denver", "Salt Lake"], "WH-SEA"),
    ]
    for cities, wh in region_map:
        if any(x in c for x in cities):
            return wh, WAREHOUSE_DB[wh]
    return "WH-ORD", WAREHOUSE_DB["WH-ORD"]


def check_warehouse_inventory(origin_city: str, item_category: str) -> dict:
    wh_id, wh = _nearest_warehouse(origin_city)
    cat = item_category.lower().strip().replace(" ", "_")
    synonyms = {
        "docs": "intl_docs",
        "documents": "intl_docs",
        "pharma": "pharma_cold",
        "cold_chain": "pharma_cold",
        "art": "artwork_crate",
        "machinery": "industrial_parts",
        "equipment": "industrial_parts",
        "dg": "hazmat_kit",
        "hazmat": "hazmat_kit",
        "letter": "general_merch",
        "envelope": "general_merch",
    }
    cat = synonyms.get(cat, cat)
    in_stock = cat in wh["inventory"] or cat == "general" or cat == "general_merch"
    return {
        "nearest_warehouse_id": wh_id,
        "warehouse_city": wh["city"],
        "region": wh["region"],
        "item_category_requested": item_category,
        "normalized_category": cat,
        "in_stock": in_stock,
        "matching_skus_sample": wh["inventory"][:4],
        "capacity_score": wh["capacity_score"],
    }


def _parse_dimensions(dim: str) -> tuple[float, float, float]:
    """Return length, width, height in cm from strings like '40x30x20cm'."""
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)",
        dim.lower().replace(" ", ""),
    )
    if not m:
        return (30.0, 20.0, 15.0)
    return (float(m.group(1)), float(m.group(2)), float(m.group(3)))


def _girth_cm(dims: tuple[float, float, float]) -> float:
    s = sorted(dims)
    return s[0] + s[1] * 2 + s[2] * 2


CARRIER_DB: dict[str, dict] = {
    "FedEx": {
        "modes": ["express", "ground", "freight"],
        "max_weight_kg": 500.0,
        "max_girth_cm": 330.0,
        "fragile_ok": True,
        "hazmat_ok": True,
        "cold_chain": True,
        "base_rate_per_kg": 2.8,
        "holiday_capacity_factor": 0.72,
    },
    "UPS": {
        "modes": ["express", "ground", "freight"],
        "max_weight_kg": 500.0,
        "max_girth_cm": 320.0,
        "fragile_ok": True,
        "hazmat_ok": True,
        "cold_chain": True,
        "base_rate_per_kg": 2.6,
        "holiday_capacity_factor": 0.7,
    },
    "USPS": {
        "modes": ["standard", "express"],
        "max_weight_kg": 31.5,
        "max_girth_cm": 274.0,
        "fragile_ok": False,
        "hazmat_ok": False,
        "cold_chain": False,
        "base_rate_per_kg": 1.4,
        "holiday_capacity_factor": 0.55,
    },
    "DHL": {
        "modes": ["express", "freight", "international"],
        "max_weight_kg": 1000.0,
        "max_girth_cm": 400.0,
        "fragile_ok": True,
        "hazmat_ok": True,
        "cold_chain": True,
        "base_rate_per_kg": 3.2,
        "holiday_capacity_factor": 0.68,
    },
    "XPO Freight": {
        "modes": ["freight", "ltl"],
        "max_weight_kg": 10000.0,
        "max_girth_cm": 800.0,
        "fragile_ok": True,
        "hazmat_ok": True,
        "cold_chain": False,
        "base_rate_per_kg": 1.1,
        "holiday_capacity_factor": 0.8,
    },
    "Old Dominion Freight": {
        "modes": ["freight", "ltl"],
        "max_weight_kg": 10000.0,
        "max_girth_cm": 800.0,
        "fragile_ok": True,
        "hazmat_ok": True,
        "cold_chain": False,
        "base_rate_per_kg": 1.05,
        "holiday_capacity_factor": 0.78,
    },
}


def _baseline_cost_from_route_result(route: dict) -> float:
    if route.get("baseline_cost_usd") is not None:
        return float(route["baseline_cost_usd"])
    rts = route.get("routes") or []
    if rts:
        return float(rts[0].get("baseline_cost_usd", 18.0))
    return 18.0


def _is_holiday_season(ship_date: str) -> bool:
    try:
        dt = datetime.strptime(ship_date[:10], "%Y-%m-%d")
    except ValueError:
        return False
    # Peak: Dec 10–24
    return dt.month == 12 and 10 <= dt.day <= 24


def get_carrier_availability(
    origin_city: str,
    destination_city: str,
    ship_date: str,
    package_weight_kg: float,
    fragile: bool,
    hazardous: bool,
    package_dimensions: str = "30x20x15cm",
) -> dict:
    o = _normalize_city(origin_city)
    d = _normalize_city(destination_city)
    route_result = check_routes(o, d)
    dims = _parse_dimensions(package_dimensions)
    girth = _girth_cm(dims)
    holiday = _is_holiday_season(ship_date)
    base_lane = _baseline_cost_from_route_result(route_result)

    carriers_out: list[dict] = []
    for name, spec in CARRIER_DB.items():
        eligible = True
        reasons: list[str] = []
        if package_weight_kg > spec["max_weight_kg"]:
            eligible = False
            reasons.append("over_weight_cap")
        if girth > spec["max_girth_cm"]:
            eligible = False
            reasons.append("oversized_girth")
        if fragile and not spec["fragile_ok"]:
            eligible = False
            reasons.append("fragile_not_supported")
        if hazardous and not spec["hazmat_ok"]:
            eligible = False
            reasons.append("hazmat_not_supported")

        base = base_lane
        if route_result.get("legs"):
            base = float(route_result.get("baseline_cost_usd", base_lane))

        rate = base + spec["base_rate_per_kg"] * float(package_weight_kg)
        if fragile:
            rate *= 1.15
        if hazardous:
            rate *= 1.35
        cap = spec["holiday_capacity_factor"] if holiday else 1.0
        availability = "available" if eligible and cap > 0.65 else "limited"
        if holiday:
            availability = "limited" if eligible else "unavailable"
        carriers_out.append(
            {
                "carrier": name,
                "eligible": eligible,
                "availability": availability if eligible else "unavailable",
                "quoted_rate_usd": round(rate, 2) if eligible else None,
                "capacity_hint": round(cap, 2),
                "holiday_peak": holiday,
                "disqualification_reasons": reasons,
                "lane_match": route_result.get("match", "unknown"),
            }
        )

    carriers_out.sort(
        key=lambda x: (x["eligible"] is False, x["quoted_rate_usd"] or 1e9)
    )
    return {
        "origin": o,
        "destination": d,
        "ship_date": ship_date,
        "package_weight_kg": package_weight_kg,
        "girth_cm": round(girth, 1),
        "holiday_peak_season": holiday,
        "carriers": carriers_out,
    }


TOOL_FUNCTIONS = {
    "check_routes": check_routes,
    "check_warehouse_inventory": check_warehouse_inventory,
    "get_carrier_availability": get_carrier_availability,
}

# === AGENT LOGIC ===


def format_input(input_data: dict) -> str:
    parts = []
    if "origin_city" in input_data:
        parts.append(f"Origin: {input_data['origin_city']}")
    if "destination_city" in input_data:
        parts.append(f"Destination: {input_data['destination_city']}")
    if "package_weight_kg" in input_data:
        parts.append(f"Weight (kg): {input_data['package_weight_kg']}")
    if "package_dimensions" in input_data:
        parts.append(f"Dimensions: {input_data['package_dimensions']}")
    if "shipment_type" in input_data:
        parts.append(f"Shipment type: {input_data['shipment_type']}")
    if "fragile" in input_data:
        parts.append(f"Fragile: {input_data['fragile']}")
    if "hazardous" in input_data:
        parts.append(f"Hazardous: {input_data['hazardous']}")
    if "requested_delivery_date" in input_data:
        parts.append(f"Requested delivery by: {input_data['requested_delivery_date']}")
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
        "recommended_route": "unknown",
        "carrier": "unknown",
        "estimated_cost": 0.0,
        "estimated_delivery_date": "unknown",
        "alternative_options": [],
        "risk_factors": ["parse_failure"],
        "reasoning": f"Parse failure: {reason[:200]}",
    }


def run(input_data: dict) -> dict:
    """Main agent entry point. Takes shipment info, returns routing recommendation."""
    user_content = format_input(input_data)
    extra = []
    if "package_dimensions" in input_data:
        extra.append(
            "When calling get_carrier_availability, use package_dimensions from the user "
            f"if needed: {input_data['package_dimensions']}"
        )
    if extra:
        user_content = user_content + "\n\n" + "\n".join(extra)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    response = call_llm(model=MODEL, messages=messages, tools=TOOLS)
    message = response.choices[0].message

    max_tool_rounds = 5
    for _ in range(max_tool_rounds):
        if not message.tool_calls:
            break

        assistant_msg: dict = {"role": "assistant", "content": message.content}
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
                if fn_name == "get_carrier_availability":
                    pd = input_data.get("package_dimensions", "30x20x15cm")
                    fn_args.setdefault("package_dimensions", pd)
                result = call_tool(fn_name, fn_args, TOOL_FUNCTIONS[fn_name])
            else:
                result = {"error": f"Unknown tool: {fn_name}"}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

        response = call_llm(model=MODEL, messages=messages, tools=TOOLS)
        message = response.choices[0].message

    return parse_output(message.content or "")
