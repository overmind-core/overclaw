"""
Sample Agent: Finance Fraud Detection

Evaluates a single card transaction using multi-step tool calls against mock
risk databases. This file is the optimization target for OverClaw.
"""

import json
from datetime import datetime, timedelta
from typing import Any

from overclaw.core.tracer import call_llm, call_tool

# === AGENT METADATA ===

AGENT_DESCRIPTION = """\
Finance Fraud Detection Agent: Takes transaction context (account_id,
transaction_amount, merchant_name, merchant_category, transaction_location,
timestamp, card_present) and returns a structured JSON fraud assessment with:
fraud_score (0-100), risk_level (critical/high/medium/low), is_fraudulent,
fraud_indicators, recommended_action (block/flag_for_review/allow/
request_verification), and reasoning. Uses tools to inspect account history,
merchant risk, velocity, and geolocation consistency.
"""

# === AGENT CONFIGURATION (optimizable) ===

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """\
You are a fraud analyst assistant for card and digital payments. Given one
transaction and metadata, use the tools to gather evidence from internal
systems. Combine signals carefully: velocity and geolocation anomalies,
merchant reputation, and account history (including prior fraud and recent
patterns). Then return a single JSON object with these exact fields:
- fraud_score (integer 0-100)
- risk_level (one of: critical, high, medium, low)
- is_fraudulent (boolean — true if you believe the transaction is likely fraud)
- fraud_indicators (array of short strings describing concrete red flags or "none")
- recommended_action (one of: block, flag_for_review, allow, request_verification)
- reasoning (brief explanation citing tool findings)

Do not include markdown fences or text outside the JSON object in your final message.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_account_history",
            "description": (
                "Retrieve recent transaction history for an account, including "
                "amounts, merchants, locations, timestamps, and card-present flags."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": "string",
                        "description": "Account identifier",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of recent transactions to return (default 12)",
                    },
                },
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_merchant",
            "description": (
                "Look up merchant reputation score, category, and whether the "
                "merchant is flagged as high-risk or on a watchlist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "merchant_name": {
                        "type": "string",
                        "description": "Merchant name as shown on the transaction",
                    },
                },
                "required": ["merchant_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_velocity_check",
            "description": (
                "Analyze transaction frequency and velocity for an account: "
                "counts in recent windows and a human-readable pattern label."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": "string",
                        "description": "Account identifier",
                    },
                    "window_hours": {
                        "type": "integer",
                        "description": "Rolling window in hours (default 24; use 1 for burst checks)",
                    },
                },
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_geolocation",
            "description": (
                "Check whether the transaction city/country is consistent with "
                "the account holder's known locations and travel profile."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": "string",
                        "description": "Account identifier",
                    },
                    "city": {
                        "type": "string",
                        "description": "Transaction city",
                    },
                    "country": {
                        "type": "string",
                        "description": "Transaction country (ISO name or common name)",
                    },
                },
                "required": ["account_id", "city", "country"],
            },
        },
    },
]

# === TOOL IMPLEMENTATIONS (fixed — do not optimize) ===

# ~10 accounts with distinct patterns (history, velocity keys, known locations).
ACCOUNT_HISTORY_DB: dict[str, list[dict[str, Any]]] = {
    "acc_001_chicago_retail": [
        {
            "amount": 42.5,
            "merchant": "City Grocery Co",
            "city": "Chicago",
            "country": "USA",
            "ts_offset_hours": 48,
            "card_present": True,
        },
        {
            "amount": 18.0,
            "merchant": "Boutique Coffee",
            "city": "Chicago",
            "country": "USA",
            "ts_offset_hours": 30,
            "card_present": True,
        },
        {
            "amount": 65.0,
            "merchant": "Shell Station",
            "city": "Chicago",
            "country": "USA",
            "ts_offset_hours": 12,
            "card_present": True,
        },
    ],
    "acc_002_frequent_traveler": [
        {
            "amount": 220.0,
            "merchant": "Hotel Grand Paris",
            "city": "Paris",
            "country": "France",
            "ts_offset_hours": 200,
            "card_present": True,
        },
        {
            "amount": 89.0,
            "merchant": "Boutique Coffee",
            "city": "Denver",
            "country": "USA",
            "ts_offset_hours": 96,
            "card_present": True,
        },
        {
            "amount": 145.0,
            "merchant": "United Airlines",
            "city": "Denver",
            "country": "USA",
            "ts_offset_hours": 72,
            "card_present": False,
        },
    ],
    "acc_003_prior_fraud": [
        {
            "amount": 499.0,
            "merchant": "Unknown Merchant XYZ",
            "city": "Miami",
            "country": "USA",
            "ts_offset_hours": 400,
            "card_present": False,
            "note": "disputed",
        },
        {
            "amount": 32.0,
            "merchant": "City Grocery Co",
            "city": "Miami",
            "country": "USA",
            "ts_offset_hours": 50,
            "card_present": True,
        },
    ],
    "acc_004_brand_new": [
        {
            "amount": 9.99,
            "merchant": "Steam Games",
            "city": "Austin",
            "country": "USA",
            "ts_offset_hours": 2,
            "card_present": False,
        },
    ],
    "acc_005_dormant_reactivated": [
        {
            "amount": 12.0,
            "merchant": "City Grocery Co",
            "city": "Portland",
            "country": "USA",
            "ts_offset_hours": 9000,
            "card_present": True,
        },
        {
            "amount": 8.5,
            "merchant": "Shell Station",
            "city": "Portland",
            "country": "USA",
            "ts_offset_hours": 8950,
            "card_present": True,
        },
        {
            "amount": 750.0,
            "merchant": "ElectroMart Online",
            "city": "Portland",
            "country": "USA",
            "ts_offset_hours": 6,
            "card_present": False,
        },
    ],
    "acc_006_card_testing": [
        {
            "amount": 1.0,
            "merchant": "QuickMart Gas",
            "city": "Phoenix",
            "country": "USA",
            "ts_offset_hours": 3,
            "card_present": False,
        },
        {
            "amount": 1.0,
            "merchant": "QuickMart Gas",
            "city": "Phoenix",
            "country": "USA",
            "ts_offset_hours": 2.5,
            "card_present": False,
        },
        {
            "amount": 1.0,
            "merchant": "QuickMart Gas",
            "city": "Phoenix",
            "country": "USA",
            "ts_offset_hours": 2,
            "card_present": False,
        },
        {
            "amount": 1.0,
            "merchant": "QuickMart Gas",
            "city": "Phoenix",
            "country": "USA",
            "ts_offset_hours": 1.5,
            "card_present": False,
        },
        {
            "amount": 1.0,
            "merchant": "QuickMart Gas",
            "city": "Phoenix",
            "country": "USA",
            "ts_offset_hours": 1,
            "card_present": False,
        },
    ],
    "acc_007_takeover_pattern": [
        {
            "amount": 55.0,
            "merchant": "Boutique Coffee",
            "city": "Seattle",
            "country": "USA",
            "ts_offset_hours": 120,
            "card_present": True,
        },
        {
            "amount": 48.0,
            "merchant": "City Grocery Co",
            "city": "Seattle",
            "country": "USA",
            "ts_offset_hours": 48,
            "card_present": True,
        },
    ],
    "acc_008_velocity_abuse": [
        {
            "amount": 40.0,
            "merchant": "Amazon Retail",
            "city": "Atlanta",
            "country": "USA",
            "ts_offset_hours": 0.4,
            "card_present": False,
        },
        {
            "amount": 35.0,
            "merchant": "Amazon Retail",
            "city": "Atlanta",
            "country": "USA",
            "ts_offset_hours": 0.35,
            "card_present": False,
        },
        {
            "amount": 50.0,
            "merchant": "Amazon Retail",
            "city": "Atlanta",
            "country": "USA",
            "ts_offset_hours": 0.3,
            "card_present": False,
        },
        {
            "amount": 45.0,
            "merchant": "Amazon Retail",
            "city": "Atlanta",
            "country": "USA",
            "ts_offset_hours": 0.25,
            "card_present": False,
        },
    ],
    "acc_009_premium_legit": [
        {
            "amount": 1200.0,
            "merchant": "Apple Store",
            "city": "San Francisco",
            "country": "USA",
            "ts_offset_hours": 168,
            "card_present": True,
        },
        {
            "amount": 85.0,
            "merchant": "Luxury Watches LLC",
            "city": "San Francisco",
            "country": "USA",
            "ts_offset_hours": 90,
            "card_present": True,
        },
    ],
    "acc_010_refund_pattern": [
        {
            "amount": 200.0,
            "merchant": "FlyByNight Tickets",
            "city": "Las Vegas",
            "country": "USA",
            "ts_offset_hours": 100,
            "card_present": False,
        },
        {
            "amount": -200.0,
            "merchant": "FlyByNight Tickets",
            "city": "Las Vegas",
            "country": "USA",
            "ts_offset_hours": 80,
            "card_present": False,
            "note": "refund",
        },
        {
            "amount": 200.0,
            "merchant": "FlyByNight Tickets",
            "city": "Las Vegas",
            "country": "USA",
            "ts_offset_hours": 60,
            "card_present": False,
        },
        {
            "amount": -200.0,
            "merchant": "FlyByNight Tickets",
            "city": "Las Vegas",
            "country": "USA",
            "ts_offset_hours": 40,
            "card_present": False,
            "note": "chargeback",
        },
    ],
}

KNOWN_LOCATIONS_DB: dict[str, list[dict[str, str]]] = {
    "acc_001_chicago_retail": [{"city": "Chicago", "country": "USA"}],
    "acc_002_frequent_traveler": [
        {"city": "Denver", "country": "USA"},
        {"city": "Paris", "country": "France"},
        {"city": "Chicago", "country": "USA"},
    ],
    "acc_003_prior_fraud": [{"city": "Miami", "country": "USA"}],
    "acc_004_brand_new": [{"city": "Austin", "country": "USA"}],
    "acc_005_dormant_reactivated": [{"city": "Portland", "country": "USA"}],
    "acc_006_card_testing": [{"city": "Phoenix", "country": "USA"}],
    "acc_007_takeover_pattern": [{"city": "Seattle", "country": "USA"}],
    "acc_008_velocity_abuse": [{"city": "Atlanta", "country": "USA"}],
    "acc_009_premium_legit": [{"city": "San Francisco", "country": "USA"}],
    "acc_010_refund_pattern": [{"city": "Las Vegas", "country": "USA"}],
}

# Precomputed velocity windows (tx counts) — rich labels for mock realism.
VELOCITY_DB: dict[str, dict[str, Any]] = {
    "acc_001_chicago_retail": {
        "tx_1h": 0,
        "tx_24h": 3,
        "tx_7d": 3,
        "pattern": "normal_steady",
    },
    "acc_002_frequent_traveler": {
        "tx_1h": 0,
        "tx_24h": 1,
        "tx_7d": 3,
        "pattern": "travel_spaced",
    },
    "acc_003_prior_fraud": {
        "tx_1h": 0,
        "tx_24h": 1,
        "tx_7d": 2,
        "pattern": "elevated_watchlist",
    },
    "acc_004_brand_new": {
        "tx_1h": 1,
        "tx_24h": 1,
        "tx_7d": 1,
        "pattern": "new_account_sparse",
    },
    "acc_005_dormant_reactivated": {
        "tx_1h": 1,
        "tx_24h": 1,
        "tx_7d": 1,
        "pattern": "dormant_then_spike",
    },
    "acc_006_card_testing": {
        "tx_1h": 5,
        "tx_24h": 5,
        "tx_7d": 5,
        "pattern": "card_testing_micro_charges",
    },
    "acc_007_takeover_pattern": {
        "tx_1h": 0,
        "tx_24h": 0,
        "tx_7d": 2,
        "pattern": "normal_prior_low_volume",
    },
    "acc_008_velocity_abuse": {
        "tx_1h": 12,
        "tx_24h": 14,
        "tx_7d": 14,
        "pattern": "burst_velocity",
    },
    "acc_009_premium_legit": {
        "tx_1h": 0,
        "tx_24h": 0,
        "tx_7d": 2,
        "pattern": "high_value_infrequent",
    },
    "acc_010_refund_pattern": {
        "tx_1h": 0,
        "tx_24h": 2,
        "tx_7d": 4,
        "pattern": "refund_chargeback_cycle",
    },
}

MERCHANT_DB: dict[str, dict[str, Any]] = {
    "amazon retail": {
        "reputation_score": 92,
        "category": "retail",
        "high_risk": False,
        "watchlist": False,
        "notes": "Major marketplace; CNP common",
    },
    "quickmart gas": {
        "reputation_score": 78,
        "category": "gas_convenience",
        "high_risk": False,
        "watchlist": False,
    },
    "cryptovault exchange": {
        "reputation_score": 35,
        "category": "crypto",
        "high_risk": True,
        "watchlist": True,
        "notes": "Irreversible transfers; fraud magnet",
    },
    "luxury watches llc": {
        "reputation_score": 88,
        "category": "luxury_retail",
        "high_risk": False,
        "watchlist": False,
    },
    "unknown merchant xyz": {
        "reputation_score": 22,
        "category": "unknown",
        "high_risk": True,
        "watchlist": True,
    },
    "steam games": {
        "reputation_score": 85,
        "category": "digital_goods",
        "high_risk": False,
        "watchlist": False,
    },
    "city grocery co": {
        "reputation_score": 90,
        "category": "grocery",
        "high_risk": False,
        "watchlist": False,
    },
    "darkweb electronics": {
        "reputation_score": 8,
        "category": "electronics",
        "high_risk": True,
        "watchlist": True,
        "notes": "Known stolen-goods conduit",
    },
    "hotel grand paris": {
        "reputation_score": 82,
        "category": "travel_lodging",
        "high_risk": False,
        "watchlist": False,
    },
    "boutique coffee": {
        "reputation_score": 88,
        "category": "food_beverage",
        "high_risk": False,
        "watchlist": False,
    },
    "flybynight tickets": {
        "reputation_score": 28,
        "category": "ticketing",
        "high_risk": True,
        "watchlist": True,
        "notes": "High dispute rate",
    },
    "shell station": {
        "reputation_score": 86,
        "category": "gas_convenience",
        "high_risk": False,
        "watchlist": False,
    },
    "apple store": {
        "reputation_score": 95,
        "category": "electronics_retail",
        "high_risk": False,
        "watchlist": False,
    },
    "foreign pharmacy online": {
        "reputation_score": 18,
        "category": "pharmacy",
        "high_risk": True,
        "watchlist": True,
    },
    "transferwise p2p": {
        "reputation_score": 72,
        "category": "money_transfer",
        "high_risk": False,
        "watchlist": False,
        "notes": "Verify beneficiary on large sends",
    },
    "electromart online": {
        "reputation_score": 70,
        "category": "electronics",
        "high_risk": False,
        "watchlist": False,
    },
    "united airlines": {
        "reputation_score": 90,
        "category": "travel",
        "high_risk": False,
        "watchlist": False,
    },
}


def _normalize_merchant_key(name: str) -> str:
    return name.lower().strip()


def _match_merchant(name: str) -> dict[str, Any]:
    key = _normalize_merchant_key(name)
    if key in MERCHANT_DB:
        return {"matched": True, "merchant_name": name, **MERCHANT_DB[key]}
    for db_key, info in MERCHANT_DB.items():
        if db_key in key or key in db_key:
            return {"matched": True, "merchant_name": name, **info}
    return {
        "matched": False,
        "merchant_name": name,
        "reputation_score": 50,
        "category": "unknown",
        "high_risk": False,
        "watchlist": False,
        "notes": "No strong match in merchant directory",
    }


def get_account_history(account_id: str, limit: int = 12) -> dict[str, Any]:
    aid = account_id.strip()
    rows = ACCOUNT_HISTORY_DB.get(aid)
    if not rows:
        return {
            "found": False,
            "account_id": aid,
            "transactions": [],
            "message": "Unknown account_id or no history on file",
        }
    lim = max(1, min(limit, 24))
    recent = rows[-lim:]
    base = datetime(2025, 3, 20, 12, 0, 0)
    out: list[dict[str, Any]] = []
    for r in recent:
        off = float(r["ts_offset_hours"])
        ts = base - timedelta(hours=off)
        item: dict[str, Any] = {
            "amount": r["amount"],
            "merchant": r["merchant"],
            "location": {"city": r["city"], "country": r["country"]},
            "timestamp_iso": ts.isoformat() + "Z",
            "card_present": r.get("card_present", False),
        }
        if "note" in r:
            item["note"] = r["note"]
        out.append(item)
    return {
        "found": True,
        "account_id": aid,
        "transaction_count_returned": len(out),
        "transactions": out,
    }


def check_merchant(merchant_name: str) -> dict[str, Any]:
    return _match_merchant(merchant_name)


def get_velocity_check(account_id: str, window_hours: int = 24) -> dict[str, Any]:
    aid = account_id.strip()
    v = VELOCITY_DB.get(aid)
    if not v:
        return {
            "found": False,
            "account_id": aid,
            "message": "No velocity profile for account",
        }
    wh = max(1, min(window_hours, 168))
    # Map window to synthetic counts for multi-step richness.
    if wh <= 1:
        count = v["tx_1h"]
        label = "1h_window"
    elif wh <= 24:
        count = v["tx_24h"]
        label = "24h_window"
    else:
        count = v["tx_7d"]
        label = "7d_window"
    burst = v["tx_1h"] >= 4 or (v["pattern"] == "burst_velocity" and wh <= 24)
    return {
        "found": True,
        "account_id": aid,
        "window_hours": wh,
        "window_label": label,
        "transaction_count": count,
        "pattern": v["pattern"],
        "burst_suspected": burst,
        "raw": {"tx_1h": v["tx_1h"], "tx_24h": v["tx_24h"], "tx_7d": v["tx_7d"]},
    }


def check_geolocation(account_id: str, city: str, country: str) -> dict[str, Any]:
    aid = account_id.strip()
    locs = KNOWN_LOCATIONS_DB.get(aid, [])
    c_norm = city.strip().lower()
    co_norm = country.strip().lower()
    if not locs:
        return {
            "found": bool(ACCOUNT_HISTORY_DB.get(aid)),
            "account_id": aid,
            "known_locations": [],
            "transaction_location": {"city": city, "country": country},
            "consistent": False,
            "anomaly": "unknown_profile",
            "traveler_profile": False,
        }
    match = any(
        x["city"].lower() == c_norm and x["country"].lower() == co_norm for x in locs
    )
    traveler = len(locs) >= 3
    anomaly = "none"
    if not match:
        anomaly = "location_not_in_usual_set"
        if not traveler:
            anomaly = "unexpected_location_non_traveler"
    return {
        "found": True,
        "account_id": aid,
        "known_locations": locs,
        "transaction_location": {"city": city, "country": country},
        "consistent": match,
        "anomaly": anomaly,
        "traveler_profile": traveler,
    }


TOOL_FUNCTIONS = {
    "get_account_history": get_account_history,
    "check_merchant": check_merchant,
    "get_velocity_check": get_velocity_check,
    "check_geolocation": check_geolocation,
}

# === AGENT LOGIC ===


def format_input(input_data: dict) -> str:
    loc = input_data.get("transaction_location") or {}
    city = loc.get("city", "")
    country = loc.get("country", "")
    parts = [
        f"account_id: {input_data.get('account_id', '')}",
        f"transaction_amount: {input_data.get('transaction_amount', '')}",
        f"merchant_name: {input_data.get('merchant_name', '')}",
        f"merchant_category: {input_data.get('merchant_category', '')}",
        f"transaction_location: {city}, {country}",
        f"timestamp: {input_data.get('timestamp', '')}",
        f"card_present: {input_data.get('card_present', '')}",
    ]
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
        "fraud_score": 0,
        "risk_level": "medium",
        "is_fraudulent": False,
        "fraud_indicators": ["parse_failure"],
        "recommended_action": "flag_for_review",
        "reasoning": f"Parse failure: {reason[:200]}",
    }


def run(input_data: dict) -> dict:
    """Main agent entry point. Takes transaction context, returns fraud assessment."""
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
