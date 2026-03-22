"""
Sample Agent: Insurance Claims Processing

Evaluates insurance claims using policy verification, claim history, damage
assessment, payout calculation, and fraud screening. This file is the
optimization target — OverClaw may iteratively improve it.
"""

import json
import re
from datetime import date, datetime
from typing import Any

from overclaw.core.tracer import call_llm, call_tool

# === AGENT METADATA ===

AGENT_DESCRIPTION = """\
Insurance Claims Processing Agent: Takes claim intake fields (policy_number,
policyholder_name, claim_type, incident_date, incident_description,
claimed_amount, supporting_documents) and returns structured JSON with:
claim_status (approved/denied/under_review/partial_approval), approved_amount,
coverage_applicable, fraud_risk (high/medium/low), denial_reasons, conditions,
and reasoning. Uses tools to verify policy, review history, assess damage,
estimate payout, and check fraud indicators.
"""

# === AGENT CONFIGURATION (optimizable) ===

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """\
You are an insurance claims processing assistant. For each claim, use the tools
in a sensible order: typically verify the policy first, check claim history,
assess how the described damage maps to coverage, calculate a payout estimate,
then run fraud screening. Combine tool results to decide outcome.

Return your final answer as JSON with these fields:
- claim_status: one of approved, denied, under_review, partial_approval
- approved_amount: number (0 if denied)
- coverage_applicable: boolean
- fraud_risk: one of high, medium, low
- denial_reasons: list of strings (empty if not denied)
- conditions: list of strings (e.g. further docs, appraisal, subrogation)
- reasoning: brief explanation tying tools and policy rules together
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "verify_policy",
            "description": (
                "Verify that an insurance policy exists, is active for the claim "
                "date, and return coverage details (limits, deductibles, type)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_number": {
                        "type": "string",
                        "description": "Policy identifier",
                    },
                    "policyholder_name": {
                        "type": "string",
                        "description": "Name of the insured policyholder",
                    },
                    "incident_date": {
                        "type": "string",
                        "description": "Claim incident date (ISO YYYY-MM-DD preferred)",
                    },
                },
                "required": ["policy_number", "policyholder_name", "incident_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_claim_history",
            "description": (
                "Retrieve prior claims for the policy to detect frequency, "
                "severity patterns, and repeat-claimant risk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_number": {
                        "type": "string",
                        "description": "Policy number to look up",
                    },
                },
                "required": ["policy_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assess_damage",
            "description": (
                "Map the incident description to coverage categories, severity, "
                "and whether exclusions or pre-existing damage may apply."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_type": {
                        "type": "string",
                        "description": "auto, home, health, life, or property",
                    },
                    "incident_description": {
                        "type": "string",
                        "description": "Narrative description of the loss or incident",
                    },
                },
                "required": ["claim_type", "incident_description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_payout",
            "description": (
                "Estimate approved payout after deductible and policy limits, "
                "using damage assessment outputs and claimed amount."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_number": {"type": "string", "description": "Policy ID"},
                    "claim_type": {
                        "type": "string",
                        "description": "Claim line type",
                    },
                    "severity_score": {
                        "type": "number",
                        "description": "Severity score from assess_damage (0-100)",
                    },
                    "claimed_amount": {
                        "type": "number",
                        "description": "Amount requested by claimant",
                    },
                    "coverage_exclusion_flag": {
                        "type": "boolean",
                        "description": (
                            "True if assess_damage indicated major exclusions"
                        ),
                    },
                },
                "required": [
                    "policy_number",
                    "claim_type",
                    "severity_score",
                    "claimed_amount",
                    "coverage_exclusion_flag",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_fraud_indicators",
            "description": (
                "Screen the claim for fraud red flags: documentation gaps, "
                "inconsistencies, timing, and behavioral signals."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_number": {"type": "string"},
                    "incident_date": {"type": "string"},
                    "claimed_amount": {"type": "number"},
                    "incident_description": {"type": "string"},
                    "supporting_documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of document descriptions provided",
                    },
                },
                "required": [
                    "policy_number",
                    "incident_date",
                    "claimed_amount",
                    "incident_description",
                    "supporting_documents",
                ],
            },
        },
    },
]

# === TOOL IMPLEMENTATIONS (fixed — do not optimize) ===

POLICY_DB: dict[str, dict[str, Any]] = {
    "POL-AUTO-10001": {
        "policyholder_name": "Jane Smith",
        "normalized_holder": "jane smith",
        "policy_type": "auto",
        "status": "active",
        "effective_from": "2023-06-01",
        "effective_to": "2026-05-31",
        "deductible": 500.0,
        "coverage_limits": {
            "collision": 50000.0,
            "comprehensive": 50000.0,
            "liability": 300000.0,
            "uninsured_motorist": 100000.0,
        },
        "endorsements": ["rental_reimbursement"],
    },
    "POL-AUTO-10002": {
        "policyholder_name": "Robert Chen",
        "normalized_holder": "robert chen",
        "policy_type": "auto",
        "status": "active",
        "effective_from": "2022-01-15",
        "effective_to": "2027-01-14",
        "deductible": 1000.0,
        "coverage_limits": {
            "collision": 35000.0,
            "comprehensive": 35000.0,
            "liability": 250000.0,
        },
        "endorsements": [],
    },
    "POL-AUTO-10003": {
        "policyholder_name": "Sarah Claims",
        "normalized_holder": "sarah claims",
        "policy_type": "auto",
        "status": "active",
        "effective_from": "2021-03-01",
        "effective_to": "2026-02-28",
        "deductible": 750.0,
        "coverage_limits": {
            "collision": 40000.0,
            "comprehensive": 40000.0,
            "liability": 200000.0,
        },
        "endorsements": ["roadside"],
    },
    "POL-AUTO-10004": {
        "policyholder_name": "Victoria Sterling",
        "normalized_holder": "victoria sterling",
        "policy_type": "auto",
        "status": "active",
        "effective_from": "2024-11-01",
        "effective_to": "2027-10-31",
        "deductible": 250.0,
        "coverage_limits": {
            "collision": 150000.0,
            "comprehensive": 150000.0,
            "liability": 500000.0,
        },
        "endorsements": ["new_car_replacement", "gap"],
    },
    "POL-HOME-20001": {
        "policyholder_name": "Alice Martinez",
        "normalized_holder": "alice martinez",
        "policy_type": "home",
        "status": "active",
        "effective_from": "2019-04-01",
        "effective_to": "2026-03-31",
        "deductible": 2500.0,
        "coverage_limits": {
            "dwelling": 450000.0,
            "other_structures": 45000.0,
            "personal_property": 225000.0,
            "loss_of_use": 90000.0,
            "liability": 300000.0,
        },
        "endorsements": ["water_backup", "earthquake_rider"],
    },
    "POL-HOME-20002": {
        "policyholder_name": "Frank Expired",
        "normalized_holder": "frank expired",
        "policy_type": "home",
        "status": "expired",
        "effective_from": "2018-01-01",
        "effective_to": "2024-12-31",
        "deductible": 1000.0,
        "coverage_limits": {"dwelling": 280000.0, "personal_property": 140000.0},
        "endorsements": [],
    },
    "POL-HOME-20003": {
        "policyholder_name": "New Homeowner",
        "normalized_holder": "new homeowner",
        "policy_type": "home",
        "status": "active",
        "effective_from": "2025-03-01",
        "effective_to": "2028-02-29",
        "deductible": 1500.0,
        "coverage_limits": {
            "dwelling": 380000.0,
            "personal_property": 190000.0,
            "liability": 200000.0,
        },
        "endorsements": [],
        "notes": "policy_within_first_30_days",
    },
    "POL-HEALTH-30001": {
        "policyholder_name": "Maria Garcia",
        "normalized_holder": "maria garcia",
        "policy_type": "health",
        "status": "active",
        "effective_from": "2024-01-01",
        "effective_to": "2025-12-31",
        "deductible": 2000.0,
        "coverage_limits": {
            "annual_out_of_pocket_max": 8000.0,
            "in_network": 2000000.0,
        },
        "coinsurance_in_network": 0.2,
        "endorsements": ["hsa_compatible"],
    },
    "POL-LIFE-40001": {
        "policyholder_name": "James Okonkwo",
        "normalized_holder": "james okonkwo",
        "policy_type": "life",
        "status": "active",
        "effective_from": "2015-06-01",
        "effective_to": "2045-05-31",
        "deductible": 0.0,
        "coverage_limits": {"death_benefit": 500000.0},
        "endorsements": ["accidental_death"],
    },
    "POL-PROP-50001": {
        "policyholder_name": "Harbor Retail LLC",
        "normalized_holder": "harbor retail llc",
        "policy_type": "property",
        "status": "active",
        "effective_from": "2023-09-01",
        "effective_to": "2026-08-31",
        "deductible": 5000.0,
        "coverage_limits": {
            "building": 1200000.0,
            "business_personal_property": 250000.0,
            "business_income": 180000.0,
            "glass": 50000.0,
        },
        "endorsements": ["equipment_breakdown"],
    },
}

CLAIM_HISTORY_DB: list[dict[str, Any]] = [
    {
        "claim_id": "CLM-2019-00412",
        "policy_number": "POL-AUTO-10001",
        "date": "2019-08-14",
        "amount": 3200.0,
        "type": "auto",
        "status": "paid",
        "description": "Rear-end collision — bumper repair",
    },
    {
        "claim_id": "CLM-2021-01890",
        "policy_number": "POL-AUTO-10001",
        "date": "2021-02-03",
        "amount": 890.0,
        "type": "auto",
        "status": "paid",
        "description": "Windshield chip repair",
    },
    {
        "claim_id": "CLM-2022-03144",
        "policy_number": "POL-AUTO-10002",
        "date": "2022-11-20",
        "amount": 15400.0,
        "type": "auto",
        "status": "paid",
        "description": "Side-impact collision",
    },
    {
        "claim_id": "CLM-2023-00877",
        "policy_number": "POL-HOME-20001",
        "date": "2023-07-10",
        "amount": 12000.0,
        "type": "home",
        "status": "paid",
        "description": "Hail damage to roof",
    },
    {
        "claim_id": "CLM-2024-00102",
        "policy_number": "POL-HOME-20001",
        "date": "2024-01-22",
        "amount": 4500.0,
        "type": "home",
        "status": "paid",
        "description": "Pipe burst — kitchen",
    },
    {
        "claim_id": "CLM-2020-02233",
        "policy_number": "POL-AUTO-10003",
        "date": "2020-05-18",
        "amount": 6700.0,
        "type": "auto",
        "status": "paid",
        "description": "Parking lot scrape",
    },
    {
        "claim_id": "CLM-2021-04401",
        "policy_number": "POL-AUTO-10003",
        "date": "2021-09-02",
        "amount": 2100.0,
        "type": "auto",
        "status": "paid",
        "description": "Theft of catalytic converter",
    },
    {
        "claim_id": "CLM-2022-01988",
        "policy_number": "POL-AUTO-10003",
        "date": "2022-04-27",
        "amount": 9800.0,
        "type": "auto",
        "status": "paid",
        "description": "Multi-vehicle accident",
    },
    {
        "claim_id": "CLM-2023-02765",
        "policy_number": "POL-AUTO-10003",
        "date": "2023-12-11",
        "amount": 1400.0,
        "type": "auto",
        "status": "paid",
        "description": "Glass only",
    },
    {
        "claim_id": "CLM-2024-03301",
        "policy_number": "POL-AUTO-10003",
        "date": "2024-06-30",
        "amount": 5200.0,
        "type": "auto",
        "status": "paid",
        "description": "Deer strike",
    },
    {
        "claim_id": "CLM-2023-01444",
        "policy_number": "POL-HOME-20002",
        "date": "2023-03-15",
        "amount": 800.0,
        "type": "home",
        "status": "denied",
        "description": "Maintenance-related leak (prior to lapse)",
    },
    {
        "claim_id": "CLM-2024-02001",
        "policy_number": "POL-HEALTH-30001",
        "date": "2024-06-01",
        "amount": 45000.0,
        "type": "health",
        "status": "paid",
        "description": "Inpatient stay — appendectomy",
    },
    {
        "claim_id": "CLM-2024-02002",
        "policy_number": "POL-PROP-50001",
        "date": "2024-11-18",
        "amount": 22000.0,
        "type": "property",
        "status": "paid",
        "description": "Water damage — supply line",
    },
    {
        "claim_id": "CLM-2025-00155",
        "policy_number": "POL-AUTO-10004",
        "date": "2025-01-10",
        "amount": 18500.0,
        "type": "auto",
        "status": "paid",
        "description": "Total loss — flood (comprehensive)",
    },
    {
        "claim_id": "CLM-2025-00200",
        "policy_number": "POL-LIFE-40001",
        "date": "2025-02-01",
        "amount": 0.0,
        "type": "life",
        "status": "withdrawn",
        "description": "Beneficiary update inquiry only",
    },
]

# Keywords per claim line → coverage buckets and exclusion hints
DAMAGE_COVERAGE_MAP: dict[str, dict[str, Any]] = {
    "auto": {
        "collision_keywords": [
            "collision",
            "rear-ended",
            "accident",
            "crash",
            "fender",
            "dent",
        ],
        "comprehensive_keywords": [
            "hail",
            "deer",
            "theft",
            "vandalism",
            "fire",
            "flood",
            "glass",
            "windshield",
        ],
        "liability_keywords": ["injury", "pedestrian", "third party", "lawsuit"],
        "wear_exclusion_keywords": ["rust", "wear and tear", "old damage"],
    },
    "home": {
        "covered_perils": [
            "fire",
            "storm",
            "hail",
            "wind",
            "hurricane",
            "tornado",
            "lightning",
            "burst pipe",
            "water",
        ],
        "flood_exclusion": ["flood", "rising water", "storm surge"],
        "earthquake_keywords": ["earthquake", "seismic"],
        "maintenance_exclusion": ["maintenance", "neglect", "slow leak", "mold years"],
    },
    "health": {
        "covered_keywords": ["surgery", "hospital", "inpatient", "procedure", "er"],
        "cosmetic_exclusion": ["cosmetic", "elective enhancement"],
    },
    "life": {
        "covered_keywords": ["death", "passed away", "deceased", "fatal"],
        "contestability_keywords": ["suicide", "within 2 years", "misrepresentation"],
    },
    "property": {
        "covered_keywords": [
            "vandalism",
            "break-in",
            "burglary",
            "fire",
            "smoke",
            "water",
            "sprinkler",
        ],
        "inventory_exclusion": ["offsite", "personal vehicle", "employee theft"],
    },
}

FRAUD_CHECKLIST = [
    "claimed_amount_round_hundreds_thousands",
    "no_supporting_documents",
    "incident_date_within_days_of_policy_start",
    "multiple_high_claims_same_year",
    "description_conflicts_with_claim_type",
    "suspiciously_vague_description",
    "duplicate_incident_language",
]


def _parse_date(s: str) -> date | None:
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def verify_policy(
    policy_number: str, policyholder_name: str, incident_date: str
) -> dict[str, Any]:
    pn = policy_number.strip().upper()
    holder_norm = policyholder_name.lower().strip()
    inc = _parse_date(incident_date)

    rec = POLICY_DB.get(pn)
    if not rec:
        return {
            "found": False,
            "active": False,
            "policy_number": pn,
            "message": "Policy number not found",
        }

    name_ok = (
        holder_norm == rec["normalized_holder"]
        or holder_norm in rec["normalized_holder"]
        or rec["normalized_holder"] in holder_norm
    )

    eff_from = _parse_date(str(rec["effective_from"]))
    eff_to = _parse_date(str(rec["effective_to"]))
    in_term = False
    if inc and eff_from and eff_to:
        in_term = eff_from <= inc <= eff_to

    active = rec["status"] == "active" and (in_term if inc else True)

    return {
        "found": True,
        "policy_number": pn,
        "policyholder_match": name_ok,
        "policy_type": rec["policy_type"],
        "status": rec["status"],
        "effective_from": rec["effective_from"],
        "effective_to": rec["effective_to"],
        "incident_within_policy_term": in_term,
        "active_for_claim": active and name_ok,
        "deductible": rec["deductible"],
        "coverage_limits": rec["coverage_limits"],
        "endorsements": rec.get("endorsements", []),
        "notes": rec.get("notes"),
    }


def check_claim_history(policy_number: str) -> dict[str, Any]:
    pn = policy_number.strip().upper()
    rows = [c for c in CLAIM_HISTORY_DB if c["policy_number"] == pn]
    total_paid = sum(c["amount"] for c in rows if c["status"] == "paid")
    return {
        "policy_number": pn,
        "prior_claim_count": len(rows),
        "claims": rows,
        "total_paid_history": round(total_paid, 2),
        "repeat_claimant": len(rows) >= 4,
    }


def assess_damage(claim_type: str, incident_description: str) -> dict[str, Any]:
    ct = claim_type.lower().strip()
    text = incident_description.lower()
    cfg = DAMAGE_COVERAGE_MAP.get(ct, {})
    severity_score = 30.0
    matched: list[str] = []
    exclusion_risk = False
    coverage_exclusion_flag = False
    notes: list[str] = []

    if ct == "auto":
        for kw in cfg.get("collision_keywords", []):
            if kw in text:
                matched.append(f"collision:{kw}")
                severity_score += 12
        for kw in cfg.get("comprehensive_keywords", []):
            if kw in text:
                matched.append(f"comprehensive:{kw}")
                severity_score += 10
        for kw in cfg.get("wear_exclusion_keywords", []):
            if kw in text:
                exclusion_risk = True
                notes.append("Possible wear/pre-existing damage language")
        primary = (
            "collision"
            if any(m.startswith("collision") for m in matched)
            else "comprehensive"
            if any(m.startswith("comprehensive") for m in matched)
            else "general_auto"
        )
    elif ct == "home":
        peril_hit = False
        for kw in cfg.get("covered_perils", []):
            if kw in text:
                matched.append(f"peril:{kw}")
                peril_hit = True
                severity_score += 14
        for kw in cfg.get("flood_exclusion", []):
            if kw in text:
                coverage_exclusion_flag = True
                notes.append("Flood/surge may be excluded without rider")
        for kw in cfg.get("maintenance_exclusion", []):
            if kw in text:
                exclusion_risk = True
        primary = "dwelling" if peril_hit else "unspecified"
    elif ct == "health":
        for kw in cfg.get("covered_keywords", []):
            if kw in text:
                matched.append(kw)
                severity_score += 15
        for kw in cfg.get("cosmetic_exclusion", []):
            if kw in text:
                coverage_exclusion_flag = True
        primary = "medical"
    elif ct == "life":
        for kw in cfg.get("covered_keywords", []):
            if kw in text:
                matched.append(kw)
                severity_score += 40
        for kw in cfg.get("contestability_keywords", []):
            if kw in text:
                notes.append("Contestability or exclusion review may apply")
        primary = "death_benefit"
    elif ct == "property":
        for kw in cfg.get("covered_keywords", []):
            if kw in text:
                matched.append(kw)
                severity_score += 12
        primary = "commercial_property"
    else:
        primary = "unknown"
        notes.append("Unknown claim_type for mapping")

    severity_score = min(100.0, severity_score + len(matched) * 2)
    if exclusion_risk:
        severity_score = max(20.0, severity_score - 15)

    return {
        "claim_type": ct,
        "primary_coverage_bucket": primary,
        "matched_signals": matched,
        "severity_score": round(severity_score, 1),
        "coverage_exclusion_flag": coverage_exclusion_flag,
        "pre_existing_or_wear_risk": exclusion_risk,
        "notes": notes,
    }


def calculate_payout(
    policy_number: str,
    claim_type: str,
    severity_score: float,
    claimed_amount: float,
    coverage_exclusion_flag: bool,
) -> dict[str, Any]:
    pn = policy_number.strip().upper()
    rec = POLICY_DB.get(pn)
    ct = claim_type.lower().strip()

    if not rec:
        return {
            "policy_number": pn,
            "estimated_approved": 0.0,
            "applied_deductible": 0.0,
            "cap_reason": "policy_not_found",
        }

    if rec["status"] != "active":
        return {
            "policy_number": pn,
            "estimated_approved": 0.0,
            "applied_deductible": rec["deductible"],
            "cap_reason": "policy_not_active",
        }

    limits = rec["coverage_limits"]
    ded = float(rec["deductible"])

    # Pick a limit bucket by claim type
    line_cap = 100000.0
    if ct == "auto":
        line_cap = (
            max(
                limits.get("collision", 0),
                limits.get("comprehensive", 0),
                limits.get("liability", 0),
            )
            or 50000.0
        )
    elif ct == "home":
        line_cap = float(
            limits.get("dwelling", limits.get("personal_property", 200000.0))
        )
    elif ct == "health":
        line_cap = float(limits.get("in_network", 1000000.0))
        coins = float(rec.get("coinsurance_in_network", 0.2))
        # Rough allowed before OOP: severity drives utilization of claimed
        raw = max(0.0, claimed_amount * (severity_score / 100.0) * (1 - coins))
        est = max(0.0, raw - ded)
        est = min(est, line_cap, claimed_amount)
        return {
            "policy_number": pn,
            "estimated_approved": round(est, 2),
            "applied_deductible": ded,
            "line_limit": line_cap,
            "cap_reason": "within_limits" if est > 0 else "deductible_or_exclusion",
        }
    elif ct == "life":
        line_cap = float(limits.get("death_benefit", 0))
        est = line_cap if severity_score >= 50 and not coverage_exclusion_flag else 0.0
        return {
            "policy_number": pn,
            "estimated_approved": round(est, 2),
            "applied_deductible": 0.0,
            "line_limit": line_cap,
            "cap_reason": "life_benefit_rule",
        }
    else:
        line_cap = float(limits.get("building", limits.get("dwelling", 500000.0)))

    if coverage_exclusion_flag:
        return {
            "policy_number": pn,
            "estimated_approved": 0.0,
            "applied_deductible": ded,
            "line_limit": line_cap,
            "cap_reason": "exclusion_flag",
        }

    # Severity-weighted fraction of claimed, minus deductible, capped
    fraction = min(1.0, max(0.0, severity_score / 100.0))
    raw_est = max(0.0, claimed_amount * fraction - ded)
    est = min(raw_est, line_cap, claimed_amount)

    return {
        "policy_number": pn,
        "estimated_approved": round(est, 2),
        "applied_deductible": ded,
        "line_limit": line_cap,
        "cap_reason": "within_limits" if est > 0 else "below_deductible_or_zero",
    }


def check_fraud_indicators(
    policy_number: str,
    incident_date: str,
    claimed_amount: float,
    incident_description: str,
    supporting_documents: list[str],
) -> dict[str, Any]:
    pn = policy_number.strip().upper()
    rec = POLICY_DB.get(pn)
    flags: list[str] = []
    score = 0

    if claimed_amount >= 1000 and claimed_amount == round(claimed_amount / 1000) * 1000:
        flags.append(FRAUD_CHECKLIST[0])
        score += 1

    docs = supporting_documents or []
    if len(docs) == 0:
        flags.append(FRAUD_CHECKLIST[1])
        score += 2

    inc = _parse_date(incident_date)
    if rec and inc:
        start = _parse_date(str(rec["effective_from"]))
        if start and 0 <= (inc - start).days <= 30:
            flags.append(FRAUD_CHECKLIST[2])
            score += 2

    hist = [c for c in CLAIM_HISTORY_DB if c["policy_number"] == pn]
    recent_high = sum(
        1 for c in hist if c["status"] == "paid" and c.get("amount", 0) >= 10000
    )
    if recent_high >= 2:
        flags.append(FRAUD_CHECKLIST[3])
        score += 1

    desc = incident_description.lower()
    ptype = (rec or {}).get("policy_type", "")
    if ptype == "auto" and "surgery" in desc:
        flags.append(FRAUD_CHECKLIST[4])
        score += 2

    if len(desc) < 40:
        flags.append(FRAUD_CHECKLIST[5])
        score += 1

    if "identical" in desc or "copy paste" in desc:
        flags.append(FRAUD_CHECKLIST[6])
        score += 2

    if score >= 5:
        level = "high"
    elif score >= 2:
        level = "medium"
    else:
        level = "low"

    return {
        "policy_number": pn,
        "fraud_risk": level,
        "fraud_score": score,
        "flags": flags,
        "document_count": len(docs),
    }


TOOL_FUNCTIONS = {
    "verify_policy": verify_policy,
    "check_claim_history": check_claim_history,
    "assess_damage": assess_damage,
    "calculate_payout": calculate_payout,
    "check_fraud_indicators": check_fraud_indicators,
}

# === AGENT LOGIC ===


def format_input(input_data: dict) -> str:
    parts = []
    if "policy_number" in input_data:
        parts.append(f"Policy number: {input_data['policy_number']}")
    if "policyholder_name" in input_data:
        parts.append(f"Policyholder: {input_data['policyholder_name']}")
    if "claim_type" in input_data:
        parts.append(f"Claim type: {input_data['claim_type']}")
    if "incident_date" in input_data:
        parts.append(f"Incident date: {input_data['incident_date']}")
    if "incident_description" in input_data:
        parts.append(f"Incident description: {input_data['incident_description']}")
    if "claimed_amount" in input_data:
        parts.append(f"Claimed amount: {input_data['claimed_amount']}")
    if "supporting_documents" in input_data:
        docs = input_data["supporting_documents"]
        if isinstance(docs, list):
            parts.append("Supporting documents: " + "; ".join(str(d) for d in docs))
        else:
            parts.append(f"Supporting documents: {docs}")
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
        "claim_status": "under_review",
        "approved_amount": 0.0,
        "coverage_applicable": False,
        "fraud_risk": "unknown",
        "denial_reasons": [],
        "conditions": [],
        "reasoning": f"Parse failure: {reason[:200]}",
    }


def run(input_data: dict) -> dict:
    """Main agent entry point. Takes claim intake, returns structured decision."""
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

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": message.content,
        }
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
