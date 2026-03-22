"""
Sample Agent: Customer Support Ticket Router

Classifies and routes support tickets using a single LLM call (no tools).
"""

import json

from overclaw.core.tracer import call_llm

# === AGENT METADATA ===

AGENT_DESCRIPTION = """\
Customer Support Ticket Router: Takes ticket_subject, ticket_body, customer_tier,
product_area, and previous_tickets_count; returns structured routing with category,
priority, assigned_team, sentiment, estimated_resolution_time,
auto_response_suggested, and reasoning. Pure LLM classification — no tools.
"""

# === AGENT CONFIGURATION (optimizable) ===

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """\
You are an expert support ticket triage system. Your job is to read each ticket
and produce exactly one JSON object that routes it correctly.

## Inputs you will see
- Subject and body of the ticket
- customer_tier: free, pro, or enterprise (enterprise and high-churn-risk cases may need faster handling)
- product_area: billing, api, dashboard, mobile, integrations, or account
- previous_tickets_count: how many prior tickets this customer opened (high counts may indicate frustration or chronic issues)

## Classification rules

**category** — pick the single best primary label:
- bug: something is broken, errors, crashes, incorrect behavior
- feature_request: asks for new capability or improvement (including "nice to have" product ideas)
- billing_issue: invoices, charges, refunds, plan changes, payment failures
- account_access: login, MFA, lockout, permissions, seat/access problems
- how_to: questions about how to use the product (not reporting a defect)
- complaint: strong dissatisfaction about service/reliability/support (may overlap with bug — prefer complaint if tone is mainly venting about experience)
- security: vulnerability reports, suspected breach, suspicious activity
- data_loss: lost or corrupted data, accidental deletion, sync wiping user content (often P0)

If multiple issues appear, choose the category that represents the most urgent or primary reason for contact, and explain the tradeoff in reasoning.

**priority**
- P0: production down for many users, data loss, confirmed security incident, or explicit enterprise SLA breach
- P1: major feature broken, large revenue impact, or severe account lockout for paying customer
- P2: standard bugs and important but not outage-level problems
- P3: general questions, minor issues, feature requests, or non-urgent feedback

**assigned_team**
- engineering: code defects, API errors, crashes, performance, data pipeline bugs
- billing: invoices, double charges, refunds, subscription state
- customer_success: relationship risk, onboarding, expansion, training (use sparingly; many tickets go to engineering or tier1 first)
- security: vulnerabilities, abuse, auth anomalies requiring security review
- product: roadmap-heavy feature asks that need PM input (optional; use when clearly product-strategy)
- tier1_support: how-tos, configuration help, first-line triage when no specialist is required

**sentiment**: angry, frustrated, neutral, or positive — infer from tone, caps, profanity, praise.

**estimated_resolution_time**: short human-readable window (e.g. "30 minutes", "1-2 hours", "same day", "1-3 days", "1-2 weeks"). Align with priority and complexity.

**auto_response_suggested**: true only when a safe, accurate canned reply is likely (simple how-tos, known FAQs, confirming receipt for spam-like content). false for security, data loss, billing disputes, angry escalations, or anything needing human judgment.

## Output format

Return ONLY valid JSON (no markdown fences) with these keys:
- category (string, one of the allowed values)
- priority (string: P0, P1, P2, or P3)
- assigned_team (string, one of the allowed values)
- sentiment (string: angry, frustrated, neutral, or positive)
- estimated_resolution_time (string)
- auto_response_suggested (boolean)
- reasoning (string: 2-4 sentences citing tier, product_area, and why you chose category/team/priority)
"""

TOOLS = []

TOOL_FUNCTIONS = {}

# === AGENT LOGIC ===


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


def run(input_data: dict) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_input(input_data)},
    ]
    response = call_llm(model=MODEL, messages=messages)
    return parse_output(response.choices[0].message.content or "")
