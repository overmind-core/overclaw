"""
Sample Agent: Lead Qualification

Takes sales lead information (company name, contact email, inquiry text) and
produces a structured qualification assessment.  This file is the optimization
target — OverClaw will iteratively improve it.
"""

import json

from overclaw.core.tracer import call_llm, call_tool

# === AGENT METADATA ===

AGENT_DESCRIPTION = """\
Lead Qualification Agent: Takes sales lead information (company_name,
contact_email, inquiry) and returns a structured JSON assessment with:
lead_score (0-100), category (hot/warm/cold), priority (high/medium/low),
recommended_action (schedule_demo/send_info/nurture/disqualify), and reasoning.
"""

# === AGENT CONFIGURATION (optimizable) ===

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """\
You are a lead qualification assistant. Given information about a sales lead,
analyze it and qualify it. Use the available tools to gather information, then
provide your assessment.

Return your final answer as JSON with these fields:
- lead_score (0-100)
- category (hot, warm, or cold)
- priority (high, medium, or low)
- recommended_action (schedule_demo, send_info, nurture, or disqualify)
- reasoning (brief explanation)
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_company",
            "description": "Search for company information",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "Company name to search",
                    }
                },
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_intent",
            "description": "Analyze buying intent from text",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_budget_signals",
            "description": "Check for budget and purchasing signals",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Inquiry text to check",
                    },
                    "company_size": {
                        "type": "string",
                        "description": "Company size category",
                    },
                },
                "required": ["text"],
            },
        },
    },
]

# === TOOL IMPLEMENTATIONS (fixed — do not optimize) ===

COMPANY_DB = {
    "techcorp solutions": {
        "size": "large",
        "industry": "technology",
        "revenue": "$50M-100M",
        "employees": 500,
    },
    "greenleaf organics": {
        "size": "small",
        "industry": "food_beverage",
        "revenue": "$1M-5M",
        "employees": 25,
    },
    "metropolis financial": {
        "size": "enterprise",
        "industry": "finance",
        "revenue": "$500M+",
        "employees": 5000,
    },
    "startupxyz": {
        "size": "startup",
        "industry": "saas",
        "revenue": "<$1M",
        "employees": 8,
    },
    "midwest manufacturing": {
        "size": "medium",
        "industry": "manufacturing",
        "revenue": "$10M-50M",
        "employees": 200,
    },
    "cloudnine hosting": {
        "size": "medium",
        "industry": "technology",
        "revenue": "$5M-10M",
        "employees": 75,
    },
    "bella's boutique": {
        "size": "micro",
        "industry": "retail",
        "revenue": "<$500K",
        "employees": 3,
    },
    "global logistics inc": {
        "size": "enterprise",
        "industry": "logistics",
        "revenue": "$200M+",
        "employees": 3000,
    },
    "innovate health": {
        "size": "medium",
        "industry": "healthcare",
        "revenue": "$20M-50M",
        "employees": 150,
    },
    "eduspark learning": {
        "size": "small",
        "industry": "education",
        "revenue": "$2M-5M",
        "employees": 30,
    },
    "quantum dynamics": {
        "size": "large",
        "industry": "technology",
        "revenue": "$100M+",
        "employees": 800,
    },
    "freshbite delivery": {
        "size": "small",
        "industry": "food_delivery",
        "revenue": "$3M-8M",
        "employees": 40,
    },
}


def search_company(company_name: str) -> dict:
    key = company_name.lower().strip()
    for db_key, info in COMPANY_DB.items():
        if db_key in key or key in db_key:
            return {"found": True, **info, "name": company_name}
    return {
        "found": False,
        "name": company_name,
        "size": "unknown",
        "industry": "unknown",
    }


INTENT_KEYWORDS = {
    "high": [
        "budget",
        "approved",
        "purchase",
        "buy",
        "implement",
        "deploy",
        "asap",
        "urgent",
        "enterprise",
        "contract",
        "ready to",
        "decision",
    ],
    "medium": [
        "looking for",
        "evaluate",
        "compare",
        "interested",
        "considering",
        "need",
        "solution",
        "team",
        "explore",
        "options",
    ],
    "low": [
        "just browsing",
        "curious",
        "maybe",
        "sometime",
        "no rush",
        "free",
        "student",
        "personal",
        "hobby",
        "learning",
    ],
}


def analyze_intent(text: str) -> dict:
    text_lower = text.lower()
    scores = {"high": 0, "medium": 0, "low": 0}
    matched: list[str] = []
    for level, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[level] += 1
                matched.append(f"{level}:{kw}")
    total = sum(scores.values()) or 1
    dominant = max(scores, key=scores.get)
    return {
        "intent_level": dominant,
        "confidence": round(scores[dominant] / total, 2),
        "signals": scores,
        "matched_keywords": matched,
    }


def check_budget_signals(text: str, company_size: str = "unknown") -> dict:
    text_lower = text.lower()
    budget_mentioned = any(
        w in text_lower
        for w in ["budget", "approved", "funding", "allocated", "$", "invest"]
    )
    timeline_mentioned = any(
        w in text_lower
        for w in [
            "q1",
            "q2",
            "q3",
            "q4",
            "this month",
            "this quarter",
            "asap",
            "immediately",
            "this year",
        ]
    )
    size_multiplier = {
        "enterprise": 1.5,
        "large": 1.3,
        "medium": 1.0,
        "small": 0.7,
        "startup": 0.5,
        "micro": 0.3,
        "unknown": 0.8,
    }
    base_score = (40 if budget_mentioned else 15) + (20 if timeline_mentioned else 5)
    return {
        "budget_mentioned": budget_mentioned,
        "timeline_mentioned": timeline_mentioned,
        "budget_confidence": min(
            base_score * size_multiplier.get(company_size, 0.8), 100
        ),
    }


TOOL_FUNCTIONS = {
    "search_company": search_company,
    "analyze_intent": analyze_intent,
    "check_budget_signals": check_budget_signals,
}

# === AGENT LOGIC ===


def format_input(input_data: dict) -> str:
    parts = []
    if "company_name" in input_data:
        parts.append(f"Company: {input_data['company_name']}")
    if "contact_email" in input_data:
        parts.append(f"Contact: {input_data['contact_email']}")
    if "inquiry" in input_data:
        parts.append(f"Inquiry: {input_data['inquiry']}")
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
        "lead_score": 0,
        "category": "unknown",
        "priority": "unknown",
        "recommended_action": "unknown",
        "reasoning": f"Parse failure: {reason[:200]}",
    }


def run(input_data: dict) -> dict:
    """Main agent entry point. Takes lead info, returns structured qualification."""
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

        # Build assistant message dict
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
