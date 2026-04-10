"""
Tool implementations for the lead qualification agent.
These are fixed — do not optimize.
"""

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
