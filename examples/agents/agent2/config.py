"""
Agent configuration: model, system prompt, and tool schemas.
"""

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
