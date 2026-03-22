"""
Sample Agent: HR Resume Screening

Evaluates candidates against job requirements using tool-backed skill extraction
and requirement matching. This file is the optimization target for OverClaw.
"""

import json
import re
from typing import Any

from overclaw.core.tracer import call_llm, call_tool

# === AGENT METADATA ===

AGENT_DESCRIPTION = """\
HR Resume Screening Agent: Takes candidate_name, resume_text, job_title,
job_id, and years_experience; uses tools to extract structured skills and match
them to role requirements. Returns JSON with overall_score (0-100),
skill_match_percentage, experience_match, culture_fit_indicators, strengths,
gaps, recommendation, and reasoning.
"""

# === AGENT CONFIGURATION (optimizable) ===

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """\
You are an HR resume screening assistant. Use the tools in order: first call
extract_skills on the resume text, then call match_job_requirements with the
job_id, the extracted_skills object returned from extract_skills, and
years_experience from the user input.

Synthesize a fair, evidence-based hiring recommendation. Consider tool outputs,
the role, and the resume narrative.

Return your final answer as JSON with these fields:
- overall_score (0-100 integer)
- skill_match_percentage (float, 0-100)
- experience_match (one of: over_qualified, strong_match, adequate, under_qualified)
- culture_fit_indicators (list of short strings)
- strengths (list of short strings)
- gaps (list of short strings)
- recommendation (one of: advance_to_interview, maybe_pile, reject)
- reasoning (brief explanation tying tools and resume together)
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_skills",
            "description": (
                "Extract and categorize skills from resume text using the internal "
                "skills taxonomy. Returns normalized skills with proficiency estimates (0-100)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resume_text": {
                        "type": "string",
                        "description": "Full or summary resume text to analyze",
                    },
                },
                "required": ["resume_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "match_job_requirements",
            "description": (
                "Compare extracted skills and experience against the job posting "
                "for the given job_id. Returns required/preferred match percentages "
                "and experience alignment hints."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": (
                            "Role key, e.g. software_engineer, data_scientist, "
                            "product_manager, devops_engineer, marketing_manager, "
                            "sales_rep, ux_designer, finance_analyst"
                        ),
                    },
                    "extracted_skills": {
                        "type": "object",
                        "description": "The full object returned by extract_skills",
                    },
                    "years_experience": {
                        "type": "integer",
                        "description": "Total relevant years of experience stated by the candidate",
                    },
                },
                "required": ["job_id", "extracted_skills", "years_experience"],
            },
        },
    },
]

# === SKILLS TAXONOMY (fixed — do not optimize) ===

# Canonical skill name -> list of surface forms / aliases (lowercase)
SKILLS_TAXONOMY: dict[str, list[str]] = {
    # Technical — languages & frameworks
    "python": ["python", "pandas", "numpy", "django", "flask", "fastapi"],
    "javascript": ["javascript", "js", "node.js", "nodejs", "react", "vue", "angular"],
    "typescript": ["typescript", "ts"],
    "java": ["java", "spring", "spring boot"],
    "go": ["golang", " go "],
    "sql": ["sql", "postgres", "postgresql", "mysql", "sqlite", "t-sql"],
    "aws": ["aws", "amazon web services", "ec2", "s3", "lambda"],
    "kubernetes": ["kubernetes", "k8s", "helm"],
    "docker": ["docker", "containerization", "containers"],
    "terraform": ["terraform", "iac", "infrastructure as code"],
    "git": ["git", "github", "gitlab", "version control"],
    "machine_learning": [
        "machine learning",
        "ml ",
        " ml,",
        "scikit",
        "tensorflow",
        "pytorch",
        "keras",
    ],
    "statistics": ["statistics", "statistical", "a/b test", "hypothesis testing"],
    "excel": ["excel", "spreadsheet", "vlookup", "pivot"],
    "tableau": ["tableau", "looker", "power bi", "powerbi"],
    "figma": ["figma", "sketch", "wireframe"],
    "seo": ["seo", "sem", "search engine"],
    "crm": ["salesforce", "hubspot", "crm"],
    "agile": ["agile", "scrum", "kanban", "sprint"],
    "product_strategy": ["product strategy", "roadmap", "okr", "prioritization"],
    "stakeholder_management": ["stakeholder", "cross-functional", "executive"],
    "public_speaking": ["presentations", "keynote", "conference talks"],
    "negotiation": ["negotiation", "contract negotiation"],
    "cobol": ["cobol", "mainframe"],
    "jquery": ["jquery"],
    "php": ["php", "laravel"],
    "ruby": ["ruby", "rails", "ruby on rails"],
    "spark": ["spark", "pyspark", "databricks"],
    "airflow": ["airflow", "workflow orchestration"],
    "linux": ["linux", "bash", "shell scripting"],
    "ci_cd": ["ci/cd", "jenkins", "github actions", "gitlab ci"],
    "networking": ["tcp/ip", "dns", "load balancing", "cdn"],
    "security": ["oauth", "soc2", "penetration", "security"],
    "accounting": ["gaap", "accounting", "bookkeeping"],
    "financial_modeling": ["financial model", "dcf", "forecasting", "valuation"],
}

# Category membership for reporting
SKILL_CATEGORIES: dict[str, str] = {
    "python": "programming",
    "javascript": "programming",
    "typescript": "programming",
    "java": "programming",
    "go": "programming",
    "php": "programming",
    "ruby": "programming",
    "sql": "data",
    "machine_learning": "data_science",
    "statistics": "data_science",
    "spark": "data_engineering",
    "airflow": "data_engineering",
    "excel": "business_analysis",
    "tableau": "business_analysis",
    "aws": "cloud",
    "kubernetes": "cloud",
    "docker": "cloud",
    "terraform": "cloud",
    "git": "engineering_practice",
    "linux": "systems",
    "ci_cd": "engineering_practice",
    "networking": "systems",
    "security": "systems",
    "figma": "design",
    "seo": "marketing",
    "crm": "sales",
    "agile": "process",
    "product_strategy": "product",
    "stakeholder_management": "leadership",
    "public_speaking": "soft_skills",
    "negotiation": "soft_skills",
    "cobol": "legacy",
    "jquery": "legacy",
    "accounting": "finance",
    "financial_modeling": "finance",
}

DEPTH_WEAK = re.compile(
    r"\b(familiar with|exposure to|basic knowledge|coursework in|toy project|"
    r"played with|dabbled)\b",
    re.I,
)
DEPTH_STRONG = re.compile(
    r"\b(lead|led|architect|principal|staff|senior|years of|production|"
    r"scaled|owned|mentor|deep expertise)\b",
    re.I,
)
GAP_PHRASE = re.compile(
    r"\b(career break|employment gap|sabbatical|time off|raising family)\b", re.I
)
FREELANCE_PHRASE = re.compile(r"\b(freelance|contractor|consultant|gig)\b", re.I)


def _estimate_proficiency(skill: str, resume_lower: str, window: int = 80) -> int:
    """Heuristic 0-100 proficiency from local context."""
    idx = resume_lower.find(skill.replace("_", " "))
    if idx < 0:
        for alias in SKILLS_TAXONOMY.get(skill, [skill]):
            idx = resume_lower.find(alias)
            if idx >= 0:
                break
    snippet = (
        resume_lower[max(0, idx - 40) : idx + window] if idx >= 0 else resume_lower
    )
    base = 55
    if DEPTH_STRONG.search(snippet):
        base += 25
    if DEPTH_WEAK.search(snippet):
        base -= 25
    if re.search(r"\b\d+\+?\s*years?\b", snippet):
        base += 10
    return max(15, min(95, base))


def extract_skills(resume_text: str) -> dict[str, Any]:
    """Extract normalized skills and proficiency estimates from resume text."""
    text = resume_text or ""
    resume_lower = text.lower()
    found: dict[str, int] = {}

    for canonical, aliases in SKILLS_TAXONOMY.items():
        for alias in aliases:
            al = alias.strip().lower()
            if al and al in resume_lower:
                prev = found.get(canonical, 0)
                prof = _estimate_proficiency(canonical, resume_lower)
                found[canonical] = max(prev, prof)
                break

    skills_by_category: dict[str, list[dict[str, Any]]] = {}
    for skill, prof in sorted(found.items(), key=lambda x: (-x[1], x[0])):
        cat = SKILL_CATEGORIES.get(skill, "general")
        skills_by_category.setdefault(cat, []).append(
            {
                "skill": skill,
                "proficiency_estimate": prof,
                "category": cat,
            }
        )

    shallow_keyword_rich = (
        len(found) >= 8 and sum(found.values()) / max(len(found), 1) < 58
    )
    signals: list[str] = []
    if GAP_PHRASE.search(resume_lower):
        signals.append("mentions_employment_gap")
    if FREELANCE_PHRASE.search(resume_lower):
        signals.append("freelance_or_contract_work")
    if re.search(
        r"\b(cs degree|b\.?s\.?\s+computer|m\.?s\.?\s+computer)\b", resume_lower
    ):
        signals.append("formal_cs_education")
    if shallow_keyword_rich:
        signals.append("possible_keyword_stuffing_or_broad_familiarity")

    return {
        "normalized_skills": list(found.keys()),
        "proficiency_by_skill": {k: v for k, v in found.items()},
        "skills_by_category": skills_by_category,
        "resume_signals": signals,
        "skills_count": len(found),
    }


JOB_REQUIREMENTS_DB: dict[str, dict[str, Any]] = {
    "software_engineer": {
        "display_title": "Software Engineer",
        "required_skills": ["python", "git", "sql"],
        "preferred_skills": ["typescript", "aws", "kubernetes", "docker"],
        "min_years": 2,
        "culture_notes": ["collaboration", "code_review", "agile"],
    },
    "data_scientist": {
        "display_title": "Data Scientist",
        "required_skills": ["python", "sql", "statistics"],
        "preferred_skills": ["machine_learning", "spark", "tableau"],
        "min_years": 2,
        "culture_notes": ["experimentation", "stakeholder_management"],
    },
    "product_manager": {
        "display_title": "Product Manager",
        "required_skills": ["agile", "product_strategy", "stakeholder_management"],
        "preferred_skills": ["sql", "excel", "public_speaking"],
        "min_years": 3,
        "culture_notes": ["cross_functional", "customer_focus"],
    },
    "devops_engineer": {
        "display_title": "DevOps Engineer",
        "required_skills": ["linux", "docker", "git", "ci_cd"],
        "preferred_skills": ["kubernetes", "terraform", "aws", "networking"],
        "min_years": 2,
        "culture_notes": ["on_call", "automation", "security"],
    },
    "marketing_manager": {
        "display_title": "Marketing Manager",
        "required_skills": ["seo", "excel", "stakeholder_management"],
        "preferred_skills": ["tableau", "public_speaking", "crm"],
        "min_years": 3,
        "culture_notes": ["brand", "analytics", "campaigns"],
    },
    "sales_rep": {
        "display_title": "Sales Representative",
        "required_skills": ["crm", "negotiation", "public_speaking"],
        "preferred_skills": ["excel", "stakeholder_management"],
        "min_years": 1,
        "culture_notes": ["quota", "relationships"],
    },
    "ux_designer": {
        "display_title": "UX Designer",
        "required_skills": ["figma", "agile", "stakeholder_management"],
        "preferred_skills": ["javascript", "public_speaking"],
        "min_years": 2,
        "culture_notes": ["user_research", "accessibility"],
    },
    "finance_analyst": {
        "display_title": "Finance Analyst",
        "required_skills": ["excel", "financial_modeling", "accounting"],
        "preferred_skills": ["sql", "stakeholder_management", "tableau"],
        "min_years": 2,
        "culture_notes": ["attention_to_detail", "compliance"],
    },
}


def _pct_matched(required: list[str], have: set[str]) -> float:
    if not required:
        return 100.0
    matched = sum(1 for s in required if s in have)
    return round(100.0 * matched / len(required), 1)


def _experience_band(min_years: int, years: int) -> str:
    if years < min_years:
        return "under_qualified"
    if years <= min_years + 2:
        return "adequate"
    if years <= min_years + 7:
        return "strong_match"
    return "over_qualified"


def match_job_requirements(
    job_id: str,
    extracted_skills: dict[str, Any],
    years_experience: int,
) -> dict[str, Any]:
    """Compare extracted skills to job_id requirements; return match metrics."""
    key = (job_id or "").strip().lower().replace(" ", "_")
    if key not in JOB_REQUIREMENTS_DB:
        return {
            "error": f"Unknown job_id: {job_id}",
            "known_job_ids": list(JOB_REQUIREMENTS_DB.keys()),
        }

    job = JOB_REQUIREMENTS_DB[key]
    prof_map = extracted_skills.get("proficiency_by_skill") or {}
    if isinstance(prof_map, str):
        try:
            prof_map = json.loads(prof_map)
        except json.JSONDecodeError:
            prof_map = {}
    normalized = extracted_skills.get("normalized_skills") or list(prof_map.keys())
    have = {s.lower() for s in normalized} | {k.lower() for k in prof_map}

    req = job["required_skills"]
    pref = job["preferred_skills"]
    required_match_pct = _pct_matched(req, have)
    preferred_match_pct = _pct_matched(pref, have)

    weighted_skill = round(0.65 * required_match_pct + 0.35 * preferred_match_pct, 1)

    min_y = int(job["min_years"])
    years = int(years_experience)
    experience_alignment = _experience_band(min_y, years)

    missing_required = [s for s in req if s not in have]
    missing_preferred = [s for s in pref if s not in have]
    matched_required = [s for s in req if s in have]
    matched_preferred = [s for s in pref if s in have]

    return {
        "job_id": key,
        "role_title": job["display_title"],
        "required_skills_match_percentage": required_match_pct,
        "preferred_skills_match_percentage": preferred_match_pct,
        "skill_match_percentage": weighted_skill,
        "min_years_required": min_y,
        "years_experience_provided": years,
        "experience_alignment": experience_alignment,
        "missing_required_skills": missing_required,
        "missing_preferred_skills": missing_preferred,
        "matched_required_skills": matched_required,
        "matched_preferred_skills": matched_preferred,
        "culture_keywords_for_role": job.get("culture_notes", []),
    }


TOOL_FUNCTIONS = {
    "extract_skills": extract_skills,
    "match_job_requirements": match_job_requirements,
}

# === AGENT LOGIC ===


def format_input(input_data: dict) -> str:
    parts = []
    if "candidate_name" in input_data:
        parts.append(f"Candidate name: {input_data['candidate_name']}")
    if "job_title" in input_data:
        parts.append(f"Role / job title: {input_data['job_title']}")
    if "job_id" in input_data:
        parts.append(f"job_id (use for match_job_requirements): {input_data['job_id']}")
    if "years_experience" in input_data:
        parts.append(f"Years of experience (integer): {input_data['years_experience']}")
    if "resume_text" in input_data:
        parts.append(f"Resume summary:\n{input_data['resume_text']}")
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
        "overall_score": 0,
        "skill_match_percentage": 0.0,
        "experience_match": "adequate",
        "culture_fit_indicators": [],
        "strengths": [],
        "gaps": [],
        "recommendation": "maybe_pile",
        "reasoning": f"Parse failure: {reason[:200]}",
    }


def run(input_data: dict) -> dict:
    """Main agent entry point. Takes screening inputs, returns structured assessment."""
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
