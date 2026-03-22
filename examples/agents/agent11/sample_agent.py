"""
Sample Agent: Education Student Performance Analysis

Takes student context (ID, name, grade level, semester, subject of concern,
optional parent notes) and returns a structured performance assessment. This
file is the optimization target — OverClaw will iteratively improve it.
"""

import json

from overclaw.core.tracer import call_llm, call_tool

# === AGENT METADATA ===

AGENT_DESCRIPTION = """\
Education Student Performance Analysis Agent: Takes student_id, student_name,
grade_level (1-12), semester (fall/spring), subject_of_concern
(math/english/science/history/art/all), and optional parent_notes. Returns a
structured JSON assessment with performance_level, trend, strengths,
areas_for_improvement, recommended_interventions, parent_conference_needed,
gifted_program_candidate, and reasoning.
"""

# === AGENT CONFIGURATION (optimizable) ===

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """\
You are an education analyst assistant. Given a student's context, use the
available tools to retrieve grade history, attendance, and learning-style
assessment data. Synthesize a clear, evidence-based performance analysis.

Return your final answer as JSON with these fields:
- performance_level (one of: excelling, on_track, at_risk, critical)
- trend (one of: improving, stable, declining)
- strengths (list of subject names or short phrases)
- areas_for_improvement (list of strings)
- recommended_interventions (list of actionable strings)
- parent_conference_needed (boolean)
- gifted_program_candidate (boolean)
- reasoning (brief explanation tying tools data to conclusions)
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_grades",
            "description": "Retrieve a student's grade history across subjects, optionally filtered by semester and/or subject.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {
                        "type": "string",
                        "description": "School student identifier",
                    },
                    "semester": {
                        "type": "string",
                        "description": "Filter: fall or spring (omit for all semesters)",
                        "enum": ["fall", "spring"],
                    },
                    "subject": {
                        "type": "string",
                        "description": "Filter: math, english, science, history, art, or omit for all subjects",
                        "enum": ["math", "english", "science", "history", "art"],
                    },
                },
                "required": ["student_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_attendance",
            "description": "Check the student's attendance record, totals, and observed patterns for a semester or full year.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {
                        "type": "string",
                        "description": "School student identifier",
                    },
                    "semester": {
                        "type": "string",
                        "description": "fall, spring, or all",
                        "enum": ["fall", "spring", "all"],
                    },
                },
                "required": ["student_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_learning_style_assessment",
            "description": "Return learning style profile, behavioral notes, participation level, and group work effectiveness.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {
                        "type": "string",
                        "description": "School student identifier",
                    },
                },
                "required": ["student_id"],
            },
        },
    },
]

# === TOOL IMPLEMENTATIONS (fixed — do not optimize) ===

# ~12–15 students with grades across 5–6 subjects over fall and spring
GRADES_DB: dict[str, dict] = {
    "STU001": {
        "student_name": "Maya Patel",
        "records": [
            {"semester": "fall", "subject": "math", "pct": 99, "letter": "A+"},
            {"semester": "fall", "subject": "english", "pct": 98, "letter": "A+"},
            {"semester": "fall", "subject": "science", "pct": 97, "letter": "A+"},
            {"semester": "fall", "subject": "history", "pct": 96, "letter": "A"},
            {"semester": "fall", "subject": "art", "pct": 100, "letter": "A+"},
            {"semester": "spring", "subject": "math", "pct": 99, "letter": "A+"},
            {"semester": "spring", "subject": "english", "pct": 99, "letter": "A+"},
            {"semester": "spring", "subject": "science", "pct": 98, "letter": "A+"},
            {"semester": "spring", "subject": "history", "pct": 97, "letter": "A+"},
            {"semester": "spring", "subject": "art", "pct": 100, "letter": "A+"},
        ],
    },
    "STU002": {
        "student_name": "Jordan Lee",
        "records": [
            {"semester": "fall", "subject": "math", "pct": 62, "letter": "D-"},
            {"semester": "fall", "subject": "english", "pct": 94, "letter": "A"},
            {"semester": "fall", "subject": "science", "pct": 68, "letter": "D+"},
            {"semester": "fall", "subject": "history", "pct": 88, "letter": "B+"},
            {"semester": "fall", "subject": "art", "pct": 91, "letter": "A-"},
            {"semester": "spring", "subject": "math", "pct": 65, "letter": "D"},
            {"semester": "spring", "subject": "english", "pct": 95, "letter": "A"},
            {"semester": "spring", "subject": "science", "pct": 70, "letter": "C-"},
            {"semester": "spring", "subject": "history", "pct": 90, "letter": "A-"},
            {"semester": "spring", "subject": "art", "pct": 92, "letter": "A-"},
        ],
    },
    "STU003": {
        "student_name": "Chris Morgan",
        "records": [
            {"semester": "fall", "subject": "math", "pct": 82, "letter": "B-"},
            {"semester": "fall", "subject": "english", "pct": 85, "letter": "B"},
            {"semester": "fall", "subject": "science", "pct": 84, "letter": "B"},
            {"semester": "fall", "subject": "history", "pct": 86, "letter": "B"},
            {"semester": "fall", "subject": "art", "pct": 88, "letter": "B+"},
            {"semester": "spring", "subject": "math", "pct": 71, "letter": "C-"},
            {"semester": "spring", "subject": "english", "pct": 74, "letter": "C"},
            {"semester": "spring", "subject": "science", "pct": 72, "letter": "C-"},
            {"semester": "spring", "subject": "history", "pct": 73, "letter": "C"},
            {"semester": "spring", "subject": "art", "pct": 76, "letter": "C+"},
        ],
    },
    "STU004": {
        "student_name": "Sam Rivera",
        "records": [
            {"semester": "spring", "subject": "math", "pct": 78, "letter": "C+"},
            {"semester": "spring", "subject": "english", "pct": 80, "letter": "B-"},
            {"semester": "spring", "subject": "science", "pct": 77, "letter": "C+"},
        ],
    },
    "STU005": {
        "student_name": "Taylor Brooks",
        "records": [
            {"semester": "fall", "subject": "math", "pct": 58, "letter": "F"},
            {"semester": "fall", "subject": "english", "pct": 72, "letter": "C-"},
            {"semester": "fall", "subject": "science", "pct": 60, "letter": "D-"},
            {"semester": "fall", "subject": "history", "pct": 70, "letter": "C-"},
            {"semester": "fall", "subject": "art", "pct": 85, "letter": "B"},
            {"semester": "spring", "subject": "math", "pct": 62, "letter": "D-"},
            {"semester": "spring", "subject": "english", "pct": 74, "letter": "C"},
            {"semester": "spring", "subject": "science", "pct": 63, "letter": "D"},
            {"semester": "spring", "subject": "history", "pct": 72, "letter": "C-"},
            {"semester": "spring", "subject": "art", "pct": 86, "letter": "B"},
        ],
    },
    "STU006": {
        "student_name": "Riley Chen",
        "records": [
            {"semester": "fall", "subject": "math", "pct": 76, "letter": "C"},
            {"semester": "fall", "subject": "english", "pct": 78, "letter": "C+"},
            {"semester": "fall", "subject": "science", "pct": 75, "letter": "C"},
            {"semester": "fall", "subject": "history", "pct": 74, "letter": "C"},
            {"semester": "fall", "subject": "art", "pct": 80, "letter": "B-"},
            {"semester": "spring", "subject": "math", "pct": 73, "letter": "C-"},
            {"semester": "spring", "subject": "english", "pct": 76, "letter": "C"},
            {"semester": "spring", "subject": "science", "pct": 72, "letter": "C-"},
            {"semester": "spring", "subject": "history", "pct": 71, "letter": "C-"},
            {"semester": "spring", "subject": "art", "pct": 79, "letter": "C+"},
        ],
    },
    "STU007": {
        "student_name": "Casey Williams",
        "records": [
            {"semester": "fall", "subject": "math", "pct": 81, "letter": "B-"},
            {"semester": "fall", "subject": "english", "pct": 68, "letter": "D+"},
            {"semester": "fall", "subject": "science", "pct": 79, "letter": "C+"},
            {"semester": "fall", "subject": "history", "pct": 84, "letter": "B"},
            {"semester": "fall", "subject": "art", "pct": 77, "letter": "C+"},
            {"semester": "spring", "subject": "math", "pct": 74, "letter": "C"},
            {"semester": "spring", "subject": "english", "pct": 88, "letter": "B+"},
            {"semester": "spring", "subject": "science", "pct": 72, "letter": "C-"},
            {"semester": "spring", "subject": "history", "pct": 80, "letter": "B-"},
            {"semester": "spring", "subject": "art", "pct": 83, "letter": "B"},
        ],
    },
    "STU008": {
        "student_name": "Jamie Ortiz",
        "records": [
            {"semester": "fall", "subject": "math", "pct": 64, "letter": "D"},
            {"semester": "fall", "subject": "english", "pct": 66, "letter": "D"},
            {"semester": "fall", "subject": "science", "pct": 65, "letter": "D"},
            {"semester": "fall", "subject": "history", "pct": 67, "letter": "D+"},
            {"semester": "fall", "subject": "art", "pct": 70, "letter": "C-"},
            {"semester": "spring", "subject": "math", "pct": 63, "letter": "D"},
            {"semester": "spring", "subject": "english", "pct": 65, "letter": "D"},
            {"semester": "spring", "subject": "science", "pct": 64, "letter": "D"},
            {"semester": "spring", "subject": "history", "pct": 66, "letter": "D"},
            {"semester": "spring", "subject": "art", "pct": 69, "letter": "D+"},
        ],
    },
    "STU009": {
        "student_name": "Drew Nguyen",
        "records": [
            {"semester": "fall", "subject": "math", "pct": 68, "letter": "D+"},
            {"semester": "fall", "subject": "english", "pct": 72, "letter": "C-"},
            {"semester": "fall", "subject": "science", "pct": 70, "letter": "C-"},
            {"semester": "fall", "subject": "history", "pct": 73, "letter": "C"},
            {"semester": "fall", "subject": "art", "pct": 75, "letter": "C"},
            {"semester": "spring", "subject": "math", "pct": 86, "letter": "B"},
            {"semester": "spring", "subject": "english", "pct": 88, "letter": "B+"},
            {"semester": "spring", "subject": "science", "pct": 87, "letter": "B"},
            {"semester": "spring", "subject": "history", "pct": 89, "letter": "B+"},
            {"semester": "spring", "subject": "art", "pct": 90, "letter": "A-"},
        ],
    },
    "STU010": {
        "student_name": "Quinn Foster",
        "records": [
            {"semester": "fall", "subject": "math", "pct": 90, "letter": "A-"},
            {"semester": "fall", "subject": "english", "pct": 92, "letter": "A-"},
            {"semester": "fall", "subject": "science", "pct": 91, "letter": "A-"},
            {"semester": "fall", "subject": "history", "pct": 89, "letter": "B+"},
            {"semester": "fall", "subject": "art", "pct": 93, "letter": "A"},
            {"semester": "spring", "subject": "math", "pct": 91, "letter": "A-"},
            {"semester": "spring", "subject": "english", "pct": 93, "letter": "A"},
            {"semester": "spring", "subject": "science", "pct": 92, "letter": "A-"},
            {"semester": "spring", "subject": "history", "pct": 90, "letter": "A-"},
            {"semester": "spring", "subject": "art", "pct": 94, "letter": "A"},
        ],
    },
    "STU011": {
        "student_name": "Skyler Kim",
        "records": [
            {"semester": "fall", "subject": "math", "pct": 92, "letter": "A-"},
            {"semester": "fall", "subject": "english", "pct": 88, "letter": "B+"},
            {"semester": "fall", "subject": "science", "pct": 90, "letter": "A-"},
            {"semester": "fall", "subject": "history", "pct": 87, "letter": "B+"},
            {"semester": "fall", "subject": "art", "pct": 94, "letter": "A"},
            {"semester": "spring", "subject": "math", "pct": 93, "letter": "A-"},
            {"semester": "spring", "subject": "english", "pct": 89, "letter": "B+"},
            {"semester": "spring", "subject": "science", "pct": 91, "letter": "A-"},
            {"semester": "spring", "subject": "history", "pct": 88, "letter": "B+"},
            {"semester": "spring", "subject": "art", "pct": 95, "letter": "A"},
        ],
    },
    "STU012": {
        "student_name": "Avery Thompson",
        "records": [
            {"semester": "fall", "subject": "math", "pct": 92, "letter": "A-"},
            {"semester": "fall", "subject": "english", "pct": 94, "letter": "A"},
            {"semester": "fall", "subject": "science", "pct": 90, "letter": "A-"},
            {"semester": "fall", "subject": "history", "pct": 93, "letter": "A"},
            {"semester": "fall", "subject": "art", "pct": 88, "letter": "B+"},
            {"semester": "spring", "subject": "math", "pct": 93, "letter": "A"},
            {"semester": "spring", "subject": "english", "pct": 95, "letter": "A"},
            {"semester": "spring", "subject": "science", "pct": 91, "letter": "A-"},
            {"semester": "spring", "subject": "history", "pct": 94, "letter": "A"},
            {"semester": "spring", "subject": "art", "pct": 89, "letter": "B+"},
        ],
    },
    "STU013": {
        "student_name": "Reese Martinez",
        "records": [
            {"semester": "fall", "subject": "math", "pct": 61, "letter": "D-"},
            {"semester": "fall", "subject": "english", "pct": 63, "letter": "D"},
            {"semester": "fall", "subject": "science", "pct": 60, "letter": "D-"},
            {"semester": "fall", "subject": "history", "pct": 65, "letter": "D"},
            {"semester": "fall", "subject": "art", "pct": 96, "letter": "A"},
            {"semester": "spring", "subject": "math", "pct": 64, "letter": "D"},
            {"semester": "spring", "subject": "english", "pct": 66, "letter": "D"},
            {"semester": "spring", "subject": "science", "pct": 62, "letter": "D-"},
            {"semester": "spring", "subject": "history", "pct": 67, "letter": "D+"},
            {"semester": "spring", "subject": "art", "pct": 97, "letter": "A+"},
        ],
    },
    "STU014": {
        "student_name": "Min-jun Park",
        "records": [
            {"semester": "fall", "subject": "math", "pct": 72, "letter": "C-"},
            {"semester": "fall", "subject": "english", "pct": 58, "letter": "F"},
            {"semester": "fall", "subject": "science", "pct": 70, "letter": "C-"},
            {"semester": "fall", "subject": "history", "pct": 68, "letter": "D+"},
            {"semester": "fall", "subject": "art", "pct": 82, "letter": "B-"},
            {"semester": "spring", "subject": "math", "pct": 76, "letter": "C"},
            {"semester": "spring", "subject": "english", "pct": 65, "letter": "D"},
            {"semester": "spring", "subject": "science", "pct": 74, "letter": "C"},
            {"semester": "spring", "subject": "history", "pct": 72, "letter": "C-"},
            {"semester": "spring", "subject": "art", "pct": 85, "letter": "B"},
        ],
    },
    "STU015": {
        "student_name": "Harper Singh",
        "records": [
            {"semester": "fall", "subject": "math", "pct": 74, "letter": "C"},
            {"semester": "fall", "subject": "english", "pct": 76, "letter": "C+"},
            {"semester": "fall", "subject": "science", "pct": 73, "letter": "C"},
            {"semester": "fall", "subject": "history", "pct": 75, "letter": "C"},
            {"semester": "fall", "subject": "art", "pct": 77, "letter": "C+"},
            {"semester": "spring", "subject": "math", "pct": 75, "letter": "C"},
            {"semester": "spring", "subject": "english", "pct": 77, "letter": "C+"},
            {"semester": "spring", "subject": "science", "pct": 74, "letter": "C"},
            {"semester": "spring", "subject": "history", "pct": 76, "letter": "C+"},
            {"semester": "spring", "subject": "art", "pct": 78, "letter": "C+"},
        ],
    },
}

ATTENDANCE_DB: dict[str, dict] = {
    "STU001": {
        "total_instructional_days": 170,
        "absences": 1,
        "tardies": 0,
        "fall": {"absences": 0, "tardies": 0, "pattern": "excellent"},
        "spring": {"absences": 1, "tardies": 0, "pattern": "excellent"},
        "notes": "No concerns; consistent punctuality.",
    },
    "STU002": {
        "total_instructional_days": 170,
        "absences": 4,
        "tardies": 2,
        "fall": {"absences": 2, "tardies": 1, "pattern": "stable"},
        "spring": {"absences": 2, "tardies": 1, "pattern": "stable"},
        "notes": "Occasional absences; no chronic pattern.",
    },
    "STU003": {
        "total_instructional_days": 170,
        "absences": 18,
        "tardies": 12,
        "fall": {"absences": 5, "tardies": 4, "pattern": "worsening"},
        "spring": {"absences": 13, "tardies": 8, "pattern": "frequent_unexcused"},
        "notes": "Rising absences and tardies in spring; Friday pattern noted.",
    },
    "STU004": {
        "total_instructional_days": 45,
        "absences": 2,
        "tardies": 1,
        "fall": {"absences": 0, "tardies": 0, "pattern": "n_a_enrolled_spring"},
        "spring": {"absences": 2, "tardies": 1, "pattern": "limited_history"},
        "notes": "Enrolled mid-year; partial-year attendance only.",
    },
    "STU005": {
        "total_instructional_days": 170,
        "absences": 6,
        "tardies": 3,
        "fall": {"absences": 3, "tardies": 2, "pattern": "stable"},
        "spring": {"absences": 3, "tardies": 1, "pattern": "stable"},
        "notes": "Some pull-outs for services documented.",
    },
    "STU006": {
        "total_instructional_days": 170,
        "absences": 9,
        "tardies": 5,
        "fall": {"absences": 5, "tardies": 3, "pattern": "irregular"},
        "spring": {"absences": 4, "tardies": 2, "pattern": "improving"},
        "notes": "Absences clustered around documented family events.",
    },
    "STU007": {
        "total_instructional_days": 170,
        "absences": 11,
        "tardies": 15,
        "fall": {"absences": 4, "tardies": 6, "pattern": "game_week_volatility"},
        "spring": {"absences": 7, "tardies": 9, "pattern": "game_week_volatility"},
        "notes": "Higher tardies during competition weeks; travel conflicts.",
    },
    "STU008": {
        "total_instructional_days": 170,
        "absences": 3,
        "tardies": 1,
        "fall": {"absences": 2, "tardies": 0, "pattern": "stable"},
        "spring": {"absences": 1, "tardies": 1, "pattern": "stable"},
        "notes": "Present consistently; no truancy flags.",
    },
    "STU009": {
        "total_instructional_days": 170,
        "absences": 5,
        "tardies": 4,
        "fall": {"absences": 3, "tardies": 3, "pattern": "moderate"},
        "spring": {"absences": 2, "tardies": 1, "pattern": "improving"},
        "notes": "Attendance improved as grades improved.",
    },
    "STU010": {
        "total_instructional_days": 170,
        "absences": 22,
        "tardies": 18,
        "fall": {"absences": 12, "tardies": 10, "pattern": "high"},
        "spring": {"absences": 10, "tardies": 8, "pattern": "high"},
        "notes": "Strong grades despite frequent absences and tardies; chronic concern.",
    },
    "STU011": {
        "total_instructional_days": 170,
        "absences": 2,
        "tardies": 1,
        "fall": {"absences": 1, "tardies": 0, "pattern": "excellent"},
        "spring": {"absences": 1, "tardies": 1, "pattern": "excellent"},
        "notes": "Age-appropriate attendance for primary grades.",
    },
    "STU012": {
        "total_instructional_days": 170,
        "absences": 3,
        "tardies": 2,
        "fall": {"absences": 1, "tardies": 1, "pattern": "college_visit_days"},
        "spring": {"absences": 2, "tardies": 1, "pattern": "college_visit_days"},
        "notes": "Some absences for college visits (excused).",
    },
    "STU013": {
        "total_instructional_days": 170,
        "absences": 4,
        "tardies": 2,
        "fall": {"absences": 2, "tardies": 1, "pattern": "stable"},
        "spring": {"absences": 2, "tardies": 1, "pattern": "stable"},
        "notes": "Typical attendance.",
    },
    "STU014": {
        "total_instructional_days": 170,
        "absences": 4,
        "tardies": 3,
        "fall": {"absences": 2, "tardies": 2, "pattern": "stable"},
        "spring": {"absences": 2, "tardies": 1, "pattern": "stable"},
        "notes": "ESL support schedule may conflict occasionally; documented.",
    },
    "STU015": {
        "total_instructional_days": 170,
        "absences": 1,
        "tardies": 0,
        "fall": {"absences": 0, "tardies": 0, "pattern": "near_perfect"},
        "spring": {"absences": 1, "tardies": 0, "pattern": "near_perfect"},
        "notes": "Excellent attendance; effort visible in participation.",
    },
}

LEARNING_STYLE_DB: dict[str, dict] = {
    "STU001": {
        "dominant_style": "visual",
        "visual_score": 0.92,
        "auditory_score": 0.78,
        "kinesthetic_score": 0.71,
        "behavioral_notes": "Leads peer study groups; no conduct issues.",
        "participation_level": "high",
        "group_work_effectiveness": "excellent",
    },
    "STU002": {
        "dominant_style": "auditory",
        "visual_score": 0.72,
        "auditory_score": 0.88,
        "kinesthetic_score": 0.55,
        "behavioral_notes": "Engages strongly in discussion; math anxiety noted.",
        "participation_level": "high",
        "group_work_effectiveness": "good",
    },
    "STU003": {
        "dominant_style": "kinesthetic",
        "visual_score": 0.65,
        "auditory_score": 0.62,
        "kinesthetic_score": 0.78,
        "behavioral_notes": "Withdrawn recently; incomplete homework trend.",
        "participation_level": "low",
        "group_work_effectiveness": "fair",
    },
    "STU004": {
        "dominant_style": "visual",
        "visual_score": 0.8,
        "auditory_score": 0.7,
        "kinesthetic_score": 0.65,
        "behavioral_notes": "Still building rapport; limited baseline.",
        "participation_level": "medium",
        "group_work_effectiveness": "unknown_limited_data",
    },
    "STU005": {
        "dominant_style": "kinesthetic",
        "visual_score": 0.68,
        "auditory_score": 0.6,
        "kinesthetic_score": 0.82,
        "behavioral_notes": "Documented need for extended time and chunking; focus varies.",
        "participation_level": "medium",
        "group_work_effectiveness": "needs_support",
    },
    "STU006": {
        "dominant_style": "auditory",
        "visual_score": 0.7,
        "auditory_score": 0.75,
        "kinesthetic_score": 0.66,
        "behavioral_notes": "Cooperative; occasional fatigue reported.",
        "participation_level": "medium",
        "group_work_effectiveness": "good",
    },
    "STU007": {
        "dominant_style": "kinesthetic",
        "visual_score": 0.7,
        "auditory_score": 0.65,
        "kinesthetic_score": 0.85,
        "behavioral_notes": "Energetic; performance varies with schedule load.",
        "participation_level": "high",
        "group_work_effectiveness": "variable",
    },
    "STU008": {
        "dominant_style": "visual",
        "visual_score": 0.75,
        "auditory_score": 0.58,
        "kinesthetic_score": 0.62,
        "behavioral_notes": "Quiet; rarely volunteers; no disruptive behavior.",
        "participation_level": "low",
        "group_work_effectiveness": "fair",
    },
    "STU009": {
        "dominant_style": "mixed",
        "visual_score": 0.76,
        "auditory_score": 0.74,
        "kinesthetic_score": 0.73,
        "behavioral_notes": "Increased engagement after mentorship match in spring.",
        "participation_level": "medium",
        "group_work_effectiveness": "improving",
    },
    "STU010": {
        "dominant_style": "visual",
        "visual_score": 0.85,
        "auditory_score": 0.8,
        "kinesthetic_score": 0.72,
        "behavioral_notes": "Self-directed learner; completes work despite absences.",
        "participation_level": "high",
        "group_work_effectiveness": "good",
    },
    "STU011": {
        "dominant_style": "kinesthetic",
        "visual_score": 0.72,
        "auditory_score": 0.68,
        "kinesthetic_score": 0.86,
        "behavioral_notes": "Primary learner; responds well to hands-on tasks.",
        "participation_level": "high",
        "group_work_effectiveness": "good",
    },
    "STU012": {
        "dominant_style": "visual",
        "visual_score": 0.88,
        "auditory_score": 0.82,
        "kinesthetic_score": 0.7,
        "behavioral_notes": "Strong self-advocacy; college essay coaching engaged.",
        "participation_level": "high",
        "group_work_effectiveness": "excellent",
    },
    "STU013": {
        "dominant_style": "visual",
        "visual_score": 0.9,
        "auditory_score": 0.72,
        "kinesthetic_score": 0.8,
        "behavioral_notes": "Strong creative portfolio; STEM confidence lower.",
        "participation_level": "high",
        "group_work_effectiveness": "good",
    },
    "STU014": {
        "dominant_style": "auditory",
        "visual_score": 0.7,
        "auditory_score": 0.68,
        "kinesthetic_score": 0.72,
        "behavioral_notes": "ESL: stronger in math/visual tasks than verbal English.",
        "participation_level": "medium",
        "group_work_effectiveness": "fair",
    },
    "STU015": {
        "dominant_style": "auditory",
        "visual_score": 0.72,
        "auditory_score": 0.76,
        "kinesthetic_score": 0.7,
        "behavioral_notes": "Consistent effort; test anxiety affects demonstration of knowledge.",
        "participation_level": "medium",
        "group_work_effectiveness": "good",
    },
}


def lookup_grades(
    student_id: str,
    semester: str | None = None,
    subject: str | None = None,
) -> dict:
    sid = student_id.strip().upper()
    if sid not in GRADES_DB:
        return {"found": False, "error": "Unknown student_id", "grades": []}
    entry = GRADES_DB[sid]
    rows = list(entry["records"])
    if semester:
        sem = semester.lower().strip()
        rows = [r for r in rows if r["semester"] == sem]
    if subject and subject.lower().strip() != "all":
        sub = subject.lower().strip()
        rows = [r for r in rows if r["subject"] == sub]
    return {
        "found": True,
        "student_id": sid,
        "student_name": entry["student_name"],
        "grades": rows,
    }


def check_attendance(student_id: str, semester: str = "all") -> dict:
    sid = student_id.strip().upper()
    if sid not in ATTENDANCE_DB:
        return {"found": False, "error": "Unknown student_id"}
    data = {k: v for k, v in ATTENDANCE_DB[sid].items()}
    sem = semester.lower().strip() if semester else "all"
    if sem in ("fall", "spring"):
        return {
            "found": True,
            "student_id": sid,
            "semester_filter": sem,
            "summary": {
                "absences": data.get(sem, {}).get("absences"),
                "tardies": data.get(sem, {}).get("tardies"),
                "pattern": data.get(sem, {}).get("pattern"),
            },
            "full_record": data,
        }
    return {
        "found": True,
        "student_id": sid,
        "semester_filter": "all",
        "full_record": data,
    }


def get_learning_style_assessment(student_id: str) -> dict:
    sid = student_id.strip().upper()
    if sid not in LEARNING_STYLE_DB:
        return {"found": False, "error": "Unknown student_id"}
    return {"found": True, "student_id": sid, "assessment": LEARNING_STYLE_DB[sid]}


TOOL_FUNCTIONS = {
    "lookup_grades": lookup_grades,
    "check_attendance": check_attendance,
    "get_learning_style_assessment": get_learning_style_assessment,
}

# === AGENT LOGIC ===


def format_input(input_data: dict) -> str:
    parts = []
    if "student_id" in input_data:
        parts.append(f"Student ID: {input_data['student_id']}")
    if "student_name" in input_data:
        parts.append(f"Student name: {input_data['student_name']}")
    if "grade_level" in input_data:
        parts.append(f"Grade level: {input_data['grade_level']}")
    if "semester" in input_data:
        parts.append(f"Semester: {input_data['semester']}")
    if "subject_of_concern" in input_data:
        parts.append(f"Subject of concern: {input_data['subject_of_concern']}")
    if input_data.get("parent_notes"):
        parts.append(f"Parent notes: {input_data['parent_notes']}")
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
        "performance_level": "unknown",
        "trend": "unknown",
        "strengths": [],
        "areas_for_improvement": [],
        "recommended_interventions": [],
        "parent_conference_needed": False,
        "gifted_program_candidate": False,
        "reasoning": f"Parse failure: {reason[:200]}",
    }


def run(input_data: dict) -> dict:
    """Main agent entry point. Takes student context, returns structured analysis."""
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
