"""Agent metadata for registry and tooling."""

AGENT_DESCRIPTION = """\
Legal Contract Review Agent: Takes contract_type (NDA, SaaS, employment,
lease, or services), contract_text, jurisdiction (US-CA, US-NY, US-TX, UK,
EU), and party_role (drafter, reviewer, or signee). Returns structured JSON
with risk_level, issues_found, missing_clauses, favorable_terms,
unfavorable_terms, overall_recommendation, and reasoning.
"""
