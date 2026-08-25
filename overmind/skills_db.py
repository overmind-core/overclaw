from dataclasses import dataclass


@dataclass
class Skill:
    name: str
    slug: str
    description: str
    version: str
    provider: str


skills = [
    Skill(
        name="Overmind",
        slug="overmind",
        description=(
            "Operate the Overmind platform via MCP — tracing and per-agent "
            "telemetry, fine-tuning, dataset upload and cleaning, evals, and "
            "optimizer experiments"
        ),
        version="1.4",
        provider="overmind-core",
    ),
    Skill(
        name="Overmind platform",
        slug="overmind-platform",
        description=(
            "Operate the Overmind platform via the overmind platform CLI — "
            "discover tools with list/describe, execute with call, and poll "
            "long jobs with job_status"
        ),
        version="1.0",
        provider="overmind-core",
    ),
]
