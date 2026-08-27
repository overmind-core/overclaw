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
]
