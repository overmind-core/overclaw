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
        name="Overmind Telemetry",
        slug="overmind-agent-telemetry",
        description=(
            "Add Overmind tracing, verify and audit instrumentation by fetching "
            "real traces, query traces via the REST API, and look up current docs"
        ),
        version="1.1",
        provider="overmind-core",
    ),
    Skill(
        name="Overmind",
        slug="overmind",
        description=(
            "Operate the Overmind platform via MCP — telemetry, fine-tuning, "
            "dataset upload and cleaning, evals, and optimizer experiments"
        ),
        version="1.2",
        provider="overmind-core",
    ),
]
