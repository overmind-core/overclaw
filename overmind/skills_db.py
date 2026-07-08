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
        name="Overmind Agent Builder",
        slug="overmind-agent-builder",
        description="Build an agent from scratch using natural language",
        version="1.0",
        provider="overmind-core",
    ),
    Skill(
        name="Overmind Telemetry",
        slug="overmind-agent-telemetry",
        description="Configure Overmind telemetry for your AI project",
        version="1.0",
        provider="overmind-core",
    ),
]
