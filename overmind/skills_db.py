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
        slug="overmind-telemetry",
        description="Configure Overmind telemetry for your AI project",
        version="1.0",
        provider="overmind-core",
    ),
    Skill(
        name="Ponytail",
        slug="ponytail",
        description="Review the optimization report and make changes to the policy, eval spec, or dataset if needed.",
        version="1.0",
        provider="DietrichGebert",
    ),
]
