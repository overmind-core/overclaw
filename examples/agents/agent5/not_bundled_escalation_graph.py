"""
Fake escalation DAG for docs / future work — exclude from agent bundle.
"""

from dataclasses import dataclass, field


@dataclass
class EscalationNode:
    name: str
    children: list[str] = field(default_factory=list)


FAKE_GRAPH: dict[str, EscalationNode] = {
    "tier1": EscalationNode("tier1", ["engineering", "billing"]),
    "engineering": EscalationNode("engineering", ["security"]),
    "billing": EscalationNode("billing", []),
    "security": EscalationNode("security", []),
}


def reachable(start: str, target: str) -> bool:
    seen: set[str] = set()
    stack = [start]
    while stack:
        n = stack.pop()
        if n == target:
            return True
        if n in seen:
            continue
        seen.add(n)
        node = FAKE_GRAPH.get(n)
        if node:
            stack.extend(node.children)
    return False
