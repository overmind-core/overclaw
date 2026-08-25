"""Types for the Overmind platform MCP proxy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ponytail: prefix heuristics — upgrade path is a server-side domain tag on each tool.
_DOMAIN_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("evals", ("list_eval", "create_eval", "get_eval", "delete_eval", "cancel_eval", "compare_eval", "update_eval")),
    ("workshop", ("workshop_", "clean_dataset", "dedupe_", "split_dataset")),
    ("finetune", ("finetune", "deploy_model", "list_finetune", "create_finetune", "cancel_finetune")),
    ("builds", ("list_build", "create_build", "get_build", "cancel_build")),
    ("capabilities", ("list_capabilities", "get_capability", "create_capability")),
    (
        "observability",
        ("list_trace", "get_trace", "list_session", "get_session", "list_span", "backfill_", "search_trace"),
    ),
    ("optimizer", ("optimizer", "backtest", "create_optimizer", "list_optimizer")),
    ("connectors", ("connector", "list_connector", "create_connector", "configure_connector")),
    ("inference", ("inference", "chat_completion", "list_model", "get_model", "delete_model")),
    ("graph", ("graph_", "graph_node", "graph_walk")),
)

KNOWN_DOMAINS = frozenset(prefix[0] for prefix in _DOMAIN_PREFIXES) | {"other"}


def infer_tool_domain(name: str) -> str:
    for domain, prefixes in _DOMAIN_PREFIXES:
        if any(name.startswith(prefix) for prefix in prefixes):
            return domain
    return "other"


@dataclass
class ToolSummary:
    name: str
    description: str
    domain: str = "other"


@dataclass
class ToolDetail:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    domain: str = "other"


@dataclass
class ToolCallResult:
    content: list[dict[str, Any]] = field(default_factory=list)
    is_error: bool = False
    structured_content: Any | None = None

    def text(self) -> str:
        parts: list[str] = []
        for block in self.content:
            if block.get("type") == "text" and block.get("text"):
                parts.append(block["text"])
        return "\n".join(parts)
