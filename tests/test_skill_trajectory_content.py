"""Guard the overmind skill docs against silent drift from the SDK surface.

The trajectory-instrumentation content — the runtime eval envelope helpers,
the Behaviour Registry anchor attributes, and the platform's locked MCP tool
names — must exist in the skill tree that `overmind skills sync overmind`
copies to installers. If a name here stops matching, either the SDK/docs
moved or the skill wasn't updated; one of them must be reconciled.
"""

from __future__ import annotations

from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1] / "skills" / "overmind"

ENVELOPE_FUNCTIONS = [
    "intent(",
    "checkpoint(",
    "expect(",
    "eval_context(",
    "end_conversation()",
]

BEHAVIOUR_ANCHOR_ATTRIBUTES = [
    "code.namespace",
    "code.function.name",
    "vcs.ref.head.revision",
]

MCP_TOOLS = [
    "get_instrumentation_context",
    "behaviour_coverage",
    "behaviour_deviations",
    "list_behaviours",
    "list_task_executions",
    "get_task_execution",
]


def _docs_text() -> str:
    files = sorted((SKILL_DIR / "references").glob("*.md")) + [SKILL_DIR / "SKILL.md"]
    return "\n".join(path.read_text() for path in files)


def test_skill_tree_mentions_runtime_envelope_functions() -> None:
    text = _docs_text()
    for fn in ENVELOPE_FUNCTIONS:
        assert fn in text


def test_skill_tree_mentions_behaviour_anchor_attributes() -> None:
    text = _docs_text()
    for attr in BEHAVIOUR_ANCHOR_ATTRIBUTES:
        assert attr in text


def test_skill_tree_mentions_locked_mcp_tool_names() -> None:
    text = _docs_text()
    for tool in MCP_TOOLS:
        assert tool in text
