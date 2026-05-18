"""Regression-guard snapshots for :mod:`overmind.prompts.policy_generator`.

Phase 4.5 of the cleanup plan extracted the shared Markdown skeleton and JSON
schema into module-level constants.  These tests pin the SHA-256 of each of
the five composed prompts so any future change to the constants OR to a
prompt body fails fast — the maintainer must either accept the new hash
(intentional copy edit) or fix the regression.
"""

from __future__ import annotations

import hashlib

from overmind.prompts.policy_generator import (
    POLICY_FROM_CODE_PROMPT,
    POLICY_FROM_DOCUMENT_PROMPT,
    POLICY_GENERATION_PROMPT,
    POLICY_IMPROVE_PROMPT,
    POLICY_REFINE_PROMPT,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# Captured from the pre-refactor state of the file so the byte-for-byte
# guarantee survives any future tinkering with the building blocks.
_EXPECTED = {
    "POLICY_GENERATION_PROMPT": (3035, "a9132b95a0b11d6645c87a3c8d7ebe1c08ce0d3f4f927bb55e7e9d748c8c1b58"),
    "POLICY_FROM_DOCUMENT_PROMPT": (1639, "3c5c36694e66bcce"),
    "POLICY_FROM_CODE_PROMPT": (1685, "31bc1ecb95ecf088"),
    "POLICY_IMPROVE_PROMPT": (2273, "e12c64cd9b4b6e61"),
    "POLICY_REFINE_PROMPT": (1424, "0fd9b388f14c0788"),
}


class TestPromptByteIdentity:
    """Each prompt's composed length and SHA prefix must stay stable."""

    _PROMPTS = {
        "POLICY_GENERATION_PROMPT": POLICY_GENERATION_PROMPT,
        "POLICY_FROM_DOCUMENT_PROMPT": POLICY_FROM_DOCUMENT_PROMPT,
        "POLICY_FROM_CODE_PROMPT": POLICY_FROM_CODE_PROMPT,
        "POLICY_IMPROVE_PROMPT": POLICY_IMPROVE_PROMPT,
        "POLICY_REFINE_PROMPT": POLICY_REFINE_PROMPT,
    }

    def test_lengths_are_stable(self):
        for name, text in self._PROMPTS.items():
            expected_len, _ = _EXPECTED[name]
            assert len(text) == expected_len, (
                f"{name} length changed: {len(text)} != {expected_len}. "
                "If intentional, update tests/test_prompts_policy_generator.py."
            )

    def test_sha_prefix_is_stable(self):
        for name, text in self._PROMPTS.items():
            _, expected_sha = _EXPECTED[name]
            actual = _sha(text)
            assert actual.startswith(expected_sha[:16]), (
                f"{name} content changed: {actual[:16]} != {expected_sha[:16]}. "
                "If intentional, update tests/test_prompts_policy_generator.py."
            )


class TestPromptFormatPlaceholders:
    """All five prompts must still expose their original ``{placeholder}`` slots."""

    def test_generation_prompt_placeholders(self):
        for slot in ("{analysis_json}", "{decision_rules}", "{hard_constraints}", "{edge_cases}", "{terminology}", "{agent_name}"):
            assert slot in POLICY_GENERATION_PROMPT

    def test_from_document_prompt_placeholders(self):
        for slot in ("{analysis_json}", "{user_document}", "{agent_name}"):
            assert slot in POLICY_FROM_DOCUMENT_PROMPT

    def test_from_code_prompt_placeholders(self):
        for slot in ("{analysis_json}", "{agent_code_section}", "{agent_name}"):
            assert slot in POLICY_FROM_CODE_PROMPT

    def test_improve_prompt_placeholders(self):
        for slot in ("{analysis_json}", "{agent_code_section}", "{existing_policy}", "{agent_name}"):
            assert slot in POLICY_IMPROVE_PROMPT

    def test_refine_prompt_placeholders(self):
        for slot in ("{analysis_json}", "{current_md}", "{current_data_json}", "{feedback}", "{additions}", "{agent_name}"):
            assert slot in POLICY_REFINE_PROMPT

    def test_prompts_format_without_keyerror(self):
        """Every prompt must accept the full kwarg set its caller passes."""
        common = dict(
            analysis_json="A",
            decision_rules="B",
            hard_constraints="C",
            edge_cases="D",
            terminology="E",
            user_document="F",
            agent_code_section="G",
            existing_policy="H",
            current_md="I",
            current_data_json="J",
            feedback="K",
            additions="L",
            agent_name="demo",
        )
        for prompt in (
            POLICY_GENERATION_PROMPT,
            POLICY_FROM_DOCUMENT_PROMPT,
            POLICY_FROM_CODE_PROMPT,
            POLICY_IMPROVE_PROMPT,
            POLICY_REFINE_PROMPT,
        ):
            slots = {k for k in common if "{" + k + "}" in prompt}
            prompt.format(**{k: common[k] for k in slots})
