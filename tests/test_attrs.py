"""Regression guards for ``overmind.attrs`` constant values.

These tests pin the wire-format namespaces so an accidental rename
("gen_ai.*" → "genai.*" or similar) does not silently break trace
ingestion.  The bug this guards against was a "genai.*" emission
mismatched with the "gen_ai.*" lookup in :mod:`overmind.optimize.trace_reader`
that made the optimizer read empty model names and zero token counts
from every LLM span.
"""

from __future__ import annotations

from overmind import attrs


class TestLLMNamespace:
    """All LLM_* constants must live in the OTel GenAI semconv namespace."""

    def test_llm_model_uses_gen_ai_namespace(self) -> None:
        assert attrs.LLM_MODEL == "gen_ai.request.model"

    def test_llm_provider_uses_gen_ai_namespace(self) -> None:
        assert attrs.LLM_PROVIDER == "gen_ai.system"

    def test_token_usage_uses_semconv_names(self) -> None:
        # The semconv uses input_tokens / output_tokens (not prompt / completion).
        assert attrs.LLM_USAGE_PROMPT_TOKENS == "gen_ai.usage.input_tokens"
        assert attrs.LLM_USAGE_COMPLETION_TOKENS == "gen_ai.usage.output_tokens"
        assert attrs.LLM_USAGE_TOTAL_TOKENS == "gen_ai.usage.total_tokens"

    def test_all_llm_constants_start_with_gen_ai(self) -> None:
        offenders = [
            (name, value)
            for name, value in vars(attrs).items()
            if name.startswith("LLM_") and isinstance(value, str) and not value.startswith("gen_ai.")
        ]
        assert offenders == [], (
            f"Found LLM_* constants outside the gen_ai.* namespace: {offenders}. "
            "The optimizer trace_reader and backend ingest only recognize gen_ai.*."
        )


class TestReaderEmitterAlignment:
    """The trace_reader's lookup keys must match what utils/llm.py emits."""

    def test_reader_lookups_match_emitted_constants(self) -> None:
        # These are the literal keys read by overmind/optimize/trace_reader.py.
        # If we ever rename the emitted constants again, this test fails fast.
        reader_keys = {
            "gen_ai.request.model",
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
            "gen_ai.usage.total_tokens",
        }
        emitted = {
            attrs.LLM_MODEL,
            attrs.LLM_USAGE_PROMPT_TOKENS,
            attrs.LLM_USAGE_COMPLETION_TOKENS,
            attrs.LLM_USAGE_TOTAL_TOKENS,
        }
        assert reader_keys == emitted


class TestNoDeadConstants:
    """Standing guard against re-introducing constants nobody emits.

    The cleanup that introduced this test deleted 26 ``attrs.*``
    constants that were defined but never referenced anywhere — keys
    like ``PROJECT_ID``, the entire ``DOCTOR_*`` family for a removed
    ``overmind doctor`` command, legacy ``LLM_*`` token-name aliases,
    and ``INPUT_DATA``/``OUTPUT_DATA``/``SCORE`` that were documented
    as observable but never actually set.

    Re-introducing one of these (without also adding an emitter) silently
    grows the wire-format schema, which makes backend ingest harder to
    reason about.  This test fails fast if any return.
    """

    REMOVED_CONSTANTS = {
        "PROJECT_ID",
        "ITERATION_ID",
        "EXPERIMENT_ID",
        "EXPERIMENT_NAME",
        "DOCTOR_AGENT_NAME",
        "DOCTOR_BUNDLE_BUILT",
        "DOCTOR_BUNDLE_FILES",
        "DOCTOR_BUNDLE_RAW_CHARS",
        "DOCTOR_BUNDLE_PROMPT_CHARS",
        "DOCTOR_HAS_EVAL_SPEC",
        "DOCTOR_HAS_INSTRUMENTED_COPY",
        "SETUP_AGENT_NAME",
        "SETUP_POLICY_MARKDOWN",
        "SETUP_POLICY_DATA",
        "SETUP_REFINED_CRITERIA",
        "SETUP_FIXED_ELEMENTS",
        "OPTIMIZE_RUN_NAME",
        "OPTIMIZE_BEST_SCORE_AFTER",
        "OPTIMIZE_DATA_LEAKAGE_COUNT",
        "OPTIMIZE_REGRESSION_FAILURES",
        "LLM_MESSAGES_COUNT",
        "LLM_TOOLS_PROVIDED",
        "LLM_TOOL_CALLS",
        "LLM_PROMPT_TOKENS",
        "LLM_COMPLETION_TOKENS",
        "LLM_TOTAL_TOKENS",
        "LLM_COST",
        "TOOL_NAME",
        "TOOL_ARG_KEYS",
        "TOOL_ERROR",
        "INPUT_DATA",
        "OUTPUT_DATA",
        "SCORE",
    }

    def test_removed_constants_stay_removed(self) -> None:
        reintroduced = sorted(name for name in self.REMOVED_CONSTANTS if hasattr(attrs, name))
        assert reintroduced == [], (
            f"These attrs constants were intentionally deleted; if you really "
            f"need one, also add an emitter and update REMOVED_CONSTANTS: {reintroduced}"
        )


class TestErrorSummaryAlias:
    """``ERROR`` is a legacy alias — keep it pointing at ``ERROR_SUMMARY``."""

    def test_error_and_error_summary_resolve_to_same_key(self) -> None:
        assert attrs.ERROR == attrs.ERROR_SUMMARY
        assert attrs.ERROR_SUMMARY == "overmind.error"

    def test_error_type_and_message_use_dotted_subnamespace(self) -> None:
        assert attrs.ERROR_TYPE == "overmind.error.type"
        assert attrs.ERROR_MESSAGE == "overmind.error.message"


class TestOptimizeBestScoreDuplicate:
    """``OPTIMIZE_REPORT_BEST_SCORE`` is deprecated.  Pin its value so the
    OTLP ingest can keep reading it during the deprecation window, and so
    new code knows to emit only ``OPTIMIZE_FINAL_BEST_SCORE``.
    """

    def test_constants_are_distinct(self) -> None:
        assert attrs.OPTIMIZE_FINAL_BEST_SCORE != attrs.OPTIMIZE_REPORT_BEST_SCORE

    def test_canonical_key_value(self) -> None:
        assert attrs.OPTIMIZE_FINAL_BEST_SCORE == "overmind.optimize.final_best_score"

    def test_legacy_key_value(self) -> None:
        assert attrs.OPTIMIZE_REPORT_BEST_SCORE == "overmind.optimize.report_best_score"
