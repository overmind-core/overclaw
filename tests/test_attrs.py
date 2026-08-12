"""Regression guards for ``overmind.attrs`` constant values.

These tests pin the wire-format namespaces so an accidental rename does not
silently break trace ingestion.

The CANONICAL token/cost namespace the Overmind server rolls up is ``genai.*``
(see ``overbae/api/overmind_attrs.py`` + ``otlp.py::_build_span_usage``).  The
SDK emits those keys AND, alongside them, the OTel GenAI semconv ``gen_ai.*``
keys (defined here as ``OTEL_*``) so OTel-native consumers and the optimiser's
:mod:`overmind.optimize.trace_reader` keep resolving model + tokens.
"""

from __future__ import annotations

from overmind import attrs


class TestLLMNamespace:
    """Canonical LLM_* constants must live in the server's ``genai.*`` namespace."""

    def test_llm_model_uses_genai_namespace(self) -> None:
        assert attrs.LLM_MODEL == "genai.model"

    def test_llm_provider_uses_genai_namespace(self) -> None:
        assert attrs.LLM_PROVIDER == "genai.provider"

    def test_token_usage_uses_canonical_names(self) -> None:
        # The server rolls up prompt/completion (not the semconv input/output).
        assert attrs.LLM_PROMPT_TOKENS == "genai.prompt_tokens"
        assert attrs.LLM_COMPLETION_TOKENS == "genai.completion_tokens"
        assert attrs.LLM_TOTAL_TOKENS == "genai.total_tokens"

    def test_cost_key_is_canonical(self) -> None:
        assert attrs.LLM_COST == "genai.cost"

    def test_all_llm_constants_start_with_genai(self) -> None:
        offenders = [
            (name, value)
            for name, value in vars(attrs).items()
            if name.startswith("LLM_") and isinstance(value, str) and not value.startswith("genai.")
        ]
        assert offenders == [], (
            f"Found LLM_* constants outside the genai.* namespace: {offenders}. "
            "The backend ingest rolls up genai.* keys; OTel semconv keys live under OTEL_*."
        )

    def test_otel_semconv_aliases_stay_in_gen_ai_namespace(self) -> None:
        # The dual-emitted OTel semconv keys the trace_reader depends on.
        assert attrs.OTEL_LLM_REQUEST_MODEL == "gen_ai.request.model"
        assert attrs.OTEL_LLM_SYSTEM == "gen_ai.system"
        assert attrs.OTEL_LLM_USAGE_PROMPT_TOKENS == "gen_ai.usage.prompt_tokens"
        assert attrs.OTEL_LLM_USAGE_COMPLETION_TOKENS == "gen_ai.usage.completion_tokens"
        assert attrs.OTEL_LLM_USAGE_TOTAL_TOKENS == "gen_ai.usage.total_tokens"


class TestReaderEmitterAlignment:
    """The trace_reader's ``gen_ai.*`` lookups must match the dual-emitted OTEL_* keys."""

    def test_reader_lookups_match_emitted_constants(self) -> None:
        # trace_reader falls back to prompt_tokens/completion_tokens (see
        # overmind/optimize/trace_reader.py); the SDK emits those forms.
        reader_keys = {
            "gen_ai.request.model",
            "gen_ai.usage.prompt_tokens",
            "gen_ai.usage.completion_tokens",
            "gen_ai.usage.total_tokens",
        }
        emitted = {
            attrs.OTEL_LLM_REQUEST_MODEL,
            attrs.OTEL_LLM_USAGE_PROMPT_TOKENS,
            attrs.OTEL_LLM_USAGE_COMPLETION_TOKENS,
            attrs.OTEL_LLM_USAGE_TOTAL_TOKENS,
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

    # NOTE: ``PROJECT_ID``, ``LLM_PROMPT_TOKENS``/``LLM_COMPLETION_TOKENS``/
    # ``LLM_TOTAL_TOKENS``/``LLM_COST`` and ``TOOL_NAME``/``TOOL_ARG_KEYS``/
    # ``TOOL_ERROR`` were re-introduced in the full-tracing-richness work and
    # now HAVE emitters (resource/identity stamping, ``utils/llm.py``, the
    # ``@tool`` decorator), so they are intentionally NOT in this set.
    REMOVED_CONSTANTS = {
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


class TestEvalEnvelopeNamespace:
    """Eval envelope wire contract v1 — event names + event attributes.

    The platform parses ``Span.events`` against exactly these strings
    (emitters live in ``overmind/evals.py``), so a rename here silently
    breaks server-side evaluation.
    """

    def test_event_names_are_pinned(self) -> None:
        assert attrs.EVAL_EXPECTATION_EVENT == "overmind.eval.expectation"
        assert attrs.EVAL_CONTEXT_EVENT == "overmind.eval.context"
        assert attrs.EVAL_CHECKPOINT_EVENT == "overmind.eval.checkpoint"
        assert attrs.EVAL_CONVERSATION_END_EVENT == "overmind.eval.conversation_end"

    def test_event_attribute_keys_are_pinned(self) -> None:
        assert attrs.EVAL_SCHEMA_VERSION == "overmind.eval.schema_version"
        assert attrs.EVAL_PAYLOAD == "overmind.eval.payload"


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
