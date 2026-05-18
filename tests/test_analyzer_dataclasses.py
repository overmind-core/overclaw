"""Tests for :class:`DiagnosisContext` / :class:`CodegenSettings`.

The dataclasses are simple value objects today, but they're at the centre
of the analyzer's hot path so a few smoke tests keep the contract pinned
as the fields evolve.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from overmind.optimize.analyzer import (
    CodegenSettings,
    DiagnosisContext,
    _build_diagnosis_context_from_kwargs,
    _run_diagnosis,
)


class TestDiagnosisContextDefaults:
    def test_minimum_required_fields(self):
        ctx = DiagnosisContext(
            agent_code="def run(x): pass",
            case_results=[],
            evaluation_results={"avg_total": 0},
            model="m",
            entrypoint_fn="run",
        )
        assert ctx.allow_model_change is False
        assert ctx.temperature == 0.7
        assert ctx.iteration_seed == 42
        assert ctx.failed_attempts is None
        assert ctx.successful_changes is None
        assert ctx.bundle is None
        assert ctx.policy_context == ""
        assert ctx.cluster_context == ""
        assert ctx.component_weights_context == ""

    def test_round_trips_through_legacy_kwargs(self):
        kwargs = dict(
            agent_code="def run(x): pass",
            case_results=[{"i": 1}],
            evaluation_results={"avg_total": 50},
            model="m",
            eval_spec={"output_fields": {}},
            failed_attempts=[{"reason": "x"}],
            successful_changes=None,
            allow_model_change=True,
            temperature=0.4,
            iteration_seed=999,
            policy_context="p",
            entrypoint_fn="run",
            cluster_context="cl",
            component_weights_context="cw",
        )
        ctx = _build_diagnosis_context_from_kwargs((), kwargs)
        assert ctx.agent_code == "def run(x): pass"
        assert ctx.eval_spec == {"output_fields": {}}
        assert ctx.allow_model_change is True
        assert ctx.temperature == 0.4
        assert ctx.iteration_seed == 999
        assert ctx.policy_context == "p"
        assert ctx.cluster_context == "cl"
        assert ctx.component_weights_context == "cw"


class TestCodegenSettingsDefaults:
    def test_defaults(self):
        cs = CodegenSettings()
        assert cs.codegen_model == ""
        assert cs.codegen_max_steps == 50
        assert cs.policy_constraints == ""
        assert cs.agent_files is None
        assert cs.num_candidates == 3
        assert cs.return_plans_only is False
        assert cs.focus_weights is None


class TestRunDiagnosisAcceptsContext:
    """The dataclass-based call shape must behave the same as kwargs."""

    @patch("overmind.utils.llm.litellm")
    def test_context_path_matches_kwargs_path(self, mock_litellm):
        diagnosis = {"root_cause": "x", "changes": [{"action": "y"}]}
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = f"```json\n{json.dumps(diagnosis)}\n```"
        mock_litellm.completion.return_value = mock_resp

        ctx = DiagnosisContext(
            agent_code="def run(x): pass",
            case_results=[],
            evaluation_results={"avg_total": 50},
            model="m",
            entrypoint_fn="run",
        )
        via_ctx = _run_diagnosis(ctx=ctx)
        via_kwargs = _run_diagnosis(
            agent_code="def run(x): pass",
            case_results=[],
            evaluation_results={"avg_total": 50},
            model="m",
            eval_spec=None,
            failed_attempts=None,
            successful_changes=None,
            allow_model_change=False,
            temperature=0.7,
            entrypoint_fn="run",
        )
        assert via_ctx == via_kwargs == diagnosis
