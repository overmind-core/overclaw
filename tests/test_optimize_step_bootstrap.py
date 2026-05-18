"""Tests for ``bootstrap_optimize_step`` (extracted from ``overmind/cli.py``).

The skill-driven optimize loop relies on three pre-``overmind.init()``
side effects:

1. Setting ``OVERMIND_API_KEY=skill-local-no-export`` when the user
   hasn't configured a real key (so the SDK can mint trace IDs locally
   without an interactive prompt).
2. Reading the W3C traceparent persisted by ``optimize-step init`` and
   exporting it as ``TRACEPARENT`` so child subprocesses stitch into the
   workflow trace.
3. Resolving the agent name from either ``args.agent`` or the skill
   state file so the dispatcher can load the right per-agent ``.env``.

These tests pin the contract of all three.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from overmind.commands.optimize_step_cmd import (
    bootstrap_optimize_step,
    resolve_agent_name_from_state,
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch):
    for key in ("OVERMIND_API_KEY", "TRACEPARENT", "OTEL_TRACEPARENT"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


class TestApiKeyPlaceholder:
    def test_installs_placeholder_when_unset(self, clean_env: pytest.MonkeyPatch):
        args = argparse.Namespace(step="init", agent="ag")
        bootstrap_optimize_step(args)
        import os

        assert os.environ["OVERMIND_API_KEY"] == "skill-local-no-export"

    def test_preserves_user_key(self, clean_env: pytest.MonkeyPatch):
        clean_env.setenv("OVERMIND_API_KEY", "real-key-123")
        args = argparse.Namespace(step="init", agent="ag")
        bootstrap_optimize_step(args)
        import os

        assert os.environ["OVERMIND_API_KEY"] == "real-key-123"


class TestTraceparentRehydration:
    def test_reads_traceparent_from_state(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ):
        tp = "00-0123456789abcdef0123456789abcdef-fedcba9876543210-01"
        state = tmp_path / "state.json"
        state.write_text(json.dumps({"agent_name": "ag", "traceparent": tp}))
        args = argparse.Namespace(step="diagnose", state=str(state))
        bootstrap_optimize_step(args)
        import os

        assert os.environ["TRACEPARENT"] == tp

    def test_init_step_does_not_rehydrate(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ):
        state = tmp_path / "state.json"
        state.write_text(json.dumps({"traceparent": "tp-should-not-load"}))
        args = argparse.Namespace(step="init", state=str(state))
        bootstrap_optimize_step(args)
        import os

        assert "TRACEPARENT" not in os.environ

    def test_existing_traceparent_wins(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ):
        clean_env.setenv("TRACEPARENT", "outer-traceparent")
        state = tmp_path / "state.json"
        state.write_text(json.dumps({"traceparent": "inner-traceparent"}))
        args = argparse.Namespace(step="diagnose", state=str(state))
        bootstrap_optimize_step(args)
        import os

        assert os.environ["TRACEPARENT"] == "outer-traceparent"

    def test_missing_state_file_does_not_raise(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ):
        args = argparse.Namespace(step="diagnose", state=str(tmp_path / "missing.json"))
        bootstrap_optimize_step(args)
        import os

        assert "TRACEPARENT" not in os.environ

    def test_blank_state_path_is_skipped(self, clean_env: pytest.MonkeyPatch):
        args = argparse.Namespace(step="diagnose", state="")
        bootstrap_optimize_step(args)
        import os

        assert "TRACEPARENT" not in os.environ


class TestAgentNameResolution:
    def test_uses_args_agent_when_present(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ):
        state = tmp_path / "state.json"
        state.write_text(json.dumps({"agent_name": "from-state"}))
        args = argparse.Namespace(step="init", agent="from-args", state=str(state))
        bootstrap_optimize_step(args)
        assert args.resolved_agent_name == "from-args"

    def test_falls_through_to_state_file(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ):
        state = tmp_path / "state.json"
        state.write_text(json.dumps({"agent_name": "from-state"}))
        args = argparse.Namespace(step="diagnose", state=str(state))
        bootstrap_optimize_step(args)
        assert args.resolved_agent_name == "from-state"

    def test_nested_config_agent_name(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ):
        state = tmp_path / "state.json"
        state.write_text(json.dumps({"config": {"agent_name": "from-config"}}))
        args = argparse.Namespace(step="diagnose", state=str(state))
        bootstrap_optimize_step(args)
        assert args.resolved_agent_name == "from-config"

    def test_returns_none_when_no_source(self, clean_env: pytest.MonkeyPatch):
        args = argparse.Namespace(step="diagnose")
        bootstrap_optimize_step(args)
        assert args.resolved_agent_name is None


class TestResolveAgentNameFromState:
    def test_returns_none_for_blank_path(self):
        assert resolve_agent_name_from_state(None) is None
        assert resolve_agent_name_from_state("") is None

    def test_returns_none_for_missing_file(self, tmp_path: Path):
        assert resolve_agent_name_from_state(str(tmp_path / "nope.json")) is None

    def test_returns_top_level_agent_name(self, tmp_path: Path):
        state = tmp_path / "state.json"
        state.write_text(json.dumps({"agent_name": "primary"}))
        assert resolve_agent_name_from_state(str(state)) == "primary"

    def test_returns_nested_agent_name(self, tmp_path: Path):
        state = tmp_path / "state.json"
        state.write_text(json.dumps({"config": {"agent_name": "nested"}}))
        assert resolve_agent_name_from_state(str(state)) == "nested"
