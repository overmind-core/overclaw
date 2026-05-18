"""Tests for the provider-instrumentation helpers in :mod:`overmind.tracing`.

Covers the unified :func:`_enable_provider` helper plus the four
``enable_*`` public wrappers:

* idempotency — repeated calls don't double-instrument
* missing-upstream behaviour — strict mode raises, normal mode logs a warning
* :func:`enable_tracing` dispatches every listed provider exactly once and
  flags unknown names without crashing
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from overmind import tracing


@pytest.fixture(autouse=True)
def _reset_provider_state():
    """Ensure each test starts with a clean ``_providers`` set and strict-mode flag."""
    original = set(tracing._providers)
    tracing._providers.clear()
    original_strict = tracing._strict_mode
    yield
    tracing._providers.clear()
    tracing._providers.update(original)
    tracing._strict_mode = original_strict


class TestEnableProvider:
    """Direct tests for the shared :func:`_enable_provider` helper."""

    def test_invokes_instrumentor_when_module_available(self):
        factory = MagicMock()
        with patch("importlib.util.find_spec", return_value=object()):
            tracing._enable_provider("demo", "demo_module", factory)

        factory.assert_called_once_with()
        factory.return_value.instrument.assert_called_once_with()
        assert "demo" in tracing._providers

    def test_is_idempotent_on_repeated_calls(self):
        factory = MagicMock()
        with patch("importlib.util.find_spec", return_value=object()):
            tracing._enable_provider("demo", "demo_module", factory)
            tracing._enable_provider("demo", "demo_module", factory)

        # Second call short-circuits — factory only runs once.
        factory.assert_called_once()

    def test_warns_when_module_missing_in_non_strict_mode(self, caplog):
        tracing._strict_mode = False
        factory = MagicMock()
        with patch("importlib.util.find_spec", return_value=None):
            tracing._enable_provider("demo", "demo_module", factory)

        factory.assert_not_called()
        assert "demo" not in tracing._providers
        assert any("not installed" in rec.message for rec in caplog.records)

    def test_raises_when_module_missing_in_strict_mode(self):
        tracing._strict_mode = True
        factory = MagicMock()
        with patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(ImportError, match="not installed"):
                tracing._enable_provider("demo", "demo_module", factory)

        factory.assert_not_called()

    def test_dotted_module_name_collapsed_for_install_hint(self):
        """``google.genai`` shouldn't appear in `pip install google.genai`."""
        tracing._strict_mode = True
        with patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(ImportError, match=r"pip install google-genai"):
                tracing._enable_provider("google", "google.genai", MagicMock())


class TestEnableTracingDispatcher:
    def test_empty_list_enables_all_known_providers(self):
        with patch.object(tracing, "_PROVIDER_ENABLERS", new={"a": MagicMock(), "b": MagicMock()}) as enablers:
            tracing.enable_tracing([])

        for enabler in enablers.values():
            enabler.assert_called_once_with()

    def test_none_short_circuits(self):
        with patch.object(tracing, "_PROVIDER_ENABLERS", new={"a": MagicMock()}) as enablers:
            tracing.enable_tracing(None)

        enablers["a"].assert_not_called()

    def test_unknown_provider_does_not_crash(self, caplog):
        tracing.enable_tracing(["definitely-not-real"])
        assert any("Unknown tracing provider" in rec.message for rec in caplog.records)

    def test_routes_named_providers_through_helpers(self):
        mock_openai = MagicMock()
        mock_anthropic = MagicMock()
        with patch.object(
            tracing,
            "_PROVIDER_ENABLERS",
            new={"openai": mock_openai, "anthropic": mock_anthropic, "google": MagicMock()},
        ):
            tracing.enable_tracing(["openai", "anthropic"])

        mock_openai.assert_called_once_with()
        mock_anthropic.assert_called_once_with()


class TestPublicEnableWrappers:
    """The four public ``enable_*`` functions delegate to the shared helper."""

    def test_all_public_enablers_route_through_helper(self):
        names = ("agno", "openai", "anthropic", "google")
        with patch.object(tracing, "_enable_provider") as enable:
            tracing.enable_agno()
            tracing.enable_openai()
            tracing.enable_anthropic()
            tracing.enable_google_genai()

        actual = [call.args[0] for call in enable.call_args_list]
        assert actual == list(names)
