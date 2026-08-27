"""Tests for the provider-instrumentation helpers in :mod:`overmind.tracing`.

Covers :func:`_enable_provider` plus the :func:`enable_tracing` dispatcher:

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

    def test_instruments_when_module_available(self):
        instrumentor_cls = MagicMock()
        module = MagicMock(DemoInstrumentor=instrumentor_cls)
        with (
            patch("importlib.util.find_spec", return_value=object()),
            patch("importlib.import_module", return_value=module),
        ):
            tracing._enable_provider("demo", "demo_module", "instr.demo", "DemoInstrumentor")

        instrumentor_cls.assert_called_once_with()
        instrumentor_cls.return_value.instrument.assert_called_once_with()
        assert "demo" in tracing._providers

    def test_is_idempotent_on_repeated_calls(self):
        instrumentor_cls = MagicMock()
        module = MagicMock(DemoInstrumentor=instrumentor_cls)
        with (
            patch("importlib.util.find_spec", return_value=object()),
            patch("importlib.import_module", return_value=module),
        ):
            tracing._enable_provider("demo", "demo_module", "instr.demo", "DemoInstrumentor")
            tracing._enable_provider("demo", "demo_module", "instr.demo", "DemoInstrumentor")

        # Second call short-circuits — the instrumentor only runs once.
        instrumentor_cls.assert_called_once()

    def test_warns_when_module_missing_in_non_strict_mode(self, caplog):
        tracing._strict_mode = False
        with patch("importlib.util.find_spec", return_value=None):
            tracing._enable_provider("demo", "demo_module", "instr.demo", "DemoInstrumentor")

        assert "demo" not in tracing._providers
        assert any("not installed" in rec.message for rec in caplog.records)

    def test_raises_when_module_missing_in_strict_mode(self):
        tracing._strict_mode = True
        with patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(ImportError, match="not installed"):
                tracing._enable_provider("demo", "demo_module", "instr.demo", "DemoInstrumentor")

    def test_dotted_module_name_collapsed_for_install_hint(self):
        """``google.genai`` shouldn't appear in `pip install google.genai`."""
        tracing._strict_mode = True
        with patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(ImportError, match=r"pip install google-genai"):
                tracing._enable_provider("google", "google.genai", "instr.google", "GoogleInstrumentor")

    def test_missing_extra_instrumentor_warns_with_extra_hint(self, caplog):
        """Target library installed but the extra-shipped instrumentor absent:
        point at the ``overmind[langchain]`` extra instead of crashing."""
        tracing._strict_mode = False
        gate, instrumentation, cls = tracing._PROVIDER_MODULES["langchain"]
        with patch("importlib.util.find_spec", side_effect=lambda mod: object() if mod == gate else None):
            tracing._enable_provider("langchain", gate, instrumentation, cls)

        assert "langchain" not in tracing._providers
        assert any("overmind[langchain]" in rec.message for rec in caplog.records)

    def test_missing_extra_instrumentor_raises_in_strict_mode(self):
        tracing._strict_mode = True
        gate, instrumentation, cls = tracing._PROVIDER_MODULES["langchain"]
        with patch("importlib.util.find_spec", side_effect=lambda mod: object() if mod == gate else None):
            with pytest.raises(ImportError, match=r"overmind\[langchain\]"):
                tracing._enable_provider("langchain", gate, instrumentation, cls)

    def test_missing_namespace_package_treated_as_absent(self):
        """find_spec raises ModuleNotFoundError when a dotted parent package
        (``openinference``) is not installed at all — must not propagate."""
        tracing._strict_mode = False

        def fake_find_spec(mod):
            if mod == "langchain_core":
                return object()
            raise ModuleNotFoundError(mod)

        with patch("importlib.util.find_spec", side_effect=fake_find_spec):
            gate, instrumentation, cls = tracing._PROVIDER_MODULES["langchain"]
            tracing._enable_provider("langchain", gate, instrumentation, cls)

        assert "langchain" not in tracing._providers

    def test_langchain_provider_registered(self):
        gate, instrumentation, cls = tracing._PROVIDER_MODULES["langchain"]
        assert gate == "langchain_core"
        assert instrumentation == "openinference.instrumentation.langchain"
        assert cls == "LangChainInstrumentor"


class TestEnableTracingDispatcher:
    def test_empty_list_enables_all_known_providers(self):
        with patch.object(tracing, "_enable_provider") as enable:
            tracing.enable_tracing([])

        enabled = [call.args[0] for call in enable.call_args_list]
        assert enabled == list(tracing._PROVIDER_MODULES)

    def test_none_short_circuits(self):
        with patch.object(tracing, "_enable_provider") as enable:
            tracing.enable_tracing(None)

        enable.assert_not_called()

    def test_unknown_provider_does_not_crash(self, caplog):
        tracing.enable_tracing(["definitely-not-real"])
        assert any("Unknown tracing provider" in rec.message for rec in caplog.records)

    def test_routes_named_providers_through_helper(self):
        with patch.object(tracing, "_enable_provider") as enable:
            tracing.enable_tracing(["openai", "anthropic"])

        enabled = [call.args[0] for call in enable.call_args_list]
        assert enabled == ["openai", "anthropic"]


class TestAutoProviders:
    """``providers="auto"`` — detect installed target libraries and enable
    every one whose instrumentor dependency is also present."""

    def test_enables_only_providers_with_library_and_instrumentor(self):
        openai_modules = {"openai", "opentelemetry.instrumentation.openai"}

        def installed(module):
            # anthropic's library present but its instrumentor absent; the
            # rest fully absent — only openai qualifies.
            return module in openai_modules or module == "anthropic"

        with (
            patch.object(tracing, "_module_installed", side_effect=installed),
            patch.object(tracing, "_enable_provider") as enable,
        ):
            tracing.enable_tracing("auto")

        assert [call.args[0] for call in enable.call_args_list] == ["openai"]

    def test_logs_resolved_list_at_info(self, caplog):
        detected = {"anthropic", "opentelemetry.instrumentation.anthropic", "openai", "opentelemetry.instrumentation.openai"}
        with (
            caplog.at_level("INFO"),
            patch.object(tracing, "_module_installed", side_effect=lambda m: m in detected),
            patch.object(tracing, "_enable_provider"),
        ):
            tracing.enable_tracing("auto")

        (record,) = [r for r in caplog.records if 'providers="auto"' in r.message]
        assert "openai, anthropic" in record.message

    def test_nothing_installed_resolves_to_none(self, caplog):
        with (
            caplog.at_level("INFO"),
            patch.object(tracing, "_module_installed", return_value=False),
            patch.object(tracing, "_enable_provider") as enable,
        ):
            tracing.enable_tracing("auto")

        enable.assert_not_called()
        assert any('resolved to: none' in r.message for r in caplog.records)

    def test_non_auto_string_raises(self):
        with pytest.raises(ValueError, match='"auto"'):
            tracing.enable_tracing("openai")
