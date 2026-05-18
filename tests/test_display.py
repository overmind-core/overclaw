"""Tests for overmind.utils.display — non-interactive fallbacks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rich.console import Console

from overmind.utils import display
from overmind.utils.display import (
    confirm_option,
    is_non_interactive,
    render_criteria_table,
    select_option,
)


class TestRenderCriteriaTable:
    """The new shared Rich criteria table renderer."""

    @staticmethod
    def _criteria() -> dict:
        return {
            "fields": {
                "priority": {"importance": "critical", "partial_credit": True},
                "score": {"importance": "important", "tolerance": 5},
                "notes": {"importance": "minor", "eval_mode": "non_empty"},
            },
            "structure_weight": 20,
        }

    @staticmethod
    def _output_schema() -> dict:
        return {
            "priority": {"type": "enum"},
            "score": {"type": "number"},
            "notes": {"type": "text"},
        }

    def _capture(self, **kwargs) -> str:
        console = Console(record=True, force_terminal=False, width=120)
        render_criteria_table(console, self._criteria(), self._output_schema(), **kwargs)
        return console.export_text()

    def test_renders_one_row_per_field_plus_structure(self):
        rendered = self._capture()
        for name in ("priority", "score", "notes", "structure"):
            assert name in rendered

    def test_scoring_details_match_field_types(self):
        rendered = self._capture()
        assert "partial credit" in rendered
        assert "tolerance" in rendered
        assert "non-empty" in rendered or "non empty" in rendered

    def test_default_title_proposed(self):
        rendered = self._capture()
        assert "Proposed Evaluation Criteria" in rendered

    def test_custom_title_used(self):
        rendered = self._capture(title="Refined Evaluation Criteria")
        assert "Refined Evaluation Criteria" in rendered
        assert "Proposed Evaluation Criteria" not in rendered

    def test_no_fields_prints_nothing(self):
        console = Console(record=True, force_terminal=False, width=120)
        render_criteria_table(console, {"fields": {}}, {}, title="Nope")
        assert "Nope" not in console.export_text()

    def test_structure_weight_rendered(self):
        rendered = self._capture()
        assert "20 pts" in rendered


# ---------------------------------------------------------------------------
# is_non_interactive
# ---------------------------------------------------------------------------


class TestIsNonInteractive:
    """Non-interactive mode is **opt-in only** so piping stdout in a real
    terminal does not silently kill the arrow-key menu."""

    def test_env_flag_truthy(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OVERMIND_NONINTERACTIVE", "1")
        assert is_non_interactive() is True

    def test_env_flag_yes(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OVERMIND_NONINTERACTIVE", "yes")
        assert is_non_interactive() is True

    def test_env_flag_falsy_value(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OVERMIND_NONINTERACTIVE", "0")
        monkeypatch.delenv("CI", raising=False)
        assert is_non_interactive() is False

    def test_ci_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OVERMIND_NONINTERACTIVE", raising=False)
        monkeypatch.setenv("CI", "true")
        assert is_non_interactive() is True

    def test_default_is_interactive(self, monkeypatch: pytest.MonkeyPatch):
        """No env signal -> stay interactive, even if stdout happens to be a pipe."""
        monkeypatch.delenv("OVERMIND_NONINTERACTIVE", raising=False)
        monkeypatch.delenv("CI", raising=False)
        assert is_non_interactive() is False


# ---------------------------------------------------------------------------
# select_option
# ---------------------------------------------------------------------------


class TestSelectOption:
    def test_non_interactive_returns_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OVERMIND_NONINTERACTIVE", "1")
        console = MagicMock()
        idx = select_option(
            ["A", "B", "C"], title="Pick", default_index=2, console=console
        )
        assert idx == 2

    def test_non_interactive_clamps_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OVERMIND_NONINTERACTIVE", "1")
        console = MagicMock()
        idx = select_option(["A", "B"], default_index=99, console=console)
        assert idx == 1

    def test_empty_options_raises(self):
        with pytest.raises(ValueError):
            select_option([])

    def test_tty_menu_normal(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OVERMIND_NONINTERACTIVE", raising=False)
        monkeypatch.delenv("CI", raising=False)
        with patch("overmind.utils.display.TerminalMenu") as mock_menu:
            mock_menu.return_value.show.return_value = 1
            idx = select_option(["A", "B", "C"], default_index=0)
            assert idx == 1

    def test_oserror_falls_back_to_text_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("OVERMIND_NONINTERACTIVE", raising=False)
        monkeypatch.delenv("CI", raising=False)
        with patch("overmind.utils.display.TerminalMenu") as mock_menu, patch(
            "overmind.utils.display.IntPrompt.ask", return_value=2
        ) as mock_int_ask:
            mock_menu.return_value.show.side_effect = OSError(
                6, "Device not configured"
            )
            idx = select_option(["A", "B", "C"], default_index=0)
            assert idx == 1  # 1-based input "2" -> 0-based 1
            mock_int_ask.assert_called_once()

    def test_oserror_then_eof_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Sandbox with no /dev/tty AND closed stdin -> use default_index."""
        monkeypatch.delenv("OVERMIND_NONINTERACTIVE", raising=False)
        monkeypatch.delenv("CI", raising=False)
        with patch("overmind.utils.display.TerminalMenu") as mock_menu, patch(
            "overmind.utils.display.IntPrompt.ask", side_effect=EOFError
        ):
            mock_menu.return_value.show.side_effect = OSError(
                6, "Device not configured"
            )
            idx = select_option(["A", "B", "C"], default_index=2)
            assert idx == 2


# ---------------------------------------------------------------------------
# confirm_option
# ---------------------------------------------------------------------------


class TestConfirmOption:
    def test_non_interactive_returns_default_true(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("OVERMIND_NONINTERACTIVE", "1")
        assert confirm_option("Continue?", default=True) is True

    def test_non_interactive_returns_default_false(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("OVERMIND_NONINTERACTIVE", "1")
        assert confirm_option("Reconfigure?", default=False) is False

    def test_tty_menu_normal_yes(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OVERMIND_NONINTERACTIVE", raising=False)
        monkeypatch.delenv("CI", raising=False)
        with patch("overmind.utils.display.TerminalMenu") as mock_menu:
            mock_menu.return_value.show.return_value = 0
            assert confirm_option("Continue?", default=True) is True

    def test_oserror_falls_back_to_text_confirm(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("OVERMIND_NONINTERACTIVE", raising=False)
        monkeypatch.delenv("CI", raising=False)
        with patch("overmind.utils.display.TerminalMenu") as mock_menu, patch(
            "overmind.utils.display.Confirm.ask", return_value=True
        ) as mock_confirm_ask:
            mock_menu.return_value.show.side_effect = OSError(
                6, "Device not configured"
            )
            assert confirm_option("Continue?", default=False) is True
            mock_confirm_ask.assert_called_once()

    def test_oserror_then_eof_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """No /dev/tty AND closed stdin -> use the supplied default."""
        monkeypatch.delenv("OVERMIND_NONINTERACTIVE", raising=False)
        monkeypatch.delenv("CI", raising=False)
        with patch("overmind.utils.display.TerminalMenu") as mock_menu, patch(
            "overmind.utils.display.Confirm.ask", side_effect=EOFError
        ):
            mock_menu.return_value.show.side_effect = OSError(
                6, "Device not configured"
            )
            assert confirm_option("Reconfigure?", default=False) is False
            assert confirm_option("Continue?", default=True) is True


# ---------------------------------------------------------------------------
# Sanity: defaults trickle through unrelated imports
# ---------------------------------------------------------------------------


def test_is_non_interactive_exported():
    """The helper is part of the public surface of display.py."""
    assert "is_non_interactive" in display.__all__
