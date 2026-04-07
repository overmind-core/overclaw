"""Tests for overclaw.utils.display — brand constants and logo rendering."""

from __future__ import annotations

from unittest.mock import MagicMock

from rich.console import Console

from overclaw.utils.display import BRAND, _load_logo_grid, render_logo


class TestBrandConstant:
    def test_is_hex_color(self):
        assert BRAND.startswith("#")
        assert len(BRAND) == 7


class TestLoadLogoGrid:
    def test_returns_list(self):
        import overclaw.utils.display as branding

        branding._logo_grid_cache = None
        grid = _load_logo_grid()
        assert isinstance(grid, list)

    def test_cache_reused(self):
        import overclaw.utils.display as branding

        branding._logo_grid_cache = None
        grid1 = _load_logo_grid()
        grid2 = _load_logo_grid()
        assert grid1 is grid2


class TestRenderLogo:
    def test_renders_without_error(self):
        console = Console(file=MagicMock(), force_terminal=True)
        render_logo(console)

    def test_small_mode(self):
        console = Console(file=MagicMock(), force_terminal=True)
        render_logo(console, small=True)

    def test_empty_grid(self):
        import overclaw.utils.display as branding

        original = branding._logo_grid_cache
        branding._logo_grid_cache = []
        console = Console(file=MagicMock(), force_terminal=True)
        render_logo(console)
        branding._logo_grid_cache = original
