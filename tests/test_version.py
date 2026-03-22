"""Tests for package version and metadata."""

from __future__ import annotations

import re


class TestVersion:
    def test_version_exists(self):
        from overclaw import __version__

        assert __version__

    def test_version_is_semver(self):
        from overclaw import __version__

        assert re.match(r"^\d+\.\d+\.\d+", __version__)
