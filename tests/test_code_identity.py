"""Round-trip tests for automatic identity stamping.

``observe()`` must stamp ``code.namespace`` / ``code.function.name`` on every
decorated span (all span types), and ``init()``'s sha detection must resolve
the running commit from env vars or ``.git/HEAD``.  Uses the repo's in-memory
span exporter pattern.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from overmind import attrs
from overmind.tracing import (
    _GIT_SHA_ENV_VARS,
    _detect_git_sha,
    entry_point,
    function,
    observe,
    retrieval,
    tool,
    workflow,
)


@pytest.fixture
def inmem(monkeypatch):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr("overmind.tracing.get_tracer", lambda: provider.get_tracer("test"))
    return provider, exporter


def _only_span(inmem):
    provider, exporter = inmem
    provider.force_flush()
    (span,) = exporter.get_finished_spans()
    return span


# ---------------------------------------------------------------------------
# code.namespace / code.function.name
# ---------------------------------------------------------------------------


def test_sync_function_carries_code_identity(inmem):
    @function()
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    span = _only_span(inmem)
    assert span.attributes[attrs.CODE_NAMESPACE] == __name__
    assert span.attributes[attrs.CODE_FUNCTION_NAME] == "test_sync_function_carries_code_identity.<locals>.add"


def test_async_function_carries_code_identity(inmem):
    @function()
    async def fetch(x):
        return x

    assert asyncio.run(fetch(7)) == 7
    span = _only_span(inmem)
    assert span.attributes[attrs.CODE_NAMESPACE] == __name__
    assert span.attributes[attrs.CODE_FUNCTION_NAME] == "test_async_function_carries_code_identity.<locals>.fetch"


def test_method_qualname_includes_class(inmem):
    class Greeter:
        @tool()
        def greet(self, name):
            return f"hi {name}"

    assert Greeter().greet("bob") == "hi bob"
    span = _only_span(inmem)
    assert span.attributes[attrs.CODE_FUNCTION_NAME] == "test_method_qualname_includes_class.<locals>.Greeter.greet"


@pytest.mark.parametrize("decorator", [observe, function, entry_point, workflow, tool, retrieval])
def test_all_decorators_stamp_code_identity(inmem, decorator):
    @decorator()
    def op():
        return 1

    assert op() == 1
    span = _only_span(inmem)
    assert span.attributes[attrs.CODE_NAMESPACE] == __name__
    assert span.attributes[attrs.CODE_FUNCTION_NAME].endswith("<locals>.op")


def test_unwraps_to_original_function(inmem):
    def original():
        return 42

    def anonymous_wrapper():
        return original()

    anonymous_wrapper.__wrapped__ = original

    traced = function()(anonymous_wrapper)
    assert traced() == 42
    span = _only_span(inmem)
    assert span.attributes[attrs.CODE_FUNCTION_NAME] == "test_unwraps_to_original_function.<locals>.original"


# ---------------------------------------------------------------------------
# vcs.ref.head.revision — git sha detection
# ---------------------------------------------------------------------------


@pytest.fixture
def no_sha_env(monkeypatch):
    for var in _GIT_SHA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _fake_git_repo(root: Path, sha: str = "a" * 40) -> str:
    git = root / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text(sha + "\n")
    return sha


def test_sha_from_head_ref_file(tmp_path, no_sha_env):
    sha = _fake_git_repo(tmp_path)
    assert _detect_git_sha(tmp_path) == sha


def test_sha_walks_up_from_subdirectory(tmp_path, no_sha_env):
    sha = _fake_git_repo(tmp_path)
    nested = tmp_path / "src" / "pkg"
    nested.mkdir(parents=True)
    assert _detect_git_sha(nested) == sha


def test_sha_detached_head(tmp_path, no_sha_env):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("b" * 40 + "\n")
    assert _detect_git_sha(tmp_path) == "b" * 40


def test_sha_from_packed_refs(tmp_path, no_sha_env):
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "packed-refs").write_text("# pack-refs with: peeled fully-peeled\n" + "c" * 40 + " refs/heads/main\n")
    assert _detect_git_sha(tmp_path) == "c" * 40


def test_env_var_override_wins(tmp_path, no_sha_env, monkeypatch):
    _fake_git_repo(tmp_path, sha="d" * 40)
    monkeypatch.setenv("OVERMIND_GIT_SHA", "e" * 40)
    assert _detect_git_sha(tmp_path) == "e" * 40


def test_undetectable_returns_none(tmp_path, no_sha_env):
    assert _detect_git_sha(tmp_path) is None


# ---------------------------------------------------------------------------
# init() wiring — sha lands on (or stays off) the OTel Resource
# ---------------------------------------------------------------------------


def _init_and_capture_resource(monkeypatch):
    from overmind import tracing

    monkeypatch.setattr(tracing, "_initialized", False)
    monkeypatch.setattr(tracing, "_tracer", None)
    monkeypatch.setattr(tracing, "OTLPSpanExporter", lambda **kw: MagicMock())
    captured = {}
    monkeypatch.setattr(
        "overmind.tracing.trace.set_tracer_provider",
        lambda provider: captured.update(resource=provider.resource),
    )
    tracing.init(overmind_api_key="test")
    return captured["resource"]


def test_init_stamps_sha_on_resource(tmp_path, no_sha_env, monkeypatch):
    sha = _fake_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    resource = _init_and_capture_resource(monkeypatch)
    assert resource.attributes[attrs.VCS_REF_HEAD_REVISION] == sha


def test_init_omits_sha_when_undetectable(tmp_path, no_sha_env, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resource = _init_and_capture_resource(monkeypatch)
    assert attrs.VCS_REF_HEAD_REVISION not in resource.attributes
