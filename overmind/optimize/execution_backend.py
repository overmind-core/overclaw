"""Execution backends used by the optimizer to run (or shadow-run) an agent.

A backend is a thin wrapper around :class:`overmind.optimize.runner.AgentRunner`
that decides *how* each case is executed and annotates the result with
provenance / confidence.  Today we ship two backends:

* :class:`SubprocessBackend` — the historical path; runs the agent in a
  plain Python subprocess.  Hardened via argv guards and a preflight import
  check (see :func:`shadow_runtime.bootstrap_source`).

* :class:`ShadowBackend` — runs the agent in a subprocess with every
  external call (LLM, HTTP, browser) intercepted.  Real LLM calls still go
  through to the real model; everything else is replayed from a cassette or
  simulated.  Produces lower-confidence traces but lets the optimizer keep
  iterating on agents that cannot be run end-to-end on Overmind's
  infrastructure.

The :class:`BackendPlan` abstraction exposes a deterministic *try-order* so
callers can fall back gracefully: ``[Subprocess → Shadow → Static]``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from overmind.optimize.cassette import Cassette, open_cassette
from overmind.optimize.failure_classifier import (
    FailureDiagnosis,
    FailureMode,
    classify_failure,
    is_recoverable_via_shadow,
)
from overmind.optimize.provenance import (
    Confidence,
    SourceTag,
    TraceSource,
    aggregate_confidence,
)
from overmind.optimize.runner import AgentRunner, RunOutput
from overmind.optimize.shadow_runtime import (
    ShadowConfig,
    read_provenance_file,
)

__all__ = [
    "BackendOutput",
    "BackendPlan",
    "ExecutionBackend",
    "ShadowBackend",
    "SubprocessBackend",
    "build_default_plan",
    "should_try_next",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class BackendOutput:
    """Result of one case executed through a backend.

    Wraps :class:`RunOutput` with Overmind-specific metadata so the optimizer
    can reason about confidence and provenance without digging into raw
    process output.
    """

    run_output: RunOutput
    backend: str
    """Which backend produced this output (e.g. ``"subprocess"`` / ``"shadow"``)."""

    diagnosis: FailureDiagnosis | None = None
    """Set when the run failed, so the caller knows *why* and whether to
    fall back to another backend."""

    provenance: list[SourceTag] = field(default_factory=list)
    """Per-call source tags collected during the run (from the shadow
    provenance sidecar, when applicable)."""

    confidence: Confidence = field(default_factory=Confidence)

    @property
    def success(self) -> bool:
        return self.run_output.success

    @property
    def data(self) -> Any:
        return self.run_output.data

    @property
    def error(self) -> str:
        return self.run_output.error


class ExecutionBackend(Protocol):
    """Common interface every backend implements."""

    name: str

    def prepare(self) -> None:
        """Set up anything the backend needs once per optimization run."""

    def run(
        self,
        input_data: Any,
        *,
        timeout: int | None = None,
        trace_file: str | Path | None = None,
    ) -> BackendOutput:
        """Execute the agent on a single input."""

    def cleanup(self) -> None:
        """Release any resources held by the backend."""


# ---------------------------------------------------------------------------
# Subprocess backend
# ---------------------------------------------------------------------------


class SubprocessBackend:
    """Standard subprocess execution.

    Hardened in two small ways compared to calling :class:`AgentRunner`
    directly:

    * ``sys.argv`` is neutralised inside the wrapper so agents with
      module-level :func:`argparse.ArgumentParser.parse_args` don't eat the
      wrapper's argv.
    * On failure we classify stderr into a :class:`FailureDiagnosis` and
      expose it to the caller so backend fallback can be informed.
    """

    name = "subprocess"

    def __init__(
        self,
        runner: AgentRunner,
        *,
        cassette: Cassette | None = None,
    ) -> None:
        self._runner = runner
        self._cassette = cassette if cassette is not None else open_cassette(None)
        self._prepared = False

    def prepare(self) -> None:
        if self._prepared:
            return
        self._runner.ensure_environment()
        self._prepared = True

    def run(
        self,
        input_data: Any,
        *,
        timeout: int | None = None,
        trace_file: str | Path | None = None,
    ) -> BackendOutput:
        # Subprocess path runs the agent for real, but if a cassette is
        # available we ship the LLM-intercept bootstrap in "record-only"
        # mode so successful runs populate the cassette.  This means the
        # first successful subprocess run warms the cassette so subsequent
        # shadow-mode runs can replay without hitting the live API.
        record_cfg: ShadowConfig | None = None
        if self._cassette.path is not None:
            record_cfg = ShadowConfig(
                enabled=False,
                cassette_path=str(self._cassette.path),
                provenance_path=None,
                simulate_browser=False,
                simulate_network=False,
            )
        ro = self._runner.run(
            input_data,
            timeout=timeout,
            trace_file=trace_file,
            shadow_config=record_cfg,
        )
        if ro.success:
            return BackendOutput(
                run_output=ro,
                backend=self.name,
                confidence=Confidence(
                    score=1.0,
                    summary={TraceSource.REAL_SUBPROCESS.value: 1},
                    reason="real subprocess execution",
                ),
            )

        diag = classify_failure(
            stderr=ro.stderr,
            returncode=ro.returncode,
            timed_out="timed out" in (ro.error or "").lower(),
            error=ro.error,
        )
        logger.info(f"SubprocessBackend: run failed mode={diag.mode.value} summary={diag.summary}")
        return BackendOutput(
            run_output=ro,
            backend=self.name,
            diagnosis=diag,
            confidence=Confidence(
                score=0.0,
                summary={},
                reason=f"subprocess failure ({diag.mode.value})",
            ),
        )

    def cleanup(self) -> None:
        try:
            self._runner.cleanup()
        except Exception as exc:
            logger.debug(f"SubprocessBackend cleanup: {exc}")


# ---------------------------------------------------------------------------
# Shadow backend
# ---------------------------------------------------------------------------


class ShadowBackend:
    """Run the agent with external calls intercepted.

    The shadow backend is identical to :class:`SubprocessBackend` except:

    * A :class:`ShadowConfig` is attached so the runner's wrapper prepends
      the shadow bootstrap (see :mod:`shadow_runtime`).
    * Provenance is read from a per-case sidecar file and aggregated into a
      :class:`Confidence` on the returned :class:`BackendOutput`.
    * Cassette misses fall back to a simulator and are tagged
      :attr:`TraceSource.SIMULATED`, so the caller can decide whether the
      signal is trustworthy enough to auto-apply changes.
    """

    name = "shadow"

    def __init__(
        self,
        runner: AgentRunner,
        *,
        cassette: Cassette,
        provenance_dir: Path,
        simulate_browser: bool = True,
        simulate_network: bool = True,
    ) -> None:
        self._runner = runner
        self._cassette = cassette
        self._provenance_dir = Path(provenance_dir)
        self._simulate_browser = simulate_browser
        self._simulate_network = simulate_network
        self._prepared = False
        self._call_counter = 0

    def prepare(self) -> None:
        if self._prepared:
            return
        self._runner.ensure_environment()
        self._provenance_dir.mkdir(parents=True, exist_ok=True)
        self._prepared = True

    def _provenance_path(self) -> Path:
        self._call_counter += 1
        return self._provenance_dir / f"prov-{self._call_counter:05d}.jsonl"

    def run(
        self,
        input_data: Any,
        *,
        timeout: int | None = None,
        trace_file: str | Path | None = None,
    ) -> BackendOutput:
        prov_path = self._provenance_path()
        cass_path = self._cassette.path

        shadow_cfg = ShadowConfig(
            enabled=True,
            cassette_path=str(cass_path) if cass_path else None,
            provenance_path=str(prov_path),
            simulate_browser=self._simulate_browser,
            simulate_network=self._simulate_network,
        )

        ro = self._runner.run(
            input_data,
            timeout=timeout,
            trace_file=trace_file,
            shadow_config=shadow_cfg,
        )

        tags = _load_source_tags(prov_path)

        if not ro.success:
            diag = classify_failure(
                stderr=ro.stderr,
                returncode=ro.returncode,
                timed_out="timed out" in (ro.error or "").lower(),
                error=ro.error,
            )
            logger.info(f"ShadowBackend: run failed mode={diag.mode.value} summary={diag.summary}")
            return BackendOutput(
                run_output=ro,
                backend=self.name,
                diagnosis=diag,
                provenance=tags,
                confidence=Confidence(
                    score=0.0,
                    summary={},
                    reason=f"shadow failure ({diag.mode.value})",
                ),
            )

        confidence = aggregate_confidence(tags)
        confidence.reason = (
            f"shadow execution ({confidence.reason})" if tags else "shadow execution (no external calls intercepted)"
        )
        return BackendOutput(
            run_output=ro,
            backend=self.name,
            provenance=tags,
            confidence=confidence,
        )

    def cleanup(self) -> None:
        try:
            self._runner.cleanup()
        except Exception as exc:
            logger.debug(f"ShadowBackend cleanup: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_source_tags(path: Path) -> list[SourceTag]:
    """Read a shadow-provenance sidecar into :class:`SourceTag` objects."""
    tags: list[SourceTag] = []
    for raw in read_provenance_file(path):
        source = raw.get("source")
        if not source:
            continue
        try:
            tags.append(SourceTag(source=TraceSource(source), reason=raw.get("reason", "")))
        except ValueError:
            continue
    return tags


# ---------------------------------------------------------------------------
# Planning — choose a backend, optionally with fallback
# ---------------------------------------------------------------------------


@dataclass
class BackendPlan:
    """Ordered list of backends the optimizer should try for each case.

    Callers invoke them in order, stopping on the first success (or on a
    :class:`FailureDiagnosis` that says further retries won't help).  This
    keeps fallback logic in one place and makes it trivial to unit test.
    """

    backends: list[ExecutionBackend]

    def __len__(self) -> int:
        return len(self.backends)

    def __iter__(self):  # type: ignore[override]
        return iter(self.backends)


def build_default_plan(
    *,
    runner: AgentRunner,
    cassette_path: Path | None,
    provenance_dir: Path,
    enable_shadow_fallback: bool = True,
    simulate_browser: bool = True,
    simulate_network: bool = True,
) -> BackendPlan:
    """Build the default [Subprocess → Shadow] fallback plan.

    When *enable_shadow_fallback* is ``False`` we return a single-backend
    plan — useful for tests that want to isolate the subprocess path.
    """
    cassette = open_cassette(cassette_path)
    subp = SubprocessBackend(runner, cassette=cassette)
    if not enable_shadow_fallback:
        return BackendPlan(backends=[subp])

    shadow = ShadowBackend(
        runner,
        cassette=cassette,
        provenance_dir=provenance_dir,
        simulate_browser=simulate_browser,
        simulate_network=simulate_network,
    )
    return BackendPlan(backends=[subp, shadow])


def should_try_next(
    diagnosis: FailureDiagnosis | None,
) -> bool:
    """Given a failed :class:`FailureDiagnosis`, decide whether to fall back.

    Non-retryable failures (missing API key, syntax error, ImportError) are
    terminal — shadow mode can't help, so we surface the error instead of
    wasting another iteration.
    """
    if diagnosis is None:
        return False
    if diagnosis.mode == FailureMode.NONE:
        return False
    return is_recoverable_via_shadow(diagnosis) and diagnosis.retryable
