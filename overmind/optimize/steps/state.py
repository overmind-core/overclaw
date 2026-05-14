"""Cross-step persistent state for the skill-driven optimize loop.

A single JSON document persisted under
``<state>/agents/<name>/experiments/skill_state.json`` that captures
*everything* the skill needs across CLI invocations:

- The full ``Config`` the user picked at ``init`` time.
- Loop control (``iteration``, ``stall_count``, ``early_stopping_triggered``).
- Best-so-far record (score, code path, files dir).
- Per-iteration history (mirrors ``Optimizer.results`` so ``report`` can
  render the same ``report.md``).
- Working directory paths (the ``agent_working`` file/dir the next
  iteration starts from).

Cross-*run* persistence (``run_state.json``) is unchanged and continues
to be owned by :class:`overmind.optimize.run_state.RunState`. This file
is for cross-*step* (within-run) state only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from overmind.optimize.config import Config


@dataclass
class SkillRunState:
    """Serializable state passed between ``overmind optimize-step`` invocations."""

    # ---- Identity & config ----
    schema_version: int = 1
    agent_name: str = ""
    state_path: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    # ---- Phase tracking ----
    phase: str = "init"
    iteration: int = 0
    early_stopping_triggered: bool = False

    # ---- Dataset (resolved & split once at init) ----
    dataset_size: int = 0
    train_size: int = 0
    holdout_size: int = 0

    # ---- Baseline ----
    baseline_score: float | None = None
    baseline_holdout_score: float | None = None

    # ---- Best-so-far ----
    best_score: float = 0.0
    best_iteration: int = 0
    best_code_path: str = ""  # Path to the entry file of best candidate
    best_files_dir: str = ""  # Multi-file bundle root (empty if single-file)
    best_case_scores: list[float] = field(default_factory=list)

    # ---- Loop control ----
    stall_count: int = 0

    # ---- Working copy paths ----
    working_path: str = ""
    working_dir: str = ""  # Multi-file working tree root
    output_dir: str = ""  # experiments/

    # ---- History ----
    results: list[dict] = field(default_factory=list)
    accepted_snapshots: list[dict] = field(default_factory=list)
    failed_attempts: list[dict] = field(default_factory=list)
    successful_changes: list[dict] = field(default_factory=list)

    # ---- Read-only baseline cache ----
    # ``{rel_path: source_text}`` for every file declared as
    # ``scope.read_only_paths``. Populated lazily on the first ``accept``
    # step (when the bundle is first built) and reused on subsequent
    # iterations so we don't re-run the BFS just to recover content the
    # user has already promised won't change. Cleared when
    # ``config.read_only_scope`` shifts (the cache key includes the
    # serialized scope list under ``read_only_baseline_key``).
    read_only_baseline: dict[str, str] = field(default_factory=dict)
    read_only_baseline_key: str = ""

    # Stable identifier for the optimize ``Job`` row this run belongs to.
    # Generated once at ``init`` and stamped on every subsequent step's
    # OTel span as ``overmind.job.id`` as a secondary coalescing key
    # (the primary one is the shared trace_id below).
    job_id: str = ""

    # W3C traceparent for the optimize workflow root span emitted by
    # ``init``.  Every subsequent ``overmind optimize-step`` CLI
    # invocation picks this up from ``skill_state.json`` and exports
    # ``TRACEPARENT`` into the environment before ``overmind.init()`` —
    # the SDK's :func:`_attach_remote_parent_if_present` then makes
    # every span created in that process a child of the root, so all
    # steps land on a single distributed trace (one trace_id, one Job
    # in the UI) without relying on ``overmind.job.id`` post-hoc
    # coalescing.
    #
    # Format: ``00-<32 hex trace_id>-<16 hex root_span_id>-01``.
    traceparent: str = ""

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> SkillRunState:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Skill run state not found at {p}. Run `overmind optimize-step init` first.")
        data = json.loads(p.read_text())
        return cls.from_dict(data, str(p))

    @classmethod
    def from_dict(cls, data: dict, state_path: str = "") -> SkillRunState:
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        if state_path:
            kwargs["state_path"] = state_path
        return cls(**kwargs)

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path or self.state_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write so a crashed step never leaves a half-written state file.
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, default=str))
        tmp.replace(target)
        self.state_path = str(target)
        return target

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    def to_config(self) -> Config:
        """Rehydrate a :class:`Config` from the persisted dict."""
        cfg_kwargs = dict(self.config)
        # Drop any unknown keys gracefully so older state files keep working.
        known = {f for f in Config.__dataclass_fields__}
        cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if k in known}
        return Config(**cfg_kwargs)

    @classmethod
    def from_config(
        cls,
        *,
        agent_name: str,
        config: Config,
        state_path: str | Path,
    ) -> SkillRunState:
        return cls(
            agent_name=agent_name,
            state_path=str(state_path),
            config=asdict(config),
        )

    # ------------------------------------------------------------------
    # Mutation helpers (small wrappers so the steps read self-documenting)
    # ------------------------------------------------------------------
    def record_iteration(self, entry: dict) -> None:
        self.results.append(entry)

    def update_best(
        self,
        *,
        score: float,
        iteration: int,
        code_path: str,
        files_dir: str = "",
        case_scores: list[float] | None = None,
    ) -> None:
        self.best_score = float(score)
        self.best_iteration = int(iteration)
        self.best_code_path = code_path
        self.best_files_dir = files_dir
        if case_scores is not None:
            self.best_case_scores = [float(s) for s in case_scores]
        self.stall_count = 0

    def bump_stall(self) -> None:
        self.stall_count += 1
