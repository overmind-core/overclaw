"""overclaw sync-optimize — upload local optimize artifacts to Overmind."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rich.console import Console

from overclaw.client import ApiReporter, flush_pending_api_updates, get_client, get_project_id
from overclaw.commands.setup_cmd import _ensure_remote_agent_id
from overclaw.core.paths import agent_experiments_dir, load_overclaw_dotenv
from overclaw.core.registry import load_registry
from overclaw.storage import configure_storage, get_storage
from overclaw.storage.api import ApiBackend


def _read_results_rows(results_path: Path) -> list[dict]:
    if not results_path.exists():
        return []
    with results_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in reader if row]


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _sync_optimize_artifacts_for_agent(
    agent_name: str,
    agent_path: str,
    console: Console,
) -> bool:
    """Sync one agent's optimize artifacts. Returns True if any sync attempted."""
    if not get_client() or not get_project_id():
        return False

    agent_id = _ensure_remote_agent_id(agent_name, agent_path, console)
    if not agent_id:
        return False

    exp_dir = agent_experiments_dir(agent_name)
    if not exp_dir.exists():
        console.print(f"[yellow]Skipping {agent_name}: no experiments directory.[/yellow]")
        return False

    results_path = exp_dir / "results.tsv"
    report_path = exp_dir / "report.md"
    best_agent_path = exp_dir / "best_agent.py"
    traces_root = exp_dir / "traces"
    rows = _read_results_rows(results_path)
    baseline_row = next((r for r in rows if r.get("iteration") == "baseline"), None)
    baseline_score = _to_float((baseline_row or {}).get("avg_score", "0"), 0.0)
    iter_rows = [
        r
        for r in rows
        if (r.get("iteration") or "").startswith("iter_")
        and (r.get("status") or "") in {"keep", "discard"}
    ]
    analyzer_model = "unknown"
    reporter = ApiReporter.create(
        agent_id=agent_id,
        analyzer_model=analyzer_model,
        num_iterations=max(len(iter_rows), 1),
        candidates_per_iteration=1,
    )
    if not reporter:
        console.print(
            f"[yellow]Skipping {agent_name}: could not create backend optimize job.[/yellow]"
        )
        return False

    configure_storage(
        agent_path=agent_path,
        agent_id=agent_id,
        job_id=reporter.job_id,
        backend="api",
    )
    storage = get_storage()
    if not isinstance(storage, ApiBackend):
        return False
    storage.set_job_id(reporter.job_id)

    reporter.on_baseline(baseline_score)

    synced_traces = 0
    if traces_root.exists():
        for trace_file in sorted(traces_root.glob("*/*.json")):
            run_name = trace_file.parent.name
            try:
                idx = int(trace_file.stem)
            except Exception:
                idx = 0
            try:
                trace_data = json.loads(trace_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            storage.save_trace(trace_data, run_name=run_name, idx=idx)
            synced_traces += 1

    for order, row in enumerate(iter_rows, start=1):
        reporter.on_iteration(
            order=order,
            avg_score=_to_float(row.get("avg_score", "0"), 0.0),
            decision=row.get("status", "discard"),
            agent_code=None,
            description=row.get("description", ""),
            dimension_scores=None,
        )

    report_md = report_path.read_text(encoding="utf-8") if report_path.exists() else None
    best_code = (
        best_agent_path.read_text(encoding="utf-8") if best_agent_path.exists() else None
    )
    best_score = max([_to_float(r.get("avg_score", "0"), 0.0) for r in iter_rows] or [baseline_score])
    reporter.on_complete(
        best_score=best_score,
        baseline_score=baseline_score,
        report_markdown=report_md,
        best_agent_code=best_code,
        backtest_results=None,
    )
    flush_pending_api_updates(timeout=30.0)
    console.print(
        f"  [dim]Synced optimize artifacts for {agent_name} "
        f"(job={reporter.job_id}, traces={synced_traces}, iterations={len(iter_rows)}).[/dim]"
    )
    return True


def main(agent_name: str | None = None) -> None:
    """Sync local optimize artifacts to Overmind for one or all agents."""
    load_overclaw_dotenv()
    console = Console()

    if not get_client() or not get_project_id():
        console.print(
            "[red]Overmind API is not configured.[/red] "
            "Set OVERMIND_API_URL, OVERMIND_API_TOKEN, and OVERMIND_PROJECT_ID in .overclaw/.env."
        )
        raise SystemExit(1)

    registry = load_registry()
    if not registry:
        console.print("[yellow]No registered agents found.[/yellow]")
        return

    names = [agent_name] if agent_name else sorted(registry.keys())
    synced_agents = 0
    skipped_agents = 0

    for name in names:
        row = registry.get(name)
        if not row:
            console.print(f"[yellow]Skipping {name}: not registered.[/yellow]")
            skipped_agents += 1
            continue

        agent_path = row.get("file_path", "")
        if not agent_path or not Path(agent_path).exists():
            console.print(
                f"[yellow]Skipping {name}: registered file missing ({agent_path or 'unknown'}).[/yellow]"
            )
            skipped_agents += 1
            continue

        if _sync_optimize_artifacts_for_agent(name, agent_path, console):
            synced_agents += 1
        else:
            skipped_agents += 1

    console.print(
        f"\n[bold green]Optimize sync complete.[/bold green] "
        f"Processed: {synced_agents} agent(s), skipped: {skipped_agents}."
    )

