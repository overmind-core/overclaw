"""Resolve agents and datasets from the Overmind platform (not local registry)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from overmind import attrs as oc_attrs
from overmind.client import get_client
from overmind.optimize.runner import AgentRunner, RunnerConfig, _current_traceparent

logger = logging.getLogger("overmind.platform_agent")


def fetch_datapoints(dataset_id: str) -> list[dict[str, Any]]:
    client = get_client()
    if client is None:
        raise RuntimeError("OVERMIND_API_KEY is not configured")

    cases: list[dict[str, Any]] = []
    page = 1
    while True:
        resp = client.datasets_datapoints_list(
            dataset_id=dataset_id,
            page=page,
            page_size=200,
            ordering="order",
        )
        for dp in resp.results or []:
            cases.append({
                "input": dp.input,
                "expected_output": dp.expected_output,
                "case_id": str(dp.id),
            })
        if not resp.next:
            break
        page += 1
    return cases


def _trace_id_from_traceparent(traceparent: str | None) -> str:
    if not traceparent:
        return ""
    parts = traceparent.split("-")
    if len(parts) >= 2:
        return parts[1]
    return ""


def run_agent_from_platform(
    *,
    payload: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    agent_path = (payload.get("agent_path") or "").strip()
    entrypoint_fn = (payload.get("entrypoint_fn") or "").strip()
    dataset_id = (payload.get("dataset_id") or "").strip()
    subset = payload.get("subset", "baseline")
    smoke_cases = int(payload.get("smoke_cases", 2))

    if not agent_path:
        raise ValueError("Command payload missing agent_path")
    if not entrypoint_fn:
        raise ValueError("Command payload missing entrypoint_fn")
    if not dataset_id:
        raise ValueError("Command payload missing dataset_id")

    file_path = (root / agent_path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Agent file not found at {file_path}")

    cases = fetch_datapoints(dataset_id)
    if subset == "smoke":
        cases = cases[:smoke_cases]

    runner = AgentRunner(
        agent_dir=root,
        entry_file=agent_path,
        entrypoint_fn=entrypoint_fn,
        config=RunnerConfig(timeout=300),
    )
    runner.ensure_environment()

    base_correlation = dict(payload.get("trace_correlation") or {})
    base_correlation.setdefault(oc_attrs.RUN_KIND, subset)

    case_results: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        inp = case.get("input") or case
        expected = case.get("expected_output")
        case_id = case.get("case_id", str(i))
        tags = {
            **base_correlation,
            oc_attrs.WORKFLOW_CASE_ID: case_id,
        }
        if payload.get("iteration") is not None:
            tags[oc_attrs.WORKFLOW_ITERATION] = str(payload.get("iteration"))
        if payload.get("candidate_index") is not None:
            tags[oc_attrs.OPTIMIZE_CANDIDATE_INDEX] = str(payload.get("candidate_index"))

        run_env = {"OVERMIND_RUN_TAGS": json.dumps(tags, default=str)}
        run_out = runner.run(inp, run_env=run_env)
        trace_id = _trace_id_from_traceparent(_current_traceparent())

        row = {
            "case_id": case_id,
            "input": inp,
            "expected_output": expected,
            "output": run_out.data if run_out.success else {},
            "error": run_out.error if not run_out.success else "",
            "success": run_out.success,
            "trace_id": trace_id,
        }
        case_results.append(row)
        results.append(row)

    runner.cleanup()
    return {
        "subset": subset,
        "agent_id": payload.get("agent_id", ""),
        "case_results": case_results,
        "results": results,
        "count": len(case_results),
    }
