"""Failure clustering and root-cause taxonomy.

Maintains a registry of failure clusters across optimization iterations (and
optionally across runs via ``RunState``).  Failed test cases are grouped by
structural signature (which eval-spec fields fail + tool-call patterns) and
optionally refined by an LLM classifier for root-cause labelling.

Cluster lifecycle:
    open → resolved (when exemplar cases all pass) → regressed (if they fail again)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class FailureCluster:
    """A group of test cases that fail for the same structural reason."""

    cluster_id: str
    root_cause: str
    mechanism: str  # wrong_tool_args | missing_tool_call | prompt_ambiguity | logic_error | format_error | unknown
    affected_fields: list[str]
    exemplar_case_indices: list[int]
    total_occurrences: int = 1
    first_seen_iteration: int = 0
    last_seen_iteration: int = 0
    resolution_status: str = "open"  # open | resolved | regressed
    resolved_at_iteration: int | None = None
    resolved_by_change: str | None = None

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "root_cause": self.root_cause,
            "mechanism": self.mechanism,
            "affected_fields": self.affected_fields,
            "exemplar_case_indices": self.exemplar_case_indices,
            "total_occurrences": self.total_occurrences,
            "first_seen_iteration": self.first_seen_iteration,
            "last_seen_iteration": self.last_seen_iteration,
            "resolution_status": self.resolution_status,
            "resolved_at_iteration": self.resolved_at_iteration,
            "resolved_by_change": self.resolved_by_change,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FailureCluster:
        return cls(
            cluster_id=d["cluster_id"],
            root_cause=d["root_cause"],
            mechanism=d["mechanism"],
            affected_fields=d.get("affected_fields", []),
            exemplar_case_indices=d.get("exemplar_case_indices", []),
            total_occurrences=d.get("total_occurrences", 1),
            first_seen_iteration=d.get("first_seen_iteration", 0),
            last_seen_iteration=d.get("last_seen_iteration", 0),
            resolution_status=d.get("resolution_status", "open"),
            resolved_at_iteration=d.get("resolved_at_iteration"),
            resolved_by_change=d.get("resolved_by_change"),
        )


# ---------------------------------------------------------------------------
# Structural signature
# ---------------------------------------------------------------------------

_MECHANISM_MAP = {
    "tool_error": "wrong_tool_args",
    "missing_tool": "missing_tool_call",
    "format": "format_error",
    "logic": "logic_error",
    "prompt": "prompt_ambiguity",
}


def _case_signature(case: dict, eval_spec: dict | None) -> tuple[str, ...]:
    """Build a hashable structural signature from a failed case.

    The signature encodes: which fields failed, whether tool errors occurred,
    and which expected tools were missing.
    """
    parts: list[str] = []
    score = case.get("score", {})
    fields = (eval_spec or {}).get("output_fields", {})

    for fname, cfg in fields.items():
        mx = cfg.get("weight", 0)
        fs = score.get(fname, 0)
        if mx > 0 and fs < mx * 0.5:
            parts.append(f"fail:{fname}")

    struct_max = (eval_spec or {}).get("structure_weight", 20)
    struct_score = score.get("structure", 0)
    if struct_max > 0 and struct_score < struct_max * 0.5:
        parts.append("fail:structure")

    tool_trace = case.get("tool_trace", [])
    has_tool_error = any(t.get("error") for t in tool_trace)
    if has_tool_error:
        parts.append("tool_error")

    called_tools = {t.get("name") for t in tool_trace}
    expected_tools = (eval_spec or {}).get("tool_config", {}).get("expected_tools", [])
    for et in expected_tools:
        tool_name = et if isinstance(et, str) else et.get("name", "")
        if tool_name and tool_name not in called_tools:
            parts.append(f"missing_tool:{tool_name}")

    return tuple(sorted(parts)) if parts else ("unknown_failure",)


def _signature_to_mechanism(sig: tuple[str, ...]) -> str:
    """Infer the primary failure mechanism from a structural signature."""
    has_tool_error = any(s == "tool_error" for s in sig)
    has_missing_tool = any(s.startswith("missing_tool:") for s in sig)
    has_structure_fail = any(s == "fail:structure" for s in sig)
    has_field_fail = any(s.startswith("fail:") and s != "fail:structure" for s in sig)

    if has_tool_error:
        return "wrong_tool_args"
    if has_missing_tool:
        return "missing_tool_call"
    if has_structure_fail and not has_field_fail:
        return "format_error"
    if has_field_fail:
        return "logic_error"
    return "unknown"


def _signature_to_fields(sig: tuple[str, ...]) -> list[str]:
    """Extract affected field names from a structural signature."""
    fields: list[str] = []
    for part in sig:
        if part.startswith("fail:"):
            fields.append(part.removeprefix("fail:"))
        elif part.startswith("missing_tool:"):
            fields.append(part.removeprefix("missing_tool:"))
    return fields


def _signature_hash(sig: tuple[str, ...]) -> str:
    raw = "|".join(sig)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class FailureRegistry:
    """Accumulates and manages failure clusters across iterations."""

    def __init__(self) -> None:
        self.clusters: dict[str, FailureCluster] = {}
        self._sig_to_cluster: dict[tuple[str, ...], str] = {}

    # -- Serialization --

    def to_dict(self) -> dict:
        return {
            "clusters": {k: v.to_dict() for k, v in self.clusters.items()},
            "sig_map": {"|".join(k): v for k, v in self._sig_to_cluster.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> FailureRegistry:
        reg = cls()
        for cid, cdata in d.get("clusters", {}).items():
            reg.clusters[cid] = FailureCluster.from_dict(cdata)
        for sig_str, cid in d.get("sig_map", {}).items():
            sig = tuple(sig_str.split("|"))
            reg._sig_to_cluster[sig] = cid
        return reg

    # -- Core operations --

    def ingest_iteration(
        self,
        iteration: int,
        case_results: list[dict],
        eval_spec: dict | None = None,
        diagnosis: dict | None = None,
    ) -> list[FailureCluster]:
        """Classify failed cases from one iteration into clusters.

        Returns a list of clusters that were created or updated.
        """
        touched: list[FailureCluster] = []
        root_cause_text = ""
        if diagnosis:
            root_cause_text = diagnosis.get("root_cause", "")

        for idx, case in enumerate(case_results):
            total = case.get("score", {}).get("total", 0)
            if total >= 80:
                continue

            sig = _case_signature(case, eval_spec)
            existing_cid = self._sig_to_cluster.get(sig)

            if existing_cid and existing_cid in self.clusters:
                cluster = self.clusters[existing_cid]
                cluster.total_occurrences += 1
                cluster.last_seen_iteration = iteration
                if idx not in cluster.exemplar_case_indices:
                    if len(cluster.exemplar_case_indices) < 5:
                        cluster.exemplar_case_indices.append(idx)
                if cluster.resolution_status == "resolved":
                    cluster.resolution_status = "regressed"
                    cluster.resolved_at_iteration = None
                    cluster.resolved_by_change = None
                touched.append(cluster)
            else:
                cid = _signature_hash(sig)
                mechanism = _signature_to_mechanism(sig)
                affected = _signature_to_fields(sig)

                label = (
                    root_cause_text[:200]
                    if root_cause_text
                    else f"Failure in {', '.join(affected) or 'unknown fields'}"
                )

                cluster = FailureCluster(
                    cluster_id=cid,
                    root_cause=label,
                    mechanism=mechanism,
                    affected_fields=affected,
                    exemplar_case_indices=[idx],
                    total_occurrences=1,
                    first_seen_iteration=iteration,
                    last_seen_iteration=iteration,
                )
                self.clusters[cid] = cluster
                self._sig_to_cluster[sig] = cid
                touched.append(cluster)

        return touched

    def update_resolution_status(
        self,
        iteration: int,
        case_results: list[dict],
        eval_spec: dict | None = None,
        change_summary: str | None = None,
    ) -> list[FailureCluster]:
        """Mark clusters as resolved if all their exemplar cases now pass.

        Returns list of newly resolved clusters.
        """
        newly_resolved: list[FailureCluster] = []

        for cluster in self.clusters.values():
            if cluster.resolution_status != "open":
                continue

            all_pass = True
            for case_idx in cluster.exemplar_case_indices:
                if case_idx >= len(case_results):
                    all_pass = False
                    break
                score = case_results[case_idx].get("score", {}).get("total", 0)
                if score < 70:
                    all_pass = False
                    break

            if all_pass and cluster.exemplar_case_indices:
                cluster.resolution_status = "resolved"
                cluster.resolved_at_iteration = iteration
                cluster.resolved_by_change = change_summary
                newly_resolved.append(cluster)

        return newly_resolved

    def get_priority_clusters(self, top_k: int = 5) -> list[FailureCluster]:
        """Rank open/regressed clusters by impact (occurrences * recency)."""
        candidates = [
            c
            for c in self.clusters.values()
            if c.resolution_status in ("open", "regressed")
        ]
        candidates.sort(
            key=lambda c: (
                c.total_occurrences * (1 + c.last_seen_iteration),
                -c.first_seen_iteration,
            ),
            reverse=True,
        )
        return candidates[:top_k]

    def get_resolved_clusters(self) -> list[FailureCluster]:
        return [c for c in self.clusters.values() if c.resolution_status == "resolved"]

    def get_open_count(self) -> int:
        return sum(
            1
            for c in self.clusters.values()
            if c.resolution_status in ("open", "regressed")
        )

    def get_resolved_count(self) -> int:
        return sum(
            1 for c in self.clusters.values() if c.resolution_status == "resolved"
        )

    def compute_component_weights(self) -> dict[str, float]:
        """Derive focus-area weights from open failure cluster mechanisms.

        Maps cluster mechanisms to optimization focus areas and weights
        them by occurrence count.
        """
        MECHANISM_TO_FOCUS: dict[str, str] = {
            "wrong_tool_args": "tool_description",
            "missing_tool_call": "tool_description",
            "format_error": "format_input",
            "logic_error": "agent_logic",
            "prompt_ambiguity": "system_prompt",
            "unknown": "agent_logic",
        }

        weights: dict[str, float] = {
            "tool_description": 0.0,
            "agent_logic": 0.0,
            "format_input": 0.0,
            "system_prompt": 0.0,
        }

        total = 0
        for cluster in self.clusters.values():
            if cluster.resolution_status not in ("open", "regressed"):
                continue
            focus = MECHANISM_TO_FOCUS.get(cluster.mechanism, "agent_logic")
            weights[focus] += cluster.total_occurrences
            total += cluster.total_occurrences

        if total > 0:
            for k in weights:
                weights[k] /= total

        return weights


def format_clusters_for_diagnosis(
    clusters: list[FailureCluster],
    max_clusters: int = 8,
) -> str:
    """Format priority clusters into a prompt section for the analyzer."""
    if not clusters:
        return "(no failure clusters identified yet)"

    lines: list[str] = []
    for i, c in enumerate(clusters[:max_clusters], 1):
        status_icon = {
            "open": "\u25cf",  # ●
            "regressed": "\u26a0",  # ⚠
            "resolved": "\u2713",  # ✓
        }.get(c.resolution_status, "?")

        lines.append(
            f"{i}. [{status_icon} {c.resolution_status.upper()}] "
            f"**{c.root_cause[:120]}**"
        )
        lines.append(
            f"   Mechanism: {c.mechanism} | "
            f"Fields: {', '.join(c.affected_fields) or 'n/a'} | "
            f"Occurrences: {c.total_occurrences} | "
            f"First seen: iter {c.first_seen_iteration} | "
            f"Last seen: iter {c.last_seen_iteration}"
        )
        if c.resolved_by_change:
            lines.append(f"   Previously fixed by: {c.resolved_by_change[:100]}")
        lines.append("")

    open_count = sum(
        1 for c in clusters if c.resolution_status in ("open", "regressed")
    )
    lines.append(
        f"**{open_count} unresolved cluster(s)** out of "
        f"{len(clusters)} shown. Prioritize the highest-occurrence "
        f"unresolved clusters."
    )

    return "\n".join(lines)
