"""Data loading, synthetic test-case generation, and seed-data augmentation.

Provides two generation paths:

Path A (no seed data) — full persona-driven pipeline:
  Phase 1: Red-team-style personas (diverse + adversarial/edge intents) from agent context
  Phase 2: Per round, each persona is handled **sequentially**; within a persona,
  the batch is split across **parallel** LLM shard calls
  Phase 3: Deduplication + schema validation + retries
  Phase 4: Coverage report

Path B (seed data exists) — analyze + optional augmentation:
  Phase 0: Schema-validate seed data
  Phase 1: LLM-based coverage analysis → gap report
  Interactive: ask user whether to augment
  Phase 2: Gap-targeted generation
  Phase 3: Merge + dedup + validate

Fast mode uses the legacy single-call generator with no personas or analysis.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import litellm

from overclaw.utils.llm import llm_completion
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from overclaw.utils.display import BRAND
from overclaw.utils.display import make_spinner_progress
from overclaw.prompts.data import (
    BATCH_GENERATION_PROMPT,
    PERSONAS_GENERATION_PROMPT,
    SYNTHETIC_DATA_LEGACY_PROMPT,
)

logger = logging.getLogger(__name__)


class DatasetGenerationError(Exception):
    """Raised when the pipeline cannot produce any valid cases."""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(path: str) -> list[dict]:
    """Load test cases from a JSON file.

    Accepts either a bare JSON array or an object with a ``test_cases`` key.
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "test_cases" in data:
        return data["test_cases"]

    raise ValueError(
        f"Unrecognized data format in {path}. "
        "Expected a JSON array or an object with a 'test_cases' key."
    )


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def _format_input_schema(eval_spec: dict) -> str:
    """Render input_schema from eval spec into a readable block."""
    schema = eval_spec.get("input_schema", {})
    if not schema:
        return ""
    lines = []
    for field, info in schema.items():
        ftype = info.get("type", "string")
        desc = info.get("description", "")
        lines.append(f"  - {field} ({ftype}): {desc}")
    return "\n".join(lines)


def _format_output_schema(eval_spec: dict) -> str:
    """Render output_fields from eval spec with types, enums, and ranges."""
    fields = eval_spec.get("output_fields", {})
    if not fields:
        return ""
    lines = []
    for field, cfg in fields.items():
        ftype = cfg.get("type", "string")
        desc = cfg.get("description", "")
        extra = ""
        if ftype == "enum":
            vals = cfg.get("values", [])
            extra = f" — allowed values: {vals}"
        elif ftype == "number":
            rng = cfg.get("range")
            if rng:
                extra = f" — range: {rng[0]}–{rng[1]}"
        eval_mode = cfg.get("eval_mode", "")
        if eval_mode:
            extra += f" (eval_mode: {eval_mode})"
        lines.append(f"  - {field} ({ftype}): {desc}{extra}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_case_against_spec(case: dict, eval_spec: dict) -> list[str]:
    """Return a list of validation error strings (empty = valid)."""
    errors: list[str] = []
    inp = case.get("input")
    out = case.get("expected_output")

    if not isinstance(inp, dict):
        errors.append("Missing or non-dict 'input'")
        return errors
    if not isinstance(out, dict):
        errors.append("Missing or non-dict 'expected_output'")
        return errors

    input_schema = eval_spec.get("input_schema", {})
    for field in input_schema:
        if field not in inp:
            errors.append(f"input missing required field '{field}'")

    output_fields = eval_spec.get("output_fields", {})
    for field, cfg in output_fields.items():
        if field not in out:
            errors.append(f"expected_output missing field '{field}'")
            continue
        val = out[field]
        ftype = cfg.get("type", "string")
        if ftype == "enum":
            allowed = cfg.get("values", [])
            if allowed and val not in allowed:
                errors.append(f"expected_output.{field} = {val!r} not in {allowed}")
        elif ftype == "number":
            if not isinstance(val, (int, float)):
                errors.append(
                    f"expected_output.{field} must be a number, got {type(val).__name__}"
                )
            else:
                rng = cfg.get("range")
                if rng and (val < rng[0] or val > rng[1]):
                    errors.append(
                        f"expected_output.{field} = {val} outside range {rng}"
                    )
        elif ftype == "text":
            if cfg.get("eval_mode") == "non_empty" and not val:
                errors.append(f"expected_output.{field} must be non-empty")

    return errors


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _safe_parse_json(text: str) -> Any:
    """Best-effort JSON extraction from LLM output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

    repaired = text.replace("'", '"')
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    return None


# Max concurrent shard calls per persona (each shard is one LLM request).
PERSONA_INNER_PARALLEL_MAX = 5

# Default LLM timeout (seconds) for persona / batch generation.
DATA_GEN_LLM_TIMEOUT = 300.0


def _llm_call(
    model: str,
    prompt: str,
    *,
    temperature: float = 0.8,
    max_tokens: int = 8000,
    timeout: float = DATA_GEN_LLM_TIMEOUT,
    max_retries: int = 3,
) -> str | None:
    """Single LLM call with retry + exponential backoff on rate limits."""
    for attempt in range(max_retries):
        try:
            response = llm_completion(
                model,
                [{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return response.choices[0].message.content
        except litellm.RateLimitError:
            wait = (2**attempt) + random.uniform(0, 1)
            logger.warning(
                "Rate limited (attempt %d/%d), waiting %.1fs",
                attempt + 1,
                max_retries,
                wait,
            )
            time.sleep(wait)
        except litellm.Timeout:
            logger.warning(
                "Timeout on LLM call (attempt %d/%d)", attempt + 1, max_retries
            )
        except Exception:
            logger.exception(
                "Unexpected error on LLM call (attempt %d/%d)", attempt + 1, max_retries
            )
            if attempt == max_retries - 1:
                raise
    return None


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _format_code_for_prompt(agent_code: str, *, max_chars: int | None = None) -> str:
    """Format agent code for embedding in a generation prompt.

    If the code already contains bundle markers (multi-file format), it is
    used as-is.  Otherwise it is optionally trimmed and wrapped in a python fence.
    """
    if not agent_code:
        return ""
    if "# ===== FILE:" in agent_code or "# === FROM:" in agent_code:
        return (
            "\n<AgentCode>\n"
            "The agent is split across multiple files. Here is the relevant code:\n"
            f"{agent_code}\n"
            "</AgentCode>\n"
        )
    trimmed = (
        agent_code[:max_chars]
        if max_chars and len(agent_code) > max_chars
        else agent_code
    )
    return f"\n<AgentCode>\n```python\n{trimmed}\n```\n</AgentCode>\n"


def _canonicalize(d: dict) -> str:
    """Deterministic JSON string for a dict, used as a dedup key."""
    return json.dumps(d, sort_keys=True, default=str).strip().lower()


def _get_key_fields(input_dict: dict) -> tuple[str, ...]:
    """Extract the most identifying values from an input dict for near-dedup."""
    vals: list[str] = []
    for v in input_dict.values():
        if isinstance(v, str) and len(v) > 2:
            vals.append(v.strip().lower())
    return tuple(sorted(vals))


def _is_near_duplicate(
    candidate_input: dict,
    existing_key_sets: set[tuple[str, ...]],
) -> bool:
    """True if the primary string fields all match an existing case."""
    keys = _get_key_fields(candidate_input)
    return keys in existing_key_sets if keys else False


# ---------------------------------------------------------------------------
# Legacy single-call generator (used by fast mode)
# ---------------------------------------------------------------------------


def generate_synthetic_data(
    agent_description: str,
    model: str,
    num_samples: int = 15,
    agent_code: str | None = None,
    policy_context: str | None = None,
) -> list[dict]:
    """Single-call synthetic generation (fast mode / backward compat).

    Kept for the fast-mode path where we skip personas and analysis.
    """
    code_section = ""
    if agent_code:
        if "# ===== FILE:" in agent_code or "# === FROM:" in agent_code:
            code_section = (
                "\nHere is the agent source code (split across multiple files) — "
                "use it to understand the exact input schema, output schema, "
                "tool definitions, and decision logic:\n"
                f"{agent_code}\n"
            )
        else:
            code_section = (
                "\nHere is the full agent source code — use it to understand the exact "
                "input schema, output schema, tool definitions, and decision logic:\n\n"
                f"```python\n{agent_code}\n```\n"
            )

    policy_section = ""
    if policy_context:
        policy_section = f"\n## Agent Policy\n\n{policy_context}\n"

    prompt = SYNTHETIC_DATA_LEGACY_PROMPT.format(
        num_samples=num_samples,
        agent_description=agent_description,
        code_section=code_section,
        policy_section=policy_section,
    )

    response = llm_completion(
        model,
        [{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=8000,
        timeout=DATA_GEN_LLM_TIMEOUT,
    )

    content = response.choices[0].message.content
    start = content.find("[")
    end = content.rfind("]") + 1
    if start >= 0 and end > start:
        cases = json.loads(content[start:end])
        if isinstance(cases, list) and len(cases) > 0:
            return cases

    raise ValueError(
        "Failed to parse synthetic data from the LLM response. "
        "Try running again or use a different model."
    )


# ===================================================================
# Full persona-driven generation pipeline
# ===================================================================


def _generate_personas(
    agent_description: str,
    agent_code: str | None,
    eval_spec: dict,
    policy_context: str | None,
    model: str,
    num_personas: int = 5,
) -> list[dict]:
    """Phase 1A: generate domain-aware personas for diverse test generation."""
    input_schema_text = _format_input_schema(eval_spec)
    output_schema_text = _format_output_schema(eval_spec)

    domain_section = ""
    if policy_context:
        domain_section = f"\n<DomainContext>\n{policy_context}\n</DomainContext>\n"

    code_section = ""
    if agent_code:
        code_section = _format_code_for_prompt(agent_code, max_chars=6000)

    prompt = PERSONAS_GENERATION_PROMPT.format(
        agent_description=agent_description,
        code_section=code_section,
        input_schema_text=input_schema_text,
        output_schema_text=output_schema_text,
        domain_section=domain_section,
        num_personas=num_personas,
    )

    raw = _llm_call(model, prompt, temperature=0.7, max_tokens=4000)
    if not raw:
        logger.warning(
            "Persona generation returned nothing; falling back to default personas"
        )
        return _default_personas(num_personas)

    parsed = _safe_parse_json(raw)
    if isinstance(parsed, dict) and "personas" in parsed:
        personas = parsed["personas"]
        if isinstance(personas, list) and len(personas) > 0:
            return personas

    logger.warning("Could not parse personas from LLM; falling back to defaults")
    return _default_personas(num_personas)


def _default_personas(n: int) -> list[dict]:
    """Fallback personas when LLM generation fails."""
    archetypes = [
        {
            "name": "Novice User",
            "role_and_background": "New employee unfamiliar with the system",
            "skill_level": "novice",
            "intent": "standard",
            "communication_style": "verbose",
            "domain_behavior": "Provides extra unnecessary detail, may miss required fields",
            "typical_scenarios": [
                "Basic straightforward request",
                "Request with missing info",
            ],
        },
        {
            "name": "Power User",
            "role_and_background": "Experienced professional who uses the system daily",
            "skill_level": "expert",
            "intent": "standard",
            "communication_style": "precise",
            "domain_behavior": "Always provides complete, well-structured inputs",
            "typical_scenarios": ["Complex multi-signal case", "Boundary-value case"],
        },
        {
            "name": "Edge Case Tester",
            "role_and_background": "QA engineer probing system limits",
            "skill_level": "expert",
            "intent": "edge_case_probing",
            "communication_style": "terse",
            "domain_behavior": "Submits minimal, ambiguous, or unusual inputs to find bugs",
            "typical_scenarios": ["Empty or minimal input", "Conflicting signals"],
        },
        {
            "name": "Adversarial User",
            "role_and_background": "Security tester or mischievous user",
            "skill_level": "intermediate",
            "intent": "adversarial",
            "communication_style": "ambiguous",
            "domain_behavior": "Submits misleading or contradictory data",
            "typical_scenarios": ["Contradictory fields", "Injection-style input"],
        },
        {
            "name": "Domain Expert",
            "role_and_background": "Senior domain specialist with deep knowledge",
            "skill_level": "expert",
            "intent": "exploratory",
            "communication_style": "precise",
            "domain_behavior": "Tests nuanced domain-specific scenarios",
            "typical_scenarios": [
                "Highly technical scenario",
                "Unusual but valid case",
            ],
        },
    ]
    return archetypes[:n]


def _generate_batch(
    persona: dict,
    agent_description: str,
    agent_code: str | None,
    eval_spec: dict,
    policy_context: str | None,
    model: str,
    batch_size: int,
    existing_cases: list[dict],
    coverage_gaps: list[dict] | None = None,
) -> list[dict]:
    """Phase 2: generate a batch of {input, expected_output} cases for one persona."""
    input_schema_text = _format_input_schema(eval_spec)
    output_schema_text = _format_output_schema(eval_spec)

    code_section = ""
    if agent_code:
        code_section = _format_code_for_prompt(agent_code)

    policy_section = ""
    if policy_context:
        policy_section = f"\n<PolicyContext>\n{policy_context}\n</PolicyContext>\n"

    existing_section = ""
    if existing_cases:
        sample = (
            existing_cases
            if len(existing_cases) <= 15
            else random.sample(existing_cases, 15)
        )
        inputs_only = [c.get("input", {}) for c in sample]
        existing_section = (
            "\n<ExistingCases>\n"
            "These cases already exist — do NOT duplicate them:\n"
            f"{json.dumps(inputs_only, indent=2, default=str)}\n"
            "</ExistingCases>\n"
        )

    gap_section = ""
    if coverage_gaps:
        gap_lines = []
        for gap in coverage_gaps[:10]:
            gap_lines.append(
                f"  - [{gap.get('severity', 'medium')}] {gap.get('area', '')}: {gap.get('description', '')}"
            )
        gap_section = (
            "\n**Priority**: Generate cases that specifically cover these gaps:\n"
            + "\n".join(gap_lines)
            + "\n"
        )

    persona_name = persona.get("name", "User")
    persona_block = (
        f"{persona_name}: {persona.get('role_and_background', '')}\n"
        f"Skill level: {persona.get('skill_level', 'intermediate')}\n"
        f"Intent: {persona.get('intent', 'standard')}\n"
        f"Style: {persona.get('communication_style', 'precise')}\n"
        f"Domain behavior: {persona.get('domain_behavior', '')}"
    )

    prompt = BATCH_GENERATION_PROMPT.format(
        agent_description=agent_description,
        code_section=code_section,
        input_schema_text=input_schema_text,
        output_schema_text=output_schema_text,
        policy_section=policy_section,
        persona_block=persona_block,
        gap_section=gap_section,
        existing_section=existing_section,
        batch_size=batch_size,
        persona_name=persona_name,
    )

    raw = _llm_call(model, prompt, temperature=0.8, max_tokens=8000)
    if not raw:
        logger.warning("Batch generation returned nothing for persona %s", persona_name)
        return []

    parsed = _safe_parse_json(raw)
    if parsed is None:
        logger.warning("Failed to parse batch JSON for persona %s", persona_name)
        return []

    if isinstance(parsed, dict) and "cases" in parsed:
        cases = parsed["cases"]
    elif isinstance(parsed, list):
        cases = parsed
    else:
        logger.warning(
            "Unexpected batch response structure for persona %s", persona_name
        )
        return []

    if not isinstance(cases, list):
        return []
    return cases


# ---------------------------------------------------------------------------
# Persona-round helpers (sequential personas, parallel shards per persona)
# ---------------------------------------------------------------------------


def _split_batch_sizes(total: int, num_shards: int) -> list[int]:
    """Split *total* cases across *num_shards* shards (positive ints summing to *total*)."""
    if total <= 0:
        return []
    n = max(1, min(num_shards, total))
    base = total // n
    rem = total % n
    return [base + (1 if i < rem else 0) for i in range(n)]


def _apply_dedup(
    raw_cases: list[dict],
    eval_spec: dict,
    seen_canonical: set[str],
    seen_key_fields: set[tuple[str, ...]],
    out: list[dict],
) -> tuple[int, int]:
    """Validate and deduplicate *raw_cases* into *out* (mutates both sets and out).

    Returns ``(added, dup_drops)`` where ``dup_drops`` is the count of cases
    that were structurally valid but dropped solely because they were duplicates
    (distinct from schema/type failures which are unretriable).
    """
    added = 0
    dup_drops = 0
    for case in raw_cases:
        if not isinstance(case, dict):
            continue
        case.pop("_meta", None)

        if eval_spec:
            errors = validate_case_against_spec(case, eval_spec)
            if errors:
                logger.debug("Discarding invalid case: %s", errors)
                continue

        inp = case.get("input", {})
        if not isinstance(inp, dict):
            continue

        canon = _canonicalize(inp)
        if canon in seen_canonical:
            dup_drops += 1
            continue
        if _is_near_duplicate(inp, seen_key_fields):
            dup_drops += 1
            continue

        seen_canonical.add(canon)
        kf = _get_key_fields(inp)
        if kf:
            seen_key_fields.add(kf)
        out.append(case)
        added += 1
    return added, dup_drops


def _retry_dropped_slots(
    retry_slots: list[int],
    personas: list[dict],
    agent_description: str,
    agent_code: str | None,
    eval_spec: dict,
    policy_context: str | None,
    model: str,
    existing_snapshot: list[dict],
    coverage_gaps: list[dict] | None,
    seen_canonical: set[str],
    seen_key_fields: set[tuple[str, ...]],
    out: list[dict],
    max_attempts: int = 3,
) -> int:
    """For each dedup-dropped slot, try to regenerate a single unique case.

    Each slot corresponds to one persona (by index).  Up to *max_attempts*
    single-case generations are tried per slot; the slot is abandoned if all
    attempts still produce a duplicate.  Runs serially to keep dedup
    state consistent.

    Returns the total number of cases successfully added across all slots.
    """
    added = 0
    for persona_idx in retry_slots:
        persona = personas[persona_idx]
        for attempt in range(max_attempts):
            batch = _generate_batch(
                persona=persona,
                agent_description=agent_description,
                agent_code=agent_code,
                eval_spec=eval_spec,
                policy_context=policy_context,
                model=model,
                batch_size=1,
                existing_cases=list(out) + existing_snapshot,
                coverage_gaps=coverage_gaps,
            )
            slot_added, _ = _apply_dedup(
                batch, eval_spec, seen_canonical, seen_key_fields, out
            )
            if slot_added > 0:
                added += slot_added
                break
            logger.debug(
                "Retry attempt %d/%d still a duplicate for persona %d",
                attempt + 1,
                max_attempts,
                persona_idx,
            )
    return added


def _per_persona_parallel_shards_round(
    round_num: int,
    personas: list[dict],
    agent_description: str,
    agent_code: str | None,
    eval_spec: dict,
    policy_context: str | None,
    model: str,
    batch_size: int,
    existing_snapshot: list[dict],
    coverage_gaps: list[dict] | None,
    console: Console,
    *,
    inner_parallel_max: int = PERSONA_INNER_PARALLEL_MAX,
) -> list[list[dict]]:
    """Run personas **one after another**; each persona uses parallel LLM shards.

    Shards split ``batch_size`` across up to *inner_parallel_max* concurrent
    ``_generate_batch`` calls so datapoints for that persona are still produced
    in parallel without firing one giant request per persona across *all*
    personas at once.
    """
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
    )  # local to avoid circular

    n = len(personas)
    raw_batches: list[list[dict]] = [[] for _ in range(n)]
    shard_count = min(inner_parallel_max, batch_size)
    sizes = _split_batch_sizes(batch_size, shard_count)

    with Progress(
        SpinnerColumn(style=BRAND),
        TextColumn(f"[bold {BRAND}]{{task.description}}"),
        console=console,
        transient=True,
    ) as progress:
        for idx, persona in enumerate(personas):
            pname = persona.get("name", f"persona {idx + 1}")
            task_id = progress.add_task(
                f"  Round {round_num} · {pname} · {len(sizes)} parallel shard(s)",
                total=None,
            )

            def _run_shard(shard_sz: int) -> list[dict]:
                if shard_sz <= 0:
                    return []
                return _generate_batch(
                    persona=persona,
                    agent_description=agent_description,
                    agent_code=agent_code,
                    eval_spec=eval_spec,
                    policy_context=policy_context,
                    model=model,
                    batch_size=shard_sz,
                    existing_cases=existing_snapshot,
                    coverage_gaps=coverage_gaps,
                )

            merged: list[dict] = []
            with ThreadPoolExecutor(max_workers=len(sizes)) as executor:
                futures = [executor.submit(_run_shard, sz) for sz in sizes]
                for fut in futures:
                    merged.extend(fut.result())

            raw_batches[idx] = merged
            progress.update(task_id, completed=True)

    return raw_batches


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------


def generate_diverse_synthetic_data(
    agent_description: str,
    model: str,
    num_samples: int = 20,
    num_personas: int = 5,
    agent_code: str | None = None,
    policy_context: str | None = None,
    eval_spec: dict | None = None,
    existing_cases: list[dict] | None = None,
    coverage_gaps: list[dict] | None = None,
    console: Console | None = None,
) -> list[dict]:
    """Full persona-driven generation pipeline.

    Returns a list of ``{input, expected_output}`` dicts. When
    *existing_cases* is provided the returned list contains ONLY the new
    cases (caller is responsible for merging).
    """
    console = console or Console()
    eval_spec = eval_spec or {}
    existing_cases = existing_cases or []

    batch_size = min(10, num_samples)
    max_empty_retries = 2

    t_start = time.monotonic()

    # Phase 1: generate personas (red-team–style diversity before batch case gen)
    console.print()
    console.print(
        f"  [bold {BRAND}]Phase 1 · Red-team personas[/bold {BRAND}]"
        f"  [dim](synthetic test design)[/dim]"
    )
    console.print(
        "  [dim]This step runs one LLM call that invents several fictional users who might "
        "interact with your agent: different roles, skill levels, and communication styles. "
        "Some personas are deliberately adversarial or edge-case–oriented so the suite "
        "exercises policy boundaries and ambiguous inputs, not only ideal requests. "
        "Right after this finishes you’ll see each persona’s name, intent, and style; "
        "Phase 2 then generates concrete [cyan]input[/cyan] / [cyan]expected_output[/cyan] "
        "cases persona by persona.[/dim]"
    )
    with make_spinner_progress(console, transient=True) as progress:
        progress.add_task(
            "  Phase 1 · Red teaming: drafting persona profiles with the model…"
        )
        personas = _generate_personas(
            agent_description,
            agent_code,
            eval_spec,
            policy_context,
            model,
            num_personas,
        )

    # Persona summary table
    persona_table = Table.grid(padding=(0, 2))
    persona_table.add_column(style=f"bold {BRAND}")
    persona_table.add_column(style="bold")
    persona_table.add_column(style="dim")
    for p in personas:
        persona_table.add_row(
            "·",
            p.get("name", "?"),
            f"{p.get('skill_level', '?')} · {p.get('intent', '?')}",
        )
    console.print()
    console.print(persona_table)

    # Phase 2: sequential personas, parallel shards within each persona
    n_personas = len(personas)
    estimated_rounds = max(1, -(-num_samples // (n_personas * batch_size)))  # ceil div
    console.print(
        f"\n  [bold {BRAND}]Phase 2 · Generating {num_samples} test cases[/bold {BRAND}]"
        f"  [dim]{n_personas} personas (one at a time) · "
        f"up to {PERSONA_INNER_PARALLEL_MAX} parallel shard(s)/persona · "
        f"{batch_size} cases/persona/round · ~{estimated_rounds} round(s)[/dim]"
    )

    new_cases: list[dict] = []
    seen_canonical: set[str] = set()
    seen_key_fields: set[tuple[str, ...]] = set()

    for c in existing_cases:
        inp = c.get("input", {})
        if isinstance(inp, dict):
            seen_canonical.add(_canonicalize(inp))
            kf = _get_key_fields(inp)
            if kf:
                seen_key_fields.add(kf)

    consecutive_empty = 0
    round_num = 0

    while len(new_cases) < num_samples and consecutive_empty <= max_empty_retries:
        round_num += 1
        # Snapshot existing cases so all parallel batches see the same context
        snapshot = list(existing_cases + new_cases)

        raw_batches = _per_persona_parallel_shards_round(
            round_num=round_num,
            personas=personas,
            agent_description=agent_description,
            agent_code=agent_code,
            eval_spec=eval_spec,
            policy_context=policy_context,
            model=model,
            batch_size=batch_size,
            existing_snapshot=snapshot,
            coverage_gaps=coverage_gaps,
            console=console,
        )

        # Deduplicate per-persona batch to track which personas had dup drops
        all_raw: list[dict] = []
        retry_slots: list[
            int
        ] = []  # one entry per dup-dropped case, holds persona index
        added = 0
        for persona_idx, batch in enumerate(raw_batches):
            all_raw.extend(batch)
            batch_added, dup_drops = _apply_dedup(
                batch, eval_spec, seen_canonical, seen_key_fields, new_cases
            )
            added += batch_added
            retry_slots.extend([persona_idx] * dup_drops)

        # Retry each dup-dropped slot: regenerate individually, up to 3 attempts each
        retry_added = 0
        if retry_slots:
            retry_added = _retry_dropped_slots(
                retry_slots=retry_slots,
                personas=personas,
                agent_description=agent_description,
                agent_code=agent_code,
                eval_spec=eval_spec,
                policy_context=policy_context,
                model=model,
                existing_snapshot=snapshot,
                coverage_gaps=coverage_gaps,
                seen_canonical=seen_canonical,
                seen_key_fields=seen_key_fields,
                out=new_cases,
            )
            added += retry_added

        pct = int(len(new_cases) / num_samples * 100)
        bar_filled = pct // 5
        bar = (
            f"[{BRAND}]{'█' * bar_filled}[/{BRAND}][dim]{'░' * (20 - bar_filled)}[/dim]"
        )
        retry_note = f"  [dim]+{retry_added} retried[/dim]" if retry_added else ""
        console.print(
            f"  [bold {BRAND}]+{added:>3}[/bold {BRAND}] unique  "
            f"{bar}  [dim]{len(new_cases)}/{num_samples}"
            f"  ({len(all_raw)} raw)[/dim]{retry_note}"
        )

        if added == 0:
            consecutive_empty += 1
        else:
            consecutive_empty = 0

    if len(new_cases) > num_samples:
        new_cases = _stratified_sample(new_cases, num_samples)

    if not new_cases:
        raise DatasetGenerationError(
            f"Generated 0 valid cases after {round_num} round(s). "
            "Check that the eval spec schemas match your agent's actual input/output format, "
            "or try a different model."
        )

    elapsed = time.monotonic() - t_start
    console.print(
        f"\n  [bold {BRAND}]✓[/bold {BRAND}]"
        f"  Generated [bold]{len(new_cases)}[/bold] test cases"
        f"  [dim]in {elapsed:.1f}s · {round_num} round(s) · "
        f"{n_personas} personas/round (sequential) · "
        f"≤{PERSONA_INNER_PARALLEL_MAX} parallel shard(s)/persona[/dim]"
    )

    # Phase 3: coverage report
    _print_coverage_report(new_cases, eval_spec, console)

    return new_cases


def _stratified_sample(cases: list[dict], n: int) -> list[dict]:
    """Downsample while preserving approximate difficulty distribution."""
    if len(cases) <= n:
        return cases
    rng = random.Random(42)
    rng.shuffle(cases)
    return cases[:n]


def _print_coverage_report(
    cases: list[dict], eval_spec: dict, console: Console
) -> None:
    """Print a summary of what the generated dataset covers."""
    if not cases:
        return

    console.print()
    console.print(
        Rule(
            f"[bold {BRAND}]Coverage Report[/bold {BRAND}]  [dim]{len(cases)} cases[/dim]",
            style="dim",
        )
    )
    console.print()

    output_fields = eval_spec.get("output_fields", {})
    enum_coverage: dict[str, set[str]] = {}
    number_stats: dict[str, list[float]] = {}

    for case in cases:
        out = case.get("expected_output", {})
        for field, cfg in output_fields.items():
            val = out.get(field)
            if val is None:
                continue
            if cfg.get("type") == "enum":
                enum_coverage.setdefault(field, set()).add(str(val))
            elif cfg.get("type") == "number" and isinstance(val, (int, float)):
                number_stats.setdefault(field, []).append(float(val))

    if not enum_coverage and not number_stats:
        console.print("  [dim]No enum or numeric fields to report.[/dim]")
        return

    if enum_coverage:
        table = Table(
            border_style=f"{BRAND}", show_header=True, header_style=f"bold {BRAND}"
        )
        table.add_column("Field", style="bold")
        table.add_column("Covered values")
        table.add_column("Missing")
        table.add_column("Pct", justify="right")
        for field, covered in sorted(enum_coverage.items()):
            allowed = set(output_fields.get(field, {}).get("values", []))
            missing = allowed - covered
            pct = len(covered) / len(allowed) * 100 if allowed else 100
            pct_style = "green" if pct == 100 else "yellow" if pct >= 50 else "red"
            table.add_row(
                field,
                ", ".join(sorted(covered)),
                f"[yellow]{', '.join(sorted(missing))}[/yellow]"
                if missing
                else "[dim]—[/dim]",
                f"[{pct_style}]{pct:.0f}%[/{pct_style}]",
            )
        console.print(table)

    if number_stats:
        table = Table(
            border_style=f"{BRAND}", show_header=True, header_style=f"bold {BRAND}"
        )
        table.add_column("Field", style="bold")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Mean", justify="right")
        table.add_column("Spec range")
        for field, vals in sorted(number_stats.items()):
            rng = output_fields.get(field, {}).get("range")
            rng_str = f"{rng[0]}–{rng[1]}" if rng else "[dim]—[/dim]"
            table.add_row(
                field,
                f"{min(vals):.1f}",
                f"{max(vals):.1f}",
                f"{sum(vals) / len(vals):.1f}",
                rng_str,
            )
        console.print(table)
