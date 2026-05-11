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

import contextvars
import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import litellm
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from overmind import SpanType, attrs, set_tag
from overmind.core.logging import stage
from overmind.prompts.data import (
    BATCH_GENERATION_PROMPT,
    PERSONAS_GENERATION_PROMPT,
    SYNTHETIC_DATA_LEGACY_PROMPT,
)
from overmind.utils.display import BRAND, make_spinner_progress
from overmind.utils.llm import llm_completion
from overmind.utils.tracing import force_flush_traces, start_child_span, traced

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
    logger.info(f"Data is a : {data}")
    if isinstance(data, dict) and "test_cases" in data:
        logger.info(f"Data is a dict with a 'test_cases' key: {data.keys()}")
        return data.get("test_cases", [])

    raise ValueError(f"Unrecognized data format in {path}. Expected a JSON array or an object with a 'test_cases' key.")


def check_consistent_fields(cases: list[dict]) -> tuple[bool, set[str], list[int]]:
    """Check whether all cases share the same top-level field names.

    Returns ``(consistent, common_fields, bad_indices)`` where
    *bad_indices* lists the positions of cases whose field set differs from
    the first case.
    """
    if not cases:
        return True, set(), []
    reference = set(cases[0].keys())
    bad: list[int] = []
    for i, case in enumerate(cases[1:], start=1):
        if set(case.keys()) != reference:
            bad.append(i)
    return (not bad), reference, bad


def _field_mapping_path(agent_name: str) -> Path:
    """Return the path for the persisted field-mapping file for *agent_name*."""

    from overmind.core.paths import agent_setup_spec_dir

    return agent_setup_spec_dir(agent_name) / "field_mapping.json"


def normalize_data_fields(
    cases: list[dict],
    console: Console,
    *,
    require_output: bool = True,
    agent_name: str | None = None,
) -> list[dict]:
    """Remap arbitrary data fields to the standard ``input`` / ``expected_output`` keys.

    If the data already uses ``input`` (and, when *require_output* is True,
    ``expected_output``) the cases are returned unchanged.  Otherwise the
    user is shown the available field names and asked to pick which one
    is the agent's input and which is the expected output.

    When the user selects a single field as the input, its value is used
    directly as the ``input`` value (good for data where one field already
    holds a dict of agent parameters).  When they choose
    ``"(all fields except output)"``, all fields other than the selected
    output field are kept as the ``input`` dict — the right default for
    flat datasets where every column is an agent parameter.

    When *agent_name* is provided:
    - If a field-mapping file already exists for that agent, the saved
      mapping is applied silently (no interactive prompt).
    - If no file exists and the user makes a selection, the mapping is
      persisted so subsequent calls (validate, optimize) skip the prompt.
    """

    from overmind.utils.display import BRAND, select_option

    if not cases:
        return cases

    # ── Alias normalisation: inputs → input, outputs → expected_output ────
    # Applied silently before any other check so the rest of the function
    # sees only the canonical key names.
    _INPUT_ALIASES = ("inputs",)
    _OUTPUT_ALIASES = ("outputs",)

    first_keys = set(cases[0].keys())
    rename: dict[str, str] = {}
    if "input" not in first_keys:
        for alias in _INPUT_ALIASES:
            if alias in first_keys:
                rename[alias] = "input"
                break
    if "expected_output" not in first_keys:
        for alias in _OUTPUT_ALIASES:
            if alias in first_keys:
                rename[alias] = "expected_output"
                break

    if rename:
        cases = [{rename.get(k, k): v for k, v in case.items()} for case in cases]

    fields = list(cases[0].keys())

    has_input = "input" in fields
    has_output = "expected_output" in fields

    if has_input and (has_output or not require_output):
        return cases

    mapping_path: Path | None = _field_mapping_path(agent_name) if agent_name else None

    # ── Try to load a previously saved mapping ────────────────────────────
    if mapping_path is not None and mapping_path.exists():
        try:
            saved = json.loads(mapping_path.read_text(encoding="utf-8"))
            input_field: str | None = saved.get("input_field")
            use_all_as_input: bool = bool(saved.get("use_all_as_input", False))
            output_field: str | None = saved.get("output_field")

            # Validate the saved mapping still applies to the current fields
            saved_fields_ok = (use_all_as_input or input_field in fields or input_field == "input") and (
                output_field is None or output_field in fields or output_field == "expected_output"
            )
            if saved_fields_ok:
                console.print(
                    f"  [dim]Using saved field mapping from "
                    f"[cyan]{mapping_path.relative_to(mapping_path.parents[2])}[/cyan]"
                    f" — input: [bold]{input_field if not use_all_as_input else '(all fields except output)'}[/bold]"
                    + (f", output: [bold]{output_field}[/bold]" if output_field else "")
                    + "[/dim]"
                )
                return _apply_field_mapping(cases, input_field, use_all_as_input, output_field)
        except Exception:
            logger.debug("Could not load saved field mapping from %s", mapping_path)

    # ── Interactive selection ─────────────────────────────────────────────
    console.print()
    console.print(
        f"  [bold {BRAND}]Data field mapping[/bold {BRAND}]  "
        "[dim]Your data does not use the standard 'input' / 'expected_output' keys.[/dim]"
    )
    console.print(f"  [dim]Available fields: {', '.join(fields)}[/dim]")
    console.print()

    if not has_input:
        ALL_FIELDS_OPTION = "(all fields except output)"
        input_choices = fields + [ALL_FIELDS_OPTION]
        input_idx = select_option(
            input_choices,
            title="Which field is the agent's input?  (single field → its value becomes 'input'; last option → all non-output fields become 'input')",
            default_index=0,
            console=console,
        )
        input_field = input_choices[input_idx]
        use_all_as_input = input_field == ALL_FIELDS_OPTION
        if use_all_as_input:
            input_field = None
    else:
        input_field = "input"
        use_all_as_input = False

    output_field = None
    if not has_output and require_output:
        remaining = [f for f in fields if f != input_field] if not use_all_as_input else fields
        SKIP_OPTION = "(none — skip expected_output)"
        output_choices = remaining + [SKIP_OPTION]
        output_idx = select_option(
            output_choices,
            title="Which field is the expected output?",
            default_index=0,
            console=console,
        )
        chosen = output_choices[output_idx]
        output_field = None if chosen == SKIP_OPTION else chosen
    elif has_output:
        output_field = "expected_output"

    # ── Persist the mapping for future calls ─────────────────────────────
    if mapping_path is not None:
        try:
            mapping_path.parent.mkdir(parents=True, exist_ok=True)
            mapping_path.write_text(
                json.dumps(
                    {
                        "input_field": input_field,
                        "use_all_as_input": use_all_as_input,
                        "output_field": output_field,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.debug("Saved field mapping to %s", mapping_path)
        except Exception:
            logger.debug("Could not save field mapping to %s", mapping_path)

    return _apply_field_mapping(cases, input_field, use_all_as_input, output_field)


def _apply_field_mapping(
    cases: list[dict],
    input_field: str | None,
    use_all_as_input: bool,
    output_field: str | None,
) -> list[dict]:
    """Remap *cases* to standard ``input`` / ``expected_output`` keys using a resolved mapping."""
    normalized: list[dict] = []
    for case in cases:
        new_case: dict = {}

        if use_all_as_input:
            new_case["input"] = {k: v for k, v in case.items() if k != output_field}
        elif input_field == "input":
            new_case["input"] = case["input"]
        else:
            new_case["input"] = case.get(input_field)

        if output_field == "expected_output":
            new_case["expected_output"] = case["expected_output"]
        elif output_field is not None:
            new_case["expected_output"] = case.get(output_field)

        # When a single field was selected as input, carry any remaining
        # fields (other than input_field and output_field) through as metadata.
        # When use_all_as_input is True every non-output field is already
        # inside new_case["input"], so nothing extra needs to be carried.
        if not use_all_as_input:
            consumed = {input_field, output_field, "input", "expected_output"} - {None}  # type: ignore[arg-type]
            for k, v in case.items():
                if k not in consumed:
                    new_case[k] = v

        normalized.append(new_case)

    return normalized


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def _format_input_schema(eval_spec: dict) -> str:
    """Render input_schema from eval spec into a readable block.

    The input_schema fields correspond EXACTLY to the entrypoint function's
    parameters.  The runner dispatches via ``fn(**input_dict)`` when the
    function has multiple parameters, so the "input" dict keys **must** be
    these field names — no extra keys, no missing required keys.
    """
    schema = eval_spec.get("input_schema", {})
    if not schema:
        return ""
    entrypoint_fn = eval_spec.get("entrypoint_fn", "")
    lines = []
    if entrypoint_fn:
        lines.append(
            f"  (These are the parameters of the entrypoint function `{entrypoint_fn}()`."
            " The input dict keys MUST be exactly these field names.)"
        )
    else:
        lines.append(
            "  (These fields map to the agent entrypoint's function parameters."
            " The input dict keys MUST be exactly these field names.)"
        )
    for field, info in schema.items():
        ftype = info.get("type", "string")
        desc = info.get("description", "")
        lines.append(f"  - {field} ({ftype}): {desc}")

    # When any param is type object/array, sub-keys mentioned in its
    # description can mislead LLMs into flattening them into the top-level
    # input dict.  Render an explicit example shape so the wrapping is
    # unambiguous.
    has_complex = any(
        (info.get("type") if isinstance(info, dict) else "string") in ("object", "array") for info in schema.values()
    )
    if has_complex:
        example: dict[str, Any] = {}
        for field, info in schema.items():
            ftype = info.get("type", "string") if isinstance(info, dict) else "string"
            if ftype == "object":
                example[field] = {"<sub_key>": "<value>", "...": "..."}
            elif ftype == "array":
                example[field] = ["<item>", "..."]
            elif ftype == "number":
                example[field] = 0
            elif ftype == "boolean":
                example[field] = False
            else:
                example[field] = "<value>"
        lines.append("")
        lines.append(
            "  Example input shape (top-level keys MUST be the entrypoint param "
            "names above; do NOT flatten sub-keys of object/array params into "
            "the top level):"
        )
        lines.append(f"    {json.dumps(example)}")
    return "\n".join(lines)


def _maybe_repair_case_input(case: dict, eval_spec: dict) -> None:
    """Best-effort fix for a common LLM mistake: when the entrypoint takes a
    single ``object``-typed parameter (e.g. ``run(input_data: dict)``), models
    often flatten the sub-keys (``company_name``, ``inquiry_text``, …) into
    the top-level ``input`` dict instead of nesting them under the param
    name.  Detect that exact shape and wrap it.  Mutates *case* in place.
    """
    if not isinstance(case, dict):
        return
    inp = case.get("input")
    if not isinstance(inp, dict):
        return
    schema = eval_spec.get("input_schema") or {}
    if len(schema) != 1:
        return
    (only_field, info) = next(iter(schema.items()))
    ftype = info.get("type", "string") if isinstance(info, dict) else "string"
    if ftype not in ("object", "array"):
        return
    # If the input already conforms (has the wrapping key) leave it alone.
    if only_field in inp and len(inp) == 1:
        return
    # Otherwise wrap whatever the LLM produced as the param value.
    case["input"] = {only_field: inp}


def _format_output_schema(eval_spec: dict) -> str:
    """Render output_fields from eval spec with types, enums, and ranges."""
    fields = eval_spec.get("output_fields", {})
    if not fields:
        return ""
    lines = []
    has_optional = False
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
        marker = ""
        if cfg.get("optional", False):
            marker = " [optional — only present on some code paths]"
            has_optional = True
        lines.append(f"  - {field} ({ftype}){marker}: {desc}{extra}")
    text = "\n".join(lines)
    if has_optional:
        text += (
            "\n\nNote: fields marked [optional] are present on some code paths "
            "and absent on others (e.g. success vs. error branches). Include "
            "them ONLY in cases whose branch actually populates them, and OMIT "
            "them from cases on the other branch."
        )
    return text


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_case_against_spec(case: dict, eval_spec: dict) -> list[str]:
    """Return a list of validation error strings (empty = valid).

    ``input`` may be a dict (mapped to entrypoint parameters via ``input_schema``)
    or a plain string (e.g. single-message agents); string inputs skip key-level
    ``input_schema`` checks.

    ``expected_output`` may be a dict (validated against ``output_fields`` when
    present) or a non-empty string (e.g. prose / markdown agents).
    """
    errors: list[str] = []
    inp = case.get("input")
    out = case.get("expected_output")

    if inp is None:
        errors.append("Missing 'input'")
        return errors

    if out is None:
        errors.append("Missing 'expected_output'")
        return errors

    if not isinstance(inp, (dict, str)):
        errors.append("'input' must be a dict or a string")
        return errors

    if isinstance(inp, dict):
        input_schema = eval_spec.get("input_schema", {})
        for field, info in input_schema.items():
            is_optional = info.get("optional", False) if isinstance(info, dict) else False
            if field not in inp and not is_optional:
                errors.append(f"input missing required field '{field}'")
        # Reject unexpected keys — the runner dispatches via **kwargs so
        # extra keys would cause a TypeError on the entrypoint function.
        if input_schema:
            extra = set(inp.keys()) - set(input_schema.keys())
            if extra:
                errors.append(
                    f"input has unexpected keys {sorted(extra)} not in entrypoint schema {sorted(input_schema.keys())}"
                )

    if isinstance(out, dict):
        output_fields = eval_spec.get("output_fields", {})
        for field, cfg in output_fields.items():
            # Mirror input_schema handling: a field marked ``optional: true``
            # may be absent.  This is essential for agents whose output is a
            # discriminated union (e.g. ``{status="success", ...}`` vs
            # ``{status="error", error_message}`` — different fields are
            # populated on different code paths).
            is_optional = cfg.get("optional", False) if isinstance(cfg, dict) else False
            if field not in out:
                if not is_optional:
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
                    errors.append(f"expected_output.{field} must be a number, got {type(val).__name__}")
                else:
                    rng = cfg.get("range")
                    if rng and (val < rng[0] or val > rng[1]):
                        errors.append(f"expected_output.{field} = {val} outside range {rng}")
            elif ftype == "text":
                if cfg.get("eval_mode") == "non_empty" and not val:
                    errors.append(f"expected_output.{field} must be non-empty")
    elif isinstance(out, str):
        if not out.strip():
            errors.append("expected_output is an empty string")

    return errors


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _safe_parse_json(text: str) -> Any:
    """Best-effort JSON extraction from LLM output.

    Handles common LLM quirks: markdown fences, leading commentary,
    unescaped control characters inside string literals, and trailing commas.
    """
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
            # Try repairing the fenced content too
            repaired = _repair_json_string(fenced.group(1).strip())
            if repaired is not None:
                return repaired

    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start >= 0 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                repaired = _repair_json_string(candidate)
                if repaired is not None:
                    return repaired

    repaired = text.replace("'", '"')
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    return None


def _repair_json_string(text: str) -> Any:
    """Try to fix common JSON issues from LLM output.

    Handles: unescaped newlines/tabs inside string values, trailing commas,
    single quotes used as string delimiters.
    """
    # Escape literal newlines and tabs that appear inside JSON string values.
    # Walk character by character to only fix characters inside quotes.
    result: list[str] = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '"' and (i == 0 or text[i - 1] != "\\"):
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == "\n":
            result.append("\\n")
        elif in_string and ch == "\r":
            result.append("\\r")
        elif in_string and ch == "\t":
            result.append("\\t")
        else:
            result.append(ch)
        i += 1
    repaired = "".join(result)

    # Remove trailing commas before } or ]
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
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
            logger.warning("Timeout on LLM call (attempt %d/%d)", attempt + 1, max_retries)
        except Exception:
            logger.exception("Unexpected error on LLM call (attempt %d/%d)", attempt + 1, max_retries)
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
    trimmed = agent_code[:max_chars] if max_chars and len(agent_code) > max_chars else agent_code
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


@traced(span_name="overmind_generate_synthetic_data", type=SpanType.FUNCTION)
def generate_synthetic_data(
    agent_description: str,
    model: str,
    num_samples: int = 15,
    agent_code: str | None = None,
    policy_context: str | None = None,
    expected_output_hint: str = "",
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

    output_format_section = ""
    if expected_output_hint:
        output_format_section = (
            f"\n## Expected Output Format\n\n"
            f"The agent's output is: **{expected_output_hint}**.\n"
            f"Shape every `expected_output` in the test cases to match this format.\n"
        )

    prompt = SYNTHETIC_DATA_LEGACY_PROMPT.format(
        num_samples=num_samples,
        agent_description=agent_description,
        code_section=code_section,
        policy_section=policy_section,
        output_format_section=output_format_section,
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
            set_tag(attrs.DATAGEN_MODE, "legacy")
            set_tag(attrs.DATAGEN_MODEL, model)
            set_tag(attrs.DATAGEN_REQUESTED_SAMPLES, str(num_samples))
            set_tag(attrs.DATAGEN_GENERATED_COUNT, str(len(cases)))
            return cases

    raise ValueError(
        "Failed to parse synthetic data from the LLM response. Try running again or use a different model."
    )


# ===================================================================
# Full persona-driven generation pipeline
# ===================================================================


@traced(span_name="overmind_generate_redteam_personas", type=SpanType.FUNCTION)
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
        logger.warning("Persona generation returned nothing; falling back to default personas")
        return _default_personas(num_personas)

    parsed = _safe_parse_json(raw)
    if isinstance(parsed, dict) and "personas" in parsed:
        personas = parsed["personas"]
        set_tag(attrs.DATAGEN_PERSONA_COUNT, str(len(personas)))
        set_tag(attrs.DATAGEN_PERSONA_SOURCE, "llm")
        if isinstance(personas, list) and len(personas) > 0:
            return personas

    logger.warning("Could not parse personas from LLM; falling back to defaults")
    fallback = _default_personas(num_personas)
    set_tag(attrs.DATAGEN_PERSONA_COUNT, str(len(fallback)))
    set_tag(attrs.DATAGEN_PERSONA_SOURCE, "fallback")
    return fallback


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
    expected_output_hint: str = "",
    variation_directive: str = "",
    temperature: float = 0.8,
) -> list[dict]:
    """Phase 2: generate a batch of {input, expected_output} cases for one persona.

    Callers can pass *variation_directive* to narrow the persona's scenario
    focus (used by shards to partition the persona's behavior space) or to
    inject anti-examples during retry (used when dedup keeps rejecting the
    same canonical cases).  Temperature can be bumped on retries to give the
    model room to explore.
    """
    input_schema_text = _format_input_schema(eval_spec)
    output_schema_text = _format_output_schema(eval_spec)

    code_section = ""
    if agent_code:
        code_section = _format_code_for_prompt(agent_code)

    policy_section = ""
    if policy_context:
        policy_section = f"\n<PolicyContext>\n{policy_context}\n</PolicyContext>\n"

    variation_section = ""
    if variation_directive:
        variation_section = f"\n<VariationDirective>\n{variation_directive.strip()}\n</VariationDirective>\n"

    existing_section = ""
    if existing_cases:
        sample = existing_cases if len(existing_cases) <= 15 else random.sample(existing_cases, 15)
        inputs_only = [c.get("input", {}) for c in sample]
        existing_section = (
            "\n<ExistingCases>\n"
            "These cases already exist — do NOT duplicate them:\n"
            f"{json.dumps(inputs_only, indent=2, default=str)}\n"
            "</ExistingCases>\n"
        )

    # Prepend the variation directive so it lands right before ExistingCases
    # in the {existing_section} template slot without a prompt-template
    # breaking change.
    existing_section = variation_section + existing_section

    gap_section = ""
    if coverage_gaps:
        gap_lines = []
        for gap in coverage_gaps[:10]:
            gap_lines.append(
                f"  - [{gap.get('severity', 'medium')}] {gap.get('area', '')}: {gap.get('description', '')}"
            )
        gap_section = (
            "\n**Priority**: Generate cases that specifically cover these gaps:\n" + "\n".join(gap_lines) + "\n"
        )

    output_format_section = ""
    if expected_output_hint:
        output_format_section = (
            "\n<OutputFormatHint>\n"
            f"The user has specified that the agent's output is: **{expected_output_hint}**.\n"
            "Shape every `expected_output` in the test cases to match this format. "
            "For example, if the output is plain text, `expected_output` should be a "
            "string (or a dict with a single key whose value is a string), not a "
            "multi-field JSON object.\n"
            "</OutputFormatHint>\n"
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
        output_format_section=output_format_section,
    )

    raw = _llm_call(model, prompt, temperature=temperature, max_tokens=16000)
    if not raw:
        logger.warning("Batch generation returned nothing for persona %s", persona_name)
        return []

    parsed = _safe_parse_json(raw)
    if parsed is None:
        logger.warning(
            "Failed to parse batch JSON for persona %s (response length: %d, first 500 chars: %.500s)",
            persona_name,
            len(raw),
            raw,
        )
        return []

    if isinstance(parsed, dict) and "cases" in parsed:
        cases = parsed["cases"]
    elif isinstance(parsed, list):
        cases = parsed
    else:
        logger.warning("Unexpected batch response structure for persona %s", persona_name)
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


# Fallback variation axes when a persona doesn't expose typical_scenarios.
# Picked to push the model toward orthogonal regions of the input space so
# parallel shards don't re-discover the same "obvious" cases.
_GENERIC_VARIATION_AXES = (
    "minimal, empty, or missing-field inputs",
    "boundary values and numeric edge cases",
    "complex multi-signal realistic inputs",
    "ambiguous or conflicting signals",
    "typical / common-path usage",
)


def _shard_variation_directive(persona: dict, shard_idx: int, num_shards: int, round_num: int = 1) -> str:
    """Build a shard-specific scenario focus so parallel shards don't collide.

    Each shard for a persona is pinned to a distinct ``typical_scenarios``
    entry (rotated by ``round_num`` so re-rounds hit fresh scenarios).  When
    the persona has fewer scenarios than shards, generic orthogonal axes
    fill the gap.
    """
    scenarios = [s for s in (persona.get("typical_scenarios") or []) if isinstance(s, str) and s.strip()]
    rotation = max(0, round_num - 1)
    if scenarios:
        focus = scenarios[(shard_idx + rotation) % len(scenarios)]
    elif num_shards > 1:
        focus = _GENERIC_VARIATION_AXES[shard_idx % len(_GENERIC_VARIATION_AXES)]
    else:
        return ""

    axis = _GENERIC_VARIATION_AXES[(shard_idx + rotation) % len(_GENERIC_VARIATION_AXES)]
    return (
        f"Within this persona's behavior space, focus THIS batch on: **{focus}**.\n"
        f"- Generate cases that specifically exercise that sub-scenario.\n"
        f"- Avoid cases that would belong to the persona's other typical scenarios.\n"
        f"- Vary inputs along this orthogonal axis: {axis}.\n"
        f"- Do not produce stock/template inputs — change phrasing, values, and structure."
    )


def _retry_variation_directive(persona: dict, rejected_inputs: list, attempt: int) -> str:
    """Build a retry directive that shows the model what NOT to produce.

    The most-recent rejected inputs (deduped) are serialized so the LLM has
    explicit anti-examples to avoid.  Temperature bump + this directive
    eliminates the "same canonical case, forever" retry loop.
    """
    persona_name = persona.get("name", "this persona")
    # Keep only the last handful to stay token-cheap and keep the signal sharp.
    recent = rejected_inputs[-8:] if rejected_inputs else []
    if not recent:
        return (
            f"The previous batch for {persona_name} produced only duplicates. "
            f"This is retry attempt {attempt}. Generate a fundamentally "
            f"different input — vary subject matter, phrasing, numeric ranges, "
            f"and structural shape simultaneously."
        )
    rejected_json = json.dumps(recent, indent=2, default=str)
    return (
        f"RETRY (attempt {attempt}) for {persona_name}.\n"
        "The inputs below were JUST REJECTED as duplicates of previously-accepted cases.\n"
        "DO NOT produce inputs structurally similar to any of them:\n"
        "<RejectedAttempts>\n"
        f"{rejected_json}\n"
        "</RejectedAttempts>\n"
        "Produce ONE case that differs along at least TWO orthogonal axes: "
        "subject matter, phrasing, structural shape, boundary values, or edge behavior."
    )


def _apply_dedup(
    raw_cases: list[dict],
    eval_spec: dict,
    seen_canonical: set[str],
    seen_key_fields: set[tuple[str, ...]],
    out: list[dict],
    rejected_inputs: list | None = None,
    schema_errors_out: list[list[str]] | None = None,
) -> tuple[int, int]:
    """Validate and deduplicate *raw_cases* into *out* (mutates both sets and out).

    Returns ``(added, dup_drops)`` where ``dup_drops`` is the count of cases
    that were structurally valid but dropped solely because they were duplicates
    (distinct from schema/type failures which are unretriable).

    When *rejected_inputs* is provided, every input that was dropped as a
    duplicate is appended to it.  Callers use this to build anti-example
    prompts for retry attempts.

    When *schema_errors_out* is provided, the error list for every
    schema-rejected case is appended to it.  This exists because schema
    rejections are silent failures by design (they don't feed retry) and
    used to mask a completely broken spec as "no unique cases generated".
    Callers surface these upstream so the user can actually see the problem.
    """
    added = 0
    dup_drops = 0
    for case in raw_cases:
        if not isinstance(case, dict):
            continue
        case.pop("_meta", None)

        if eval_spec:
            _maybe_repair_case_input(case, eval_spec)
            errors = validate_case_against_spec(case, eval_spec)
            if errors:
                logger.debug("Discarding invalid case: %s", errors)
                if schema_errors_out is not None:
                    schema_errors_out.append(errors)
                continue

        inp = case.get("input", {})

        if isinstance(inp, dict):
            canon = _canonicalize(inp)
        else:
            canon = str(inp).strip().lower()

        if canon in seen_canonical:
            dup_drops += 1
            if rejected_inputs is not None:
                rejected_inputs.append(inp)
            continue
        if isinstance(inp, dict) and _is_near_duplicate(inp, seen_key_fields):
            dup_drops += 1
            if rejected_inputs is not None:
                rejected_inputs.append(inp)
            continue

        seen_canonical.add(canon)
        kf = _get_key_fields(inp) if isinstance(inp, dict) else ()
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
    max_attempts: int = 2,
    expected_output_hint: str = "",
    rejected_by_persona: dict[int, list[dict]] | None = None,
    skip_personas: set[int] | None = None,
    per_persona_added: dict[int, int] | None = None,
) -> int:
    """For each dedup-dropped slot, try to regenerate a single unique case.

    Each slot corresponds to one persona (by index).  Up to *max_attempts*
    single-case generations are tried per slot; the slot is abandoned if all
    attempts still produce a duplicate.  Runs serially to keep dedup
    state consistent.

    Diversification strategy (added to break the "same canonical case
    forever" retry loop):

    * ``rejected_by_persona`` carries the inputs that dedup just rejected
      for each persona.  They're injected into the retry prompt as
      explicit anti-examples so the LLM has a concrete "do not reproduce"
      signal instead of just a larger ExistingCases list.
    * Temperature is bumped to ``1.0`` on retries (the original pass uses
      ``0.8``) to widen the sampling distribution.
    * Each failed retry attempt adds its own output to the anti-example
      list so subsequent attempts see even more rejected patterns.
    * ``skip_personas`` short-circuits slots whose persona has already met
      quota — a narrow persona (e.g. "Novice User") will no longer block
      the whole round once it has produced its share.

    Returns the total number of cases successfully added across all slots.
    """
    added = 0
    rejected_by_persona = rejected_by_persona or {}
    skip_personas = skip_personas or set()
    per_persona_added = per_persona_added if per_persona_added is not None else {}

    for persona_idx in retry_slots:
        if persona_idx in skip_personas:
            logger.debug(
                "Skipping retry slot for persona %d (quota already met)",
                persona_idx,
            )
            continue

        persona = personas[persona_idx]
        # Start with the inputs that dedup rejected for this persona in the
        # main round; append any further duplicates we generate during retry.
        anti_examples = list(rejected_by_persona.get(persona_idx, []))

        for attempt in range(max_attempts):
            directive = _retry_variation_directive(persona, anti_examples, attempt + 1)
            with start_child_span("overmind_datagen_retry", span_type=SpanType.FUNCTION):
                set_tag(attrs.DATAGEN_PERSONA_IDX, str(persona_idx))
                set_tag(attrs.DATAGEN_RETRY_ATTEMPT, str(attempt + 1))
                set_tag(attrs.DATAGEN_RETRY_MAX_ATTEMPTS, str(max_attempts))
                set_tag(attrs.DATAGEN_ANTI_EXAMPLES, str(len(anti_examples)))
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
                    expected_output_hint=expected_output_hint,
                    variation_directive=directive,
                    temperature=1.0,
                )
            retry_rejected: list = []
            slot_added, _ = _apply_dedup(
                batch,
                eval_spec,
                seen_canonical,
                seen_key_fields,
                out,
                rejected_inputs=retry_rejected,
            )
            if slot_added > 0:
                added += slot_added
                per_persona_added[persona_idx] = per_persona_added.get(persona_idx, 0) + slot_added
                logger.debug(
                    "Retry succeeded for persona %d on attempt %d/%d",
                    persona_idx,
                    attempt + 1,
                    max_attempts,
                )
                break
            # Feed this attempt's output into the next attempt's anti-examples.
            anti_examples.extend(retry_rejected)
            logger.debug(
                "Retry attempt %d/%d still a duplicate for persona %d (anti_examples=%d)",
                attempt + 1,
                max_attempts,
                persona_idx,
                len(anti_examples),
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
    expected_output_hint: str = "",
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

            shard_directives = [
                _shard_variation_directive(persona, shard_idx=s, num_shards=len(sizes), round_num=round_num)
                for s in range(len(sizes))
            ]

            def _run_shard(shard_sz: int, directive: str) -> list[dict]:
                if shard_sz <= 0:
                    return []
                return _generate_batch(
                    persona=persona,  # noqa: B023
                    agent_description=agent_description,
                    agent_code=agent_code,
                    eval_spec=eval_spec,
                    policy_context=policy_context,
                    model=model,
                    batch_size=shard_sz,
                    existing_cases=existing_snapshot,
                    coverage_gaps=coverage_gaps,
                    expected_output_hint=expected_output_hint,
                    variation_directive=directive,
                )

            merged: list[dict] = []
            with (
                start_child_span("overmind_datagen_persona", span_type=SpanType.FUNCTION),
                stage(
                    "data.phase2.persona",
                    logger=logger,
                    round=round_num,
                    persona_idx=idx,
                    persona=pname,
                    intent=persona.get("intent", "?"),
                    shards=len(sizes),
                    batch_size=batch_size,
                    directives=[d.splitlines()[0] if d else "(none)" for d in shard_directives],
                ) as pinfo,
            ):
                set_tag(attrs.DATAGEN_ROUND, str(round_num))
                set_tag(attrs.DATAGEN_PERSONA_IDX, str(idx))
                set_tag(attrs.DATAGEN_PERSONA_NAME, str(pname))
                set_tag(attrs.DATAGEN_PERSONA_INTENT, str(persona.get("intent", "?")))
                set_tag(attrs.DATAGEN_PERSONA_SHARDS, str(len(sizes)))
                with ThreadPoolExecutor(max_workers=len(sizes)) as executor:
                    # Propagate the OTel/contextvars context so spans
                    # emitted inside each shard (e.g. overmind_llm_completion)
                    # nest under the parent setup workflow span instead of
                    # becoming orphan root traces.
                    parent_ctx = contextvars.copy_context()
                    futures = [
                        executor.submit(
                            parent_ctx.copy().run,
                            _run_shard,
                            sz,
                            shard_directives[s],
                        )
                        for s, sz in enumerate(sizes)
                    ]
                    for fut in futures:
                        try:
                            merged.extend(fut.result())
                        except Exception:
                            logger.exception("Synthetic shard failed for persona idx=%s", idx)
                pinfo["raw_cases"] = len(merged)

            raw_batches[idx] = merged
            progress.update(task_id, completed=True)
            force_flush_traces()

    return raw_batches


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------


@traced(span_name="overmind_generate_diverse_data", type=SpanType.WORKFLOW)
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
    expected_output_hint: str = "",
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
    console.print(f"  [bold {BRAND}]Phase 1 · Red-team personas[/bold {BRAND}]  [dim](synthetic test design)[/dim]")
    console.print(
        "  [dim]This step runs one LLM call that invents several fictional users who might "
        "interact with your agent: different roles, skill levels, and communication styles. "
        "Some personas are deliberately adversarial or edge-case–oriented so the suite "
        "exercises policy boundaries and ambiguous inputs, not only ideal requests. "
        "Right after this finishes you’ll see each persona’s name, intent, and style; "
        "Phase 2 then generates concrete [cyan]input[/cyan] / [cyan]expected_output[/cyan] "
        "cases persona by persona.[/dim]"
    )
    with (
        stage(
            "data.phase1.red_team_personas",
            logger=logger,
            num_personas=num_personas,
            model=model,
        ) as phase1,
        make_spinner_progress(console, transient=True) as progress,
    ):
        progress.add_task("  Phase 1 · Red teaming: drafting persona profiles with the model…")
        personas = _generate_personas(
            agent_description,
            agent_code,
            eval_spec,
            policy_context,
            model,
            num_personas,
        )
        phase1["got_personas"] = len(personas)
        phase1["names"] = [p.get("name", "?") for p in personas]
        phase1["intents"] = [p.get("intent", "?") for p in personas]

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
        else:
            seen_canonical.add(str(inp).strip().lower())

    consecutive_empty = 0
    round_num = 0

    # Soft per-persona quota so a narrow persona (e.g. "Novice User") can't
    # monopolize retry budget once it has produced its fair share.  +1 slack
    # lets an over-performing persona slightly exceed its quota before we
    # start skipping it.
    per_persona_added: dict[int, int] = {idx: 0 for idx in range(len(personas))}
    quota_per_persona = max(2, -(-num_samples // max(1, len(personas))) + 1)

    while len(new_cases) < num_samples and consecutive_empty <= max_empty_retries:
        round_num += 1
        snapshot = list(existing_cases + new_cases)

        # Per-round child span so progress flushes to the trace UI even
        # while the outer pipeline span is still open.
        with (
            start_child_span(
                f"overmind_datagen_round_{round_num}",
                span_type=SpanType.FUNCTION,
            ),
            stage(
                "data.phase2.generation_round",
                logger=logger,
                round=round_num,
                target=num_samples,
                have=len(new_cases),
                snapshot_size=len(snapshot),
                quota_per_persona=quota_per_persona,
                per_persona=dict(per_persona_added),
            ) as phase2,
        ):
            set_tag(attrs.DATAGEN_ROUND, str(round_num))
            set_tag(attrs.DATAGEN_TARGET, str(num_samples))
            set_tag(attrs.DATAGEN_HAVE_BEFORE, str(len(new_cases)))
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
                expected_output_hint=expected_output_hint,
            )

            # Dedup per-persona batch, capturing each persona's rejected
            # inputs so retries get concrete "do not reproduce" anti-examples.
            all_raw: list[dict] = []
            retry_slots: list[int] = []
            rejected_by_persona: dict[int, list[dict]] = {}
            schema_errors_this_round: list[list[str]] = []
            added = 0
            for persona_idx, batch in enumerate(raw_batches):
                all_raw.extend(batch)
                persona_rejected: list = []
                batch_added, dup_drops = _apply_dedup(
                    batch,
                    eval_spec,
                    seen_canonical,
                    seen_key_fields,
                    new_cases,
                    rejected_inputs=persona_rejected,
                    schema_errors_out=schema_errors_this_round,
                )
                added += batch_added
                per_persona_added[persona_idx] += batch_added
                retry_slots.extend([persona_idx] * dup_drops)
                if persona_rejected:
                    rejected_by_persona[persona_idx] = persona_rejected

            schema_drops = len(schema_errors_this_round)
            if schema_drops:
                # Never let a spec mismatch masquerade as "0 unique cases".
                # Surface up to 3 unique error patterns so the user sees the
                # real problem immediately.
                unique_patterns: list[str] = []
                seen_patterns: set[str] = set()
                for errs in schema_errors_this_round:
                    if not errs:
                        continue
                    key = " | ".join(sorted(errs))
                    if key in seen_patterns:
                        continue
                    seen_patterns.add(key)
                    unique_patterns.append(key)
                    if len(unique_patterns) >= 3:
                        break
                logger.warning(
                    "Schema validation dropped %d/%d raw case(s) this round; distinct error patterns: %s",
                    schema_drops,
                    len(all_raw),
                    unique_patterns,
                )
                # If the ENTIRE round was killed by schema rejects, that's a
                # spec/LLM contract mismatch — shout at the user so they don't
                # sit watching "+0 unique" rounds forever.
                if added == 0 and not retry_slots:
                    console.print(
                        f"  [bold red]![/bold red]  {schema_drops}/{len(all_raw)} "
                        f"raw cases were rejected by the output schema. "
                        f"[dim]Likely an output_fields mismatch "
                        f"(e.g. fields that should be marked optional).[/dim]"
                    )
                    for patt in unique_patterns:
                        console.print(f"    [dim]· {patt}[/dim]")

            # Skip retries for personas that have already met quota: they're
            # statistically more likely to keep producing duplicates and
            # burning retry budget.
            skip_personas = {idx for idx, n in per_persona_added.items() if n >= quota_per_persona}

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
                    expected_output_hint=expected_output_hint,
                    rejected_by_persona=rejected_by_persona,
                    skip_personas=skip_personas,
                    per_persona_added=per_persona_added,
                )
                added += retry_added

            phase2["raw_cases"] = len(all_raw)
            phase2["added_unique"] = added
            phase2["schema_drops"] = schema_drops
            phase2["retry_slots"] = len(retry_slots)
            phase2["retry_skipped"] = sum(1 for idx in retry_slots if idx in skip_personas)
            phase2["retry_added"] = retry_added
            phase2["total_after"] = len(new_cases)
            phase2["per_persona_after"] = dict(per_persona_added)

        pct = int(len(new_cases) / num_samples * 100)
        bar_filled = pct // 5
        bar = f"[{BRAND}]{'█' * bar_filled}[/{BRAND}][dim]{'░' * (20 - bar_filled)}[/dim]"
        retry_note = f"  [dim]+{retry_added} retried[/dim]" if retry_added else ""
        schema_note = f"  [dim red]-{schema_drops} schema[/dim red]" if schema_drops else ""
        console.print(
            f"  [bold {BRAND}]+{added:>3}[/bold {BRAND}] unique  "
            f"{bar}  [dim]{len(new_cases)}/{num_samples}"
            f"  ({len(all_raw)} raw)[/dim]{retry_note}{schema_note}"
        )
        force_flush_traces()

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

    set_tag(attrs.DATAGEN_MODE, "diverse_persona_pipeline")
    set_tag(attrs.DATAGEN_MODEL, model)
    set_tag(attrs.DATAGEN_REQUESTED_SAMPLES, str(num_samples))
    set_tag(attrs.DATAGEN_GENERATED_COUNT, str(len(new_cases)))
    set_tag(attrs.DATAGEN_ROUNDS, str(round_num))
    set_tag(attrs.DATAGEN_ELAPSED_SECONDS, f"{elapsed:.1f}")
    set_tag(attrs.DATAGEN_EXISTING_CASES, str(len(existing_cases)))
    if coverage_gaps:
        set_tag(attrs.DATAGEN_COVERAGE_GAP_COUNT, str(len(coverage_gaps)))

    # Phase 3: coverage report
    with stage(
        "data.phase3.coverage_report",
        logger=logger,
        cases=len(new_cases),
    ):
        _print_coverage_report(new_cases, eval_spec, console)

    return new_cases


def _stratified_sample(cases: list[dict], n: int) -> list[dict]:
    """Downsample while preserving approximate difficulty distribution."""
    if len(cases) <= n:
        return cases
    rng = random.Random(42)
    rng.shuffle(cases)
    return cases[:n]


def _print_coverage_report(cases: list[dict], eval_spec: dict, console: Console) -> None:
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
        if not isinstance(out, dict):
            continue
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
        table = Table(border_style=f"{BRAND}", show_header=True, header_style=f"bold {BRAND}")
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
                f"[yellow]{', '.join(sorted(missing))}[/yellow]" if missing else "[dim]—[/dim]",
                f"[{pct_style}]{pct:.0f}%[/{pct_style}]",
            )
        console.print(table)

    if number_stats:
        table = Table(border_style=f"{BRAND}", show_header=True, header_style=f"bold {BRAND}")
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
