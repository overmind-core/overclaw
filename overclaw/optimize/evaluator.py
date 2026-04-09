"""Spec-driven evaluation scoring for agent outputs.

Reads an evaluation spec (e.g. ``agents/<name>/eval_spec/eval_spec.json``) produced by ``setup.py``
and scores agent outputs dynamically based on the configured field types,
weights, and tolerances.

Supports four scoring layers:
1. **Mechanical** — deterministic field-level matching (enum, number, text, boolean)
2. **LLM-as-Judge** — semantic quality assessment via a strong model
3. **Tool Usage** — checks tool call completeness, argument quality, and data chaining
4. **Type Correctness** — penalizes outputs with wrong field types
"""

import json
import logging
import re
import time
from pathlib import Path

from overclaw.utils.llm import llm_completion
from overclaw.prompts.evaluator import (
    LLM_JUDGE_BATCH_PROMPT,
    LLM_JUDGE_PROMPT,
    LLM_TEXT_FIELD_JUDGE_PROMPT,
)

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "they",
        "them",
        "their",
        "what",
        "which",
        "who",
        "whom",
    }
)

_JUDGE_MAX_RETRIES = 3
_JUDGE_RETRY_BACKOFF = 1.5
_JUDGE_FALLBACK_SCORE = 0.5


class SpecEvaluator:
    """Scores agent outputs according to an evaluation spec."""

    def __init__(
        self,
        spec_path: str,
        llm_judge_model: str | None = None,
        policy_judge_rubric: str = "",
    ):
        with open(spec_path) as f:
            self.spec = json.load(f)
        self.fields: dict[str, dict] = self.spec["output_fields"]
        self.structure_weight: float = self.spec.get("structure_weight", 20)
        self.llm_judge_model = llm_judge_model
        self.tool_config: dict = self.spec.get("tool_config", {})
        self.policy_judge_rubric = policy_judge_rubric
        self._validate_spec()

    def _validate_spec(self) -> None:
        """Sanity-check that spec weights are internally consistent."""
        import warnings

        total_declared = self.spec.get("total_points", 100)
        field_sum = sum(float(cfg.get("weight", 0)) for cfg in self.fields.values())
        other = (
            self.structure_weight
            + float(self.spec.get("tool_usage_weight", 0))
            + float(self.spec.get("llm_judge_weight", 0))
        )
        actual_total = field_sum + other
        if abs(actual_total - total_declared) > 1.0:
            warnings.warn(
                f"Eval spec weight mismatch: field weights ({field_sum:.0f}) + "
                f"structure ({self.structure_weight:.0f}) + extras "
                f"({other - self.structure_weight:.0f}) = {actual_total:.0f}, "
                f"but total_points is {total_declared}. "
                f"Scores may not reach the declared maximum.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_output(
        self,
        output: dict,
        expected: dict,
        input_data: dict | None = None,
        tool_trace: list[dict] | None = None,
        *,
        _skip_judge: bool = False,
    ) -> dict:
        """Score a single output against expected ground truth.

        Returns a dict with per-dimension scores and a ``total`` (0–100).
        When ``_skip_judge`` is True, the LLM judge call is deferred (used
        by ``evaluate_batch`` to batch judge calls for efficiency).
        """
        scores: dict[str, float] = {}

        # --- Structure scoring (presence check) ---
        expected_fields = list(self.fields.keys())
        field_importances = {
            name: cfg.get("importance", "important")
            for name, cfg in self.fields.items()
        }
        weighted_present = 0.0
        weighted_total = 0.0
        for f in expected_fields:
            imp_mult = {"critical": 3, "important": 2, "minor": 1}.get(
                field_importances.get(f, "important"), 2
            )
            weighted_total += imp_mult
            val = output.get(f)
            if val is not None and val != "":
                weighted_present += imp_mult
        scores["structure"] = (
            weighted_present / max(weighted_total, 1)
        ) * self.structure_weight

        # --- Per-field mechanical scoring ---
        for field_name, config in self.fields.items():
            ftype = config["type"]
            if ftype == "enum":
                scores[field_name] = self._score_enum(
                    output.get(field_name), expected.get(field_name), config
                )
            elif ftype == "number":
                scores[field_name] = self._score_number(
                    output.get(field_name), expected.get(field_name), config
                )
            elif ftype == "text":
                scores[field_name] = self._score_text(
                    output.get(field_name),
                    expected.get(field_name),
                    config,
                    field_name=field_name,
                    input_data=input_data,
                )
            elif ftype == "boolean":
                weight = config.get("weight", 0)
                match = output.get(field_name) == expected.get(field_name)
                scores[field_name] = float(weight) if match else 0.0

        # --- Type correctness penalty ---
        type_penalty = self._check_type_correctness(output)
        scores["type_correctness_penalty"] = type_penalty

        # --- Cross-field consistency ---
        consistency_penalty = self._check_cross_field_consistency(output, expected)
        scores["consistency_penalty"] = consistency_penalty

        # --- Tool usage scoring ---
        tool_score = self._score_tool_usage(tool_trace)
        tool_weight = self.spec.get("tool_usage_weight", 0)
        scores["tool_usage"] = tool_score * tool_weight

        # --- LLM-as-Judge (blended in if configured) ---
        judge_weight = self.spec.get("llm_judge_weight", 0)
        if not _skip_judge and self.llm_judge_model and judge_weight > 0 and input_data:
            judge_score = self._score_with_llm_judge(input_data, expected, output)
            scores["llm_judge"] = judge_score * judge_weight

        scores["total"] = max(0.0, sum(scores.values()))
        return scores

    def evaluate_batch(self, results: list[dict]) -> dict:
        """Evaluate a batch of ``{"output": …, "expected": …}`` dicts.

        When LLM judge is enabled, cases are scored mechanically first, then
        judge calls are batched (up to 5 cases per LLM call) for efficiency.
        Pre-scored items that lack an ``llm_judge`` key are also queued for
        batch judging, so callers can pass ``_skip_judge=True`` during
        per-case scoring and let this method handle judge calls in bulk.
        """
        if not results:
            return {"avg_total": 0.0, "count": 0, "individual_scores": []}

        judge_weight = self.spec.get("llm_judge_weight", 0)
        use_judge = bool(self.llm_judge_model and judge_weight > 0)

        all_scores: list[dict] = []
        needs_judge: list[int] = []

        for idx, r in enumerate(results):
            if "score" in r and isinstance(r["score"], dict):
                all_scores.append(r["score"])
                if use_judge and "llm_judge" not in r["score"] and r.get("input"):
                    needs_judge.append(idx)
            else:
                score = self.evaluate_output(
                    r["output"],
                    r["expected"],
                    input_data=r.get("input"),
                    tool_trace=r.get("tool_trace"),
                    _skip_judge=use_judge,
                )
                all_scores.append(score)
                if use_judge and r.get("input"):
                    needs_judge.append(idx)

        # Batch LLM judge calls
        if needs_judge:
            judge_fail_count = 0
            BATCH_SIZE = 5
            for batch_start in range(0, len(needs_judge), BATCH_SIZE):
                batch_indices = needs_judge[batch_start : batch_start + BATCH_SIZE]
                batch_items = [(idx, results[idx]) for idx in batch_indices]

                if len(batch_items) == 1:
                    idx, r = batch_items[0]
                    js = self._score_with_llm_judge(
                        r.get("input", {}), r.get("expected", {}), r.get("output", {})
                    )
                    if js == _JUDGE_FALLBACK_SCORE:
                        judge_fail_count += 1
                    all_scores[idx]["llm_judge"] = js * judge_weight
                    all_scores[idx]["total"] = max(0.0, sum(all_scores[idx].values()))
                else:
                    judge_scores = self._score_batch_with_llm_judge(batch_items)
                    for (idx, _), js in zip(batch_items, judge_scores):
                        if js == _JUDGE_FALLBACK_SCORE:
                            judge_fail_count += 1
                        all_scores[idx]["llm_judge"] = js * judge_weight
                        all_scores[idx]["total"] = max(
                            0.0, sum(all_scores[idx].values())
                        )

            if judge_fail_count > 0:
                fail_pct = judge_fail_count / len(needs_judge) * 100
                logger.warning(
                    "LLM judge failed on %d/%d cases (%.0f%%). "
                    "Fallback score %.1f used for failed cases.",
                    judge_fail_count,
                    len(needs_judge),
                    fail_pct,
                    _JUDGE_FALLBACK_SCORE,
                )

        keys = [k for k in all_scores[0] if k != "total"]
        avg: dict[str, float] = {}
        for k in keys:
            avg[f"avg_{k}"] = sum(s.get(k, 0) for s in all_scores) / len(all_scores)
        avg["avg_total"] = sum(s["total"] for s in all_scores) / len(all_scores)
        avg["count"] = len(all_scores)
        avg["individual_scores"] = all_scores  # type: ignore[assignment]
        return avg

    def get_dimension_labels(self) -> list[tuple[str, str]]:
        """Return ``(display_name, avg_key)`` pairs for reporting."""
        labels: list[tuple[str, str]] = [("Structure", "avg_structure")]
        for field_name in self.fields:
            display = field_name.replace("_", " ").title()
            labels.append((display, f"avg_{field_name}"))
        if self.spec.get("tool_usage_weight", 0) > 0:
            labels.append(("Tool Usage", "avg_tool_usage"))
        if self.spec.get("llm_judge_weight", 0) > 0:
            labels.append(("LLM Judge", "avg_llm_judge"))
        labels.append(("Type Correctness", "avg_type_correctness_penalty"))
        labels.append(("Consistency", "avg_consistency_penalty"))
        return labels

    def get_max_scores(self) -> dict[str, float]:
        """Return the maximum possible score for each dimension key."""
        maxes: dict[str, float] = {"avg_structure": float(self.structure_weight)}
        for field_name, config in self.fields.items():
            maxes[f"avg_{field_name}"] = float(config.get("weight", 0))
        tw = self.spec.get("tool_usage_weight", 0)
        if tw > 0:
            maxes["avg_tool_usage"] = float(tw)
        jw = self.spec.get("llm_judge_weight", 0)
        if jw > 0:
            maxes["avg_llm_judge"] = float(jw)
        maxes["avg_type_correctness_penalty"] = 0.0
        maxes["avg_consistency_penalty"] = 0.0
        return maxes

    # ------------------------------------------------------------------
    # Type correctness
    # ------------------------------------------------------------------

    def _check_type_correctness(self, output: dict) -> float:
        """Penalize outputs where field values have the wrong type.

        Returns a negative penalty (or 0 if all types are correct).
        Each type error deducts 2 points, capped at -10.
        """
        errors = 0
        for field_name, config in self.fields.items():
            actual = output.get(field_name)
            if actual is None:
                continue
            ftype = config["type"]
            if ftype == "number":
                try:
                    float(actual)
                except (ValueError, TypeError):
                    errors += 1
            elif ftype == "boolean":
                if not isinstance(actual, bool):
                    if not (isinstance(actual, (int, float)) and actual in (0, 1)):
                        errors += 1
            elif ftype == "enum":
                valid = {v.lower() for v in config.get("values", [])}
                if str(actual).lower().strip() not in valid:
                    errors += 1
            elif ftype == "text":
                if not isinstance(actual, str):
                    errors += 1
        return max(-10.0, -2.0 * errors)

    # ------------------------------------------------------------------
    # Cross-field consistency
    # ------------------------------------------------------------------

    def _check_cross_field_consistency(self, output: dict, expected: dict) -> float:
        """Penalize outputs where fields contradict each other.

        Returns a negative penalty (or 0 if consistent).
        Supports rule types: ``correlation`` (number vs enum) and
        ``ordering`` (number A <= number B).
        """
        rules = self.spec.get("consistency_rules", [])
        if not rules:
            rules = self._infer_consistency_rules(output)

        penalty = 0.0
        for rule in rules:
            field_a = rule.get("field_a", "")
            field_b = rule.get("field_b", "")
            val_a = output.get(field_a)
            val_b = output.get(field_b)
            if val_a is None or val_b is None:
                continue

            rule_type = rule.get("type", "correlation")
            if rule_type == "correlation":
                if self._is_contradictory(field_a, val_a, field_b, val_b):
                    penalty -= rule.get("penalty", 3.0)
            elif rule_type == "ordering":
                try:
                    num_a = float(val_a)
                    num_b = float(val_b)
                except (ValueError, TypeError):
                    continue
                op = rule.get("operator", "<=")
                violated = False
                if op == "<=" and num_a > num_b:
                    violated = True
                elif op == "<" and num_a >= num_b:
                    violated = True
                elif op == ">=" and num_a < num_b:
                    violated = True
                elif op == ">" and num_a <= num_b:
                    violated = True
                if violated:
                    penalty -= rule.get("penalty", 3.0)

        return penalty

    def _infer_consistency_rules(self, output: dict) -> list[dict]:
        """Auto-detect basic consistency expectations from field types."""
        rules: list[dict] = []
        number_fields = [n for n, c in self.fields.items() if c.get("type") == "number"]
        enum_fields = [n for n, c in self.fields.items() if c.get("type") == "enum"]

        for nf in number_fields:
            for ef in enum_fields:
                rules.append(
                    {
                        "field_a": nf,
                        "field_b": ef,
                        "type": "correlation",
                        "penalty": 3.0,
                    }
                )
        return rules

    def _is_contradictory(self, field_a: str, val_a, field_b: str, val_b) -> bool:
        """Heuristic: detect when a numeric field and an enum field diverge.

        Enum values are listed best-first (index 0 = best) per the setup
        prompt convention, so enum_normalized 0.0 = best, 1.0 = worst.
        A contradiction is when number is high but enum says worst, or
        number is low but enum says best.
        """
        config_a = self.fields.get(field_a, {})
        config_b = self.fields.get(field_b, {})

        if config_a.get("type") == "number" and config_b.get("type") == "enum":
            try:
                num_val = float(val_a)
            except (ValueError, TypeError):
                return False

            rng = config_a.get("range", [0, 100])
            if len(rng) == 2:
                normalized = (num_val - rng[0]) / max(rng[1] - rng[0], 1)
            else:
                normalized = num_val / 100.0

            values = config_b.get("values", [])
            if not values or len(values) < 2:
                return False

            str_val = str(val_b).lower().strip()
            if str_val not in [v.lower() for v in values]:
                return False

            val_position = next(
                (i for i, v in enumerate(values) if v.lower() == str_val),
                -1,
            )
            if val_position < 0:
                return False

            # 0 = best (first), 1 = worst (last)
            enum_normalized = val_position / max(len(values) - 1, 1)

            # High number but worst enum, or low number but best enum
            if normalized > 0.7 and enum_normalized > 0.7:
                return True
            if normalized < 0.3 and enum_normalized < 0.3:
                return True

        return False

    # ------------------------------------------------------------------
    # Tool usage scoring
    # ------------------------------------------------------------------

    def _score_tool_usage(self, tool_trace: list[dict] | None) -> float:
        """Score tool usage quality from trace data (0.0–1.0)."""
        if not tool_trace:
            return 0.0

        expected_tools = self.tool_config.get("expected_tools", [])
        param_constraints = self.tool_config.get("param_constraints", {})
        dependencies = self.tool_config.get("dependencies", [])

        if not expected_tools and not param_constraints and not dependencies:
            return 1.0

        sub_scores: list[float] = []

        # Completeness: were all expected tools called?
        if expected_tools:
            called_tools = {t.get("name", "") for t in tool_trace}
            completeness = len(called_tools & set(expected_tools)) / max(
                len(expected_tools), 1
            )
            sub_scores.append(completeness)

        # Argument quality: were enum-like args valid?
        if param_constraints:
            arg_score = self._score_tool_arguments(tool_trace, param_constraints)
            sub_scores.append(arg_score)

        # Dependency chaining: did outputs flow correctly between tools?
        if dependencies:
            chain_score = self._score_tool_chaining(tool_trace, dependencies)
            sub_scores.append(chain_score)

        return sum(sub_scores) / max(len(sub_scores), 1)

    @staticmethod
    def _score_tool_arguments(tool_trace: list[dict], constraints: dict) -> float:
        """Check if tool arguments match known valid values."""
        checks = 0
        valid = 0
        for call in tool_trace:
            name = call.get("name", "")
            args = call.get("args", {})
            if name not in constraints:
                continue
            for param, allowed in constraints[name].items():
                if param not in args:
                    continue
                checks += 1
                if str(args[param]).lower().strip() in [
                    str(v).lower() for v in allowed
                ]:
                    valid += 1
        return valid / max(checks, 1)

    @staticmethod
    def _score_tool_chaining(tool_trace: list[dict], dependencies: list[dict]) -> float:
        """Check if tool outputs were correctly propagated to dependent tools."""
        checks = 0
        correct = 0

        tool_results: dict[str, dict] = {}
        for call in tool_trace:
            name = call.get("name", "")
            tool_results[name] = call.get("result", {})

        for dep in dependencies:
            source_tool = dep.get("from_tool", "")
            source_field = dep.get("from_field", "")
            target_tool = dep.get("to_tool", "")
            target_param = dep.get("to_param", "")

            if source_tool not in tool_results:
                continue

            target_call = next(
                (c for c in tool_trace if c.get("name") == target_tool), None
            )
            if not target_call:
                continue

            checks += 1
            expected_val = tool_results[source_tool].get(source_field)
            actual_val = target_call.get("args", {}).get(target_param)

            if (
                expected_val is not None
                and str(expected_val).lower() == str(actual_val or "").lower()
            ):
                correct += 1

        return correct / max(checks, 1)

    # ------------------------------------------------------------------
    # LLM-as-Judge
    # ------------------------------------------------------------------

    def _score_with_llm_judge(
        self, input_data: dict, expected: dict, output: dict
    ) -> float:
        """Use a strong model to assess semantic quality. Returns 0.0–1.0.

        Retries up to ``_JUDGE_MAX_RETRIES`` times with exponential backoff
        on transient failures before falling back to the sentinel score.
        """
        criteria_parts = []
        for fname, config in self.fields.items():
            label = fname.replace("_", " ").title()
            imp = config.get("importance", "important")
            criteria_parts.append(f"- {label} ({imp}): {config.get('description', '')}")

        policy_section = ""
        if self.policy_judge_rubric:
            policy_section = (
                "\n## Agent Policy Rules\n" + self.policy_judge_rubric + "\n"
            )

        prompt = LLM_JUDGE_PROMPT.format(
            input_json=json.dumps(input_data, indent=2),
            expected_json=json.dumps(expected, indent=2),
            actual_json=json.dumps(output, indent=2),
            criteria="\n".join(criteria_parts),
            policy_rubric=policy_section,
        )

        last_exc: Exception | None = None
        for attempt in range(_JUDGE_MAX_RETRIES):
            try:
                resp = llm_completion(
                    self.llm_judge_model,
                    [{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=400,
                )
                content = resp.choices[0].message.content or ""
                score = self._parse_judge_scores(content)
                if score != _JUDGE_FALLBACK_SCORE:
                    return score
                logger.debug("Judge parse returned fallback on attempt %d", attempt + 1)
            except Exception as exc:
                last_exc = exc
                if attempt < _JUDGE_MAX_RETRIES - 1:
                    time.sleep(_JUDGE_RETRY_BACKOFF**attempt)

        logger.warning(
            "LLM judge failed after %d attempts%s — using fallback %.1f",
            _JUDGE_MAX_RETRIES,
            f": {last_exc}" if last_exc else "",
            _JUDGE_FALLBACK_SCORE,
        )
        return _JUDGE_FALLBACK_SCORE

    def _score_batch_with_llm_judge(
        self, batch_items: list[tuple[int, dict]]
    ) -> list[float]:
        """Score multiple cases in a single LLM judge call. Returns list of 0.0–1.0.

        Retries on failure with exponential backoff.
        """
        criteria_parts = []
        for fname, config in self.fields.items():
            label = fname.replace("_", " ").title()
            imp = config.get("importance", "important")
            criteria_parts.append(f"- {label} ({imp}): {config.get('description', '')}")

        policy_section = ""
        if self.policy_judge_rubric:
            policy_section = (
                "\n## Agent Policy Rules\n" + self.policy_judge_rubric + "\n"
            )

        case_blocks = []
        for case_num, (idx, r) in enumerate(batch_items):
            case_blocks.append(
                f"### Case {case_num + 1} (id: {case_num + 1})\n"
                f"**Input:** {json.dumps(r.get('input', {}), indent=2)}\n"
                f"**Expected:** {json.dumps(r.get('expected', {}), indent=2)}\n"
                f"**Actual:** {json.dumps(r.get('output', {}), indent=2)}"
            )

        prompt = LLM_JUDGE_BATCH_PROMPT.format(
            criteria="\n".join(criteria_parts),
            policy_rubric=policy_section,
            cases_block="\n\n".join(case_blocks),
        )

        fallback = [_JUDGE_FALLBACK_SCORE] * len(batch_items)
        last_exc: Exception | None = None
        for attempt in range(_JUDGE_MAX_RETRIES):
            try:
                max_tokens = 200 * len(batch_items)
                resp = llm_completion(
                    self.llm_judge_model,
                    [{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content or ""
                start = content.find("[")
                end = content.rfind("]") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(content[start:end])
                    if isinstance(parsed, list) and len(parsed) >= len(batch_items):
                        return [
                            self._compute_judge_score(p)
                            for p in parsed[: len(batch_items)]
                        ]
            except Exception as exc:
                last_exc = exc
                if attempt < _JUDGE_MAX_RETRIES - 1:
                    time.sleep(_JUDGE_RETRY_BACKOFF**attempt)

        if last_exc:
            logger.warning(
                "Batch judge failed after %d attempts: %s",
                _JUDGE_MAX_RETRIES,
                last_exc,
            )
        return fallback

    def _parse_judge_scores(self, content: str) -> float:
        """Parse a single judge response JSON into a 0.0–1.0 score."""
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(content[start:end])
            return self._compute_judge_score(parsed)
        return _JUDGE_FALLBACK_SCORE

    def _compute_judge_score(self, parsed: dict) -> float:
        """Compute weighted judge score from parsed dimension scores."""
        sc = parsed.get("semantic_correctness", 5)
        ic = parsed.get("internal_consistency", 5)
        rq = parsed.get("reasoning_quality", 5)
        pc = parsed.get("policy_compliance", 10)
        if self.policy_judge_rubric:
            return (sc * 0.35 + ic * 0.2 + rq * 0.15 + pc * 0.3) / 10.0
        return (sc * 0.5 + ic * 0.3 + rq * 0.2) / 10.0

    # ------------------------------------------------------------------
    # Private scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_enum(actual, expected, config: dict) -> float:
        weight = config.get("weight", 0)
        actual_str = str(actual or "").lower().strip()
        expected_str = str(expected or "").lower().strip()

        if actual_str == expected_str:
            return float(weight)

        valid = {v.lower() for v in config.get("values", [])}
        if config.get("partial_credit") and actual_str in valid:
            return float(config.get("partial_score", max(1, round(weight * 0.2))))

        return 0.0

    @staticmethod
    def _score_number(actual, expected, config: dict) -> float:
        weight = config.get("weight", 0)
        try:
            actual_val = float(actual or 0)
            expected_val = float(expected or 0)
        except (ValueError, TypeError):
            return 0.0

        diff = abs(actual_val - expected_val)

        bands = config.get("tolerance_bands", [])
        if bands:
            for band in sorted(bands, key=lambda b: b["within"]):
                if diff <= band["within"]:
                    return weight * band["score_pct"]
            return 0.0

        tolerance = config.get("tolerance", 10)
        if diff <= tolerance:
            return float(weight)
        if diff <= tolerance * 2:
            return weight * 0.5
        return 0.0

    def _score_text(
        self,
        actual,
        expected,
        config: dict,
        field_name: str = "",
        input_data: dict | None = None,
    ) -> float:
        weight = config.get("weight", 0)
        mode = config.get("eval_mode", "non_empty")

        if mode == "skip":
            return 0.0
        if mode == "non_empty":
            return float(weight) if actual and str(actual).strip() else 0.0
        if mode == "similarity":
            return self._text_similarity(actual, expected) * weight
        if mode == "keyword_coverage":
            return self._text_keyword_coverage(actual, expected) * weight
        if mode == "llm_judge":
            return (
                self._text_field_judge(actual, expected, config, field_name, input_data)
                * weight
            )
        return 0.0

    @staticmethod
    def _text_similarity(actual, expected) -> float:
        """Token-based similarity between actual and expected text (0.0–1.0).

        Combines Jaccard overlap with expected-token coverage and a length
        ratio penalty. Fully deterministic — no API calls.
        """
        actual_str = str(actual or "").strip()
        expected_str = str(expected or "").strip()

        if not actual_str and not expected_str:
            return 1.0
        if not actual_str:
            return 0.0
        if not expected_str:
            return 1.0 if actual_str else 0.0

        actual_tokens = set(re.findall(r"\b\w+\b", actual_str.lower()))
        expected_tokens = set(re.findall(r"\b\w+\b", expected_str.lower()))

        if not expected_tokens:
            return 1.0 if actual_tokens else 0.5

        intersection = actual_tokens & expected_tokens
        union = actual_tokens | expected_tokens
        jaccard = len(intersection) / max(len(union), 1)
        coverage = len(intersection) / len(expected_tokens)

        len_ratio = len(actual_str) / max(len(expected_str), 1)
        length_penalty = 1.0
        if len_ratio < 0.3:
            length_penalty = len_ratio / 0.3
        elif len_ratio > 3.0:
            length_penalty = max(0.5, 1.0 - (len_ratio - 3.0) * 0.1)

        return (coverage * 0.6 + jaccard * 0.4) * length_penalty

    @staticmethod
    def _text_keyword_coverage(actual, expected) -> float:
        """Fraction of significant keywords from expected that appear in actual (0.0–1.0)."""
        expected_str = str(expected or "").strip()
        actual_str = str(actual or "").strip()

        if not expected_str:
            return 1.0 if actual_str else 0.0
        if not actual_str:
            return 0.0

        expected_tokens = set(re.findall(r"\b\w+\b", expected_str.lower()))
        keywords = expected_tokens - _STOPWORDS
        if not keywords:
            return 1.0 if actual_str else 0.0

        actual_lower = actual_str.lower()
        found = sum(1 for kw in keywords if kw in actual_lower)
        return found / len(keywords)

    def _text_field_judge(
        self,
        actual,
        expected,
        config: dict,
        field_name: str,
        input_data: dict | None,
    ) -> float:
        """Per-field LLM judge for text quality (0.0–1.0)."""
        if not self.llm_judge_model:
            return self._text_similarity(actual, expected)

        actual_str = str(actual or "").strip()
        expected_str = str(expected or "").strip()

        if not actual_str:
            return 0.0
        if not expected_str:
            return 1.0 if actual_str else 0.0

        prompt = LLM_TEXT_FIELD_JUDGE_PROMPT.format(
            field_name=field_name.replace("_", " ").title(),
            field_description=config.get("description", ""),
            expected_text=expected_str,
            actual_text=actual_str,
            input_json=json.dumps(input_data or {}, indent=2),
        )

        for attempt in range(_JUDGE_MAX_RETRIES):
            try:
                resp = llm_completion(
                    self.llm_judge_model,
                    [{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=150,
                )
                content = resp.choices[0].message.content or ""
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(content[start:end])
                    raw = parsed.get("score", 5)
                    return max(0.0, min(1.0, float(raw) / 10.0))
            except Exception:
                if attempt < _JUDGE_MAX_RETRIES - 1:
                    time.sleep(_JUDGE_RETRY_BACKOFF**attempt)

        return self._text_similarity(actual, expected)


def has_entrypoint(code: str, fn_name: str) -> bool:
    """Return True if *code* exposes a top-level function named *fn_name*.

    OverClaw loads agent files as Python modules and calls the registered
    entry function for every test case.  Uses AST parsing for accuracy,
    with a string-based fallback for non-parseable code.
    """
    from overclaw.utils.code import has_entrypoint_ast

    if has_entrypoint_ast(code, fn_name):
        return True
    return f"def {fn_name}(" in code or f"def {fn_name} (" in code


def has_run_entrypoint(code: str) -> bool:
    """Backward-compatible alias for ``has_entrypoint(code, 'run')``."""
    return has_entrypoint(code, "run")


def load_evaluator(
    spec_path: str,
    llm_judge_model: str | None = None,
    policy_judge_rubric: str = "",
) -> SpecEvaluator:
    """Load a SpecEvaluator from an eval spec JSON file."""
    if not Path(spec_path).exists():
        raise FileNotFoundError(
            f"Evaluation spec not found at {spec_path}.\n"
            "Run OverClaw setup first (`overclaw setup`) to generate evaluation criteria."
        )
    return SpecEvaluator(
        spec_path,
        llm_judge_model=llm_judge_model,
        policy_judge_rubric=policy_judge_rubric,
    )
