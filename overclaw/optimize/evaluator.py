"""Spec-driven evaluation scoring for agent outputs.

Reads an evaluation spec (e.g. ``agents/<name>/eval_spec/eval_spec.json``) produced by ``setup.py``
and scores agent outputs dynamically based on the configured field types,
weights, and tolerances.

Supports three scoring layers:
1. **Mechanical** — deterministic field-level matching (enum, number, text, boolean)
2. **LLM-as-Judge** — semantic quality assessment via a strong model
3. **Tool Usage** — checks tool call completeness, argument quality, and data chaining
"""

import json
from pathlib import Path

from overclaw.utils.llm import llm_completion
from overclaw.prompts.evaluator import LLM_JUDGE_BATCH_PROMPT, LLM_JUDGE_PROMPT


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

        # --- Mechanical scoring ---
        expected_fields = list(self.fields.keys())
        present = sum(
            1 for f in expected_fields if f in output and output[f] not in (None, "", 0)
        )
        scores["structure"] = (
            present / max(len(expected_fields), 1)
        ) * self.structure_weight

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
                scores[field_name] = self._score_text(output.get(field_name), config)
            elif ftype == "boolean":
                weight = config.get("weight", 0)
                match = output.get(field_name) == expected.get(field_name)
                scores[field_name] = float(weight) if match else 0.0

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
            BATCH_SIZE = 5
            for batch_start in range(0, len(needs_judge), BATCH_SIZE):
                batch_indices = needs_judge[batch_start : batch_start + BATCH_SIZE]
                batch_items = [(idx, results[idx]) for idx in batch_indices]

                if len(batch_items) == 1:
                    idx, r = batch_items[0]
                    js = self._score_with_llm_judge(
                        r.get("input", {}), r.get("expected", {}), r.get("output", {})
                    )
                    all_scores[idx]["llm_judge"] = js * judge_weight
                    all_scores[idx]["total"] = max(0.0, sum(all_scores[idx].values()))
                else:
                    judge_scores = self._score_batch_with_llm_judge(batch_items)
                    for (idx, _), js in zip(batch_items, judge_scores):
                        all_scores[idx]["llm_judge"] = js * judge_weight
                        all_scores[idx]["total"] = max(
                            0.0, sum(all_scores[idx].values())
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
        if any(
            s.get("consistency_penalty", 0) != 0
            for s in (self.spec.get("_last_scores", []) or [{}])
        ):
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
        maxes["avg_consistency_penalty"] = 0.0
        return maxes

    # ------------------------------------------------------------------
    # Cross-field consistency
    # ------------------------------------------------------------------

    def _check_cross_field_consistency(self, output: dict, expected: dict) -> float:
        """Penalize outputs where fields contradict each other.

        Returns a negative penalty (or 0 if consistent).
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
        """Use a strong model to assess semantic quality. Returns 0.0–1.0."""
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

        try:
            resp = llm_completion(
                self.llm_judge_model,
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=400,
            )
            content = resp.choices[0].message.content or ""
            return self._parse_judge_scores(content)
        except Exception:
            pass

        return 0.5

    def _score_batch_with_llm_judge(
        self, batch_items: list[tuple[int, dict]]
    ) -> list[float]:
        """Score multiple cases in a single LLM judge call. Returns list of 0.0–1.0."""
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

        fallback = [0.5] * len(batch_items)
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
                        self._compute_judge_score(p) for p in parsed[: len(batch_items)]
                    ]
        except Exception:
            pass

        return fallback

    def _parse_judge_scores(self, content: str) -> float:
        """Parse a single judge response JSON into a 0.0–1.0 score."""
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(content[start:end])
            return self._compute_judge_score(parsed)
        return 0.5

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

    @staticmethod
    def _score_text(actual, config: dict) -> float:
        weight = config.get("weight", 0)
        mode = config.get("eval_mode", "non_empty")

        if mode == "skip":
            return 0.0
        if mode == "non_empty":
            return float(weight) if actual and str(actual).strip() else 0.0
        return 0.0


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
