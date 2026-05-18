"""Span attribute / tag keys used across Overmind.

Every tag attached to an Overmind trace span (via ``overmind.set_tag`` or
``start_span(..., attributes=...)``) is defined here so the schema of
attributes we emit is auditable from a single file.  Add new tags here
first, then import the constant at the call site — never inline a raw
``"overmind.*"`` / ``"gen_ai.*"`` / ``"tool.*"`` string elsewhere.

Naming rule
-----------
Keys under ``overmind.*`` use dotted segments
(``overmind.agent.id``, ``overmind.error.type``, …); never a single
underscore-blob like ``overmind.agent_id``.

Deprecations
------------
Some constants exist as **legacy aliases** for backward compatibility
with older OTLP payloads.  Each deprecation is documented inline next
to the constant; new call sites must prefer the canonical key.  Current
legacy aliases:

* :data:`ERROR` → use :data:`ERROR_SUMMARY`.
* :data:`OPTIMIZE_REPORT_BEST_SCORE` → use
  :data:`OPTIMIZE_FINAL_BEST_SCORE`.

Keep ``overbae/api/overmind_attrs.py`` in lockstep with this file when
adding or renaming keys.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Top-level command tag (set on every CLI entry-point span).
# ---------------------------------------------------------------------------
COMMAND = "overmind.command"

# ---------------------------------------------------------------------------
# Generic lifecycle tags — explicit status / progress markers we set on any
# span so the platform can surface "what's happening?" without parsing the
# OTel ``status_code`` directly.
# ---------------------------------------------------------------------------
STATUS = "overmind.status"  # "running" | "success" | "failed" | "cancelled"
ERROR_TYPE = "overmind.error.type"  # exception class name, when STATUS == "failed"
ERROR_MESSAGE = "overmind.error.message"  # short, scrubbed message
PROGRESS_PHASE = "overmind.progress.phase"  # human label for current pipeline step
PROGRESS_CURRENT = "overmind.progress.current"  # current step index (1-based)
PROGRESS_TOTAL = "overmind.progress.total"  # total number of steps
DURATION_SECONDS = "overmind.duration.seconds"  # walltime of the wrapped span

# ---------------------------------------------------------------------------
# Top-level resource / context tags (link spans to server-side entities).
# ---------------------------------------------------------------------------
AGENT_ID = "overmind.agent.id"
JOB_ID = "overmind.job.id"

# ---------------------------------------------------------------------------
# Agent registry / `overmind agent ...` commands
# (overmind/commands/agent_cmd.py)
# ---------------------------------------------------------------------------
AGENT_NAME = "overmind.agent.name"
# Tracelo­op-compatible workflow label attached to every downstream span by
# the on-start processor.  Stored in the OTel context so child spans pick
# it up automatically.
WORKFLOW_NAME = "overmind.workflow.name"
# Stable per-conversation identifier used by chat agents to group multiple
# traces under the same user-visible session.
CONVERSATION_ID = "overmind.conversation.id"
AGENT_ENTRYPOINT = "overmind.agent.entrypoint"
AGENT_NEW_ENTRYPOINT = "overmind.agent.new_entrypoint"
AGENT_OLD_ENTRYPOINT = "overmind.agent.old_entrypoint"
AGENT_FILE_PATH = "overmind.agent.file_path"
AGENT_FUNCTION_NAME = "overmind.agent.function_name"
AGENT_REGISTERED_COUNT = "overmind.agent.registered_count"
AGENT_REMOVED = "overmind.agent.removed"
AGENT_FILE_EXISTS = "overmind.agent.file_exists"
AGENT_SETUP_SPEC_READY = "overmind.agent.setup_spec_ready"
AGENT_EXPERIMENT_FILE_COUNT = "overmind.agent.experiment_file_count"

# ---------------------------------------------------------------------------
# `overmind init` (overmind/commands/init_cmd.py)
# ---------------------------------------------------------------------------
INIT_ENV_PATH = "overmind.init.env_path"
INIT_HAS_OPENAI_KEY = "overmind.init.has_openai_key"
INIT_HAS_ANTHROPIC_KEY = "overmind.init.has_anthropic_key"
INIT_HAS_OVERMIND_TOKEN = "overmind.init.has_overmind_token"
INIT_ANALYZER_MODEL = "overmind.init.analyzer_model"
INIT_HAS_SYNTHETIC_DATAGEN_MODEL = "overmind.init.has_synthetic_datagen_model"

# ---------------------------------------------------------------------------
# `overmind setup` (overmind/commands/setup_cmd.py +
# overmind/setup/{questionnaire,policy_generator}.py)
# ---------------------------------------------------------------------------
SETUP_FAST = "overmind.setup.fast"
SETUP_MODEL = "overmind.setup.model"
SETUP_HAS_POLICY = "overmind.setup.has_policy"
SETUP_HAS_SEED_DATA = "overmind.setup.has_seed_data"
SETUP_POLICY_PATH = "overmind.setup.policy_path"
SETUP_DATA_PATH = "overmind.setup.data_path"
SETUP_STORAGE_BACKEND = "overmind.setup.storage_backend"
SETUP_AGENT_PATH = "overmind.setup.agent_path"
SETUP_ENTRYPOINT_FN = "overmind.setup.entrypoint_fn"
SETUP_ANALYZER_MODEL = "overmind.setup.analyzer_model"
SETUP_PHASE = "overmind.setup.phase"
SETUP_EVAL_SPEC_FIELD_COUNT = "overmind.setup.eval_spec_field_count"
SETUP_EVAL_SPEC_HAS_TOOLS = "overmind.setup.eval_spec_has_tools"
SETUP_EVAL_SPEC_HAS_JUDGE = "overmind.setup.eval_spec_has_judge"
SETUP_EVAL_SPEC_HAS_POLICY = "overmind.setup.eval_spec_has_policy"
SETUP_EVAL_SPEC_STRUCTURE_WEIGHT = "overmind.setup.eval_spec_structure_weight"
SETUP_EVAL_SPEC_TOOL_COUNT = "overmind.setup.eval_spec_tool_count"
SETUP_EVAL_SPEC_CONSISTENCY_RULE_COUNT = "overmind.setup.eval_spec_consistency_rule_count"
SETUP_CRITERIA_SOURCE = "overmind.setup.criteria_source"
SETUP_POLICY_SOURCE = "overmind.setup.policy_source"
SETUP_AGENT_POLICY_MARKDOWN = "overmind.agent.policy.markdown"
SETUP_AGENT_POLICY_DATA = "overmind.agent.policy.data"
# Agent description and full eval spec snapshot (text / JSON string).
SETUP_AGENT_DESCRIPTION = "overmind.setup.agent_description"
SETUP_EVAL_SPEC = "overmind.setup.eval_spec"
# Post-analysis artefacts from the setup pipeline.
SETUP_PROPOSED_CRITERIA = "overmind.setup.proposed_criteria"
SETUP_TOOL_ANALYSIS = "overmind.setup.tool_analysis"
SETUP_INPUT_SCHEMA = "overmind.setup.input_schema"
SETUP_OUTPUT_SCHEMA = "overmind.setup.output_schema"
SETUP_CONSISTENCY_RULES = "overmind.setup.consistency_rules"
SETUP_TOOLS_SUMMARY = "overmind.setup.tools_summary"
SETUP_DECISION_LOGIC = "overmind.setup.decision_logic"
SETUP_SCOPE = "overmind.setup.scope"
SETUP_OPTIMIZABLE_ELEMENTS = "overmind.setup.optimizable_elements"

# ---------------------------------------------------------------------------
# `overmind optimize` (overmind/commands/optimize_cmd.py)
# ---------------------------------------------------------------------------
OPTIMIZE_AGENT_NAME = "overmind.optimize.agent_name"
OPTIMIZE_AGENT_PATH = "overmind.optimize.agent_path"
OPTIMIZE_FAST = "overmind.optimize.fast"
OPTIMIZE_ENTRYPOINT_FN = "overmind.optimize.entrypoint_fn"
OPTIMIZE_STORAGE_BACKEND = "overmind.optimize.storage_backend"
OPTIMIZE_ANALYZER_MODEL = "overmind.optimize.analyzer_model"
OPTIMIZE_LLM_JUDGE_MODEL = "overmind.optimize.llm_judge_model"
OPTIMIZE_ITERATIONS = "overmind.optimize.iterations"
OPTIMIZE_CANDIDATES_PER_ITERATION = "overmind.optimize.candidates_per_iteration"
OPTIMIZE_PARALLEL = "overmind.optimize.parallel"
OPTIMIZE_MAX_WORKERS = "overmind.optimize.max_workers"
OPTIMIZE_RUNS_PER_EVAL = "overmind.optimize.runs_per_eval"
OPTIMIZE_REGRESSION_THRESHOLD = "overmind.optimize.regression_threshold"
OPTIMIZE_HOLDOUT_RATIO = "overmind.optimize.holdout_ratio"
OPTIMIZE_HOLDOUT_ENFORCEMENT = "overmind.optimize.holdout_enforcement"
OPTIMIZE_EARLY_STOPPING_PATIENCE = "overmind.optimize.early_stopping_patience"
OPTIMIZE_CROSS_RUN_PERSISTENCE = "overmind.optimize.cross_run_persistence"
OPTIMIZE_FAILURE_CLUSTERING = "overmind.optimize.failure_clustering"
OPTIMIZE_ADAPTIVE_FOCUS = "overmind.optimize.adaptive_focus"
OPTIMIZE_MODEL_BACKTESTING = "overmind.optimize.model_backtesting"
OPTIMIZE_BACKTEST_MODELS = "overmind.optimize.backtest_models"
OPTIMIZE_EVAL_SPEC_PATH = "overmind.optimize.eval_spec_path"
OPTIMIZE_DATA_PATH = "overmind.optimize.data_path"
# Full eval_spec.json (JSON string) for OTLP ingest to refresh Agent eval fields during optimize.
OPTIMIZE_EVAL_SPEC = "overmind.optimize.eval_spec"

# Optimizer pipeline (overmind/optimize/optimizer.py) — runtime tags emitted
# from the optimizer's spans during a run.
OPTIMIZE_PHASE = "overmind.optimize.phase"
OPTIMIZE_DATASET_TOTAL = "overmind.optimize.dataset_total"
OPTIMIZE_DATASET_TRAIN = "overmind.optimize.dataset_train"
OPTIMIZE_DATASET_HOLDOUT = "overmind.optimize.dataset_holdout"
OPTIMIZE_BASELINE_SCORE = "overmind.optimize.baseline_score"
OPTIMIZE_ITERATION = "overmind.optimize.iteration"
OPTIMIZE_TOTAL_ITERATIONS = "overmind.optimize.total_iterations"
OPTIMIZE_BEST_SCORE_BEFORE = "overmind.optimize.best_score_before"
OPTIMIZE_STALL_COUNT = "overmind.optimize.stall_count"
OPTIMIZE_TEMPERATURE = "overmind.optimize.temperature"
OPTIMIZE_N_CANDIDATES_GENERATED = "overmind.optimize.n_candidates_generated"
OPTIMIZE_N_CANDIDATES_VALID = "overmind.optimize.n_candidates_valid"
OPTIMIZE_CANDIDATE_INDEX = "overmind.optimize.candidate_index"
OPTIMIZE_CANDIDATE_METHOD = "overmind.optimize.candidate_method"
OPTIMIZE_CANDIDATE_SCORE = "overmind.optimize.candidate_score"
OPTIMIZE_CANDIDATE_ADJUSTED_SCORE = "overmind.optimize.candidate_adjusted_score"
OPTIMIZE_COMPLEXITY_PENALTY = "overmind.optimize.complexity_penalty"
OPTIMIZE_ITERATION_DECISION = "overmind.optimize.iteration_decision"
OPTIMIZE_ITERATION_SCORE = "overmind.optimize.iteration_score"
OPTIMIZE_ITERATION_IMPROVEMENT = "overmind.optimize.iteration_improvement"
OPTIMIZE_ITERATION_REASON = "overmind.optimize.iteration_reason"
# Skill-driven flow identifiers — each ``overmind optimize-step`` call
# stamps its phase here so OTLP can attribute every span back to its
# step (init / baseline / diagnose / evaluate / accept / report).
OPTIMIZE_STEP = "overmind.optimize.step"
# Terminal state of the optimize run.  ``running`` while in flight,
# flipped to ``completed`` / ``failed`` / ``cancelled`` by the step
# that owns end-of-run.  OTLP uses this (alongside the legacy
# ``is_optimize_root`` heuristic) to drive ``Job.status``.
OPTIMIZE_RUN_STATUS = "overmind.optimize.run_status"
# Per-dimension scores for the iteration (e.g. ``{"correctness": 87.0,
# "tool_use": 64.2}``).  Serialised as JSON because OTel attributes
# cannot carry nested dicts directly.
OPTIMIZE_ITERATION_DIMENSION_SCORES = "overmind.optimize.iteration_dimension_scores"
# Full agent source code produced by the iteration's best candidate.
# Stamped on the ``optimizer.iteration`` span so OTLP can project it
# onto ``JobIteration.agent_code``.
OPTIMIZE_ITERATION_AGENT_CODE = "overmind.optimize.iteration_agent_code"
# JSON-encoded list of human-readable change summaries (the
# ``suggestions`` field of the best candidate).  OTLP folds these into
# ``JobIteration.description``.
OPTIMIZE_ITERATION_SUGGESTIONS = "overmind.optimize.iteration_suggestions"
OPTIMIZE_ACCEPTED = "overmind.optimize.accepted"
OPTIMIZE_FINAL_BEST_SCORE = "overmind.optimize.final_best_score"
# Final report.md text produced by ``Optimizer._generate_report``.
# Stamped on the optimize run span so OTLP can persist it on the
# corresponding ``Job.report_markdown`` row.
OPTIMIZE_REPORT_MARKDOWN = "overmind.optimize.report_markdown"
# Final best agent code text (``Job.best_agent_code``).
OPTIMIZE_BEST_AGENT_CODE = "overmind.optimize.best_agent_code"
OPTIMIZE_HOLDOUT_SCORE = "overmind.optimize.holdout_score"
OPTIMIZE_HOLDOUT_BASELINE_SCORE = "overmind.optimize.holdout_baseline_score"
OPTIMIZE_HOLDOUT_IMPROVEMENT = "overmind.optimize.holdout_improvement"
OPTIMIZE_BLENDED_IMPROVEMENT = "overmind.optimize.blended_improvement"
OPTIMIZE_HOLDOUT_REVERTED = "overmind.optimize.holdout_reverted"
# Train-vs-holdout score gap (positive = train score is higher than
# holdout, i.e. overfitting).  Projected onto
# ``Job.result["holdout"]["overfit_gap"]``.
OPTIMIZE_OVERFIT_GAP = "overmind.optimize.overfit_gap"
# End-of-run summary counters, projected onto ``Job.result["summary"]``.
OPTIMIZE_TOTAL_ACCEPTED = "overmind.optimize.total_accepted"
OPTIMIZE_TOTAL_REJECTED = "overmind.optimize.total_rejected"
# Final headline improvement (``best_score - baseline_score``) — drives
# ``Job.improvement`` in the OTLP ingest.
OPTIMIZE_REPORT_IMPROVEMENT = "overmind.optimize.report_improvement"
# DEPRECATED: redundant with ``OPTIMIZE_FINAL_BEST_SCORE``.  Kept for
# backward compatibility with the legacy ``ApiReporter.on_complete``
# payload; new call sites should emit only ``OPTIMIZE_FINAL_BEST_SCORE``.
# Plan p6-1 retires this once OTLP ingest stops reading it.
OPTIMIZE_REPORT_BEST_SCORE = "overmind.optimize.report_best_score"
OPTIMIZE_BACKTEST_MODEL = "overmind.optimize.backtest_model"
OPTIMIZE_BACKTEST_SCORE = "overmind.optimize.backtest_score"

# ---------------------------------------------------------------------------
# Evaluator (overmind/optimize/evaluator.py)
# ---------------------------------------------------------------------------
EVAL_BATCH_SIZE = "overmind.eval.batch_size"
EVAL_AVG_TOTAL = "overmind.eval.avg_total"
EVAL_USED_LLM_JUDGE = "overmind.eval.used_llm_judge"
EVAL_JUDGE_NEEDS_COUNT = "overmind.eval.judge_needs_count"
EVAL_JUDGE_FAIL_COUNT = "overmind.eval.judge_fail_count"
EVAL_JUDGE_FAIL_PCT = "overmind.eval.judge_fail_pct"
EVAL_AVG_CONFIDENCE = "overmind.eval.avg_confidence"
EVAL_SOURCE_SUMMARY = "overmind.eval.source_summary"
# Per-run agent execution telemetry (run_agent_on_dataset / parallel /
# sequential subprocess spans).  These let the platform UI distinguish
# baseline / candidate / holdout runs and surface backend health.
RUN_AGENT_RUN_NAME = "overmind.run.run_name"
RUN_AGENT_AGENT_PATH = "overmind.run.agent_path"
RUN_AGENT_CASES_TOTAL = "overmind.run.cases_total"
RUN_AGENT_CASES_SUCCEEDED = "overmind.run.cases_succeeded"
RUN_AGENT_CASES_FAILED = "overmind.run.cases_failed"
RUN_AGENT_CASES_WITH_PROVENANCE = "overmind.run.cases_with_provenance"
RUN_AGENT_PARALLEL = "overmind.run.parallel"
RUN_AGENT_MAX_WORKERS = "overmind.run.max_workers"
RUN_AGENT_BACKENDS = "overmind.run.backends"
RUN_AGENT_BACKEND_USED = "overmind.run.backend_used"
# Aggregated batch evaluation results — stamped on the
# ``optimizer.run_agent_on_dataset`` and ``optimizer.build_eval_results``
# spans once every case has been scored.
RUN_AGENT_AVG_SCORE = "overmind.run.avg_score"
RUN_AGENT_DIMENSION_SCORES = "overmind.run.dimension_scores"
RUN_AGENT_PASS_RATE = "overmind.run.pass_rate"
RUN_AGENT_DURATION_SECONDS = "overmind.run.duration_seconds"
RUN_AGENT_PER_CASE_SCORES = "overmind.run.per_case_scores"

# Per-case telemetry (``optimizer.run_case_with_plan``) — one span per
# datapoint, lets the UI drill down from a run into the individual
# subprocess invocations and the backend fallback decisions.
RUN_CASE_RUN_NAME = "overmind.run.case.run_name"
RUN_CASE_INDEX = "overmind.run.case.index"
RUN_CASE_INPUT_KEYS = "overmind.run.case.input_keys"
RUN_CASE_INPUT_CHARS = "overmind.run.case.input_chars"
RUN_CASE_BACKEND_ATTEMPTS = "overmind.run.case.backend_attempts"
RUN_CASE_BACKENDS_TRIED = "overmind.run.case.backends_tried"
RUN_CASE_BACKEND_USED = "overmind.run.case.backend_used"
RUN_CASE_SUCCESS = "overmind.run.case.success"
RUN_CASE_RETURNCODE = "overmind.run.case.returncode"
RUN_CASE_ERROR = "overmind.run.case.error"
RUN_CASE_OUTPUT_CHARS = "overmind.run.case.output_chars"
RUN_CASE_DURATION_SECONDS = "overmind.run.case.duration_seconds"
RUN_CASE_PROVENANCE_COUNT = "overmind.run.case.provenance_count"
RUN_CASE_CONFIDENCE = "overmind.run.case.confidence"

# ``optimizer.evaluate_worktree`` — invoked from the skill-driven
# `overmind optimize-step evaluate` flow, scores a worktree's candidate
# agent against the training set.  These tags let the UI surface
# per-candidate / per-iteration scores even when the optimizer is being
# driven from a host coding agent rather than the legacy loop.
OPTIMIZE_WORKTREE_RUN_NAME = "overmind.optimize.worktree.run_name"
OPTIMIZE_WORKTREE_ENTRY_PATH = "overmind.optimize.worktree.entry_path"
OPTIMIZE_WORKTREE_CASES_TOTAL = "overmind.optimize.worktree.cases_total"
OPTIMIZE_WORKTREE_AVG_SCORE = "overmind.optimize.worktree.avg_score"
OPTIMIZE_WORKTREE_DIMENSION_SCORES = "overmind.optimize.worktree.dimension_scores"
OPTIMIZE_WORKTREE_PASS_RATE = "overmind.optimize.worktree.pass_rate"
OPTIMIZE_WORKTREE_DURATION_SECONDS = "overmind.optimize.worktree.duration_seconds"

# ---------------------------------------------------------------------------
# Cross-run optimization state (overmind/optimize/run_state.py)
# ---------------------------------------------------------------------------
RUN_STATE_TOTAL_RUNS = "overmind.run_state.total_runs"
RUN_STATE_REGRESSION_CASES = "overmind.run_state.regression_cases"
RUN_STATE_FAILURE_CLUSTERS = "overmind.run_state.failure_clusters"
RUN_STATE_LATEST_BASELINE = "overmind.run_state.latest_baseline"
RUN_STATE_LATEST_FINAL = "overmind.run_state.latest_final"
RUN_STATE_LATEST_ACCEPTED = "overmind.run_state.latest_accepted"
RUN_STATE_LATEST_REJECTED = "overmind.run_state.latest_rejected"

# ---------------------------------------------------------------------------
# Candidate generation (overmind/optimize/analyzer.py)
# ---------------------------------------------------------------------------
CANDIDATES_REQUESTED = "overmind.candidates.requested"
CANDIDATES_PRODUCED = "overmind.candidates.produced"
CANDIDATES_METHODS = "overmind.candidates.methods"
CANDIDATES_HAS_DIAGNOSIS = "overmind.candidates.has_diagnosis"
CANDIDATES_USE_BUNDLE = "overmind.candidates.use_bundle"
CANDIDATES_HAS_ROOT_CAUSE = "overmind.candidates.has_root_cause"
CANDIDATES_FALLBACK = "overmind.candidates.fallback"

# ---------------------------------------------------------------------------
# Synthetic data generation (overmind/optimize/data.py)
# ---------------------------------------------------------------------------
DATAGEN_MODE = "overmind.datagen.mode"
DATAGEN_MODEL = "overmind.datagen.model"
DATAGEN_REQUESTED_SAMPLES = "overmind.datagen.requested_samples"
DATAGEN_GENERATED_COUNT = "overmind.datagen.generated_count"
DATAGEN_PERSONA_COUNT = "overmind.datagen.persona_count"
DATAGEN_PERSONA_SOURCE = "overmind.datagen.persona_source"
DATAGEN_PERSONA_IDX = "overmind.datagen.persona_idx"
DATAGEN_PERSONA_NAME = "overmind.datagen.persona_name"
DATAGEN_PERSONA_INTENT = "overmind.datagen.persona_intent"
DATAGEN_PERSONA_SHARDS = "overmind.datagen.persona_shards"
DATAGEN_RETRY_ATTEMPT = "overmind.datagen.retry_attempt"
DATAGEN_RETRY_MAX_ATTEMPTS = "overmind.datagen.retry_max_attempts"
DATAGEN_ANTI_EXAMPLES = "overmind.datagen.anti_examples"
DATAGEN_ROUND = "overmind.datagen.round"
DATAGEN_ROUNDS = "overmind.datagen.rounds"
DATAGEN_TARGET = "overmind.datagen.target"
DATAGEN_HAVE_BEFORE = "overmind.datagen.have_before"
DATAGEN_ELAPSED_SECONDS = "overmind.datagen.elapsed_seconds"
DATAGEN_EXISTING_CASES = "overmind.datagen.existing_cases"
DATAGEN_COVERAGE_GAP_COUNT = "overmind.datagen.coverage_gap_count"

# ---------------------------------------------------------------------------
# Seed data validation / coverage (overmind/optimize/data_analyzer.py)
# ---------------------------------------------------------------------------
SEED_VALIDATION_TOTAL_CASES = "overmind.seed_validation.total_cases"
SEED_VALIDATION_VALID_COUNT = "overmind.seed_validation.valid_count"
SEED_VALIDATION_INVALID_COUNT = "overmind.seed_validation.invalid_count"
SEED_COVERAGE_QUALITY_SCORE = "overmind.seed_coverage.quality_score"
SEED_COVERAGE_CASE_COUNT = "overmind.seed_coverage.case_count"
SEED_COVERAGE_GAP_COUNT = "overmind.seed_coverage.gap_count"
SEED_COVERAGE_UNCOVERED_RULE_COUNT = "overmind.seed_coverage.uncovered_rule_count"
SEED_COVERAGE_SUGGESTED_ADDITIONAL_CASES = "overmind.seed_coverage.suggested_additional_cases"

# ---------------------------------------------------------------------------
# Coding agent (overmind/coding_agent/{__init__,agent}.py)
# ---------------------------------------------------------------------------
CODING_AGENT_MODEL = "overmind.coding_agent.model"
CODING_AGENT_INPUT_FILE_COUNT = "overmind.coding_agent.input_file_count"
CODING_AGENT_MODIFIED_FILE_COUNT = "overmind.coding_agent.modified_file_count"
CODING_AGENT_STEPS_TAKEN = "overmind.coding_agent.steps_taken"
CODING_AGENT_MAX_STEPS = "overmind.coding_agent.max_steps"
CODING_AGENT_TOKENS_IN = "overmind.coding_agent.tokens_in"
CODING_AGENT_TOKENS_OUT = "overmind.coding_agent.tokens_out"
CODING_AGENT_LOOP_STEPS = "overmind.coding_agent.loop_steps"
CODING_AGENT_EXIT_REASON = "overmind.coding_agent.exit_reason"

# ---------------------------------------------------------------------------
# LLM call metadata (overmind/utils/llm.py)
#
# Aligned with the OpenTelemetry GenAI semantic conventions so the keys
# the SDK emits match what trace consumers (the optimizer's trace_reader,
# backend ingest, third-party auto-instrumentors like
# ``opentelemetry-instrumentation-openai``) expect.  Token counts use
# ``input_tokens`` / ``output_tokens`` per the semconv (NOT
# ``prompt_tokens`` / ``completion_tokens``).
# ---------------------------------------------------------------------------
LLM_MODEL = "gen_ai.request.model"
LLM_PROVIDER = "gen_ai.system"
LLM_ERROR = "gen_ai.error.type"
LLM_ELAPSED_SECONDS = "gen_ai.elapsed_seconds"
LLM_REQUEST_MESSAGE_COUNT = "gen_ai.request.message_count"
LLM_REQUEST_MESSAGE_CHARS = "gen_ai.request.message_chars"
LLM_REQUEST_TOOL_COUNT = "gen_ai.request.tool_count"
LLM_REQUEST_KWARGS = "gen_ai.request.kwargs"
LLM_USAGE_PROMPT_TOKENS = "gen_ai.usage.input_tokens"
LLM_USAGE_COMPLETION_TOKENS = "gen_ai.usage.output_tokens"
LLM_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"

# ---------------------------------------------------------------------------
# Span-level classification / error summary
# ---------------------------------------------------------------------------
# Human-readable summary string attached when a span fails. ``ERROR_TYPE``
# and ``ERROR_MESSAGE`` carry the structured (class name + scrubbed
# message) pair; ``ERROR_SUMMARY`` is the free-form line the UI renders
# next to the failed iteration.  Aliased as ``ERROR`` for backward
# compatibility with older spans.
ERROR_SUMMARY = "overmind.error"
ERROR = ERROR_SUMMARY  # Legacy alias — prefer ``ERROR_SUMMARY``.
# Explicit span-type override — set to SpanType values when the heuristic
# classification would be wrong (e.g. "entry_point", "tool", "workflow").
SPAN_TYPE = "overmind.span.type"
