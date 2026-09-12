"""Typed wire objects for retained text judgments; validate before interpreting.

These types describe the two standalone measurement contracts. They do not
qualify a collector, authorize a plan, or confer an evidence acceptance scope.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class JudgeRubric(TypedDict):
    text: str
    sha256: str


class JudgeDemonstration(TypedDict):
    input: str
    answer: str
    rating: str


class JudgeReference(TypedDict):
    id: str
    text: str
    sha256: str


class JudgePrompt(TypedDict):
    system: str
    template: str
    demonstrations: list[JudgeDemonstration]
    references: list[JudgeReference]


class JudgeModelIdentity(TypedDict):
    kind: Literal["hosted_api", "local_weights"]
    weights_sha256: str | None


class JudgeConfig(TypedDict):
    temperature: str
    top_p: str
    max_output_tokens: int
    seed: int | None


class JudgeIdentity(TypedDict):
    provider: str
    requested_model: str
    approved_resolved_models: list[str]
    model_identity: JudgeModelIdentity
    config: JudgeConfig
    tools: Literal[False]


class JudgeParser(TypedDict):
    id: Literal["json-rating-v1"]


class JudgeRating(TypedDict):
    label: str
    value: str


class JudgeScale(TypedDict):
    id: Literal["bounded-discrete-v1"]
    ratings: list[JudgeRating]


class JudgeCaseUnit(TypedDict):
    case_id: str
    unit_id: str


class JudgeSampling(TypedDict):
    basis: Literal["curated_benchmark"]
    unit_weighting: Literal["equal"]
    within_unit_weighting: Literal["equal_cases"]
    case_units: list[JudgeCaseUnit]


class JudgeSchedule(TypedDict):
    order: Literal["case-side-repetition-lexicographic-v1"]
    trial_id_scheme: Literal["plan-case-side-repetition-sha256-v1"]
    repetitions: int
    max_attempts: int
    attempt_selection: Literal["first_completed_response"]
    retry_on: list[Literal["transport_error"]]
    cache: Literal["forbid"]
    expected_trials: int


class JudgeAnswerBinding(TypedDict):
    case_id: str
    baseline_answer_sha256: str
    subject_answer_sha256: str
    baseline_request_sha256: str
    subject_request_sha256: str


class JudgeMeasurementPlan(TypedDict):
    format: Literal["invarlock/judge-measurement-plan-v1"]
    profile_id: Literal["text-frozen-answer-v1"]
    case_set_sha256: str
    baseline_run_sha256: str
    subject_run_sha256: str
    rubric: JudgeRubric
    prompt: JudgePrompt
    judge: JudgeIdentity
    parser: JudgeParser
    scale: JudgeScale
    sampling: JudgeSampling
    schedule: JudgeSchedule
    answer_bindings: list[JudgeAnswerBinding]


class JudgeJsonBlob(TypedDict):
    media_type: Literal["application/json"]
    text: str
    sha256: str


class JudgeAttemptError(TypedDict):
    code: str
    message: str


class JudgeUsage(TypedDict):
    input_tokens: int
    output_tokens: int


class JudgeSourceMapping(TypedDict):
    source_id: str
    scorer_id: str
    model_event_id: str
    record_index: int
    attempt_index: int


class JudgeAttempt(TypedDict):
    attempt: int
    role: Literal["judge"]
    resolved_model: str | None
    status: Literal[
        "completed", "refusal", "transport_error", "timeout_ambiguous", "cancelled"
    ]
    request: JudgeJsonBlob
    response: JudgeJsonBlob | None
    request_id: str | None
    finish_reason: str | None
    error: JudgeAttemptError | None
    usage: JudgeUsage | None
    cache: Literal["none", "reused"]
    source: JudgeSourceMapping


class JudgeParseResult(TypedDict):
    status: Literal["ok", "invalid", "refusal", "unavailable"]
    rating: str | None
    value: str | None


class JudgeTrial(TypedDict):
    trial_id: str
    case_id: str
    side: Literal["baseline", "subject"]
    repetition: int
    answer_sha256: str
    plan_sha256: str
    status: Literal["complete", "incomplete"]
    attempts: list[JudgeAttempt]
    selected_attempt: int | None
    parse: JudgeParseResult


class JudgeRetainedSource(TypedDict):
    source_id: str
    profile: Literal["retained-judge-json-v1"]
    encoding: Literal["utf-8"]
    byte_size: int
    media_type: Literal["application/json"]
    content: str
    sha256: str


class JudgeCompleteness(TypedDict):
    status: Literal["complete", "incomplete"]
    expected_trials: int
    recorded_trials: int
    completed_trials: int


class JudgeMeasurements(TypedDict):
    format: Literal["invarlock/judge-measurements-v1"]
    profile_id: Literal["text-frozen-answer-v1"]
    plan_sha256: str
    source_profile: Literal["retained-judge-json-v1"]
    sources: list[JudgeRetainedSource]
    trials: list[JudgeTrial]
    completeness: JudgeCompleteness


class JudgeAnalysisPolicyDocument(TypedDict):
    """Standalone decision policy; decimal fields use canonical string values."""

    format: Literal["invarlock/judge-analysis-policy-v1"]
    plan_sha256: str
    metric_name: str
    decision_role: Literal["required", "advisory"]
    direction: Literal["higher", "lower"]
    method: Literal["fixed-benchmark-hoeffding-v1"]
    alpha: str
    comparison_family_size: int
    minimum_units: int
    maximum_interval_width: str
    allowed_degradation: str
    subject_bound: str | None
