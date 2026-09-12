"""Replay-backed bounded analysis of a frozen judge measurement schedule."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from decimal import Context, Decimal, localcontext
from fractions import Fraction
from typing import Any, Literal

from invarlock.judge_measurement_types import JudgeMeasurementPlan, JudgeMeasurements
from invarlock.judge_measurements.contracts import (
    canonical_payload,
    validate_measurements,
)
from invarlock.judge_measurements.statistics import (
    METHOD_ID,
    Decision,
    Direction,
    GateDecision,
    Interval,
    ScheduledCase,
    Score,
    collapse_scores,
    combine_decisions,
    decide_effect,
    decide_subject_bound,
    hoeffding_interval,
)

type JsonValue = str | int | bool | None | list[JsonValue] | dict[str, JsonValue]
ASSUMPTIONS = (
    "Declared units are independent; distributions may differ between units.",
    "Scores remain within the declared rating scale under the frozen judge protocol.",
    "Cases, answers, unit grouping, schedule, and decision policy are fixed before judging.",
    "Units have equal weight; cases within units and repetitions within cases have equal weight.",
    "Repetitions and cases within a unit do not increase the number of independent units.",
    "The declared comparison family includes both published intervals and all enclosing claims.",
)
ESTIMAND = (
    "Equal-unit average expected judge score on the fixed scheduled benchmark; "
    "the paired effect is subject minus baseline. No population generalization is asserted."
)


@dataclass(frozen=True)
class JudgeAnalysisPolicy:
    direction: Direction
    allowed_degradation: Decimal
    alpha: Decimal = Decimal("0.05")
    comparison_family_size: int = 2
    required: bool = True
    minimum_units: int = 1
    maximum_interval_width: Decimal = Decimal("2")
    subject_bound: Decimal | None = None

    def __post_init__(self) -> None:
        if self.direction not in ("higher", "lower"):
            raise ValueError("direction must be higher or lower")
        for name in ("allowed_degradation", "alpha", "maximum_interval_width"):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite():
                raise ValueError(f"{name} must be a finite Decimal")
        if self.allowed_degradation < 0:
            raise ValueError("allowed_degradation must be nonnegative")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be strictly between zero and one")
        if self.maximum_interval_width < 0:
            raise ValueError("maximum_interval_width must be nonnegative")
        for name, minimum in (("comparison_family_size", 2), ("minimum_units", 1)):
            count = getattr(self, name)
            if isinstance(count, bool) or not isinstance(count, int) or count < minimum:
                raise ValueError(f"{name} must be an integer at least {minimum}")
        if not isinstance(self.required, bool):
            raise ValueError("required must be boolean")
        if self.subject_bound is not None and (
            not isinstance(self.subject_bound, Decimal)
            or not self.subject_bound.is_finite()
        ):
            raise ValueError("subject_bound must be a finite Decimal or None")


@dataclass(frozen=True)
class CoverageCounts:
    expected_trials: int
    recorded_trials: int
    completed_trials: int
    incomplete_trials: int
    scheduled_cases: int
    scheduled_units: int
    complete_units: int
    repetitions: int


@dataclass(frozen=True)
class AnalysisGate:
    name: Literal["paired_effect", "subject_bound"]
    decision: Decision
    reasons: tuple[str, ...]


def _serializable(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    raise TypeError(f"unsupported analysis value: {type(value).__name__}")


@dataclass(frozen=True)
class JudgeAnalysisResult:
    plan_sha256: str
    measurements_sha256: str
    policy: JudgeAnalysisPolicy
    sampling_basis: Literal["curated_benchmark"]
    counts: CoverageCounts
    effect_interval: Interval | None
    subject_interval: Interval | None
    subject_interval_role: Literal["descriptive", "decision"]
    gates: tuple[AnalysisGate, ...]
    decision: Decision
    reasons: tuple[str, ...]
    method: str = METHOD_ID
    estimand: str = ESTIMAND
    assumptions: tuple[str, ...] = ASSUMPTIONS

    def to_dict(self) -> dict[str, JsonValue]:
        """Return fresh JSON-safe data; decimal values retain their exact strings."""
        result = _serializable(asdict(self))
        assert isinstance(result, dict)
        return result


def _coverage(plan: JudgeMeasurementPlan, data: JudgeMeasurements) -> CoverageCounts:
    case_units = {
        case["case_id"]: case["unit_id"] for case in plan["sampling"]["case_units"]
    }
    expected = Counter(case_units.values())
    completed: Counter[str] = Counter()
    for trial in data["trials"]:
        if trial["status"] == "complete" and trial["parse"]["status"] == "ok":
            completed[case_units[trial["case_id"]]] += 1
    repetitions = plan["schedule"]["repetitions"]
    complete_units = sum(
        completed[unit] == cases * repetitions * 2 for unit, cases in expected.items()
    )
    expected_trials = plan["schedule"]["expected_trials"]
    completed_trials = sum(completed.values())
    return CoverageCounts(
        expected_trials,
        len(data["trials"]),
        completed_trials,
        expected_trials - completed_trials,
        len(case_units),
        len(expected),
        complete_units,
        repetitions,
    )


def _gate(
    name: Literal["paired_effect", "subject_bound"],
    interval: Interval,
    policy: JudgeAnalysisPolicy,
) -> AnalysisGate:
    reasons = []
    if interval.unit_count < policy.minimum_units:
        reasons.append("minimum_units_not_met")
    # Exact rational subtraction avoids ambient Decimal context rounding at
    # the inclusive width threshold.
    if Fraction(interval.upper) - Fraction(interval.lower) > Fraction(
        policy.maximum_interval_width
    ):
        reasons.append("maximum_interval_width_exceeded")
    if reasons:
        return AnalysisGate(name, "insufficient_evidence", tuple(reasons))
    if name == "paired_effect":
        decision = decide_effect(
            interval, direction=policy.direction, margin=policy.allowed_degradation
        )
    else:
        assert policy.subject_bound is not None
        decision = decide_subject_bound(
            interval, direction=policy.direction, bound=policy.subject_bound
        )
    if decision == "insufficient_evidence":
        return AnalysisGate(name, decision, ("interval_crosses_decision_threshold",))
    return AnalysisGate(name, decision, ())


def analyze_measurements(
    plan: JudgeMeasurementPlan,
    measurements: JudgeMeasurements,
    policy: JudgeAnalysisPolicy,
    *,
    baseline_run: dict[str, Any],
    subject_run: dict[str, Any],
) -> JudgeAnalysisResult:
    """Validate replay and analyze only a completely observed planned schedule.

    A valid retained bundle containing failed trials yields insufficient
    evidence, never complete-case inference. Contract violations raise
    JudgeMeasurementContractError. An advisory metric keeps its own result;
    combine_analysis_results excludes it from the required conjunction.
    """
    validate_measurements(
        measurements, plan, baseline_run=baseline_run, subject_run=subject_run
    )
    counts = _coverage(plan, measurements)
    effect = subject = None
    gates: tuple[AnalysisGate, ...] = ()
    reasons: tuple[str, ...] = ()
    if counts.incomplete_trials:
        decision: Decision = "insufficient_evidence"
        reasons = ("incomplete_planned_schedule",)
    else:
        values = [Decimal(rating["value"]) for rating in plan["scale"]["ratings"]]
        lower, upper = min(values), max(values)
        # The wire contract fixes scores in [0, 1], so exact support subtraction
        # can be expressed as a terminating decimal independently of context.
        precision = max(
            100,
            upper.adjusted()
            - min(int(lower.as_tuple().exponent), int(upper.as_tuple().exponent))
            + 3,
        )
        with localcontext(Context(prec=precision)):
            width = upper - lower
        collapsed = collapse_scores(
            (
                ScheduledCase(case["case_id"], case["unit_id"])
                for case in plan["sampling"]["case_units"]
            ),
            (
                Score(
                    trial["case_id"],
                    trial["side"],
                    trial["repetition"] - 1,
                    Decimal(trial["parse"]["value"]),
                )
                for trial in measurements["trials"]
                if trial["status"] == "complete"
                and trial["parse"]["status"] == "ok"
                and trial["parse"]["value"] is not None
            ),
            repetitions=counts.repetitions,
            lower_bound=lower,
            upper_bound=upper,
        )
        effect = hoeffding_interval(
            (unit.effect for unit in collapsed.units),
            lower_bound=width.copy_negate(),
            upper_bound=width,
            alpha=policy.alpha,
            comparisons=policy.comparison_family_size,
        )
        subject = hoeffding_interval(
            (unit.subject for unit in collapsed.units),
            lower_bound=lower,
            upper_bound=upper,
            alpha=policy.alpha,
            comparisons=policy.comparison_family_size,
        )
        gates = (_gate("paired_effect", effect, policy),)
        if policy.subject_bound is not None:
            gates += (_gate("subject_bound", subject, policy),)
        decision = combine_decisions(GateDecision(gate.decision) for gate in gates)
        reasons = tuple(sorted({reason for gate in gates for reason in gate.reasons}))
    return JudgeAnalysisResult(
        plan_sha256=measurements["plan_sha256"],
        measurements_sha256=hashlib.sha256(canonical_payload(measurements)).hexdigest(),
        policy=policy,
        sampling_basis=plan["sampling"]["basis"],
        counts=counts,
        effect_interval=effect,
        subject_interval=subject,
        subject_interval_role="descriptive"
        if policy.subject_bound is None
        else "decision",
        gates=gates,
        decision=decision,
        reasons=reasons,
    )


def combine_analysis_results(results: Iterable[JudgeAnalysisResult]) -> Decision:
    """Combine required metrics with an explicitly shared comparison family.

    Each result reserves two interval claims, even if data is incomplete or
    the metric is advisory. Results must use the same alpha and a declared
    family size at least twice the number of combined results.
    """
    family = tuple(results)
    if family:
        alpha = family[0].policy.alpha
        if any(result.policy.alpha != alpha for result in family):
            raise ValueError("combined results must declare the same family alpha")
        if any(
            result.policy.comparison_family_size < 2 * len(family) for result in family
        ):
            raise ValueError("comparison family is too small for combined intervals")
    return combine_decisions(
        GateDecision(result.decision, result.policy.required) for result in family
    )
