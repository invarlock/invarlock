"""Bounded inference over a fixed scheduled benchmark.

The sampling assumption is independence between units, not random sampling of
benchmark cases or identical unit distributions. The target is the equal-unit
average expected score on the fixed benchmark. Repetitions and cases within a
unit never increase the inference sample size. Scores within a unit may depend
on each other. This module cannot establish independence from recorded scores.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import (
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    localcontext,
)
from fractions import Fraction
from typing import Literal

METHOD_ID = "fixed-benchmark-hoeffding-v1"
DEFAULT_QUANTUM = Decimal("0.000000000000001")
_PRECISION = 100
Decision = Literal["pass", "regression", "insufficient_evidence"]
Direction = Literal["higher", "lower"]
Side = Literal["baseline", "subject"]


@dataclass(frozen=True)
class ScheduledCase:
    case_id: str
    unit_id: str


@dataclass(frozen=True)
class Score:
    case_id: str
    side: Side
    repetition: int
    value: Decimal


@dataclass(frozen=True)
class UnitMean:
    unit_id: str
    baseline: Fraction
    subject: Fraction

    @property
    def effect(self) -> Fraction:
        """Subject minus baseline, before orienting for metric direction."""
        return self.subject - self.baseline


@dataclass(frozen=True)
class CollapsedScores:
    units: tuple[UnitMean, ...]
    case_count: int
    repetitions: int

    @property
    def unit_count(self) -> int:
        return len(self.units)


@dataclass(frozen=True)
class Interval:
    mean: Decimal
    lower: Decimal
    upper: Decimal
    unit_count: int
    alpha: Decimal
    comparisons: int
    method: str = METHOD_ID


@dataclass(frozen=True)
class GateDecision:
    decision: Decision
    required: bool = True


def _finite(value: Decimal, name: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{name} must be a finite Decimal")
    return value


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _direction(direction: Direction) -> None:
    if direction not in ("higher", "lower"):
        raise ValueError("direction must be higher or lower")


def collapse_scores(
    cases: Iterable[ScheduledCase],
    scores: Iterable[Score],
    *,
    repetitions: int,
    lower_bound: Decimal,
    upper_bound: Decimal,
) -> CollapsedScores:
    """Require every slot, then equally average repeats, cases, and units.

    Repetition indexes are zero based. There must be exactly one score for every
    scheduled case, side, and repetition, with no additional observations.
    Fractions preserve exact decimal scores through unequal group sizes and
    nonterminating averages; interval arithmetic is deterministic Decimal.
    """
    _positive_int(repetitions, "repetitions")
    _finite(lower_bound, "lower_bound")
    _finite(upper_bound, "upper_bound")
    if lower_bound > upper_bound:
        raise ValueError("lower_bound must not exceed upper_bound")
    scheduled: dict[str, str] = {}
    for case in cases:
        if not case.case_id or not case.unit_id or case.case_id in scheduled:
            raise ValueError(
                "scheduled cases need unique case IDs and nonempty unit IDs"
            )
        scheduled[case.case_id] = case.unit_id
    if not scheduled:
        raise ValueError("at least one scheduled case is required")
    observations: dict[tuple[str, Side, int], Fraction] = {}
    for score in scores:
        if (
            score.case_id not in scheduled
            or score.side not in ("baseline", "subject")
            or isinstance(score.repetition, bool)
            or not isinstance(score.repetition, int)
            or not 0 <= score.repetition < repetitions
        ):
            raise ValueError("score does not match a scheduled slot")
        _finite(score.value, "score")
        if not lower_bound <= score.value <= upper_bound:
            raise ValueError("score is outside the declared bounds")
        slot = (score.case_id, score.side, score.repetition)
        if slot in observations:
            raise ValueError("duplicate score slot")
        observations[slot] = Fraction(score.value)
    if len(observations) != len(scheduled) * 2 * repetitions:
        raise ValueError("missing required score slots")
    grouped: dict[str, list[tuple[Fraction, Fraction]]] = {}
    for case_id, unit_id in sorted(scheduled.items()):
        baseline = (
            sum(
                (observations[case_id, "baseline", r] for r in range(repetitions)),
                Fraction(),
            )
            / repetitions
        )
        subject = (
            sum(
                (observations[case_id, "subject", r] for r in range(repetitions)),
                Fraction(),
            )
            / repetitions
        )
        grouped.setdefault(unit_id, []).append((baseline, subject))
    units = tuple(
        UnitMean(
            unit_id,
            sum((b for b, _ in values), Fraction()) / len(values),
            sum((s for _, s in values), Fraction()) / len(values),
        )
        for unit_id, values in sorted(grouped.items())
    )
    return CollapsedScores(units, len(scheduled), repetitions)


def hoeffding_interval(
    values: Iterable[Decimal | Fraction],
    *,
    lower_bound: Decimal,
    upper_bound: Decimal,
    alpha: Decimal = Decimal("0.05"),
    comparisons: int = 1,
    quantum: Decimal = DEFAULT_QUANTUM,
) -> Interval:
    """Two-sided Hoeffding interval with Bonferroni family error control.

    Each value is one independent bounded unit mean. For n units and common
    range width w, radius = w * sqrt(log(2*K/alpha)/(2*n)). K is the fixed
    number of comparisons in the declared family, including advisory gates.
    No independence between comparisons is required. A constant observed
    sample still has positive radius unless the declared support is constant.
    """
    _finite(lower_bound, "lower_bound")
    _finite(upper_bound, "upper_bound")
    _finite(alpha, "alpha")
    _finite(quantum, "quantum")
    _positive_int(comparisons, "comparisons")
    if lower_bound > upper_bound:
        raise ValueError("lower_bound must not exceed upper_bound")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    if quantum <= 0 or quantum.as_tuple().digits != (1,):
        raise ValueError("quantum must be a positive power of ten")
    exact: list[Fraction] = []
    for value in values:
        if isinstance(value, Decimal):
            _finite(value, "unit value")
        elif not isinstance(value, Fraction):
            raise ValueError("unit values must be finite Decimals or Fractions")
        item = Fraction(value)
        if not Fraction(lower_bound) <= item <= Fraction(upper_bound):
            raise ValueError("unit value is outside the declared bounds")
        exact.append(item)
    if not exact:
        raise ValueError("at least one independent unit is required")
    mean = sum(exact, Fraction()) / len(exact)
    # Never inherit caller rounding, precision, or exponent limits. Directed
    # operations bound all arithmetic; ln and sqrt are correctly rounded to
    # nearest by Decimal, so their next representable value bounds them above.
    precision = max(
        _PRECISION,
        lower_bound.adjusted() - int(quantum.as_tuple().exponent) + 20,
        upper_bound.adjusted() - int(quantum.as_tuple().exponent) + 20,
    )
    with localcontext(
        Context(
            prec=precision,
            Emax=999999999,
            Emin=-999999999,
            rounding=ROUND_CEILING,
        )
    ) as ctx:
        width = upper_bound - lower_bound
        log_argument = Decimal(2 * comparisons) / alpha
        log_upper = log_argument.ln().next_plus()
        root_upper = (log_upper / Decimal(2 * len(exact))).sqrt().next_plus()
        radius = width * root_upper
        mean_upper = Decimal(mean.numerator) / Decimal(mean.denominator)
        upper = min(upper_bound, mean_upper + radius).quantize(
            quantum, rounding=ROUND_CEILING
        )
        ctx.rounding = ROUND_FLOOR
        mean_lower = Decimal(mean.numerator) / Decimal(mean.denominator)
        lower = max(lower_bound, mean_lower - radius).quantize(
            quantum, rounding=ROUND_FLOOR
        )
        ctx.rounding = ROUND_HALF_EVEN
        reported_mean = (Decimal(mean.numerator) / Decimal(mean.denominator)).quantize(
            quantum
        )
    return Interval(reported_mean, lower, upper, len(exact), alpha, comparisons)


def decide_effect(
    interval: Interval,
    *,
    direction: Direction,
    margin: Decimal,
    minimum_units: int = 1,
) -> Decision:
    """Decide a subject-minus-baseline interval against tolerated degradation."""
    _direction(direction)
    _finite(margin, "margin")
    _positive_int(minimum_units, "minimum_units")
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    if interval.unit_count < minimum_units:
        return "insufficient_evidence"
    # copy_negate avoids caller-context rounding in unary minus on Decimal.
    if direction == "higher":
        lower, upper = interval.lower, interval.upper
    else:
        lower, upper = interval.upper.copy_negate(), interval.lower.copy_negate()
    threshold = margin.copy_negate()
    if lower >= threshold:
        return "pass"
    if upper < threshold:
        return "regression"
    return "insufficient_evidence"


def decide_subject_bound(
    interval: Interval,
    *,
    direction: Direction,
    bound: Decimal,
    minimum_units: int = 1,
) -> Decision:
    """Decide an absolute floor (higher) or ceiling (lower) for the subject."""
    _direction(direction)
    _finite(bound, "bound")
    _positive_int(minimum_units, "minimum_units")
    if interval.unit_count < minimum_units:
        return "insufficient_evidence"
    if direction == "higher":
        if interval.lower >= bound:
            return "pass"
        if interval.upper < bound:
            return "regression"
    else:
        if interval.upper <= bound:
            return "pass"
        if interval.lower > bound:
            return "regression"
    return "insufficient_evidence"


def combine_decisions(gates: Iterable[GateDecision]) -> Decision:
    """Advisory results do not change the conjunction of required gates."""
    required: list[Decision] = []
    for gate in gates:
        if gate.decision not in ("pass", "regression", "insufficient_evidence"):
            raise ValueError("unknown gate decision")
        if not isinstance(gate.required, bool):
            raise ValueError("required must be boolean")
        if gate.required:
            required.append(gate.decision)
    if "regression" in required:
        return "regression"
    if not required or "insufficient_evidence" in required:
        return "insufficient_evidence"
    return "pass"
