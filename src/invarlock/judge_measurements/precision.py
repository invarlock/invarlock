"""Advisory precision bounds for a complete, predeclared judge schedule."""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, Context, Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
from itertools import repeat
from typing import Any

from invarlock.judge_measurement_types import JudgeMeasurementPlan
from invarlock.judge_measurements.analysis import JudgeAnalysisPolicy
from invarlock.judge_measurements.statistics import (
    DEFAULT_QUANTUM,
    METHOD_ID,
    hoeffding_interval,
)

UNIT_SEARCH_LIMIT = 10_000


def _decimal(value: Fraction) -> str:
    # Width bounds are multiples of the interval's decimal quantum.
    with localcontext(Context(prec=100)):
        return str(Decimal(value.numerator) / Decimal(value.denominator))


@lru_cache(maxsize=256)
def _width_bounds(
    units: int, lower: Decimal, upper: Decimal, alpha: Decimal, comparisons: int
) -> tuple[Fraction, Fraction]:
    """Enclose all rounded widths, including support clipping at either edge.

    Before outward quantization, the minimum width is min(range, radius)
    and the maximum is min(range, 2 * radius). Endpoints attain the minimum;
    the support midpoint attains the maximum. Outward rounding can add less
    than two quanta. Keep that allowance on both bounds rather than assuming
    the midpoint also maximizes the rounded width for every fractional mean.
    """
    widths = []
    for mean in (
        Fraction(lower),
        Fraction(upper),
        (Fraction(lower) + Fraction(upper)) / 2,
    ):
        interval = hoeffding_interval(
            repeat(mean, units),
            lower_bound=lower,
            upper_bound=upper,
            alpha=alpha,
            comparisons=comparisons,
        )
        widths.append(Fraction(interval.upper) - Fraction(interval.lower))
    padding = 2 * Fraction(DEFAULT_QUANTUM)
    with localcontext(Context(prec=100)):
        support_width = Fraction(
            upper.quantize(DEFAULT_QUANTUM, rounding=ROUND_CEILING)
        ) - Fraction(lower.quantize(DEFAULT_QUANTUM, rounding=ROUND_FLOOR))
    return max(Fraction(0), min(widths[:2]) - padding), min(
        support_width, widths[2] + padding
    )


def _interval_precision(
    units: int, lower: Decimal, upper: Decimal, policy: JudgeAnalysisPolicy
) -> dict[str, Any]:
    def bounds(count: int) -> tuple[Fraction, Fraction]:
        return _width_bounds(
            count, lower, upper, policy.alpha, policy.comparison_family_size
        )

    minimum, maximum = bounds(units)
    limit = Fraction(policy.maximum_interval_width)
    status = (
        "within_limit"
        if maximum <= limit
        else "unattainable"
        if minimum > limit
        else "not_guaranteed"
    )
    needed = None
    if bounds(UNIT_SEARCH_LIMIT)[1] <= limit:
        left, right = 1, UNIT_SEARCH_LIMIT
        while left < right:
            middle = (left + right) // 2
            if bounds(middle)[1] <= limit:
                right = middle
            else:
                left = middle + 1
        needed = left
    return {
        "minimum_width_lower_bound": _decimal(minimum),
        "maximum_width_upper_bound": _decimal(maximum),
        "status": status,
        "units_for_guaranteed_width": needed,
    }


def plan_precision(
    plan: JudgeMeasurementPlan, policy: JudgeAnalysisPolicy
) -> dict[str, Any]:
    """Forecast width only; callers supply an already validated plan and policy.

    This does not use ratings, alter execution readiness, change the policy,
    or predict acceptance. Counts assume every planned trial completes and
    the declared units are independent. The search is bounded by plan capacity.
    """
    units = len({case["unit_id"] for case in plan["sampling"]["case_units"]})
    if not 1 <= units <= UNIT_SEARCH_LIMIT:
        raise ValueError("precision planning requires 1 to 10000 independent units")
    values = [Decimal(rating["value"]) for rating in plan["scale"]["ratings"]]
    lower, upper = min(values), max(values)
    precision = max(
        100,
        upper.adjusted()
        - min(int(lower.as_tuple().exponent), int(upper.as_tuple().exponent))
        + 3,
    )
    with localcontext(Context(prec=precision)):
        width = upper - lower
    effect = _interval_precision(units, width.copy_negate(), width, policy)
    subject = _interval_precision(units, lower, upper, policy)
    subject["role"] = "descriptive" if policy.subject_bound is None else "decision"
    return {
        "method": METHOD_ID,
        "independent_units": units,
        "minimum_units": policy.minimum_units,
        "minimum_units_met": units >= policy.minimum_units,
        "maximum_interval_width": str(policy.maximum_interval_width),
        "unit_search_limit": UNIT_SEARCH_LIMIT,
        "paired_effect": effect,
        "subject_score": subject,
        "scope": "complete_planned_schedule",
    }
