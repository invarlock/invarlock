from __future__ import annotations

import copy
import itertools
import json
from decimal import ROUND_UP, Decimal, Inexact, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

from invarlock.judge_measurements.analysis import decode_analysis_policy
from invarlock.judge_measurements.contracts import measurement_plan_digest
from invarlock.judge_measurements.precision import _width_bounds, plan_precision
from invarlock.judge_measurements.statistics import hoeffding_interval

FIXTURES = Path(__file__).parents[1] / "fixtures" / "judge_measurements"
D = Decimal


def forecast(n=40, width="0.2", scale=("0", "1"), repetitions=1, minimum=1):
    plan = json.loads((FIXTURES / "plan.json").read_text())
    plan["sampling"]["case_units"] = [
        {"case_id": str(i), "unit_id": str(i)} for i in range(n)
    ]
    binding = plan["answer_bindings"][0]
    plan["answer_bindings"] = [{**binding, "case_id": str(i)} for i in range(n)]
    plan["schedule"]["repetitions"] = repetitions
    plan["schedule"]["expected_trials"] = n * 2 * repetitions
    plan["scale"]["ratings"] = [
        {"label": str(i), "value": value} for i, value in enumerate(scale)
    ]
    policy = json.loads((FIXTURES / "analysis_policy.json").read_text())
    policy.update(
        plan_sha256=measurement_plan_digest(plan),
        maximum_interval_width=width,
        minimum_units=minimum,
        comparison_family_size=4,
        subject_bound="0.6",
    )
    return plan, decode_analysis_policy(policy, plan=plan)


def test_forecast_detects_impossible_width_without_observations():
    plan, policy = forecast(n=40, width="0.2")
    result = plan_precision(plan, policy)
    assert result["paired_effect"]["status"] == "unattainable"
    assert result["paired_effect"]["units_for_guaranteed_width"] == 1016
    assert result["subject_score"]["units_for_guaranteed_width"] == 254
    assert result["minimum_units_met"] is True


@pytest.mark.parametrize(
    "n,width,status,needed",
    [
        (422, "0.32", "within_limit", 397),
        (211, "0.32", "not_guaranteed", 397),
        (1288, "0.18", "within_limit", 1254),
    ],
)
def test_retained_study_precision_requirements(n, width, status, needed):
    plan, policy = forecast(n=n, width=width)
    result = plan_precision(plan, policy)
    assert result["paired_effect"]["status"] == status
    assert result["paired_effect"]["units_for_guaranteed_width"] == needed


def test_repetitions_and_cases_inside_units_do_not_improve_forecast():
    plan, policy = forecast()
    expected = plan_precision(plan, policy)
    plan["schedule"]["repetitions"] = 20
    plan["sampling"]["case_units"] += [
        {"case_id": f"extra-{i}", "unit_id": str(i)} for i in range(40)
    ]
    assert plan_precision(plan, policy) == expected


def test_count_and_width_are_separate_requirements():
    plan, policy = forecast(n=422, width="0.32", minimum=500)
    result = plan_precision(plan, policy)
    assert result["minimum_units_met"] is False
    assert result["paired_effect"]["status"] == "within_limit"
    assert result["minimum_units"] == 500


def test_forecast_does_not_mutate_inputs_or_inherit_decimal_context():
    plan, policy = forecast(scale=("0.123456789012345", "0.987654321098765"))
    original = copy.deepcopy(plan)
    expected = plan_precision(plan, policy)
    _width_bounds.cache_clear()
    with localcontext() as ctx:
        ctx.prec = 3
        ctx.rounding = ROUND_UP
        ctx.traps[Inexact] = True
        assert plan_precision(plan, policy) == expected
    assert plan == original


@pytest.mark.parametrize(
    "scale", [("0", "1"), ("0.2", "0.7"), ("0.500000000000001", "0.500000000000002")]
)
def test_forecast_encloses_every_small_sample_interval(scale):
    # Enumerate all score outcomes, including fractional means and clipping.
    n = 5
    plan, policy = forecast(n=n, scale=scale)
    result = plan_precision(plan, policy)
    low, high = map(D, scale)
    width = high - low
    for key, support, choices in (
        (
            "paired_effect",
            (-width, width),
            (-Fraction(width), Fraction(0), Fraction(width)),
        ),
        (
            "subject_score",
            (low, high),
            (Fraction(low), (Fraction(low) + Fraction(high)) / 2, Fraction(high)),
        ),
    ):
        lower = Fraction(result[key]["minimum_width_lower_bound"])
        upper = Fraction(result[key]["maximum_width_upper_bound"])
        for values in itertools.product(choices, repeat=n):
            interval = hoeffding_interval(
                values, lower_bound=support[0], upper_bound=support[1], comparisons=4
            )
            actual = Fraction(interval.upper) - Fraction(interval.lower)
            assert lower <= actual <= upper


@pytest.mark.parametrize(
    "n,alpha,comparisons",
    [
        (11, "0.05", 2),
        (40, "0.05", 4),
        (1288, "0.000000000000001", 4),
        (10000, "0.999999999999999", 1000000),
    ],
)
def test_bounds_cover_fractional_means_and_unclipped_rounding(n, alpha, comparisons):
    lower, upper = D("0.123456789012345"), D("0.987654321098765")
    minimum, maximum = _width_bounds(n, lower, upper, D(alpha), comparisons)
    for position in range(18):
        mean = Fraction(lower) + (Fraction(upper) - Fraction(lower)) * position / 17
        interval = hoeffding_interval(
            itertools.repeat(mean, n),
            lower_bound=lower,
            upper_bound=upper,
            alpha=D(alpha),
            comparisons=comparisons,
        )
        width = Fraction(interval.upper) - Fraction(interval.lower)
        assert minimum <= width <= maximum


def test_unavailable_count_is_explicit_and_descriptive_subject_is_labeled():
    plan, policy = forecast(width="0.000000000000001")
    policy = type(policy)(**{**policy.__dict__, "subject_bound": None})
    result = plan_precision(plan, policy)
    assert result["paired_effect"]["units_for_guaranteed_width"] is None
    assert result["unit_search_limit"] == 10000
    assert result["subject_score"]["role"] == "descriptive"


def test_zero_unit_plan_is_rejected():
    plan, policy = forecast()
    plan["sampling"]["case_units"] = []
    with pytest.raises(ValueError, match="independent unit"):
        plan_precision(plan, policy)
