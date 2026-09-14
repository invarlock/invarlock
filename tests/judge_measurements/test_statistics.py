from decimal import ROUND_UP, Decimal, Inexact, localcontext
from fractions import Fraction

import pytest

from invarlock.judge_measurements.statistics import (
    METHOD_ID,
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

D = Decimal


def collapse(cases, scores, repetitions=1):
    return collapse_scores(
        cases, scores, repetitions=repetitions, lower_bound=D(0), upper_bound=D(1)
    )


def interval(values, **kwargs):
    return hoeffding_interval(values, lower_bound=D(0), upper_bound=D(1), **kwargs)


def fixed_interval(lower, upper, n=10):
    return Interval(D(0), D(lower), D(upper), n, D(".05"), 1)


def test_constant_observations_do_not_imply_zero_uncertainty():
    actual = interval([D(".5")] * 100)
    assert actual.mean == D(".500000000000000")
    assert actual.lower == D(".364189848425938")
    assert actual.upper == D(".635810151574062")
    assert actual.unit_count == 100
    assert actual.method == METHOD_ID


def test_support_clipping_and_single_unit():
    assert interval([D(0)]).lower == 0
    assert interval([D(0)]).upper == 1
    assert interval([D(1)]).lower == 0
    assert interval([D(1)]).upper == 1
    result = hoeffding_interval([D(".3")], lower_bound=D(".3"), upper_bound=D(".3"))
    assert result.lower == result.upper == D(".3")


def test_bonferroni_widens_interval_and_more_units_narrow_it():
    original = interval([D(".5")] * 100)
    multiple = interval([D(".5")] * 100, comparisons=10)
    larger = interval([D(".5")] * 1000)
    assert multiple.lower < original.lower < larger.lower
    assert multiple.upper > original.upper > larger.upper
    assert multiple.comparisons == 10
    assert multiple.alpha == D(".05")


def test_outward_quantization_and_fraction_means():
    values = [Fraction(1, 3)] * 200
    fine = interval(values, quantum=D("1e-30"))
    coarse = interval(values, quantum=D(".001"))
    assert coarse.lower <= fine.lower <= fine.upper <= coarse.upper
    assert coarse.mean == D(".333")
    assert coarse.lower.as_tuple().exponent == -3


def test_results_ignore_caller_decimal_context_and_input_order():
    values = [Fraction(1, 3), D(".1234567890123456789"), D(1)] * 90
    expected = interval(values)
    with localcontext() as ctx:
        ctx.prec = 3
        ctx.rounding = ROUND_UP
        ctx.traps[Inexact] = True
        assert interval(reversed(values)) == expected


def test_equal_unit_weight_and_equal_case_weight_inside_units():
    cases = [
        ScheduledCase("a", "u1"),
        ScheduledCase("b", "u1"),
        ScheduledCase("c", "u2"),
    ]
    scores = [
        Score(case, side, rep, D(value))
        for case, values in (("a", ("0", "1")), ("b", ("0", "0")), ("c", ("0", "1")))
        for side, value in zip(("baseline", "subject"), values, strict=True)
        for rep in range(2)
    ]
    result = collapse(cases, scores, repetitions=2)
    assert result.unit_count == 2
    assert result.case_count == 3
    assert [unit.subject for unit in result.units] == [Fraction(1, 2), Fraction(1)]
    assert interval(unit.subject for unit in result.units).mean == D(".75")
    assert collapse(reversed(cases), reversed(scores), repetitions=2) == result


def test_repeated_scores_do_not_inflate_sample_size_or_change_interval():
    cases = [ScheduledCase(str(i), str(i // 3)) for i in range(300)]

    def run(repetitions):
        result = collapse(
            cases,
            (
                Score(case.case_id, side, r, value)
                for case in cases
                for side, value in (("baseline", D(".3")), ("subject", D(".6")))
                for r in range(repetitions)
            ),
            repetitions,
        )
        return interval(unit.effect for unit in result.units)

    assert run(1) == run(7)
    assert run(7).unit_count == 100


def test_repetition_then_case_average_is_exact():
    cases = [ScheduledCase("a", "u"), ScheduledCase("b", "u"), ScheduledCase("c", "u")]
    scores = [
        Score(case.case_id, side, r, D(int(case.case_id == "a" and r == 0)))
        for case in cases
        for side in ("baseline", "subject")
        for r in range(3)
    ]
    with localcontext() as ctx:
        ctx.prec = 2
        result = collapse(cases, scores, repetitions=3)
    assert result.units[0].subject == Fraction(1, 9)
    assert result.units[0].effect == 0


@pytest.mark.parametrize(
    "scores,match",
    [
        ([], "missing required"),
        ([Score("a", "baseline", 0, D(0))], "missing required"),
        ([Score("a", "baseline", 0, D(0))] * 2, "duplicate"),
        ([Score("unknown", "baseline", 0, D(0))], "scheduled slot"),
        ([Score("a", "baseline", 1, D(0))], "scheduled slot"),
        ([Score("a", "baseline", True, D(0))], "scheduled slot"),
        ([Score("a", "other", 0, D(0))], "scheduled slot"),
        ([Score("a", "baseline", 0, D("1.1"))], "outside"),
        ([Score("a", "baseline", 0, D("NaN"))], "finite"),
    ],
)
def test_incomplete_extra_duplicate_and_invalid_scores_fail(scores, match):
    with pytest.raises(ValueError, match=match):
        collapse([ScheduledCase("a", "u")], scores)


@pytest.mark.parametrize(
    "cases", [[], [ScheduledCase("a", "")], [ScheduledCase("a", "u")] * 2]
)
def test_invalid_schedules_fail(cases):
    with pytest.raises(ValueError):
        collapse(cases, [])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": D(0)},
        {"alpha": D(1)},
        {"alpha": D("NaN")},
        {"comparisons": 0},
        {"comparisons": True},
        {"comparisons": 1.5},
        {"quantum": D(0)},
        {"quantum": D(".03")},
        {"quantum": D("Infinity")},
    ],
)
def test_invalid_interval_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        interval([D(".5")], **kwargs)


@pytest.mark.parametrize("values", [[], [D(2)], [D("NaN")], [D("Infinity")], [0.5]])
def test_invalid_unit_values_fail(values):
    with pytest.raises(ValueError):
        interval(values)


@pytest.mark.parametrize(
    "lower,upper,expected",
    [
        ("-.1", ".1", "pass"),
        ("-.2", "-.11", "regression"),
        ("-.2", "-.1", "insufficient_evidence"),
        ("-.2", ".1", "insufficient_evidence"),
        ("-.1", "-.1", "pass"),
    ],
)
def test_effect_threshold_equality_and_direction(lower, upper, expected):
    assert (
        decide_effect(fixed_interval(lower, upper), direction="higher", margin=D(".1"))
        == expected
    )
    assert (
        decide_effect(
            fixed_interval(D(upper).copy_negate(), D(lower).copy_negate()),
            direction="lower",
            margin=D(".1"),
        )
        == expected
    )


@pytest.mark.parametrize(
    "lower,upper,direction,expected",
    [
        (".5", ".8", "higher", "pass"),
        (".4", ".49", "higher", "regression"),
        (".4", ".5", "higher", "insufficient_evidence"),
        (".2", ".5", "lower", "pass"),
        (".51", ".6", "lower", "regression"),
        (".5", ".6", "lower", "insufficient_evidence"),
        (".5", ".5", "lower", "pass"),
    ],
)
def test_subject_bound_threshold_equality(lower, upper, direction, expected):
    assert (
        decide_subject_bound(
            fixed_interval(lower, upper), direction=direction, bound=D(".5")
        )
        == expected
    )


def test_minimum_units_precedes_decisions():
    assert (
        decide_effect(
            fixed_interval(".5", ".8", n=2),
            direction="higher",
            margin=D(0),
            minimum_units=3,
        )
        == "insufficient_evidence"
    )
    assert (
        decide_subject_bound(
            fixed_interval(".5", ".8", n=2),
            direction="higher",
            bound=D(".2"),
            minimum_units=3,
        )
        == "insufficient_evidence"
    )


def test_advisory_results_do_not_change_required_conjunction():
    assert (
        combine_decisions([GateDecision("pass"), GateDecision("regression", False)])
        == "pass"
    )
    assert (
        combine_decisions([GateDecision("insufficient_evidence"), GateDecision("pass")])
        == "insufficient_evidence"
    )
    assert (
        combine_decisions(
            [GateDecision("insufficient_evidence"), GateDecision("regression")]
        )
        == "regression"
    )
    assert combine_decisions([GateDecision("pass", False)]) == "insufficient_evidence"
    assert combine_decisions([]) == "insufficient_evidence"


def test_decisions_ignore_low_precision_callers():
    data = fixed_interval("-.10000000001", "-.10000000001")
    with localcontext() as ctx:
        ctx.prec = 1
        ctx.traps[Inexact] = True
        assert decide_effect(data, direction="higher", margin=D(".1")) == "regression"
        assert decide_effect(data, direction="lower", margin=D(0)) == "pass"


def test_paired_effect_uses_full_declared_difference_support():
    subject = interval([D(".5")] * 100)
    effect = hoeffding_interval([D(0)] * 100, lower_bound=D(-1), upper_bound=D(1))
    assert effect.lower == D("-.271620303148124")
    assert effect.upper == D(".271620303148124")
    assert effect.upper == 2 * (subject.upper - subject.mean)


@pytest.mark.parametrize("repetitions", [0, -1, True, 1.5])
def test_invalid_repetition_count_fails(repetitions):
    with pytest.raises(ValueError, match="positive integer"):
        collapse([ScheduledCase("a", "u")], [], repetitions=repetitions)


def test_reversed_support_fails_in_both_paths():
    with pytest.raises(ValueError, match="lower_bound"):
        hoeffding_interval([D(0)], lower_bound=D(1), upper_bound=D(0))
    with pytest.raises(ValueError, match="lower_bound"):
        collapse_scores([], [], repetitions=1, lower_bound=D(1), upper_bound=D(0))


def test_invalid_decision_configuration_fails():
    data = fixed_interval(0, 1)
    with pytest.raises(ValueError, match="direction"):
        decide_effect(data, direction="unknown", margin=D(0))
    with pytest.raises(ValueError, match="nonnegative"):
        decide_effect(data, direction="higher", margin=D(-1))
    with pytest.raises(ValueError, match="positive integer"):
        decide_effect(data, direction="higher", margin=D(0), minimum_units=0)
    with pytest.raises(ValueError, match="finite"):
        decide_subject_bound(data, direction="higher", bound=D("NaN"))
    with pytest.raises(ValueError, match="unknown gate"):
        combine_decisions([GateDecision("unknown")])
    with pytest.raises(ValueError, match="boolean"):
        combine_decisions([GateDecision("pass", required=1)])


def test_high_magnitude_inputs_quantize_without_precision_loss():
    result = hoeffding_interval(
        [D("1e100")], lower_bound=D("1e100"), upper_bound=D("1e100")
    )
    assert result.lower == result.upper == result.mean == D("1e100")
