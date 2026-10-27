"""Independent arithmetic expectations; no acceptance assertions for known errors."""

import itertools
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "audit"))
from reference_numerics import (
    exact_mean,
    exact_ratio,
    exhaustive_ratios,
    hoeffding_family_tail,
    scalar_request,
    scheduled_nll_interval,
    sign_enumeration_probability,
)


def test_exact_means_preserve_large_integer_and_finite_cancellation():
    assert exact_mean([2**60 + 1]) - exact_mean([2**60]) == 1
    assert exact_mean([1e308, -1e308]) == 0
    assert exact_ratio([1, 3], [2, 3]) == Fraction(5, 4)


def test_binary_counts_effects_and_probability_by_sign_enumeration():
    from invarlock.paired_exact_match import paired_exact_match_statistics

    for n in range(1, 6):
        for pairs in itertools.product(((0, 0), (0, 1), (1, 0), (1, 1)), repeat=n):
            baseline, subject = map(list, zip(*pairs, strict=False))
            losses = sum(b > s for b, s in pairs)
            gains = sum(s > b for b, s in pairs)
            result = paired_exact_match_statistics(baseline, subject)
            assert result.baseline_pass_count == sum(baseline)
            assert result.subject_pass_count == sum(subject)
            assert result.effect_size_pp == pytest.approx(
                float(Fraction(100 * (gains - losses), n))
            )
            assert result.mcnemar_exact_two_sided_p_value == float(
                sign_enumeration_probability(losses, gains)
            )


@pytest.mark.parametrize(
    "baseline,subject", [([1, 3], [2, 3]), ([1, 2, 4], [3, 2, 2]), ([2, 2], [0, 0])]
)
def test_nll_tiny_enumerated_population_and_independent_scheduled_arithmetic(
    baseline, subject
):
    from invarlock.evaluation_comparison.likelihood import likelihood_statistics

    population = exhaustive_ratios(baseline, subject)
    ratio, interval = likelihood_statistics(baseline, subject, "sha256:" + "ab" * 32)
    assert ratio == pytest.approx(float(exact_ratio(baseline, subject)))
    assert interval["lower"] == pytest.approx(float(population[0]))
    assert interval["upper"] == pytest.approx(float(population[-1]))
    independent = scheduled_nll_interval(baseline, subject, bytes.fromhex("ab" * 32))
    assert tuple(map(float, independent)) == pytest.approx(
        (interval["lower"], interval["upper"])
    )


def test_hoeffding_endpoints_satisfy_family_tail_inequality():
    from invarlock.judge_measurements.statistics import hoeffding_interval

    interval = hoeffding_interval(
        [Fraction(1, 3)] * 100,
        lower_bound=Decimal(0),
        upper_bound=Decimal(1),
        comparisons=2,
    )
    # Exact rational mean avoids trusting the reported, rounded center.
    from decimal import localcontext

    with localcontext() as context:
        context.prec = 150
        mean = Decimal(1) / 3
        for endpoint in (interval.lower, interval.upper):
            tail = hoeffding_family_tail(abs(endpoint - mean), 1, 100, 2)
            assert tail <= Decimal(".05")
            assert tail > Decimal(".04999999999999")


def test_judge_unequal_group_sizes_and_missing_slot():
    from invarlock.judge_measurements.statistics import (
        ScheduledCase,
        Score,
        collapse_scores,
    )

    cases = [ScheduledCase("a", "A"), ScheduledCase("b", "A"), ScheduledCase("c", "B")]
    scores = [
        Score(case.case_id, side, repetition, Decimal(value))
        for case, value in zip(cases, (1, 0, 1), strict=False)
        for side in ("baseline", "subject")
        for repetition in range(2)
    ]
    kwargs = {"repetitions": 2, "lower_bound": Decimal(0), "upper_bound": Decimal(1)}
    result = collapse_scores(cases, scores, **kwargs)
    assert [unit.subject for unit in result.units] == [Fraction(1, 2), Fraction(1)]
    assert exact_mean(unit.subject for unit in result.units) == Fraction(3, 4)
    assert collapse_scores(cases[::-1], scores[::-1], **kwargs) == result
    with pytest.raises(ValueError, match="missing"):
        collapse_scores(cases, scores[:-1], **kwargs)


def test_public_scalar_means_pair_order_and_missingness():
    from invarlock.evaluation_comparison.comparison import compare_runs

    request = scalar_request([1, 2, 4], [2, 3, 5])
    result = compare_runs(**request)["metrics"][0]
    assert result["baseline_mean"] == pytest.approx(float(Fraction(7, 3)))
    assert result["subject_mean"] == pytest.approx(float(Fraction(10, 3)))
    assert result["interval"]["lower"] == result["interval"]["upper"] == 1
    assert result["decision"] == "pass"
    for key in ("baseline", "subject"):
        request[key]["records"].reverse()
    assert compare_runs(**request)["metrics"][0] == result
    request["subject"]["records"][0]["scores"] = {}
    missing = compare_runs(**request)["metrics"][0]
    assert missing["decision"] == "insufficient_evidence"
    assert missing["baseline_mean"] is None
    assert len(missing["missing_ids"]) == 1
