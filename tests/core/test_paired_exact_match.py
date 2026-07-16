from __future__ import annotations

import math
from itertools import product

import pytest

from invarlock.paired_exact_match import (
    MAX_PAIRED_EXACT_MATCH_OUTCOMES,
    PAIRED_CONFIDENCE_INTERVAL_METHOD,
    PairedExactMatchError,
    paired_exact_match_statistics,
)


def test_paired_statistics_replay_directional_counts_and_effect() -> None:
    statistics = paired_exact_match_statistics(
        [True, True, True, False, False, False],
        [True, False, False, True, False, False],
    )

    assert statistics.pair_count == 6
    assert statistics.baseline_pass_count == 3
    assert statistics.subject_pass_count == 2
    assert statistics.baseline_pass_subject_fail_count == 2
    assert statistics.baseline_fail_subject_pass_count == 1
    assert statistics.both_pass_count == 1
    assert statistics.both_fail_count == 2
    assert statistics.effect_size_pp == pytest.approx(-100.0 / 6.0)
    assert statistics.mcnemar_exact_two_sided_p_value == 1.0
    interval = statistics.effect_size_confidence_interval
    assert interval.method == PAIRED_CONFIDENCE_INTERVAL_METHOD
    assert interval.confidence_level == 0.95
    assert interval.lower_pp <= statistics.effect_size_pp <= interval.upper_pp


def test_exact_two_sided_mcnemar_probability_uses_only_discordant_pairs() -> None:
    statistics = paired_exact_match_statistics(
        [True] * 10 + [True, False, False],
        [False] * 10 + [True, False, False],
    )

    assert statistics.baseline_pass_subject_fail_count == 10
    assert statistics.baseline_fail_subject_pass_count == 0
    assert statistics.mcnemar_exact_two_sided_p_value == pytest.approx(2 / 1024)
    assert statistics.effect_size_pp == pytest.approx(-1000 / 13)

    asymmetric = paired_exact_match_statistics(
        [True] * 5 + [False], [False] * 5 + [True]
    )
    assert asymmetric.mcnemar_exact_two_sided_p_value == pytest.approx(14 / 64)


def test_balanced_discordant_pairs_have_no_effect_and_unit_p_value() -> None:
    statistics = paired_exact_match_statistics(
        [True, False, True, False],
        [False, True, True, False],
    )

    assert statistics.baseline_pass_subject_fail_count == 1
    assert statistics.baseline_fail_subject_pass_count == 1
    assert statistics.effect_size_pp == 0.0
    assert statistics.mcnemar_exact_two_sided_p_value == 1.0


def test_confidence_interval_is_deterministic_and_bounded() -> None:
    baseline = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    subject = [1, 0, 0, 1, 1, 0, 0, 0]

    first = paired_exact_match_statistics(baseline, subject)
    second = paired_exact_match_statistics(tuple(baseline), tuple(subject))

    assert first == second
    interval = first.effect_size_confidence_interval
    assert interval.lower_pp == pytest.approx(-41.08300125374949)
    assert interval.upper_pp == pytest.approx(41.08300125374949)
    assert -100.0 <= interval.lower_pp <= interval.upper_pp <= 100.0


def test_all_concordant_pairs_have_no_mcnemar_signal() -> None:
    statistics = paired_exact_match_statistics(
        [True, False, True, False], [True, False, True, False]
    )

    assert statistics.mcnemar_exact_two_sided_p_value == 1.0
    assert statistics.effect_size_pp == 0.0
    assert statistics.effect_size_confidence_interval.lower_pp == pytest.approx(0.0)
    assert statistics.effect_size_confidence_interval.upper_pp == pytest.approx(0.0)


def test_small_sample_intervals_are_bounded_and_contain_the_paired_effect() -> None:
    for baseline in product((False, True), repeat=4):
        for subject in product((False, True), repeat=4):
            statistics = paired_exact_match_statistics(baseline, subject)
            interval = statistics.effect_size_confidence_interval
            assert -100.0 <= interval.lower_pp <= statistics.effect_size_pp
            assert statistics.effect_size_pp <= interval.upper_pp <= 100.0


def test_swapping_sides_reverses_effect_counts_and_interval() -> None:
    baseline = [True, True, False, False, False]
    subject = [False, True, True, True, False]

    forward = paired_exact_match_statistics(baseline, subject)
    reverse = paired_exact_match_statistics(subject, baseline)

    assert forward.baseline_pass_subject_fail_count == (
        reverse.baseline_fail_subject_pass_count
    )
    assert forward.baseline_fail_subject_pass_count == (
        reverse.baseline_pass_subject_fail_count
    )
    assert forward.effect_size_pp == -reverse.effect_size_pp
    assert forward.mcnemar_exact_two_sided_p_value == (
        reverse.mcnemar_exact_two_sided_p_value
    )
    assert forward.effect_size_confidence_interval.lower_pp == pytest.approx(
        -reverse.effect_size_confidence_interval.upper_pp
    )
    assert forward.effect_size_confidence_interval.upper_pp == pytest.approx(
        -reverse.effect_size_confidence_interval.lower_pp
    )


@pytest.mark.parametrize(
    ("baseline", "subject", "message"),
    [
        ([], [], "non-empty"),
        ([True], [True, False], "same number"),
        ("1", [True], "sequence"),
        ([True], b"1", "sequence"),
        ([0.5], [True], "binary"),
        ([math.nan], [True], "binary"),
        ([math.inf], [True], "binary"),
        ([10**10_000], [True], "binary"),
        ([None], [True], "binary"),
        ([True], ["pass"], "binary"),
    ],
)
def test_paired_statistics_fail_closed_on_malformed_inputs(
    baseline: object, subject: object, message: str
) -> None:
    with pytest.raises(PairedExactMatchError, match=message):
        paired_exact_match_statistics(baseline, subject)  # type: ignore[arg-type]


def test_paired_statistics_enforce_the_authenticated_record_limit() -> None:
    outcomes = [False] * (MAX_PAIRED_EXACT_MATCH_OUTCOMES + 1)

    with pytest.raises(PairedExactMatchError, match="10_000-pair limit"):
        paired_exact_match_statistics(outcomes, outcomes)
