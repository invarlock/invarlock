"""Verifier-owned statistics for paired exact-match outcomes.

The functions in this module consume only paired binary outcomes.  They do not
trust aggregate values supplied by an evaluator, and they do not make an
acceptance decision.  Evidence verification can therefore replay the counts,
effect size, significance test, and confidence interval from authenticated
per-record outcomes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

type ExactMatchOutcome = bool | int | float

MAX_PAIRED_EXACT_MATCH_OUTCOMES: Final = 10_000
PAIRED_CONFIDENCE_LEVEL: Final = 0.95
PAIRED_CONFIDENCE_INTERVAL_METHOD: Final = "newcombe_hybrid_score_paired_v1"
_Z_975: Final = 1.959963984540054


class PairedExactMatchError(ValueError):
    """Raised when paired exact-match outcomes are malformed."""


@dataclass(frozen=True)
class PairedConfidenceInterval:
    """A deterministic interval for the subject-minus-baseline effect."""

    method: str
    confidence_level: float
    lower_pp: float
    upper_pp: float


@dataclass(frozen=True)
class PairedExactMatchStatistics:
    """Verifier-replayed statistics for one paired exact-match comparison."""

    pair_count: int
    baseline_pass_count: int
    subject_pass_count: int
    baseline_pass_subject_fail_count: int
    baseline_fail_subject_pass_count: int
    both_pass_count: int
    both_fail_count: int
    mcnemar_exact_two_sided_p_value: float
    effect_size_pp: float
    effect_size_confidence_interval: PairedConfidenceInterval


def _binary_outcomes(
    values: Sequence[ExactMatchOutcome], *, label: str
) -> tuple[bool, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise PairedExactMatchError(f"{label} outcomes must be a sequence")
    if not values:
        raise PairedExactMatchError(f"{label} outcomes must be non-empty")
    if len(values) > MAX_PAIRED_EXACT_MATCH_OUTCOMES:
        raise PairedExactMatchError(
            f"{label} outcomes exceed the "
            f"{MAX_PAIRED_EXACT_MATCH_OUTCOMES:_}-pair limit"
        )

    normalized: list[bool] = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            normalized.append(value)
        elif isinstance(value, int):
            if value not in {0, 1}:
                raise PairedExactMatchError(
                    f"{label} outcome {index} must be binary (zero or one)"
                )
            normalized.append(value == 1)
        elif isinstance(value, float):
            if not math.isfinite(value) or value not in {0.0, 1.0}:
                raise PairedExactMatchError(
                    f"{label} outcome {index} must be binary (zero or one)"
                )
            normalized.append(value == 1.0)
        else:
            raise PairedExactMatchError(
                f"{label} outcome {index} must be binary (zero or one)"
            )
    return tuple(normalized)


def _mcnemar_exact_two_sided(
    *, baseline_pass_subject_fail: int, baseline_fail_subject_pass: int
) -> float:
    discordant = baseline_pass_subject_fail + baseline_fail_subject_pass
    if discordant == 0:
        return 1.0

    tail = min(baseline_pass_subject_fail, baseline_fail_subject_pass)
    coefficient = 1
    cumulative = 1
    for successes in range(1, tail + 1):
        coefficient = coefficient * (discordant - successes + 1) // successes
        cumulative += coefficient
    probability = Fraction(2 * cumulative, 1 << discordant)
    return float(min(probability, Fraction(1, 1)))


def _wilson_score_interval(*, successes: int, count: int) -> tuple[float, float]:
    proportion = successes / count
    z_squared = _Z_975 * _Z_975
    denominator = 1.0 + z_squared / count
    center = (proportion + z_squared / (2.0 * count)) / denominator
    radius = (
        _Z_975
        * math.sqrt(
            proportion * (1.0 - proportion) / count + z_squared / (4.0 * count * count)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _paired_effect_confidence_interval(
    *,
    pair_count: int,
    both_pass_count: int,
    baseline_pass_subject_fail_count: int,
    baseline_fail_subject_pass_count: int,
) -> PairedConfidenceInterval:
    baseline_pass_count = both_pass_count + baseline_pass_subject_fail_count
    subject_pass_count = both_pass_count + baseline_fail_subject_pass_count
    baseline_rate = baseline_pass_count / pair_count
    subject_rate = subject_pass_count / pair_count
    effect = subject_rate - baseline_rate
    if baseline_pass_subject_fail_count == 0 and baseline_fail_subject_pass_count == 0:
        return PairedConfidenceInterval(
            method=PAIRED_CONFIDENCE_INTERVAL_METHOD,
            confidence_level=PAIRED_CONFIDENCE_LEVEL,
            lower_pp=0.0,
            upper_pp=0.0,
        )

    subject_lower, subject_upper = _wilson_score_interval(
        successes=subject_pass_count, count=pair_count
    )
    baseline_lower, baseline_upper = _wilson_score_interval(
        successes=baseline_pass_count, count=pair_count
    )

    denominator = math.sqrt(
        subject_rate * (1.0 - subject_rate) * baseline_rate * (1.0 - baseline_rate)
    )
    if denominator == 0.0:
        correlation = (
            1.0
            if baseline_pass_subject_fail_count == 0
            and baseline_fail_subject_pass_count == 0
            else 0.0
        )
    else:
        joint_pass_rate = both_pass_count / pair_count
        correlation = (joint_pass_rate - subject_rate * baseline_rate) / denominator
        correlation = max(-1.0, min(1.0, correlation))

    lower_subject_distance = subject_rate - subject_lower
    upper_baseline_distance = baseline_upper - baseline_rate
    lower_radicand = (
        lower_subject_distance**2
        + upper_baseline_distance**2
        - 2.0 * correlation * lower_subject_distance * upper_baseline_distance
    )
    upper_subject_distance = subject_upper - subject_rate
    lower_baseline_distance = baseline_rate - baseline_lower
    upper_radicand = (
        upper_subject_distance**2
        + lower_baseline_distance**2
        - 2.0 * correlation * upper_subject_distance * lower_baseline_distance
    )
    lower = max(-1.0, effect - math.sqrt(max(0.0, lower_radicand)))
    upper = min(1.0, effect + math.sqrt(max(0.0, upper_radicand)))
    return PairedConfidenceInterval(
        method=PAIRED_CONFIDENCE_INTERVAL_METHOD,
        confidence_level=PAIRED_CONFIDENCE_LEVEL,
        lower_pp=lower * 100.0,
        upper_pp=upper * 100.0,
    )


def paired_exact_match_statistics(
    baseline_outcomes: Sequence[ExactMatchOutcome],
    subject_outcomes: Sequence[ExactMatchOutcome],
) -> PairedExactMatchStatistics:
    """Replay paired exact-match statistics from binary per-record outcomes.

    ``effect_size_pp`` and its interval use subject minus baseline orientation,
    so positive values indicate improvement and negative values indicate
    regression.  The exact two-sided McNemar p-value is conditional on the
    number of discordant pairs.
    """

    baseline = _binary_outcomes(baseline_outcomes, label="baseline")
    subject = _binary_outcomes(subject_outcomes, label="subject")
    if len(baseline) != len(subject):
        raise PairedExactMatchError(
            "baseline and subject outcomes must contain the same number of pairs"
        )

    both_pass = 0
    both_fail = 0
    baseline_pass_subject_fail = 0
    baseline_fail_subject_pass = 0
    for baseline_passed, subject_passed in zip(baseline, subject, strict=True):
        if baseline_passed and subject_passed:
            both_pass += 1
        elif baseline_passed:
            baseline_pass_subject_fail += 1
        elif subject_passed:
            baseline_fail_subject_pass += 1
        else:
            both_fail += 1

    pair_count = len(baseline)
    baseline_pass_count = both_pass + baseline_pass_subject_fail
    subject_pass_count = both_pass + baseline_fail_subject_pass
    effect_size_pp = (
        (baseline_fail_subject_pass - baseline_pass_subject_fail) / pair_count * 100.0
    )
    if effect_size_pp == 0.0:
        effect_size_pp = 0.0

    return PairedExactMatchStatistics(
        pair_count=pair_count,
        baseline_pass_count=baseline_pass_count,
        subject_pass_count=subject_pass_count,
        baseline_pass_subject_fail_count=baseline_pass_subject_fail,
        baseline_fail_subject_pass_count=baseline_fail_subject_pass,
        both_pass_count=both_pass,
        both_fail_count=both_fail,
        mcnemar_exact_two_sided_p_value=_mcnemar_exact_two_sided(
            baseline_pass_subject_fail=baseline_pass_subject_fail,
            baseline_fail_subject_pass=baseline_fail_subject_pass,
        ),
        effect_size_pp=effect_size_pp,
        effect_size_confidence_interval=_paired_effect_confidence_interval(
            pair_count=pair_count,
            both_pass_count=both_pass,
            baseline_pass_subject_fail_count=baseline_pass_subject_fail,
            baseline_fail_subject_pass_count=baseline_fail_subject_pass,
        ),
    )


__all__ = [
    "MAX_PAIRED_EXACT_MATCH_OUTCOMES",
    "PAIRED_CONFIDENCE_INTERVAL_METHOD",
    "PAIRED_CONFIDENCE_LEVEL",
    "PairedConfidenceInterval",
    "PairedExactMatchError",
    "PairedExactMatchStatistics",
    "paired_exact_match_statistics",
]
