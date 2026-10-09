from __future__ import annotations

import itertools

import pytest

import invarlock.exact_match_sensitivity as sensitivity_module
from invarlock.evidence_pack_contract import (
    PAIRED_RECORDS_FORMAT,
    build_comparison_report,
)
from invarlock.exact_match_sensitivity import exact_match_sensitivity


def verdict(baseline, subject, policy):
    return build_comparison_report(
        comparison_id="oracle",
        paired_records={
            "format": PAIRED_RECORDS_FORMAT,
            "metric": "exact_match",
            "schedule_sha256": "0" * 64,
            "records": [
                {"baseline": {"score": b}, "subject": {"score": s}}
                for b, s in zip(baseline, subject, strict=True)
            ],
        },
        policy={"resolved_policy": {"metrics": {"exact_match": policy}}},
        policy_digest="sha256:" + "1" * 64,
    )["verdict"]


@pytest.mark.parametrize(
    "policy",
    [
        {"delta_min_pp": -20},
        {
            "delta_min_pp": -20,
            "minimum_record_count": 4,
            "maximum_interval_width_pp": 80,
        },
        {"delta_min_pp": -20, "minimum_side_accuracy": 0.5},
        {
            "delta_min_pp": -20,
            "minimum_record_count": 4,
            "maximum_interval_width_pp": 80,
            "minimum_side_accuracy": 0.5,
        },
    ],
)
def test_search_matches_exhaustive_record_edits_and_replays_witness(policy):
    for n in range(1, 6):
        for b in range(n + 1):
            baseline = [1] * b + [0] * (n - b)
            outcomes = list(itertools.product([0, 1], repeat=n))
            expected = {s: verdict(baseline, s, policy) for s in outcomes}
            # One representative of every possible 2x2 table; the oracle still
            # enumerates individual records rather than trusting table arithmetic.
            for a in range(b + 1):
                for c in range(n - b + 1):
                    subject = (1,) * a + (0,) * (b - a) + (1,) * c + (0,) * (n - b - c)
                    distance = min(
                        (
                            sum(x != y for x, y in zip(subject, other, strict=True))
                            for other in outcomes
                            if expected[other] != expected[subject]
                        ),
                        default=None,
                    )
                    result = exact_match_sensitivity(
                        baseline, subject, policy=policy, max_changes=n
                    )
                    assert result["original_verdict"] == expected[subject]
                    assert result["minimum_changes"] == distance
                    if distance is not None:
                        assert result["status"] == "exact"
                        witness = list(subject)
                        for i in result["witness"]["subject_flip_indices"]:
                            witness[i] = 1 - witness[i]
                        assert (
                            len(result["witness"]["subject_flip_indices"]) == distance
                        )
                        assert verdict(baseline, witness, policy) != expected[subject]
                    else:
                        assert result["status"] == "no_flip_possible"


def test_known_four_flip_example_and_unchanged_inputs():
    b = [1] * 75 + [0] * 25
    s = [1] * 72 + [0] * 3 + [1] * 3 + [0] * 22
    before = s.copy()
    result = exact_match_sensitivity(b, s, policy={"delta_min_pp": -10})
    assert result["minimum_changes"] == 4
    assert result["original_verdict"] == "pass"
    assert s == before


def test_partial_ring_reports_only_completed_radius():
    result = exact_match_sensitivity(
        [1] * 50 + [0] * 50,
        [1] * 50 + [0] * 50,
        policy={"delta_min_pp": -100},
        max_changes=8,
        max_states=1,
    )
    assert result["status"] == "lower_bound"
    assert result["checked_through_changes"] == 0
    assert result["states_examined"] == 1
    assert result["minimum_changes"] is None


def test_radius_limit_is_not_exact_distance():
    result = exact_match_sensitivity(
        [1] * 7500 + [0] * 2500,
        [1] * 7500 + [0] * 2500,
        policy={"delta_min_pp": -10},
        max_changes=8,
    )
    assert result["status"] == "lower_bound"
    assert result["checked_through_changes"] == 8
    assert result["states_examined"] <= 2048


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_changes": True},
        {"max_changes": 33},
        {"max_states": 0},
        {"max_states": 5001},
    ],
)
def test_invalid_search_bounds(kwargs):
    with pytest.raises(ValueError):
        exact_match_sensitivity([1], [1], policy={"delta_min_pp": -10}, **kwargs)


@pytest.mark.parametrize(
    "b,s,policy",
    [
        ([1], [2], {"delta_min_pp": -10}),
        ([1], [], {"delta_min_pp": -10}),
        ([1], [1, 0], {"delta_min_pp": -10}),
        ([1] * 10001, [1] * 10001, {"delta_min_pp": -10}),
        ([1], [1], {"delta_min_pp": -10, "minimum_record_count": 1}),
        ([1], [1], {"delta_min_pp": float("nan")}),
    ],
)
def test_rejects_invalid_outcomes_or_native_policy(b, s, policy):
    with pytest.raises(ValueError):
        exact_match_sensitivity(b, s, policy=policy)


def test_improving_subject_can_fail_the_width_gate():
    b = [1] + [0] * 19
    s = [0, 1] + [0] * 18
    policy = {
        "delta_min_pp": -100.0,
        "minimum_record_count": 1,
        "maximum_interval_width_pp": 39.30266123732387,
    }
    assert verdict(b, s, policy) == "pass"
    improved = s.copy()
    improved[2] = 1
    assert verdict(b, improved, policy) == "fail"
    result = exact_match_sensitivity(b, s, policy=policy)
    assert result["minimum_changes"] == 1
    assert result["witness"]["gates"]["effect_floor"] is True
    assert (
        result["witness"]["sample_qualification"]["interval_width"]["passed"] is False
    )


@pytest.mark.parametrize("divergent_call", [1, 2], ids=["original", "witness"])
def test_rejects_drift_from_native_report_arithmetic(monkeypatch, divergent_call):
    # Future report changes must not silently leave a valid-looking advisory
    # whose starting verdict or purported opposite-verdict witness is stale.
    native_builder = sensitivity_module.build_comparison_report
    calls = 0

    def divergent_report(**kwargs):
        nonlocal calls
        calls += 1
        report = native_builder(**kwargs)
        if calls == divergent_call:
            report["verdict"] = "fail" if report["verdict"] == "pass" else "pass"
        return report

    monkeypatch.setattr(sensitivity_module, "build_comparison_report", divergent_report)
    message = (
        "arithmetic disagrees with the native report"
        if divergent_call == 1
        else "witness failed native replay"
    )
    with pytest.raises(RuntimeError, match=message):
        exact_match_sensitivity(
            [1] * 75 + [0] * 25,
            [1] * 72 + [0] * 3 + [1] * 3 + [0] * 22,
            policy={"delta_min_pp": -10},
        )
    assert calls == divergent_call
