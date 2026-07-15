from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy

import pytest

from invarlock.eval.guard_metric_impact import (
    arm_facts_match_measurements,
    build_guard_metric_bare_report,
    compute_guard_metric_impact,
    degradation_within_limit,
    extract_guard_metric_arm_facts,
    guard_metric_impact_payload_errors,
    guard_metric_schedule_digest,
    metric_value_from_arm_facts,
)


def _digest(ids: list[int]) -> str:
    encoded = json.dumps(ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_ppl_contract_replays_weighted_arm_facts_and_relative_increase() -> None:
    bare_report = {
        "evaluation_windows": {
            "final": {
                "window_ids": [11, 12],
                "logloss": [math.log(2.0), math.log(4.0)],
                "token_counts": [3, 1],
            }
        }
    }
    guarded_report = {
        "evaluation_windows": {
            "final": {
                "window_ids": [11, 12],
                "logloss": [math.log(2.2), math.log(4.4)],
                "token_counts": [3, 1],
            }
        }
    }

    bare_facts = extract_guard_metric_arm_facts(bare_report, "ppl_causal")
    guarded_facts = extract_guard_metric_arm_facts(guarded_report, "ppl_causal")
    assert bare_facts == {
        "weighted_logloss_sum": pytest.approx(3 * math.log(2.0) + math.log(4.0)),
        "token_count": 4,
        "example_ids_digest": _digest([11, 12]),
    }
    bare_value = metric_value_from_arm_facts("ppl_causal", bare_facts)
    guarded_value = metric_value_from_arm_facts("ppl_causal", guarded_facts)
    measurement = compute_guard_metric_impact("ppl_causal", bare_value, guarded_value)

    assert measurement is not None
    assert measurement.direction == "lower"
    assert measurement.degradation_basis == "relative_increase"
    assert measurement.degradation == pytest.approx(0.1)
    assert measurement.display_value == pytest.approx(10.0)
    assert measurement.display_unit == "percent"
    assert arm_facts_match_measurements(
        "ppl_causal", bare_facts, guarded_facts, bare_value, guarded_value
    )


def test_accuracy_contract_replays_counts_and_absolute_drop() -> None:
    bare_report = {
        "metrics": {"classification": {"final": {"correct_total": 80, "total": 100}}},
        "evaluation_windows": {"final": {"example_ids": [1, 2, 3]}},
    }
    guarded_report = {
        "metrics": {"classification": {"final": {"correct_total": 79, "total": 100}}},
        "evaluation_windows": {"final": {"example_ids": [1, 2, 3]}},
    }

    bare_facts = extract_guard_metric_arm_facts(bare_report, "accuracy")
    guarded_facts = extract_guard_metric_arm_facts(guarded_report, "accuracy")
    measurement = compute_guard_metric_impact("accuracy", 0.8, 0.79)

    assert bare_facts == {
        "correct": 80,
        "total": 100,
        "example_ids_digest": _digest([1, 2, 3]),
    }
    assert measurement is not None
    assert measurement.direction == "higher"
    assert measurement.degradation_basis == "absolute_drop"
    assert measurement.degradation == pytest.approx(0.01)
    assert measurement.display_value == pytest.approx(1.0)
    assert measurement.display_unit == "percentage_points"
    assert arm_facts_match_measurements(
        "accuracy", bare_facts, guarded_facts, 0.8, 0.79
    )


def test_arm_facts_reject_mismatched_identity_or_copied_metric_value() -> None:
    bare = {"correct": 8, "total": 10, "example_ids_digest": "a" * 64}
    guarded = {"correct": 8, "total": 10, "example_ids_digest": "b" * 64}

    assert not arm_facts_match_measurements("accuracy", bare, guarded, 0.8, 0.8)
    guarded["example_ids_digest"] = bare["example_ids_digest"]
    assert not arm_facts_match_measurements("accuracy", bare, guarded, 0.8, 0.7)


def test_accuracy_improvement_is_negative_degradation() -> None:
    measurement = compute_guard_metric_impact("accuracy", 0.8, 0.82)

    assert measurement is not None
    assert measurement.degradation == pytest.approx(-0.02)
    assert measurement.display_value == pytest.approx(-2.0)


@pytest.mark.parametrize(
    ("metric_kind", "bare", "guarded"),
    [("ppl_causal", 100.0, 101.0), ("accuracy", 0.8, 0.79)],
)
def test_degradation_limit_accepts_exact_decimal_boundary(
    metric_kind: str,
    bare: float,
    guarded: float,
) -> None:
    measurement = compute_guard_metric_impact(metric_kind, bare, guarded)

    assert measurement is not None
    assert degradation_within_limit(
        degradation=measurement.degradation,
        degradation_limit=0.01,
    )


@pytest.mark.parametrize(
    ("metric_kind", "bare", "guarded"),
    [("ppl_causal", 100.0, 101.0001), ("accuracy", 0.8, 0.789999)],
)
def test_degradation_limit_rejects_values_just_over_boundary(
    metric_kind: str,
    bare: float,
    guarded: float,
) -> None:
    measurement = compute_guard_metric_impact(metric_kind, bare, guarded)

    assert measurement is not None
    assert not degradation_within_limit(
        degradation=measurement.degradation,
        degradation_limit=0.01,
    )


def test_payload_rejects_self_consistent_guarded_forgery_unbound_to_report() -> None:
    subject = {
        "primary_metric": {"kind": "ppl_causal", "final": 2.0},
        "evaluation_windows": {
            "final": {
                "window_ids": [7],
                "logloss": [math.log(2.0)],
                "token_counts": [1],
            }
        },
    }
    bare_report = {
        "primary_metric": {"kind": "ppl_causal", "final": 2.0},
        "final": {"window_ids": [7], "logloss": [math.log(2.0)], "token_counts": [1]},
    }
    bare_facts = extract_guard_metric_arm_facts(bare_report, "ppl_causal")
    forged_guarded = {
        "weighted_logloss_sum": math.log(3.0),
        "token_count": 1,
        "example_ids_digest": _digest([7]),
    }
    payload = {
        "metric_kind": "ppl_causal",
        "direction": "lower",
        "degradation_basis": "relative_increase",
        "bare_value": 2.0,
        "guarded_value": 3.0,
        "bare_facts": bare_facts,
        "guarded_facts": forged_guarded,
        "degradation": 0.5,
        "degradation_limit": 1.0,
        "display_value": 50.0,
        "display_unit": "percent",
        "evaluated": True,
        "passed": True,
        "checks": {
            "metric_kind_matches": True,
            "measurements_valid": True,
            "guard_metric_impact": True,
            "arm_facts_replay": True,
        },
        "schedule_digest": guard_metric_schedule_digest(subject, "ppl_causal"),
        "bare_report": bare_report,
    }

    errors = guard_metric_impact_payload_errors(
        payload,
        subject_report=subject,
        require_bare_report=True,
    )

    assert "guard metric impact guarded arm is not bound to the report" in errors


def test_payload_rejects_false_arm_replay_check_even_when_passed_is_true() -> None:
    bare_report = {
        "primary_metric": {"kind": "accuracy", "final": 0.8},
        "final": {"correct_total": 8, "total": 10},
    }
    facts = {"correct": 8, "total": 10}
    payload = {
        "metric_kind": "accuracy",
        "direction": "higher",
        "degradation_basis": "absolute_drop",
        "bare_value": 0.8,
        "guarded_value": 0.8,
        "bare_facts": facts,
        "guarded_facts": facts,
        "degradation": 0.0,
        "degradation_limit": 0.01,
        "display_value": 0.0,
        "display_unit": "percentage_points",
        "evaluated": True,
        "passed": True,
        "checks": {"arm_facts_replay": False},
        "bare_report": bare_report,
    }

    assert "guard metric impact checks are incomplete or failed" in (
        guard_metric_impact_payload_errors(payload, require_bare_report=True)
    )


def test_bare_report_envelope_round_trips_through_payload_replay() -> None:
    report = {
        "status": "success",
        "metrics": {
            "primary_metric": {"kind": "accuracy", "final": 0.8},
            "classification": {"final": {"correct_total": 8, "total": 10}},
        },
        "evaluation_windows": {"final": {"example_ids": list(range(10))}},
    }
    envelope = build_guard_metric_bare_report(report, "accuracy")
    facts = extract_guard_metric_arm_facts(report, "accuracy")
    payload = {
        "metric_kind": "accuracy",
        "direction": "higher",
        "degradation_basis": "absolute_drop",
        "bare_value": 0.8,
        "guarded_value": 0.8,
        "bare_facts": facts,
        "guarded_facts": facts,
        "bare_report": envelope,
        "degradation": 0.0,
        "degradation_limit": 0.01,
        "display_value": 0.0,
        "display_unit": "percentage_points",
        "evaluated": True,
        "passed": True,
        "checks": {
            "metric_kind_matches": True,
            "measurements_valid": True,
            "guard_metric_impact": True,
            "arm_facts_replay": True,
        },
        "schedule_digest": guard_metric_schedule_digest(report, "accuracy"),
    }

    assert envelope is not None
    assert envelope["status"] == "success"
    assert (
        guard_metric_impact_payload_errors(
            payload,
            subject_report=report,
            require_bare_report=True,
        )
        == []
    )
    payload["unexpected_field"] = "untrusted"
    assert (
        "guard metric impact payload contains unsupported fields: unexpected_field"
        in guard_metric_impact_payload_errors(
            payload,
            subject_report=report,
            require_bare_report=True,
        )
    )


@pytest.mark.parametrize("status", ["failed", None])
def test_payload_rejects_missing_or_unsuccessful_bare_execution_status(
    status: str | None,
) -> None:
    report = {
        "metrics": {
            "primary_metric": {"kind": "accuracy", "final": 0.8},
            "classification": {"final": {"correct_total": 8, "total": 10}},
        },
        "evaluation_windows": {"final": {"example_ids": list(range(10))}},
    }
    if status is not None:
        report["status"] = status
    facts = extract_guard_metric_arm_facts(report, "accuracy")
    payload = {
        "metric_kind": "accuracy",
        "direction": "higher",
        "degradation_basis": "absolute_drop",
        "bare_value": 0.8,
        "guarded_value": 0.8,
        "bare_facts": facts,
        "guarded_facts": facts,
        "bare_report": build_guard_metric_bare_report(report, "accuracy"),
        "degradation": 0.0,
        "degradation_limit": 0.01,
        "display_value": 0.0,
        "display_unit": "percentage_points",
        "evaluated": True,
        "passed": True,
        "checks": {
            "metric_kind_matches": True,
            "measurements_valid": True,
            "guard_metric_impact": True,
            "arm_facts_replay": True,
        },
    }

    assert (
        "guard metric impact bare arm does not retain a successful execution status"
        in (guard_metric_impact_payload_errors(payload, require_bare_report=True))
    )


@pytest.mark.parametrize(
    "field",
    ["direction", "degradation_basis", "display_unit", "degradation", "display_value"],
)
def test_payload_rejects_semantic_field_tampering(field: str) -> None:
    report = {
        "primary_metric": {"kind": "accuracy", "final": 0.8},
        "metrics": {"classification": {"final": {"correct_total": 8, "total": 10}}},
        "evaluation_windows": {"final": {"example_ids": list(range(10))}},
    }
    facts = extract_guard_metric_arm_facts(report, "accuracy")
    payload = {
        "metric_kind": "accuracy",
        "direction": "higher",
        "degradation_basis": "absolute_drop",
        "bare_value": 0.8,
        "guarded_value": 0.8,
        "bare_facts": facts,
        "guarded_facts": facts,
        "bare_report": build_guard_metric_bare_report(report, "accuracy"),
        "degradation": 0.0,
        "degradation_limit": 0.01,
        "display_value": 0.0,
        "display_unit": "percentage_points",
        "evaluated": True,
        "passed": True,
        "checks": {
            "metric_kind_matches": True,
            "measurements_valid": True,
            "guard_metric_impact": True,
            "arm_facts_replay": True,
        },
        "schedule_digest": guard_metric_schedule_digest(report, "accuracy"),
    }
    tampered = deepcopy(payload)
    tampered[field] = (
        "wrong" if field in {"direction", "degradation_basis", "display_unit"} else 1.0
    )

    assert guard_metric_impact_payload_errors(
        tampered, subject_report=report, require_bare_report=True
    )


@pytest.mark.parametrize(
    "mutation", ["deleted", "kind", "count_key", "length", "ordered_ids"]
)
def test_payload_rejects_bare_envelope_mutations(mutation: str) -> None:
    report = {
        "primary_metric": {"kind": "ppl_causal", "final": 2.0},
        "evaluation_windows": {
            "final": {
                "window_ids": [1, 2],
                "logloss": [math.log(2.0), math.log(2.0)],
                "token_counts": [1, 1],
            }
        },
    }
    facts = extract_guard_metric_arm_facts(report, "ppl_causal")
    payload = {
        "metric_kind": "ppl_causal",
        "direction": "lower",
        "degradation_basis": "relative_increase",
        "bare_value": 2.0,
        "guarded_value": 2.0,
        "bare_facts": facts,
        "guarded_facts": facts,
        "bare_report": build_guard_metric_bare_report(report, "ppl_causal"),
        "degradation": 0.0,
        "degradation_limit": 0.01,
        "display_value": 0.0,
        "display_unit": "percent",
        "evaluated": True,
        "passed": True,
        "checks": {
            "metric_kind_matches": True,
            "measurements_valid": True,
            "guard_metric_impact": True,
            "arm_facts_replay": True,
        },
        "schedule_digest": guard_metric_schedule_digest(report, "ppl_causal"),
    }
    tampered = deepcopy(payload)
    if mutation == "deleted":
        tampered.pop("bare_report")
    elif mutation == "kind":
        tampered["bare_report"]["primary_metric"]["kind"] = "accuracy"
    elif mutation == "count_key":
        counts = tampered["bare_report"]["final"].pop("token_counts")
        tampered["bare_report"]["final"]["masked_token_counts"] = counts
    elif mutation == "length":
        tampered["bare_report"]["final"]["token_counts"].pop()
    else:
        tampered["bare_report"]["final"]["window_ids"].reverse()

    assert "guard metric impact bare arm is not bound to retained evidence" in (
        guard_metric_impact_payload_errors(tampered, require_bare_report=True)
    )


def test_bare_envelope_preserves_top_level_final_window_ids() -> None:
    report = {
        "status": "success",
        "primary_metric": {"kind": "ppl_causal", "final": 2.0},
        "final": {
            "window_ids": [1, 2],
            "logloss": [math.log(2.0), math.log(2.0)],
            "token_counts": [1, 1],
        },
    }
    facts = extract_guard_metric_arm_facts(report, "ppl_causal")
    envelope = build_guard_metric_bare_report(report, "ppl_causal")

    assert envelope is not None
    assert envelope["final"]["window_ids"] == [1, 2]
    payload = {
        "metric_kind": "ppl_causal",
        "direction": "lower",
        "degradation_basis": "relative_increase",
        "bare_value": 2.0,
        "guarded_value": 2.0,
        "bare_facts": facts,
        "guarded_facts": facts,
        "bare_report": envelope,
        "degradation": 0.0,
        "degradation_limit": 0.01,
        "display_value": 0.0,
        "display_unit": "percent",
        "evaluated": True,
        "passed": True,
        "checks": {
            "metric_kind_matches": True,
            "measurements_valid": True,
            "guard_metric_impact": True,
            "arm_facts_replay": True,
        },
        "schedule_digest": guard_metric_schedule_digest(report, "ppl_causal"),
    }

    assert guard_metric_impact_payload_errors(payload, require_bare_report=True) == []


@pytest.mark.parametrize("metric_kind", ["ppl_causal", "ppl_mlm", "ppl_seq2seq"])
def test_all_ppl_kinds_treat_improvement_as_negative_degradation(
    metric_kind: str,
) -> None:
    measurement = compute_guard_metric_impact(metric_kind, 10.0, 9.0)

    assert measurement is not None
    assert measurement.degradation == pytest.approx(-0.1)
    assert measurement.display_value == pytest.approx(-10.0)


@pytest.mark.parametrize(
    ("kind", "bare", "guarded"),
    [
        ("ppl_causal", True, 1.0),
        ("ppl_causal", 0.0, 1.0),
        ("ppl_causal", float("nan"), 1.0),
        ("accuracy", -0.1, 0.5),
        ("accuracy", 0.5, 1.1),
        ("accuracy", True, 0.5),
    ],
)
def test_metric_contract_rejects_invalid_values(
    kind: str, bare: object, guarded: object
) -> None:
    assert compute_guard_metric_impact(kind, bare, guarded) is None
