from __future__ import annotations

import pytest

from invarlock.reporting.validation.report import compute_validation_flags


def _valid_inputs() -> dict[str, object]:
    return {
        "ppl": {"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0},
        "spectral": {"caps_applied": 0, "max_caps": 5},
        "rmt": {"stable": True},
        "invariants": {"status": "pass"},
    }


@pytest.mark.parametrize(
    ("section", "expected_flag"),
    [
        ("ppl", "preview_final_drift_acceptable"),
        ("spectral", "spectral_stable"),
        ("rmt", "rmt_stable"),
        ("invariants", "invariants_pass"),
    ],
)
def test_empty_required_guard_evidence_fails_closed(
    section: str,
    expected_flag: str,
) -> None:
    inputs = _valid_inputs()
    inputs[section] = {}

    flags = compute_validation_flags(**inputs)

    assert flags[expected_flag] is False


def test_empty_ppl_evidence_fails_both_ppl_gates() -> None:
    inputs = _valid_inputs()
    inputs["ppl"] = {}

    flags = compute_validation_flags(**inputs)

    assert flags["preview_final_drift_acceptable"] is False
    assert flags["primary_metric_acceptable"] is False


def test_explicit_valid_guard_evidence_remains_accepted() -> None:
    flags = compute_validation_flags(**_valid_inputs())

    assert flags["preview_final_drift_acceptable"] is True
    assert flags["primary_metric_acceptable"] is True
    assert flags["spectral_stable"] is True
    assert flags["rmt_stable"] is True
    assert flags["invariants_pass"] is True


def test_tiny_relax_does_not_turn_missing_drift_evidence_into_a_pass() -> None:
    inputs = _valid_inputs()
    inputs["ppl"] = {"ratio_vs_baseline": 1.0}

    flags = compute_validation_flags(**inputs, tiny_relax=True)

    assert flags["preview_final_drift_acceptable"] is False
