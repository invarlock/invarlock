from __future__ import annotations

import math

from invarlock.reporting import guard_warnings


def test_guard_warning_scalar_parsers_reject_boolean_and_nonfinite_values() -> None:
    assert guard_warnings._finite_float(True) is None
    assert guard_warnings._finite_float(math.inf) is None
    assert guard_warnings._finite_int(False) is None
    assert guard_warnings._finite_int(-1) is None


def test_guard_section_prefers_post_edit_and_fills_empty_fields() -> None:
    payload = {
        "invariants": {"violations": []},
        "metrics": {"invariants": {"summary": {"warnings": 1}}},
        "guards": [
            {"name": "invariants", "metrics": {"warning_violations": 1}},
            {
                "name": "invariants_post",
                "stage": "post",
                "metrics": {"warning_violations": 2},
                "violations": [{"name": "post"}],
            },
        ],
    }
    section = guard_warnings._guard_section(payload, "invariants")
    assert section["warning_violations"] == 2
    assert section["violations"] == [{"name": "post"}]


def test_warning_builder_omits_empty_coordinates() -> None:
    warning = guard_warnings._warning(
        guard="rmt",
        kind="movement",
        message="message",
        policy_gate="unknown",
        family="",
        module="",
        baseline={},
        subject={},
    )
    assert "family" not in warning
    assert "module" not in warning
    assert warning["baseline"] == {}
    assert warning["subject"] == {}


def test_spectral_module_extraction_filters_noise_and_uses_family_defaults() -> None:
    spectral = {
        "family_caps": {"attention": {"kappa": 2.0}},
        "violations": [
            {},
            {"module": "ignored", "type": "diagnostic"},
            {"module": "m1", "family": "attention", "z": 3.0},
            {"module": "m2", "z_score": 4.0, "kappa": 3.0},
        ],
        "top_z_scores": {
            "attention": [
                {"module": "inside", "z": 1.0},
                {"module": "outside", "z": -3.0},
            ],
            "missing-policy": [{"module": "ignored", "z": 9.0}],
        },
    }
    modules = guard_warnings._spectral_capped_modules(spectral)
    assert ("attention", "m1") in modules
    assert ("unknown", "m2") in modules
    assert ("attention", "outside") in modules
    assert ("attention", "inside") not in modules
    assert guard_warnings._family_kappa(spectral, None) is None


def test_spectral_count_and_deadband_warnings_are_baseline_relative() -> None:
    count_warning = guard_warnings._spectral_warnings(
        subject={"spectral": {"caps_applied": 2}},
        baseline={"spectral": {"caps_applied": 1}},
        validation={"spectral_stable": True},
    )
    assert count_warning[0]["kind"] == "cap_count_increase"
    assert count_warning[0]["message"].startswith("Policy passes")

    subject = {
        "spectral": {
            "deadband": -1,
            "violations": [
                {"module": "same", "family": "attention", "z_score": 3.0},
                {"module": "missing-z", "family": "attention"},
            ],
        }
    }
    baseline = {
        "spectral": {
            "violations": [
                {"module": "same", "family": "attention", "z_score": 2.0},
                {"module": "missing-z", "family": "attention", "z_score": 2.0},
            ]
        }
    }
    warnings = guard_warnings._spectral_warnings(
        subject=subject, baseline=baseline, validation={"spectral_stable": False}
    )
    assert [warning["kind"] for warning in warnings] == [
        "capped_module_z_score_increase"
    ]


def test_rmt_warning_uses_family_when_module_is_absent() -> None:
    warnings = guard_warnings._rmt_warnings(
        subject={"rmt": {"epsilon_violations": [{"family": "mlp"}]}},
        baseline={},
        validation={"rmt_stable": True},
    )
    assert warnings[0]["kind"] == "new_epsilon_violation"
    assert warnings[0]["family"] == "mlp"
    assert "module" not in warnings[0]


def test_variance_and_invariant_warning_signals_require_new_subject_evidence() -> None:
    subject = {
        "variance": {
            "enabled": True,
            "predictive_gate": {"delta_ci": [0.1, 0.2]},
        },
        "invariants": {"summary": {"warning_violations": 2}},
    }
    warnings = guard_warnings.build_guard_warnings(
        subject=subject,
        baseline={"invariants": {"warnings": 1}},
        validation={"invariants_pass": True},
    )["warnings"]
    assert {warning["kind"] for warning in warnings} == {
        "new_predictive_signal",
        "warning_count_increase",
    }
    assert all(warning["message"].startswith("Policy passes") for warning in warnings)

    assert (
        guard_warnings._variance_warnings(
            subject=subject,
            baseline={"variance": {"enabled": True, "ab_test": {"passed": True}}},
        )
        == []
    )
