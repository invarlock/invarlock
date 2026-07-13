from __future__ import annotations

from pathlib import Path

import invarlock.reporting.verify_check_helpers_consistency as verify_helpers_mod


def test_validate_variance_enablement_rejects_missing_gate_provenance() -> None:
    report = {
        "resolved_policy": {"variance": {"min_effect_lognll": 0.0}},
        "variance": {
            "enabled": True,
            "predictive_gate": {
                "evaluated": True,
                "passed": False,
                "delta_ci": [-0.003, 0.0],
                "mean_delta": -0.001,
            },
            "ab_test": {"seed": 123, "windows_used": 2},
        },
    }

    errors = verify_helpers_mod._validate_variance_enablement(report)  # noqa: SLF001

    assert any("predictive_gate.passed" in error for error in errors)
    assert any("variance.predictive_gate.delta_ci" in error for error in errors)
    assert any("variance.ab_test.provenance" in error for error in errors)


def test_validate_variance_enablement_accepts_complete_enabled_evidence() -> None:
    report = {
        "resolved_policy": {"variance": {"min_effect_lognll": 0.001}},
        "variance": {
            "enabled": True,
            "predictive_gate": {
                "evaluated": True,
                "passed": True,
                "delta_ci": [-0.004, -0.002],
                "mean_delta": -0.003,
            },
            "ab_test": {
                "seed": 123,
                "windows_used": 2,
                "provenance": {"window_ids": [11, 12]},
            },
        },
    }

    assert (
        verify_helpers_mod._validate_variance_enablement(report) == []  # noqa: SLF001
    )


def test_validate_evaluation_report_payload_runs_variance_enablement_lint(
    tmp_path: Path,
) -> None:
    report = {
        "variance": {
            "enabled": True,
            "predictive_gate": {
                "passed": True,
                "delta_ci": [-0.003, 0.001],
                "mean_delta": -0.001,
            },
            "ab_test": {"seed": 123, "windows_used": 2},
        }
    }

    errors = verify_helpers_mod._validate_evaluation_report_payload(  # noqa: SLF001
        tmp_path / "evaluation.report.json",
        load_evaluation_report_fn=lambda _path: report,
        validate_report_fn=lambda _report: True,
        validate_report_schema_strict_fn=lambda _report: True,
        validate_primary_metric_fn=lambda _report: [],
        validate_pairing_fn=lambda _report: [],
        validate_counts_fn=lambda _report: [],
        validate_logspace_ci_identity_fn=lambda _report, profile=None: [],
        validate_drift_band_fn=lambda _report: [],
        validate_primary_metric_policy_fn=lambda _report, profile=None: [],
        apply_profile_lints_fn=lambda _report: [],
        validate_tokenizer_hash_fn=lambda _report: [],
        validate_measurement_contracts_fn=lambda _report, profile=None: [],
    )

    assert any("variance.predictive_gate.delta_ci" in error for error in errors)
    assert any("variance.ab_test.provenance" in error for error in errors)
