from __future__ import annotations

import math

import invarlock.calibration as calibration


def test_summarize_null_sweep_reports_covers_bad_multiple_testing_and_tiny_alpha() -> (
    None
):
    report = {
        "guards": [
            {
                "name": "spectral",
                "metrics": {
                    "family_z_summary": {"ffn": {"max": float("inf")}},
                    "multiple_testing": {
                        "method": "bh",
                        "alpha": object(),
                        "m": object(),
                    },
                    "multiple_testing_selection": {"family_pvalues": {"ffn": 0.25}},
                },
                "violations": ["warn"],
            }
        ]
    }

    summary = calibration.summarize_null_sweep_reports(
        [report],
        tier="balanced",
        target_any_warning_rate=0.0,
    )

    assert summary["recommendations"]["multiple_testing"]["method"] == "bh"
    assert summary["recommendations"]["family_caps"] == {}

    tiny_alpha_report = {
        "guards": [
            {
                "name": "spectral",
                "metrics": {
                    "family_z_summary": {"attn": {"max": 1.5}},
                    "multiple_testing": {"method": "bh", "alpha": 1e-7, "m": 4},
                    "multiple_testing_selection": {"family_pvalues": {"attn": 1e-4}},
                },
                "violations": ["warn"],
            }
        ]
    }

    tiny_alpha_summary = calibration.summarize_null_sweep_reports(
        [tiny_alpha_report],
        tier="balanced",
        target_any_warning_rate=0.0,
    )

    assert tiny_alpha_summary["recommendations"]["multiple_testing"]["alpha"] == 1e-6
    assert math.isclose(
        tiny_alpha_summary["recommendations"]["family_caps"]["attn"],
        round(1.5 * 1.05, 3),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_summarize_null_sweep_reports_skips_invalid_reports_and_non_finite_caps() -> (
    None
):
    summary = calibration.summarize_null_sweep_reports(
        [
            "bad",
            {},
            {"guards": "bad"},
            {
                "guards": [
                    {
                        "name": "spectral",
                        "metrics": {
                            "family_z_summary": {"ffn": {"max": float("nan")}},
                            "family_z_quantiles": {"attn": {"max": float("inf")}},
                            "multiple_testing_selection": {
                                "family_pvalues": {"ffn": "bad", "attn": 2.0},
                                "family_violation_counts": {"ffn": "bad"},
                            },
                            "caps_applied": "bad",
                        },
                        "violations": [],
                    }
                ]
            },
        ],
        tier="balanced",
        target_any_warning_rate=2.0,
    )

    assert summary["recommendations"]["family_caps"] == {}
    assert summary["recommendations"]["multiple_testing"]["alpha"] == 0.05
    assert summary["observed"]["candidate_violations_by_family_total"] == {}


def test_summarize_null_sweep_reports_covers_inner_multiple_testing_coercion_failures(
    monkeypatch,
) -> None:
    report = {
        "guards": [
            {
                "name": "spectral",
                "metrics": {
                    "family_z_summary": {"ffn": {"max": float("nan")}},
                    "multiple_testing_selection": {"family_pvalues": {"ffn": 0.25}},
                },
                "violations": ["warn"],
            }
        ]
    }

    monkeypatch.setattr(
        calibration,
        "_extract_multiple_testing",
        lambda _metrics: {"method": "bh", "alpha": object(), "m": object()},
    )

    summary = calibration.summarize_null_sweep_reports(
        [report],
        tier="balanced",
        target_any_warning_rate=0.0,
    )

    assert summary["recommendations"]["multiple_testing"]["alpha"] == 0.05
    assert summary["recommendations"]["family_caps"] == {}


def test_extract_multiple_testing_handles_missing_alpha_and_m_fields() -> None:
    assert calibration._extract_multiple_testing(
        {"multiple_testing": {"method": "bh"}}
    ) == {"method": "bh"}


def test_summarize_null_sweep_reports_covers_nonfinite_caps_and_full_alpha_grid(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        calibration,
        "_extract_family_max_z",
        lambda _metrics: {"ffn": float("inf")},
    )
    monkeypatch.setattr(
        calibration,
        "_extract_multiple_testing",
        lambda _metrics: {"method": "bh", "alpha": 2.0},
    )

    summary = calibration.summarize_null_sweep_reports(
        [
            {
                "guards": [
                    {
                        "name": "spectral",
                        "metrics": {
                            "multiple_testing_selection": {
                                "family_pvalues": {"ffn": 0.0}
                            }
                        },
                        "violations": ["warn"],
                    }
                ]
            }
        ],
        tier="balanced",
        target_any_warning_rate=-1.0,
    )

    assert summary["recommendations"]["family_caps"] == {}
    assert summary["recommendations"]["multiple_testing"]["alpha"] == 2.0
