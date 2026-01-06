from __future__ import annotations

from invarlock.reporting import guards_analysis as GA


def test_measurement_contract_digest_handles_bad_str() -> None:
    class _BadStr:
        def __str__(self) -> str:  # pragma: no cover
            raise RuntimeError("boom")

    assert GA._measurement_contract_digest({"x": _BadStr()}) is None


def test_measurement_contract_digest_success_and_empty() -> None:
    assert GA._measurement_contract_digest({}) is None
    digest = GA._measurement_contract_digest({"x": 1})
    assert isinstance(digest, str) and len(digest) == 16


def test_extract_invariants_covers_fail_and_warn_paths() -> None:
    report_fail = {
        "metrics": {
            "invariants": {
                "ok": {"passed": True},
                "bad_with_list": {
                    "passed": False,
                    "violations": [
                        {"type": "mismatch", "severity": "warning", "foo": 1},
                        "bad",
                    ],
                },
                "bad_no_list": {
                    "passed": False,
                    "type": "oops",
                    "message": "nope",
                    "x": 1,
                },
                "bad_scalar": False,
            }
        },
        "guards": [
            {
                "name": "invariants",
                "metrics": {"fatal_violations": 1, "warning_violations": 0},
                "violations": [
                    {"check": "x", "type": "violation", "severity": "warning"},
                    "bad",
                ],
            }
        ],
    }
    out_fail = GA._extract_invariants(report_fail)
    assert out_fail["status"] == "fail"
    assert out_fail["failures"]

    report_warn = {
        "metrics": {"invariants": {}},
        "guards": [
            {
                "name": "invariants",
                "metrics": {"fatal_violations": 0, "warning_violations": 1},
                "violations": [
                    {"check": "x", "type": "violation", "severity": "warning"}
                ],
            }
        ],
    }
    out_warn = GA._extract_invariants(report_warn)
    assert out_warn["status"] == "warn"


def test_extract_invariants_guard_entry_no_violations_keeps_pass() -> None:
    report = {
        "metrics": {"invariants": {}},
        "guards": [
            {
                "name": "invariants",
                "metrics": {"fatal_violations": 0, "warning_violations": 0},
                "violations": [],
            }
        ],
    }
    out = GA._extract_invariants(report)
    assert out["status"] == "pass"


def test_extract_spectral_analysis_caps_applied_int_fallback() -> None:
    report = {
        "guards": [
            {
                "name": "spectral",
                "policy": {"max_caps": "bad"},
                "metrics": {
                    "violations_detected": "not-an-int",
                    "max_caps": "bad",
                    "max_spectral_norm_final": "bad",
                    "mean_spectral_norm_final": "bad",
                },
            }
        ]
    }
    out = GA._extract_spectral_analysis(report, baseline={})
    assert out["caps_applied"] == 0
    assert isinstance(out.get("max_caps"), int)


def test_extract_spectral_analysis_baseline_metrics_spectral_not_dict() -> None:
    baseline = {"metrics": {"spectral": ["bad"]}}
    out = GA._extract_spectral_analysis({"guards": []}, baseline=baseline)
    assert out["evaluated"] is False


def test_extract_rmt_analysis_edge_risk_paths_and_contract_hashes() -> None:
    contract = {"estimator": {"type": "power_iter"}}
    baseline = {
        "rmt": {
            "measurement_contract": contract,
            "edge_risk_by_family": {"attn": 1.0, "bad": float("nan")},
        }
    }
    report = {
        "guards": [
            {
                "name": "rmt",
                "policy": {"epsilon": {"attn": 0.1}, "epsilon_default": 0.2},
                "metrics": {
                    # Leave base empty so we fall back to baseline_edge_by_family.
                    "edge_risk_by_family_base": {},
                    "edge_risk_by_family": {
                        "attn": 2.0,
                        "ffn": 0.0,
                        "bad": float("inf"),
                    },
                    "epsilon_by_family": {},
                    "epsilon_default": 0.3,
                    "measurement_contract": contract,
                },
            }
        ]
    }
    out = GA._extract_rmt_analysis(report, baseline)
    assert out["evaluated"] is True
    assert out["measurement_contract_match"] is True
    assert out["epsilon_violations"]
    assert out["families"]["attn"]["ratio"] == 2.0


def test_extract_rmt_analysis_legacy_covers_guard_and_fallback_paths() -> None:
    report_guard = {
        "guards": [
            {
                "name": "rmt",
                "policy": {
                    "epsilon": {"attn": 0.1, "ffn": 0.2, "bad": "x"},
                    "margin": 1.5,
                    "deadband": 0.1,
                    "epsilon_default": 0.15,
                },
                "metrics": {
                    "epsilon_default": 0.12,
                    "outliers_per_family": {"attn": 3, "ffn": 1},
                    "baseline_outliers_per_family": {"attn": 2, "ffn": 1},
                    "max_mp_ratio_final": 3.0,
                    "mean_mp_ratio_final": 2.0,
                    "flagged_rate": 0.9,
                },
            }
        ]
    }
    baseline_with_rmt = {
        "rmt": {
            "outliers": 3,
            "max_mp_ratio_final": 2.0,
            "mean_mp_ratio_final": 1.5,
        }
    }
    out_guard = GA._extract_rmt_analysis_legacy(report_guard, baseline_with_rmt)
    assert out_guard["evaluated"] is True
    assert out_guard["families"]
    assert out_guard["status"] in {"stable", "unstable"}
    assert out_guard["margin"] == 1.5
    assert out_guard["deadband"] == 0.1

    # Baseline-less path: triggers conservative baseline fallback when bare==0.
    report_conservative = {
        "guards": [
            {
                "name": "rmt",
                "metrics": {"outliers_per_family": {"attn": 2}, "max_ratio": 1.0},
                "policy": {"epsilon_by_family": {"attn": 0.1}},
            }
        ]
    }
    out_conservative = GA._extract_rmt_analysis_legacy(report_conservative, {})
    assert out_conservative["outliers_bare"] == max(
        0, out_conservative["outliers_guarded"] - 1
    )

    # Guard-absent fallback path: pulls from metrics.rmt + top-level rmt.families.
    report_fallback = {
        "guards": [],
        "metrics": {"rmt": {"outliers": 2, "stable": True}},
        "rmt": {
            "families": {
                "attn": {"outliers_guarded": 1, "outliers_bare": 1, "epsilon": 0.1}
            }
        },
    }
    out_fallback = GA._extract_rmt_analysis_legacy(report_fallback, {})
    assert out_fallback["stable"] is True
    assert out_fallback["outliers_guarded"] == 2


def test_extract_variance_analysis_provenance_window_ids_and_ratio_ci_fail() -> None:
    report = {
        "guards": [
            {
                "name": "variance",
                "metrics": {
                    # Keep gain unset so it falls back to metrics.variance for gain/ppl.
                    "ve_enabled": False,
                    "ratio_ci": "bad",
                    "predictive_gate": {"enabled": True},
                    "ab_seed_used": 123,
                    "ab_windows_used": 7,
                    "ab_provenance": {
                        "nested": [
                            {"window_ids": [1, 2.0, "x"]},
                            {"more": {"window_ids": [3]}},
                        ]
                    },
                },
                "policy": {"mode": "ab"},
            }
        ],
        "metrics": {"variance": {"gain": 0.1, "ppl_no_ve": 10.0, "ppl_with_ve": 9.0}},
    }
    out = GA._extract_variance_analysis(report)
    assert out["enabled"] is False
    assert out["gain"] == 0.1
    assert out["ppl_no_ve"] == 10.0
    assert out["ppl_with_ve"] == 9.0
    assert out["ab_test"]["provenance"]["window_ids"] == [1, 2, 3]


def test_extract_variance_analysis_handles_non_dict_variance_metrics() -> None:
    out = GA._extract_variance_analysis(
        {"guards": [], "metrics": {"variance": ["bad"]}}
    )
    assert out["gain"] is None


def test_extract_variance_analysis_keeps_existing_window_ids() -> None:
    report = {
        "guards": [
            {
                "name": "variance",
                "metrics": {"ab_provenance": {"window_ids": [3, 1, 2]}},
            }
        ]
    }
    out = GA._extract_variance_analysis(report)
    assert out["ab_test"]["provenance"]["window_ids"] == [3, 1, 2]
