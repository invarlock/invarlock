# ruff: noqa: F405
from __future__ import annotations

from tests.reporting._support_report_builder import *  # noqa: F401,F403,F405


class TestExtractSpectralAnalysis:
    """Targeted coverage for _extract_spectral_analysis."""

    def test_extract_spectral_analysis_normalizes_policy_and_top_violations(self):
        report = create_mock_run_report()
        spectral_guard = _build_spectral_guard_with_z_scores()
        report["guards"] = [spectral_guard]
        baseline = {
            "spectral": {
                "max_spectral_norm": 1.5,
                "mean_spectral_norm": 1.1,
            }
        }

        result = _extract_spectral_analysis(report, baseline)

        assert result["caps_applied_by_family"] == {"ffn": 2, "attn": 1}
        assert result["bh_family_count"] == 4
        assert result["multiple_testing"]["m"] == 4
        assert result["policy"]["correction_enabled"] is False
        assert result["policy"]["max_spectral_norm"] is None
        assert result["family_z_quantiles"]["ffn"]["count"] == 4
        assert len(result["top_z_scores"]["attn"]) == 3
        assert {v["module"] for v in result["top_violations"]} == {
            "ffn.0.w2",
            "attn.0.wk",
        }

    def test_extract_spectral_analysis_derives_quantiles_from_z_scores(self):
        report = {
            "meta": {"auto": {"tier": "balanced"}},
            "guards": [
                {
                    "name": "spectral",
                    "policy": {
                        "multiple_testing": {"method": "bh", "alpha": 0.05},
                    },
                    "metrics": {
                        "violations_detected": 0,
                        "final_z_scores": {
                            "layers.0.mlp.c_fc": 1.5,
                            "layers.1.mlp.c_fc": "bad-data",
                            "layers.2.attn.proj": -2.4,
                            "layers.3.attn.proj": -0.6,
                            "layers.4.other.adapter": 0.2,
                        },
                        "module_family_map": {
                            "layers.0.mlp.c_fc": "ffn",
                            "layers.1.mlp.c_fc": "ffn",
                            "layers.2.attn.proj": "attn",
                            "layers.3.attn.proj": "attn",
                            "layers.4.other.adapter": "other",
                        },
                    },
                }
            ],
        }
        baseline = {
            "spectral": {
                "max_spectral_norm": 2.0,
                "mean_spectral_norm": 1.0,
            }
        }

        result = _extract_spectral_analysis(report, baseline)

        quantiles = result["family_z_quantiles"]
        assert set(quantiles) == {"ffn", "attn", "other"}
        assert quantiles["ffn"]["count"] == 1  # only numeric entry retained
        assert quantiles["attn"]["max"] == pytest.approx(2.4)
        assert result["top_z_scores"]["attn"][0]["z"] == pytest.approx(2.4)
        assert result["max_caps"] is None or result["max_caps"] >= 0

    def test_extract_spectral_analysis_uses_guard_level_final_z_scores(self):
        report = {
            "meta": {"auto": {"tier": "balanced"}},
            "guards": [
                {
                    "name": "spectral",
                    "policy": {"sigma_quantile": 0.95},
                    "metrics": {"violations_detected": 0},
                    "final_z_scores": {
                        "layers.0.mlp.c_fc": 1.25,
                        "layers.1.attn.proj": -2.75,
                        "layers.2.other.adapter": 0.4,
                    },
                    "module_family_map": {
                        "layers.0.mlp.c_fc": "ffn",
                        "layers.1.attn.proj": "attn",
                        "layers.2.other.adapter": "other",
                    },
                }
            ],
        }
        baseline = {"spectral": {}}

        result = _extract_spectral_analysis(report, baseline)

        quantiles = result["family_z_quantiles"]
        assert quantiles["ffn"]["max"] == pytest.approx(1.25)
        assert quantiles["attn"]["q99"] == pytest.approx(2.75)
        assert result["top_z_scores"]["attn"][0]["module"] == "layers.1.attn.proj"

    def test_extract_spectral_analysis_uses_metrics_fallback(self):
        report = {
            "metrics": {
                "spectral": {
                    "sigma_ratios": [1.25, 0.95, 1.05],
                }
            }
        }
        baseline = {}

        result = _extract_spectral_analysis(report, baseline)

        summary = result["summary"]
        assert summary["max_sigma_ratio"] == pytest.approx(1.25)
        assert summary["median_sigma_ratio"] == pytest.approx(1.05)
        assert result["caps_applied"] == 0

    def test_extract_spectral_analysis_with_rich_metrics(self):
        report = {
            "meta": {"auto": {"tier": "balanced"}},
            "guards": [
                {
                    "name": "spectral",
                    "policy": {
                        "sigma_quantile": 0.95,
                        "deadband": 0.1,
                        "max_caps": 5,
                        "family_caps": {
                            "ffn": {"kappa": 2.5},
                            "attn": {"kappa": 2.8},
                        },
                        "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
                    },
                    "metrics": {
                        "modules_checked": 64,
                        "violations_detected": 2,
                        "caps_exceeded": False,
                        "max_caps": 5,
                        "max_spectral_norm_final": 7.5,
                        "mean_spectral_norm_final": 5.0,
                        "family_caps": {
                            "ffn": {"kappa": 2.6},
                            "attn": {"kappa": 2.9},
                        },
                        "family_z_summary": {
                            "ffn": {
                                "max": 2.4,
                                "mean": 1.2,
                                "count": 32,
                                "violations": 1,
                            },
                            "attn": {
                                "max": 2.1,
                                "mean": 1.1,
                                "count": 32,
                                "violations": 0,
                            },
                        },
                        "family_z_quantiles": {
                            "ffn": {"q95": 2.2, "q99": 2.3, "max": 2.4, "count": 32},
                        },
                        "top_z_scores": {
                            "ffn": [
                                {"module": "mlp.c_fc", "z": 2.4},
                                {"module": "mlp.c_proj", "z": 2.1},
                            ],
                        },
                    },
                    "violations": [
                        {
                            "check": "max_sigma",
                            "type": "spectral_violation",
                            "severity": "warning",
                            "module": "mlp.c_fc",
                            "family": "ffn",
                            "kappa": 2.5,
                            "z_score": 2.6,
                        }
                    ],
                }
            ],
            "metrics": {
                "spectral": {"sigma_ratios": [1.0, 1.1, 0.9]},
            },
        }
        baseline = {
            "spectral": {
                "max_spectral_norm": 6.0,
                "mean_spectral_norm": 4.5,
            },
            "metrics": {
                "spectral": {
                    "max_spectral_norm_final": 6.0,
                    "mean_spectral_norm_final": 4.5,
                }
            },
        }

        result = _extract_spectral_analysis(report, baseline)

        assert result["caps_applied"] == 2
        assert result["summary"]["max_sigma_ratio"] == pytest.approx(7.5 / 6.0)
        assert result["families"]["ffn"]["violations"] == 1
        assert "top_z_scores" in result

    def test_extract_spectral_analysis_filters_invalid_top_z_scores(self):
        report = create_mock_run_report()
        spectral_guard = _build_spectral_guard_with_z_scores()
        spectral_guard["metrics"]["top_z_scores"] = {
            "ffn": [
                {"module": "ffn.0.w2", "z": 3.0},
                {"module": "ffn.0.w1", "z": "bad"},
                "not-a-dict",
            ],
            "attn": "ignore",
        }
        report["guards"] = [spectral_guard]

        result = _extract_spectral_analysis(report, baseline={})

        assert result["top_z_scores"]["ffn"] == [
            {"module": "ffn.0.w2", "z": pytest.approx(3.0)}
        ]
        attn_entries = result["top_z_scores"]["attn"]
        assert len(attn_entries) == 3
        assert all(isinstance(entry["z"], float) for entry in attn_entries)


class TestExtractRMTAnalysis:
    """Exercise _extract_rmt_analysis edge cases."""

    def test_extract_rmt_analysis_with_family_metrics(self):
        report = {
            "meta": {"auto": {"tier": "balanced"}},
            "guards": [
                {
                    "name": "rmt",
                    "policy": {"deadband": 0.1},
                    "metrics": {
                        "edge_risk_by_family_base": {"ffn": 1.0, "attn": 1.0},
                        "edge_risk_by_family": {"ffn": 1.05, "attn": 1.07},
                        "epsilon_by_family": {"ffn": 0.1, "attn": 0.08},
                    },
                }
            ],
        }
        baseline = {"rmt": {}}

        result = _extract_rmt_analysis(report, baseline)

        assert result["edge_risk_by_family_base"]["ffn"] == pytest.approx(1.0)
        assert result["edge_risk_by_family"]["attn"] == pytest.approx(1.07)
        assert result["epsilon_by_family"]["ffn"] == pytest.approx(0.1)
        assert isinstance(result["stable"], bool)
        assert result["max_edge_ratio"] == pytest.approx(1.07)

    def test_extract_rmt_analysis_without_guard_falls_back(self):
        report = {
            "meta": {"auto": {"tier": "balanced"}},
            "guards": [],
            "metrics": {},
        }
        baseline = {}

        result = _extract_rmt_analysis(report, baseline)

        assert result["evaluated"] is False
        assert result["families"]["ffn"]["epsilon"] == pytest.approx(0.01)
        assert result["status"] in {"stable", "unstable"}


class TestExtractVarianceAnalysis:
    """Cover variance guard metadata extraction."""

    def test_extract_variance_analysis_with_ab_metadata(self):
        report = {
            "guards": [
                {
                    "name": "variance",
                    "metrics": {
                        "ve_enabled": False,
                        "gain": 0.002,
                        "ppl_no_ve": 50.0,
                        "ppl_with_ve": 49.95,
                        "ratio_ci": (0.99, 1.01),
                        "calibration": {"windows": 12},
                        "tap": "mlp.c_proj",
                        "scope": "ffn",
                        "predictive_gate": {"enabled": True},
                        "ab_seed_used": 123,
                        "ab_windows_used": 24,
                        "ab_provenance": "synthetic",
                        "ab_point_estimates": {"mean": 0.002},
                    },
                }
            ]
        }

        result = _extract_variance_analysis(report)

        assert result["enabled"] is False
        assert result["gain"] == pytest.approx(0.002)
        assert result["ppl_no_ve"] == 50.0
        assert result["ab_test"]["seed"] == 123

    def test_extract_variance_analysis_top_level_metrics(self):
        report = {
            "variance": {
                "metrics": {
                    "ve_enabled": True,
                    "gain": 0.002,
                    "ratio_ci": (0.98, 0.99),
                    "calibration": {"coverage": 10, "requested": 12, "status": "ok"},
                    "tap": "mlp.c_proj",
                }
            },
            "metrics": {"variance": {"gain": 0.002}},
        }

        result = _extract_variance_analysis(report)

        assert result["enabled"] is False
        assert result["gain"] == 0.002


class TestExtractInvariants:
    """Exercise invariant extraction for guard-driven failures."""

    def test_extract_invariants_with_guard_metrics(self):
        report = {
            "metrics": {
                "invariants": {
                    "weight_norm": {
                        "passed": False,
                        "message": "too large",
                    }
                }
            },
            "guards": [
                {
                    "name": "invariants",
                    "metrics": {
                        "checks_performed": 3,
                        "violations_found": 1,
                        "fatal_violations": 1,
                        "warning_violations": 0,
                    },
                    "violations": [
                        {
                            "check": "weight_norm",
                            "severity": "error",
                            "type": "fatal",
                            "detail": {"module": "mlp"},
                        }
                    ],
                }
            ],
        }

        invariants = _extract_invariants(report)

        assert invariants["status"] == "fail"
        assert invariants["summary"]["fatal_violations"] == 1
        assert invariants["failures"]
