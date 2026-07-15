# ruff: noqa: F405
from __future__ import annotations

from tests.reporting._support_guard_metric_impact import canonical_ppl_impact
from tests.reporting._support_report_builder import *  # noqa: F401,F403,F405


class TestEffectivePoliciesAndResolution:
    """Exercise policy extraction and resolution helpers."""

    def test_extract_effective_policies_fills_from_metrics(self):
        report = {
            "meta": {"auto": {"tier": "balanced"}},
            "guards": [
                {
                    "name": "spectral",
                    "policy": {},
                    "metrics": {
                        "caps_applied": 1,
                        "sigma_quantile": 0.92,
                        "deadband": 0.08,
                        "max_caps": 4,
                        "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
                    },
                },
                {
                    "name": "rmt",
                    "policy": {},
                    "metrics": {
                        "deadband_used": 0.09,
                        "margin_used": 1.4,
                        "epsilon_default": 0.08,
                        "epsilon_by_family": {"ffn": 0.1},
                    },
                },
                {
                    "name": "variance",
                    "policy": {},
                    "metrics": {
                        "scope": "ffn",
                        "min_gain_threshold": 0.001,
                        "ve_enabled": True,
                    },
                },
                {
                    "name": "invariants",
                    "policy": {},
                    "metrics": {
                        "checks_performed": 3,
                        "violations_found": 1,
                    },
                },
            ],
        }

        policies = _extract_effective_policies(report)

        assert policies["spectral"]["caps_applied"] == 1
        assert policies["rmt"]["deadband"] == pytest.approx(0.09)
        assert policies["variance"]["scope"] == "ffn"
        assert policies["invariants"]["checks_performed"] == 3

    def test_build_resolved_policies_merges_defaults(self):
        spectral = {
            "sigma_quantile": 0.97,
            "deadband": 0.12,
            "max_caps": 6,
            "family_caps": {"ffn": {"kappa": 2.7}},
            "multiple_testing": {"method": "bh", "alpha": 0.04, "m": 3},
        }
        rmt = {"deadband": 0.12, "margin": 1.6, "epsilon_by_family": {"ffn": 0.09}}
        variance = {
            "predictive_gate": {"sided": "one_sided"},
            "min_effect_lognll": 0.0008,
        }

        resolved = _build_resolved_policies("balanced", spectral, rmt, variance)

        assert resolved["spectral"]["sigma_quantile"] == pytest.approx(0.97)
        assert resolved["rmt"]["epsilon_by_family"]["ffn"] == pytest.approx(0.09)
        assert resolved["variance"]["predictive_one_sided"] is True
        # Balanced tier enforces default min-effect value
        assert resolved["variance"]["min_effect_lognll"] == pytest.approx(0.0)

    def test_compute_policy_and_report_digest_helpers(self):
        policy = {"spectral": {"sigma_quantile": 0.95}}
        digest = _compute_policy_digest(policy)
        assert isinstance(digest, str) and len(digest) == 16

        report = {
            "meta": {
                "model_id": "m",
                "adapter": "hf_causal",
                "commit": "abc",
                "ts": "2024",
            },
            "edit": {"name": "structured", "plan_digest": "plan"},
            "metrics": {"ppl_preview": 10.0, "ppl_final": 11.0, "ppl_ratio": 1.1},
        }
        report_digest = _compute_report_digest(report)
        assert isinstance(report_digest, str) and len(report_digest) == 64

    def test_compute_validation_flags_variants(self):
        ppl = {
            "preview_final_ratio": 1.08,
            "ratio_vs_baseline": 1.05,
            "ratio_ci": (1.0, 1.12),
        }
        spectral = {"caps_applied": 6, "max_caps": 5}
        rmt = {"stable": False}
        invariants = {"status": "fail"}
        guard_metric_impact = {"degradation": 1.05, "degradation_limit": 0.02}

        flags = _compute_validation_flags(
            ppl,
            spectral,
            rmt,
            invariants,
            tier="balanced",
            guard_metric_impact=guard_metric_impact,
        )

        assert flags["preview_final_drift_acceptable"] is False
        assert flags["primary_metric_acceptable"] is False
        assert flags["spectral_stable"] is False
        assert flags["rmt_stable"] is False
        assert flags["invariants_pass"] is False
        assert flags["guard_metric_impact_acceptable"] is False

    def test_compute_validation_flags_target_ratio(self):
        ppl = {
            "preview_final_ratio": 1.02,
            "ratio_vs_baseline": 1.03,
            "ratio_ci": (0.99, 1.05),
        }
        spectral = {"caps_applied": 2, "max_caps": 5}
        rmt = {"stable": True}
        invariants = {"status": "pass"}
        guard_metric_impact = canonical_ppl_impact(10.0, 10.1, degradation_limit=0.02)

        flags = _compute_validation_flags(
            ppl,
            spectral,
            rmt,
            invariants,
            tier="balanced",
            target_ratio=1.05,
            guard_metric_impact=guard_metric_impact,
        )

        assert flags["primary_metric_acceptable"] is True
        assert flags["spectral_stable"] is True
        assert flags["guard_metric_impact_acceptable"] is True


class TestExtractStructuralDeltas:
    """Ensure structural delta extraction covers quant/SVD branches."""

    def test_extract_structural_deltas_quant(self):
        report = {
            "edit": {
                "name": "quant_rtn",
                "deltas": {
                    "params_changed": 10,
                    "heads_pruned": 0,
                    "neurons_pruned": 0,
                    "layers_modified": 2,
                    "bitwidth_map": {
                        "mlp.c_fc": {"bitwidth": 8, "group_size": 32},
                    },
                },
                "plan": {
                    "algorithm": "quant",
                    "scope": "attn",
                    "ranking": "magnitude",
                    "budgets": {"head_budget": {"ratio": 0.2}},
                    "seed": 99,
                    "plan_digest": "quant_plan_energy_0.5",
                },
            },
            "meta": {"seed": 11},
        }

        result = _extract_structural_deltas(report)

        diagnostics = result["compression_diagnostics"]
        assert diagnostics["parameter_analysis"]["bitwidth"]["value"] == 8
        assert diagnostics["algorithm_details"]["modules_quantized"] == 1

    def test_extract_structural_deltas_svd(self):
        report = {
            "edit": {
                "name": "svd95",
                "deltas": {
                    "params_changed": 5,
                    "heads_pruned": 0,
                    "neurons_pruned": 0,
                    "layers_modified": 1,
                    "rank_map": {
                        "mlp.c_fc": {"skipped": False},
                        "mlp.c_proj": {"skipped": True},
                    },
                },
                "plan": {
                    "algorithm": "svd",
                    "scope": "ffn",
                    "plan_digest": "svd_ffn_energy_0.3",
                },
            }
        }

        result = _extract_structural_deltas(report)

        diagnostics = result["compression_diagnostics"]
        assert diagnostics["target_analysis"]["modules_modified"] == 1
        assert diagnostics["parameter_analysis"]["frac"]["effectiveness"] in {
            "applied",
            "too_conservative",
        }


class TestEvaluationReportAnalyticsHelpers:
    """Cover remaining analytics helpers in the report assembly owners."""

    def test_analyze_bitwidth_map(self):
        bitwidth_map = {
            "module1": {"bitwidth": 8},
            "module2": {"bitwidth": 4},
        }
        summary = _analyze_bitwidth_map(bitwidth_map)
        assert summary["total_modules"] == 2
        assert summary["min_bitwidth"] == 4

    def test_compute_savings_summary_rank_map(self):
        deltas = {
            "rank_map": {
                "layer1": {
                    "realized_params_saved": 10,
                    "theoretical_params_saved": 12,
                    "deploy_mode": "decompose",
                },
                "layer2": {
                    "realized_params_saved": 0,
                    "theoretical_params_saved": 5,
                },
            }
        }
        summary = _compute_savings_summary(deltas)
        assert summary["total_realized_params_saved"] == 10
        assert summary["mode"] in {"realized", "theoretical"}
        assert summary["total_theoretical_params_saved"] == 17

    def test_compute_savings_summary_summary_only(self):
        deltas = {
            "savings": {
                "total_realized_params_saved": 0,
                "total_theoretical_params_saved": 40,
                "deploy_mode": "theoretical",
            }
        }
        summary = _compute_savings_summary(deltas)
        assert summary["mode"] == "theoretical"
        assert summary["total_theoretical_params_saved"] == 40

    def test_extract_rank_information(self):
        edit_config = {"rank_policy": "energy", "frac": 0.2}
        deltas = {
            "rank_map": {
                "layer1": {
                    "baseline_rank": 256,
                    "target_rank": 128,
                    "energy_captured": 0.95,
                }
            }
        }
        rank_info = _extract_rank_information(edit_config, deltas)
        per_module = rank_info["per_module"]
        assert per_module["layer1"]["realized_params_saved"] is None
        assert "savings_summary" in rank_info

    def test_generate_run_id_uses_existing(self):
        report = {"meta": {"run_id": "existing-run-id"}}
        assert _generate_run_id(report) == "existing-run-id"

    def test_compute_evaluation_report_hash_ignores_artifacts(self):
        evaluation_report = {
            "schema_version": "v1",
            "run_id": "abc123",
            "meta": {"model_id": "m"},
            "artifacts": {"generated_at": "now"},
        }
        hash_with_artifacts = _compute_report_hash(evaluation_report)
        evaluation_report.pop("artifacts")
        hash_without_artifacts = _compute_report_hash(evaluation_report)
        assert hash_with_artifacts == hash_without_artifacts
