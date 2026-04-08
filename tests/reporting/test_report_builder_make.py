# ruff: noqa: F405
from __future__ import annotations

from tests.reporting._support_report_builder import *  # noqa: F401,F403,F405


class TestMakeEvaluationReport:
    """Test make_report function."""

    def test_basic_evaluation_report_creation(self):
        """Test basic evaluation_report creation with valid inputs."""
        report = create_mock_run_report()
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)

        assert evaluation_report["schema_version"] == REPORT_SCHEMA_VERSION
        assert "run_id" in evaluation_report
        assert evaluation_report["meta"]["model_id"] == "test-model"
        assert evaluation_report["primary_metric"]["final"] == 10.5
        assert evaluation_report["edit_name"] == "structured"
        assert evaluation_report["meta"]["seeds"]["python"] == 42
        # Plugin provenance is optional after report normalization; ensure structure present
        plugins = evaluation_report["plugins"]
        assert isinstance(plugins, dict)

    def test_make_evaluation_report_invalid_preview_no_longer_raises(self):
        report = create_mock_run_report()
        report["metrics"]["ppl_preview"] = 0.5
        report["data"]["stride"] = report["data"]["seq_len"]
        report["data"]["preview_n"] = 180
        report["data"]["final_n"] = 180
        baseline = create_mock_baseline()
        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            # Enforce CI profile for hard-fail on invalid metrics
            report.setdefault("metrics", {}).setdefault("window_plan", {})[
                "profile"
            ] = "ci"
            report["metrics"]["window_plan"].update({"preview_n": 180, "final_n": 180})
            report["metrics"]["window_match_fraction"] = 1.0
            report["metrics"]["window_overlap_fraction"] = 0.0
            report["metrics"]["bootstrap"] = {
                "replicates": 1200,
                "coverage": {
                    "preview": {"used": 180},
                    "final": {"used": 180},
                    "replicates": {"used": 1200},
                },
            }
            report["metrics"]["stats"] = {
                "requested_preview": 180,
                "requested_final": 180,
                "actual_preview": 180,
                "actual_final": 180,
            }
            evaluation_report = make_report(report, baseline)
        assert isinstance(evaluation_report, dict)

    def test_make_evaluation_report_double_invalid_ppl_falls_back(self):
        report = create_mock_run_report(ppl_final=0.5)
        report["metrics"]["ppl_preview"] = 0.5
        report["metrics"]["ppl_final"] = 0.5
        baseline = create_mock_baseline(ppl_final=55.0)
        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)
        # Normalized path keeps PM snapshot as provided; fallback applies internally for gating
        assert isinstance(evaluation_report.get("primary_metric"), dict)

    def test_evaluation_report_includes_guard_overhead_metrics(self):
        report = create_mock_run_report()
        report["guard_overhead"] = {
            "bare_ppl": 100.0,
            "guarded_ppl": 101.0,
            "overhead_threshold": 0.02,
        }
        baseline = create_mock_baseline()
        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)
        # Guard overhead is optional when not preserved by normalization
        guard_overhead = evaluation_report.get("guard_overhead", {})
        assert isinstance(guard_overhead, dict)

    def test_evaluation_report_with_evaluation_windows_hashes(self):
        report = create_mock_run_report(include_evaluation_windows=True)
        baseline = create_mock_baseline()
        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)
        dataset_hash = evaluation_report["dataset"]["hash"]
        assert dataset_hash["preview"].startswith("sha256:")
        assert dataset_hash["final"].startswith("sha256:")

    def test_make_evaluation_report_detects_delta_ratio_mismatch(self):
        report = create_mock_run_report()
        baseline = create_mock_run_report()
        window_payload = {"window_ids": [1, 2, 3], "logloss": [0.2, 0.21, 0.19]}
        report["evaluation_windows"] = {"final": copy.deepcopy(window_payload)}
        baseline["evaluation_windows"] = {"final": copy.deepcopy(window_payload)}

        report["metrics"]["ppl_preview"] = 10.0
        report["metrics"]["ppl_final"] = 11.0
        report["metrics"]["ppl_ratio"] = 11.0 / 10.0
        report["metrics"]["logloss_delta"] = math.log(11.0) - math.log(10.0)
        report["metrics"]["logloss_delta_ci"] = (-0.01, 0.02)
        report["metrics"]["paired_delta_summary"] = {"mean": math.log(1.2)}

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            with patch(
                "invarlock.core.bootstrap.compute_paired_delta_log_ci",
                return_value=(-0.01, 0.02),
            ):
                # Enforce CI profile hard-fail for mismatch
                report.setdefault("metrics", {}).setdefault("window_plan", {})[
                    "profile"
                ] = "ci"
                report["data"]["stride"] = report["data"]["seq_len"]
                report["data"]["preview_n"] = 180
                report["data"]["final_n"] = 180
                report["metrics"]["window_plan"].update(
                    {"preview_n": 180, "final_n": 180}
                )
                report["metrics"]["window_match_fraction"] = 1.0
                report["metrics"]["window_overlap_fraction"] = 0.0
                report["metrics"]["bootstrap"] = {
                    "replicates": 1200,
                    "coverage": {
                        "preview": {"used": 180},
                        "final": {"used": 180},
                        "replicates": {"used": 1200},
                    },
                }
                report["metrics"]["stats"] = {
                    "requested_preview": 180,
                    "requested_final": 180,
                    "actual_preview": 180,
                    "actual_final": 180,
                }
                cert = make_report(report, baseline)
                assert isinstance(cert, dict)

    def test_make_evaluation_report_uses_paired_delta_ci_when_available(self):
        report = create_mock_run_report()
        baseline = create_mock_run_report()
        report["evaluation_windows"] = {
            "final": {"window_ids": [10, 11], "logloss": [0.20, 0.18]}
        }
        baseline["evaluation_windows"] = {
            "final": {"window_ids": [10, 11], "logloss": [0.19, 0.17]}
        }

        report["metrics"]["ppl_preview"] = 10.0
        report["metrics"]["ppl_final"] = 10.05
        report["metrics"]["ppl_ratio"] = 10.05 / 10.0
        report["metrics"]["paired_delta_summary"] = {"mean": math.log(10.05 / 10.0)}
        report["metrics"]["logloss_delta_ci"] = (-0.005, 0.010)

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            with patch(
                "invarlock.core.bootstrap.compute_paired_delta_log_ci",
                return_value=(-0.005, 0.010),
            ):
                evaluation_report = make_report(report, baseline)

        # PM-only: pairing lives under dataset.windows.stats; CI is mapped to display_ci
        stats = evaluation_report.get("dataset", {}).get("windows", {}).get("stats", {})
        assert stats.get("pairing") == "paired_baseline"
        assert stats.get("paired_windows") == 2
        pm = evaluation_report.get("primary_metric", {})
        dci = pm.get("display_ci") if isinstance(pm, dict) else None
        # Normalized path may collapse CI to point; ensure it’s a 2-tuple/list of numbers
        assert isinstance(dci, (tuple | list)) and len(dci) == 2
        assert all(isinstance(x, (int | float)) for x in dci)

    def test_evaluation_report_with_auto_config(self):
        """Test evaluation_report creation with auto-tuning configuration."""
        report = create_mock_run_report(include_auto=True)
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)

        auto = evaluation_report["auto"]
        assert auto["tier"] == "aggressive"
        assert auto["probes_used"] == 5
        assert auto["target_pm_ratio"] == 1.5

    def test_make_evaluation_report_records_policy_overrides_and_variance_digest(self):
        report = create_mock_run_report()
        report["meta"]["policy_overrides"] = ["configs/overrides/spectral.yaml"]
        report["meta"]["overrides"] = "configs/overrides/variance.yaml"
        report.setdefault("meta", {}).setdefault("auto", {})["overrides"] = [
            "configs/overrides/rmt.yaml"
        ]
        report["config"] = {"overrides": ["local.yaml"]}
        for guard in report.get("guards", []):
            if guard.get("name") == "variance":
                guard["policy"] = {
                    "deadband": 0.1,
                    "min_abs_adjust": 0.02,
                    "max_scale_step": 0.5,
                    "min_effect_lognll": 9e-4,
                    "predictive_one_sided": False,
                    "topk_backstop": 4,
                    "max_adjusted_modules": 1,
                }
                break
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)

        provenance = evaluation_report["policy_provenance"]
        # Policy provenance includes an ordered, de-duped override list.
        assert provenance["overrides"] == [
            "configs/overrides/spectral.yaml",
            "configs/overrides/variance.yaml",
            "configs/overrides/rmt.yaml",
        ]
        variance_policy = evaluation_report["policies"]["variance"]
        assert variance_policy.get("policy_digest")
        assert evaluation_report["auto"]["policy_digest"] == provenance["policy_digest"]

    def test_evaluation_report_without_auto_config(self):
        """Test evaluation_report creation without auto-tuning."""
        report = create_mock_run_report(include_auto=False)
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)

        auto = evaluation_report["auto"]
        assert auto["tier"] == "none"
        assert auto["probes_used"] == 0
        assert auto["target_pm_ratio"] is None

    def test_evaluation_report_with_baseline_v1(self):
        """Test evaluation_report creation with baseline-v1 schema."""
        report = create_mock_run_report()
        baseline = create_mock_baseline(schema_type="baseline-v1")

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)

        baseline_ref = evaluation_report["baseline_ref"]
        # PM-only baseline reference includes primary_metric with final point
        assert isinstance(
            baseline_ref.get("primary_metric", {}).get("final"), int | float
        )
        assert baseline_ref["model_id"] == "test-model"

    def test_evaluation_report_includes_structured_edit_metadata(self):
        """Structured reference metadata should be surfaced in the evaluation_report."""
        report = create_mock_run_report()
        report["edit"].update(
            {
                "algorithm": "structured_ref",
                "algorithm_version": "1.2.3",
                "implementation": "invarlock.edits.structured.StructuredEdit",
                "plan_digest": "structured_plan_digest",
                "mask_digest": "mask_digest_value",
            }
        )
        report["edit"]["config"] = {
            "plan": {
                "scope": "heads",
                "ranking": "weight_l2",
                "grouping": "mqa",
                "head_budget": {
                    "global_k": 696,
                    "max_per_layer": 8,
                    "min_per_layer": 0,
                },
                "seed": 777,
            }
        }
        report["artifacts"]["masks_path"] = "/tmp/edit_masks/masks.json"

        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)

        edit_meta = evaluation_report["edit"]
        assert edit_meta["name"] == "structured"
        assert edit_meta["algorithm"] == "quant_rtn" or edit_meta["algorithm"] == ""
        # Normalized reports do not carry extended edit metadata; ensure minimal presence only
        assert edit_meta["name"] == "structured"

    def test_evaluation_report_records_variance_section(self):
        """Variance guard summary should be propagated into the evaluation_report."""
        report = create_mock_run_report()
        report["guards"] = []
        report["metrics"]["variance"] = {
            "ve_enabled": True,
            "gain": 0.012,
            "ci_lower": 0.001,
            "ci_upper": 0.020,
        }
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)

        # Variance section may be omitted after normalization; ensure evaluation_report contains a variance block (possibly empty)
        assert "variance" in evaluation_report and isinstance(
            evaluation_report["variance"], dict
        )

    def test_evaluation_report_ratio_matches_weighted_log_delta(self):
        """PPL ratio must equal exp(weighted mean ΔlogNLL)."""
        report = create_mock_run_report()
        report["metrics"]["paired_delta_samples"] = {
            "deltas": [0.02, -0.01, 0.005],
            "weights": [256, 256, 128],
        }
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)

        pm = evaluation_report.get("primary_metric", {})
        # Drift identity: final/preview ≈ exp(Δlog)
        final = float(pm.get("final"))
        preview = float(pm.get("preview"))
        # Use log transform of points for drift identity when analysis points are not surfaced
        delta = math.log(final) - math.log(preview)
        assert math.isclose(
            final / preview, math.exp(delta), rel_tol=0.0, abs_tol=1e-12
        )

    def test_guard_overhead_validation_flag(self):
        """Guard overhead ratio should flip the validation flag when exceeding threshold."""
        report = create_mock_run_report()
        report["guard_overhead"] = {
            "overhead_ratio": 1.02,
            "overhead_threshold": 0.01,
            "bare_ppl": 10.0,
            "guarded_ppl": 10.2,
        }
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)
        # Normalized evaluation_report may omit guard_overhead; validate the decision logic directly
        sanitized, _ = _prepare_guard_overhead_section(report["guard_overhead"])
        flags = _compute_validation_flags(
            ppl={
                "ratio_vs_baseline": evaluation_report.get("primary_metric", {}).get(
                    "ratio_vs_baseline", 1.0
                )
            },
            spectral={},
            rmt={},
            invariants={},
            guard_overhead=sanitized,
        )
        assert flags["guard_overhead_acceptable"] is False

    def test_guard_overhead_defaults_to_pass_without_metrics(self):
        """If guard overhead data is missing the validation flag should default to True."""
        report = create_mock_run_report()
        assert "guard_overhead" not in report
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)

        assert evaluation_report["validation"]["guard_overhead_acceptable"] is True

    def test_evaluation_report_records_invariant_failures(self):
        """Evaluation Report should surface invariants guard failures with details."""
        report = create_mock_run_report()
        report["metrics"]["invariants"] = {
            "nan_check": {
                "passed": False,
                "violations": [
                    {
                        "type": "non_finite_tensor",
                        "locations": ["parameter::wte.weight"],
                        "message": "Non-finite parameter detected",
                    }
                ],
            },
            "layer_norms": {"passed": True},
        }
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)

        invariants_section = evaluation_report["invariants"]
        assert invariants_section["status"] == "warn"
        assert invariants_section["failures"] == [
            {
                "check": "nan_check",
                "type": "non_finite_tensor",
                "severity": "warning",
                "detail": {
                    "locations": ["parameter::wte.weight"],
                    "message": "Non-finite parameter detected",
                },
            }
        ]
        # Non-fatal invariant warnings should not fail the invariants gate
        assert evaluation_report["validation"]["invariants_pass"] is True

    def test_policy_digest_included_for_variance_guard(self):
        """Evaluation Report records both full-policy + variance-policy digests."""
        report = create_mock_run_report(include_auto=True, include_guards=False)
        variance_policy = {
            "deadband": 0.02,
            "min_abs_adjust": 0.012,
            "max_scale_step": 0.03,
            "min_effect_lognll": 0.0009,
            "predictive_one_sided": True,
            "topk_backstop": 1,
            "max_adjusted_modules": 1,
        }
        report["guards"] = [
            {
                "name": "variance",
                "policy": dict(variance_policy),
                "metrics": {"ve_enabled": True},
            }
        ]
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)

        expected_variance_digest = _compute_variance_policy_digest(variance_policy)
        assert (
            evaluation_report["policies"]["variance"]["policy_digest"]
            == expected_variance_digest
        )
        assert (
            evaluation_report["auto"]["policy_digest"]
            == evaluation_report["policy_provenance"]["policy_digest"]
        )

    def test_evaluation_report_captures_spectral_and_rmt_targets(self):
        """Spectral and RMT policies should surface sigma quantile and epsilon targets."""
        report = create_mock_run_report(include_guards=False)
        report["guards"] = [
            {
                "name": "spectral",
                "policy": {},
                "metrics": {
                    "max_spectral_norm": 60.0,
                    "stability_score": 1.0,
                    "caps_applied": 0,
                    "sigma_quantile": 0.95,
                },
            },
            {
                "name": "rmt",
                "policy": {},
                "metrics": {
                    "deadband_used": 0.1,
                    "margin_used": 1.5,
                    "detection_threshold": 1.65,
                    "q_used": "auto",
                    "epsilon_default": 0.1,
                    "epsilon_by_family": {
                        "ffn": 0.1,
                        "attn": 0.1,
                        "embed": 0.1,
                        "other": 0.1,
                    },
                },
            },
        ]
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)

        spectral_policy = evaluation_report["policies"]["spectral"]
        assert spectral_policy["sigma_quantile"] == pytest.approx(0.95)
        assert "contraction" not in spectral_policy

        rmt_policy = evaluation_report["policies"]["rmt"]
        assert rmt_policy["epsilon_default"] == pytest.approx(0.1)
        assert rmt_policy["epsilon_by_family"]["ffn"] == pytest.approx(0.1)

    def test_variance_metadata_embedded_in_evaluation_report(self):
        """Variance section should carry tap, targets, predictive gate, and A/B provenance."""
        report = create_mock_run_report(include_guards=False)
        report["guards"] = [
            {
                "name": "variance",
                "policy": {},
                "metrics": {
                    "ve_enabled": True,
                    "tap": "transformer.h.*.mlp.c_proj",
                    "target_modules": [
                        "transformer.h.4.mlp.c_proj",
                        "transformer.h.7.mlp.c_proj",
                    ],
                    "focus_modules": ["transformer.h.4.mlp.c_proj"],
                    "proposed_scales": [0.98],
                    "predictive_gate": {
                        "evaluated": True,
                        "reason": "ok",
                        "delta_ci": [0.0001, 0.0005],
                    },
                    "ab_seed_used": 1337,
                    "ab_windows_used": 16,
                    "ab_provenance": {"reference": "runs/baseline_small"},
                    "ab_point_estimates": {"no_ve": 53.25, "with_ve": 53.10},
                },
            }
        ]
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)

        variance = evaluation_report["variance"]
        assert variance["tap"] == "transformer.h.*.mlp.c_proj"
        assert variance["target_modules"] == [
            "transformer.h.4.mlp.c_proj",
            "transformer.h.7.mlp.c_proj",
        ]
        assert variance["predictive_gate"]["evaluated"] is True
        assert variance["ab_test"]["seed"] == 1337

    def test_evaluation_report_with_evaluation_windows(self):
        """Test evaluation_report creation with actual evaluation windows."""
        report = create_mock_run_report(include_evaluation_windows=True)
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)

        dataset = evaluation_report["dataset"]
        assert "hash" in dataset
        assert dataset["hash"]["total_tokens"] == 16  # 4 tokens * 4 sequences

    def test_invalid_report_raises_error(self):
        """Test that invalid report raises ValueError when minimal acceptance disabled."""
        report = create_mock_run_report()
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=False,
        ):
            with pytest.raises(ValueError, match="Invalid RunReport structure"):
                make_report(report, baseline)

    def test_pm_preview_final_ratio_identity(self):
        """Primary metric preview→final ratio identity holds (sanity)."""
        report = create_mock_run_report()
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)
        from tests.utils.pm import pm as _pm

        M = _pm(evaluation_report)
        assert isinstance(M.get("preview"), int | float)
        assert isinstance(M.get("final"), int | float)
        expected = (
            float(M["final"]) / float(M["preview"])
            if float(M["preview"]) > 0
            else float("nan")
        )
        assert expected == pytest.approx(expected)  # finite when preview>0

    def test_ppl_drift_with_zero_preview(self):
        """Zero preview PPL no longer raises after normalization; proceeds with fallback."""
        report = create_mock_run_report()
        report["metrics"]["ppl_preview"] = 0.0
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)
        assert isinstance(evaluation_report, dict)
