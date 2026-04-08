# ruff: noqa: F405
from __future__ import annotations

from tests.reporting._support_report_builder import *  # noqa: F401,F403,F405


class TestPrivateHelperFunctions:
    """Test private helper functions."""

    def test_normalize_baseline_runreport(self):
        """Test _normalize_baseline with RunReport format."""
        baseline = create_mock_run_report(model_id="baseline-model", ppl_final=8.5)
        # Fix: make it a proper baseline
        baseline["edit"]["name"] = "baseline"

        result = _normalize_baseline(baseline)

        assert result["model_id"] == "baseline-model"
        assert result["ppl_final"] == 8.5
        assert "run_id" in result

    def test_normalize_baseline_v1_schema(self):
        """Test _normalize_baseline with baseline-v1 schema."""
        baseline = create_mock_baseline(schema_type="baseline-v1", ppl_final=7.8)

        result = _normalize_baseline(baseline)

        assert result["model_id"] == "test-model"
        assert result["ppl_final"] == 7.8
        assert result["run_id"] == "baseline123456789"[:16]

    def test_normalize_baseline_normalized_format(self):
        """Test _normalize_baseline with already normalized format."""
        baseline = create_mock_baseline(schema_type="normalized")

        result = _normalize_baseline(baseline)

        assert result == baseline  # Should return as-is

    def test_normalize_baseline_invalid_input(self):
        """Test _normalize_baseline with invalid input."""
        with pytest.raises(ValueError, match="Baseline must be a RunReport dict"):
            _normalize_baseline("invalid_input")

    def test_extract_dataset_info(self):
        """Test _extract_dataset_info function."""
        report = create_mock_run_report()

        result = _extract_dataset_info(report)

        assert result["provider"] == "wikitext"
        assert result["split"] == "test"
        assert result["seq_len"] == 1024
        assert result["windows"]["preview"] == 10
        assert result["windows"]["final"] == 50

    def test_compute_actual_window_hashes_with_windows(self):
        """Test _compute_actual_window_hashes with evaluation windows."""
        report = create_mock_run_report(include_evaluation_windows=True)

        result = _compute_actual_window_hashes(report)

        assert result["preview"].startswith("sha256:")
        assert result["final"].startswith("sha256:")
        assert result["total_tokens"] == 16

    def test_compute_actual_window_hashes_prefers_explicit_hashes(self):
        """Explicit preview/final hashes should be preferred when present."""
        report = create_mock_run_report(include_evaluation_windows=True)
        data_cfg = report["data"]
        data_cfg["preview_hash"] = "deadbeef" * 4
        data_cfg["final_hash"] = "cafebabe" * 4
        data_cfg["preview_total_tokens"] = 1024
        data_cfg["final_total_tokens"] = 2048
        data_cfg["dataset_hash"] = "dataset123"

        hashes = _compute_actual_window_hashes(report)

        assert hashes["preview"] == "blake2s:deadbeefdeadbeefdeadbeefdeadbeef"
        assert hashes["final"] == "blake2s:cafebabecafebabecafebabecafebabe"
        assert hashes["dataset"] == "dataset123"
        assert hashes["total_tokens"] == 3072
        assert hashes["preview_tokens"] == 1024
        assert hashes["final_tokens"] == 2048

    def test_compute_actual_window_hashes_fallback(self):
        """Test _compute_actual_window_hashes fallback to config-based hash."""
        report = create_mock_run_report(include_evaluation_windows=False)

        result = _compute_actual_window_hashes(report)

        assert result["preview"].startswith("sha256:")
        assert result["final"].startswith("sha256:")
        assert result["total_tokens"] == 61440  # (10 + 50) * 1024

    def test_extract_invariants_pass(self):
        """Test _extract_invariants with passing invariants."""
        report = create_mock_run_report()

        result = _extract_invariants(report)

        assert result["status"] == "pass"
        assert result["post"] == "pass"
        assert result["pre"] == "pass"

    def test_extract_invariants_fail(self):
        """Test _extract_invariants with failing invariants."""
        report = create_mock_run_report()
        report["metrics"]["invariants"] = {
            "weight_norm": {"passed": False},
            "activation_range": {"passed": True},
        }

        result = _extract_invariants(report)

        assert result["status"] == "fail"
        assert result["post"] == "fail"

    def test_extract_invariants_warns_on_guard_violations(self):
        """Guard-provided warning violations should mark status as warn."""
        report = create_mock_run_report()
        report["metrics"]["invariants"] = {}
        report.setdefault("guards", []).append(
            {
                "name": "invariants",
                "metrics": {
                    "checks_performed": 3,
                    "violations_found": 1,
                    "fatal_violations": 0,
                    "warning_violations": 1,
                },
                "violations": [
                    {
                        "check": "tokenizer_alignment",
                        "type": "mismatch",
                        "severity": "warning",
                        "detail": {"field": "tokenizer_hash"},
                    }
                ],
            }
        )

        result = _extract_invariants(report)

        assert result["status"] == "warn"
        assert result["summary"]["warning_violations"] == 1
        assert result["failures"][0]["check"] == "tokenizer_alignment"

    def test_extract_invariants_empty(self):
        """Test _extract_invariants with empty invariants."""
        report = create_mock_run_report()
        report["metrics"]["invariants"] = {}

        result = _extract_invariants(report)

        assert result["status"] == "pass"  # Empty treated as pass

    def test_extract_spectral_analysis(self):
        """Test _extract_spectral_analysis function."""
        report = create_mock_run_report()
        baseline = create_mock_baseline()

        result = _extract_spectral_analysis(report, baseline)

        assert result["caps_applied"] == 2  # From mock guards
        assert "summary" in result
        assert result["summary"]["status"] == "capped"
        assert result["max_caps"] == 5
        assert result["summary"]["max_caps"] == 5
        assert result["summary"]["caps_exceeded"] is False
        assert result["multiple_testing"]["method"] == "bh"
        assert "multipletesting" not in result
        assert "contraction" not in result["summary"]

    def test_extract_spectral_analysis_no_caps(self):
        """Test _extract_spectral_analysis with no caps applied."""
        report = create_mock_run_report(include_guards=False)
        baseline = create_mock_baseline()

        result = _extract_spectral_analysis(report, baseline)

        assert result["caps_applied"] == 0
        assert result["summary"]["status"] == "stable"
        assert result["summary"].get("max_caps") is not None

    def test_extract_rmt_analysis(self):
        """Test _extract_rmt_analysis function."""
        report = create_mock_run_report()
        baseline = create_mock_baseline(
            schema_type="normalized"
        )  # Use normalized format with RMT data

        result = _extract_rmt_analysis(report, baseline)

        assert result["epsilon_default"] == pytest.approx(0.01)
        assert result["stable"] is True
        assert result["status"] == "stable"
        assert result["families"]["embed"]["epsilon"] == pytest.approx(0.01)

    def test_extract_rmt_analysis_calculated_stability(self):
        """Test _extract_rmt_analysis with calculated stability."""
        report = {
            "meta": {"auto": {"tier": "balanced"}},
            "guards": [
                {
                    "name": "rmt",
                    "metrics": {
                        "edge_risk_by_family_base": {"ffn": 1.0},
                        "edge_risk_by_family": {"ffn": 1.25},
                        "epsilon_by_family": {"ffn": 0.1},
                    },
                }
            ],
            "metrics": {},
        }
        baseline = {"rmt": {}}

        result = _extract_rmt_analysis(report, baseline)

        # Should calculate stability from the ε-band when no explicit stable flag is present.
        assert result["stable"] is False
        assert result["status"] == "unstable"

    def test_extract_variance_analysis_enabled(self):
        """Test _extract_variance_analysis with variance enabled."""
        report = create_mock_run_report()

        result = _extract_variance_analysis(report)

        assert result["enabled"] is True
        assert result["gain"] == 1.8

    def test_extract_variance_analysis_disabled(self):
        """Test _extract_variance_analysis with variance disabled."""
        report = create_mock_run_report(include_guards=False)

        result = _extract_variance_analysis(report)

        assert result["enabled"] is False
        assert result["gain"] is None

    def test_extract_structural_deltas(self):
        """Test _extract_structural_deltas function."""
        report = create_mock_run_report()

        result = _extract_structural_deltas(report)

        assert result["params_changed"] == 1000
        # Legacy pruning-related fields are no longer emitted
        assert "heads_pruned" not in result
        assert "neurons_pruned" not in result
        assert result["layers_modified"] == 3
        assert result["sparsity"] == 0.1

    def test_extract_structural_deltas_with_bitwidths(self):
        """Test _extract_structural_deltas with bitwidth information."""
        report = create_mock_run_report()
        report["edit"]["deltas"]["bitwidth_map"] = {"layer_0": 8, "layer_1": 4}

        result = _extract_structural_deltas(report)

        assert "bitwidths" in result
        assert result["bitwidths"] == {"layer_0": 8, "layer_1": 4}

    def test_extract_effective_policies(self):
        """Test _extract_effective_policies function."""
        report = create_mock_run_report()

        result = _extract_effective_policies(report)

        assert "spectral" in result
        assert "rmt" in result
        assert "variance" in result
        assert result["spectral"]["sigma_quantile"] == 0.95

    def test_extract_effective_policies_no_guards(self):
        """Test _extract_effective_policies with no guards."""
        report = create_mock_run_report(include_guards=False)

        result = _extract_effective_policies(report)

        # Should create default policies
        assert "spectral" in result
        assert "rmt" in result

    def test_compute_validation_flags(self):
        """Test _compute_validation_flags function."""
        ppl = {"ratio_vs_baseline": 1.05}
        spectral = {"caps_applied": 2, "max_caps": 5}
        rmt = {"stable": True}
        invariants = {"status": "pass"}

        result = _compute_validation_flags(ppl, spectral, rmt, invariants)

        assert result["primary_metric_acceptable"] is True
        assert result["invariants_pass"] is True
        assert result["spectral_stable"] is True  # 2 < 5
        assert result["rmt_stable"] is True

    def test_compute_validation_flags_failures(self):
        """Test _compute_validation_flags with failures."""
        ppl = {"ratio_vs_baseline": 3.0}  # Too high
        spectral = {"caps_applied": 10, "max_caps": 5, "caps_exceeded": True}
        rmt = {"stable": False}
        invariants = {"status": "fail"}

        result = _compute_validation_flags(ppl, spectral, rmt, invariants)

        assert result["primary_metric_acceptable"] is False
        assert result["invariants_pass"] is False
        assert result["spectral_stable"] is False
        assert result["rmt_stable"] is False

    def test_ppl_ratio_gate_balanced_threshold(self):
        """Balanced tier allows ratios up to 1.10 inclusive."""
        ppl = {"ratio_vs_baseline": 1.1, "ratio_ci": (1.02, 1.10)}
        spectral = {"caps_applied": 0, "max_caps": 5}
        rmt = {"stable": True}
        invariants = {"status": "pass"}

        result = _compute_validation_flags(
            ppl, spectral, rmt, invariants, tier="balanced"
        )

        assert result["primary_metric_acceptable"] is True

    def test_ppl_ratio_gate_conservative_fails(self):
        """Conservative tier tightens PPL ratio to 1.05."""
        ppl = {"ratio_vs_baseline": 1.06, "ratio_ci": (1.04, 1.07)}
        spectral = {"caps_applied": 0, "max_caps": 3}
        rmt = {"stable": True}
        invariants = {"status": "pass"}

        result = _compute_validation_flags(
            ppl, spectral, rmt, invariants, tier="conservative"
        )

        assert result["primary_metric_acceptable"] is False

    def test_generate_run_id(self):
        """Test _generate_run_id function."""
        report = create_mock_run_report()

        run_id = _generate_run_id(report)

        assert isinstance(run_id, str)
        assert len(run_id) == 16  # SHA256 hash truncated to 16 chars

    def test_generate_run_id_consistent(self):
        """Test _generate_run_id is consistent for same input."""
        report = create_mock_run_report()

        run_id1 = _generate_run_id(report)
        run_id2 = _generate_run_id(report)

        assert run_id1 == run_id2

    def test_compute_evaluation_report_hash(self):
        """Test _compute_report_hash function."""
        evaluation_report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "run_id": "test123",
            "artifacts": {"path": "/some/path"},  # Should be excluded
        }

        cert_hash = _compute_report_hash(evaluation_report)

        assert isinstance(cert_hash, str)
        assert len(cert_hash) == 16

    def test_compute_evaluation_report_hash_excludes_artifacts(self):
        """Test that evaluation_report hash excludes artifacts section."""
        cert1 = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "run_id": "test123",
            "artifacts": {"path": "/path1"},
        }

        cert2 = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "run_id": "test123",
            "artifacts": {"path": "/path2"},  # Different artifacts
        }

        hash1 = _compute_report_hash(cert1)
        hash2 = _compute_report_hash(cert2)

        assert hash1 == hash2  # Should be same since artifacts excluded


class TestModuleExports:
    """Test stable schema constants exposed by report assembly owners."""

    def test_schema_version_constant(self):
        """Test that REPORT_SCHEMA_VERSION is properly defined."""
        assert REPORT_SCHEMA_VERSION == "v1"


if __name__ == "__main__":
    pytest.main([__file__])
