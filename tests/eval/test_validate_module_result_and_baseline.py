"""
Comprehensive test coverage for InvarLock validation module.
Tests for validate.py to achieve 80%+ coverage.
"""

import json
import tempfile
from pathlib import Path

import pytest

from invarlock.reporting.validate import (
    ValidationResult,
    _validate_invariants,
    _validate_structural_counts,
    load_baseline,
    save_baseline,
    validate_against_baseline,
    validate_drift_gate,
    validate_guard_metric_impact,
)

if __name__ == "__main__":
    pytest.main([__file__])


class TestValidateResult:
    """Test ValidationResult class."""

    def test_init_basic(self):
        """Test basic initialization."""
        result = ValidationResult(
            passed=True,
            checks={"test": True},
            metrics={"accuracy": 0.95},
            messages=["All good"],
        )

        assert result.passed is True
        assert result.checks == {"test": True}
        assert result.metrics == {"accuracy": 0.95}
        assert result.messages == ["All good"]
        assert result.warnings == []
        assert result.errors == []

    def test_init_with_warnings_errors(self):
        """Test initialization with warnings and errors."""
        result = ValidationResult(
            passed=False,
            checks={"test": False},
            metrics={},
            messages=[],
            warnings=["Minor issue"],
            errors=["Major problem"],
        )

        assert result.warnings == ["Minor issue"]
        assert result.errors == ["Major problem"]

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = ValidationResult(
            passed=True,
            checks={"ppl_check": True},
            metrics={"ppl_ratio": 1.25},
            messages=["Success"],
            warnings=["Warning"],
            errors=[],
        )

        expected = {
            "passed": True,
            "checks": {"ppl_check": True},
            "metrics": {"ppl_ratio": 1.25},
            "messages": ["Success"],
            "warnings": ["Warning"],
            "errors": [],
        }

        assert result.to_dict() == expected

    def test_summary_passed(self):
        """Test summary for passed validation."""
        result = ValidationResult(
            passed=True,
            checks={"check1": True, "check2": True},
            metrics={},
            messages=["All checks passed"],
            warnings=[],
            errors=[],
        )

        summary = result.summary()
        assert "✓ PASSED" in summary
        assert "(2/2 checks passed)" in summary
        assert "✓ check1" in summary
        assert "✓ check2" in summary
        assert "All checks passed" in summary

    def test_summary_failed(self):
        """Test summary for failed validation."""
        result = ValidationResult(
            passed=False,
            checks={"check1": True, "check2": False},
            metrics={},
            messages=["Some issues"],
            warnings=["Warning message"],
            errors=["Error message"],
        )

        summary = result.summary()
        assert "✗ FAILED" in summary
        assert "(1/2 checks passed)" in summary
        assert "✓ check1" in summary
        assert "✗ check2" in summary
        assert "Some issues" in summary
        assert "⚠️ Warning message" in summary
        assert "❌ Error message" in summary


class TestValidateAgainstBaseline:
    """Test validate_against_baseline function."""

    def test_against_baseline_success(self):
        """Test successful validation with all checks passing."""
        run_report = {
            "metrics": {
                "primary_metric": {"kind": "ppl_causal", "ratio_vs_baseline": 1.25},
                "invariants_passed": True,
            },
            "param_reduction_ratio": 0.020,  # Use exactly same value to pass tolerance
            "heads_pruned": 16,
            "neurons_pruned": 1024,
            "layers_modified": 8,
        }

        baseline = {
            "ratio_vs_baseline": 1.26,
            "param_reduction_ratio": 0.020,  # Same value
            "heads_pruned": 16,
            "neurons_pruned": 1024,
            "layers_modified": 8,
        }

        result = validate_against_baseline(run_report, baseline)

        assert result.passed is True
        assert "ratio_tolerance" in result.checks
        assert "param_ratio_tolerance" in result.checks
        assert "ratio_bounds" in result.checks
        assert all(result.checks.values())

    def test_against_baseline_ppl_ratio_tolerance_failure(self):
        """Test PPL ratio tolerance failure."""
        run_report = {
            "metrics": {
                "primary_metric": {"kind": "ppl_causal", "ratio_vs_baseline": 1.50}
            }
        }
        baseline = {"ratio_vs_baseline": 1.25}

        result = validate_against_baseline(run_report, baseline, tol_ratio=0.02)

        assert result.passed is False
        assert result.checks["ratio_tolerance"] is False
        assert "Primary metric ratio deviation" in " ".join(result.messages)

    def test_against_baseline_param_ratio_tolerance_failure(self):
        """Test parameter ratio tolerance failure."""
        run_report = {
            "ppl_ratio": 1.25,
            "param_reduction_ratio": 0.10,  # Way too high
        }
        baseline = {"ppl_ratio": 1.25, "param_reduction_ratio": 0.02}

        result = validate_against_baseline(run_report, baseline, tol_param_ratio=0.02)

        assert result.passed is False
        assert result.checks["param_ratio_tolerance"] is False
        assert "Parameter ratio deviation" in " ".join(result.messages)

    def test_against_baseline_ppl_bounds_failure(self):
        """Test PPL bounds check failure."""
        run_report = {
            "metrics": {
                "primary_metric": {"kind": "ppl_causal", "ratio_vs_baseline": 2.0}
            }
        }
        baseline = {"ratio_vs_baseline": 1.25}

        result = validate_against_baseline(
            run_report, baseline, ratio_bounds=(1.25, 1.32)
        )

        assert result.passed is False
        assert result.checks["ratio_bounds"] is False
        assert "outside acceptable bounds" in " ".join(result.messages)

    def test_against_baseline_alternative_metric_extraction(self):
        """Test alternative ways of extracting metrics."""
        run_report = {
            "metrics": {
                "primary_metric": {"kind": "ppl_causal", "ratio_vs_baseline": 1.25}
            },
            "parameters_removed": 1000,
            "original_params": 50000,
        }
        baseline = {"ratio_vs_baseline": 1.25, "param_reduction_ratio": 0.02}

        result = validate_against_baseline(run_report, baseline)

        assert result.metrics.get("current_ratio") == 1.25
        assert result.metrics["current_param_ratio"] == 0.02

    def test_against_baseline_missing_metrics(self):
        """Test handling of missing metrics in run report."""
        run_report = {}  # No metrics
        baseline = {"ppl_ratio": 1.25, "param_reduction_ratio": 0.02}

        result = validate_against_baseline(run_report, baseline)

        assert result.passed is False
        assert len(result.errors) >= 2
        assert "Cannot extract ratio_vs_baseline" in " ".join(result.errors)
        assert "Cannot extract parameter reduction ratio" in " ".join(result.errors)

    def test_against_baseline_missing_baseline_metrics(self):
        """Test handling of missing baseline metrics."""
        run_report = {"ppl_ratio": 1.25, "param_reduction_ratio": 0.02}
        baseline = {}  # No baseline metrics

        result = validate_against_baseline(run_report, baseline)

        assert result.passed is False
        assert "Baseline missing ratio_vs_baseline" in result.errors
        assert "Baseline missing param_reduction_ratio" in result.errors

    def test_against_baseline_structural_exact_disabled(self):
        """Test with structural validation disabled."""
        run_report = {"ppl_ratio": 1.25}
        baseline = {"ppl_ratio": 1.25}

        result = validate_against_baseline(run_report, baseline, structural_exact=False)

        assert "structural_counts" not in result.checks
        assert result.passed is False

    def test_against_baseline_exception_handling(self):
        """Test exception handling in validation."""
        # Create invalid data that will cause an exception
        run_report = {
            "metrics": {
                "primary_metric": {"kind": "ppl_causal", "ratio_vs_baseline": "invalid"}
            }
        }
        baseline = {"ratio_vs_baseline": 1.25}

        result = validate_against_baseline(run_report, baseline)

        assert result.passed is False
        # PM-only: treat as extract failure rather than exception flag
        assert result.checks.get("ratio_tolerance") is False
        assert len(result.errors) > 0
        assert "Cannot extract ratio_vs_baseline" in result.errors[0]

    def test_against_baseline_accuracy_delta_bounds_success(self):
        run_report = {
            "metrics": {
                "primary_metric": {
                    "kind": "accuracy",
                    "delta_vs_baseline_pp": -0.4,
                },
                "invariants_passed": True,
            },
            "param_reduction_ratio": 0.02,
        }
        baseline = {"param_reduction_ratio": 0.02}

        result = validate_against_baseline(
            run_report,
            baseline,
            delta_bounds_pp=(-1.0, 0.0),
            structural_exact=False,
        )

        assert result.passed is True
        assert result.checks["delta_bounds_pp"] is True
        assert "Δpp -0.40 within acceptable bounds" in " ".join(result.messages)

    def test_against_baseline_accuracy_without_delta_bounds_sets_false(self):
        run_report = {
            "metrics": {"primary_metric": {"kind": "accuracy"}},
            "param_reduction_ratio": 0.02,
        }
        baseline = {"param_reduction_ratio": 0.02}

        result = validate_against_baseline(
            run_report,
            baseline,
            delta_bounds_pp=None,
            structural_exact=False,
        )

        assert result.passed is False
        assert result.checks["delta_bounds_pp"] is False
        assert "Cannot extract delta_vs_baseline_pp" in " ".join(result.errors)

    def test_against_baseline_accuracy_with_invalid_delta_bounds_skips_check(self):
        run_report = {
            "metrics": {
                "primary_metric": {
                    "kind": "accuracy",
                    "delta_vs_baseline_pp": -0.4,
                },
                "invariants_passed": True,
            },
            "param_reduction_ratio": 0.02,
        }
        baseline = {"param_reduction_ratio": 0.02}

        result = validate_against_baseline(
            run_report,
            baseline,
            delta_bounds_pp=(0.0,),
            structural_exact=False,
        )

        assert result.passed is True
        assert "delta_bounds_pp" not in result.checks

    def test_against_baseline_accuracy_delta_bounds_failure_message(self):
        run_report = {
            "metrics": {
                "primary_metric": {
                    "kind": "accuracy",
                    "delta_vs_baseline_pp": -2.0,
                }
            },
            "param_reduction_ratio": 0.02,
        }
        baseline = {"param_reduction_ratio": 0.02}

        result = validate_against_baseline(
            run_report,
            baseline,
            delta_bounds_pp=(-1.0, 0.0),
            structural_exact=False,
        )

        assert result.passed is False
        assert result.checks["delta_bounds_pp"] is False
        assert "outside acceptable bounds" in " ".join(result.messages)

    def test_against_baseline_invariants_failure_adds_error(self):
        run_report = {
            "metrics": {
                "primary_metric": {"kind": "ppl_causal", "ratio_vs_baseline": 1.25}
            },
            "param_reduction_ratio": 0.02,
            "guard_reports": {"invariants_guard": {"passed": False}},
        }
        baseline = {"ratio_vs_baseline": 1.25, "param_reduction_ratio": 0.02}

        result = validate_against_baseline(
            run_report,
            baseline,
            structural_exact=False,
        )

        assert result.passed is False
        assert result.checks["invariants"] is False
        assert "Model invariants evidence is missing or failed" in result.errors

    def test_against_baseline_pm_kind_lookup_failure_falls_back_to_none(self):
        class KindBoomDict(dict):
            def get(self, key, default=None):
                if key == "kind":
                    raise RuntimeError("boom")
                return super().get(key, default)

        run_report = {
            "metrics": {
                "primary_metric": KindBoomDict(
                    {"ratio_vs_baseline": 1.25, "kind": "ppl_causal"}
                ),
                "invariants_passed": True,
            },
            "param_reduction_ratio": 0.02,
        }
        baseline = {"ratio_vs_baseline": 1.25, "param_reduction_ratio": 0.02}

        result = validate_against_baseline(
            run_report,
            baseline,
            structural_exact=False,
        )

        assert result.passed is True
        assert result.checks["ratio_tolerance"] is True
        assert result.metrics["current_ratio"] == 1.25

    def test_against_baseline_unexpected_exception_returns_validation_error(self):
        class ExplodingMetrics(dict):
            def get(self, key, default=None):
                if key == "primary_metric":
                    raise RuntimeError("explode")
                return super().get(key, default)

        run_report = {"metrics": ExplodingMetrics({"sentinel": True})}
        baseline = {"ratio_vs_baseline": 1.25, "param_reduction_ratio": 0.02}

        result = validate_against_baseline(run_report, baseline)

        assert result.passed is False
        assert result.checks == {"validation_error": False}
        assert "Validation failed with exception: explode" in result.errors


class TestDriftAndOverheadValidation:
    def test_validate_drift_gate_success(self):
        result = validate_drift_gate(
            {"metrics": {"primary_metric": {"preview": 10.0, "final": 10.2}}}
        )

        assert result.passed is True
        assert result.checks["drift_gate"] is True
        assert result.metrics["drift_ratio"] == pytest.approx(1.02)

    def test_validate_drift_gate_failure(self):
        result = validate_drift_gate(
            {"metrics": {"primary_metric": {"preview": 10.0, "final": 11.0}}}
        )

        assert result.passed is False
        assert result.checks["drift_gate"] is False
        assert "Drift gate FAILED" in " ".join(result.errors)

    def test_validate_drift_gate_missing_values(self):
        result = validate_drift_gate({"metrics": {"primary_metric": {"preview": 0.0}}})

        assert result.passed is False
        assert result.checks["drift_gate"] is False
        assert "Cannot calculate drift" in " ".join(result.errors)

    def test_validate_drift_gate_exception_path(self):
        class ExplodingMetrics(dict):
            def get(self, key, default=None):
                if key == "primary_metric":
                    raise RuntimeError("drift boom")
                return super().get(key, default)

        result = validate_drift_gate({"metrics": ExplodingMetrics({"sentinel": True})})

        assert result.passed is False
        assert result.checks == {"drift_gate_error": False}
        assert "Drift gate validation failed: drift boom" in result.errors

    def test_validate_guard_metric_impact_success(self):
        result = validate_guard_metric_impact(
            {"metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}}},
            {"metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.05}}},
            degradation_limit=0.01,
        )

        assert result.passed is True
        assert result.checks["guard_metric_impact"] is True
        assert result.metrics["display_value"] == pytest.approx(0.5)

    def test_validate_guard_metric_impact_failure(self):
        result = validate_guard_metric_impact(
            {"metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}}},
            {"metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.3}}},
            degradation_limit=0.01,
        )

        assert result.passed is False
        assert result.checks["guard_metric_impact"] is False
        assert "Guard metric impact FAILED" in " ".join(result.errors)

    def test_validate_guard_metric_impact_missing_values(self):
        result = validate_guard_metric_impact(
            {"metrics": {"primary_metric": {"kind": "ppl_causal", "final": 0.0}}},
            {"metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.3}}},
        )

        assert result.passed is False
        assert result.checks["guard_metric_impact"] is False
        assert "Cannot calculate guard metric impact" in " ".join(result.errors)

    def test_validate_guard_metric_impact_rejects_non_mapping_primary_metric(self):
        result = validate_guard_metric_impact(
            {"metrics": {"primary_metric": "bad"}},
            {"metrics": {"primary_metric": "bad"}},
        )

        assert result.passed is False
        assert result.checks["guard_metric_impact"] is False
        assert "Cannot calculate guard metric impact" in " ".join(result.errors)

    def test_validate_guard_metric_impact_exception_path(self):
        class ExplodingMetrics(dict):
            def get(self, key, default=None):
                if key == "primary_metric":
                    raise RuntimeError("impact boom")
                return super().get(key, default)

        result = validate_guard_metric_impact(
            {"metrics": ExplodingMetrics({"sentinel": True})},
            {"metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}}},
        )

        assert result.passed is False
        assert result.checks == {"guard_metric_impact_error": False}
        assert "Guard metric impact validation failed: impact boom" in result.errors


class TestValidateStructuralCounts:
    """Test _validate_structural_counts function."""

    def test_exact_structural_matches(self):
        """Test exact structural count matches."""
        run_report = {"heads_pruned": 16, "neurons_pruned": 1024, "layers_modified": 8}
        baseline = {"heads_pruned": 16, "neurons_pruned": 1024, "layers_modified": 8}

        result = _validate_structural_counts(run_report, baseline)

        assert all(result["checks"].values())
        assert len(result["warnings"]) == 0
        assert "count matches" in " ".join(result["messages"])

    def test_structural_mismatches(self):
        """Test structural count mismatches."""
        run_report = {"heads_pruned": 16, "neurons_pruned": 1024, "layers_modified": 8}
        baseline = {
            "heads_pruned": 20,  # Different
            "neurons_pruned": 2048,  # Different
            "layers_modified": 10,  # Different
        }

        result = _validate_structural_counts(run_report, baseline)

        assert not any(result["checks"].values())
        assert "mismatch" in " ".join(result["messages"])

    def test_structural_missing_data(self):
        """Test handling of missing structural data."""
        run_report = {}  # No structural data
        baseline = {"heads_pruned": 16}

        result = _validate_structural_counts(run_report, baseline)

        assert result["checks"]["layers_count_exact"] is False
        assert len(result["warnings"]) > 0
        assert "Cannot validate" in " ".join(result["warnings"])

    def test_structural_nested_metrics(self):
        """Test extracting structural counts from nested metrics."""
        run_report = {
            "metrics": {
                "heads_pruned": 16,
                "neurons_pruned": 1024,
                "layers_modified": 8,
            }
        }
        baseline = {"heads_pruned": 16, "neurons_pruned": 1024, "layers_modified": 8}

        result = _validate_structural_counts(run_report, baseline)

        assert all(result["checks"].values())


class TestValidateInvariants:
    """Test _validate_invariants function."""

    def test_invariants_passed_in_guard_reports(self):
        """Test invariants validation from guard reports."""
        run_report = {"guard_reports": {"invariants_guard": {"passed": True}}}

        result = _validate_invariants(run_report)
        assert result is True

    def test_invariants_failed_in_guard_reports(self):
        """Test failed invariants in guard reports."""
        run_report = {"guard_reports": {"invariants_checker": {"passed": False}}}

        result = _validate_invariants(run_report)
        assert result is False

    def test_invariants_in_metrics(self):
        """Test invariants validation from metrics."""
        run_report = {"metrics": {"invariants_passed": True}}

        result = _validate_invariants(run_report)
        assert result is True

    def test_no_invariants_found(self):
        """Test when no invariants check is found."""
        run_report = {
            "guard_reports": {"other_guard": {"passed": True}},
            "metrics": {"accuracy": 0.95},
        }

        result = _validate_invariants(run_report)
        assert result is False


class TestValidateFileIO:
    """Test file I/O functions."""

    def test_fileio_load_baseline_success(self):
        """Test successful baseline loading."""
        test_baseline = {"ppl_ratio": 1.25, "param_reduction_ratio": 0.02}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_baseline, f)
            baseline_path = Path(f.name)

        try:
            loaded = load_baseline(baseline_path)
            assert loaded == test_baseline
        finally:
            baseline_path.unlink()

    def test_fileio_load_baseline_missing_file(self):
        """Test loading non-existent baseline file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_baseline(Path("/nonexistent/file.json"))

        assert "Baseline file not found" in str(exc_info.value)

    def test_fileio_load_baseline_invalid_json(self):
        """Test loading invalid JSON baseline."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            baseline_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                load_baseline(baseline_path)
            assert "Invalid JSON" in str(exc_info.value)
        finally:
            baseline_path.unlink()

    def test_fileio_load_baseline_rejects_non_object_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([1, 2, 3], f)
            baseline_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                load_baseline(baseline_path)
            assert "Baseline file must contain a JSON object" in str(exc_info.value)
        finally:
            baseline_path.unlink()

    def test_fileio_save_baseline(self):
        """Test saving baseline to file."""
        test_baseline = {"ppl_ratio": 1.25, "test": True}

        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_path = Path(temp_dir) / "subdir" / "baseline.json"

            save_baseline(test_baseline, baseline_path)

            # Verify file was created and contains correct data
            assert baseline_path.exists()
            with open(baseline_path) as f:
                loaded = json.load(f)
            assert loaded == test_baseline
