"""
Comprehensive test coverage for InvarLock validation module.
Tests for validate.py to achieve 80%+ coverage.
"""

import json
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

from invarlock.reporting.validate import (
    ValidationResult,
    _validate_invariants,
    _validate_structural_counts,
    create_baseline_from_report,
    load_baseline,
    save_baseline,
    validate_against_baseline,
    validate_gpt2_small_wt2_baseline,
)


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
                "primary_metric": {"kind": "ppl_causal", "ratio_vs_baseline": 1.25}
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

        assert result.checks["structural_counts"] is True

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

        # Should not fail on missing data
        assert all(result["checks"].values())
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
        assert result is None


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


class TestValidateCreateBaseline:
    """Test create_baseline_from_report function."""

    def test_create_baseline_basic(self):
        """Test basic baseline creation."""
        run_report = {
            "metrics": {
                "primary_metric": {"kind": "ppl_causal", "ratio_vs_baseline": 1.25}
            },
            "param_reduction_ratio": 0.02,
            "heads_pruned": 16,
            "neurons_pruned": 1024,
            "layers_modified": 8,
        }

        baseline = create_baseline_from_report(run_report)

        assert baseline["ratio_vs_baseline"] == 1.25
        assert baseline["param_reduction_ratio"] == 0.02
        assert baseline["heads_pruned"] == 16
        assert baseline["neurons_pruned"] == 1024
        assert baseline["layers_modified"] == 8
        assert baseline["baseline_created"] is True
        assert baseline["source"] == "run_report"

    def test_create_baseline_alternative_metrics(self):
        """Test baseline creation with alternative metric names."""
        run_report = {
            "metrics": {
                "primary_metric": {"kind": "ppl_causal", "ratio_vs_baseline": 1.25},
                "heads_pruned": 20,
                "neurons_pruned": 2048,
            },
            "parameters_removed": 1000,
            "original_params": 50000,
        }

        baseline = create_baseline_from_report(run_report)

        assert baseline["ratio_vs_baseline"] == 1.25
        assert baseline["param_reduction_ratio"] == 0.02
        assert baseline["heads_pruned"] == 20
        assert baseline["neurons_pruned"] == 2048

    def test_create_baseline_with_sparsity(self):
        """Test baseline creation with sparsity metrics."""
        run_report = {
            "ppl_ratio": 1.25,
            "actual_sparsity": {
                "head_sparsity": 0.1,
                "neuron_sparsity": 0.15,
                "weight_sparsity": 0.05,
            },
        }

        baseline = create_baseline_from_report(run_report)

        assert baseline["head_sparsity"] == 0.1
        assert baseline["neuron_sparsity"] == 0.15
        assert baseline["weight_sparsity"] == 0.05

    def test_create_baseline_minimal_data(self):
        """Test baseline creation with minimal data."""
        run_report = {}

        baseline = create_baseline_from_report(run_report)

        # Should still have metadata
        assert baseline["baseline_created"] is True
        assert baseline["source"] == "run_report"


class TestValidateGpt2Baseline:
    """Test validate_gpt2_small_wt2_baseline function."""

    @patch("invarlock.reporting.validate.load_baseline")
    def test_gpt2_validation_with_baseline_file(self, mock_load):
        """Test GPT-2 validation with existing baseline file."""
        mock_baseline = {
            "ratio_vs_baseline": 1.285,
            "param_reduction_ratio": 0.022,
            "heads_pruned": 16,
            "neurons_pruned": 1024,
            "layers_modified": 8,
        }
        mock_load.return_value = mock_baseline

        run_report = {
            "metrics": {
                "primary_metric": {"kind": "ppl_causal", "ratio_vs_baseline": 1.285}
            },
            "param_reduction_ratio": 0.022,
            "heads_pruned": 16,
            "neurons_pruned": 1024,
            "layers_modified": 8,
        }

        result = validate_gpt2_small_wt2_baseline(run_report)

        assert result.passed is True
        mock_load.assert_called_once()

    @patch("invarlock.reporting.validate.load_baseline")
    def test_gpt2_validation_missing_baseline(self, mock_load):
        """Test GPT-2 validation with missing baseline file."""
        mock_load.side_effect = FileNotFoundError("File not found")

        run_report = {"ppl_ratio": 1.285, "param_reduction_ratio": 0.022}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_gpt2_small_wt2_baseline(run_report)

            # Should use default baseline and show warning
            assert len(w) == 1
            assert "Baseline file not found" in str(w[0].message)

        # Should still validate against default values
        assert isinstance(result, ValidationResult)

    def test_gpt2_validation_custom_baseline_path(self):
        """Test GPT-2 validation with custom baseline path."""
        test_baseline = {"ppl_ratio": 1.30, "param_reduction_ratio": 0.025}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_baseline, f)
            baseline_path = Path(f.name)

        try:
            run_report = {
                "metrics": {
                    "primary_metric": {"kind": "ppl_causal", "ratio_vs_baseline": 1.295}
                },
                "param_reduction_ratio": 0.024,
            }
            result = validate_gpt2_small_wt2_baseline(run_report, baseline_path)

            assert isinstance(result, ValidationResult)
        finally:
            baseline_path.unlink()


class TestValidateIntegration:
    """Test integration scenarios."""

    def test_integration_full_validation_workflow(self):
        """Test complete validation workflow."""
        # Create comprehensive run report
        run_report = {
            "metrics": {
                "primary_metric": {"kind": "ppl_causal", "ratio_vs_baseline": 1.28},
                "invariants_passed": True,
            },
            "param_reduction_ratio": 0.021,
            "heads_pruned": 16,
            "neurons_pruned": 1024,
            "layers_modified": 8,
            "guard_reports": {"invariants_guard": {"passed": True}},
            "actual_sparsity": {"head_sparsity": 0.1, "neuron_sparsity": 0.15},
        }

        # Create baseline from report
        baseline = create_baseline_from_report(run_report)

        # Validate against baseline
        result = validate_against_baseline(run_report, baseline)

        assert result.passed is True
        assert "invariants" in result.checks
        assert result.checks["invariants"] is True

        # Test serialization
        result_dict = result.to_dict()
        assert "passed" in result_dict
        assert "checks" in result_dict

        # Test summary
        summary = result.summary()
        assert "PASSED" in summary

    def test_integration_validation_mixed_results(self):
        """Test validation with some passing and some failing checks."""
        run_report = {
            "metrics": {
                "primary_metric": {"kind": "ppl_causal", "ratio_vs_baseline": 2.0}
            },  # Too high
            "param_reduction_ratio": 0.02,  # Good
            "heads_pruned": 16,  # Good
            "neurons_pruned": 999,  # Different from baseline (will fail)
        }

        baseline = {
            "ratio_vs_baseline": 1.25,
            "param_reduction_ratio": 0.02,
            "heads_pruned": 16,
            "neurons_pruned": 1024,
        }

        result = validate_against_baseline(run_report, baseline)

        assert result.passed is False
        assert result.checks["param_ratio_tolerance"] is True  # Should pass
        assert result.checks["ratio_tolerance"] is False  # Should fail
        assert result.checks["ratio_bounds"] is False  # Should fail

        summary = result.summary()
        assert "FAILED" in summary
        # Account for all possible checks (ppl_ratio_tolerance, param_ratio_tolerance, ppl_bounds, structural checks)
        assert "checks passed" in summary


if __name__ == "__main__":
    pytest.main([__file__])
