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
    create_baseline_from_report,
    validate_against_baseline,
    validate_gpt2_small_wt2_baseline,
)

if __name__ == "__main__":
    pytest.main([__file__])


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

    def test_create_baseline_ignores_unparseable_ratio(self):
        run_report = {
            "metrics": {
                "layers_modified": 3,
                "primary_metric": {
                    "kind": "ppl_causal",
                    "ratio_vs_baseline": object(),
                },
            },
            "parameters_removed": 10,
            "original_params": 100,
        }

        baseline = create_baseline_from_report(run_report)

        assert "ratio_vs_baseline" not in baseline
        assert baseline["param_reduction_ratio"] == 0.1
        assert baseline["layers_modified"] == 3


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
