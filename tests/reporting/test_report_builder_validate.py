# ruff: noqa: F405
from __future__ import annotations

from tests.reporting._support_report_builder import *  # noqa: F401,F403,F405


class TestValidateEvaluationReport:
    """Test validate_report function."""

    def test_valid_evaluation_report(self):
        """Test validation of a valid evaluation_report (PM-only)."""
        evaluation_report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "run_id": "test123",
            "meta": {"model_id": "m"},
            "auto": {"tier": "balanced", "probes_used": 0, "target_pm_ratio": None},
            "dataset": {
                "provider": "dummy",
                "seq_len": 8,
                "windows": {"preview": 1, "final": 1},
            },
            "baseline_ref": {},
            "primary_metric": {
                "kind": "ppl_causal",
                "final": 11.0,
                "preview": 10.0,
                "ratio_vs_baseline": 1.1,
                "display_ci": [10.0, 12.0],
            },
            "invariants": {},
            "spectral": {},
            "rmt": {},
            "variance": {},
            "structure": {},
            "policies": {},
            "plugins": {"adapter": {}, "edit": {}, "guards": []},
            "artifacts": {"events_path": "", "logs_path": "", "generated_at": "now"},
            "validation": {
                "primary_metric_acceptable": True,
                "preview_final_drift_acceptable": True,
                "invariants_pass": True,
                "spectral_stable": True,
                "rmt_stable": True,
                "guard_overhead_acceptable": True,
            },
        }

        assert validate_report(evaluation_report) is True

    def test_invalid_schema_version(self):
        """Test validation fails with wrong schema version."""
        evaluation_report = {"schema_version": "wrong-version", "run_id": "test123"}

        assert validate_report(evaluation_report) is False

    def test_missing_required_fields(self):
        """Test validation fails with missing required fields."""
        evaluation_report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "run_id": "test123",
            # Missing other required fields
        }

        assert validate_report(evaluation_report) is False

    def test_invalid_ppl_metrics(self):
        """Test validation fails with invalid PPL metrics."""
        evaluation_report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "run_id": "test123",
            "meta": {},
            "auto": {},
            "dataset": {},
            "baseline_ref": {},
            "ppl": {
                "preview": "not_a_number",  # Should be numeric
                "final": 11.0,
                "ratio_vs_baseline": 1.1,
                "drift": 1.05,
            },
            "invariants": {},
            "spectral": {},
            "rmt": {},
            "variance": {},
            "structure": {},
            "policies": {},
            "plugins": {"adapter": {}, "edit": {}, "guards": []},
            "artifacts": {},
            "validation": {
                "primary_metric_acceptable": True,
                "invariants_pass": True,
                "spectral_stable": True,
                "rmt_stable": True,
            },
        }

        assert validate_report(evaluation_report) is False

    def test_invalid_validation_flags(self):
        """Test validation fails with invalid validation flags."""
        evaluation_report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "run_id": "test123",
            "meta": {},
            "auto": {},
            "dataset": {},
            "baseline_ref": {},
            "ppl": {
                "preview": 10.0,
                "final": 11.0,
                "ratio_vs_baseline": 1.1,
                "drift": 1.05,
            },
            "invariants": {},
            "spectral": {},
            "rmt": {},
            "variance": {},
            "structure": {},
            "policies": {},
            "plugins": {"adapter": {}, "edit": {}, "guards": []},
            "artifacts": {},
            "validation": {
                "primary_metric_acceptable": "not_boolean",  # Should be boolean
                "preview_final_drift_acceptable": True,
                "invariants_pass": True,
                "spectral_stable": True,
                "rmt_stable": True,
                "guard_overhead_acceptable": True,
            },
        }

        assert validate_report(evaluation_report) is False

    def test_exception_handling(self):
        """Test validation handles exceptions gracefully."""
        # Invalid structure that would raise exceptions in try-except block
        # Test with dict that raises KeyError/TypeError/ValueError (caught exceptions)
        evaluation_report = {"invalid": "structure"}
        assert validate_report(evaluation_report) is False

        # Test with malformed dictionary structure
        evaluation_report = {"schema_version": "v1", "ppl": "not_a_dict"}
        assert validate_report(evaluation_report) is False

        # Test AttributeError case (None input) - should raise AttributeError
        with pytest.raises(AttributeError):
            validate_report(None)

        # Test AttributeError case (string input) - should raise AttributeError
        with pytest.raises(AttributeError):
            validate_report("not_a_dict")
