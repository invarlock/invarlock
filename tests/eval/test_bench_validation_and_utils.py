"""
Test coverage for bench.py - targeting missing coverage areas.

This module provides comprehensive tests for the InvarLock benchmark module,
focusing on areas likely to be uncovered to push coverage from 76% to 80%+.
"""

import pytest

from invarlock.eval.bench import (
    BenchmarkConfig,
    BenchmarkSummary,
    RunResult,
    ScenarioConfig,
    ScenarioResult,
    ValidationGates,
    _config_to_dict,
    _scenario_result_to_dict,
    _summary_to_step14_json,
    generate_scenarios,
    resolve_epsilon_from_runtime,
)
from invarlock.reporting.report_types import create_empty_report


def _report_with_artifacts(report_path: str = "report.json") -> dict[str, object]:
    report = create_empty_report()
    report["artifacts"]["report_path"] = report_path
    return report


class TestValidationGates:
    """Test validation gate logic."""

    def test_validate_catastrophic_spike_rate_pass(self):
        """Test catastrophic spike validation passes."""
        comparison = {"catastrophic_spike": False}
        assert ValidationGates.validate_catastrophic_spike_rate(comparison) is True

    def test_validate_catastrophic_spike_rate_fail(self):
        """Test catastrophic spike validation fails."""
        comparison = {"catastrophic_spike": True}
        assert ValidationGates.validate_catastrophic_spike_rate(comparison) is False

    def test_validate_catastrophic_spike_rate_missing_key(self):
        """Test catastrophic spike validation with missing key."""
        comparison = {}
        assert ValidationGates.validate_catastrophic_spike_rate(comparison) is True

    def test_validate_tying_violations_pass(self):
        """Test tying violations validation passes."""
        comparison = {"tying_violations_post": 0}
        assert ValidationGates.validate_tying_violations(comparison) is True

    def test_validate_tying_violations_fail(self):
        """Test tying violations validation fails."""
        comparison = {"tying_violations_post": 1}
        assert ValidationGates.validate_tying_violations(comparison) is False

    def test_validate_rmt_outliers_boundary_conditions(self):
        """Test RMT outliers validation at boundary conditions."""
        # Exact equality case
        comparison = {"rmt_outliers_bare": 2, "rmt_outliers_guarded": 2}
        assert ValidationGates.validate_rmt_outliers(comparison, 0.0) is True

        # Just within threshold
        comparison = {"rmt_outliers_bare": 2, "rmt_outliers_guarded": 3}
        assert (
            ValidationGates.validate_rmt_outliers(comparison, 0.5) is True
        )  # ceil(2 * 1.5) = 3

        # Just over threshold
        comparison = {"rmt_outliers_bare": 2, "rmt_outliers_guarded": 4}
        assert (
            ValidationGates.validate_rmt_outliers(comparison, 0.5) is False
        )  # ceil(2 * 1.5) = 3, got 4

    def test_validate_primary_metric_overhead_nan_handling(self):
        """Test primary metric overhead validation with NaN."""
        comparison = {"primary_metric_overhead": float("nan")}
        assert ValidationGates.validate_primary_metric_overhead(comparison) is True

    def test_validate_time_overhead_nan_handling(self):
        """Test time overhead validation with NaN."""
        comparison = {"guard_overhead_time": float("nan")}
        assert ValidationGates.validate_time_overhead(comparison) is True

    def test_validate_memory_overhead_nan_handling(self):
        """Test memory overhead validation with NaN."""
        comparison = {"guard_overhead_mem": float("nan")}
        assert ValidationGates.validate_memory_overhead(comparison) is True


class TestResolveEpsilonFromRuntimeEdgeCases:
    """Additional epsilon resolution edge cases."""

    def test_resolve_epsilon_from_runtime_rmt_without_deadband(self):
        """RMT guard without deadband should fall back to default epsilon."""
        report = create_empty_report()
        report["guards"] = [{"name": "rmt", "policy": {}}]

        epsilon = resolve_epsilon_from_runtime(report)
        assert epsilon == 0.10

    def test_validate_primary_metric_overhead_thresholds(self):
        comparison = {"primary_metric_overhead": 0.009}
        assert ValidationGates.validate_primary_metric_overhead(
            comparison, threshold=0.01
        )
        comparison["primary_metric_overhead"] = 0.02
        assert (
            ValidationGates.validate_primary_metric_overhead(comparison, threshold=0.01)
            is False
        )

    def test_validate_time_and_memory_overhead_thresholds(self):
        comparison = {"guard_overhead_time": 0.14, "guard_overhead_mem": 0.09}
        assert ValidationGates.validate_time_overhead(comparison, threshold=0.15)
        assert ValidationGates.validate_memory_overhead(comparison, threshold=0.10)
        comparison["guard_overhead_time"] = 0.2
        comparison["guard_overhead_mem"] = 0.11
        assert (
            ValidationGates.validate_time_overhead(comparison, threshold=0.15) is False
        )
        assert (
            ValidationGates.validate_memory_overhead(comparison, threshold=0.10)
            is False
        )

    def test_validate_all_gates_comprehensive(self):
        """Test comprehensive gate validation."""
        comparison = {
            "catastrophic_spike": False,
            "tying_violations_post": 0,
            "rmt_outliers_bare": 2,
            "rmt_outliers_guarded": 2,
            "primary_metric_overhead": 0.005,  # 0.5%
            "guard_overhead_time": 0.10,  # 10%
            "guard_overhead_mem": 0.08,  # 8%
        }

        config = BenchmarkConfig(edits=["structured"], tiers=["balanced"], probes=[0])
        gates = ValidationGates.validate_all_gates(comparison, config, 0.1)

        assert all(gates.values())
        assert len(gates) == 6  # spike, tying, rmt, quality, time, mem


class TestUtilityFunctions:
    """Test utility functions."""

    def test_generate_scenarios_cartesian_product(self):
        """Test scenario generation creates Cartesian product."""
        config = BenchmarkConfig(
            edits=["structured", "quant_rtn"],
            tiers=["balanced", "aggressive"],
            probes=[0, 2],
        )

        scenarios = generate_scenarios(config)
        assert len(scenarios) == 8  # 2 × 2 × 2

        # Check all combinations exist
        combinations = [(s.edit, s.tier, s.probes) for s in scenarios]
        expected = [
            ("structured", "balanced", 0),
            ("structured", "balanced", 2),
            ("structured", "aggressive", 0),
            ("structured", "aggressive", 2),
            ("quant_rtn", "balanced", 0),
            ("quant_rtn", "balanced", 2),
            ("quant_rtn", "aggressive", 0),
            ("quant_rtn", "aggressive", 2),
        ]
        assert sorted(combinations) == sorted(expected)

    def test_resolve_epsilon_from_runtime_with_rmt_guard(self):
        """Test epsilon resolution from RMT guard report."""
        report = create_empty_report()
        report["guards"] = [
            {"name": "rmt", "policy": {"deadband": 0.05}},
            {"name": "spectral", "policy": {"sigma_quantile": 0.9}},
        ]

        epsilon = resolve_epsilon_from_runtime(report)
        assert epsilon == 0.05

    def test_resolve_epsilon_from_runtime_no_rmt_guard(self):
        """Test epsilon resolution fallback when no RMT guard."""
        report = create_empty_report()
        report["guards"] = [{"name": "spectral", "policy": {"sigma_quantile": 0.9}}]

        epsilon = resolve_epsilon_from_runtime(report)
        assert epsilon == 0.10  # Default fallback

    def test_resolve_epsilon_from_runtime_empty_guards(self):
        """Test epsilon resolution with empty guards."""
        report = create_empty_report()

        epsilon = resolve_epsilon_from_runtime(report)
        assert epsilon == 0.10  # Default fallback


class TestOutputGeneration:
    """Test output generation functions."""

    def test_scenario_result_to_dict_complete(self):
        """Test scenario result to dict conversion."""
        config = ScenarioConfig(edit="structured", tier="balanced", probes=2)
        bare_result = RunResult("bare", create_empty_report(), success=True)
        guarded_result = RunResult("guarded", create_empty_report(), success=True)

        scenario_result = ScenarioResult(
            config=config,
            bare_result=bare_result,
            guarded_result=guarded_result,
            metrics={"primary_metric_overhead": 0.01},
            gates={"spike": True, "quality": True},
            probes_used=2,
            epsilon_used=0.1,
        )

        result_dict = _scenario_result_to_dict(scenario_result)

        assert result_dict["edit"] == "structured"
        assert result_dict["tier"] == "balanced"
        assert result_dict["probes"] == 2
        assert result_dict["bare_success"] is True
        assert result_dict["guarded_success"] is True

    def test_scenario_result_to_dict_no_results(self):
        """Test scenario result to dict with no run results."""
        config = ScenarioConfig(edit="structured", tier="balanced", probes=2)
        scenario_result = ScenarioResult(config=config)

        result_dict = _scenario_result_to_dict(scenario_result)

        assert result_dict["bare_success"] is False
        assert result_dict["guarded_success"] is False

    def test_config_to_dict_complete(self):
        """Test benchmark config to dict conversion."""
        config = BenchmarkConfig(
            edits=["structured"],
            tiers=["balanced"],
            probes=[0],
            epsilon=0.05,
        )

        config_dict = _config_to_dict(config)

        assert config_dict["edits"] == ["structured"]
        assert config_dict["epsilon"] == pytest.approx(0.05)
        assert "strict" not in config_dict

    def test_summary_to_step14_json_skipped_scenario(self):
        """Test JSON generation with skipped scenario."""
        config = BenchmarkConfig(edits=["structured"], tiers=["balanced"], probes=[0])
        scenario_config = ScenarioConfig(edit="structured", tier="balanced", probes=0)

        skipped_result = ScenarioResult(
            config=scenario_config, skipped=True, skip_reason="missing dependency"
        )

        summary = BenchmarkSummary(
            config=config,
            scenarios=[skipped_result],
            overall_pass=True,
            timestamp="2023-01-01T00:00:00",
            execution_time_seconds=10.0,
        )

        json_data = _summary_to_step14_json(summary)

        scenario = json_data["scenarios"][0]
        assert scenario["skip"] is True
        assert scenario["skip_reason"] == "missing dependency"
        assert scenario["primary_metric_bare"] is None
        assert scenario["pass"]["spike"] is None
