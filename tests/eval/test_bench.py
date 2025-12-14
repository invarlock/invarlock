"""
Test coverage for bench.py - targeting missing coverage areas.

This module provides comprehensive tests for the InvarLock benchmark module,
focusing on areas likely to be uncovered to push coverage from 76% to 80%+.
"""

import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from invarlock.eval.bench import (
    BenchmarkConfig,
    BenchmarkSummary,
    ConfigurationManager,
    DependencyChecker,
    MetricsAggregator,
    RunResult,
    ScenarioConfig,
    ScenarioResult,
    ValidationGates,
    _config_to_dict,
    _generate_step14_markdown,
    _scenario_result_to_dict,
    _summary_to_step14_json,
    execute_scenario,
    execute_single_run,
    generate_scenarios,
    main,
    resolve_epsilon_from_runtime,
    run_guard_effect_benchmark,
)
from invarlock.reporting.report_types import create_empty_report


class TestScenarioConfig:
    """Test ScenarioConfig class and its post-init logic."""

    def test_ci_profile_defaults(self):
        """Test CI profile sets correct defaults."""
        config = ScenarioConfig(
            edit="quant_rtn", tier="balanced", probes=2, profile="ci"
        )
        assert config.preview_n == 50
        assert config.final_n == 50

    def test_release_profile_defaults(self):
        """Test release profile sets correct defaults."""
        config = ScenarioConfig(
            edit="quant_rtn", tier="balanced", probes=2, profile="release"
        )
        assert config.preview_n == 100
        assert config.final_n == 100

    def test_release_profile_preserves_custom_preview_and_final(self):
        """Custom preview_n/final_n should not be overridden in release profile."""
        config = ScenarioConfig(
            edit="quant_rtn",
            tier="balanced",
            probes=2,
            profile="release",
            preview_n=10,
            final_n=20,
        )
        assert config.preview_n == 10
        assert config.final_n == 20

    def test_invalid_profile_raises_error(self):
        """Test invalid profile raises ValueError."""
        with pytest.raises(ValueError, match="Unknown profile: invalid"):
            ScenarioConfig(
                edit="quant_rtn", tier="balanced", probes=2, profile="invalid"
            )

    def test_custom_preview_final_n_preserved(self):
        """Test custom preview_n and final_n are preserved."""
        config = ScenarioConfig(
            edit="quant_rtn",
            tier="balanced",
            probes=2,
            profile="ci",
            preview_n=25,
            final_n=75,
        )
        assert config.preview_n == 25
        assert config.final_n == 75


class TestBenchmarkConfig:
    """Test BenchmarkConfig class and its post-init logic."""

    def test_strict_mode_sets_epsilon_zero(self):
        """Test strict mode sets epsilon to 0."""
        config = BenchmarkConfig(
            edits=["quant_rtn"], tiers=["balanced"], probes=[0], strict=True
        )
        assert config.epsilon == 0.0

    def test_output_dir_path_conversion(self):
        """Test output_dir is converted to Path object."""
        config = BenchmarkConfig(
            edits=["quant_rtn"], tiers=["balanced"], probes=[0], output_dir="test_dir"
        )
        assert isinstance(config.output_dir, Path)
        assert config.output_dir.name == "test_dir"


class TestDependencyChecker:
    """Test dependency checking functionality."""

    def test_check_external_deps_always_available(self):
        """Test external deps check returns available (placeholder)."""
        available, message = DependencyChecker.check_external_deps()
        assert available is True
        assert message == "Available"

    def test_check_peft_deprecated(self):
        """External fine-tuning adapters are not supported."""
        available, message = DependencyChecker.check_peft()
        assert available is False
        assert message == "unsupported edit"

    def test_check_edit_dependencies_builtin_edits(self):
        """Test built-in edits are always available."""
        for edit in ["quant_rtn"]:
            available, message = DependencyChecker.check_edit_dependencies(edit)
            assert available is True
            assert message == "Available"

    def test_check_edit_dependencies_unknown_edit(self):
        """Test unknown edit types return available (fallback)."""
        available, message = DependencyChecker.check_edit_dependencies("unknown_edit")
        assert available is False
        assert message == "unsupported edit"


class TestConfigurationManager:
    """Test configuration generation logic."""

    def test_create_base_config_structure(self):
        """Test base configuration structure."""
        scenario = ScenarioConfig(edit="quant_rtn", tier="balanced", probes=2)
        config = ConfigurationManager.create_base_config(scenario)

        assert "model" in config
        assert "dataset" in config
        assert "edit" in config
        assert "eval" in config
        assert "output" in config

        assert config["model"]["id"] == "gpt2"
        assert config["dataset"]["provider"] == "wikitext2"
        assert config["edit"]["name"] == "quant_rtn"

    def test_get_edit_plan_quant_rtn_ci(self):
        """Test quant_rtn edit plan for CI profile."""
        plan = ConfigurationManager._get_edit_plan("quant_rtn", "ci")
        assert plan["bitwidth"] == 8
        assert plan["per_channel"] is True
        assert plan["group_size"] == 128
        assert plan["scope"] == "ffn"

    # lowrank_svd and structured plans are purged

    def test_get_edit_plan_unknown_edit(self):
        """Test unknown edit returns empty plan."""
        plan = ConfigurationManager._get_edit_plan("unknown_edit", "ci")
        assert plan == {}

    def test_create_bare_config(self):
        """Test bare configuration creation."""
        scenario = ScenarioConfig(edit="quant_rtn", tier="balanced", probes=2)
        config = ConfigurationManager.create_bare_config(scenario)

        assert config["auto"]["enabled"] is False
        assert config["guards"]["order"] == ["invariants"]
        assert config["guards"]["invariants"]["mode"] == "warn"

    def test_create_guarded_config(self):
        """Test guarded configuration creation."""
        scenario = ScenarioConfig(edit="quant_rtn", tier="balanced", probes=2)
        config = ConfigurationManager.create_guarded_config(scenario)

        assert config["auto"]["enabled"] is True
        assert config["auto"]["tier"] == "balanced"
        assert config["auto"]["probes"] == 2
        assert "invariants" in config["guards"]["order"]
        assert "spectral" in config["guards"]["order"]
        assert "rmt" in config["guards"]["order"]


class TestMetricsAggregator:
    """Test metrics aggregation and computation."""

    def test_extract_core_metrics_empty_report(self):
        """Test extracting metrics from empty report."""
        report = create_empty_report()
        # Remove any existing metrics to test empty case
        if "metrics" in report:
            del report["metrics"]
        metrics = MetricsAggregator.extract_core_metrics(report)

        for key in [
            "primary_metric_preview",
            "primary_metric_final",
            "latency_ms_per_tok",
            "memory_mb_peak",
        ]:
            assert math.isnan(metrics[key])

    def test_extract_core_metrics_handles_bad_values(self):
        report = {
            "metrics": {
                "primary_metric": {"preview": "bad", "final": object()},
                "latency_ms_per_tok": 1.0,
                "memory_mb_peak": 2.0,
            }
        }
        metrics = MetricsAggregator.extract_core_metrics(report)
        assert math.isnan(metrics["primary_metric_preview"])
        assert math.isnan(metrics["primary_metric_final"])

    def test_extract_core_metrics_non_dict_primary_metric_and_meta(self):
        """When primary_metric/meta are non-dicts, fallbacks should still behave."""
        report = {
            "metrics": {"primary_metric": 123},  # not a dict
            "meta": "not-a-dict",
        }
        metrics = MetricsAggregator.extract_core_metrics(report)
        # No crash and all derived metrics remain NaN
        assert math.isnan(metrics["primary_metric_preview"])
        assert math.isnan(metrics["primary_metric_final"])
        assert math.isnan(metrics["duration_s"])

    def test_extract_core_metrics_populated_report(self):
        """Test extracting metrics from populated report."""
        report = create_empty_report()
        report["metrics"] = {
            "primary_metric": {"kind": "perplexity", "preview": 45.0, "final": 46.0},
            "latency_ms_per_tok": 12.5,
            "memory_mb_peak": 2048.0,
        }

        metrics = MetricsAggregator.extract_core_metrics(report)
        assert metrics["primary_metric_preview"] == 45.0
        assert metrics["primary_metric_final"] == 46.0
        assert metrics["latency_ms_per_tok"] == 12.5
        assert metrics["memory_mb_peak"] == 2048.0

    def test_extract_core_metrics_duration_from_meta_fields(self):
        """Duration should be taken from duration_s then duration meta fields."""
        # duration_s takes precedence
        report = {"metrics": {}, "meta": {"duration_s": 1.5}}
        metrics = MetricsAggregator.extract_core_metrics(report)
        assert metrics["duration_s"] == 1.5

        # Fallback to legacy duration when duration_s is absent
        report = {"metrics": {}, "meta": {"duration": 2.0}}
        metrics = MetricsAggregator.extract_core_metrics(report)
        assert metrics["duration_s"] == 2.0

    def test_extract_guard_metrics_empty_report(self):
        """Test extracting guard metrics from empty report."""
        report = create_empty_report()
        metrics = MetricsAggregator.extract_guard_metrics(report)

        assert metrics["rmt_outliers"] == 0
        assert metrics["tying_violations_post"] == 0
        assert metrics["catastrophic_spike"] is False

    def test_extract_guard_metrics_populated_report(self):
        """Test extracting guard metrics from populated report."""
        report = create_empty_report()
        report["metrics"] = {"rmt": {"outliers": 3}, "invariants": {"violations": 2}}
        report["flags"] = {"guard_recovered": True}

        metrics = MetricsAggregator.extract_guard_metrics(report)
        assert metrics["rmt_outliers"] == 3
        assert metrics["tying_violations_post"] == 2
        assert metrics["catastrophic_spike"] is True

    def test_extract_guard_metrics_prefers_structured_guard_reports(self):
        """Structured guard reports should override metrics fallbacks when present."""
        report = create_empty_report()
        # Metric fallbacks that would otherwise be used
        report["metrics"] = {
            "rmt": {"outliers": 7},
            "invariants": {"violations": 5},
        }
        # Structured guard entries should take precedence
        report["guards"] = [
            {
                "name": "rmt",
                "metrics": {"layers_flagged": 4},
                "violations": [],
            },
            {
                "name": "invariants",
                "metrics": {"violations_found": 1},
                "violations": [{"id": 1}, {"id": 2}],
            },
        ]

        metrics = MetricsAggregator.extract_guard_metrics(report)
        # rmt_outliers sourced from structured guard metrics
        assert metrics["rmt_outliers"] == 4
        # tying_violations_post sourced from structured guard metrics
        assert metrics["tying_violations_post"] == 1

    def test_extract_guard_metrics_non_list_guards_falls_back_to_metrics(self):
        """When guards is not a list, fall back to metrics-based paths."""
        report = create_empty_report()
        report["guards"] = "not-a-list"
        report["metrics"] = {
            "rmt": {"outliers": 1},
            "invariants": {"violations": 2},
        }
        report["meta"] = {"rollback_reason": None}
        metrics = MetricsAggregator.extract_guard_metrics(report)
        assert metrics["rmt_outliers"] == 1
        assert metrics["tying_violations_post"] == 2
        assert metrics["catastrophic_spike"] is False

    def test_extract_guard_metrics_len_violations_used_when_metrics_missing(self):
        """Structured invariants guard falls back to len(violations) when needed."""
        report = create_empty_report()
        report["guards"] = [
            {
                "name": "invariants",
                "metrics": {"violations_found": "not-a-number"},
                "violations": [{"id": 1}, {"id": 2}],
            }
        ]
        report["metrics"] = {}
        metrics = MetricsAggregator.extract_guard_metrics(report)
        assert metrics["tying_violations_post"] == 2

    def test_compute_comparison_metrics_full_overhead_paths(self):
        """Exercise primary/time/mem overhead branches with finite baselines."""
        bare_report = create_empty_report()
        bare_report["metrics"] = {
            "primary_metric": {"kind": "ppl_causal", "final": 10.0},
            "latency_ms_per_tok": 1.0,
            "memory_mb_peak": 100.0,
        }
        bare_report["meta"] = {"duration_s": 1.0}

        guarded_report = create_empty_report()
        guarded_report["metrics"] = {
            "primary_metric": {"kind": "ppl_causal", "final": 11.0},
            "latency_ms_per_tok": 1.2,
            "memory_mb_peak": 120.0,
        }
        guarded_report["meta"] = {"duration_s": 1.5}

        bare_result = RunResult("bare", bare_report, success=True)
        guarded_result = RunResult("guarded", guarded_report, success=True)

        comparison = MetricsAggregator.compute_comparison_metrics(
            bare_result, guarded_result
        )

        assert comparison["primary_metric_overhead"] == pytest.approx(0.1)
        assert comparison["guard_overhead_time"] == pytest.approx(0.5)
        assert comparison["guard_overhead_mem"] == pytest.approx(0.2)

    def test_compute_comparison_metrics_invalid_inputs(self):
        """Test comparison with invalid inputs."""
        bare_result = None
        guarded_result = None

        comparison = MetricsAggregator.compute_comparison_metrics(
            bare_result, guarded_result
        )
        assert comparison == {}

        # Test with unsuccessful results
        bare_result = RunResult("bare", create_empty_report(), success=False)
        guarded_result = RunResult("guarded", create_empty_report(), success=True)

        comparison = MetricsAggregator.compute_comparison_metrics(
            bare_result, guarded_result
        )
        assert comparison == {}

    def test_compute_comparison_metrics_nan_handling(self):
        """Test comparison metrics with NaN values."""
        bare_report = create_empty_report()
        bare_report["metrics"] = {
            "primary_metric": {"kind": "perplexity", "final": float("nan")},
            "latency_ms_per_tok": 0.0,
            "memory_mb_peak": 0.0,
        }

        guarded_report = create_empty_report()
        guarded_report["metrics"] = {
            "primary_metric": {"kind": "perplexity", "final": 46.0},
            "latency_ms_per_tok": float("nan"),
            "memory_mb_peak": float("nan"),
        }

        bare_result = RunResult("bare", bare_report, success=True)
        guarded_result = RunResult("guarded", guarded_report, success=True)

        comparison = MetricsAggregator.compute_comparison_metrics(
            bare_result, guarded_result
        )

        assert math.isnan(comparison["primary_metric_overhead"])
        assert math.isnan(comparison["guard_overhead_time"])
        assert math.isnan(comparison["guard_overhead_mem"])

    def test_compute_comparison_metrics_zero_division_handling(self):
        """Test comparison metrics with zero base values."""
        bare_report = create_empty_report()
        bare_report["metrics"] = {
            "primary_metric": {"kind": "perplexity", "final": 0.0},
            "latency_ms_per_tok": 0.0,
            "memory_mb_peak": 0.0,
        }

        guarded_report = create_empty_report()
        guarded_report["metrics"] = {
            "primary_metric": {"kind": "perplexity", "final": 46.0},
            "latency_ms_per_tok": 12.0,
            "memory_mb_peak": 2000.0,
        }

        bare_result = RunResult("bare", bare_report, success=True)
        guarded_result = RunResult("guarded", guarded_report, success=True)

        comparison = MetricsAggregator.compute_comparison_metrics(
            bare_result, guarded_result
        )

        assert math.isnan(comparison["primary_metric_overhead"])
        assert math.isnan(comparison["guard_overhead_time"])
        assert math.isnan(comparison["guard_overhead_mem"])


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
            strict=True,
        )

        config_dict = _config_to_dict(config)

        assert config_dict["edits"] == ["structured"]
        # strict=True overrides epsilon to 0.0 in post_init
        assert config_dict["epsilon"] == 0.0
        assert config_dict["strict"] is True

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


class TestCLIAndMain:
    """Test CLI argument parsing and main function."""

    @patch("sys.argv", ["bench.py", "--edits", "quant_rtn", "--profile", "ci"])
    @patch("invarlock.eval.bench.run_guard_effect_benchmark")
    def test_main_basic_invocation(self, mock_benchmark):
        """Test basic main function invocation."""
        mock_benchmark.return_value = {"overall_pass": True}

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        mock_benchmark.assert_called_once()

    @patch("sys.argv", ["bench.py", "--edits", "invalid_edit"])
    def test_main_invalid_edit_type(self):
        """Test main function with invalid edit type."""
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    @patch("sys.argv", ["bench.py", "--edits", "quant_rtn", "--tiers", "invalid_tier"])
    def test_main_invalid_tier(self):
        """Test main function with invalid tier."""
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    @patch("sys.argv", ["bench.py", "--edits", "quant_rtn", "--probes", "-1"])
    def test_main_invalid_probe_count(self):
        """Test main function with invalid probe count."""
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    @patch("sys.argv", ["bench.py", "--edits", "quant_rtn", "--profile", "ci"])
    @patch("invarlock.eval.bench.run_guard_effect_benchmark")
    def test_main_benchmark_failure(self, mock_benchmark):
        """Test main function when benchmark fails gates."""
        mock_benchmark.return_value = {"overall_pass": False}

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    @patch("sys.argv", ["bench.py", "--edits", "quant_rtn", "--profile", "ci"])
    @patch("invarlock.eval.bench.run_guard_effect_benchmark")
    def test_main_keyboard_interrupt(self, mock_benchmark):
        """Test main function with keyboard interrupt."""
        mock_benchmark.side_effect = KeyboardInterrupt()

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    @patch("sys.argv", ["bench.py", "--edits", "quant_rtn", "--profile", "ci"])
    @patch("invarlock.eval.bench.run_guard_effect_benchmark")
    def test_main_exception_handling(self, mock_benchmark):
        """Test main function exception handling."""
        mock_benchmark.side_effect = RuntimeError("Test error")

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    @patch(
        "sys.argv",
        ["bench.py", "--edits", "quant_rtn", "--profile", "ci", "--verbose"],
    )
    @patch("invarlock.eval.bench.run_guard_effect_benchmark")
    def test_main_exception_verbose_traces(self, mock_benchmark):
        mock_benchmark.side_effect = RuntimeError("boom")
        with pytest.raises(SystemExit):
            main()


class TestExecuteSingleRun:
    """Test execute_single_run function."""

    def test_execute_single_run_success(self, monkeypatch):
        """Test successful single run execution."""
        from types import SimpleNamespace

        from invarlock.eval.data import EvaluationWindow

        class DummyProfile:
            def make_tokenizer(self):
                return object(), "tokhash"

        class DummyProvider:
            def windows(  # noqa: PLR0913
                self,
                tokenizer,  # noqa: ARG002
                *,
                seq_len: int,
                stride: int,  # noqa: ARG002
                preview_n: int,
                final_n: int,
                seed: int,  # noqa: ARG002
                split: str,  # noqa: ARG002
            ):
                preview = EvaluationWindow(
                    input_ids=[[1] * seq_len for _ in range(preview_n)],
                    attention_masks=[[1] * seq_len for _ in range(preview_n)],
                    indices=list(range(preview_n)),
                )
                final = EvaluationWindow(
                    input_ids=[[2] * seq_len for _ in range(final_n)],
                    attention_masks=[[1] * seq_len for _ in range(final_n)],
                    indices=list(range(final_n)),
                )
                return preview, final

        class DummyAdapter:
            def load_model(self, model_id: str, device: str = "auto", **_kwargs):
                return SimpleNamespace(name=f"{model_id}:{device}")

            def snapshot(self, _model):
                return b"snapshot"

            def restore(self, _model, _blob):
                return None

        class DummyEdit:
            name = "quant_rtn"

        class DummyRegistry:
            def get_adapter(self, _name: str):
                return DummyAdapter()

            def get_edit(self, _name: str):
                return DummyEdit()

            def get_guard(self, _name: str):
                raise KeyError("no guards in stub")

        class DummyCoreReport:
            def __init__(self):
                self.meta = {"duration": 0.01, "guard_recovered": False}
                self.edit = {
                    "plan_digest": "pd",
                    "deltas": {"params_changed": 0, "layers_modified": 0},
                }
                self.metrics = {
                    "primary_metric": {
                        "kind": "ppl_causal",
                        "preview": 1.0,
                        "final": 1.0,
                    },
                    "latency_ms_per_tok": 1.0,
                    "memory_mb_peak": 1.0,
                }
                self.guards = {}
                self.evaluation_windows = {"preview": {}, "final": {}}
                self.status = "success"

        def _fake_execute(*_a, **_k):
            return DummyCoreReport()

        monkeypatch.setattr(
            "invarlock.model_profile.detect_model_profile",
            lambda *_a, **_k: DummyProfile(),
        )
        monkeypatch.setattr(
            "invarlock.eval.data.get_provider", lambda *_a, **_k: DummyProvider()
        )
        monkeypatch.setattr(
            "invarlock.core.registry.get_registry", lambda: DummyRegistry()
        )
        monkeypatch.setattr("invarlock.core.runner.CoreRunner.execute", _fake_execute)
        monkeypatch.setattr(
            "invarlock.guards.rmt.capture_baseline_mp_stats", lambda *_a, **_k: {}
        )
        monkeypatch.setattr(
            "invarlock.guards.rmt.rmt_detect",
            lambda *_a, **_k: {"n_layers_flagged": 0},
        )

        scenario = ScenarioConfig(edit="quant_rtn", tier="balanced", probes=2)
        run_config = ConfigurationManager.create_bare_config(scenario)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = execute_single_run(run_config, scenario, "bare", Path(temp_dir))

        assert result.success is True
        assert result.run_type == "bare"
        assert result.report["meta"]["model_id"] == "gpt2"
        assert result.report["edit"]["name"] == "quant_rtn"

    def test_execute_single_run_exception_handling(self, monkeypatch):
        """Test exception handling in single run execution."""
        scenario = ScenarioConfig(edit="quant_rtn", tier="balanced", probes=2)
        run_config = ConfigurationManager.create_bare_config(scenario)

        monkeypatch.setattr(
            "invarlock.model_profile.detect_model_profile",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = execute_single_run(run_config, scenario, "bare", Path(temp_dir))

        assert result.success is False
        assert result.error_message is not None
        assert "boom" in result.error_message

    def test_execute_single_run_reuses_runtime_without_recomputing_baselines(
        self, monkeypatch
    ):
        """When runtime is pre-populated, heavy setup branches are skipped."""
        from types import SimpleNamespace

        from invarlock.eval.data import EvaluationWindow

        class DummyProfile:
            def make_tokenizer(self):
                return object(), "tokhash"

        class DummyProvider:
            def windows(  # noqa: PLR0913
                self,
                tokenizer,  # noqa: ARG002
                *,
                seq_len: int,
                stride: int,  # noqa: ARG002
                preview_n: int,
                final_n: int,
                seed: int,  # noqa: ARG002
                split: str,  # noqa: ARG002
            ):
                preview = EvaluationWindow(
                    input_ids=[[1] * seq_len for _ in range(preview_n)],
                    attention_masks=[[1] * seq_len for _ in range(preview_n)],
                    indices=list(range(preview_n)),
                )
                final = EvaluationWindow(
                    input_ids=[[2] * seq_len for _ in range(final_n)],
                    attention_masks=[[1] * seq_len for _ in range(final_n)],
                    indices=list(range(final_n)),
                )
                return preview, final

        class DummyAdapter:
            def load_model(self, model_id: str, device: str = "auto", **_kwargs):
                return SimpleNamespace(name=f"{model_id}:{device}")

            def snapshot(self, _model):
                return b"snapshot"

            def restore(self, _model, _blob):
                return None

        class DummyEdit:
            name = "quant_rtn"

        class DummyRegistry:
            def get_adapter(self, _name: str):
                return DummyAdapter()

            def get_edit(self, _name: str):
                return DummyEdit()

            def get_guard(self, _name: str):
                raise KeyError("no guards in stub")

        class DummyCoreReport:
            def __init__(self):
                self.meta = {"duration": 0.01, "guard_recovered": False}
                self.edit = {
                    "plan_digest": "pd",
                    "deltas": {"params_changed": 0, "layers_modified": 0},
                }
                self.metrics = {
                    "primary_metric": {
                        "kind": "ppl_causal",
                        "preview": 1.0,
                        "final": 1.0,
                    },
                    "latency_ms_per_tok": 1.0,
                    "memory_mb_peak": 1.0,
                }
                self.guards = {}
                self.evaluation_windows = {"preview": {}, "final": {}}
                self.status = "success"

        def _fake_execute(*_a, **_k):
            return DummyCoreReport()

        calls = {"capture": 0, "provider": 0}

        def _fake_capture(*_a, **_k):
            calls["capture"] += 1
            return {}

        def _fake_provider(*_a, **_k):
            calls["provider"] += 1
            return DummyProvider()

        monkeypatch.setattr(
            "invarlock.model_profile.detect_model_profile",
            lambda *_a, **_k: DummyProfile(),
        )
        monkeypatch.setattr("invarlock.eval.data.get_provider", _fake_provider)
        monkeypatch.setattr(
            "invarlock.core.registry.get_registry", lambda: DummyRegistry()
        )
        monkeypatch.setattr("invarlock.core.runner.CoreRunner.execute", _fake_execute)
        monkeypatch.setattr(
            "invarlock.guards.rmt.capture_baseline_mp_stats", _fake_capture
        )
        monkeypatch.setattr(
            "invarlock.guards.rmt.rmt_detect",
            lambda *_a, **_k: {"n_layers_flagged": 0},
        )

        scenario = ScenarioConfig(edit="quant_rtn", tier="balanced", probes=2)
        run_config = ConfigurationManager.create_guarded_config(scenario)

        runtime = {
            "adapter": DummyAdapter(),
            "model": SimpleNamespace(name="m"),
            "baseline_snapshot": b"snapshot",
            "pairing_schedule": {"preview": {}, "final": {}},
            "calibration_data": [],
            "tokenizer_hash": "tokhash",
            "split": "validation",
            "dataset_name": "wikitext2",
            "rmt_baseline_mp_stats": {"layer": {}},
            "rmt_baseline_sigmas": {"layer": 0.1},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = execute_single_run(
                run_config, scenario, "guarded", Path(temp_dir), runtime=runtime
            )

        assert result.success is True
        # When runtime is pre-populated, provider and capture helpers are never called.
        assert calls["provider"] == 0
        assert calls["capture"] == 0


class TestExecuteScenario:
    """Test execute_scenario function."""

    @patch("invarlock.eval.bench.DependencyChecker.check_edit_dependencies")
    def test_execute_scenario_dependency_failure(self, mock_check_deps):
        """Test scenario execution with dependency failure."""
        mock_check_deps.return_value = (False, "missing dependency")

        scenario = ScenarioConfig(edit="unknown_edit", tier="balanced", probes=2)
        config = BenchmarkConfig(edits=["unknown_edit"], tiers=["balanced"], probes=[2])

        with tempfile.TemporaryDirectory() as temp_dir:
            result = execute_scenario(scenario, config, Path(temp_dir))

        assert result.skipped is True
        assert result.skip_reason == "missing dependency"

    @patch("invarlock.eval.bench.resolve_epsilon_from_runtime")
    @patch("invarlock.eval.bench.execute_single_run")
    @patch("invarlock.eval.bench.DependencyChecker.check_edit_dependencies")
    def test_execute_scenario_epsilon_fallback_when_guarded_fails(
        self,
        mock_check_deps,
        mock_execute_single_run,
        mock_resolve_epsilon,
    ):
        mock_check_deps.return_value = (True, "ok")
        bare = RunResult(
            run_type="bare",
            report=create_empty_report(),
            success=True,
            error_message=None,
        )
        guarded = RunResult(
            run_type="guarded",
            report=create_empty_report(),
            success=False,
            error_message="boom",
        )
        mock_execute_single_run.side_effect = [bare, guarded]

        scenario = ScenarioConfig(edit="quant_rtn", tier="balanced", probes=1)
        config = BenchmarkConfig(
            edits=["quant_rtn"], tiers=["balanced"], probes=[1], epsilon=None
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = execute_scenario(scenario, config, Path(temp_dir))

        assert pytest.approx(result.epsilon_used, rel=1e-9) == 0.10
        mock_resolve_epsilon.assert_not_called()

    @patch("invarlock.eval.bench.ValidationGates.validate_all_gates")
    @patch("invarlock.eval.bench.execute_single_run")
    @patch("invarlock.eval.bench.DependencyChecker.check_edit_dependencies")
    def test_execute_scenario_success_uses_validation_gates(
        self,
        mock_check_deps,
        mock_execute_single_run,
        mock_validate_all_gates,
    ):
        """Successful bare/guarded runs should flow through ValidationGates."""
        mock_check_deps.return_value = (True, "ok")
        bare = RunResult(
            run_type="bare",
            report=create_empty_report(),
            success=True,
            error_message=None,
        )
        guarded = RunResult(
            run_type="guarded",
            report=create_empty_report(),
            success=True,
            error_message=None,
        )
        mock_execute_single_run.side_effect = [bare, guarded]
        mock_validate_all_gates.return_value = {
            "spike": True,
            "tying": True,
            "rmt": True,
            "quality": True,
            "time": True,
            "mem": True,
        }

        scenario = ScenarioConfig(edit="quant_rtn", tier="balanced", probes=1)
        config = BenchmarkConfig(
            edits=["quant_rtn"], tiers=["balanced"], probes=[1], epsilon=None
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = execute_scenario(scenario, config, Path(temp_dir))

        mock_validate_all_gates.assert_called_once()
        assert result.gates["quality"] is True


class TestRunGuardEffectBenchmark:
    """Test the main benchmark function."""

    def test_run_guard_effect_benchmark_basic(self, monkeypatch):
        """Test basic benchmark execution."""
        monkeypatch.setattr(
            "invarlock.eval.bench.execute_scenario",
            lambda scenario, cfg, output_dir: ScenarioResult(
                config=scenario,
                metrics={"primary_metric_overhead": 0.0, "guard_overhead_time": 0.0},
                gates={
                    "spike": True,
                    "tying": True,
                    "rmt": True,
                    "quality": True,
                    "time": True,
                    "mem": True,
                },
                probes_used=scenario.probes,
                epsilon_used=0.1,
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_guard_effect_benchmark(
                edits=["quant_rtn"],
                tiers=["balanced"],
                probes=[0],
                profile="ci",
                output_dir=temp_dir,
            )

        assert "overall_pass" in result
        assert "execution_time_seconds" in result
        assert "scenarios" in result
        assert len(result["scenarios"]) == 1

    def test_run_guard_effect_benchmark_multiple_scenarios(self, monkeypatch):
        """Test benchmark with multiple scenarios."""
        monkeypatch.setattr(
            "invarlock.eval.bench.execute_scenario",
            lambda scenario, cfg, output_dir: ScenarioResult(
                config=scenario,
                metrics={"primary_metric_overhead": 0.0, "guard_overhead_time": 0.0},
                gates={
                    "spike": True,
                    "tying": True,
                    "rmt": True,
                    "quality": True,
                    "time": True,
                    "mem": True,
                },
                probes_used=scenario.probes,
                epsilon_used=0.1,
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_guard_effect_benchmark(
                edits=["quant_rtn"],
                tiers=["balanced", "aggressive"],
                probes=[0, 2],
                profile="ci",
                output_dir=temp_dir,
            )

        assert len(result["scenarios"]) == 4  # 1 edit × 2 tiers × 2 probes

    def test_run_guard_effect_benchmark_sets_overall_fail(self, monkeypatch, tmp_path):
        scenario_cfg = ScenarioConfig(edit="quant_rtn", tier="balanced", probes=1)
        fail_result = ScenarioResult(
            config=scenario_cfg,
            metrics={"primary_metric_overhead": 0.0, "guard_overhead_time": 0.0},
            gates={"quality": False, "spike": True, "rmt": True},
            skipped=False,
            probes_used=1,
            epsilon_used=0.05,
        )
        monkeypatch.setattr(
            "invarlock.eval.bench.generate_scenarios", lambda cfg: [scenario_cfg]
        )
        monkeypatch.setattr(
            "invarlock.eval.bench.execute_scenario",
            lambda scenario, cfg, output_dir: fail_result,
        )

        result = run_guard_effect_benchmark(
            edits=["quant_rtn"],
            tiers=["balanced"],
            probes=[1],
            profile="ci",
            output_dir=tmp_path,
        )

        assert result["overall_pass"] is False


class TestMarkdownGeneration:
    """Test Markdown generation edge cases."""

    def test_generate_step14_markdown_with_edge_cases(self):
        """Test Markdown generation with various edge cases."""
        config = BenchmarkConfig(edits=["quant_rtn"], tiers=["balanced"], probes=[0])
        scenario_config = ScenarioConfig(edit="quant_rtn", tier="balanced", probes=0)

        failing_result = ScenarioResult(
            config=scenario_config,
            metrics={
                "primary_metric_overhead": float("nan"),
                "guard_overhead_time": 0.20,
                "guard_overhead_mem": 0.20,
                "rmt_outliers_bare": 3,
                "rmt_outliers_guarded": 1,
            },
            gates={"spike": False, "rmt": False, "quality": False},
        )
        skipped_result = ScenarioResult(
            config=scenario_config,
            skipped=True,
            skip_reason="not available",
        )

        summary = BenchmarkSummary(
            config=config,
            scenarios=[failing_result, skipped_result],
            overall_pass=False,
            timestamp="2023-01-01T00:00:00",
            execution_time_seconds=10.0,
        )

        markdown = _generate_step14_markdown(summary)

        assert "❌ FAIL" in markdown
        assert "🔴 +20.0%" in markdown  # High time overhead
        assert "SKIP: not available" in markdown
        assert "❌📈" in markdown  # Failed spike gate indicator
        assert "❌🔬" in markdown  # Failed RMT gate
        assert "❌📊" in markdown  # Failed quality gate
