"""
Test coverage for bench.py - targeting missing coverage areas.

This module provides comprehensive tests for the InvarLock benchmark module,
focusing on areas likely to be uncovered to push coverage from 76% to 80%+.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from invarlock.eval.bench import (
    BenchmarkConfig,
    BenchmarkSummary,
    RunResult,
    ScenarioConfig,
    ScenarioResult,
    _generate_step14_markdown,
    execute_scenario,
    run_guard_effect_benchmark,
)
from invarlock.reporting.report_types import create_empty_report


def _report_with_artifacts(report_path: str = "report.json") -> dict[str, object]:
    report = create_empty_report()
    report["artifacts"]["report_path"] = report_path
    return report


class TestExecuteScenario:
    """Test execute_scenario function."""

    @patch("invarlock.eval.bench_runner.DependencyChecker.check_edit_dependencies")
    def test_execute_scenario_dependency_failure(self, mock_check_deps):
        """Test scenario execution with dependency failure."""
        mock_check_deps.return_value = (False, "missing dependency")

        scenario = ScenarioConfig(edit="unknown_edit", tier="balanced", probes=2)
        config = BenchmarkConfig(edits=["unknown_edit"], tiers=["balanced"], probes=[2])

        with tempfile.TemporaryDirectory() as temp_dir:
            result = execute_scenario(scenario, config, Path(temp_dir))

        assert result.skipped is True
        assert result.skip_reason == "missing dependency"

    @patch("invarlock.eval.bench_runner.resolve_epsilon_from_runtime")
    @patch("invarlock.eval.bench_runner.execute_single_run")
    @patch("invarlock.eval.bench_runner.DependencyChecker.check_edit_dependencies")
    def test_execute_scenario_epsilon_fallback_when_guarded_fails(
        self,
        mock_check_deps,
        mock_execute_single_run,
        mock_resolve_epsilon,
    ):
        mock_check_deps.return_value = (True, "ok")
        bare = RunResult(
            run_type="bare",
            report=_report_with_artifacts("bare.report.json"),
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

    @patch("invarlock.eval.bench_runner.ValidationGates.validate_all_gates")
    @patch("invarlock.eval.bench_runner.execute_single_run")
    @patch("invarlock.eval.bench_runner.DependencyChecker.check_edit_dependencies")
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
            report=_report_with_artifacts("bare.report.json"),
            success=True,
            error_message=None,
        )
        guarded = RunResult(
            run_type="guarded",
            report=_report_with_artifacts("guarded.report.json"),
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
            with (
                patch(
                    "invarlock.reporting.report_make.make_report",
                    return_value=create_empty_report(),
                ),
                patch(
                    "invarlock.reporting.telemetry.telemetry_output_enabled",
                    return_value=False,
                ),
            ):
                result = execute_scenario(scenario, config, Path(temp_dir))

        mock_validate_all_gates.assert_called_once()
        assert result.gates["quality"] is True

    @patch("invarlock.eval.bench_runner.execute_single_run")
    @patch("invarlock.eval.bench_runner.DependencyChecker.check_edit_dependencies")
    def test_execute_scenario_success_requires_report_artifacts(
        self,
        mock_check_deps,
        mock_execute_single_run,
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
            report=_report_with_artifacts("guarded.report.json"),
            success=True,
            error_message=None,
        )
        mock_execute_single_run.side_effect = [bare, guarded]

        scenario = ScenarioConfig(edit="quant_rtn", tier="balanced", probes=1)
        config = BenchmarkConfig(edits=["quant_rtn"], tiers=["balanced"], probes=[1])

        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(
                RuntimeError, match="bare run report is missing artifacts.report_path"
            ):
                execute_scenario(scenario, config, Path(temp_dir))

    @patch("invarlock.eval.bench_runner.execute_single_run")
    @patch("invarlock.eval.bench_runner.DependencyChecker.check_edit_dependencies")
    def test_execute_scenario_report_generation_failure_raises(
        self,
        mock_check_deps,
        mock_execute_single_run,
    ):
        mock_check_deps.return_value = (True, "ok")
        bare = RunResult(
            run_type="bare",
            report=_report_with_artifacts("bare.report.json"),
            success=True,
            error_message=None,
        )
        guarded = RunResult(
            run_type="guarded",
            report=_report_with_artifacts("guarded.report.json"),
            success=True,
            error_message=None,
        )
        mock_execute_single_run.side_effect = [bare, guarded]

        scenario = ScenarioConfig(edit="quant_rtn", tier="balanced", probes=1)
        config = BenchmarkConfig(edits=["quant_rtn"], tiers=["balanced"], probes=[1])

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch(
                    "invarlock.reporting.report_make.make_report",
                    side_effect=RuntimeError("report boom"),
                ),
                patch(
                    "invarlock.reporting.telemetry.telemetry_output_enabled",
                    return_value=False,
                ),
                pytest.raises(
                    RuntimeError,
                    match="Evaluation report generation failed for quant_rtn__balanced__p1",
                ),
            ):
                execute_scenario(scenario, config, Path(temp_dir))


class TestRunGuardEffectBenchmark:
    """Test the main benchmark function."""

    def test_run_guard_effect_benchmark_basic(self, monkeypatch):
        """Test basic benchmark execution."""
        monkeypatch.setattr(
            "invarlock.eval.bench_runner.execute_scenario",
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
            "invarlock.eval.bench_runner.execute_scenario",
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
            "invarlock.eval.bench_runner.generate_scenarios", lambda cfg: [scenario_cfg]
        )
        monkeypatch.setattr(
            "invarlock.eval.bench_runner.execute_scenario",
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
