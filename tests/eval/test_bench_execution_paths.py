"""
Test coverage for bench.py - targeting missing coverage areas.

This module provides comprehensive tests for the InvarLock benchmark module,
focusing on areas likely to be uncovered to push coverage from 76% to 80%+.
"""

import math
from pathlib import Path

import pytest

from invarlock.eval.bench_policy import (
    BenchmarkConfig,
    BenchmarkSummary,
    MetricsAggregator,
    RunResult,
    ScenarioConfig,
    ScenarioResult,
    generate_step14_markdown,
)
from invarlock.eval.bench_runner import (
    execute_scenario,
    execute_single_run,
)
from invarlock.reporting.report_types import create_empty_report


def _report_with_artifacts(report_path: str = "report.json") -> dict[str, object]:
    report = create_empty_report()
    report["artifacts"]["report_path"] = report_path
    return report


def test_execute_single_run_plan_digest_none_and_non_dict_guards_are_tolerated(
    monkeypatch, tmp_path: Path
) -> None:
    import types

    import invarlock.core.registry as core_registry
    import invarlock.core.runner as core_runner
    import invarlock.eval.bench_runner as bench_runner_mod

    class _Adapter:
        def restore(self, _model, _blob):  # noqa: ANN001
            return None

    class _Registry:
        def get_edit(self, _name: str):  # noqa: ANN001
            return object()

    class _CoreRunner:
        def execute(self, **_kwargs):  # noqa: ANN001
            return types.SimpleNamespace(
                meta={},
                edit={"plan_digest": None, "deltas": {}},
                metrics={},
                evaluation_windows=[],
                guards={"variance": "skip-me"},
                status="ok",
            )

    monkeypatch.setattr(core_registry, "get_registry", lambda: _Registry())
    monkeypatch.setattr(core_runner, "CoreRunner", _CoreRunner)
    monkeypatch.setattr(
        bench_runner_mod.rmt_detection,
        "rmt_detect",
        lambda **_k: {"n_layers_flagged": 0},
    )

    scenario = ScenarioConfig(
        edit="quant_rtn", tier="balanced", probes=0, profile="ci", device="cpu"
    )
    runtime = {
        "adapter": _Adapter(),
        "model": object(),
        "baseline_snapshot": b"blob",
        "pairing_schedule": {},
        "calibration_data": [],
        "rmt_baseline_mp_stats": {},
        "rmt_baseline_sigmas": {},
        "dataset_name": "wikitext2",
        "split": "validation",
    }

    result = execute_single_run(
        {"dataset": {"provider": "wikitext2"}, "edit": {"plan": {}}},
        scenario,
        "bare",
        tmp_path,
        runtime=runtime,
    )
    assert result.success is True
    assert result.report["edit"]["plan_digest"] == ""
    assert result.report["guards"] == []

    class _CoreRunnerNonDictMetricsAndMixedGuards:
        def execute(self, **_kwargs):  # noqa: ANN001
            return types.SimpleNamespace(
                meta={},
                edit={"plan_digest": None, "deltas": {}},
                metrics=[],
                evaluation_windows=[],
                guards={
                    "skip": "not-a-dict",
                    "spectral": {"passed": True, "decision": "allow"},
                },
                status="ok",
            )

    monkeypatch.setattr(
        core_runner, "CoreRunner", _CoreRunnerNonDictMetricsAndMixedGuards
    )
    result = execute_single_run(
        {"dataset": {"provider": "wikitext2"}, "edit": {"plan": {}}},
        scenario,
        "bare",
        tmp_path,
        runtime=runtime,
    )
    assert result.success is True
    assert isinstance(result.report["metrics"], dict)
    assert "primary_metric" in result.report["metrics"]
    assert len(result.report["guards"]) == 1
    assert result.report["guards"][0]["name"] == "spectral"
    assert result.report["guards"][0]["passed"] is True
    assert result.report["guards"][0]["decision"] == "allow"

    class _CoreRunnerNonDictGuards:
        def execute(self, **_kwargs):  # noqa: ANN001
            return types.SimpleNamespace(
                meta={},
                edit={"plan_digest": None, "deltas": {}},
                metrics={},
                evaluation_windows=[],
                guards=[],
                status="ok",
            )

    monkeypatch.setattr(core_runner, "CoreRunner", _CoreRunnerNonDictGuards)
    result = execute_single_run(
        {"dataset": {"provider": "wikitext2"}, "edit": {"plan": {}}},
        scenario,
        "bare",
        tmp_path,
        runtime=runtime,
    )
    assert result.success is True
    assert result.report["guards"] == []


def test_execute_scenario_writes_pairing_schedule_and_telemetry_summary(
    monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import invarlock.eval.bench_runner as bench_runner_mod
    import invarlock.reporting.report_builder_support as telemetry_mod

    bare = RunResult("bare", _report_with_artifacts("bare.json"), success=True)
    guarded = RunResult("guarded", _report_with_artifacts("guarded.json"), success=True)

    def _fake_execute_single_run(_cfg, _scenario, run_type, _dir, *, runtime):  # noqa: ANN001
        runtime["pairing_schedule"] = {"preview": {}, "final": {}}
        return bare if run_type == "bare" else guarded

    monkeypatch.setattr(
        bench_runner_mod, "execute_single_run", _fake_execute_single_run
    )
    monkeypatch.setattr(
        bench_runner_mod,
        "resolve_epsilon_from_runtime",
        lambda _report: 0.2,
    )
    monkeypatch.setattr(
        "invarlock.reporting.report_make.make_report",
        lambda guarded_report, bare_report: {  # noqa: ARG005
            "summary": {"status": "ok"}
        },
    )
    monkeypatch.setattr(telemetry_mod, "telemetry_output_enabled", lambda: True)
    monkeypatch.setattr(
        telemetry_mod,
        "telemetry_summary_line",
        lambda _report: "telemetry summary",
    )

    caplog.set_level("INFO")
    scenario = ScenarioConfig(edit="quant_rtn", tier="balanced", probes=1)
    config = BenchmarkConfig(
        edits=["quant_rtn"],
        tiers=["balanced"],
        probes=[1],
        dataset="wikitext2",
        output_dir=tmp_path,
    )

    result = execute_scenario(scenario, config, tmp_path)
    scenario_slug = f"{scenario.edit}__{scenario.tier}__p{scenario.probes}"
    scenario_dir = tmp_path / "scenarios" / scenario_slug
    assert (scenario_dir / "pairing_schedule.json").exists()
    assert "evaluation_report" in result.artifacts
    assert "telemetry summary" in caplog.text


def test_execute_scenario_writes_report_without_telemetry_summary_line(
    monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import invarlock.eval.bench_runner as bench_runner_mod
    import invarlock.reporting.report_builder_support as telemetry_mod

    bare = RunResult("bare", _report_with_artifacts("bare.json"), success=True)
    guarded = RunResult("guarded", _report_with_artifacts("guarded.json"), success=True)

    def _fake_execute_single_run(_cfg, _scenario, run_type, _dir, *, runtime):  # noqa: ANN001
        runtime["pairing_schedule"] = {"preview": {}, "final": {}}
        return bare if run_type == "bare" else guarded

    monkeypatch.setattr(
        bench_runner_mod, "execute_single_run", _fake_execute_single_run
    )
    monkeypatch.setattr(
        bench_runner_mod,
        "resolve_epsilon_from_runtime",
        lambda _report: 0.2,
    )
    monkeypatch.setattr(
        "invarlock.reporting.report_make.make_report",
        lambda guarded_report, bare_report: {  # noqa: ARG005
            "summary": {"status": "ok"}
        },
    )
    monkeypatch.setattr(telemetry_mod, "telemetry_output_enabled", lambda: True)
    monkeypatch.setattr(telemetry_mod, "telemetry_summary_line", lambda _report: "")

    caplog.set_level("INFO")
    scenario = ScenarioConfig(edit="quant_rtn", tier="balanced", probes=1)
    config = BenchmarkConfig(
        edits=["quant_rtn"],
        tiers=["balanced"],
        probes=[1],
        dataset="wikitext2",
        output_dir=tmp_path,
    )

    result = execute_scenario(scenario, config, tmp_path)
    assert "evaluation_report" in result.artifacts
    assert "telemetry summary" not in caplog.text


def test_execute_scenario_defaults_epsilon_when_runs_fail_without_reports(
    monkeypatch, tmp_path: Path
) -> None:
    import invarlock.eval.bench_runner as bench_runner_mod

    bare = RunResult("bare", create_empty_report(), success=False, error_message="bare")
    guarded = RunResult(
        "guarded", create_empty_report(), success=False, error_message="guarded"
    )

    monkeypatch.setattr(
        bench_runner_mod,
        "execute_single_run",
        lambda _cfg, _scenario, run_type, _dir, *, runtime: (  # noqa: ARG005
            bare if run_type == "bare" else guarded
        ),
    )

    scenario = ScenarioConfig(edit="quant_rtn", tier="balanced", probes=1)
    config = BenchmarkConfig(
        edits=["quant_rtn"],
        tiers=["balanced"],
        probes=[1],
        dataset="wikitext2",
        output_dir=tmp_path,
    )

    result = execute_scenario(scenario, config, tmp_path)
    assert result.epsilon_used == 0.10
    assert "bare_report" not in result.artifacts
    assert result.metrics == {"error_bare": "bare", "error_guarded": "guarded"}


def test_execute_scenario_uses_explicit_epsilon_without_runtime_resolution(
    monkeypatch, tmp_path: Path
) -> None:
    import invarlock.eval.bench_runner as bench_runner_mod

    bare = RunResult("bare", create_empty_report(), success=False, error_message="bare")
    guarded = RunResult(
        "guarded", create_empty_report(), success=False, error_message="guarded"
    )

    monkeypatch.setattr(
        bench_runner_mod,
        "execute_single_run",
        lambda _cfg, _scenario, run_type, _dir, *, runtime: (  # noqa: ARG005
            bare if run_type == "bare" else guarded
        ),
    )
    monkeypatch.setattr(
        bench_runner_mod,
        "resolve_epsilon_from_runtime",
        lambda _report: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    scenario = ScenarioConfig(edit="quant_rtn", tier="balanced", probes=1)
    config = BenchmarkConfig(
        edits=["quant_rtn"],
        tiers=["balanced"],
        probes=[1],
        dataset="wikitext2",
        output_dir=tmp_path,
        epsilon=0.33,
    )

    result = execute_scenario(scenario, config, tmp_path)
    assert result.epsilon_used == 0.33

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

        assert math.isnan(comparison["guard_primary_metric_impact"])
        assert math.isnan(comparison["guard_runtime_overhead"])
        assert math.isnan(comparison["guard_memory_overhead"])


def test_generate_step14_markdown_uses_dash_for_missing_runtime_overhead() -> None:
    summary = BenchmarkSummary(
        config=BenchmarkConfig(
            edits=["quant_rtn"],
            tiers=["balanced"],
            probes=[0],
            output_dir=Path("bench"),
        ),
        scenarios=[
            ScenarioResult(
                config=ScenarioConfig(
                    edit="quant_rtn",
                    tier="balanced",
                    probes=0,
                ),
                metrics={
                    "guard_primary_metric_impact": 0.0,
                    "guard_runtime_overhead": float("nan"),
                    "guard_memory_overhead": 0.0,
                    "rmt_outliers_bare": 0,
                    "rmt_outliers_guarded": 0,
                },
                gates={"spike": True, "rmt": True, "quality": True},
            )
        ],
        overall_pass=True,
        timestamp="2026-04-08T00:00:00",
        execution_time_seconds=0.1,
    )

    markdown = generate_step14_markdown(summary)

    assert (
        "| quant_rtn | balanced | 0 | ✅ PASS | 🟢 +0.0% | - | 🟢 +0.0% |" in markdown
    )

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

        assert math.isnan(comparison["guard_primary_metric_impact"])
        assert math.isnan(comparison["guard_runtime_overhead"])
        assert math.isnan(comparison["guard_memory_overhead"])
