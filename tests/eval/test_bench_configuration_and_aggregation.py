"""
Test coverage for bench.py - targeting missing coverage areas.

This module provides comprehensive tests for the InvarLock benchmark module,
focusing on areas likely to be uncovered to push coverage from 76% to 80%+.
"""

import math
from pathlib import Path

import pytest

from invarlock.eval.bench import (
    BenchmarkConfig,
    ConfigurationManager,
    DependencyChecker,
    MetricsAggregator,
    RunResult,
    ScenarioConfig,
    execute_single_run,
)
from invarlock.reporting.report_types import create_empty_report


def _report_with_artifacts(report_path: str = "report.json") -> dict[str, object]:
    report = create_empty_report()
    report["artifacts"]["report_path"] = report_path
    return report


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

    def test_epsilon_override_preserved(self):
        """Explicit epsilon override is preserved."""
        config = BenchmarkConfig(
            edits=["quant_rtn"], tiers=["balanced"], probes=[0], epsilon=0.0
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

        # Fallback to duration when duration_s is absent
        report = {"metrics": {}, "meta": {"duration": 2.0}}
        metrics = MetricsAggregator.extract_core_metrics(report)
        assert metrics["duration_s"] == 2.0

    def test_extract_core_metrics_tolerates_primary_metric_and_meta_get_errors(self):
        class _ExplodingGetDict(dict):
            def get(self, *_args, **_kwargs):  # type: ignore[override]
                raise TypeError("boom")

        report = {
            "metrics": {
                "primary_metric": _ExplodingGetDict(),
                "latency_ms_per_tok": 1.0,
                "memory_mb_peak": 2.0,
            },
            "meta": _ExplodingGetDict(),
        }

        metrics = MetricsAggregator.extract_core_metrics(report)

        assert math.isnan(metrics["primary_metric_preview"])
        assert math.isnan(metrics["primary_metric_final"])
        assert math.isnan(metrics["duration_s"])

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

    def test_extract_guard_metrics_skips_non_dict_guards_and_handles_bad_fallback_types(
        self,
    ):
        report = create_empty_report()
        report["guards"] = ["not-a-dict"]
        report["metrics"] = {"rmt": "bad", "invariants": "bad"}
        report["meta"] = {"rollback_reason": "boom"}

        metrics = MetricsAggregator.extract_guard_metrics(report)
        assert metrics["rmt_outliers"] == 0
        assert metrics["tying_violations_post"] == 0
        assert metrics["catastrophic_spike"] is True

    def test_extract_guard_metrics_rmt_structured_loop_exhaustion_uses_fallback(self):
        """When structured RMT metrics are non-numeric, loop should exhaust and fallback."""
        report = create_empty_report()
        report["guards"] = [
            {"name": "rmt", "metrics": {"layers_flagged": "bad"}, "violations": []}
        ]
        report["metrics"] = {"rmt": {"outliers": 9}}

        metrics = MetricsAggregator.extract_guard_metrics(report)
        assert metrics["rmt_outliers"] == 9

    def test_extract_guard_metrics_invariants_violations_not_list_skips_len_fallback(
        self,
    ):
        report = create_empty_report()
        report["guards"] = [
            {
                "name": "invariants",
                "metrics": {"violations_found": "bad"},
                "violations": "bad",
            }
        ]
        report["metrics"] = {"invariants": {"violations": 7}}

        metrics = MetricsAggregator.extract_guard_metrics(report)
        assert metrics["tying_violations_post"] == 7

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

    def test_compute_comparison_metrics_time_overhead_falls_back_to_latency(self):
        bare_report = create_empty_report()
        bare_report["metrics"] = {
            "primary_metric": {"kind": "ppl_causal", "final": 10.0},
            "latency_ms_per_tok": 1.0,
            "memory_mb_peak": 100.0,
        }
        bare_report["meta"] = {"duration_s": "bad"}

        guarded_report = create_empty_report()
        guarded_report["metrics"] = {
            "primary_metric": {"kind": "ppl_causal", "final": 12.0},
            "latency_ms_per_tok": 1.5,
            "memory_mb_peak": 110.0,
        }
        guarded_report["meta"] = {"duration_s": None}

        bare_result = RunResult("bare", bare_report, success=True)
        guarded_result = RunResult("guarded", guarded_report, success=True)

        comparison = MetricsAggregator.compute_comparison_metrics(
            bare_result, guarded_result
        )
        assert comparison["guard_overhead_time"] == pytest.approx(0.5)


def test_execute_single_run_skips_tokenizer_hash_and_duration_paths(
    monkeypatch, tmp_path: Path
):
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
            # Match the attribute shape expected by execute_single_run
            return types.SimpleNamespace(
                meta={"duration": "bad"},
                edit={"plan_digest": "pd", "deltas": {}},
                metrics={"rmt": "bad"},
                guards={},
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
    run_config: dict = {"dataset": {"provider": "wikitext2"}, "edit": {"plan": {}}}
    runtime = {
        "adapter": _Adapter(),
        "model": object(),
        "baseline_snapshot": b"blob",
        "pairing_schedule": {},
        "calibration_data": [],
        "rmt_baseline_mp_stats": {},
        "rmt_baseline_sigmas": {},
        "tokenizer_hash": None,
        "dataset_name": "wikitext2",
        "split": "validation",
    }

    result = execute_single_run(run_config, scenario, "bare", tmp_path, runtime=runtime)
    assert result.success is True
    assert "tokenizer_hash" not in result.report.get("meta", {})
    assert "duration_s" not in result.report.get("meta", {})


def test_execute_single_run_invalid_edit_payload_surfaces_failure(
    monkeypatch, tmp_path: Path
):
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
                edit="not-a-dict",
                metrics={"rmt": {}},
                guards={},
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
    run_config: dict = {"dataset": {"provider": "wikitext2"}, "edit": {"plan": {}}}
    runtime = {
        "adapter": _Adapter(),
        "model": object(),
        "baseline_snapshot": b"blob",
        "pairing_schedule": {},
        "calibration_data": [],
        "rmt_baseline_mp_stats": {},
        "rmt_baseline_sigmas": {},
        "tokenizer_hash": None,
        "dataset_name": "wikitext2",
        "split": "validation",
    }

    result = execute_single_run(run_config, scenario, "bare", tmp_path, runtime=runtime)
    assert result.success is False
    assert result.error_message is not None
    assert "invalid edit metadata payload" in result.error_message

    class _CoreRunnerNonStringPlanDigest:
        def execute(self, **_kwargs):  # noqa: ANN001
            return types.SimpleNamespace(
                meta={},
                edit={"plan_digest": 123, "deltas": {}},
                metrics={"rmt": {}},
                guards={},
                status="ok",
            )

    monkeypatch.setattr(core_runner, "CoreRunner", _CoreRunnerNonStringPlanDigest)
    result = execute_single_run(run_config, scenario, "bare", tmp_path, runtime=runtime)
    assert result.success is False
    assert result.error_message is not None
    assert "non-string plan_digest" in result.error_message

    class _CoreRunnerNonDictDeltas:
        def execute(self, **_kwargs):  # noqa: ANN001
            return types.SimpleNamespace(
                meta={},
                edit={"plan_digest": "pd", "deltas": []},
                metrics={"rmt": {}},
                guards={},
                status="ok",
            )

    monkeypatch.setattr(core_runner, "CoreRunner", _CoreRunnerNonDictDeltas)
    result = execute_single_run(run_config, scenario, "bare", tmp_path, runtime=runtime)
    assert result.success is False
    assert result.error_message is not None
    assert "invalid edit delta payload" in result.error_message


def test_bench_runner_helper_validations() -> None:
    import invarlock.eval.bench_runner as bench_runner_mod

    with pytest.raises(RuntimeError, match="dataset section must be a mapping"):
        bench_runner_mod._assign_dataset_provider(
            {"dataset": []}, "wikitext2", run_label="bare"
        )

    ok = RunResult("bare", _report_with_artifacts("report.json"), success=True)
    assert (
        bench_runner_mod._extract_success_report_path(ok, run_label="bare")
        == "report.json"
    )

    with pytest.raises(RuntimeError, match="mapping report payload"):
        bench_runner_mod._extract_success_report_path(
            RunResult("bare", "bad", success=True), run_label="bare"
        )
    with pytest.raises(RuntimeError, match="artifacts metadata"):
        bench_runner_mod._extract_success_report_path(
            RunResult("bare", {"artifacts": []}, success=True),
            run_label="bare",
        )
    with pytest.raises(RuntimeError, match="artifacts.report_path"):
        bench_runner_mod._extract_success_report_path(
            RunResult("bare", {"artifacts": {}}, success=True),
            run_label="bare",
        )


def test_execute_single_run_tolerates_missing_guards_attribute(
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
                edit={"plan_digest": "pd", "deltas": {}},
                metrics={},
                evaluation_windows={},
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
    assert result.report["guards"] == []
