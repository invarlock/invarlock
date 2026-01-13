from __future__ import annotations

import pytest

from invarlock.eval import bench as bench_mod


def test_scenario_config_profiles_and_validation():
    cfg_ci = bench_mod.ScenarioConfig(edit="quant_rtn", tier="balanced", probes=0)
    assert cfg_ci.preview_n == 50 and cfg_ci.final_n == 50

    cfg_release = bench_mod.ScenarioConfig(
        edit="quant_rtn", tier="balanced", probes=0, profile="release"
    )
    assert cfg_release.preview_n == 100 and cfg_release.final_n == 100

    with pytest.raises(ValueError):
        bench_mod.ScenarioConfig(
            edit="quant_rtn", tier="balanced", probes=0, profile="unknown"
        )


def test_benchmark_config_epsilon_override_sets_epsilon():
    cfg = bench_mod.BenchmarkConfig(
        edits=["quant_rtn"],
        tiers=["balanced"],
        probes=[0],
        epsilon=0.0,
        output_dir="bench-out",
    )
    assert cfg.epsilon == 0.0
    assert cfg.output_dir.name == "bench-out"


def test_dependency_checker_only_allows_quant_rtn():
    ok, msg = bench_mod.DependencyChecker.check_edit_dependencies("quant_rtn")
    assert ok is True and "Available" in msg
    bad, reason = bench_mod.DependencyChecker.check_edit_dependencies("other")
    assert bad is False and "unsupported" in reason
