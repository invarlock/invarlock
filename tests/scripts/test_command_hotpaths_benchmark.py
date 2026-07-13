from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path("scripts/checks/benchmark_command_hotpaths.py")


def _load_benchmark_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_command_hotpaths", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _result(*, median: float, input_digest: str = "a", output_digest: str = "b"):
    return {
        "median_walltime_seconds": median,
        "input_sha256": input_digest * 64,
        "output_sha256": output_digest * 64,
    }


def test_benchmark_defaults_exercise_every_local_hotpath() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--repeat", "1", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["schema"] == "invarlock.command_hotpaths_benchmark.v1"
    assert payload["publication"] == "local-only"
    assert payload["results_persisted"] is False
    assert payload["repeat"] == 1
    assert set(payload["benchmarks"]) == {
        "bootstrap",
        "evidence-snapshot-verify",
        "report-assembly",
        "schema-validation",
    }
    assert set(payload["platform"]) == {
        "machine",
        "python_implementation",
        "python_version",
        "release",
        "system",
    }
    assert payload["commit"]["sha"] == "unknown" or len(payload["commit"]["sha"]) == 40
    assert isinstance(payload["commit"]["dirty"], bool)

    for benchmark in payload["benchmarks"].values():
        assert benchmark["repeat"] == 1
        assert benchmark["median_walltime_seconds"] > 0
        assert benchmark["p95_walltime_seconds"] > 0
        assert benchmark["peak_rss_bytes"] > 0
        assert len(benchmark["input_sha256"]) == 64
        assert len(benchmark["output_sha256"]) == 64
        assert benchmark["cuda"] is None


def test_benchmark_selection_runs_only_requested_hotpath() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--benchmark",
            "schema-validation",
            "--repeat",
            "1",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert set(json.loads(result.stdout)["benchmarks"]) == {"schema-validation"}


def test_cli_acceptance_comparison_controls_exit_status(tmp_path: Path) -> None:
    measured = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--benchmark",
            "schema-validation",
            "--repeat",
            "1",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    baseline = json.loads(measured.stdout)
    baseline_path = tmp_path / "baseline.json"
    baseline["benchmarks"]["schema-validation"]["median_walltime_seconds"] = 1_000.0
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    command = [
        sys.executable,
        str(SCRIPT),
        "--benchmark",
        "schema-validation",
        "--repeat",
        "1",
        "--baseline-json",
        str(baseline_path),
        "--target",
        "schema-validation",
        "--json",
    ]

    accepted = subprocess.run(command, check=False, capture_output=True, text=True)
    assert accepted.returncode == 0
    assert json.loads(accepted.stdout)["acceptance"]["passed"] is True

    baseline["benchmarks"]["schema-validation"]["median_walltime_seconds"] = 1e-12
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    rejected = subprocess.run(command, check=False, capture_output=True, text=True)
    assert rejected.returncode == 1
    assert json.loads(rejected.stdout)["acceptance"]["passed"] is False


def test_acceptance_comparison_requires_target_gain_without_other_regressions() -> None:
    benchmark = _load_benchmark_module()
    baseline = {
        "benchmarks": {
            "schema-validation": _result(median=1.0),
            "bootstrap": _result(median=2.0, input_digest="c", output_digest="d"),
        }
    }
    current = {
        "benchmarks": {
            "schema-validation": _result(median=0.89),
            "bootstrap": _result(median=2.0, input_digest="c", output_digest="d"),
        }
    }

    comparison = benchmark.compare_benchmark_results(
        current,
        baseline,
        targets={"schema-validation"},
        minimum_target_improvement=0.10,
        maximum_untargeted_regression=0.0,
    )

    assert comparison["passed"] is True
    assert comparison["benchmarks"]["schema-validation"]["passed"] is True
    assert comparison["benchmarks"]["bootstrap"]["passed"] is True


@pytest.mark.parametrize(
    ("current", "expected_reason"),
    [
        (
            {
                "schema-validation": _result(median=0.95),
                "bootstrap": _result(median=2.0, input_digest="c", output_digest="d"),
            },
            "target improvement below threshold",
        ),
        (
            {
                "schema-validation": _result(median=0.89),
                "bootstrap": _result(median=2.01, input_digest="c", output_digest="d"),
            },
            "untargeted regression above threshold",
        ),
        (
            {
                "schema-validation": _result(median=0.89, output_digest="e"),
                "bootstrap": _result(median=2.0, input_digest="c", output_digest="d"),
            },
            "output digest mismatch",
        ),
    ],
)
def test_acceptance_comparison_fails_closed(
    current: dict[str, dict[str, object]], expected_reason: str
) -> None:
    benchmark = _load_benchmark_module()
    baseline = {
        "benchmarks": {
            "schema-validation": _result(median=1.0),
            "bootstrap": _result(median=2.0, input_digest="c", output_digest="d"),
        }
    }

    comparison = benchmark.compare_benchmark_results(
        {"benchmarks": current},
        baseline,
        targets={"schema-validation"},
        minimum_target_improvement=0.10,
        maximum_untargeted_regression=0.0,
    )

    assert comparison["passed"] is False
    assert expected_reason in {
        item["reason"]
        for item in comparison["benchmarks"].values()
        if item["passed"] is False
    }
