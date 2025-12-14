from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from invarlock.eval import bench
from invarlock.eval.bench_regression import BENCH_GOLDEN_ID, BENCH_GOLDEN_SHA256
from invarlock.reporting.report_types import create_empty_report

TESTS_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TESTS_ROOT.parent
GOLDEN_PATH = TESTS_ROOT / "fixtures" / "benchmarks" / "guard_effect_golden.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _drop_artifacts(payload: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(payload))
    for scenario in out.get("scenarios", []):
        if isinstance(scenario, dict):
            scenario.pop("artifacts", None)
    return out


def _sort_scenarios(payload: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = payload.get("scenarios", [])
    if not isinstance(scenarios, list):
        return []

    def _key(item: dict[str, Any]) -> tuple:
        return (item.get("edit"), item.get("tier"), int(item.get("probes", 0)))

    return sorted(
        [s for s in scenarios if isinstance(s, dict)],
        key=_key,
    )


def _assert_float_close(actual: Any, expected: Any, *, tol: float = 1e-9) -> None:
    assert isinstance(actual, int | float)
    assert isinstance(expected, int | float)
    assert actual == pytest.approx(float(expected), abs=tol, rel=tol)


def test_bench_golden_hash_and_changelog_guard() -> None:
    assert GOLDEN_PATH.is_file()
    assert _sha256(GOLDEN_PATH) == BENCH_GOLDEN_SHA256
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert BENCH_GOLDEN_ID in changelog
    assert BENCH_GOLDEN_SHA256 in changelog


def test_bench_policy_regression_against_golden(tmp_path: Path, monkeypatch) -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    def stub_execute_single_run(  # noqa: PLR0913
        run_config: dict[str, Any],  # noqa: ARG001
        scenario: bench.ScenarioConfig,
        run_type: str,
        output_dir: Path,  # noqa: ARG001
        *,
        runtime: dict[str, Any] | None = None,  # noqa: ARG001
    ) -> bench.RunResult:
        report = create_empty_report()
        report["meta"].update(
            {
                "model_id": scenario.model_id,
                "adapter": scenario.adapter,
                "device": str(scenario.device),
                "seed": scenario.seed,
                "commit": "",
                "ts": "2025-01-01T00:00:00",
            }
        )
        report["data"].update(
            {
                "dataset": "synthetic",
                "split": "validation",
                "seq_len": scenario.seq_len,
                "stride": scenario.stride,
                "preview_n": scenario.preview_n,
                "final_n": scenario.final_n,
            }
        )
        report["edit"].update({"name": scenario.edit, "plan_digest": "pd"})

        if scenario.tier == "balanced":
            guarded_final = 10.09
            guarded_duration = 1.13
            guarded_mem = 1090.0
        elif scenario.tier == "conservative":
            guarded_final = 10.08
            guarded_duration = 1.14
            guarded_mem = 1095.0
        else:  # pragma: no cover
            guarded_final = 10.10
            guarded_duration = 1.10
            guarded_mem = 1080.0

        bare_final = 10.0
        bare_duration = 1.0
        bare_mem = 1000.0

        if run_type == "bare":
            pm_final = bare_final
            duration_s = bare_duration
            mem = bare_mem
            outliers = 2
        else:
            pm_final = guarded_final
            duration_s = guarded_duration
            mem = guarded_mem
            outliers = 3

        report["meta"]["duration_s"] = duration_s
        report["metrics"].update(
            {
                "primary_metric": {
                    "kind": "ppl_causal",
                    "preview": 10.0,
                    "final": pm_final,
                },
                "latency_ms_per_tok": 1.0,
                "memory_mb_peak": mem,
                "rmt": {"outliers": outliers},
                "invariants": {"violations": 0},
            }
        )
        if run_type == "guarded":
            report["guards"] = [
                {
                    "name": "rmt",
                    "policy": {"deadband": 0.10, "margin": 1.5},
                    "metrics": {"outliers_total": outliers},
                    "violations": [],
                },
                {
                    "name": "invariants",
                    "policy": {},
                    "metrics": {"violations_found": 0},
                    "violations": [],
                },
            ]
        report["flags"].update({"guard_recovered": False, "rollback_reason": None})
        return bench.RunResult(run_type=run_type, report=report, success=True)

    monkeypatch.setattr(bench, "execute_single_run", stub_execute_single_run)

    bench.run_guard_effect_benchmark(
        edits=["quant_rtn"],
        tiers=["balanced", "conservative"],
        probes=[0],
        profile="ci",
        output_dir=tmp_path,
        dataset="synthetic",
    )

    generated = json.loads(
        (tmp_path / "results" / "guard_effect.json").read_text(encoding="utf-8")
    )
    generated = _drop_artifacts(generated)

    assert generated.get("schema_version") == golden.get("schema_version")
    assert generated.get("profile") == golden.get("profile")
    assert generated.get("seed") == golden.get("seed")
    assert generated.get("epsilon") == golden.get("epsilon")

    got_scenarios = _sort_scenarios(generated)
    exp_scenarios = _sort_scenarios(golden)
    assert len(got_scenarios) == len(exp_scenarios)

    float_keys = {
        "primary_metric_bare",
        "primary_metric_guarded",
        "primary_metric_overhead",
        "latency_bare",
        "latency_guarded",
        "guard_overhead_time",
        "mem_bare",
        "mem_guarded",
        "guard_overhead_mem",
        "epsilon",
    }

    for got, exp in zip(got_scenarios, exp_scenarios, strict=False):
        assert (got.get("edit"), got.get("tier"), got.get("probes")) == (
            exp.get("edit"),
            exp.get("tier"),
            exp.get("probes"),
        )
        for key, exp_val in exp.items():
            if key in {"pass"}:
                assert got.get(key) == exp_val
                continue
            if key in float_keys:
                _assert_float_close(got.get(key), exp_val, tol=1e-6)
                continue
            assert got.get(key) == exp_val


pytestmark = pytest.mark.regression
