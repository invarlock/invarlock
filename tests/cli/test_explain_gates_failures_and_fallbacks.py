from __future__ import annotations

import json

import pytest
import typer
from typer.testing import CliRunner

from invarlock.cli.app import app
from tests.cli._support_console import RecordingConsole


def test_explain_gates_tokens_below_floor_and_drift_fail(monkeypatch, tmp_path):
    rep = tmp_path / "rep.json"
    base = tmp_path / "base.json"
    rep.write_text(json.dumps({}), encoding="utf-8")
    base.write_text(json.dumps({}), encoding="utf-8")

    from invarlock.cli.commands import explain_gates as mod

    def _fake_cert(_report, _baseline):
        return {
            "auto": {"tier": "balanced"},
            "validation": {
                "hysteresis_applied": False,
                "primary_metric_acceptable": True,
                "preview_final_drift_acceptable": False,
            },
            "telemetry": {"preview_total_tokens": 10_000, "final_total_tokens": 10_000},
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 11.0,
                "ratio_vs_baseline": 1.01,
                "display_ci": [0.99, 1.02],
            },
        }

    monkeypatch.setattr(mod, "make_report", _fake_cert)
    r = CliRunner().invoke(
        app,
        [
            "report",
            "explain",
            "--subject-report",
            str(rep),
            "--baseline-report",
            str(base),
        ],
    )
    assert r.exit_code == 0
    # tokens below floor
    assert "below floor" in r.stdout.lower()
    # drift fail
    assert "Gate: Drift" in r.stdout and "FAIL" in r.stdout


def test_explain_gates_handles_failures_and_threshold_edges(monkeypatch, tmp_path):
    rep = tmp_path / "rep.json"
    base = tmp_path / "base.json"
    rep.write_text(
        json.dumps(
            {"provenance": {"dataset_split": "validation", "split_fallback": True}}
        ),
        encoding="utf-8",
    )
    base.write_text(json.dumps({}), encoding="utf-8")

    from invarlock.cli.commands import explain_gates as mod

    class FlakyTelemetry(dict):
        def __init__(self):
            super().__init__(preview_total_tokens=12_000, final_total_tokens=30_000)
            self._raised = False

        def get(self, key, default=None):
            if not self._raised and key == "preview_total_tokens":
                self._raised = True
                raise ValueError("boom")
            return super().get(key, default)

    def fake_cert(_report, _baseline):
        return {
            "auto": {"tier": "balanced"},
            "validation": {
                "primary_metric_acceptable": False,
                "preview_final_drift_acceptable": True,
                "guard_overhead_acceptable": False,
            },
            "telemetry": FlakyTelemetry(),
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 1.0,
                "final": 0.97,
                "ratio_vs_baseline": 1.2,
            },
            "guard_overhead": {
                "threshold_percent": "n/a",
                "overhead_threshold": 0.05,
                "overhead_percent": 2.5,
            },
        }

    monkeypatch.setattr(mod, "make_report", fake_cert)
    result = CliRunner().invoke(
        app,
        [
            "report",
            "explain",
            "--subject-report",
            str(rep),
            "--baseline-report",
            str(base),
        ],
    )
    assert "Dataset split: validation (fallback)" in result.stdout
    assert "observed: 1.200x" in result.stdout
    assert "observed: 0.970" in result.stdout
    assert "+5.0%" in result.stdout


def test_explain_gates_dataset_split_handles_exception(monkeypatch, tmp_path):
    rep = tmp_path / "rep.json"
    base = tmp_path / "base.json"
    rep.write_text("{}", encoding="utf-8")
    base.write_text("{}", encoding="utf-8")

    from invarlock.cli.commands import explain_gates as mod

    class BadMapping(dict):
        def get(self, *_args, **_kwargs):  # pragma: no cover - invoked via command
            raise RuntimeError("broken")

    calls = {"count": 0}

    def fake_loads(payload):
        calls["count"] += 1
        if calls["count"] == 1:
            return BadMapping()
        return {}

    monkeypatch.setattr(mod.json, "loads", fake_loads)
    monkeypatch.setattr(
        mod,
        "make_report",
        lambda *_: {
            "auto": {"tier": "balanced"},
            "validation": {"primary_metric_acceptable": True},
            "telemetry": {"preview_total_tokens": 0, "final_total_tokens": 0},
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": [1.0, 1.0],
            },
        },
    )

    mod.explain_gates_command(
        subject_report=str(rep),
        baseline_report=str(base),
    )
    assert calls["count"] >= 2


def test_explain_gates_missing_and_load_failures(tmp_path, monkeypatch) -> None:
    from invarlock.cli.commands import explain_gates as mod

    console = RecordingConsole()
    monkeypatch.setattr(mod, "console", console)

    missing = tmp_path / "missing.json"
    present = tmp_path / "present.json"
    present.write_text("{}", encoding="utf-8")

    with pytest.raises(typer.Exit) as missing_exc:
        mod.explain_gates_command(
            subject_report=str(missing),
            baseline_report=str(present),
        )
    assert missing_exc.value.exit_code == 1
    assert any(
        "Missing --subject-report or --baseline-report file" in line
        for line in console.lines
    )

    bad = tmp_path / "bad.json"
    bad.write_text("{not-json", encoding="utf-8")
    console.calls.clear()
    with pytest.raises(typer.Exit) as load_exc:
        mod.explain_gates_command(
            subject_report=str(bad),
            baseline_report=str(present),
        )
    assert load_exc.value.exit_code == 1
    assert any("Failed to load inputs" in line for line in console.lines)


def test_explain_gates_info_tail_and_ratio_only_overhead(tmp_path, monkeypatch) -> None:
    from invarlock.cli.commands import explain_gates as mod

    report = tmp_path / "report.json"
    baseline = tmp_path / "baseline.json"
    report.write_text(
        json.dumps(
            {"provenance": {"dataset_split": "validation", "split_fallback": True}}
        ),
        encoding="utf-8",
    )
    baseline.write_text("{}", encoding="utf-8")

    console = RecordingConsole()
    monkeypatch.setattr(mod, "console", console)
    monkeypatch.setattr(
        mod,
        "get_tier_policies",
        lambda: {"balanced": {"metrics": {}}, "aggressive": {"metrics": "invalid"}},
    )
    monkeypatch.setattr(
        mod,
        "make_report",
        lambda *_args: {
            "auto": {"tier": "balanced", "tiny_relax": True},
            "validation": {
                "primary_metric_acceptable": False,
                "preview_final_drift_acceptable": True,
                "guard_overhead_acceptable": True,
            },
            "resolved_policy": {"metrics": "ignore-me"},
            "telemetry": {"preview_total_tokens": 1, "final_total_tokens": 2},
            "primary_metric_tail": {
                "evaluated": False,
                "mode": "warn",
                "policy": {"quantile": "oops"},
                "stats": {"epsilon": 1e-6},
            },
            "guard_overhead": {"overhead_threshold": 0.05, "overhead_ratio": 1.03},
        },
    )

    mod.explain_gates_command(
        subject_report=str(report),
        baseline_report=str(baseline),
    )

    joined = console.joined()
    assert "threshold: unavailable" in joined
    assert "tiny relax enabled" in joined
    assert "status: INFO" in joined
    assert "Dataset split: validation (fallback)" in joined
    assert "observed: 1.030x" in joined
    assert "threshold: ≤ +5.0%" in joined


def test_explain_gates_hysteresis_warn_tail_and_drift_defaults(
    tmp_path, monkeypatch
) -> None:
    from invarlock.cli.commands import explain_gates as mod

    report = tmp_path / "report.json"
    baseline = tmp_path / "baseline.json"
    report.write_text("{}", encoding="utf-8")
    baseline.write_text("{}", encoding="utf-8")

    console = RecordingConsole()
    monkeypatch.setattr(mod, "console", console)
    monkeypatch.setattr(
        mod,
        "get_tier_policies",
        lambda: {
            "balanced": {
                "metrics": {
                    "pm_ratio": {"ratio_limit_base": 1.05, "hysteresis_ratio": 0.02}
                }
            }
        },
    )
    monkeypatch.setattr(
        mod,
        "make_report",
        lambda *_args: {
            "auto": {"tier": "balanced"},
            "validation": {
                "primary_metric_acceptable": True,
                "preview_final_drift_acceptable": False,
                "hysteresis_applied": True,
                "guard_overhead_acceptable": False,
            },
            "resolved_policy": {
                "metrics": {
                    "pm_ratio": {
                        "ratio_limit_base": 1.05,
                        "hysteresis_ratio": 0.02,
                        "min_tokens": 1,
                    }
                }
            },
            "telemetry": {"preview_total_tokens": 10, "final_total_tokens": 10},
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 0.0,
                "final": 1.0,
                "ratio_vs_baseline": 1.01,
                "display_ci": [1.0],
                "drift_band": {"min": -1.0, "max": 0.0},
            },
            "primary_metric_tail": {
                "evaluated": True,
                "passed": False,
                "mode": "warn",
                "policy": {"quantile": 0.95, "quantile_max": 0.2, "mass_max": 0.1},
                "stats": {"q95": 0.3, "tail_mass": 0.2},
            },
            "guard_overhead": {"threshold_percent": "n/a"},
        },
    )

    mod.explain_gates_command(
        subject_report=str(report),
        baseline_report=str(baseline),
    )

    joined = console.joined()
    assert "status: PASS" in joined
    assert "effective threshold = 1.070x" in joined
    assert "Gate: Primary Metric Tail" in joined
    assert "status: WARN" in joined
    assert "observed: P95=0.3000" in joined
    assert "tail_mass: Pr[ΔlogNLL > ε]=0.2000" in joined
    assert "threshold: 0.95–1.05x" in joined
    assert "observed: N/A" in joined
    assert "status: FAIL" in joined
