from __future__ import annotations

import json

from typer.testing import CliRunner

from invarlock.cli.app import app


def test_explain_gates_tokens_below_floor_and_drift_fail(monkeypatch, tmp_path):
    rep = tmp_path / "rep.json"
    base = tmp_path / "base.json"
    rep.write_text(json.dumps({}), encoding="utf-8")
    base.write_text(json.dumps({}), encoding="utf-8")

    from invarlock.cli.commands import explain_gates as mod

    def _fake_cert(_report, _baseline):  # type: ignore[no-untyped-def]
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

    monkeypatch.setattr(mod, "make_certificate", _fake_cert)
    r = CliRunner().invoke(
        app, ["report", "explain", "--report", str(rep), "--baseline", str(base)]
    )
    assert r.exit_code == 0
    # tokens below floor
    assert "below floor" in r.stdout.lower()
    # drift fail
    assert "Gate: Drift" in r.stdout and "FAIL" in r.stdout


def test_explain_gates_handles_edge_cases(monkeypatch, tmp_path):
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

    def fake_cert(_report, _baseline):  # type: ignore[no-untyped-def]
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

    monkeypatch.setattr(mod, "make_certificate", fake_cert)
    result = CliRunner().invoke(
        app, ["report", "explain", "--report", str(rep), "--baseline", str(base)]
    )
    assert result.exit_code == 0
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
        "make_certificate",
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

    result = CliRunner().invoke(
        app, ["report", "explain", "--report", str(rep), "--baseline", str(base)]
    )
    assert result.exit_code == 0
