from __future__ import annotations

import json

from typer.testing import CliRunner

from invarlock.cli.app import app


def test_explain_gates_missing_files_exits_with_message(tmp_path):
    r = CliRunner().invoke(
        app,
        [
            "report",
            "explain",
            "--report",
            str(tmp_path / "missing1.json"),
            "--baseline",
            str(tmp_path / "missing2.json"),
        ],
    )
    assert r.exit_code == 1
    assert "Missing --report or --baseline file" in r.stdout


def test_explain_gates_invalid_json(tmp_path):
    rep = tmp_path / "rep.json"
    base = tmp_path / "base.json"
    rep.write_text("{invalid", encoding="utf-8")
    base.write_text("{invalid", encoding="utf-8")
    r = CliRunner().invoke(
        app, ["report", "explain", "--report", str(rep), "--baseline", str(base)]
    )
    assert r.exit_code == 1
    assert "Failed to load inputs" in r.stdout


def test_explain_gates_hysteresis_and_overhead_rendering(monkeypatch, tmp_path):
    # Create minimal valid JSON files (their content will be ignored by our patch)
    rep = tmp_path / "rep.json"
    base = tmp_path / "base.json"
    rep.write_text(json.dumps({}), encoding="utf-8")
    base.write_text(json.dumps({}), encoding="utf-8")

    # Patch make_certificate to a simple, controlled payload
    from invarlock.cli.commands import explain_gates as mod

    def _fake_cert(_report, _baseline):  # type: ignore[no-untyped-def]
        return {
            "auto": {"tier": "balanced"},
            "validation": {
                "hysteresis_applied": True,
                "primary_metric_acceptable": True,
                "guard_overhead_acceptable": True,
            },
            "telemetry": {"preview_total_tokens": 30000, "final_total_tokens": 30000},
            "ppl": {
                "ratio_vs_baseline": 1.01,
                "ratio_ci": [0.99, 1.02],
                "drift_ci": [0.98, 1.02],
                "preview_final_ratio": 1.0,
            },
            "guard_overhead": {"overhead_ratio": 1.015, "overhead_threshold": 0.02},
        }

    monkeypatch.setattr(mod, "make_certificate", _fake_cert)
    r = CliRunner().invoke(
        app, ["report", "explain", "--report", str(rep), "--baseline", str(base)]
    )
    assert r.exit_code == 0
    # Hysteresis note printed
    assert "hysteresis applied" in r.stdout.lower()
    # Overhead ratio rendered
    assert "1.015x" in r.stdout
