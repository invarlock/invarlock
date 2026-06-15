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
            "--subject-report",
            str(tmp_path / "missing1.json"),
            "--baseline-report",
            str(tmp_path / "missing2.json"),
        ],
    )
    assert r.exit_code == 2
    assert "Path not found" in r.stdout


def test_explain_gates_invalid_json(tmp_path):
    rep = tmp_path / "rep.json"
    base = tmp_path / "base.json"
    rep.write_text("{invalid", encoding="utf-8")
    base.write_text("{invalid", encoding="utf-8")
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
    assert r.exit_code == 2
    assert "not valid JSON" in r.stdout


def test_explain_gates_hysteresis_and_overhead_rendering(monkeypatch, tmp_path):
    # Create minimal valid JSON files (their content will be ignored by our patch)
    rep = tmp_path / "rep.json"
    base = tmp_path / "base.json"
    rep.write_text(json.dumps({}), encoding="utf-8")
    base.write_text(json.dumps({}), encoding="utf-8")

    # Patch make_report to a simple, controlled payload
    from invarlock.cli.commands import explain_gates as mod

    def _fake_cert(_report, _baseline):
        return {
            "auto": {"tier": "balanced"},
            "validation": {
                "hysteresis_applied": True,
                "primary_metric_acceptable": True,
                "preview_final_drift_acceptable": True,
                "guard_overhead_acceptable": True,
            },
            "telemetry": {"preview_total_tokens": 30000, "final_total_tokens": 30000},
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.01,
                "display_ci": [0.99, 1.02],
            },
            "guard_overhead": {"overhead_ratio": 1.015, "overhead_threshold": 0.02},
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
    # Hysteresis note printed
    assert "hysteresis applied" in r.stdout.lower()
    # Overhead ratio rendered
    assert "1.015x" in r.stdout


def test_explain_gates_accuracy_uses_delta_pp_and_audit_outline(monkeypatch, tmp_path):
    rep = tmp_path / "rep.json"
    base = tmp_path / "base.json"
    rep.write_text(json.dumps({}), encoding="utf-8")
    base.write_text(json.dumps({}), encoding="utf-8")

    from invarlock.cli.commands import explain_gates as mod

    def _fake_cert(_report, _baseline):
        return {
            "meta": {"model_id": "vision-model", "adapter": "hf_multimodal"},
            "auto": {"tier": "balanced"},
            "dataset": {
                "provider": "vision_text",
                "windows": {"preview": 2, "final": 2},
                "hash": {"source": "manifest"},
            },
            "provenance": {"provider_digest": {"ids_sha256": "abc"}},
            "policy_digest": {"thresholds_hash": "1234567890abcdef"},
            "validation": {
                "primary_metric_acceptable": True,
                "preview_final_drift_acceptable": True,
                "invariants_pass": True,
                "spectral_stable": True,
                "rmt_stable": True,
            },
            "primary_metric": {
                "kind": "accuracy",
                "preview": 0.850,
                "final": 0.855,
                "ratio_vs_baseline": +0.50,
                "display_ci": [-0.1, 1.1],
            },
            "resolved_policy": {
                "metrics": {
                    "accuracy": {
                        "delta_min_pp": -1.0,
                        "preview_final_delta_pp_max": 0.01,
                    }
                }
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
    assert "observed: +0.50 pp" in r.stdout
    assert "threshold: ≥ -1.00 pp" in r.stdout
    assert "observed: +0.50 pp" in r.stdout
    assert "threshold: ≤ ±1.00 pp" in r.stdout
    assert "Evidence And Provenance" in r.stdout
    assert "Dataset: vision_text" in r.stdout
