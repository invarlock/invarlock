from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import typer

from invarlock.cli.commands.verify import verify_command


def _write_cert(tmp_path: Path, payload: dict, name: str) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _minimal_base_cert_skeleton() -> dict:
    return {
        "schema_version": "v1",
        "run_id": "run-xyz",
        "artifacts": {"generated_at": "2024-01-01T00:00:00"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "unit",
            "seq_len": 8,
            "windows": {
                "preview": 2,
                "final": 2,
                "stats": {
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
                    "paired_windows": 2,
                },
            },
        },
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
        "baseline_ref": {
            "run_id": "base-xyz",
            "model_id": "m",
            "primary_metric": {"kind": "ppl_causal", "final": 10.0},
        },
        "artifacts_extra": {},
    }


def test_verify_accuracy_recompute_success_json(tmp_path: Path, capsys) -> None:
    cert = _minimal_base_cert_skeleton()
    # Accuracy PM with matching aggregates
    cert["primary_metric"] = {
        "kind": "accuracy",
        "final": 0.8,
        "preview": 0.8,
        "ratio_vs_baseline": 0.0,
        "display_ci": [0.8, 0.8],
    }
    cert.setdefault("metrics", {})["classification"] = {"n_correct": 8, "n_total": 10}
    cert["baseline_ref"]["primary_metric"] = {"kind": "accuracy", "final": 0.75}
    p = _write_cert(tmp_path, cert, "acc.json")
    with pytest.raises(typer.Exit) as ei:
        verify_command([p], baseline=None, profile="dev", json_out=True)
    out = json.loads(capsys.readouterr().out)
    assert out["resolution"]["exit_code"] == 0
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 0


def test_verify_ppl_recompute_success_json(tmp_path: Path, capsys) -> None:
    cert = _minimal_base_cert_skeleton()
    # ppl-like PM; final equals exp(mean logloss)
    pm_final = 10.0
    cert["primary_metric"] = {
        "kind": "ppl_causal",
        "final": pm_final,
        "preview": pm_final,
        "ratio_vs_baseline": 1.0,
        "display_ci": [1.0, 1.0],
    }
    # Provide evaluation_windows for recompute
    cert["evaluation_windows"] = {
        "final": {"logloss": [math.log(pm_final)], "token_counts": [1]}
    }
    p = _write_cert(tmp_path, cert, "ppl.json")
    with pytest.raises(typer.Exit) as ei:
        verify_command([p], baseline=None, profile="dev", json_out=True, tolerance=1e-9)
    out = json.loads(capsys.readouterr().out)
    assert out["resolution"]["exit_code"] == 0
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 0


def test_verify_ppl_recompute_analysis_point_success(tmp_path: Path, capsys) -> None:
    cert = _minimal_base_cert_skeleton()
    pm_final = 7.0
    # analysis_point_final = ln(final)
    ap_final = math.log(pm_final)
    cert["primary_metric"] = {
        "kind": "ppl_causal",
        "final": pm_final,
        "preview": pm_final,
        "analysis_point_final": ap_final,
        "ratio_vs_baseline": 1.0,
        "display_ci": [1.0, 1.0],
    }
    cert["evaluation_windows"] = {"final": {"logloss": [ap_final], "token_counts": [1]}}
    # Align baseline to avoid ratio mismatch errors (1.0 = 7.0 / 7.0)
    cert["baseline_ref"]["primary_metric"]["final"] = pm_final
    p = _write_cert(tmp_path, cert, "ppl_ap.json")
    with pytest.raises(typer.Exit) as ei:
        verify_command([p], baseline=None, profile="dev", json_out=True, tolerance=1e-9)
    out = json.loads(capsys.readouterr().out)
    assert out["resolution"]["exit_code"] == 0
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 0


def test_verify_accuracy_dev_missing_aggregates_warns(tmp_path: Path, capsys) -> None:
    cert = _minimal_base_cert_skeleton()
    cert["primary_metric"] = {
        "kind": "accuracy",
        "final": 0.8,
        "preview": 0.8,
        "ratio_vs_baseline": 0.0,
        "display_ci": [0.8, 0.8],
    }
    cert["baseline_ref"]["primary_metric"] = {"kind": "accuracy", "final": 0.75}
    # No classification aggregates → dev path prints warning and continues
    p = _write_cert(tmp_path, cert, "acc_warn.json")
    # Human mode
    verify_command([p], baseline=None, profile="dev", json_out=False)
    out = capsys.readouterr().out
    assert "Cannot recompute accuracy" in out
