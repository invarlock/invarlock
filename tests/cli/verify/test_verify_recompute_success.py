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
        "delta_vs_baseline_pp": 5.0,
        "display_ci": [5.0, 5.0],
        "n_final": 200,
    }
    cert.setdefault("metrics", {})["classification"] = {
        "n_correct": 160,
        "n_total": 200,
    }
    cert["dataset"]["windows"]["preview"] = 200
    cert["dataset"]["windows"]["final"] = 200
    cert["dataset"]["windows"]["stats"]["paired_windows"] = 200
    cert["dataset"]["windows"]["stats"]["coverage"]["preview"]["used"] = 200
    cert["dataset"]["windows"]["stats"]["coverage"]["final"]["used"] = 200
    cert["baseline_ref"]["primary_metric"] = {"kind": "accuracy", "final": 0.75}
    p = _write_cert(tmp_path, cert, "acc.json")
    with pytest.raises(typer.Exit) as ei:
        verify_command([p], baseline=None, profile="dev", json_out=True)
    out = json.loads(capsys.readouterr().out)
    assert "resolution" not in out
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 0
    result = out["results"][0]
    assert "final" not in result
    assert result["delta_vs_baseline_pp"] == 5.0
    assert "ratio_vs_baseline" not in result


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
    assert "resolution" not in out
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
    assert "resolution" not in out
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 0


def test_verify_accuracy_dev_rejects_missing_aggregates(tmp_path: Path, capsys) -> None:
    cert = _minimal_base_cert_skeleton()
    cert["primary_metric"] = {
        "kind": "accuracy",
        "final": 0.8,
        "preview": 0.8,
        "delta_vs_baseline_pp": 5.0,
        "display_ci": [5.0, 5.0],
    }
    cert["baseline_ref"]["primary_metric"] = {"kind": "accuracy", "final": 0.75}
    # Missing classification aggregates cannot substantiate the accuracy result.
    p = _write_cert(tmp_path, cert, "acc_warn.json")
    with pytest.raises(SystemExit) as exc:
        verify_command([p], baseline=None, profile="dev", json_out=False)
    out = capsys.readouterr().out
    assert exc.value.code == 1
    assert "requires measured classification aggregates" in out
