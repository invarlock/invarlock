from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from invarlock.cli.commands.verify import verify_command


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _mk_acc_cert(pm_final: float, *, delta: float | None = None) -> dict:
    return {
        "schema_version": "v1",
        "run_id": "r",
        "artifacts": {"generated_at": "t"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "p",
            "seq_len": 8,
            "windows": {
                "preview": 0,
                "final": 0,
                "stats": {
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "coverage": {"preview": {"used": 0}, "final": {"used": 0}},
                    "paired_windows": 0,
                },
            },
        },
        "primary_metric": {
            "kind": "accuracy",
            "final": pm_final,
            "ratio_vs_baseline": (delta if delta is not None else 0.0),
            "display_ci": [pm_final, pm_final],
        },
        "metrics": {"classification": {"n_correct": 8, "n_total": 10}},
        "baseline_ref": {"primary_metric": {"kind": "accuracy", "final": 0.8}},
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
    }


def test_verify_accuracy_recompute_mismatch_dev(tmp_path: Path, capsys) -> None:
    cert = _mk_acc_cert(pm_final=0.75)
    p = _write(tmp_path / "c.json", cert)
    with pytest.raises(typer.Exit):
        verify_command([p], baseline=None, profile="dev", json_out=True)
    out = json.loads(capsys.readouterr().out)
    assert out["resolution"]["exit_code"] != 0


def test_verify_accuracy_recompute_missing_ci_profile(tmp_path: Path) -> None:
    # Remove aggregates; expect E004 in CI profile
    cert = _mk_acc_cert(pm_final=0.80)
    cert["metrics"]["classification"] = {}
    p = _write(tmp_path / "c.json", cert)
    with pytest.raises(typer.Exit) as ei:
        verify_command([p], baseline=None, profile="ci", json_out=True)
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) != 0
