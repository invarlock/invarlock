from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from invarlock.cli.commands.verify import verify_command


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _base_cert() -> dict:
    return {
        "schema_version": "v1",
        "run_id": "r1",
        "artifacts": {"generated_at": "t"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "p",
            "seq_len": 8,
            "windows": {
                "preview": 1,
                "final": 1,
                "stats": {
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "coverage": {"preview": {"used": 1}, "final": {"used": 1}},
                    "paired_windows": 1,
                },
            },
        },
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": 10.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
        },
        "baseline_ref": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
    }


def test_verify_profile_lints_equals_gte_lte(tmp_path: Path, capsys) -> None:
    # Passing lints
    cert_ok = _base_cert()
    cert_ok["meta"]["model_profile"] = {
        "cert_lints": [
            {
                "type": "equals",
                "path": "dataset.seq_len",
                "value": 8,
                "message": "seq len eq",
            },
            {
                "type": "gte",
                "path": "primary_metric.final",
                "value": 9.0,
                "message": "pm gte",
            },
            {
                "type": "lte",
                "path": "primary_metric.final",
                "value": 11.0,
                "message": "pm lte",
            },
        ]
    }
    p_ok = _write(tmp_path / "ok.json", cert_ok)
    with pytest.raises(typer.Exit) as ei_ok:
        verify_command([p_ok], baseline=None, profile="dev", json_out=True)
    out_ok = json.loads(capsys.readouterr().out)
    assert out_ok["resolution"]["exit_code"] == 0
    assert getattr(ei_ok.value, "exit_code", getattr(ei_ok.value, "code", None)) == 0

    # Failing lints (equals and gte)
    cert_bad = _base_cert()
    cert_bad["meta"]["model_profile"] = {
        "cert_lints": [
            {
                "type": "equals",
                "path": "dataset.seq_len",
                "value": 16,
                "message": "seq len eq",
            },
            {
                "type": "gte",
                "path": "primary_metric.final",
                "value": 11.1,
                "message": "pm gte",
            },
        ]
    }
    p_bad = _write(tmp_path / "bad.json", cert_bad)
    with pytest.raises(typer.Exit) as ei_bad:
        verify_command([p_bad], baseline=None, profile="dev", json_out=True)
    out_bad = json.loads(capsys.readouterr().out)
    assert out_bad["resolution"]["exit_code"] != 0
    assert getattr(ei_bad.value, "exit_code", getattr(ei_bad.value, "code", None)) != 0
