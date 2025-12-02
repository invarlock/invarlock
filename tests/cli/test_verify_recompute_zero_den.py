from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from invarlock.cli.commands.verify import verify_command


def _write_cert(tmp_path: Path, payload: dict, name: str) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _minimal_cert() -> dict:
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


def test_verify_ppl_recompute_zero_denominator_is_noop(tmp_path: Path, capsys) -> None:
    cert = _minimal_cert()
    # ppl-like PM
    cert["primary_metric"] = {
        "kind": "ppl_causal",
        "final": 10.0,
        "preview": 10.0,
        "ratio_vs_baseline": 1.0,
        "display_ci": [1.0, 1.0],
    }
    # Provide evaluation_windows with zero total token count → recompute is a no-op
    cert["evaluation_windows"] = {
        "final": {"logloss": [1.0, 1.1], "token_counts": [0, 0]}
    }
    p = _write_cert(tmp_path, cert, "ppl_zero_den.json")
    with pytest.raises(typer.Exit) as ei:
        verify_command([p], baseline=None, profile="dev", json_out=True)
    out = json.loads(capsys.readouterr().out)
    assert out["resolution"]["exit_code"] == 0
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 0
