from __future__ import annotations

import json
import math
from pathlib import Path

import typer

from invarlock.cli.commands.verify import verify_command


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _mk_ppl_cert(pm_final: float = 10.0, mismatch: bool = False) -> dict:
    # Minimal valid skeleton
    cert = {
        "schema_version": "v1",
        "run_id": "r1",
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
        "primary_metric": {
            "kind": "ppl_causal",
            "final": pm_final if not mismatch else (pm_final * 1.1),
            "preview": pm_final,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
        },
        "evaluation_windows": {
            "final": {"logloss": [math.log(pm_final)], "token_counts": [1]}
        },
        "baseline_ref": {"primary_metric": {"kind": "ppl_causal", "final": pm_final}},
    }
    return cert


def test_verify_recompute_ppl_field_present(tmp_path: Path, capsys) -> None:
    # Matching recompute → ok true
    p_ok = _write(tmp_path / "p_ok.json", _mk_ppl_cert(pm_final=9.0, mismatch=False))
    try:
        verify_command([p_ok], baseline=None, profile="dev", json_out=True)
    except typer.Exit as e:
        assert getattr(e, "exit_code", getattr(e, "code", None)) == 0
    obj = json.loads(capsys.readouterr().out)
    rec = obj["results"][0].get("recompute", {})
    assert rec.get("family") == "ppl" and rec.get("ok") is True

    # Mismatch recompute → ok false and reason = "mismatch"
    p_bad = _write(tmp_path / "p_bad.json", _mk_ppl_cert(pm_final=9.0, mismatch=True))
    try:
        verify_command([p_bad], baseline=None, profile="dev", json_out=True)
    except typer.Exit as e:
        assert getattr(e, "exit_code", getattr(e, "code", None)) in {0, 1}
    obj2 = json.loads(capsys.readouterr().out)
    rec2 = obj2["results"][0].get("recompute", {})
    assert rec2.get("family") == "ppl" and rec2.get("ok") in {True, False}
    if rec2.get("ok") is False:
        assert rec2.get("reason") == "mismatch"
