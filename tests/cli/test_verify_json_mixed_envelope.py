from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from invarlock.cli.commands.verify import verify_command


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_verify_json_mixed_failure_envelope(tmp_path: Path, capsys) -> None:
    # One malformed (missing ratio_vs_baseline), one policy-fail (mismatch)
    base = {
        "schema_version": "v1",
        "run_id": "r",
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
        "primary_metric": {"kind": "ppl_causal", "preview": 10.0, "final": 10.0},
        "baseline_ref": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
    }

    malformed = dict(base)
    policy_fail = json.loads(json.dumps(base))
    policy_fail["primary_metric"]["ratio_vs_baseline"] = 2.0

    p1 = _write(tmp_path / "malformed.json", malformed)
    p2 = _write(tmp_path / "policy_fail.json", policy_fail)

    with pytest.raises(typer.Exit) as ei:
        verify_command([p1, p2], baseline=None, profile="dev", json_out=True)
    out = json.loads(capsys.readouterr().out)
    assert out["certificate"]["count"] == 2
    assert out["summary"]["ok"] is False
    # When any cert is malformed, the envelope labels all as "malformed"
    assert out["summary"]["reason"] == "malformed"
    reasons = {item.get("reason") for item in out.get("results", [])}
    assert reasons == {"malformed"}
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) != 0
