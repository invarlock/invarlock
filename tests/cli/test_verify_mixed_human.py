from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from invarlock.cli.app import app


def _ok_cert() -> dict:
    return {
        "schema_version": "v1",
        "run_id": "r1",
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 10.0,
            "preview": 10.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
        },
        "dataset": {
            "provider": "ds",
            "seq_len": 1,
            "windows": {
                "preview": 1,
                "final": 1,
                "stats": {
                    "coverage": {"preview": {"used": 1}, "final": {"used": 1}},
                    "paired_windows": 1,
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                },
            },
        },
        "meta": {
            "model_id": "m",
            "adapter": "hf",
            "seed": 1,
            "device": "cpu",
            "ts": "2024-01-01T00:00:00",
            "auto": None,
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "plugins": {},
        "flags": {"guard_recovered": False, "rollback_reason": None},
        "baseline_ref": {"primary_metric": {"final": 10.0}},
    }


def test_verify_human_mode_mixed_pass_fail(tmp_path: Path):
    ok = _ok_cert()
    bad = _ok_cert()
    bad["primary_metric"]["ratio_vs_baseline"] = 2.0
    p1 = tmp_path / "ok.json"
    p2 = tmp_path / "bad.json"
    p1.write_text(json.dumps(ok))
    p2.write_text(json.dumps(bad))

    res = CliRunner().invoke(app, ["verify", str(p1), str(p2)])
    assert res.exit_code == 1
    out = res.stdout
    assert "PASS" in out and "FAIL" in out
