from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from invarlock.cli.app import app


def _cert_with_windows_and_inconsistent_final() -> dict:
    # PM final (display) is 2.0, but evaluation_windows imply exp(mean)=1.0 → mismatch
    return {
        "schema_version": "v1",
        "run_id": "r1",
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 2.0,
            "preview": 1.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
        },
        "evaluation_windows": {
            "final": {"logloss": [0.0, 0.0], "token_counts": [1, 1]}
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
        "baseline_ref": {"primary_metric": {"final": 1.0}},
    }


def test_verify_detects_display_mismatch_from_recompute(tmp_path: Path):
    cert = _cert_with_windows_and_inconsistent_final()
    p = tmp_path / "cert.json"
    p.write_text(json.dumps(cert))
    res = CliRunner().invoke(app, ["verify", str(p)])
    assert res.exit_code == 1
    assert "Display mismatch" in res.stdout
