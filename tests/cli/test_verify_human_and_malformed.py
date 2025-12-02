from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from invarlock.cli.app import app


def _cert_ok() -> dict:
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
        # Minimal required fields for schema
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
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 4,
            "stride": 4,
            "preview_n": 1,
            "final_n": 1,
        },
        "baseline_ref": {"primary_metric": {"final": 10.0}},
    }


def test_verify_human_mode_prints_ok_line(tmp_path: Path):
    cert = _cert_ok()
    p = tmp_path / "cert.json"
    p.write_text(json.dumps(cert))
    # No --json: exercise human-mode success print branch
    # Use top-level verify (human-mode by default)
    res = CliRunner().invoke(app, ["verify", str(p)])
    assert res.exit_code == 0
    assert "VERIFY OK" in res.stdout
    assert "metric=ppl_causal" in res.stdout


def test_verify_json_out_malformed_json_triggers_malformed_reason(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{not-json}")
    res = CliRunner().invoke(app, ["verify", "--json", str(p)])
    # Should emit one JSON line with malformed reason and non-zero exit
    lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj.get("summary", {}).get("reason") == "malformed"
    assert obj.get("resolution", {}).get("exit_code") in (1, 2)


def test_verify_human_mode_failure_prints_fail_lines(tmp_path: Path):
    cert = _cert_ok()
    # Mismatch ratio to force policy-fail (not malformed)
    cert["primary_metric"]["ratio_vs_baseline"] = 2.0
    p = tmp_path / "cert.json"
    p.write_text(json.dumps(cert))
    res = CliRunner().invoke(app, ["verify", str(p)])
    assert res.exit_code in (1, 2)
    # Expect FAIL printed and some explanation
    assert "FAIL" in res.stdout
    assert "ratio" in res.stdout.lower()
