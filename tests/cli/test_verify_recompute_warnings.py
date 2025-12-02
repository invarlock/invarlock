from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from invarlock.cli.app import app


def _base_cert(kind: str) -> dict:
    cert = {
        "schema_version": "v1",
        "run_id": "r1",
        "primary_metric": {
            "kind": kind,
            "final": 1.0,
            "preview": 1.0,
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
        "baseline_ref": {"primary_metric": {"final": 1.0}},
    }
    return cert


def test_verify_dev_warns_when_ppl_basis_cannot_recompute(tmp_path: Path):
    cert = _base_cert("ppl_causal")
    # No evaluation_windows.final → dev prints yellow warning, still passes
    p = tmp_path / "cert.json"
    p.write_text(json.dumps(cert))
    res = CliRunner().invoke(app, ["verify", str(p)])
    assert res.exit_code == 0


def test_verify_dev_warns_when_accuracy_cannot_recompute(tmp_path: Path):
    cert = _base_cert("accuracy")
    p = tmp_path / "cert_acc.json"
    p.write_text(json.dumps(cert))
    res = CliRunner().invoke(app, ["verify", str(p)])
    assert res.exit_code == 0


def test_verify_json_success_multiple_certs(tmp_path: Path):
    c1 = _base_cert("ppl_causal")
    c2 = _base_cert("ppl_causal")
    p1 = tmp_path / "c1.json"
    p2 = tmp_path / "c2.json"
    p1.write_text(json.dumps(c1))
    p2.write_text(json.dumps(c2))
    res = CliRunner().invoke(app, ["verify", "--json", str(p1), str(p2)])
    lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj.get("certificate", {}).get("count") == 2
