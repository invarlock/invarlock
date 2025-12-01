from __future__ import annotations

import json
from pathlib import Path

from invarlock.cli.commands.verify import verify_command


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_verify_human_success_prints_width(tmp_path: Path, capsys) -> None:
    cert = {
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
            "display_ci": [0.95, 1.05],
        },
        # ppl.stats path so human summary can extract n={preview/final}
        "ppl": {"stats": {"coverage": {"preview": {"used": 1}, "final": {"used": 1}}}},
        "baseline_ref": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
    }
    p = _write(tmp_path / "c.json", cert)
    verify_command([p], baseline=None, profile="dev", json_out=False)
    out = capsys.readouterr().out
    assert "VERIFY OK" in out and "width=" in out and "ci=[0.950000,1.050000]" in out
