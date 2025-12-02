import json
from pathlib import Path

from typer.testing import CliRunner

from invarlock.cli.app import app


def test_verify_json_single_line(tmp_path: Path):
    cert = {
        "schema_version": "v1",
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 1.0,
            "preview": 1.0,
            "ratio_vs_baseline": 1.0,
        },
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.0]}},
        "dataset": {
            "windows": {
                "preview": 1,
                "final": 1,
                "stats": {
                    "coverage": {"final": {"used": 1}, "preview": {"used": 1}},
                    "paired_windows": 1,
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                },
            }
        },
    }
    p = tmp_path / "cert.json"
    p.write_text(json.dumps(cert))
    res = CliRunner().invoke(app, ["verify", "--json", str(p)])
    lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1, f"Expected one JSON line, got {len(lines)}"
    obj = json.loads(lines[0])
    assert obj.get("format_version") == "verify-v1"
    assert obj.get("resolution", {}).get("exit_code") in (0, 1, 2)
