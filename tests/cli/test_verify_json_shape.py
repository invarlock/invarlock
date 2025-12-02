import json
from pathlib import Path

from typer.testing import CliRunner

from invarlock.cli.app import app


def test_verify_json_envelope_is_stable(tmp_path: Path):
    # Minimal passing-ish v1 cert with PM ratio
    cert = {
        "schema_version": "v1",
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 10.0,
            "preview": 10.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
        },
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [2.3]}},
        "dataset": {
            "windows": {
                "preview": 0,
                "final": 1,
                "stats": {
                    "coverage": {"final": {"used": 1}, "preview": {"used": 0}},
                    "paired_windows": 0,
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                },
            }
        },
    }
    path = tmp_path / "cert.json"
    path.write_text(json.dumps(cert))
    res = CliRunner().invoke(app, ["verify", "--json", str(path)])
    assert res.exit_code in (0, 1, 2)
    last = res.stdout.strip().splitlines()[-1]
    obj = json.loads(last)
    assert obj.get("format_version") == "verify-v1"
    assert {"summary", "resolution"} <= set(obj.keys())
    assert "exit_code" in (obj.get("resolution") or {})
