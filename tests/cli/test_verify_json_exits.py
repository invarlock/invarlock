import json
from pathlib import Path

from typer.testing import CliRunner

from invarlock.cli.app import app


def _invoke_verify_json(cert_obj: dict, tmp_path: Path):
    p = tmp_path / "cert.json"
    p.write_text(json.dumps(cert_obj))
    res = CliRunner().invoke(app, ["verify", "--json", str(p)])
    assert res.exit_code in (0, 1, 2)
    payload = json.loads(res.stdout.strip().splitlines()[-1])
    return res, payload


def test_verify_json_malformed_missing_ratio(tmp_path: Path):
    bad = {
        "schema_version": "v1",
        "primary_metric": {"kind": "ppl_causal", "final": 10.0, "preview": 10.0},
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
    res, obj = _invoke_verify_json(bad, tmp_path)
    assert obj.get("format_version") == "verify-v1"
    assert obj.get("summary", {}).get("reason") == "malformed"
    assert obj.get("resolution", {}).get("exit_code") in (1, 2)
    assert isinstance(obj.get("results"), list)
    assert obj["results"][0].get("reason") in {"malformed", "policy_fail"}


def test_verify_json_policy_fail_reason(tmp_path: Path):
    # Cert with drift out of band to trigger policy fail
    cert = {
        "schema_version": "v1",
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 2.0,
            "preview": 1.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
        },
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [2.3]}},
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
    res, obj = _invoke_verify_json(cert, tmp_path)
    assert obj.get("format_version") == "verify-v1"
    assert obj.get("summary", {}).get("reason") in {"policy_fail", "malformed"}
    assert obj.get("resolution", {}).get("exit_code") in (1, 2)
