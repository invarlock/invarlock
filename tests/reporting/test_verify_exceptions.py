from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from invarlock.cli.app import app


def test_verify_json_malformed_schema_raises_validation_error(tmp_path: Path) -> None:
    # Minimal JSON that parses but violates certificate schema/shape
    bad = {"schema_version": "v1", "run_id": "", "primary_metric": {}}
    p = tmp_path / "bad_cert.json"
    p.write_text(json.dumps(bad))

    res = CliRunner().invoke(app, ["verify", "--json", str(p)])
    assert res.exit_code != 0
    obj = json.loads(res.stdout.splitlines()[0])
    # Expect structured error with typed ValidationError (E601)
    err = obj.get("error", {})
    assert err.get("category") in {"ValidationError", "InvarlockError"}
    assert err.get("code") == "E601"
    # Reason may be policy_fail or malformed; ensure JSON present
    assert obj.get("summary", {}).get("ok") is False


def test_verify_json_recompute_mismatch_metrics_error_in_ci(tmp_path: Path) -> None:
    # Construct a certificate with ppl kind and mismatched ratio vs baseline
    cert = {
        "schema_version": "v1",
        "run_id": "r1",
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 12.0,
            "preview": 10.0,
            "ratio_vs_baseline": 1.2345,
            "display_ci": [1.0, 1.0],
        },
        "provenance": {"provider_digest": {"ids_sha256": "deadbeef"}},
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
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 4,
            "stride": 4,
            "preview_n": 1,
            "final_n": 1,
        },
        "baseline_ref": {"primary_metric": {"final": 10.0}},
        # Provide evaluation windows with mismatched recompute vs displayed
        "evaluation_windows": {
            "final": {
                "logloss": [0.0, 1.0],
                "token_counts": [1, 1],
            }
        },
    }

    p = tmp_path / "mismatch_cert.json"
    p.write_text(json.dumps(cert))

    # Use CI profile to treat recompute mismatch as fatal
    res = CliRunner().invoke(app, ["verify", "--json", "--profile", "ci", str(p)])
    assert res.exit_code != 0
    obj = json.loads(res.stdout.splitlines()[0])
    err = obj.get("error", {})
    assert err.get("code") == "E602"
    assert err.get("category") in {"MetricsError", "InvarlockError"}
