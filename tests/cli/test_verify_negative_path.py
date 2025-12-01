from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.cli.commands.verify import verify_command
from invarlock.reporting.certificate import make_certificate


def _mk_basic_reports() -> tuple[dict, dict]:
    report = {
        "meta": {"model_id": "m", "adapter": "hf_gpt2", "device": "cpu", "seed": 1},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 50.0,
                "final": 51.0,
                "ratio_vs_baseline": 1.02,
            }
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": [1, 2],
                "logloss": [1.0, 1.1],
                "token_counts": [100, 200],
            },
            "final": {
                "window_ids": [3, 4],
                "logloss": [1.1, 1.2],
                "token_counts": [100, 200],
            },
        },
        "guards": [],
        "artifacts": {"events_path": "", "logs_path": ""},
    }
    baseline = {
        "run_id": "baseline-1",
        "model_id": "m",
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "final": 50.0, "preview": 50.0},
            "bootstrap": {"replicates": 200, "alpha": 0.05},
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": [1, 2],
                "logloss": [1.0, 1.1],
                "token_counts": [100, 200],
            },
            "final": {
                "window_ids": [3, 4],
                "logloss": [1.0, 1.1],
                "token_counts": [100, 200],
            },
        },
        "guards": [],
        "artifacts": {"events_path": "", "logs_path": ""},
    }
    return report, baseline


def test_verify_negative_path_rejects_tampered_ratio(tmp_path: Path, capsys):
    report, baseline = _mk_basic_reports()
    cert = make_certificate(report, baseline)
    # Tamper ratio_vs_baseline to trigger mismatch
    cert["primary_metric"]["ratio_vs_baseline"] = 1.2345
    # Ensure baseline_ref has PM for mismatch detection
    cert.setdefault("baseline_ref", {}).setdefault("primary_metric", {})["final"] = 50.0

    cert_path = tmp_path / "cert.json"
    cert_path.write_text(json.dumps(cert))

    # Direct invocation must disable JSON to get human-readable output
    with pytest.raises(SystemExit):
        verify_command([cert_path], json_out=False)
    out = capsys.readouterr().out
    assert "Primary metric ratio mismatch" in out
