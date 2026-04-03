from __future__ import annotations

import json
import math
from pathlib import Path

from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.reporting import verify_contract as verify_mod


def _attestation_gate_cert() -> dict:
    return {
        "schema_version": "v1",
        "run_id": "runtime-attestation-gate",
        "artifacts": {"generated_at": "2024-01-01T00:00:00Z"},
        "plugins": {},
        "meta": {},
        "provenance": {"provider_digest": {"ids_sha256": "subject-ids"}},
        "dataset": {
            "windows": {
                "preview": 1,
                "final": 1,
                "stats": {
                    "coverage": {"preview": {"used": 1}, "final": {"used": 1}},
                    "paired_windows": 1,
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                },
            }
        },
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 9.0,
            "final": 9.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
        },
        "evaluation_windows": {
            "final": {"logloss": [math.log(9.0)], "token_counts": [1]}
        },
        "baseline_ref": {"primary_metric": {"kind": "ppl_causal", "final": 9.0}},
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
    }


def test_verify_fails_closed_without_runtime_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cert_path = tmp_path / "evaluation.report.json"
    cert_path.write_text(json.dumps(_attestation_gate_cert()), encoding="utf-8")
    monkeypatch.setattr(
        verify_mod, "_validate_evaluation_report_payload", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        verify_mod,
        "_warn_adapter_family_mismatch",
        lambda *args, **kwargs: None,
        raising=True,
    )

    result = CliRunner().invoke(
        app,
        ["verify", str(cert_path)],
        env={"INVARLOCK_ALLOW_UNATTESTED_ARTIFACTS": "0"},
    )

    assert result.exit_code == 1
    assert "runtime.manifest.json missing for" in result.output


def test_verify_allows_unattested_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cert_path = tmp_path / "evaluation.report.json"
    cert_path.write_text(json.dumps(_attestation_gate_cert()), encoding="utf-8")
    monkeypatch.setattr(
        verify_mod, "_validate_evaluation_report_payload", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        verify_mod,
        "_warn_adapter_family_mismatch",
        lambda *args, **kwargs: None,
        raising=True,
    )

    result = CliRunner().invoke(
        app,
        ["verify", "--assurance", "trusted-local", str(cert_path)],
        env={"INVARLOCK_ALLOW_UNATTESTED_ARTIFACTS": "0"},
    )

    assert result.exit_code == 0
    assert "VERIFY OK" in result.output
