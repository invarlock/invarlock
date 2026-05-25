from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.cli.commands.verify import verify_command
from invarlock.core.assurance_contract import ASSURANCE_CLAIM_SET, CANONICAL_GUARD_CHAIN
from invarlock.core.exceptions import ConfigError
from invarlock.reporting import verify_contract as verify_mod
from invarlock.runtime_security import (
    RUNTIME_MANIFEST_FILENAME,
    RUNTIME_MANIFEST_VERSION,
    RUNTIME_VERIFIER_CONTRACT_VERSION,
)

_VALID_TEST_IMAGE_DIGEST = "sha256:" + ("a" * 64)


def _provenance_gate_cert() -> dict:
    return {
        "schema_version": "v1",
        "run_id": "runtime-provenance-gate",
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


def _strict_provenance_gate_cert() -> dict:
    payload = _provenance_gate_cert()
    payload["plugins"] = {"guards": list(CANONICAL_GUARD_CHAIN)}
    payload["guards"] = [{"name": name} for name in CANONICAL_GUARD_CHAIN]
    payload["context"] = {
        "profile": "ci",
        "runtime": {"execution_mode": "container"},
    }
    payload["auto"] = {"tier": "balanced"}
    payload["meta"] = {"profile": "ci"}
    payload["spectral"] = {"supported": True, "status": "pass"}
    payload["rmt"] = {"supported": True, "status": "pass"}
    payload["variance"] = {"supported": True, "status": "pass"}
    payload["invariants"] = {"supported": True, "status": "pass"}
    payload["primary_metric"]["ci"] = [0.0, 0.0]
    payload["assurance"] = {
        "mode": "strict",
        "profile": "ci",
        "tier": "balanced",
        "claim_set": ASSURANCE_CLAIM_SET,
        "canonical_guard_chain": list(CANONICAL_GUARD_CHAIN),
        "guard_chain_observed": list(CANONICAL_GUARD_CHAIN),
        "canonical_guard_chain_enforced": True,
        "fallback_fields_used": False,
        "runtime_provenance_verified": False,
        "runtime_provenance_declared": "container",
        "runtime_provenance_verification_status": "pending",
        "verdict": "pending_verifier",
        "report_local_verdict": "pass",
        "verified_assurance_verdict": "pending",
        "blocking_reasons": [],
    }
    return payload


def _write_runtime_manifest(
    report_path: Path, *, execution_mode: str = "container"
) -> Path:
    payload = {
        "manifest_version": RUNTIME_MANIFEST_VERSION,
        "generated_at_utc": "2026-05-24T00:00:00+00:00",
        "verifier_contract_version": RUNTIME_VERIFIER_CONTRACT_VERSION,
        "report": {
            "path": str(report_path.resolve()),
            "filename": report_path.name,
            "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        },
        "config": {"path": None, "sha256": None, "source": "missing"},
        "execution_mode": execution_mode,
        "runtime": {
            "image_ref": "ghcr.io/invarlock/invarlock-runtime:test",
            "image_digest": _VALID_TEST_IMAGE_DIGEST,
            "container_execution": execution_mode == "container",
            "allow_network": False,
            "allow_remote_code": False,
            "allow_third_party_plugins": False,
        },
    }
    path = report_path.parent / RUNTIME_MANIFEST_FILENAME
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def test_verify_fails_closed_without_runtime_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cert_path = tmp_path / "evaluation.report.json"
    cert_path.write_text(json.dumps(_provenance_gate_cert()), encoding="utf-8")
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
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )

    assert result.exit_code == 1
    assert "runtime.manifest.json missing for" in result.output


def test_verify_allows_unverified_provenance_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cert_path = tmp_path / "evaluation.report.json"
    cert_path.write_text(json.dumps(_provenance_gate_cert()), encoding="utf-8")
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
        ["verify", "--runtime-provenance", "host", str(cert_path)],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )

    assert result.exit_code == 0
    assert "VERIFY OK" in result.output


def test_strict_verify_accepts_pending_report_with_valid_runtime_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cert_path = tmp_path / "evaluation.report.json"
    cert_path.write_text(json.dumps(_strict_provenance_gate_cert()), encoding="utf-8")
    _write_runtime_manifest(cert_path)
    monkeypatch.setattr(
        verify_mod, "_validate_evaluation_report_payload", lambda *args, **kwargs: []
    )

    result = CliRunner().invoke(
        app,
        ["verify", "--assurance", "strict", str(cert_path)],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )

    assert result.exit_code == 0
    assert "VERIFY OK" in result.output


def test_verify_json_reports_runtime_provenance_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cert_path = tmp_path / "evaluation.report.json"
    cert_path.write_text(json.dumps(_strict_provenance_gate_cert()), encoding="utf-8")
    _write_runtime_manifest(cert_path)
    monkeypatch.setattr(
        verify_mod, "_validate_evaluation_report_payload", lambda *args, **kwargs: []
    )

    result = CliRunner().invoke(
        app,
        ["verify", "--assurance", "strict", "--json", str(cert_path)],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    verification = payload["results"][0]["verification"]
    assert verification["runtime_provenance"]["status"] == "verified"
    assert verification["runtime_provenance"]["verified"] is True
    assert verification["runtime_provenance"]["issues"] == []


def test_strict_verify_rejects_host_runtime_provenance_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cert_path = tmp_path / "evaluation.report.json"
    cert_path.write_text(json.dumps(_strict_provenance_gate_cert()), encoding="utf-8")
    monkeypatch.setattr(
        verify_mod, "_validate_evaluation_report_payload", lambda *args, **kwargs: []
    )

    result = CliRunner().invoke(
        app,
        [
            "verify",
            "--assurance",
            "strict",
            "--runtime-provenance",
            "host",
            str(cert_path),
        ],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )

    assert result.exit_code == 1
    assert "verified runtime provenance" in result.output


def test_strict_verify_rejects_host_runtime_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cert_path = tmp_path / "evaluation.report.json"
    cert_path.write_text(json.dumps(_strict_provenance_gate_cert()), encoding="utf-8")
    _write_runtime_manifest(cert_path, execution_mode="host")
    monkeypatch.setattr(
        verify_mod, "_validate_evaluation_report_payload", lambda *args, **kwargs: []
    )

    result = CliRunner().invoke(
        app,
        ["verify", "--assurance", "strict", str(cert_path)],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )

    assert result.exit_code == 1
    assert "marks evaluation.report.json as 'host'" in result.output
    assert "verified runtime provenance" in result.output


def test_strict_verify_rejects_invalid_runtime_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cert_path = tmp_path / "evaluation.report.json"
    cert_path.write_text(json.dumps(_strict_provenance_gate_cert()), encoding="utf-8")
    (tmp_path / RUNTIME_MANIFEST_FILENAME).write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(
        verify_mod, "_validate_evaluation_report_payload", lambda *args, **kwargs: []
    )

    result = CliRunner().invoke(
        app,
        ["verify", "--assurance", "strict", str(cert_path)],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )

    assert result.exit_code == 1
    assert "runtime.manifest.json is invalid" in result.output
    assert "verified runtime provenance" in result.output


def test_strict_verify_rejects_runtime_manifest_digest_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cert_path = tmp_path / "evaluation.report.json"
    cert_path.write_text(json.dumps(_strict_provenance_gate_cert()), encoding="utf-8")
    manifest_path = _write_runtime_manifest(cert_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["report"]["sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        verify_mod, "_validate_evaluation_report_payload", lambda *args, **kwargs: []
    )

    result = CliRunner().invoke(
        app,
        ["verify", "--assurance", "strict", str(cert_path)],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )

    assert result.exit_code == 1
    assert "report digest mismatch" in result.output
    assert "verified runtime provenance" in result.output


def test_verify_command_rejects_invalid_runtime_provenance_value(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="Runtime provenance must be one of: container, host.",
    ):
        verify_command(
            [tmp_path / "evaluation.report.json"],
            runtime_provenance="not-a-real-mode",
        )


def test_verify_json_reports_resolution_for_preload_config_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cert_path = tmp_path / "evaluation.report.json"

    def _boom(*args, **kwargs):
        raise ConfigError(code="E201", message="bad cfg")

    monkeypatch.setattr(verify_mod, "_load_evaluation_report", _boom, raising=True)

    with pytest.raises(typer.Exit) as exc_info:
        verify_command([cert_path], baseline=None, profile="dev", json_out=True)

    assert exc_info.value.exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["resolution"]["exit_code"] == 2
