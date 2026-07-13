from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from invarlock import __version__
from invarlock.cli.app import app
from invarlock.cli.commands.verify import verify_command
from invarlock.core.exceptions import ConfigError
from invarlock.reporting import verify_adapter_family
from invarlock.reporting import verify_contract as verify_mod
from invarlock.runtime_security import (
    RUNTIME_MANIFEST_FILENAME,
)
from tests.cli._support_verify_runtime_provenance import (
    _VALID_TEST_IMAGE_DIGEST,
    _matching_strict_policy_pack,
    _matching_strict_ppl_baseline,
    _provenance_gate_cert,
    _strict_provenance_gate_cert,
    _write_runtime_manifest,
)


def _write_strict_baseline(tmp_path: Path) -> Path:
    baseline_path = tmp_path / "trusted-baseline.json"
    baseline_path.write_text(
        json.dumps(_matching_strict_ppl_baseline()),
        encoding="utf-8",
    )
    return baseline_path


def _write_strict_policy_pack(tmp_path: Path) -> Path:
    policy_path = tmp_path / "trusted-policy-pack.json"
    policy_path.write_text(
        json.dumps(_matching_strict_policy_pack()),
        encoding="utf-8",
    )
    return policy_path


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
        verify_adapter_family,
        "warn_adapter_family_mismatch",
        lambda *args, **kwargs: None,
        raising=True,
    )

    result = CliRunner().invoke(
        app,
        ["verify", str(cert_path)],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )

    assert result.exit_code != 0
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
        verify_adapter_family,
        "warn_adapter_family_mismatch",
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


def test_strict_verify_rejects_manifest_binding_without_external_image_pin(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cert_path = tmp_path / "evaluation.report.json"
    cert_path.write_text(json.dumps(_strict_provenance_gate_cert()), encoding="utf-8")
    _write_runtime_manifest(cert_path)
    baseline_path = _write_strict_baseline(tmp_path)
    policy_path = _write_strict_policy_pack(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "verify",
            "--profile",
            "ci",
            "--assurance",
            "strict",
            "--baseline",
            str(baseline_path),
            "--policy-pack",
            str(policy_path),
            str(cert_path),
        ],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )

    assert result.exit_code != 0
    assert "independently supplied runtime image digest" in result.output


def test_strict_verify_accepts_external_runtime_image_digest_pin(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cert_path = tmp_path / "evaluation.report.json"
    cert_path.write_text(json.dumps(_strict_provenance_gate_cert()), encoding="utf-8")
    _write_runtime_manifest(cert_path)
    baseline_path = _write_strict_baseline(tmp_path)
    policy_path = _write_strict_policy_pack(tmp_path)
    monkeypatch.setattr(
        verify_mod, "_validate_evaluation_report_payload", lambda *args, **kwargs: []
    )

    result = CliRunner().invoke(
        app,
        [
            "verify",
            "--profile",
            "ci",
            "--assurance",
            "strict",
            "--expected-runtime-image-digest",
            _VALID_TEST_IMAGE_DIGEST,
            "--baseline",
            str(baseline_path),
            "--policy-pack",
            str(policy_path),
            str(cert_path),
        ],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )

    assert result.exit_code == 0
    assert "VERIFY OK" in result.output


def test_strict_verify_rejects_permissive_runtime_allowances(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cert_path = tmp_path / "evaluation.report.json"
    cert_path.write_text(json.dumps(_strict_provenance_gate_cert()), encoding="utf-8")
    manifest_path = _write_runtime_manifest(cert_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime"]["allow_remote_code"] = True
    manifest["runtime"]["allow_third_party_plugins"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    baseline_path = _write_strict_baseline(tmp_path)
    policy_path = _write_strict_policy_pack(tmp_path)
    monkeypatch.setattr(
        verify_mod, "_validate_evaluation_report_payload", lambda *args, **kwargs: []
    )

    result = CliRunner().invoke(
        app,
        [
            "verify",
            "--profile",
            "ci",
            "--assurance",
            "strict",
            "--expected-runtime-image-digest",
            _VALID_TEST_IMAGE_DIGEST,
            "--baseline",
            str(baseline_path),
            "--policy-pack",
            str(policy_path),
            str(cert_path),
        ],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )

    assert result.exit_code != 0
    assert "strict runtime forbids allow_remote_code=true" in result.output
    assert "strict runtime forbids allow_third_party_plugins=true" in result.output


def test_verify_json_reports_runtime_provenance_status(
    tmp_path: Path,
) -> None:
    cert_path = tmp_path / "evaluation.report.json"
    cert_path.write_text(json.dumps(_strict_provenance_gate_cert()), encoding="utf-8")
    _write_runtime_manifest(cert_path)
    baseline_path = _write_strict_baseline(tmp_path)
    policy_path = _write_strict_policy_pack(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "verify",
            "--profile",
            "ci",
            "--assurance",
            "strict",
            "--expected-runtime-image-digest",
            _VALID_TEST_IMAGE_DIGEST,
            "--baseline",
            str(baseline_path),
            "--policy-pack",
            str(policy_path),
            "--json",
            str(cert_path),
        ],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    verification = payload["results"][0]["verification"]
    assert (
        verification["runtime_provenance"]["status"] == "expected_image_digest_matched"
    )
    assert verification["runtime_provenance"]["verified"] is True
    assert verification["runtime_provenance"]["binding_verified"] is True
    assert verification["runtime_provenance"]["expected_digest_matched"] is True
    assert verification["runtime_provenance"]["issues"] == []
    receipt = verification["receipt"]
    dataset_provider = {"kind": "test-fixture"}
    assert receipt == {
        "format_version": "invarlock.verify-receipt.v1",
        "signed": False,
        "subject_report_sha256": hashlib.sha256(cert_path.read_bytes()).hexdigest(),
        "baseline_report_sha256": hashlib.sha256(
            baseline_path.read_bytes()
        ).hexdigest(),
        "policy_pack_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "subject_provider_digest": {
            "ids_sha256": "strict-schedule-ids",
            "tokenizer_sha256": "strict-tokenizer",
        },
        "baseline_provider_digest": {
            "ids_sha256": "strict-schedule-ids",
            "tokenizer_sha256": "strict-tokenizer",
        },
        "dataset_provider": dataset_provider,
        "dataset_provider_sha256": (
            "sha256:c86f4e23865c38f089c00c9f03d79884a489d6770c24ca0cbd02a12f09fa58bd"
        ),
        "verifier": {"package": "invarlock", "version": __version__},
        "inputs": {
            "profile": "ci",
            "assurance_mode": "strict",
            "report_assurance_mode": "strict",
            "warning_policy": "pass",
            "expected_runtime_image_digest": _VALID_TEST_IMAGE_DIGEST,
            "expected_policy_digest": json.loads(
                policy_path.read_text(encoding="utf-8")
            )["policy_digest"],
        },
        "reported_policy_digest": None,
    }


def test_report_mode_labels_self_declared_runtime_as_manifest_bound(
    tmp_path: Path,
) -> None:
    cert_path = tmp_path / "evaluation.report.json"
    cert = _provenance_gate_cert()
    cert_path.write_text(json.dumps(cert), encoding="utf-8")
    _write_runtime_manifest(cert_path)
    result = CliRunner().invoke(
        app,
        ["verify", "--assurance", "off", "--json", str(cert_path)],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    runtime = payload["results"][0]["verification"]["runtime_provenance"]
    assert runtime["status"] == "manifest_bound"
    assert runtime["binding_verified"] is True
    assert runtime["expected_digest_matched"] is False
    assert runtime["verified"] is False


def test_strict_verify_rejects_host_runtime_provenance_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cert_path = tmp_path / "evaluation.report.json"
    cert_path.write_text(json.dumps(_strict_provenance_gate_cert()), encoding="utf-8")
    baseline_path = _write_strict_baseline(tmp_path)
    policy_path = _write_strict_policy_pack(tmp_path)
    monkeypatch.setattr(
        verify_mod, "_validate_evaluation_report_payload", lambda *args, **kwargs: []
    )

    result = CliRunner().invoke(
        app,
        [
            "verify",
            "--profile",
            "ci",
            "--assurance",
            "strict",
            "--runtime-provenance",
            "host",
            "--baseline",
            str(baseline_path),
            "--policy-pack",
            str(policy_path),
            str(cert_path),
        ],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )

    assert result.exit_code == 1
    assert "independently supplied runtime image digest" in result.output


def test_strict_verify_rejects_host_runtime_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cert_path = tmp_path / "evaluation.report.json"
    cert_path.write_text(json.dumps(_strict_provenance_gate_cert()), encoding="utf-8")
    _write_runtime_manifest(cert_path, execution_mode="host")
    baseline_path = _write_strict_baseline(tmp_path)
    policy_path = _write_strict_policy_pack(tmp_path)
    monkeypatch.setattr(
        verify_mod, "_validate_evaluation_report_payload", lambda *args, **kwargs: []
    )

    result = CliRunner().invoke(
        app,
        [
            "verify",
            "--profile",
            "ci",
            "--assurance",
            "strict",
            "--baseline",
            str(baseline_path),
            "--policy-pack",
            str(policy_path),
            str(cert_path),
        ],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )

    assert result.exit_code == 1
    assert "marks evaluation.report.json as 'host'" in result.output
    assert "independently supplied runtime image digest" in result.output


def test_strict_verify_rejects_invalid_runtime_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cert_path = tmp_path / "evaluation.report.json"
    cert_path.write_text(json.dumps(_strict_provenance_gate_cert()), encoding="utf-8")
    (tmp_path / RUNTIME_MANIFEST_FILENAME).write_text("{not-json", encoding="utf-8")
    baseline_path = _write_strict_baseline(tmp_path)
    policy_path = _write_strict_policy_pack(tmp_path)
    monkeypatch.setattr(
        verify_mod, "_validate_evaluation_report_payload", lambda *args, **kwargs: []
    )

    result = CliRunner().invoke(
        app,
        [
            "verify",
            "--profile",
            "ci",
            "--assurance",
            "strict",
            "--baseline",
            str(baseline_path),
            "--policy-pack",
            str(policy_path),
            str(cert_path),
        ],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )

    assert result.exit_code == 1
    assert "runtime.manifest.json is invalid" in result.output
    assert "independently supplied runtime image digest" in result.output


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
    baseline_path = _write_strict_baseline(tmp_path)
    policy_path = _write_strict_policy_pack(tmp_path)
    monkeypatch.setattr(
        verify_mod, "_validate_evaluation_report_payload", lambda *args, **kwargs: []
    )

    result = CliRunner().invoke(
        app,
        [
            "verify",
            "--profile",
            "ci",
            "--assurance",
            "strict",
            "--baseline",
            str(baseline_path),
            "--policy-pack",
            str(policy_path),
            str(cert_path),
        ],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )

    assert result.exit_code == 1
    assert "report digest mismatch" in result.output


def test_strict_verify_rejects_runtime_image_digest_pin_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cert_path = tmp_path / "evaluation.report.json"
    cert_path.write_text(json.dumps(_strict_provenance_gate_cert()), encoding="utf-8")
    _write_runtime_manifest(cert_path)
    baseline_path = _write_strict_baseline(tmp_path)
    policy_path = _write_strict_policy_pack(tmp_path)
    monkeypatch.setattr(
        verify_mod, "_validate_evaluation_report_payload", lambda *args, **kwargs: []
    )

    result = CliRunner().invoke(
        app,
        [
            "verify",
            "--profile",
            "ci",
            "--assurance",
            "strict",
            "--expected-runtime-image-digest",
            "sha256:" + ("b" * 64),
            "--baseline",
            str(baseline_path),
            "--policy-pack",
            str(policy_path),
            str(cert_path),
        ],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )

    assert result.exit_code == 1
    assert "runtime image digest mismatch" in result.output


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

    monkeypatch.setattr(
        verify_mod,
        "_load_evaluation_report_snapshot",
        _boom,
        raising=True,
    )

    with pytest.raises(typer.Exit) as exc_info:
        verify_command([cert_path], baseline=None, profile="dev", json_out=True)

    assert exc_info.value.exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["resolution"]["exit_code"] == 2
