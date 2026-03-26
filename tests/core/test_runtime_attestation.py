from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import invarlock.core.runtime_attestation as attestation


def test_configure_runtime_security_forwards_allowances(
    monkeypatch,
) -> None:
    captured: dict[str, bool] = {}

    def _capture(**kwargs: bool) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(attestation, "apply_runtime_allowances", _capture)

    attestation.configure_runtime_security(
        allow_network=True,
        allow_host_execution=True,
        allow_third_party_plugins=True,
        allow_remote_code=True,
        allow_unattested_artifacts=True,
    )

    assert captured == {
        "allow_network": True,
        "allow_host_execution": True,
        "allow_third_party_plugins": True,
        "allow_remote_code": True,
        "allow_unattested_artifacts": True,
    }


def test_verify_runtime_attestation_short_circuits_when_unattested_allowed(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(attestation, "unattested_artifacts_allowed", lambda: False)
    assert (
        attestation.verify_runtime_attestation(
            tmp_path / "report.json", allow_unattested=True
        )
        == []
    )

    monkeypatch.setattr(attestation, "unattested_artifacts_allowed", lambda: True)
    assert attestation.verify_runtime_attestation(tmp_path / "report.json") == []


def test_verify_runtime_attestation_handles_missing_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    report = tmp_path / "report.json"
    manifest = tmp_path / "runtime.manifest.json"

    monkeypatch.setattr(attestation, "unattested_artifacts_allowed", lambda: False)
    monkeypatch.setattr(
        attestation,
        "load_runtime_manifest",
        lambda path: (manifest, None),
    )

    errors = attestation.verify_runtime_attestation(report)
    assert errors == [
        "runtime.manifest.json missing or unreadable for report.json; pass --allow-unattested-artifacts to override."
    ]


def test_verify_runtime_attestation_rejects_non_container_execution_mode(
    monkeypatch, tmp_path: Path
) -> None:
    report = tmp_path / "report.json"
    manifest = tmp_path / "runtime.manifest.json"

    monkeypatch.setattr(attestation, "unattested_artifacts_allowed", lambda: False)
    monkeypatch.setattr(
        attestation,
        "load_runtime_manifest",
        lambda path: (manifest, {"execution_mode": "host"}),
    )

    errors = attestation.verify_runtime_attestation(report)
    assert errors == [
        "runtime.manifest.json marks report.json as 'host'; pass --allow-unattested-artifacts to override."
    ]


def test_verify_runtime_attestation_requires_verifier_binary(
    monkeypatch, tmp_path: Path
) -> None:
    report = tmp_path / "report.json"
    manifest = tmp_path / "runtime.manifest.json"

    monkeypatch.setattr(attestation, "unattested_artifacts_allowed", lambda: False)
    monkeypatch.setattr(
        attestation,
        "load_runtime_manifest",
        lambda path: (manifest, {"execution_mode": "container"}),
    )
    monkeypatch.setattr(attestation, "runtime_verifier_binary", lambda: "verify-bin")
    monkeypatch.setattr(attestation.shutil, "which", lambda binary: None)

    errors = attestation.verify_runtime_attestation(report)
    assert errors == [
        "Runtime verifier 'verify-bin' is not installed; cannot verify report.json."
    ]


def test_verify_runtime_attestation_handles_subprocess_outcomes(
    monkeypatch, tmp_path: Path
) -> None:
    report = tmp_path / "report.json"
    manifest = tmp_path / "runtime.manifest.json"

    monkeypatch.setattr(attestation, "unattested_artifacts_allowed", lambda: False)
    monkeypatch.setattr(
        attestation,
        "load_runtime_manifest",
        lambda path: (manifest, {"execution_mode": "container"}),
    )
    monkeypatch.setattr(attestation, "runtime_verifier_binary", lambda: "verify-bin")
    monkeypatch.setattr(attestation.shutil, "which", lambda binary: "/usr/bin/verify")

    monkeypatch.setattr(
        attestation.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    assert attestation.verify_runtime_attestation(report) == []

    monkeypatch.setattr(
        attestation.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout='{"errors": ["hash mismatch", "digest missing"]}',
            stderr="",
        ),
    )
    assert attestation.verify_runtime_attestation(report) == [
        "hash mismatch",
        "digest missing",
    ]

    monkeypatch.setattr(
        attestation.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="not-json",
            stderr="",
        ),
    )
    assert attestation.verify_runtime_attestation(report) == ["not-json"]

    monkeypatch.setattr(
        attestation.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr=""),
    )
    assert attestation.verify_runtime_attestation(report) == [
        "Runtime verifier failed for report.json."
    ]
