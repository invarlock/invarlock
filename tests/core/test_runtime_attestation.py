from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import invarlock.runtime_attestation as attestation


def test_configure_runtime_security_forwards_allowances(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    reset_token = object()
    policy = attestation.build_runtime_security_policy(
        allow_network=True,
        allow_host_execution=True,
        allow_third_party_plugins=True,
        allow_remote_code=True,
        allow_unattested_artifacts=True,
    )

    def _capture(**kwargs: object) -> object:
        captured.update(kwargs)
        return reset_token

    resets: list[object] = []

    monkeypatch.setattr(attestation, "build_runtime_security_policy", lambda **kwargs: policy)
    monkeypatch.setattr(attestation, "apply_runtime_allowances", _capture)
    monkeypatch.setattr(attestation, "reset_runtime_allowances", lambda token: resets.append(token))

    with attestation.configure_runtime_security(
        allow_network=True,
        allow_host_execution=True,
        allow_third_party_plugins=True,
        allow_remote_code=True,
        allow_unattested_artifacts=True,
    ):
        assert captured == {"policy": policy}

    assert resets == [reset_token]


def test_verify_runtime_attestation_short_circuits_when_unattested_allowed(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(attestation, "unattested_artifacts_allowed", lambda: False)
    result = attestation.verify_runtime_attestation(
        tmp_path / "report.json", allow_unattested=True
    )
    assert result.skipped is True
    assert result.issues == ()

    monkeypatch.setattr(attestation, "unattested_artifacts_allowed", lambda: True)
    result = attestation.verify_runtime_attestation(tmp_path / "report.json")
    assert result.skipped is True
    assert result.issues == ()


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

    result = attestation.verify_runtime_attestation(report)
    assert [issue.message for issue in result.issues] == [
        "runtime.manifest.json missing or unreadable for report.json."
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

    result = attestation.verify_runtime_attestation(report)
    assert [issue.message for issue in result.issues] == [
        "runtime.manifest.json marks report.json as 'host'."
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

    result = attestation.verify_runtime_attestation(report)
    assert [issue.message for issue in result.issues] == [
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
    result = attestation.verify_runtime_attestation(report)
    assert result.verified is True
    assert result.issues == ()

    monkeypatch.setattr(
        attestation.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout='{"errors": ["hash mismatch", "digest missing"]}',
            stderr="",
        ),
    )
    result = attestation.verify_runtime_attestation(report)
    assert [issue.message for issue in result.issues] == [
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
    result = attestation.verify_runtime_attestation(report)
    assert [issue.message for issue in result.issues] == ["not-json"]

    monkeypatch.setattr(
        attestation.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr=""),
    )
    result = attestation.verify_runtime_attestation(report)
    assert [issue.message for issue in result.issues] == [
        "Runtime verifier failed for report.json."
    ]


def test_verify_runtime_attestation_handles_timeout(
    monkeypatch, tmp_path: Path
) -> None:
    report = tmp_path / "report.json"
    manifest = tmp_path / "runtime.manifest.json"
    seen: dict[str, object] = {}

    monkeypatch.setattr(attestation, "unattested_artifacts_allowed", lambda: False)
    monkeypatch.setattr(
        attestation,
        "load_runtime_manifest",
        lambda path: (manifest, {"execution_mode": "container"}),
    )
    monkeypatch.setattr(attestation, "runtime_verifier_binary", lambda: "verify-bin")
    monkeypatch.setattr(attestation.shutil, "which", lambda binary: "/usr/bin/verify")

    def _run(command, capture_output=False, text=False, check=False, timeout=None):
        seen["timeout"] = timeout
        raise attestation.subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(attestation.subprocess, "run", _run)

    result = attestation.verify_runtime_attestation(report)

    assert result.verified is False
    assert [issue.message for issue in result.issues] == [
        "Runtime verifier timed out for report.json."
    ]
    assert seen["timeout"] == attestation._RUNTIME_VERIFIER_TIMEOUT_SECONDS
