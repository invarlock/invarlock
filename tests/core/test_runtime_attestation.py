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

    monkeypatch.setattr(
        attestation, "build_runtime_security_policy", lambda **kwargs: policy
    )
    monkeypatch.setattr(attestation, "apply_runtime_allowances", _capture)
    monkeypatch.setattr(
        attestation, "reset_runtime_allowances", lambda token: resets.append(token)
    )

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
        lambda path: attestation.RuntimeManifestLoadResult(
            path=manifest,
            payload=None,
            issue_code=attestation.RuntimeManifestLoadIssueCode.MISSING,
        ),
    )

    result = attestation.verify_runtime_attestation(report)
    assert (
        result.issues[0].code
        == attestation.RuntimeAttestationIssueCode.MANIFEST_MISSING
    )
    assert [issue.message for issue in result.issues] == [
        "runtime.manifest.json missing for report.json."
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
        lambda path: attestation.RuntimeManifestLoadResult(
            path=manifest,
            payload={"execution_mode": "host"},
        ),
    )

    result = attestation.verify_runtime_attestation(report)
    assert [issue.message for issue in result.issues] == [
        "runtime.manifest.json marks report.json as 'host'."
    ]


def test_verify_runtime_attestation_uses_python_runtime_verifier(
    monkeypatch, tmp_path: Path
) -> None:
    report = tmp_path / "report.json"
    manifest = tmp_path / "runtime.manifest.json"

    monkeypatch.setattr(attestation, "unattested_artifacts_allowed", lambda: False)
    monkeypatch.setattr(
        attestation,
        "load_runtime_manifest",
        lambda path: attestation.RuntimeManifestLoadResult(
            path=manifest,
            payload={"execution_mode": "container"},
        ),
    )
    monkeypatch.setattr(
        attestation,
        "verify_runtime_manifest",
        lambda report_path, manifest_path: SimpleNamespace(
            ok=True,
            errors=(),
            report=str(report_path),
            manifest=str(manifest_path),
        ),
    )

    result = attestation.verify_runtime_attestation(report)
    assert result.verified is True
    assert result.issues == ()


def test_verify_runtime_attestation_reports_python_verifier_failures(
    monkeypatch, tmp_path: Path
) -> None:
    report = tmp_path / "report.json"
    manifest = tmp_path / "runtime.manifest.json"

    monkeypatch.setattr(attestation, "unattested_artifacts_allowed", lambda: False)
    monkeypatch.setattr(
        attestation,
        "load_runtime_manifest",
        lambda path: attestation.RuntimeManifestLoadResult(
            path=manifest,
            payload={"execution_mode": "container"},
        ),
    )
    monkeypatch.setattr(
        attestation,
        "verify_runtime_manifest",
        lambda report_path, manifest_path: SimpleNamespace(
            ok=False,
            errors=("hash mismatch", "digest missing"),
            report=str(report_path),
            manifest=str(manifest_path),
        ),
    )

    result = attestation.verify_runtime_attestation(report)
    assert result.verified is False
    assert [issue.message for issue in result.issues] == [
        "hash mismatch",
        "digest missing",
    ]


def test_verify_runtime_attestation_distinguishes_invalid_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    report = tmp_path / "report.json"
    manifest = tmp_path / "runtime.manifest.json"

    monkeypatch.setattr(attestation, "unattested_artifacts_allowed", lambda: False)
    monkeypatch.setattr(
        attestation,
        "load_runtime_manifest",
        lambda path: attestation.RuntimeManifestLoadResult(
            path=manifest,
            payload=None,
            issue_code=attestation.RuntimeManifestLoadIssueCode.INVALID_JSON,
            issue_message="runtime.manifest.json is not valid JSON",
        ),
    )

    result = attestation.verify_runtime_attestation(report)

    assert result.verified is False
    assert (
        result.issues[0].code
        == attestation.RuntimeAttestationIssueCode.MANIFEST_INVALID
    )
    assert result.issues[0].message == (
        "runtime.manifest.json is invalid for report.json: "
        "runtime.manifest.json is not valid JSON."
    )
