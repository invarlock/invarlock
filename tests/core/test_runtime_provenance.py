from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import invarlock.runtime_provenance as provenance


def test_configure_runtime_security_forwards_allowances(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    reset_token = object()
    policy = provenance.build_runtime_security_policy(
        allow_network=True,
        allow_host_execution=True,
        allow_third_party_plugins=True,
        allow_remote_code=True,
        allow_unverified_provenance=True,
    )

    def _capture(**kwargs: object) -> object:
        captured.update(kwargs)
        return reset_token

    resets: list[object] = []

    monkeypatch.setattr(
        provenance, "build_runtime_security_policy", lambda **kwargs: policy
    )
    monkeypatch.setattr(provenance, "apply_runtime_allowances", _capture)
    monkeypatch.setattr(
        provenance, "reset_runtime_allowances", lambda token: resets.append(token)
    )

    with provenance.configure_runtime_security(
        allow_network=True,
        allow_host_execution=True,
        allow_third_party_plugins=True,
        allow_remote_code=True,
        allow_unverified_provenance=True,
    ):
        assert captured == {"policy": policy}

    assert resets == [reset_token]


def test_verify_runtime_provenance_short_circuits_when_unverified_provenance_allowed(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(provenance, "unverified_provenance_allowed", lambda: False)
    result = provenance.verify_runtime_provenance(
        tmp_path / "report.json", allow_unverified=True
    )
    assert result.skipped is True
    assert result.issues == ()

    monkeypatch.setattr(provenance, "unverified_provenance_allowed", lambda: True)
    result = provenance.verify_runtime_provenance(tmp_path / "report.json")
    assert result.skipped is True
    assert result.issues == ()


def test_verify_runtime_provenance_handles_missing_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    report = tmp_path / "report.json"
    manifest = tmp_path / "runtime.manifest.json"

    monkeypatch.setattr(provenance, "unverified_provenance_allowed", lambda: False)
    monkeypatch.setattr(
        provenance,
        "load_runtime_manifest",
        lambda path: provenance.RuntimeManifestLoadResult(
            path=manifest,
            payload=None,
            issue_code=provenance.RuntimeManifestLoadIssueCode.MISSING,
        ),
    )

    result = provenance.verify_runtime_provenance(report)
    assert (
        result.issues[0].code == provenance.RuntimeProvenanceIssueCode.MANIFEST_MISSING
    )
    assert [issue.message for issue in result.issues] == [
        "runtime.manifest.json missing for report.json."
    ]


def test_verify_runtime_provenance_rejects_non_container_execution_mode(
    monkeypatch, tmp_path: Path
) -> None:
    report = tmp_path / "report.json"
    manifest = tmp_path / "runtime.manifest.json"

    monkeypatch.setattr(provenance, "unverified_provenance_allowed", lambda: False)
    monkeypatch.setattr(
        provenance,
        "load_runtime_manifest",
        lambda path: provenance.RuntimeManifestLoadResult(
            path=manifest,
            payload={"execution_mode": "host"},
        ),
    )

    result = provenance.verify_runtime_provenance(report)
    assert [issue.message for issue in result.issues] == [
        "runtime.manifest.json marks report.json as 'host'."
    ]


def test_verify_runtime_provenance_uses_python_runtime_verifier(
    monkeypatch, tmp_path: Path
) -> None:
    report = tmp_path / "report.json"
    manifest = tmp_path / "runtime.manifest.json"

    monkeypatch.setattr(provenance, "unverified_provenance_allowed", lambda: False)
    monkeypatch.setattr(
        provenance,
        "load_runtime_manifest",
        lambda path: provenance.RuntimeManifestLoadResult(
            path=manifest,
            payload={"execution_mode": "container"},
        ),
    )
    monkeypatch.setattr(
        provenance,
        "verify_runtime_manifest",
        lambda report_path, manifest_path: SimpleNamespace(
            ok=True,
            errors=(),
            report=str(report_path),
            manifest=str(manifest_path),
        ),
    )

    result = provenance.verify_runtime_provenance(report)
    assert result.verified is True
    assert result.issues == ()


def test_verify_runtime_provenance_reports_python_verifier_failures(
    monkeypatch, tmp_path: Path
) -> None:
    report = tmp_path / "report.json"
    manifest = tmp_path / "runtime.manifest.json"

    monkeypatch.setattr(provenance, "unverified_provenance_allowed", lambda: False)
    monkeypatch.setattr(
        provenance,
        "load_runtime_manifest",
        lambda path: provenance.RuntimeManifestLoadResult(
            path=manifest,
            payload={"execution_mode": "container"},
        ),
    )
    monkeypatch.setattr(
        provenance,
        "verify_runtime_manifest",
        lambda report_path, manifest_path: SimpleNamespace(
            ok=False,
            errors=("hash mismatch", "digest missing"),
            report=str(report_path),
            manifest=str(manifest_path),
        ),
    )

    result = provenance.verify_runtime_provenance(report)
    assert result.verified is False
    assert [issue.message for issue in result.issues] == [
        "hash mismatch",
        "digest missing",
    ]


def test_verify_runtime_provenance_distinguishes_invalid_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    report = tmp_path / "report.json"
    manifest = tmp_path / "runtime.manifest.json"

    monkeypatch.setattr(provenance, "unverified_provenance_allowed", lambda: False)
    monkeypatch.setattr(
        provenance,
        "load_runtime_manifest",
        lambda path: provenance.RuntimeManifestLoadResult(
            path=manifest,
            payload=None,
            issue_code=provenance.RuntimeManifestLoadIssueCode.INVALID_JSON,
            issue_message="runtime.manifest.json is not valid JSON",
        ),
    )

    result = provenance.verify_runtime_provenance(report)

    assert result.verified is False
    assert (
        result.issues[0].code == provenance.RuntimeProvenanceIssueCode.MANIFEST_INVALID
    )
    assert result.issues[0].message == (
        "runtime.manifest.json is invalid for report.json: "
        "runtime.manifest.json is not valid JSON."
    )
