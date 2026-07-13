from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import invarlock.runtime_provenance as provenance
from invarlock.runtime_security import (
    RUNTIME_MANIFEST_VERSION,
    RUNTIME_VERIFIER_CONTRACT_VERSION,
)

_TEST_IMAGE_DIGEST = "sha256:" + ("a" * 64)


def _runtime_manifest_payload(report: Path, *, report_sha256: str) -> dict:
    return {
        "manifest_version": RUNTIME_MANIFEST_VERSION,
        "generated_at_utc": "2026-07-09T00:00:00+00:00",
        "verifier_contract_version": RUNTIME_VERIFIER_CONTRACT_VERSION,
        "report": {
            "path": str(report.resolve()),
            "filename": report.name,
            "sha256": report_sha256,
        },
        "config": {"path": None, "sha256": None, "source": "missing"},
        "execution_mode": "container",
        "runtime": {
            "image_ref": "ghcr.io/invarlock/runtime:test",
            "image_digest": _TEST_IMAGE_DIGEST,
            "container_execution": True,
            "allow_network": False,
            "allow_remote_code": False,
            "allow_third_party_plugins": False,
        },
    }


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
    report.write_text("{}", encoding="utf-8")
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
        "verify_runtime_manifest_snapshot",
        lambda report_bytes, manifest_payload, *, report, manifest, **kwargs: (
            SimpleNamespace(
                ok=True,
                errors=(),
                report=str(report),
                manifest=str(manifest),
            )
        ),
    )

    result = provenance.verify_runtime_provenance(report)
    assert result.verified is False
    assert result.binding_verified is True
    assert result.expected_digest_matched is False
    assert result.trust_status == "manifest_bound"
    assert result.issues == ()


def test_verify_runtime_provenance_requires_external_digest_for_independent_trust(
    monkeypatch, tmp_path: Path
) -> None:
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "runtime.manifest.json"
    expected = "sha256:" + ("a" * 64)

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
        "verify_runtime_manifest_snapshot",
        lambda report_bytes, manifest_payload, *, report, manifest, **kwargs: (
            SimpleNamespace(
                ok=True,
                errors=(),
                report=str(report),
                manifest=str(manifest),
                binding_verified=True,
                expected_digest_matched=True,
                trust_status="expected_image_digest_matched",
                declared_image_digest=expected,
            )
        ),
    )

    result = provenance.verify_runtime_provenance(
        report, expected_image_digest=expected
    )

    assert result.verified is True
    assert result.binding_verified is True
    assert result.expected_digest_matched is True
    assert result.trust_status == "expected_image_digest_matched"


def test_verify_runtime_provenance_reports_python_verifier_failures(
    monkeypatch, tmp_path: Path
) -> None:
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
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
        "verify_runtime_manifest_snapshot",
        lambda report_bytes, manifest_payload, *, report, manifest, **kwargs: (
            SimpleNamespace(
                ok=False,
                errors=("hash mismatch", "digest missing"),
                report=str(report),
                manifest=str(manifest),
            )
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


def test_verify_runtime_provenance_manifest_swap_cannot_repair_loaded_snapshot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    report = tmp_path / "evaluation.report.json"
    report_bytes = b'{"schema_version":"v1"}\n'
    report.write_bytes(report_bytes)
    manifest_path = tmp_path / "runtime.manifest.json"
    valid_manifest = _runtime_manifest_payload(
        report,
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
    )
    invalid_manifest = _runtime_manifest_payload(
        report,
        report_sha256="0" * 64,
    )
    manifest_path.write_text(json.dumps(invalid_manifest), encoding="utf-8")
    original_load = provenance.load_runtime_manifest

    def _load_then_repair(path: Path):
        snapshot = original_load(path)
        manifest_path.write_text(json.dumps(valid_manifest), encoding="utf-8")
        return snapshot

    monkeypatch.setattr(provenance, "load_runtime_manifest", _load_then_repair)

    result = provenance.verify_runtime_provenance(
        report,
        expected_image_digest=_TEST_IMAGE_DIGEST,
        report_bytes=report_bytes,
    )

    assert result.verified is False
    assert result.binding_verified is False
    assert any("report digest mismatch" in issue.message for issue in result.issues)
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == valid_manifest
