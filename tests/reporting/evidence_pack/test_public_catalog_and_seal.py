from __future__ import annotations

import json
from pathlib import Path

import pytest

import invarlock.evidence_pack as evidence_pack_mod
from invarlock.evidence_catalog import (
    EvidenceCatalogError,
    catalog_digest,
    load_evidence_catalog,
)
from invarlock.evidence_catalog_contracts.set_verifier import (
    verify_evidence_pack_set,
)
from invarlock.evidence_pack import verify_evidence_pack
from invarlock.evidence_pack_support import EvidencePackResult, EvidencePackStatus
from invarlock.reporting.verify_contract import VerifyExecutionResult, VerifyOutcome
from tests._support_evidence_pack_signing import sign_manifest
from tests.reporting.evidence_pack._support_catalog_evidence import (
    IMAGE_DIGEST,
    SIGNER_FINGERPRINT,
    SOURCE_BUNDLE_DIGEST,
    SOURCE_COMMIT,
    assemble_signed_catalog_pack,
    catalog_entry,
    write_catalog_evidence_fixture,
    write_json,
)


def _write_set_catalog(tmp_path: Path) -> Path:
    preset_digest = "sha256:" + ("a" * 64)
    entries = [
        catalog_entry(
            lane_id=lane_id,
            model_id=f"org/{lane_id}",
            preset_digest=preset_digest,
        )
        for lane_id in ("text-a", "text-b")
    ]
    return write_json(
        tmp_path / "catalog.json",
        {
            "format_version": "invarlock/evidence-catalog-v1",
            "entry_count": len(entries),
            "entries": entries,
        },
    )


def _write_pack(path: Path, binding: dict[str, object], catalog_path: Path) -> Path:
    path.mkdir()
    write_json(path / "manifest.json", {"catalog": binding})
    (path / "metadata").mkdir()
    (path / "metadata" / "catalog.json").write_bytes(catalog_path.read_bytes())
    write_json(
        path / "metadata" / "source_repo.json",
        {
            "format_version": "invarlock/source-provenance-v1",
            "commit": SOURCE_COMMIT,
            "source_bundle_sha256": SOURCE_BUNDLE_DIGEST,
            "dirty": False,
        },
    )
    write_json(path / "reports" / "one" / "evaluation.report.json", {"ok": True})
    write_json(
        path / "reports" / "one" / "runtime.manifest.json",
        {"runtime": {"image_digest": IMAGE_DIGEST}},
    )
    (path / "checksums.sha256").write_text(
        "0" * 64 + "  placeholder\n", encoding="utf-8"
    )
    return path


def _pinned_verification(*_args, **_kwargs) -> EvidencePackResult:
    return EvidencePackResult(
        payload={
            "ok": True,
            "authenticity": "pinned",
            "signer_fingerprint": SIGNER_FINGERPRINT,
        },
        status=EvidencePackStatus.OK,
    )


def _fake_report_verification(*_args, **_kwargs) -> VerifyExecutionResult:
    return VerifyExecutionResult(
        outcome=VerifyOutcome.OK,
        payload={"ok": True},
        diagnostics=(),
    )


def _seal_fixture(fixture, out_dir: Path) -> EvidencePackResult:
    _pack, fingerprint = assemble_signed_catalog_pack(fixture, out_dir)
    return EvidencePackResult(
        payload={
            "ok": True,
            "signature": {"signer_fingerprint": fingerprint},
        },
        status=EvidencePackStatus.OK,
    )


def test_verify_set_requires_exact_coverage_and_records_pinned_signers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    catalog_path = _write_set_catalog(tmp_path)
    catalog = load_evidence_catalog(catalog_path)
    pack_a = _write_pack(
        tmp_path / "pack-a",
        catalog.binding_for("text-a", path="metadata/catalog.json"),
        catalog_path,
    )
    pack_b = _write_pack(
        tmp_path / "pack-b",
        catalog.binding_for("text-b", path="metadata/catalog.json"),
        catalog_path,
    )
    receipt = tmp_path / "set-receipt.json"
    monkeypatch.setattr(
        "invarlock.evidence_catalog_contracts.set_verifier.verify_evidence_pack",
        _pinned_verification,
    )

    result = verify_evidence_pack_set(
        catalog_path=catalog_path,
        pack_dirs=[pack_b, pack_a],
        receipt_path=receipt,
        expected_catalog_digest=catalog_digest(catalog_path),
        expected_source_commit=SOURCE_COMMIT,
        expected_source_bundle_digest=SOURCE_BUNDLE_DIGEST,
        expected_runtime_image_digest=IMAGE_DIGEST,
        expected_fingerprint=SIGNER_FINGERPRINT,
    )

    assert result.status is EvidencePackStatus.OK
    assert result.payload["catalog_digest"] == catalog_digest(catalog_path)
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert [item["entry_id"] for item in receipt_payload["packs"]] == [
        "text-a",
        "text-b",
    ]
    assert {item["authenticity"] for item in receipt_payload["packs"]} == {"pinned"}
    assert {item["signer_fingerprint"] for item in receipt_payload["packs"]} == {
        SIGNER_FINGERPRINT
    }
    assert receipt_payload["source_commit"] == SOURCE_COMMIT
    assert receipt_payload["source_bundle_digest"] == SOURCE_BUNDLE_DIGEST
    assert receipt_payload["runtime_image_digest"] == IMAGE_DIGEST
    assert {item["source_commit"] for item in receipt_payload["packs"]} == {
        SOURCE_COMMIT
    }


def test_verify_set_rejects_same_signer_evidence_from_another_source_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    catalog_path = _write_set_catalog(tmp_path)
    catalog = load_evidence_catalog(catalog_path)
    pack = _write_pack(
        tmp_path / "pack",
        catalog.binding_for("text-a", path="metadata/catalog.json"),
        catalog_path,
    )
    source_path = pack / "metadata" / "source_repo.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["commit"] = "1" * 40
    write_json(source_path, source)
    monkeypatch.setattr(
        "invarlock.evidence_catalog_contracts.set_verifier.verify_evidence_pack",
        _pinned_verification,
    )

    result = verify_evidence_pack_set(
        catalog_path=catalog_path,
        pack_dirs=[pack],
        receipt_path=tmp_path / "receipt.json",
        expected_catalog_digest=catalog_digest(catalog_path),
        expected_source_commit=SOURCE_COMMIT,
        expected_source_bundle_digest=SOURCE_BUNDLE_DIGEST,
        expected_runtime_image_digest=IMAGE_DIGEST,
        expected_fingerprint=SIGNER_FINGERPRINT,
    )

    assert result.status is EvidencePackStatus.FORMAT
    assert result.payload["packs"][0]["source_commit"] == "1" * 40
    assert result.payload["packs"][0]["ok"] is False
    assert "source_commit_mismatch" in result.payload["packs"][0]["errors"]


def test_verify_set_uses_one_snapshot_for_provenance_and_strict_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    catalog_path = _write_set_catalog(tmp_path)
    catalog = load_evidence_catalog(catalog_path)
    pack = _write_pack(
        tmp_path / "pack",
        catalog.binding_for("text-a", path="metadata/catalog.json"),
        catalog_path,
    )
    source_path = pack / "metadata" / "source_repo.json"

    def swap_source_after_snapshot(
        snapshotted_pack: Path, *_args: object, **_kwargs: object
    ) -> EvidencePackResult:
        assert snapshotted_pack != pack
        assert (
            json.loads(
                (snapshotted_pack / "metadata" / "source_repo.json").read_text(
                    encoding="utf-8"
                )
            )["commit"]
            == SOURCE_COMMIT
        )
        replacement = tmp_path / "replacement-source.json"
        write_json(
            replacement,
            {
                "format_version": "invarlock/source-provenance-v1",
                "commit": "1" * 40,
                "source_bundle_sha256": SOURCE_BUNDLE_DIGEST,
                "dirty": False,
            },
        )
        replacement.replace(source_path)
        return _pinned_verification()

    monkeypatch.setattr(
        "invarlock.evidence_catalog_contracts.set_verifier.verify_evidence_pack",
        swap_source_after_snapshot,
    )

    result = verify_evidence_pack_set(
        catalog_path=catalog_path,
        pack_dirs=[pack],
        receipt_path=tmp_path / "receipt.json",
        expected_catalog_digest=catalog_digest(catalog_path),
        expected_source_commit=SOURCE_COMMIT,
        expected_source_bundle_digest=SOURCE_BUNDLE_DIGEST,
        expected_runtime_image_digest=IMAGE_DIGEST,
        expected_fingerprint=SIGNER_FINGERPRINT,
    )

    pack_result = result.payload["packs"][0]
    assert pack_result["source_commit"] == SOURCE_COMMIT
    assert pack_result["ok"] is False
    assert "pack_provenance_invalid" in pack_result["errors"]


def test_catalog_rejects_duplicate_json_keys_and_closed_schema_fields(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"format_version":"invarlock/evidence-catalog-v1",'
        '"entry_count":0,"entry_count":0,"entries":[]}',
        encoding="utf-8",
    )
    with pytest.raises(EvidenceCatalogError, match="duplicate"):
        load_evidence_catalog(duplicate)

    catalog_path = _write_set_catalog(tmp_path)
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    payload["operational_placement"] = {"host": "not-allowed"}
    write_json(catalog_path, payload)
    with pytest.raises(EvidenceCatalogError, match="unsupported field"):
        load_evidence_catalog(catalog_path)


def test_verify_set_rejects_duplicate_and_missing_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    catalog_path = _write_set_catalog(tmp_path)
    catalog = load_evidence_catalog(catalog_path)
    pack_a = _write_pack(
        tmp_path / "pack-a",
        catalog.binding_for("text-a", path="metadata/catalog.json"),
        catalog_path,
    )
    pack_a_again = _write_pack(
        tmp_path / "pack-a-again",
        catalog.binding_for("text-a", path="metadata/catalog.json"),
        catalog_path,
    )
    monkeypatch.setattr(
        "invarlock.evidence_catalog_contracts.set_verifier.verify_evidence_pack",
        _pinned_verification,
    )

    result = verify_evidence_pack_set(
        catalog_path=catalog_path,
        pack_dirs=[pack_a, pack_a_again],
        receipt_path=tmp_path / "receipt.json",
        expected_catalog_digest=catalog_digest(catalog_path),
        expected_source_commit=SOURCE_COMMIT,
        expected_source_bundle_digest=SOURCE_BUNDLE_DIGEST,
        expected_runtime_image_digest=IMAGE_DIGEST,
        expected_fingerprint=SIGNER_FINGERPRINT,
    )

    assert result.status is EvidencePackStatus.FORMAT
    assert result.payload["missing_entry_ids"] == ["text-b"]
    assert result.payload["duplicate_entry_ids"] == ["text-a"]


def test_verify_set_receipt_never_copies_nested_error_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    catalog_path = _write_set_catalog(tmp_path)
    catalog = load_evidence_catalog(catalog_path)
    pack = _write_pack(
        tmp_path / "pack",
        catalog.binding_for("text-a", path="metadata/catalog.json"),
        catalog_path,
    )
    receipt = tmp_path / "receipt.json"

    def failed(*_args, **_kwargs) -> EvidencePackResult:
        return EvidencePackResult(
            payload={
                "ok": False,
                "errors": [f"input missing at {tmp_path}"],
                "authenticity": "mismatch",
            },
            status=EvidencePackStatus.FORMAT,
        )

    monkeypatch.setattr(
        "invarlock.evidence_catalog_contracts.set_verifier.verify_evidence_pack",
        failed,
    )
    result = verify_evidence_pack_set(
        catalog_path=catalog_path,
        pack_dirs=[pack],
        receipt_path=receipt,
        expected_catalog_digest=catalog_digest(catalog_path),
        expected_source_commit=SOURCE_COMMIT,
        expected_source_bundle_digest=SOURCE_BUNDLE_DIGEST,
        expected_runtime_image_digest=IMAGE_DIGEST,
        expected_fingerprint=SIGNER_FINGERPRINT,
    )

    assert result.status is EvidencePackStatus.FORMAT
    assert str(tmp_path) not in receipt.read_text(encoding="utf-8")


def test_verify_set_requires_an_independent_trust_anchor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    catalog_path = _write_set_catalog(tmp_path)
    catalog = load_evidence_catalog(catalog_path)
    pack = _write_pack(
        tmp_path / "pack",
        catalog.binding_for("text-a", path="metadata/catalog.json"),
        catalog_path,
    )
    monkeypatch.setattr(
        "invarlock.evidence_catalog_contracts.set_verifier.verify_evidence_pack",
        lambda *_args, **_kwargs: pytest.fail("nested verification must not run"),
    )

    result = verify_evidence_pack_set(
        catalog_path=catalog_path,
        pack_dirs=[pack],
        receipt_path=tmp_path / "receipt.json",
        expected_catalog_digest=catalog_digest(catalog_path),
        expected_source_commit=SOURCE_COMMIT,
        expected_source_bundle_digest=SOURCE_BUNDLE_DIGEST,
        expected_runtime_image_digest=IMAGE_DIGEST,
    )

    assert result.status is EvidencePackStatus.USAGE
    assert result.payload["errors"] == ["independent_trust_anchor_required"]


def test_verify_set_requires_a_well_formed_independent_catalog_anchor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    catalog_path = _write_set_catalog(tmp_path)
    monkeypatch.setattr(
        "invarlock.evidence_catalog_contracts.set_verifier.verify_evidence_pack",
        lambda *_args, **_kwargs: pytest.fail("nested verification must not run"),
    )

    result = verify_evidence_pack_set(
        catalog_path=catalog_path,
        pack_dirs=[],
        receipt_path=tmp_path / "receipt.json",
        expected_catalog_digest="",
        expected_source_commit=SOURCE_COMMIT,
        expected_source_bundle_digest=SOURCE_BUNDLE_DIGEST,
        expected_runtime_image_digest=IMAGE_DIGEST,
        expected_fingerprint=SIGNER_FINGERPRINT,
    )

    assert result.status is EvidencePackStatus.USAGE
    assert result.payload["errors"] == ["independent_catalog_anchor_required"]


def test_verify_set_rejects_a_different_independent_catalog_anchor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    catalog_path = _write_set_catalog(tmp_path)
    monkeypatch.setattr(
        "invarlock.evidence_catalog_contracts.set_verifier.verify_evidence_pack",
        lambda *_args, **_kwargs: pytest.fail("nested verification must not run"),
    )

    result = verify_evidence_pack_set(
        catalog_path=catalog_path,
        pack_dirs=[],
        receipt_path=tmp_path / "receipt.json",
        expected_catalog_digest="sha256:" + ("f" * 64),
        expected_source_commit=SOURCE_COMMIT,
        expected_source_bundle_digest=SOURCE_BUNDLE_DIGEST,
        expected_runtime_image_digest=IMAGE_DIGEST,
        expected_fingerprint=SIGNER_FINGERPRINT,
    )

    assert result.status is EvidencePackStatus.FORMAT
    assert result.payload["errors"] == ["catalog_digest_mismatch"]


def test_verify_set_rejects_receipts_inside_packs_without_writing(
    tmp_path: Path,
) -> None:
    catalog_path = _write_set_catalog(tmp_path)
    catalog = load_evidence_catalog(catalog_path)
    pack = _write_pack(
        tmp_path / "pack",
        catalog.binding_for("text-a", path="metadata/catalog.json"),
        catalog_path,
    )
    receipt = pack / "receipt.json"

    with pytest.raises(EvidenceCatalogError, match="outside every pack"):
        verify_evidence_pack_set(
            catalog_path=catalog_path,
            pack_dirs=[pack],
            receipt_path=receipt,
            expected_catalog_digest=catalog_digest(catalog_path),
            expected_source_commit=SOURCE_COMMIT,
            expected_source_bundle_digest=SOURCE_BUNDLE_DIGEST,
            expected_runtime_image_digest=IMAGE_DIGEST,
            expected_fingerprint=SIGNER_FINGERPRINT,
        )
    assert not receipt.exists()


def test_direct_verification_derives_catalog_profile_and_rejects_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = write_catalog_evidence_fixture(tmp_path)
    monkeypatch.setattr(
        evidence_pack_mod, "_run_verify_command", _fake_report_verification
    )
    pack = tmp_path / "pack"
    sealed = _seal_fixture(fixture, pack)
    assert sealed.status is EvidencePackStatus.OK, sealed.payload
    fingerprint = sealed.payload["signature"]["signer_fingerprint"]
    verify_kwargs = {
        "strict": True,
        "report_assurance": "strict",
        "expected_fingerprint": fingerprint,
        "expected_catalog_digest": catalog_digest(fixture.catalog),
        "expected_runtime_image_digest": IMAGE_DIGEST,
        "policy_pack_path": fixture.policy_pack,
    }

    derived = verify_evidence_pack(pack, **verify_kwargs)
    assert derived.status is EvidencePackStatus.OK, derived.payload

    mismatched = verify_evidence_pack(pack, profile="ci", **verify_kwargs)
    assert mismatched.status is EvidencePackStatus.USAGE
    assert any(
        "does not match the authenticated catalog profile" in error
        for error in mismatched.payload["errors"]
    )


def test_authenticated_catalog_relabel_is_rejected_by_direct_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = write_catalog_evidence_fixture(tmp_path, include_relabel_target=True)
    monkeypatch.setattr(
        evidence_pack_mod, "_run_verify_command", _fake_report_verification
    )
    pack = tmp_path / "pack"
    sealed = _seal_fixture(fixture, pack)
    assert sealed.status is EvidencePackStatus.OK
    fingerprint = sealed.payload["signature"]["signer_fingerprint"]

    verified = verify_evidence_pack(
        pack,
        strict=True,
        profile="release",
        report_assurance="strict",
        expected_fingerprint=fingerprint,
        expected_catalog_digest=catalog_digest(fixture.catalog),
        expected_runtime_image_digest=IMAGE_DIGEST,
        policy_pack_path=fixture.policy_pack,
    )
    assert verified.status is EvidencePackStatus.OK

    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    catalog = load_evidence_catalog(fixture.catalog)
    manifest["catalog"] = catalog.binding_for("text-b", path="metadata/catalog.json")
    write_json(manifest_path, manifest)
    sign_manifest(manifest_path, signing_key_path=fixture.signing_key)

    relabelled = verify_evidence_pack(
        pack,
        strict=True,
        profile="release",
        report_assurance="strict",
        expected_fingerprint=fingerprint,
        expected_catalog_digest=catalog_digest(fixture.catalog),
        expected_runtime_image_digest=IMAGE_DIGEST,
        policy_pack_path=fixture.policy_pack,
    )
    assert relabelled.status is EvidencePackStatus.INTEGRITY
    assert any(
        "catalog evidence material cannot be loaded" in error
        for error in relabelled.payload["errors"]
    )


def test_direct_verification_rejects_a_missing_required_catalog_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = write_catalog_evidence_fixture(tmp_path)
    monkeypatch.setattr(
        evidence_pack_mod, "_run_verify_command", _fake_report_verification
    )
    pack = tmp_path / "pack"
    sealed = _seal_fixture(fixture, pack)
    assert sealed.status is EvidencePackStatus.OK
    fingerprint = sealed.payload["signature"]["signer_fingerprint"]
    (pack / "metadata" / "preset.yaml").unlink()

    result = verify_evidence_pack(
        pack,
        strict=True,
        profile="release",
        report_assurance="strict",
        expected_fingerprint=fingerprint,
        expected_catalog_digest=catalog_digest(fixture.catalog),
        expected_runtime_image_digest=IMAGE_DIGEST,
        policy_pack_path=fixture.policy_pack,
    )

    assert result.status is EvidencePackStatus.INTEGRITY
    assert result.payload["ok"] is False
