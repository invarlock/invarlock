from __future__ import annotations

from pathlib import Path

import invarlock.evidence_pack_integrity as evidence_pack_integrity_mod
from tests.reporting.test_evidence_pack_helper_signature_and_manifest import (
    _sign_pack,
    _write_json,
    _write_manifest_and_checksums,
    _write_pack_scaffold,
    evidence_pack_mod,
)


def test_expected_fingerprint_normalization_rejects_missing_and_malformed() -> None:
    assert evidence_pack_integrity_mod.normalize_expected_fingerprint(None) is None
    assert evidence_pack_integrity_mod.normalize_expected_fingerprint("nope") is None


def test_verify_evidence_pack_rejects_malformed_expected_fingerprint(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()

    result = evidence_pack_mod.verify_evidence_pack(
        pack_dir,
        skip_verify=True,
        expected_fingerprint="nope",
    )

    assert result.status.value == 2
    assert result.payload["authenticity"] == "unpinned"
    assert "--expected-fingerprint must be a sha256" in result.payload["errors"][0]


def test_verify_signature_enforces_expected_fingerprint(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
    )
    fingerprint = _sign_pack(pack_dir, tmp_path)

    errors, warnings, fingerprint_out = evidence_pack_mod._verify_signature(
        pack_dir,
        strict=False,
        expected_fingerprints=frozenset({fingerprint}),
    )
    assert errors == []
    assert warnings == []
    assert fingerprint_out == fingerprint

    bad_fingerprint = "sha256:" + ("0" * 64)
    errors, warnings, fingerprint_out = evidence_pack_mod._verify_signature(
        pack_dir,
        strict=False,
        expected_fingerprints=frozenset({bad_fingerprint}),
    )
    assert "manifest signature signer mismatch" in errors[0]
    assert bad_fingerprint in errors[0]
    assert warnings == []
    assert fingerprint_out == fingerprint


def test_verify_evidence_pack_reports_authenticity_with_pin_and_trust_store(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "pack"
    report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
    )
    fingerprint = _sign_pack(pack_dir, tmp_path)

    unpinned = evidence_pack_mod.verify_evidence_pack(pack_dir, skip_verify=True)
    assert unpinned.status.value == 0
    assert unpinned.payload["authenticity"] == "unpinned"
    assert unpinned.payload["signer_fingerprint"] == fingerprint

    pinned = evidence_pack_mod.verify_evidence_pack(
        pack_dir,
        skip_verify=True,
        expected_fingerprint=fingerprint,
    )
    assert pinned.status.value == 0
    assert pinned.payload["authenticity"] == "pinned"

    trust_store = tmp_path / "trusted-signers.json"
    _write_json(trust_store, {"trusted_signers": [fingerprint]})
    trusted = evidence_pack_mod.verify_evidence_pack(
        pack_dir,
        skip_verify=True,
        trust_store_path=trust_store,
    )
    assert trusted.status.value == 0
    assert trusted.payload["authenticity"] == "pinned"
    assert trusted.payload["trust_store"] == str(trust_store)

    mismatch = evidence_pack_mod.verify_evidence_pack(
        pack_dir,
        skip_verify=True,
        expected_fingerprint="sha256:" + ("0" * 64),
    )
    assert mismatch.status.value != 0
    assert mismatch.payload["authenticity"] == "mismatch"
    assert "signer mismatch" in mismatch.payload["errors"][0]
