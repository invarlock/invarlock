from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519

from invarlock import evidence_pack_integrity as integrity
from invarlock.evidence_pack_contract import canonical_json_bytes


def _signature_bundle(
    manifest: bytes,
    key: ed25519.Ed25519PrivateKey,
    *,
    recorded_fingerprint: str | None = None,
) -> dict[str, object]:
    public_key = key.public_key()
    return {
        "format": integrity.EVIDENCE_PACK_SIGNATURE_FORMAT,
        "algorithm": "ed25519",
        "signing_key_fingerprint": (
            recorded_fingerprint or integrity.public_key_fingerprint(public_key)
        ),
        "public_key": {
            "encoding": "pem",
            "value": public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode("ascii"),
        },
        "signature": {
            "encoding": "base64",
            "value": base64.b64encode(key.sign(manifest)).decode("ascii"),
        },
    }


def _signed_pack(
    tmp_path: Path, *, manifest_signer: str | None = None
) -> tuple[Path, str]:
    root = tmp_path / "pack"
    root.mkdir()
    key = ed25519.Ed25519PrivateKey.generate()
    fingerprint = integrity.public_key_fingerprint(key.public_key())
    manifest = canonical_json_bytes(
        {"signing_key_fingerprint": manifest_signer or fingerprint}
    )
    (root / "manifest.json").write_bytes(manifest)
    (root / integrity.MANIFEST_SIGNATURE_FILENAME).write_bytes(
        canonical_json_bytes(_signature_bundle(manifest, key))
    )
    return root, fingerprint


def test_expected_fingerprint_normalization_is_closed() -> None:
    fingerprint = "sha256:" + "a" * 64

    assert (
        integrity.normalize_expected_fingerprint(f"  {fingerprint.upper()}  ")
        == fingerprint
    )
    assert integrity.normalize_expected_fingerprint(None) is None
    assert integrity.normalize_expected_fingerprint("sha256:" + "g" * 64) is None
    assert integrity.normalize_expected_fingerprint("a" * 64) is None


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ([], "JSON object"),
        ({}, "missing or empty"),
        ({"checksums_sha256_digest": "wrong"}, "digest mismatch"),
    ],
)
def test_manifest_must_bind_the_exact_checksum_ledger(
    manifest: object, message: str
) -> None:
    assert message in " ".join(
        integrity.verify_manifest_binds_checksums_payload(manifest, b"ledger\n")
    )
    digest = hashlib.sha256(b"ledger\n").hexdigest()
    assert (
        integrity.verify_manifest_binds_checksums_payload(
            {"checksums_sha256_digest": digest}, b"ledger\n"
        )
        == []
    )


def test_checksum_parser_reports_malformed_and_canonical_duplicates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pack"
    root.mkdir()
    digest = "A" * 64
    (root / "checksums.sha256").write_text(
        f"{digest}  ./payload.json\nnot-a-checksum\n{digest.lower()}  payload.json\n",
        encoding="utf-8",
    )

    entries, errors = integrity.parse_checksums(root)

    assert entries == [
        (digest.lower(), "./payload.json"),
        (digest.lower(), "payload.json"),
    ]
    assert any("not a valid sha256 entry" in error for error in errors)
    assert any("duplicates path 'payload.json'" in error for error in errors)
    assert integrity.canonicalize_checksum_path(".\\nested\\file.json") == (
        "nested/file.json"
    )


def test_checksum_verifier_rejects_traversal_missing_symlink_and_mismatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pack"
    root.mkdir()
    payload = root / "payload.bin"
    payload.write_bytes(b"payload")
    target = tmp_path / "outside.bin"
    target.write_bytes(b"outside")
    (root / "linked.bin").symlink_to(target)
    correct = hashlib.sha256(payload.read_bytes()).hexdigest()
    (root / "checksums.sha256").write_text(
        f"{correct}  payload.bin\n"
        f"{'0' * 64}  missing.bin\n"
        f"{'0' * 64}  ../outside.bin\n"
        f"{'0' * 64}  linked.bin\n"
        f"{'0' * 64}  payload.bin\n",
        encoding="utf-8",
    )

    errors, covered = integrity.verify_checksums(root)

    joined = "\n".join(errors)
    assert "missing from pack" in joined
    assert "escapes the pack root" in joined
    assert "checksum mismatch" in joined
    assert {"payload.bin", "missing.bin", "../outside.bin", "linked.bin"} <= covered


def test_extra_file_policy_distinguishes_strict_errors_from_diagnostics(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pack"
    root.mkdir()
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    (root / "extra.txt").write_text("extra", encoding="utf-8")

    strict_errors, strict_warnings = integrity.verify_no_extra_files(
        root, covered_paths=set(), strict=True
    )
    diagnostic_errors, diagnostic_warnings = integrity.verify_no_extra_files(
        root, covered_paths=set(), strict=False
    )

    assert strict_warnings == []
    assert "extra.txt" in strict_errors[0]
    assert diagnostic_errors == []
    assert diagnostic_warnings == strict_errors
    assert integrity.verify_no_extra_files(
        root, covered_paths={"extra.txt"}, strict=True
    ) == ([], [])


def test_signature_presence_and_file_type_are_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "pack"
    root.mkdir()

    assert integrity.verify_signature(root, strict=True)[0] == [
        "manifest.signature.json missing"
    ]
    assert integrity.verify_signature(root, strict=False)[1] == [
        "manifest.signature.json missing"
    ]

    target = tmp_path / "signature.json"
    target.write_text("{}", encoding="utf-8")
    (root / integrity.MANIFEST_SIGNATURE_FILENAME).symlink_to(target)
    errors, warnings, fingerprint = integrity.verify_signature(root, strict=True)
    assert errors == ["manifest.signature.json must be a regular file"]
    assert warnings == []
    assert fingerprint is None


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"[]", "JSON object"),
        (b'{"format":"x"}', "fields are invalid"),
        (b'{"key":1,"key":2}', "duplicate key"),
    ],
)
def test_signature_bundle_rejects_open_or_ambiguous_shapes(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    root = tmp_path / "pack"
    root.mkdir()
    (root / integrity.MANIFEST_SIGNATURE_FILENAME).write_bytes(payload)

    errors, warnings, fingerprint = integrity.verify_signature(root, strict=True)

    assert message in " ".join(errors)
    assert warnings == []
    assert fingerprint is None


def test_signature_bundle_validates_every_cryptographic_field(tmp_path: Path) -> None:
    root = tmp_path / "pack"
    root.mkdir()
    manifest = b"{}\n"
    key = ed25519.Ed25519PrivateKey.generate()
    base = _signature_bundle(manifest, key)
    mutations = [
        ({"format": "other"}, "format must"),
        ({"algorithm": "rsa"}, "algorithm must"),
        ({"public_key": []}, "public_key is invalid"),
        ({"public_key": {"encoding": "der", "value": "x"}}, "public_key is invalid"),
        ({"signature": []}, "signature is invalid"),
        ({"signature": {"encoding": "hex", "value": "x"}}, "signature is invalid"),
        ({"signing_key_fingerprint": "invalid"}, "fingerprint is invalid"),
    ]
    for index, (mutation, message) in enumerate(mutations):
        payload = {**base, **mutation}
        path = root / integrity.MANIFEST_SIGNATURE_FILENAME
        path.write_bytes(canonical_json_bytes(payload))
        errors, _warnings, _fingerprint = integrity.verify_signature(root, strict=True)
        assert message in " ".join(errors), index


def test_signature_verification_pins_key_and_manifest_identity(tmp_path: Path) -> None:
    root, fingerprint = _signed_pack(tmp_path)

    assert integrity.verify_signature(
        root, strict=True, expected_fingerprints={fingerprint}
    ) == ([], [], fingerprint)

    errors, _warnings, observed = integrity.verify_signature(
        root, strict=True, expected_fingerprints={"sha256:" + "0" * 64}
    )
    assert observed == fingerprint
    assert "signer mismatch" in " ".join(errors)


def test_signature_rejects_embedded_key_mismatch_non_ed25519_and_bad_base64(
    tmp_path: Path,
) -> None:
    root, fingerprint = _signed_pack(tmp_path, manifest_signer="sha256:" + "0" * 64)
    errors, _warnings, observed = integrity.verify_signature(root, strict=True)
    assert observed == fingerprint
    assert errors == ["manifest signing key does not match its signature"]

    manifest = (root / "manifest.json").read_bytes()
    key = ed25519.Ed25519PrivateKey.generate()
    mismatched = _signature_bundle(
        manifest, key, recorded_fingerprint="sha256:" + "1" * 64
    )
    (root / integrity.MANIFEST_SIGNATURE_FILENAME).write_bytes(
        canonical_json_bytes(mismatched)
    )
    errors, _warnings, observed = integrity.verify_signature(root, strict=True)
    assert "fingerprint does not match" in " ".join(errors)
    assert observed is not None

    non_ed = ec.generate_private_key(ec.SECP256R1()).public_key()
    non_ed_payload = _signature_bundle(manifest, key)
    non_ed_payload["public_key"] = {
        "encoding": "pem",
        "value": non_ed.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("ascii"),
    }
    (root / integrity.MANIFEST_SIGNATURE_FILENAME).write_bytes(
        canonical_json_bytes(non_ed_payload)
    )
    assert "must be Ed25519" in " ".join(
        integrity.verify_signature(root, strict=True)[0]
    )

    bad_signature = _signature_bundle(manifest, key)
    assert isinstance(bad_signature["signature"], dict)
    bad_signature["signature"]["value"] = "not-base64"  # type: ignore[index]
    (root / integrity.MANIFEST_SIGNATURE_FILENAME).write_bytes(
        canonical_json_bytes(bad_signature)
    )
    assert "signature verification failed" in " ".join(
        integrity.verify_signature(root, strict=True)[0]
    )
