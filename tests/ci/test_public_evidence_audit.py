from __future__ import annotations

import base64
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519

from invarlock.engine import evaluate_request_file, verify_evidence
from invarlock.evidence_pack_integrity import public_key_fingerprint

DIGEST = "sha256:" + "a" * 64


def _load() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "checks" / "check_public_evidence.py"
    spec = importlib.util.spec_from_file_location("public_evidence_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_index(root: Path, **updates: object) -> None:
    payload: dict[str, object] = {
        "format_version": "invarlock/public-evidence-index-v1",
        "status": "not_created",
        "status_label": "Evidence not yet created",
        "carrier_policy": {"installed_wheel": "compact_index_only"},
        "evidence_count": 0,
        "evidence_file_count": 0,
        "evidence_size_bytes": 0,
        "entries": [],
    }
    payload.update(updates)
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# Public evidence\n", encoding="utf-8")
    (root / "evidence_index.json").write_text(json.dumps(payload), encoding="utf-8")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _private_key_bytes(key: ed25519.Ed25519PrivateKey) -> bytes:
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _write_local_publication(
    module: ModuleType,
    root: Path,
    *,
    successful_receipt: bool = True,
) -> tuple[Path, Path]:
    repository = Path(__file__).resolve().parents[2]
    evaluation = root.parent / "evaluation"
    evaluation.mkdir(parents=True)
    for directory in ("inputs", "policy", "import"):
        shutil.copytree(repository / "examples" / directory, evaluation / directory)
    shutil.copy2(repository / "examples" / "request.yaml", evaluation / "request.yaml")
    evidence_signing_key = ed25519.Ed25519PrivateKey.generate()
    evidence_signing_key_path = evaluation / "evidence-signer.pem"
    evidence_signing_key_path.write_bytes(_private_key_bytes(evidence_signing_key))
    evidence_signing_key_path.chmod(0o600)
    evaluated = evaluate_request_file(
        evaluation / "request.yaml",
        signing_key_path=evidence_signing_key_path,
    )

    entry_root = root / "evidence" / "local"
    pack = entry_root / "evidence"
    entry_root.mkdir(parents=True)
    shutil.copytree(evaluated.evidence_path, pack)
    verifier_key = ed25519.Ed25519PrivateKey.generate()
    verifier_key_path = root.parent / "verifier.pem"
    verifier_key_path.write_bytes(_private_key_bytes(verifier_key))
    verifier_key_path.chmod(0o600)
    receipt = entry_root / "verification.receipt.json"
    expected_inputs = {
        role: json.loads(
            (pack / "inputs" / f"{role}.json").read_text(encoding="utf-8")
        )["digest"]
        for role in ("baseline", "subject", "dataset")
    }
    verify_evidence(
        pack,
        policy_path=evaluation / "policy" / "acceptance.json",
        expected_baseline_artifact=expected_inputs["baseline"],
        expected_subject_artifact=expected_inputs["subject"],
        expected_schedule=expected_inputs["dataset"],
        expected_baseline_runtime="sha256:" + "1" * 64,
        expected_subject_runtime="sha256:" + "2" * 64,
        expected_signer=public_key_fingerprint(evidence_signing_key.public_key()),
        receipt_path=receipt,
        verifier_signing_key_path=verifier_key_path,
        verifier_identity="tests.public-evidence-audit",
    )
    if not successful_receipt:
        receipt.chmod(0o644)
        receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
        statement = receipt_value["statement"]
        statement["verdict"] = {
            "ok": False,
            "integrity_ok": True,
            "policy_verdict": "fail",
            "verification_status": 7,
        }
        receipt_value["signature"]["value"] = base64.b64encode(
            verifier_key.sign(_canonical_json_bytes(statement))
        ).decode("ascii")
        receipt.write_bytes(_canonical_json_bytes(receipt_value))
    (entry_root / "evidence.meta.json").write_text(
        json.dumps(
            {
                "format_version": "invarlock/public-evidence-meta-v1",
                "summary": "Authenticated example comparison",
                "artifact_paths": {
                    "evidence_pack": "evidence",
                    "verification_receipt": "verification.receipt.json",
                },
            }
        ),
        encoding="utf-8",
    )
    pack_summary = module._artifact_summary(pack, source_root=root)
    receipt_summary = module._artifact_summary(receipt, source_root=root)
    _write_index(
        root,
        status="available",
        status_label="Evidence available",
        evidence_count=1,
        evidence_file_count=pack_summary["file_count"] + 1,
        evidence_size_bytes=(
            pack_summary["size_bytes"] + receipt_summary["size_bytes"]
        ),
        entries=[
            {
                "slug": "local",
                "path": "public_evidence/evidence/local",
                "evidence_class": "signed_evidence_pack",
                "summary": "Local evidence",
                "artifacts": {
                    "evidence_pack": pack_summary,
                    "verification_receipt": receipt_summary,
                },
            }
        ],
    )
    return pack, receipt


def _valid_receipt_anchors() -> dict[str, object]:
    return {
        "policy_digest": DIGEST,
        "artifact_digests": {"baseline": DIGEST, "subject": DIGEST},
        "schedule_digest": DIGEST,
        "runtime_digests": {"baseline": DIGEST, "subject": DIGEST},
        "pack_signer_fingerprint": DIGEST,
    }


def _valid_external_entry(slug: str = "external") -> dict[str, object]:
    return {
        "slug": slug,
        "path": f"public_evidence/evidence/{slug}",
        "evidence_class": "signed_evidence_pack",
        "summary": "External evidence",
        "artifacts": {
            "evidence_pack": {
                "kind": "directory",
                "path": f"public_evidence/evidence/{slug}/evidence",
                "file_count": 3,
                "size_bytes": 30,
                "control_hashes": {"manifest.json": DIGEST},
                "external_asset": {
                    "url": "https://example.com/evidence.tar.zst",
                    "sha256": DIGEST,
                },
            },
            "verification_receipt": {
                "kind": "file",
                "path": f"public_evidence/evidence/{slug}/verification.receipt.json",
                "size_bytes": 10,
                "sha256": DIGEST,
                "external_asset": {
                    "url": "https://example.com/verification.receipt.json",
                    "sha256": DIGEST,
                },
            },
        },
    }


def test_empty_public_evidence_is_an_explicit_valid_state(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    _write_index(root)

    assert module.check_public_evidence(root) == []


def test_empty_public_evidence_requires_the_not_created_label(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    _write_index(root, status="available", status_label="Available")

    errors = module.check_public_evidence(root)

    assert any("Evidence not yet created" in error for error in errors)


def test_public_evidence_rejects_obsolete_and_private_markers(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    _write_index(root)
    index = root / "evidence_index.json"
    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["note"] = "frozen-v1 at /root/private"
    index.write_text(json.dumps(payload), encoding="utf-8")

    errors = module.check_public_evidence(root)

    assert any("obsolete marker" in error for error in errors)
    assert any("private marker" in error for error in errors)


def test_output_text_that_resembles_a_drive_prefix_is_not_a_host_path(
    tmp_path: Path,
) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    _write_local_publication(module, root)
    index = root / "evidence_index.json"
    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["entries"][0]["summary"] = "F:\n"
    index.write_text(json.dumps(payload), encoding="utf-8")

    assert module.check_public_evidence(root) == []


def test_windows_host_path_is_rejected_in_a_typed_artifact_path() -> None:
    module = _load()

    assert not module._safe_logical_path(
        "C:\\Users\\operator\\evidence.json",
        prefix="public_evidence/evidence/example/",
    )


def test_public_evidence_rejects_unindexed_surfaces(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    _write_index(root)
    (root / "historical_failures.json").write_text("{}", encoding="utf-8")

    errors = module.check_public_evidence(root)

    assert errors == ["unexpected public evidence surfaces: historical_failures.json"]


def test_public_evidence_ignores_untracked_macos_directory_metadata(
    tmp_path: Path,
) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    _write_index(root)
    (root / ".DS_Store").write_bytes(b"local Finder state")
    evidence_root = root / "evidence"
    evidence_root.mkdir()
    (evidence_root / ".DS_Store").write_bytes(b"local Finder state")

    assert module.check_public_evidence(root) == []


@pytest.mark.parametrize("entry_kind", ["directory", "symlink"])
def test_public_evidence_rejects_non_file_macos_metadata_surfaces(
    tmp_path: Path, entry_kind: str
) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    _write_index(root)
    metadata = root / ".DS_Store"
    if entry_kind == "directory":
        metadata.mkdir()
        (metadata / "unreviewed.txt").write_text("not evidence", encoding="utf-8")
    else:
        target = tmp_path / "unreviewed.txt"
        target.write_text("not evidence", encoding="utf-8")
        metadata.symlink_to(target)

    assert module.check_public_evidence(root) == [
        "unexpected public evidence surfaces: .DS_Store"
    ]


def test_external_entries_can_coexist_without_local_directories(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    digest = "sha256:" + "a" * 64
    entry = {
        "slug": "external",
        "path": "public_evidence/evidence/external",
        "evidence_class": "signed_evidence_pack",
        "summary": "External evidence",
        "artifacts": {
            "evidence_pack": {
                "kind": "directory",
                "path": "public_evidence/evidence/external/evidence",
                "file_count": 3,
                "size_bytes": 30,
                "control_hashes": {"manifest.json": digest},
                "external_asset": {
                    "url": "https://example.com/evidence.tar.zst",
                    "sha256": digest,
                },
            },
            "verification_receipt": {
                "kind": "file",
                "path": "public_evidence/evidence/external/verification.receipt.json",
                "size_bytes": 10,
                "sha256": digest,
                "external_asset": {
                    "url": "https://example.com/verification.receipt.json",
                    "sha256": digest,
                },
            },
        },
    }
    _write_index(
        root,
        status="available",
        status_label="Evidence available",
        evidence_count=1,
        evidence_file_count=4,
        evidence_size_bytes=40,
        entries=[entry],
    )
    (root / "evidence").mkdir()

    assert module.check_public_evidence(root) == []


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/evidence.tar.zst",
        "https://user:secret@example.com/evidence.tar.zst",
        "https://example.com/evidence.tar.zst?token=secret",
        "https://example.com/evidence.tar.zst#signed-token",
        "https:///evidence.tar.zst",
        "https://[invalid/evidence.tar.zst",
    ],
)
def test_external_entries_reject_unsafe_or_credential_bearing_urls(
    tmp_path: Path,
    url: str,
) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    digest = "sha256:" + "a" * 64
    entry = {
        "slug": "external",
        "path": "public_evidence/evidence/external",
        "evidence_class": "signed_evidence_pack",
        "summary": "External evidence",
        "artifacts": {
            "evidence_pack": {
                "kind": "directory",
                "path": "public_evidence/evidence/external/evidence",
                "file_count": 3,
                "size_bytes": 30,
                "control_hashes": {"manifest.json": digest},
                "external_asset": {"url": url, "sha256": digest},
            },
            "verification_receipt": {
                "kind": "file",
                "path": "public_evidence/evidence/external/verification.receipt.json",
                "size_bytes": 10,
                "sha256": digest,
                "external_asset": {"url": url, "sha256": digest},
            },
        },
    }
    _write_index(
        root,
        status="available",
        status_label="Evidence available",
        evidence_count=1,
        evidence_file_count=4,
        evidence_size_bytes=40,
        entries=[entry],
    )
    (root / "evidence").mkdir()

    errors = module.check_public_evidence(root)

    assert any("credential-free HTTPS" in error for error in errors)


def test_local_publication_accepts_a_manifest_bound_signed_receipt(
    tmp_path: Path,
) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    _write_local_publication(module, root)

    assert module.check_public_evidence(root) == []


def test_local_artifact_summary_must_match_published_bytes(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    _write_local_publication(module, root)
    index_path = root / "evidence_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    pack_summary = index["entries"][0]["artifacts"]["evidence_pack"]
    pack_summary["size_bytes"] += 1
    index["evidence_size_bytes"] += 1
    index_path.write_text(json.dumps(index), encoding="utf-8")

    errors = module.check_public_evidence(root)

    assert any("artifact summary does not match its bytes" in error for error in errors)


def test_local_receipt_must_bind_the_published_manifest(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    pack, _ = _write_local_publication(module, root)
    manifest = pack / "manifest.json"
    manifest.chmod(0o644)
    manifest.write_text(
        json.dumps({"format": "invarlock/evidence-pack-v1", "changed": True}),
        encoding="utf-8",
    )

    index_path = root / "evidence_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    old_summary = index["entries"][0]["artifacts"]["evidence_pack"]
    new_summary = module._artifact_summary(pack, source_root=root)
    index["entries"][0]["artifacts"]["evidence_pack"] = new_summary
    index["evidence_size_bytes"] += (
        new_summary["size_bytes"] - old_summary["size_bytes"]
    )
    index_path.write_text(json.dumps(index), encoding="utf-8")

    errors = module.check_public_evidence(root)

    assert any("does not bind the pack manifest" in error for error in errors)


def test_local_receipt_signature_must_verify(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    _, receipt = _write_local_publication(module, root)
    receipt.chmod(0o644)
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["signature"]["value"] = base64.b64encode(b"\0" * 64).decode("ascii")
    receipt.write_text(json.dumps(value), encoding="utf-8")

    index_path = root / "evidence_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"][0]["artifacts"]["verification_receipt"] = module._artifact_summary(
        receipt, source_root=root
    )
    index_path.write_text(json.dumps(index), encoding="utf-8")

    errors = module.check_public_evidence(root)

    assert any("signature verification failed" in error for error in errors)


def test_local_receipt_requires_artifact_and_schedule_anchors(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    _, receipt = _write_local_publication(module, root)
    receipt.chmod(0o644)
    value = json.loads(receipt.read_text(encoding="utf-8"))
    anchors = value["statement"]["anchors"]
    del anchors["artifact_digests"]
    del anchors["schedule_digest"]
    receipt.write_text(json.dumps(value), encoding="utf-8")

    index_path = root / "evidence_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"][0]["artifacts"]["verification_receipt"] = module._artifact_summary(
        receipt, source_root=root
    )
    index_path.write_text(json.dumps(index), encoding="utf-8")

    errors = module.check_public_evidence(root)

    assert any("anchor fields are invalid" in error for error in errors)


def test_local_receipt_must_record_successful_strict_acceptance(
    tmp_path: Path,
) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    _write_local_publication(module, root, successful_receipt=False)

    errors = module.check_public_evidence(root)

    assert any("must record successful strict acceptance" in error for error in errors)


def test_local_metadata_fields_are_closed(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    pack, _receipt = _write_local_publication(module, root)
    metadata_path = pack.parent / "evidence.meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["review_note"] = "undeclared metadata"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    errors = module.check_public_evidence(root)

    assert any("metadata fields are not closed" in error for error in errors)


def test_index_artifact_paths_must_remain_under_their_slug(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    _write_local_publication(module, root)
    index_path = root / "evidence_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"][0]["artifacts"]["evidence_pack"]["path"] = (
        "public_evidence/evidence/another-entry/evidence"
    )
    index_path.write_text(json.dumps(index), encoding="utf-8")

    errors = module.check_public_evidence(root)

    assert any("unsafe evidence_pack path" in error for error in errors)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"policy_digest": "bad"}, "policy digest is invalid"),
        ({"pack_signer_fingerprint": "bad"}, "pack signer is invalid"),
        ({"artifact_digests": []}, "artifact anchors are invalid"),
        (
            {"artifact_digests": {"baseline": DIGEST, "subject": "bad"}},
            "artifact anchors are invalid",
        ),
        ({"schedule_digest": "bad"}, "schedule anchor is invalid"),
        ({"runtime_digests": []}, "runtime anchors are invalid"),
        (
            {"runtime_digests": {"baseline": DIGEST, "subject": "bad"}},
            "runtime anchors are invalid",
        ),
    ],
)
def test_receipt_anchor_validation_rejects_invalid_bound_materials(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    module = _load()
    anchors = _valid_receipt_anchors()
    anchors.update(mutation)
    errors: list[str] = []

    module._check_receipt_anchors(errors, tmp_path / "receipt.json", anchors)

    assert any(message in error for error in errors)


def test_receipt_anchor_validation_rejects_open_shape(tmp_path: Path) -> None:
    module = _load()
    errors: list[str] = []

    module._check_receipt_anchors(errors, tmp_path / "receipt.json", {"extra": True})

    assert errors and "anchor fields are invalid" in errors[0]


def test_v2_receipt_anchor_rejects_invalid_request_digest(tmp_path: Path) -> None:
    module = _load()
    anchors = _valid_receipt_anchors()
    anchors["request_digest"] = "not-a-digest"
    errors: list[str] = []

    module._check_receipt_anchors(
        errors,
        tmp_path / "receipt.json",
        anchors,
        require_request=True,
    )

    assert errors == [
        f"{tmp_path / 'receipt.json'}: signed receipt request digest is invalid"
    ]


def test_pack_request_context_rejects_mismatched_digest_and_malformed_request(
    tmp_path: Path,
) -> None:
    module = _load()
    pack = tmp_path / "evidence"
    pack.mkdir()
    request_path = pack / "request.json"
    request_path.write_text("[]\n", encoding="utf-8")
    manifest_path = pack / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"evidence": {"request": {"path": "request.json", "digest": DIGEST}}}
        ),
        encoding="utf-8",
    )
    receipt = tmp_path / "receipt.json"
    errors: list[str] = []

    request_digest, uses_llama_cpp = module._public_pack_request_context(
        errors,
        receipt=receipt,
        manifest_path=manifest_path,
    )

    assert request_digest == module._sha256_bytes(request_path.read_bytes())
    assert uses_llama_cpp is False
    assert errors == [f"{receipt}: pack request digest does not match manifest"]


def test_pack_request_context_rejects_noncanonical_reference_path(
    tmp_path: Path,
) -> None:
    module = _load()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"evidence": {"request": {"path": "../request.json", "digest": DIGEST}}}
        ),
        encoding="utf-8",
    )
    errors: list[str] = []

    request_digest, uses_llama_cpp = module._public_pack_request_context(
        errors,
        receipt=tmp_path / "receipt.json",
        manifest_path=manifest_path,
    )

    assert request_digest is None
    assert uses_llama_cpp is False
    assert errors == [f"{manifest_path}: request reference path is invalid"]


@pytest.mark.parametrize(
    ("receipt_format", "request_anchor", "message"),
    [
        (
            "invarlock/evidence-verification-receipt-v1",
            None,
            "llama_cpp evidence requires signed receipt format v2",
        ),
        (
            "invarlock/evidence-verification-receipt-v2",
            DIGEST,
            "signed request anchor does not bind the pack request",
        ),
    ],
)
def test_signed_receipt_fail_closes_gguf_request_binding(
    tmp_path: Path,
    receipt_format: str,
    request_anchor: str | None,
    message: str,
) -> None:
    module = _load()
    pack = tmp_path / "evidence"
    pack.mkdir()
    request_path = pack / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "comparison": {
                    "baseline": {"runtime": {"provider": "llama_cpp"}},
                    "subject": {"runtime": {"provider": "llama_cpp"}},
                }
            }
        ),
        encoding="utf-8",
    )
    request_digest = module._sha256_bytes(request_path.read_bytes())
    manifest_path = pack / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "evidence": {
                    "request": {"path": "request.json", "digest": request_digest}
                }
            }
        ),
        encoding="utf-8",
    )
    anchors = _valid_receipt_anchors()
    if request_anchor is not None:
        anchors["request_digest"] = request_anchor
    statement = {
        "format": receipt_format,
        "pack_manifest_digest": module._sha256_bytes(manifest_path.read_bytes()),
        "anchors": anchors,
        "verifier": {
            "identity": "tests.verifier",
            "signing_key_fingerprint": DIGEST,
            "trust_profile_digest": None,
        },
        "verdict": {
            "ok": True,
            "integrity_ok": True,
            "policy_verdict": "pass",
            "verification_status": 0,
        },
    }
    errors: list[str] = []

    module._check_signed_receipt(
        errors,
        receipt=tmp_path / "receipt.json",
        value={"statement": statement, "signature": {}},
        manifest_path=manifest_path,
    )

    assert any(message in error for error in errors)


@pytest.mark.parametrize(
    ("verifier", "message"),
    [
        ({"identity": "test"}, "verifier fields are invalid"),
        (
            {
                "identity": "contains spaces",
                "signing_key_fingerprint": DIGEST,
                "trust_profile_digest": None,
            },
            "verifier identity is invalid",
        ),
        (
            {
                "identity": "tests.verifier",
                "signing_key_fingerprint": "bad",
                "trust_profile_digest": None,
            },
            "verifier fingerprint is invalid",
        ),
        (
            {
                "identity": "tests.verifier",
                "signing_key_fingerprint": DIGEST,
                "trust_profile_digest": "bad",
            },
            "trust profile digest is invalid",
        ),
    ],
)
def test_receipt_verifier_validation_rejects_ambiguous_identity(
    tmp_path: Path,
    verifier: dict[str, object],
    message: str,
) -> None:
    module = _load()
    errors: list[str] = []

    module._check_receipt_verifier(errors, tmp_path / "receipt.json", verifier)

    assert any(message in error for error in errors)


@pytest.mark.parametrize(
    ("verdict", "message"),
    [
        ({"ok": True}, "verdict fields are invalid"),
        (
            {
                "ok": 1,
                "integrity_ok": True,
                "policy_verdict": "pass",
                "verification_status": 0,
            },
            "verdict booleans are invalid",
        ),
        (
            {
                "ok": True,
                "integrity_ok": True,
                "policy_verdict": "unknown",
                "verification_status": 0,
            },
            "policy verdict is invalid",
        ),
        (
            {
                "ok": True,
                "integrity_ok": True,
                "policy_verdict": "pass",
                "verification_status": True,
            },
            "verification status is invalid",
        ),
    ],
)
def test_receipt_verdict_validation_rejects_invalid_acceptance_claims(
    tmp_path: Path,
    verdict: dict[str, object],
    message: str,
) -> None:
    module = _load()
    errors: list[str] = []

    module._check_receipt_verdict(errors, tmp_path / "receipt.json", verdict)

    assert any(message in error for error in errors)


@pytest.mark.parametrize(
    ("signature", "message"),
    [
        ({}, "signature is required"),
        (
            {
                "algorithm": "ed25519",
                "format": "invarlock/evidence-verification-receipt-signature-v1",
                "public_key": [],
                "value": "",
            },
            "public key is invalid",
        ),
        (
            {
                "algorithm": "ed25519",
                "format": "invarlock/evidence-verification-receipt-signature-v1",
                "public_key": {"encoding": "pem", "value": 7},
                "value": "",
            },
            "public key is invalid",
        ),
        (
            {
                "algorithm": "ed25519",
                "format": "invarlock/evidence-verification-receipt-signature-v1",
                "public_key": {"encoding": "pem", "value": "not a pem"},
                "value": "",
            },
            "public key is invalid",
        ),
    ],
)
def test_receipt_public_key_rejects_malformed_signature_blocks(
    tmp_path: Path,
    signature: dict[str, object],
    message: str,
) -> None:
    module = _load()
    errors: list[str] = []

    public_key = module._receipt_public_key(
        errors, tmp_path / "receipt.json", signature
    )

    assert public_key is None
    assert any(message in error for error in errors)


def test_receipt_public_key_rejects_a_valid_non_ed25519_key(tmp_path: Path) -> None:
    module = _load()
    public_key = ec.generate_private_key(ec.SECP256R1()).public_key()
    signature = {
        "algorithm": "ed25519",
        "format": "invarlock/evidence-verification-receipt-signature-v1",
        "public_key": {
            "encoding": "pem",
            "value": public_key.public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode("ascii"),
        },
        "value": "",
    }
    errors: list[str] = []

    assert (
        module._receipt_public_key(errors, tmp_path / "receipt.json", signature) is None
    )
    assert "public key is not Ed25519" in " ".join(errors)


def test_signed_receipt_rejects_missing_statement(tmp_path: Path) -> None:
    module = _load()
    errors: list[str] = []

    module._check_signed_receipt(
        errors,
        receipt=tmp_path / "receipt.json",
        value={"signature": {}},
        manifest_path=tmp_path / "manifest.json",
    )

    assert any("fields are not closed" in error for error in errors)
    assert any("statement is required" in error for error in errors)


def test_signed_receipt_rejects_invalid_manifest_claim_and_unsigned_shape(
    tmp_path: Path,
) -> None:
    module = _load()
    errors: list[str] = []
    statement = {
        "format": "future",
        "pack_manifest_digest": "bad",
        "anchors": _valid_receipt_anchors(),
        "verifier": {
            "identity": "tests.verifier",
            "signing_key_fingerprint": DIGEST,
            "trust_profile_digest": None,
        },
        "verdict": {
            "ok": True,
            "integrity_ok": True,
            "policy_verdict": "pass",
            "verification_status": 0,
        },
        "undeclared": True,
    }

    module._check_signed_receipt(
        errors,
        receipt=tmp_path / "receipt.json",
        value={"statement": statement, "signature": {}},
        manifest_path=tmp_path / "manifest.json",
    )

    assert any("statement fields are invalid" in error for error in errors)
    assert any("receipt format is invalid" in error for error in errors)
    assert any("manifest digest is invalid" in error for error in errors)


def test_signed_receipt_rejects_unreadable_claimed_manifest(tmp_path: Path) -> None:
    module = _load()
    errors: list[str] = []
    statement = {
        "format": "invarlock/evidence-verification-receipt-v1",
        "pack_manifest_digest": DIGEST,
        "anchors": _valid_receipt_anchors(),
        "verifier": {
            "identity": "tests.verifier",
            "signing_key_fingerprint": DIGEST,
            "trust_profile_digest": None,
        },
        "verdict": {
            "ok": True,
            "integrity_ok": True,
            "policy_verdict": "pass",
            "verification_status": 0,
        },
    }

    module._check_signed_receipt(
        errors,
        receipt=tmp_path / "receipt.json",
        value={"statement": statement, "signature": {}},
        manifest_path=tmp_path / "missing-manifest.json",
    )

    assert any("could not read pack manifest" in error for error in errors)


@pytest.mark.parametrize(
    "value",
    [None, "/absolute/path", "public_evidence/evidence/demo/../secret", "wrong/root"],
)
def test_safe_logical_path_rejects_noncanonical_or_escaping_values(
    value: object,
) -> None:
    module = _load()

    assert not module._safe_logical_path(value, prefix="public_evidence/evidence/demo/")


def test_safe_external_url_rejects_non_text() -> None:
    module = _load()

    assert not module._safe_external_url(None)


def test_index_entry_rejects_non_object_and_invalid_slug(tmp_path: Path) -> None:
    module = _load()
    errors: list[str] = []

    module._check_index_entry(errors, [], tmp_path)
    module._check_index_entry(errors, {"slug": "bad/slug"}, tmp_path)

    assert "public evidence entry must be an object" in errors
    assert "public evidence entry slug is invalid" in errors


def test_index_entry_rejects_wrong_class_path_and_missing_artifacts(
    tmp_path: Path,
) -> None:
    module = _load()
    errors: list[str] = []
    entry = {
        "slug": "demo",
        "path": "elsewhere",
        "evidence_class": "historical",
        "artifacts": {},
    }

    module._check_index_entry(errors, entry, tmp_path / "public_evidence")

    assert any(
        "evidence_class must be signed_evidence_pack" in error for error in errors
    )
    assert any("entry path must be" in error for error in errors)
    assert any("missing evidence_pack" in error for error in errors)
    assert any("missing verification_receipt" in error for error in errors)


@pytest.mark.parametrize(
    ("summary", "message"),
    [
        (
            {
                "kind": "file",
                "path": "public_evidence/evidence/demo/receipt.json",
                "size_bytes": 1,
                "sha256": DIGEST,
                "undeclared": True,
            },
            "summary fields are not closed",
        ),
        (
            {
                "kind": "archive",
                "path": "public_evidence/evidence/demo/evidence",
                "size_bytes": 1,
            },
            "kind must be file or directory",
        ),
        (
            {
                "kind": "file",
                "path": "public_evidence/evidence/demo/receipt.json",
                "size_bytes": True,
                "sha256": DIGEST,
            },
            "artifact size is invalid",
        ),
        (
            {
                "kind": "file",
                "path": "public_evidence/evidence/demo/receipt.json",
                "size_bytes": 1,
                "sha256": "bad",
            },
            "artifact digest is invalid",
        ),
        (
            {
                "kind": "directory",
                "path": "public_evidence/evidence/demo/evidence",
                "size_bytes": 1,
                "file_count": -1,
                "control_hashes": {},
            },
            "artifact file count is invalid",
        ),
        (
            {
                "kind": "directory",
                "path": "public_evidence/evidence/demo/evidence",
                "size_bytes": 1,
                "file_count": 1,
                "control_hashes": {"unexpected.json": DIGEST},
            },
            "artifact control hashes are invalid",
        ),
    ],
)
def test_artifact_summary_rejects_invalid_shape_and_totals(
    tmp_path: Path,
    summary: dict[str, object],
    message: str,
) -> None:
    module = _load()
    errors: list[str] = []
    entry = {
        "slug": "demo",
        "artifacts": {"evidence_pack": summary},
    }

    module._check_artifact_summary(
        errors, entry, "evidence_pack", tmp_path / "public_evidence"
    )

    assert any(message in error for error in errors)


def test_artifact_summary_rejects_open_external_asset_and_carrier_conflict(
    tmp_path: Path,
) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    local = root / "evidence" / "demo" / "receipt.json"
    local.parent.mkdir(parents=True)
    local.write_text("{}\n", encoding="utf-8")
    summary = {
        "kind": "file",
        "path": "public_evidence/evidence/demo/receipt.json",
        "size_bytes": local.stat().st_size,
        "sha256": module._sha256_bytes(local.read_bytes()),
        "external_asset": {
            "url": "http://example.com/receipt.json",
            "sha256": "bad",
            "token": "secret",
        },
    }
    errors: list[str] = []

    module._check_artifact_summary(
        errors,
        {"slug": "demo", "artifacts": {"verification_receipt": summary}},
        "verification_receipt",
        root,
    )

    assert any("external asset fields are not closed" in error for error in errors)
    assert any("credential-free HTTPS" in error for error in errors)
    assert any("external asset digest is invalid" in error for error in errors)
    assert any("one publication carrier" in error for error in errors)


def test_artifact_summary_rejects_symlinked_local_carrier(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    local = root / "evidence" / "demo" / "receipt.json"
    local.parent.mkdir(parents=True)
    local.symlink_to(outside)
    errors: list[str] = []

    module._check_artifact_summary(
        errors,
        {
            "slug": "demo",
            "artifacts": {
                "verification_receipt": {
                    "kind": "file",
                    "path": "public_evidence/evidence/demo/receipt.json",
                    "size_bytes": outside.stat().st_size,
                    "sha256": module._sha256_bytes(outside.read_bytes()),
                }
            },
        },
        "verification_receipt",
        root,
    )

    assert any("symlinks are not allowed" in error for error in errors)


def test_artifact_totals_ignore_open_or_untyped_entries() -> None:
    module = _load()

    assert module._artifact_totals(
        [
            None,
            {"artifacts": None},
            {"artifacts": {"invalid": None}},
            {
                "artifacts": {
                    "directory": {
                        "kind": "directory",
                        "file_count": 2,
                        "size_bytes": 30,
                    },
                    "file": {"kind": "file", "size_bytes": 10},
                    "bools": {
                        "kind": "directory",
                        "file_count": True,
                        "size_bytes": False,
                    },
                }
            },
        ]
    ) == (3, 40)


def test_local_entry_reports_pack_format_anchor_and_rendered_signer_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    pack, _receipt = _write_local_publication(module, root)
    manifest_path = pack / "manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_bytes())
    manifest["format"] = "invarlock/evidence-pack-v99"
    manifest["inputs"]["baseline"]["material_digest"] = "sha256:" + "0" * 64
    manifest_path.write_bytes(_canonical_json_bytes(manifest))
    monkeypatch.setattr(
        module,
        "render_evidence",
        lambda _pack: SimpleNamespace(evidence_signer="sha256:" + "9" * 64),
    )
    errors: list[str] = []

    module._check_local_entry(errors, root / "evidence/local")

    joined = "\n".join(errors)
    assert "only the canonical invarlock/evidence-pack-v1" in joined
    assert "baseline anchor does not bind the pack manifest" in joined
    assert "does not match the verified pack signer" in joined


def test_local_entry_rejects_missing_pack_before_receipt_processing(
    tmp_path: Path,
) -> None:
    module = _load()
    entry = tmp_path / "evidence/local"
    entry.mkdir(parents=True)
    entry.joinpath("evidence.meta.json").write_text(
        json.dumps(
            {
                "format_version": "invarlock/public-evidence-meta-v1",
                "summary": "Missing pack",
                "artifact_paths": {
                    "evidence_pack": "evidence",
                    "verification_receipt": "verification.receipt.json",
                },
            }
        ),
        encoding="utf-8",
    )
    errors: list[str] = []

    module._check_local_entry(errors, entry)

    assert errors == [f"{entry / 'evidence'}: evidence pack is missing or unsafe"]


def test_local_tree_rejects_unindexed_directory_and_loose_file(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    evidence_root = root / "evidence"
    (evidence_root / "orphan").mkdir(parents=True)
    (evidence_root / "loose.txt").write_text("unexpected\n", encoding="utf-8")
    errors: list[str] = []

    module._check_local_evidence_tree(errors, root, [])

    assert "every local evidence directory must appear in the index" in errors
    assert any("unexpected files" in error for error in errors)
    assert any("missing safe evidence.meta.json" in error for error in errors)


def test_public_evidence_reports_missing_root_readme_and_index(tmp_path: Path) -> None:
    module = _load()
    missing = tmp_path / "missing"

    assert module.check_public_evidence(missing) == [
        f"public evidence root not found: {missing.resolve()}"
    ]

    root = tmp_path / "public_evidence"
    root.mkdir()
    errors = module.check_public_evidence(root)
    assert "public_evidence/README.md is required" in errors
    assert "public_evidence/evidence_index.json is required" in errors


def test_public_evidence_rejects_invalid_index_json(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    root.mkdir()
    (root / "README.md").write_text("# Evidence\n", encoding="utf-8")
    (root / "evidence_index.json").write_text("{", encoding="utf-8")

    errors = module.check_public_evidence(root)

    assert errors and "Expecting property name" in errors[0]


def test_public_evidence_rejects_index_format_and_summary_total_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    monkeypatch.setattr(module, "_validate_index", lambda *_args: None)
    root = tmp_path / "public_evidence"
    _write_index(
        root,
        format_version="invarlock/public-evidence-index-v99",
        evidence_file_count=1,
        evidence_size_bytes=1,
    )

    errors = module.check_public_evidence(root)

    assert "public evidence index format is invalid" in errors
    assert "evidence_file_count must match artifact summaries" in errors
    assert "evidence_size_bytes must match artifact summaries" in errors


def test_public_evidence_checks_packaged_index_at_canonical_root(
    tmp_path: Path,
) -> None:
    module = _load()
    root = tmp_path / "public_evidence"
    packaged = tmp_path / "packaged"
    _write_index(root)
    packaged.mkdir()
    (packaged / "evidence_index.json").write_text("{}\n", encoding="utf-8")
    module.SOURCE_ROOT = root
    module.PACKAGED_ROOT = packaged

    errors = module.check_public_evidence(root)

    assert "source and packaged public evidence indexes differ" in errors


def test_public_evidence_cli_reports_success_and_failure(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "checks"
        / "check_public_evidence.py"
    )
    valid = tmp_path / "valid"
    invalid = tmp_path / "invalid"
    _write_index(valid)

    passed = subprocess.run(
        [sys.executable, str(script), "--root", str(valid)],
        capture_output=True,
        text=True,
        check=False,
    )
    failed = subprocess.run(
        [sys.executable, str(script), "--root", str(invalid)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert passed.returncode == 0
    assert "audit passed" in passed.stdout
    assert failed.returncode == 1
    assert "public evidence root not found" in failed.stdout
