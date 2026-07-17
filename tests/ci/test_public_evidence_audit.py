from __future__ import annotations

import base64
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.engine import evaluate_request_file, verify_evidence
from invarlock.evidence_pack_integrity import public_key_fingerprint


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
