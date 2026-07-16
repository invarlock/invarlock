#!/usr/bin/env python3
"""Audit the canonical public evidence index and any local evidence packs."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from invarlock.evidence_pack_integrity import (  # noqa: E402
    public_key_fingerprint,
)
from invarlock.evidence_reporting import (  # noqa: E402
    EvidenceReportError,
    render_evidence,
)
from scripts.checks.sync_packaged_public_evidence import (  # noqa: E402
    EVIDENCE_DIRNAME,
    INDEX_FILENAME,
    INDEX_FORMAT_VERSION,
    PACKAGED_ROOT,
    SOURCE_ROOT,
    _artifact_summary,
    _read_object,
    _validate_index,
    _validate_metadata,
)

_DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")
_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_RECEIPT_FORMAT = "invarlock/evidence-verification-receipt-v1"
_RECEIPT_SIGNATURE_FORMAT = "invarlock/evidence-verification-receipt-signature-v1"
_PRIVATE_MARKERS = (
    "/Users/",
    "/home/",
    "/root/",
    "ssh root@",
    "INVARLOCK_SIGNING_KEY",
    "PRIVATE KEY",
)
_OBSOLETE_MARKERS = (
    "published_basis",
    "frozen-v1",
    "catalog_evidence_index",
    "catalog_evidence/",
)


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


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _check_signed_receipt(
    errors: list[str],
    *,
    receipt: Path,
    value: dict[str, Any],
    manifest_path: Path,
) -> None:
    if set(value) != {"statement", "signature"}:
        errors.append(f"{receipt}: signed receipt fields are not closed")

    statement = value.get("statement")
    signature = value.get("signature")
    if not isinstance(statement, dict):
        errors.append(f"{receipt}: signed verification receipt statement is required")
        return
    expected_statement_fields = {
        "format",
        "pack_manifest_digest",
        "anchors",
        "verifier",
        "verdict",
    }
    if set(statement) != expected_statement_fields:
        errors.append(f"{receipt}: signed receipt statement fields are invalid")
    if statement.get("format") != _RECEIPT_FORMAT:
        errors.append(f"{receipt}: signed receipt format is invalid")

    manifest_claim = statement.get("pack_manifest_digest")
    if not _valid_digest(manifest_claim):
        errors.append(f"{receipt}: signed receipt manifest digest is invalid")
    else:
        try:
            manifest_digest = _sha256_bytes(manifest_path.read_bytes())
        except OSError as exc:
            errors.append(f"{manifest_path}: could not read pack manifest: {exc}")
        else:
            if manifest_claim != manifest_digest:
                errors.append(
                    f"{receipt}: signed receipt does not bind the pack manifest"
                )

    anchors = statement.get("anchors")
    if not isinstance(anchors, dict) or set(anchors) != {
        "policy_digest",
        "artifact_digests",
        "schedule_digest",
        "runtime_digests",
        "pack_signer_fingerprint",
    }:
        errors.append(f"{receipt}: signed receipt anchor fields are invalid")
    else:
        if not _valid_digest(anchors.get("policy_digest")):
            errors.append(f"{receipt}: signed receipt policy digest is invalid")
        if not _valid_digest(anchors.get("pack_signer_fingerprint")):
            errors.append(f"{receipt}: signed receipt pack signer is invalid")
        artifacts = anchors.get("artifact_digests")
        if (
            not isinstance(artifacts, dict)
            or set(artifacts) != {"baseline", "subject"}
            or any(not _valid_digest(digest) for digest in artifacts.values())
        ):
            errors.append(f"{receipt}: signed receipt artifact anchors are invalid")
        if not _valid_digest(anchors.get("schedule_digest")):
            errors.append(f"{receipt}: signed receipt schedule anchor is invalid")
        runtimes = anchors.get("runtime_digests")
        if (
            not isinstance(runtimes, dict)
            or set(runtimes) != {"baseline", "subject"}
            or any(not _valid_digest(digest) for digest in runtimes.values())
        ):
            errors.append(f"{receipt}: signed receipt runtime anchors are invalid")

    verifier = statement.get("verifier")
    if not isinstance(verifier, dict) or set(verifier) != {
        "identity",
        "signing_key_fingerprint",
    }:
        errors.append(f"{receipt}: signed receipt verifier fields are invalid")
        recorded_fingerprint = None
    else:
        identity = verifier.get("identity")
        if not isinstance(identity, str) or _IDENTITY.fullmatch(identity) is None:
            errors.append(f"{receipt}: signed receipt verifier identity is invalid")
        recorded_fingerprint = verifier.get("signing_key_fingerprint")
        if not _valid_digest(recorded_fingerprint):
            errors.append(f"{receipt}: signed receipt verifier fingerprint is invalid")

    verdict = statement.get("verdict")
    if not isinstance(verdict, dict) or set(verdict) != {
        "ok",
        "integrity_ok",
        "policy_verdict",
        "verification_status",
    }:
        errors.append(f"{receipt}: signed receipt verdict fields are invalid")
    else:
        ok = verdict.get("ok")
        integrity_ok = verdict.get("integrity_ok")
        policy_verdict = verdict.get("policy_verdict")
        status = verdict.get("verification_status")
        if not isinstance(ok, bool) or not isinstance(integrity_ok, bool):
            errors.append(f"{receipt}: signed receipt verdict booleans are invalid")
        if policy_verdict not in {"pass", "fail", None}:
            errors.append(f"{receipt}: signed receipt policy verdict is invalid")
        if isinstance(status, bool) or not isinstance(status, int) or status < 0:
            errors.append(f"{receipt}: signed receipt verification status is invalid")
        if not (
            ok is True
            and integrity_ok is True
            and policy_verdict == "pass"
            and status == 0
        ):
            errors.append(
                f"{receipt}: signed receipt must record successful strict acceptance"
            )

    if (
        not isinstance(signature, dict)
        or set(signature) != {"algorithm", "format", "public_key", "value"}
        or signature.get("format") != _RECEIPT_SIGNATURE_FORMAT
        or signature.get("algorithm") != "ed25519"
    ):
        errors.append(f"{receipt}: signed verification receipt signature is required")
        return

    public_key_block = signature.get("public_key")
    if (
        not isinstance(public_key_block, dict)
        or set(public_key_block)
        != {
            "encoding",
            "value",
        }
        or public_key_block.get("encoding") != "pem"
    ):
        errors.append(f"{receipt}: signed receipt public key is invalid")
        return
    public_key_value = public_key_block.get("value")
    public_key: ed25519.Ed25519PublicKey | None = None
    if not isinstance(public_key_value, str):
        errors.append(f"{receipt}: signed receipt public key is invalid")
    else:
        try:
            loaded = serialization.load_pem_public_key(public_key_value.encode("ascii"))
            if not isinstance(loaded, ed25519.Ed25519PublicKey):
                raise TypeError("public key is not Ed25519")
            public_key = loaded
        except (TypeError, UnicodeEncodeError, ValueError) as exc:
            errors.append(f"{receipt}: signed receipt public key is invalid: {exc}")
    if public_key is None:
        return

    if recorded_fingerprint != public_key_fingerprint(public_key):
        errors.append(
            f"{receipt}: signed receipt verifier fingerprint does not match its key"
        )
    try:
        encoded_signature = signature.get("value")
        if not isinstance(encoded_signature, str):
            raise ValueError("signature value is not text")
        signature_bytes = base64.b64decode(encoded_signature, validate=True)
        public_key.verify(signature_bytes, _canonical_json_bytes(statement))
    except (
        InvalidSignature,
        TypeError,
        ValueError,
        binascii.Error,
    ):
        errors.append(f"{receipt}: signed receipt signature verification failed")


def _safe_logical_path(value: object, *, prefix: str) -> bool:
    if not isinstance(value, str):
        return False
    path = PurePosixPath(value)
    return (
        value == path.as_posix()
        and not path.is_absolute()
        and ".." not in path.parts
        and value.startswith(prefix)
    )


def _safe_external_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and bool(parsed.hostname)
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )


def _check_artifact_summary(
    errors: list[str], entry: dict[str, Any], role: str, root: Path
) -> None:
    artifacts = entry.get("artifacts")
    summary = artifacts.get(role) if isinstance(artifacts, dict) else None
    if not isinstance(summary, dict):
        errors.append(f"{entry.get('slug', '<entry>')}: missing {role}")
        return
    logical = summary.get("path")
    slug = entry.get("slug")
    prefix = f"public_evidence/evidence/{slug}/"
    if not isinstance(slug, str) or not _safe_logical_path(logical, prefix=prefix):
        errors.append(f"{entry.get('slug', '<entry>')}: unsafe {role} path")
        return
    local = root.parent / str(logical)
    external = summary.get("external_asset")
    kind = summary.get("kind")
    common_fields = {"kind", "path", "size_bytes"}
    if kind == "file":
        required_fields = common_fields | {"sha256"}
    elif kind == "directory":
        required_fields = common_fields | {"file_count", "control_hashes"}
    else:
        errors.append(f"{logical}: artifact kind must be file or directory")
        return
    if set(summary) != required_fields | ({"external_asset"} if external else set()):
        errors.append(f"{logical}: artifact summary fields are not closed")
    size = summary.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        errors.append(f"{logical}: artifact size is invalid")
    if kind == "file" and _DIGEST.fullmatch(str(summary.get("sha256") or "")) is None:
        errors.append(f"{logical}: artifact digest is invalid")
    if kind == "directory":
        count = summary.get("file_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            errors.append(f"{logical}: artifact file count is invalid")
        controls = summary.get("control_hashes")
        if not isinstance(controls, dict) or any(
            name not in {"manifest.json", "manifest.signature.json", "checksums.sha256"}
            or _DIGEST.fullmatch(str(digest)) is None
            for name, digest in (controls.items() if isinstance(controls, dict) else ())
        ):
            errors.append(f"{logical}: artifact control hashes are invalid")
    if not local.exists() and not isinstance(external, dict):
        errors.append(f"{logical}: missing local artifact and external_asset")
    if isinstance(external, dict):
        if set(external) != {"sha256", "url"}:
            errors.append(f"{logical}: external asset fields are not closed")
        if not _safe_external_url(external.get("url")):
            errors.append(
                f"{logical}: external asset URL must be credential-free HTTPS "
                "without query or fragment"
            )
        if _DIGEST.fullmatch(str(external.get("sha256") or "")) is None:
            errors.append(f"{logical}: external asset digest is invalid")
        if local.exists():
            errors.append(f"{logical}: artifact must use one publication carrier")
    if local.is_symlink():
        errors.append(f"{logical}: symlinks are not allowed")
    elif local.exists():
        try:
            observed = _artifact_summary(local, source_root=root)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
        else:
            expected = {
                key: value for key, value in summary.items() if key != "external_asset"
            }
            if observed != expected:
                errors.append(f"{logical}: artifact summary does not match its bytes")


def _check_local_entry(errors: list[str], entry_root: Path) -> None:
    metadata_path = entry_root / "evidence.meta.json"
    if not metadata_path.is_file() or metadata_path.is_symlink():
        errors.append(f"{entry_root}: missing safe evidence.meta.json")
        return
    try:
        metadata = _read_object(metadata_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
        return
    try:
        artifact_paths, _summary = _validate_metadata(metadata_path, metadata)
    except ValueError as exc:
        errors.append(str(exc))
        return
    pack_name = artifact_paths.get("evidence_pack")
    receipt_name = artifact_paths.get("verification_receipt")
    assert isinstance(pack_name, str) and isinstance(receipt_name, str)
    pack = entry_root / pack_name
    receipt = entry_root / receipt_name
    if not pack.is_dir() or pack.is_symlink():
        errors.append(f"{pack}: evidence pack is missing or unsafe")
        return
    manifest_path = pack / "manifest.json"
    try:
        manifest = _read_object(manifest_path)
        receipt_value = _read_object(receipt)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
        return
    if manifest.get("format") != "evidence-pack-v1":
        errors.append(f"{pack}: only the canonical evidence-pack-v1 is publishable")
    _check_signed_receipt(
        errors,
        receipt=receipt,
        value=receipt_value,
        manifest_path=manifest_path,
    )
    statement = receipt_value.get("statement")
    anchors = statement.get("anchors") if isinstance(statement, dict) else None
    artifacts = anchors.get("artifact_digests") if isinstance(anchors, dict) else None
    schedule = anchors.get("schedule_digest") if isinstance(anchors, dict) else None
    runtimes = anchors.get("runtime_digests") if isinstance(anchors, dict) else None
    policy_digest = anchors.get("policy_digest") if isinstance(anchors, dict) else None
    signer = (
        anchors.get("pack_signer_fingerprint") if isinstance(anchors, dict) else None
    )
    if (
        isinstance(artifacts, dict)
        and set(artifacts) == {"baseline", "subject"}
        and all(_valid_digest(value) for value in artifacts.values())
        and _valid_digest(schedule)
        and isinstance(runtimes, dict)
        and set(runtimes) == {"baseline", "subject"}
        and all(_valid_digest(value) for value in runtimes.values())
        and _valid_digest(policy_digest)
        and _valid_digest(signer)
    ):
        inputs = manifest.get("inputs")
        expected_materials = {
            "baseline": artifacts["baseline"],
            "subject": artifacts["subject"],
            "dataset": schedule,
            "policy": policy_digest,
            "baseline_runtime": runtimes["baseline"],
            "subject_runtime": runtimes["subject"],
        }
        if not isinstance(inputs, dict):
            errors.append(f"{manifest_path}: manifest inputs are invalid")
        else:
            for role, expected_digest in expected_materials.items():
                reference = inputs.get(role)
                observed_digest = (
                    reference.get("material_digest")
                    if isinstance(reference, dict)
                    else None
                )
                if observed_digest != expected_digest:
                    errors.append(
                        f"{receipt}: signed receipt {role} anchor does not bind "
                        "the pack manifest"
                    )
        if manifest.get("signing_key_fingerprint") != signer:
            errors.append(
                f"{receipt}: signed receipt signer anchor does not bind the pack"
            )
        try:
            rendered = render_evidence(pack)
        except EvidenceReportError as exc:
            errors.append(f"{pack}: signed pack validation failed: {exc}")
        else:
            if rendered.evidence_signer != signer:
                errors.append(
                    f"{receipt}: signed receipt signer anchor does not match "
                    "the verified pack signer"
                )


def check_public_evidence(root: Path = SOURCE_ROOT) -> list[str]:
    errors: list[str] = []
    root = root.resolve()
    if not root.is_dir():
        return [f"public evidence root not found: {root}"]
    if not (root / "README.md").is_file():
        errors.append("public_evidence/README.md is required")
    index_path = root / INDEX_FILENAME
    if not index_path.is_file():
        errors.append(f"public_evidence/{INDEX_FILENAME} is required")
        return errors
    try:
        index = _read_object(index_path)
        _validate_index(index_path, index)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
        return errors
    if index.get("format_version") != INDEX_FORMAT_VERSION:
        errors.append("public evidence index format is invalid")
    encoded = index_path.read_text(encoding="utf-8")
    for marker in _PRIVATE_MARKERS:
        if marker in encoded:
            errors.append(f"public evidence index contains private marker {marker!r}")
    for marker in _OBSOLETE_MARKERS:
        if marker in encoded:
            errors.append(f"public evidence index contains obsolete marker {marker!r}")
    entries = index.get("entries")
    assert isinstance(entries, list)
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            errors.append("public evidence entry must be an object")
            continue
        slug = raw_entry.get("slug")
        if not isinstance(slug, str) or not slug or "/" in slug:
            errors.append("public evidence entry slug is invalid")
            continue
        if raw_entry.get("evidence_class") != "signed_evidence_pack":
            errors.append(f"{slug}: evidence_class must be signed_evidence_pack")
        expected_path = f"public_evidence/{EVIDENCE_DIRNAME}/{slug}"
        if raw_entry.get("path") != expected_path:
            errors.append(f"{slug}: entry path must be {expected_path}")
        _check_artifact_summary(errors, raw_entry, "evidence_pack", root)
        _check_artifact_summary(errors, raw_entry, "verification_receipt", root)
    artifact_file_count = 0
    artifact_size_bytes = 0
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            continue
        artifacts = raw_entry.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        for summary in artifacts.values():
            if not isinstance(summary, dict):
                continue
            count = 1 if summary.get("kind") == "file" else summary.get("file_count")
            size = summary.get("size_bytes")
            if isinstance(count, int) and not isinstance(count, bool):
                artifact_file_count += count
            if isinstance(size, int) and not isinstance(size, bool):
                artifact_size_bytes += size
    if index.get("evidence_file_count") != artifact_file_count:
        errors.append("evidence_file_count must match artifact summaries")
    if index.get("evidence_size_bytes") != artifact_size_bytes:
        errors.append("evidence_size_bytes must match artifact summaries")
    evidence_root = root / EVIDENCE_DIRNAME
    if evidence_root.is_dir():
        local_entries = sorted(
            path for path in evidence_root.iterdir() if path.is_dir()
        )
        indexed = {entry.get("slug") for entry in entries if isinstance(entry, dict)}
        if not {path.name for path in local_entries}.issubset(indexed):
            errors.append("every local evidence directory must appear in the index")
        for entry_root in local_entries:
            _check_local_entry(errors, entry_root)
        unexpected_children = sorted(
            item.name for item in evidence_root.iterdir() if not item.is_dir()
        )
        if unexpected_children:
            errors.append(
                "unexpected files in public evidence directory: "
                + ", ".join(unexpected_children)
            )
    allowed_root_files = {"README.md", INDEX_FILENAME}
    unexpected = sorted(
        item.name
        for item in root.iterdir()
        if item.name not in allowed_root_files and item.name != EVIDENCE_DIRNAME
    )
    if unexpected:
        errors.append("unexpected public evidence surfaces: " + ", ".join(unexpected))
    if root == SOURCE_ROOT.resolve():
        packaged_index = PACKAGED_ROOT / INDEX_FILENAME
        if (
            not packaged_index.is_file()
            or packaged_index.read_bytes() != index_path.read_bytes()
        ):
            errors.append("source and packaged public evidence indexes differ")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE_ROOT)
    args = parser.parse_args(argv)
    errors = check_public_evidence(args.root)
    if errors:
        for error in errors:
            print(error)
        return 1
    print("Public evidence audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
