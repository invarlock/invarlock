"""Atomic publication for the first captured-evaluation evidence handoff."""

from __future__ import annotations

import hashlib
import os
import secrets
import shutil
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.captured_contracts import (
    CapturedContractError,
    CapturedSnapshot,
    check_sizes,
    checksums,
    load_payloads,
    manifest_signature,
    read_file,
    secure_directory,
    validate_contract,
)
from invarlock.captured_normalization import captured_comparison_id
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.filesystem import (
    AtomicDirectoryExistsError,
    AtomicDirectoryPublicationError,
    publish_directory_no_replace,
)


class CapturedEvidenceError(ValueError):
    """Raised when captured evidence cannot be safely published."""


def _private_key(path: Path) -> ed25519.Ed25519PrivateKey:
    try:
        key = serialization.load_pem_private_key(
            read_file(path, 65536),
            password=None,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise CapturedEvidenceError(f"could not load signing key: {exc}") from exc
    if not isinstance(key, ed25519.Ed25519PrivateKey):
        raise CapturedEvidenceError("captured evidence signing key must be Ed25519")
    return key


def _checksums(files: dict[str, bytes]) -> bytes:
    return checksums(files)


def _digest_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def publish_captured_evidence(
    destination: Path,
    *,
    baseline: dict[str, Any],
    subject: dict[str, Any],
    policy: dict[str, Any],
    comparison: dict[str, Any],
    request_digest: str,
    signing_key_path: Path | None,
    unsigned: bool,
    normalized_request: dict[str, Any] | None = None,
) -> Path:
    """Publish one fixed, no-clobber captured evidence directory."""
    if unsigned and signing_key_path is not None:
        raise CapturedEvidenceError("--unsigned cannot be combined with a signing key")
    if not unsigned and signing_key_path is None:
        raise CapturedEvidenceError(
            "captured evaluation requires a signing key unless unsigned mode is explicit"
        )

    baseline_bytes = canonical_json_bytes(baseline)
    subject_bytes = canonical_json_bytes(subject)
    policy_bytes = canonical_json_bytes(policy)
    comparison_bytes = canonical_json_bytes(comparison)
    request = normalized_request or {
        "format": "invarlock/evaluation-request-v2-normalized",
        "execution": {"mode": "captured"},
        "comparison": {"policy_digest": hashlib.sha256(policy_bytes).hexdigest()},
    }
    request_bytes = canonical_json_bytes(request)
    if request_digest != _digest_bytes(request_bytes):
        raise CapturedEvidenceError("request digest does not match normalized request")
    manifest: dict[str, Any] = {
        "format": "invarlock/evidence-pack-v2",
        "kind": "captured",
        "authentication": "unsigned_local" if unsigned else "signed",
        "comparison_id": captured_comparison_id(
            request_digest=request_digest,
            baseline_run_digest=_digest_bytes(baseline_bytes),
            subject_run_digest=_digest_bytes(subject_bytes),
            policy_digest=_digest_bytes(policy_bytes),
        ),
        "request_digest": request_digest,
        "signing_key_fingerprint": None,
        "files": {
            "request": {"path": "request.json", "digest": _digest_bytes(request_bytes)},
            "policy": {
                "path": "inputs/policy.json",
                "digest": _digest_bytes(policy_bytes),
            },
            "baseline": {
                "path": "records/baseline.json",
                "digest": _digest_bytes(baseline_bytes),
            },
            "subject": {
                "path": "records/subject.json",
                "digest": _digest_bytes(subject_bytes),
            },
            "report": {
                "path": "reports/evaluation.report.json",
                "digest": _digest_bytes(comparison_bytes),
            },
        },
        "checksums": "checksums.sha256",
    }
    files = {
        "request.json": request_bytes,
        "inputs/policy.json": policy_bytes,
        "records/baseline.json": baseline_bytes,
        "records/subject.json": subject_bytes,
        "reports/evaluation.report.json": comparison_bytes,
    }
    checksums = _checksums(files)
    manifest["checksums_sha256_digest"] = hashlib.sha256(checksums).hexdigest()
    manifest_bytes = canonical_json_bytes(manifest)
    signature: bytes | None = None
    if not unsigned:
        assert signing_key_path is not None
        key = _private_key(signing_key_path)
        manifest["signing_key_fingerprint"] = public_key_fingerprint(key.public_key())
        manifest["signature"] = "manifest.signature.json"
        manifest_bytes = canonical_json_bytes(manifest)
        signature = canonical_json_bytes(manifest_signature(manifest_bytes, key))
    files["checksums.sha256"] = checksums
    files["manifest.json"] = manifest_bytes
    if signature is not None:
        files["manifest.signature.json"] = signature

    destination = Path(destination).absolute()
    staging: Path | None = None
    cleanup: int | None = None
    try:
        check_sizes(files)
        validate_contract(manifest)
        load_payloads(CapturedSnapshot(files))
        with secure_directory(destination.parent, create=True) as parent:
            stage_name = ".captured-evidence-" + secrets.token_hex(16)
            cleanup = os.dup(parent)
            os.mkdir(stage_name, mode=0o700, dir_fd=parent)
            staging = destination.parent / stage_name
            with secure_directory(staging) as stage:
                for directory in ("inputs", "records", "reports"):
                    os.mkdir(directory, mode=0o700, dir_fd=stage)
                for name, payload in files.items():
                    path = Path(name)
                    with secure_directory(staging / path.parent) as output_parent:
                        descriptor = os.open(
                            path.name,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                            0o600,
                            dir_fd=output_parent,
                        )
                        with os.fdopen(descriptor, "wb") as handle:
                            handle.write(payload)
                            handle.flush()
                            os.fchmod(handle.fileno(), 0o444)
                            os.fsync(handle.fileno())
                        os.fsync(output_parent)
        publish_directory_no_replace(staging, destination)
    except AtomicDirectoryExistsError as exc:
        raise CapturedEvidenceError(
            f"captured evidence destination already exists: {destination}"
        ) from exc
    except (AtomicDirectoryPublicationError, CapturedContractError) as exc:
        raise CapturedEvidenceError(
            f"could not publish captured evidence: {exc}"
        ) from exc
    except CapturedEvidenceError:
        raise
    except OSError as exc:
        raise CapturedEvidenceError(
            f"could not publish captured evidence: {exc}"
        ) from exc
    finally:
        if cleanup is not None:
            try:
                if staging is not None:
                    shutil.rmtree(staging.name, dir_fd=cleanup)
            except FileNotFoundError:
                pass
            finally:
                os.close(cleanup)
    return destination


__all__ = ["CapturedEvidenceError", "publish_captured_evidence"]
