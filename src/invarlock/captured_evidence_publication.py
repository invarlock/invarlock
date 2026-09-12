"""Atomic publication for the first captured-evaluation evidence handoff."""

from __future__ import annotations

import hashlib
import os
import secrets
from collections.abc import Iterator
from contextlib import contextmanager, suppress
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


@contextmanager
def _opened_directory(parent: int, name: str) -> Iterator[int]:
    """Own one descriptor for the complete surrounding publication scope."""
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent,
    )
    try:
        yield descriptor
    finally:
        try:
            os.close(descriptor)
        except OSError:
            # Never retry an uncertain close; the descriptor may be reused.
            pass


@contextmanager
def _opened_directories(parent: int) -> Iterator[dict[str, int]]:
    """Retain the fixed captured-evidence directories through publication."""
    with (
        _opened_directory(parent, "inputs") as inputs,
        _opened_directory(parent, "records") as records,
        _opened_directory(parent, "reports") as reports,
    ):
        yield {"inputs": inputs, "records": records, "reports": reports}


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


def _remove_empty_bound_directory(parent: int, name: str, descriptor: int) -> None:
    """Check retained identity before best-effort removal of an empty entry."""
    named = os.stat(name, dir_fd=parent, follow_symlinks=False)
    opened = os.fstat(descriptor)
    if (named.st_dev, named.st_ino) == (opened.st_dev, opened.st_ino):
        os.rmdir(name, dir_fd=parent)


def _cleanup_staging(
    parent: int,
    name: str,
    stage: int,
    directories: dict[str, int],
    files: dict[str, bytes],
) -> None:
    """Best-effort bounded cleanup through descriptors owned by this attempt.

    Never recursively traverse a staged pathname: it may now name another tree.
    Same-user mutation within the retained directories remains a trust boundary.
    """
    # A moved stage may already be published when a post-rename check fails.
    # Leave such trees intact; cleanup only uses the retained descriptors below.
    named = os.stat(name, dir_fd=parent, follow_symlinks=False)
    opened = os.fstat(stage)
    if (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
        return
    for filename in files:
        path = Path(filename)
        descriptor = (
            stage if path.parent == Path(".") else directories.get(str(path.parent))
        )
        if descriptor is not None:
            with suppress(OSError):
                os.unlink(path.name, dir_fd=descriptor)
    for directory, descriptor in directories.items():
        with suppress(OSError):
            _remove_empty_bound_directory(stage, directory, descriptor)
    with suppress(OSError):
        _remove_empty_bound_directory(parent, name, stage)


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
    retained_stage: int | None = None
    directories: dict[str, int] = {}
    cleanup_attempted = False
    published = False
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
                # The secure context verifies the original name on exit, before
                # publication renames it. Keep its opened inode through rename.
                retained_stage = os.dup(stage)
                for directory in ("inputs", "records", "reports"):
                    os.mkdir(directory, mode=0o700, dir_fd=stage)
        assert retained_stage is not None
        with _opened_directories(retained_stage) as directories:
            try:
                for name, payload in files.items():
                    path = Path(name)
                    output_parent = (
                        retained_stage
                        if path.parent == Path(".")
                        else directories[str(path.parent)]
                    )
                    descriptor = os.open(
                        path.name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=output_parent,
                    )
                    try:
                        handle = os.fdopen(descriptor, "wb")
                    except BaseException:
                        os.close(descriptor)
                        raise
                    with handle:
                        handle.write(payload)
                        handle.flush()
                        os.fchmod(handle.fileno(), 0o444)
                        os.fsync(handle.fileno())
                    os.fsync(output_parent)
                for directory, descriptor in directories.items():
                    named = os.stat(
                        directory, dir_fd=retained_stage, follow_symlinks=False
                    )
                    opened = os.fstat(descriptor)
                    if (named.st_dev, named.st_ino, named.st_mode) != (
                        opened.st_dev,
                        opened.st_ino,
                        opened.st_mode,
                    ):
                        raise CapturedEvidenceError(
                            "captured staging directory identity changed"
                        )
                publish_directory_no_replace(
                    staging,
                    destination,
                    expected_source_fd=retained_stage,
                    expected_source_parent_fd=cleanup,
                )
                published = True
            finally:
                cleanup_attempted = True
                if not published and cleanup is not None:
                    assert staging is not None
                    with suppress(OSError):
                        _cleanup_staging(
                            cleanup,
                            staging.name,
                            retained_stage,
                            directories,
                            files,
                        )
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
        if (
            not published
            and not cleanup_attempted
            and cleanup is not None
            and retained_stage is not None
        ):
            assert staging is not None
            with suppress(OSError):
                with _opened_directories(retained_stage) as directories:
                    _cleanup_staging(
                        cleanup, staging.name, retained_stage, directories, files
                    )
        for retained_descriptor in (retained_stage, cleanup):
            if retained_descriptor is not None:
                with suppress(OSError):
                    os.close(retained_descriptor)
    return destination


__all__ = ["CapturedEvidenceError", "publish_captured_evidence"]
