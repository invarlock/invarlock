"""Caller-owned trust material for the maintained integration examples."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.evidence_pack_integrity import public_key_fingerprint

_MAX_TRUST_FILE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class TrustMaterial:
    """Resolved paths and fingerprints for one caller-owned trust root."""

    evidence_key: Path
    verifier_key: Path
    trust_root: Path
    trusted_inputs: Path
    independent_policy: Path
    evidence_fingerprint: str
    verifier_fingerprint: str


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _outside(root: Path, candidate: Path, *, label: str) -> None:
    try:
        candidate.relative_to(root)
    except ValueError:
        return
    raise ValueError(f"{label} must remain outside the transaction workspace")


def _directory_flags() -> int:
    directory = getattr(os, "O_DIRECTORY", None)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(directory, int) or not isinstance(nofollow, int):
        raise ValueError("secure trust-material directory access is unavailable")
    return os.O_RDONLY | directory | nofollow | getattr(os, "O_CLOEXEC", 0)


def _open_directory(path: Path, *, label: str) -> int:
    """Open every directory component without following a symlink."""

    absolute = _absolute(path)
    flags = _directory_flags()
    descriptor: int | None = None
    try:
        descriptor = os.open(absolute.anchor, flags)
        relative = absolute.relative_to(Path(absolute.anchor))
        for component in relative.parts:
            if component in {"", ".", ".."}:
                raise ValueError(f"{label} contains an unsafe directory component")
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except (OSError, ValueError) as exc:
        if descriptor is not None:
            os.close(descriptor)
        if isinstance(exc, ValueError):
            raise
        raise ValueError(
            f"{label} must be an existing directory without symlinks"
        ) from exc


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_file_at(parent: int, name: str, *, label: str) -> bytes:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int) or name in {"", ".", ".."} or "/" in name:
        raise ValueError(f"{label} has an unsafe path")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent)
    except OSError as exc:
        raise ValueError(
            f"{label} could not be opened without following links"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor, min(1024 * 1024, _MAX_TRUST_FILE_BYTES + 1 - total)
            )
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_TRUST_FILE_BYTES:
                raise ValueError(f"{label} is too large")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise ValueError(f"{label} changed while being read")
        return b"".join(chunks)
    except OSError as exc:
        raise ValueError(f"{label} could not be read") from exc
    finally:
        os.close(descriptor)


def read_external_file(path: Path, *, label: str) -> bytes:
    """Read a caller-owned regular file without following directory links."""

    candidate = _absolute(path)
    parent = _open_directory(candidate.parent, label=f"{label} parent")
    try:
        return _read_file_at(parent, candidate.name, label=label)
    finally:
        os.close(parent)


def _write_new_file_at(
    parent: int,
    name: str,
    payload: bytes,
    *,
    mode: int,
    final_mode: int | None = None,
) -> None:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int) or name in {"", ".", ".."} or "/" in name:
        raise ValueError("trust-material file path is unsafe")
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(name, flags, mode, dir_fd=parent)
    except OSError as exc:
        raise ValueError(
            "trust-material file could not be created exclusively"
        ) from exc
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, mode if final_mode is None else final_mode)
    except OSError as exc:
        raise ValueError("trust-material file could not be written securely") from exc
    finally:
        os.close(descriptor)


def load_external_key(
    path: Path,
    *,
    transaction_root: Path,
    label: str,
) -> tuple[Path, bytes, str]:
    """Load one caller-owned Ed25519 private key without accepting symlinks."""

    candidate = _absolute(path)
    transaction = _absolute(transaction_root)
    _outside(transaction, candidate, label=label)
    transaction_descriptor = _open_directory(transaction, label="transaction workspace")
    os.close(transaction_descriptor)
    try:
        payload = read_external_file(candidate, label=label)
        key = serialization.load_pem_private_key(payload, password=None)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain an Ed25519 private key") from exc
    if not isinstance(key, ed25519.Ed25519PrivateKey):
        raise ValueError(f"{label} must contain an Ed25519 private key")
    return candidate, payload, public_key_fingerprint(key.public_key())


def validate_new_trust_root(path: Path, *, transaction_root: Path) -> Path:
    """Validate a caller-owned trust root before doing transaction work."""

    transaction = _absolute(transaction_root)
    root = _absolute(path)
    _outside(transaction, root, label="trust root")
    parent = root.parent
    transaction_descriptor = _open_directory(transaction, label="transaction workspace")
    os.close(transaction_descriptor)
    _outside(transaction, parent, label="trust root parent")
    parent_descriptor = _open_directory(parent, label="trust root parent")
    try:
        try:
            os.stat(root.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ValueError(
                "trust root must be new and outside the transaction"
            ) from exc
        else:
            raise ValueError("trust root must be new and outside the transaction")
    finally:
        os.close(parent_descriptor)
    return root


def create_trust_material(
    *,
    transaction_root: Path,
    evidence_key: Path,
    verifier_key_bytes: bytes,
    evidence_fingerprint: str,
    verifier_fingerprint: str,
    trust_root: Path,
    policy_bytes: bytes,
    verifier_identity: str,
    anchors: Mapping[str, str],
) -> TrustMaterial:
    """Create a new trust-root directory from caller-owned key material."""

    root = validate_new_trust_root(trust_root, transaction_root=transaction_root)

    trust_payload = (
        json.dumps(
            {
                "format": "invarlock/trust-inputs-v1",
                "policy": {"path": "policy/acceptance.json"},
                "anchors": dict(anchors),
                "verifier": {
                    "identity": verifier_identity,
                    "signing_key_path": "verifier.pem",
                },
                "allow_installed_scorers": False,
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    parent_descriptor = _open_directory(root.parent, label="trust root parent")
    root_descriptor: int | None = None
    policy_descriptor: int | None = None
    try:
        try:
            os.mkdir(root.name, 0o700, dir_fd=parent_descriptor)
        except OSError as exc:
            raise ValueError("trust root could not be created exclusively") from exc
        root_descriptor = os.open(
            root.name, _directory_flags(), dir_fd=parent_descriptor
        )
        os.mkdir("policy", 0o700, dir_fd=root_descriptor)
        policy_descriptor = os.open(
            "policy", _directory_flags(), dir_fd=root_descriptor
        )
        _write_new_file_at(
            root_descriptor, "verifier.pem", verifier_key_bytes, mode=0o600
        )
        _write_new_file_at(
            policy_descriptor,
            "acceptance.json",
            policy_bytes,
            mode=0o600,
            final_mode=0o444,
        )
        _write_new_file_at(
            root_descriptor,
            "trusted-inputs.json",
            trust_payload,
            mode=0o600,
            final_mode=0o444,
        )
    except OSError as exc:
        raise ValueError("trust root could not be created securely") from exc
    finally:
        if policy_descriptor is not None:
            os.close(policy_descriptor)
        if root_descriptor is not None:
            os.close(root_descriptor)
        os.close(parent_descriptor)
    verifier_path = root / "verifier.pem"
    policy_path = root / "policy/acceptance.json"
    trust_path = root / "trusted-inputs.json"
    return TrustMaterial(
        evidence_key=evidence_key,
        verifier_key=verifier_path,
        trust_root=root,
        trusted_inputs=trust_path,
        independent_policy=policy_path,
        evidence_fingerprint=evidence_fingerprint,
        verifier_fingerprint=verifier_fingerprint,
    )


__all__ = [
    "TrustMaterial",
    "create_trust_material",
    "load_external_key",
    "read_external_file",
    "validate_new_trust_root",
]
