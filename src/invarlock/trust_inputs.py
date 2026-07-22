"""Closed caller-owned trust inputs for independent evidence verification."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from jsonschema import Draft202012Validator

from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
)
from invarlock.public_contracts import load_trust_inputs_schema

MAX_TRUST_INPUTS_BYTES = 256 * 1024
_MAX_POLICY_BYTES = 4 * 1024 * 1024
_MAX_SIGNING_KEY_BYTES = 64 * 1024
_CLOSE_ON_EXEC = getattr(os, "O_CLOEXEC", 0)


class TrustInputsError(ValueError):
    """Raised when a trust-input profile is malformed or unsafe."""


@dataclass(frozen=True)
class TrustInputs:
    """Resolved independent roots and the digest of their closed profile."""

    policy_path: Path
    policy_bytes: bytes = field(repr=False)
    expected_artifact_digests: Mapping[str, str]
    expected_schedule_digest: str
    expected_runtime_digests: Mapping[str, str]
    expected_signer_fingerprint: str
    expected_request_digest: str | None
    verifier_identity: str
    verifier_signing_key_path: Path
    verifier_signing_key_bytes: bytes = field(repr=False)
    allow_installed_scorers: bool
    profile_digest: str


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


def _directory_open_flags() -> int:
    directory = getattr(os, "O_DIRECTORY", None)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(directory, int) or not isinstance(nofollow, int):
        raise TrustInputsError(
            "secure descriptor-relative trust-input loading is unavailable"
        )
    return os.O_RDONLY | directory | nofollow | _CLOSE_ON_EXEC


def _file_open_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int):
        raise TrustInputsError(
            "secure descriptor-relative trust-input loading is unavailable"
        )
    return os.O_RDONLY | nofollow | _CLOSE_ON_EXEC


def _absolute_profile(path: Path) -> Path:
    """Return an absolute lexical path without resolving any symlink."""

    return Path(os.path.abspath(os.fspath(path)))


def _open_directory_without_links(path: Path, *, label: str) -> int:
    """Open an absolute directory without following any path component."""

    if not path.is_absolute():
        raise TrustInputsError(f"{label} parent path is invalid")
    flags = _directory_open_flags()
    try:
        descriptor = os.open(path.anchor, flags)
    except OSError as exc:
        raise TrustInputsError(
            f"{label} parent must be an existing non-symlink directory"
        ) from exc
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise TrustInputsError(
                    f"{label} parent must be an existing non-symlink directory"
                ) from exc
            os.close(descriptor)
            descriptor = child
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise TrustInputsError(
                f"{label} parent must be an existing non-symlink directory"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _safe_relative_parts(relative: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(relative, str):
        raise TrustInputsError(f"{label} path is invalid")
    parts = tuple(relative.split("/"))
    if relative.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise TrustInputsError(f"{label} path is unsafe")
    return parts


def _read_relative_regular_file(
    root_fd: int,
    parts: tuple[str, ...],
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    """Read one root-confined file through no-follow descriptor traversal."""

    current_fd = os.dup(root_fd)
    try:
        for index, component in enumerate(parts):
            final = index == len(parts) - 1
            flags = _file_open_flags() if final else _directory_open_flags()
            try:
                child_fd = os.open(component, flags, dir_fd=current_fd)
            except OSError as exc:
                raise TrustInputsError(
                    f"{label} could not be opened without following symlinks"
                ) from exc
            os.close(current_fd)
            current_fd = child_fd
        opened = os.fstat(current_fd)
        if not stat.S_ISREG(opened.st_mode):
            raise TrustInputsError(f"{label} must be a real regular file")
        if opened.st_size > max_bytes:
            raise TrustInputsError(f"{label} exceeds the {max_bytes}-byte size limit")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(current_fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > max_bytes:
            raise TrustInputsError(f"{label} exceeds the {max_bytes}-byte size limit")
        after = os.fstat(current_fd)
        identity = lambda value: (  # noqa: E731 - compact immutable stat projection
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(opened) != identity(after):
            raise TrustInputsError(f"{label} changed while being read")
        return payload
    finally:
        os.close(current_fd)


def load_trust_inputs(
    path: Path,
    *,
    verifier_key_bytes_override: bytes | None = None,
) -> TrustInputs:
    """Load a closed profile without following profile or referenced-file symlinks.

    ``verifier_key_bytes_override`` is reserved for replaying a completed signed
    receipt with separately captured public key bytes. The profile digest remains
    the digest of the original profile, including its signer-key path, while the
    caller remains responsible for authenticating the supplied override.
    """

    profile = _absolute_profile(Path(path))
    parent_fd = _open_directory_without_links(
        profile.parent,
        label="trust-input profile",
    )
    try:
        raw = _read_relative_regular_file(
            parent_fd,
            (profile.name,),
            label="trust-input profile",
            max_bytes=MAX_TRUST_INPUTS_BYTES,
        )
        payload = parse_json_bytes(raw, label="trust-input profile")
        if not isinstance(payload, dict):
            raise TrustInputsError("trust-input profile must decode to a JSON object")
        errors = sorted(
            Draft202012Validator(load_trust_inputs_schema()).iter_errors(payload),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
        if errors:
            first = errors[0]
            location = ".".join(str(part) for part in first.absolute_path) or "<root>"
            raise TrustInputsError(
                f"trust-input profile schema failed at {location}: {first.message}"
            )

        policy = payload["policy"]
        anchors = payload["anchors"]
        verifier = payload["verifier"]
        assert isinstance(policy, dict)
        assert isinstance(anchors, dict)
        assert isinstance(verifier, dict)
        policy_parts = _safe_relative_parts(policy["path"], label="policy")
        signing_key_parts = _safe_relative_parts(
            verifier["signing_key_path"],
            label="verifier signing key",
        )
        policy_bytes = _read_relative_regular_file(
            parent_fd,
            policy_parts,
            label="policy",
            max_bytes=_MAX_POLICY_BYTES,
        )
        if verifier_key_bytes_override is None:
            signing_key_bytes = _read_relative_regular_file(
                parent_fd,
                signing_key_parts,
                label="verifier signing key",
                max_bytes=_MAX_SIGNING_KEY_BYTES,
            )
        else:
            if not isinstance(verifier_key_bytes_override, bytes):
                raise TrustInputsError("verifier key override must be exact bytes")
            if len(verifier_key_bytes_override) > _MAX_SIGNING_KEY_BYTES:
                raise TrustInputsError(
                    "verifier key override exceeds the 65536-byte size limit"
                )
            signing_key_bytes = verifier_key_bytes_override
        canonical = _canonical_json_bytes(payload)
        return TrustInputs(
            policy_path=profile.parent.joinpath(*policy_parts),
            policy_bytes=policy_bytes,
            expected_artifact_digests=MappingProxyType(
                {
                    "baseline": str(anchors["baseline_artifact_digest"]),
                    "subject": str(anchors["subject_artifact_digest"]),
                }
            ),
            expected_schedule_digest=str(anchors["schedule_digest"]),
            expected_runtime_digests=MappingProxyType(
                {
                    "baseline": str(anchors["baseline_runtime_digest"]),
                    "subject": str(anchors["subject_runtime_digest"]),
                }
            ),
            expected_signer_fingerprint=str(anchors["evidence_signer_fingerprint"]),
            expected_request_digest=(
                str(anchors["request_digest"]) if "request_digest" in anchors else None
            ),
            verifier_identity=str(verifier["identity"]),
            verifier_signing_key_path=profile.parent.joinpath(*signing_key_parts),
            verifier_signing_key_bytes=signing_key_bytes,
            allow_installed_scorers=bool(payload["allow_installed_scorers"]),
            profile_digest=f"sha256:{hashlib.sha256(canonical).hexdigest()}",
        )
    except StrictJsonError as exc:
        raise TrustInputsError(str(exc)) from exc
    finally:
        os.close(parent_fd)


__all__ = [
    "MAX_TRUST_INPUTS_BYTES",
    "TrustInputs",
    "TrustInputsError",
    "load_trust_inputs",
]
