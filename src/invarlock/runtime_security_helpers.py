"""Small fail-closed runtime identity primitives for InvarLock evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from invarlock.evidence_pack_json import read_regular_file_bytes
from invarlock.public_contracts import RUNTIME_MANIFEST_CONTRACT_VERSION

ALLOW_NETWORK_ENV = "INVARLOCK_ALLOW_NETWORK"
ALLOW_REMOTE_CODE_ENV = "INVARLOCK_ALLOW_REMOTE_CODE"
ALLOW_THIRD_PARTY_PLUGINS_ENV = "INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS"
CONTAINER_EXECUTION_ENV = "INVARLOCK_CONTAINER_EXECUTION"
RUNTIME_IMAGE_ENV = "INVARLOCK_RUNTIME_IMAGE"
RUNTIME_IMAGE_DIGEST_ENV = "INVARLOCK_RUNTIME_IMAGE_DIGEST"
RUNTIME_MANIFEST_FILENAME = "runtime.manifest.json"
RUNTIME_MANIFEST_VERSION = 1
RUNTIME_VERIFIER_CONTRACT_VERSION = RUNTIME_MANIFEST_CONTRACT_VERSION
RUNTIME_IMAGE_DEFAULT = "ghcr.io/invarlock/invarlock-runtime:latest"

_SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimeManifestExecution:
    """Authenticated execution facts recorded into a runtime manifest."""

    execution_mode: str
    container_execution: bool
    image_ref: str
    image_digest: str | None
    allow_network: bool
    allow_remote_code: bool
    allow_third_party_plugins: bool


@dataclass(frozen=True)
class RuntimeProviderManifestFiles:
    """Sibling provider evidence files bound by a runtime manifest."""

    receipt: Path
    scoring_observation: Path
    artifact_identity: Path


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def network_allowed() -> bool:
    return _enabled(ALLOW_NETWORK_ENV)


def remote_code_allowed() -> bool:
    return _enabled(ALLOW_REMOTE_CODE_ENV)


def third_party_plugins_allowed() -> bool:
    return _enabled(ALLOW_THIRD_PARTY_PLUGINS_ENV)


def running_inside_container() -> bool:
    return _enabled(CONTAINER_EXECUTION_ENV)


def _regular_file_marker_present(path: str) -> bool:
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


def _read_bounded_kernel_file(path: str, *, max_bytes: int = 16 * 1024) -> bytes | None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None
        payload = os.read(descriptor, max_bytes + 1)
        return payload if len(payload) <= max_bytes else None
    except OSError:
        return None
    finally:
        os.close(descriptor)


def strict_container_boundary_present() -> bool:
    """Require explicit InvarLock intent and kernel-visible container evidence."""

    if not running_inside_container():
        return False
    if any(
        _regular_file_marker_present(path)
        for path in ("/.dockerenv", "/run/.containerenv")
    ):
        return True
    cgroup = _read_bounded_kernel_file("/proc/1/cgroup")
    return cgroup is not None and any(
        marker in cgroup.lower()
        for marker in (b"docker", b"containerd", b"kubepods", b"libpod")
    )


def current_execution_mode() -> str:
    return "container" if running_inside_container() else "host"


def _declared_runtime_image_digest(image_ref: str) -> str | None:
    explicit_raw = os.environ.get(RUNTIME_IMAGE_DIGEST_ENV, "").strip()
    embedded_raw = image_ref.rsplit("@", 1)[1] if "@" in image_ref else ""
    for label, value in (
        (RUNTIME_IMAGE_DIGEST_ENV, explicit_raw),
        ("runtime image reference", embedded_raw),
    ):
        if value and not _SHA256_DIGEST_RE.fullmatch(value):
            raise RuntimeError(f"{label} must use lowercase sha256:<64 hex>")
    if explicit_raw and embedded_raw and explicit_raw != embedded_raw:
        raise RuntimeError(
            "declared runtime image digest does not match the image reference"
        )
    return explicit_raw or embedded_raw or None


def resolve_runtime_image_digest() -> str | None:
    image_ref = os.environ.get(RUNTIME_IMAGE_ENV, "").strip()
    return _declared_runtime_image_digest(image_ref)


def resolve_runtime_image() -> str:
    """Return a digest-bearing image reference whenever a digest is declared."""

    image_ref = os.environ.get(RUNTIME_IMAGE_ENV, "").strip() or RUNTIME_IMAGE_DEFAULT
    digest = _declared_runtime_image_digest(image_ref)
    if digest is None or "@" in image_ref or image_ref == digest:
        return image_ref
    return f"{image_ref}@{digest}"


def _runtime_provenance_image_ref(image_ref: str, image_digest: str | None) -> str:
    if image_digest is None or not _SHA256_DIGEST_RE.fullmatch(image_digest):
        raise RuntimeError("runtime provenance requires a lowercase image digest")
    if (
        not image_ref
        or image_ref != image_ref.strip()
        or image_ref.startswith(("/", "\\"))
        or (len(image_ref) >= 2 and image_ref[0].isalpha() and image_ref[1] == ":")
        or any(ord(character) < 32 for character in image_ref)
        or image_ref.count("@") > 1
    ):
        raise RuntimeError("runtime image reference must be a portable reference")
    embedded = image_ref.rsplit("@", 1)[1] if "@" in image_ref else None
    if embedded is not None and not image_ref.split("@", 1)[0]:
        raise RuntimeError("runtime image reference must name an image")
    if embedded is not None and not _SHA256_DIGEST_RE.fullmatch(embedded):
        raise RuntimeError("runtime image reference contains an invalid digest")
    if embedded is not None and embedded != image_digest:
        raise RuntimeError("runtime image reference and digest do not agree")
    if image_ref == image_digest or "@" in image_ref:
        return image_ref
    return f"{image_ref}@{image_digest}"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"runtime JSON value has unsupported type {type(value).__name__}")


def serialize_canonical_json(payload: Any) -> str:
    return json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(read_regular_file_bytes(path, label="runtime manifest input"))


__all__ = [
    "ALLOW_NETWORK_ENV",
    "ALLOW_REMOTE_CODE_ENV",
    "ALLOW_THIRD_PARTY_PLUGINS_ENV",
    "CONTAINER_EXECUTION_ENV",
    "RUNTIME_IMAGE_DIGEST_ENV",
    "RUNTIME_IMAGE_ENV",
    "RUNTIME_MANIFEST_FILENAME",
    "RUNTIME_MANIFEST_VERSION",
    "RUNTIME_VERIFIER_CONTRACT_VERSION",
    "RuntimeManifestExecution",
    "RuntimeProviderManifestFiles",
    "current_execution_mode",
    "network_allowed",
    "remote_code_allowed",
    "resolve_runtime_image",
    "resolve_runtime_image_digest",
    "running_inside_container",
    "strict_container_boundary_present",
    "third_party_plugins_allowed",
]
