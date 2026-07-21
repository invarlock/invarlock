"""Cryptographic and filesystem integrity for the canonical evidence pack."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock import evidence_pack_json as evidence_pack_json_mod

MANIFEST_SIGNATURE_FILENAME = "manifest.signature.json"
EVIDENCE_PACK_SIGNATURE_FORMAT = "invarlock/evidence-pack-signature-v1"
CONTROL_FILES = {
    "checksums.sha256",
    "manifest.json",
    MANIFEST_SIGNATURE_FILENAME,
}
CHECKSUM_LINE_RE = re.compile(r"^([A-Fa-f0-9]{64}) [ *](.+)$")
SIGNING_KEY_FINGERPRINT_RE = re.compile(r"sha256:[a-f0-9]{64}")


def _load_json(path: Path) -> Any:
    return evidence_pack_json_mod.load_json(path, label="JSON input")


def _json_load_error_types() -> tuple[type[BaseException], ...]:
    return (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        evidence_pack_json_mod.StrictJsonError,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path_hex(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _normalize_pack_path(pack_dir: Path, rel_path: str) -> Path | None:
    """Resolve one portable pack path without accepting links or traversal."""

    if pack_dir.is_symlink() or not rel_path or "\\" in rel_path:
        return None
    parts = rel_path.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        return None
    current = pack_dir
    for index, part in enumerate(parts):
        current = current / part
        try:
            current.lstat()
        except OSError:
            break
        if current.is_symlink():
            return None
        if index < len(parts) - 1 and not current.is_dir():
            return None
    candidate = pack_dir.joinpath(*parts).resolve()
    try:
        candidate.relative_to(pack_dir.resolve())
    except ValueError:
        return None
    return candidate


def public_key_fingerprint(public_key: ed25519.Ed25519PublicKey) -> str:
    key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return f"sha256:{hashlib.sha256(key_bytes).hexdigest()}"


def normalize_expected_fingerprint(value: str | None) -> str | None:
    """Normalize a caller-pinned signing key fingerprint."""

    if value is None:
        return None
    normalized = value.strip().lower()
    if SIGNING_KEY_FINGERPRINT_RE.fullmatch(normalized) is None:
        return None
    return normalized


def verify_manifest_binds_checksums_payload(
    payload: Any, checksums_payload: bytes
) -> list[str]:
    if not isinstance(payload, dict):
        return ["manifest must decode to a JSON object"]
    expected = payload.get("checksums_sha256_digest")
    if not isinstance(expected, str) or not expected:
        return ["manifest.json checksums_sha256_digest is missing or empty"]
    actual = _sha256_bytes(checksums_payload)
    if expected != actual:
        return [f"checksums.sha256 digest mismatch (expected {expected}, got {actual})"]
    return []


def canonicalize_checksum_path(rel_path: str) -> str:
    canonical = rel_path.replace("\\", "/")
    while canonical.startswith("./"):
        canonical = canonical[2:]
    return canonical


def parse_checksums(pack_dir: Path) -> tuple[list[tuple[str, str]], list[str]]:
    entries: list[tuple[str, str]] = []
    errors: list[str] = []
    try:
        raw = evidence_pack_json_mod.read_regular_file_bytes(
            pack_dir / "checksums.sha256",
            label="checksums.sha256",
            max_bytes=1024 * 1024,
        )
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError, evidence_pack_json_mod.StrictJsonError) as exc:
        return [], [f"checksums.sha256 could not be read safely: {exc}"]
    seen_paths: set[str] = set()
    for index, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line:
            continue
        match = CHECKSUM_LINE_RE.fullmatch(raw_line)
        if match is None:
            errors.append(f"checksums.sha256 line {index} is not a valid sha256 entry")
            continue
        digest, rel_path = match.groups()
        canonical_path = canonicalize_checksum_path(rel_path)
        if canonical_path in seen_paths:
            errors.append(
                f"checksums.sha256 line {index} duplicates path {canonical_path!r}"
            )
        seen_paths.add(canonical_path)
        entries.append((digest.lower(), rel_path))
    return entries, errors


def verify_checksums(pack_dir: Path) -> tuple[list[str], set[str]]:
    entries, errors = parse_checksums(pack_dir)
    covered_paths: set[str] = set()
    for digest, rel_path in entries:
        canonical_rel_path = canonicalize_checksum_path(rel_path)
        covered_paths.add(canonical_rel_path)
        resolved = _normalize_pack_path(pack_dir, canonical_rel_path)
        if resolved is None:
            errors.append(f"checksums entry escapes the pack root: {rel_path}")
            continue
        if not resolved.is_file():
            errors.append(f"checksums entry missing from pack: {rel_path}")
            continue
        actual = _sha256_path_hex(resolved)
        if actual != digest:
            errors.append(
                f"checksum mismatch for {rel_path} (expected {digest}, got {actual})"
            )
    return errors, covered_paths


def verify_no_extra_files(
    pack_dir: Path, *, covered_paths: set[str], strict: bool
) -> tuple[list[str], list[str]]:
    actual_paths = {
        path.relative_to(pack_dir).as_posix()
        for path in pack_dir.rglob("*")
        if path.is_file()
    }
    extras = sorted(actual_paths - covered_paths - CONTROL_FILES)
    if not extras:
        return [], []
    message = "Pack contains extra files not covered by checksums.sha256: " + ", ".join(
        extras
    )
    return ([message], []) if strict else ([], [message])


def _load_signature_bundle(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        payload = _load_json(path)
    except _json_load_error_types() as exc:
        return None, [f"{MANIFEST_SIGNATURE_FILENAME} is not valid JSON: {exc}"]
    if not isinstance(payload, dict):
        return None, [f"{MANIFEST_SIGNATURE_FILENAME} must decode to a JSON object"]
    if set(payload) != {
        "format",
        "algorithm",
        "signing_key_fingerprint",
        "public_key",
        "signature",
    }:
        return None, [f"{MANIFEST_SIGNATURE_FILENAME} fields are invalid"]
    errors: list[str] = []
    if payload.get("format") != EVIDENCE_PACK_SIGNATURE_FORMAT:
        errors.append(
            f"{MANIFEST_SIGNATURE_FILENAME} format must be "
            f"{EVIDENCE_PACK_SIGNATURE_FORMAT!r}"
        )
    if payload.get("algorithm") != "ed25519":
        errors.append(f"{MANIFEST_SIGNATURE_FILENAME} algorithm must be 'ed25519'")
    public_key = payload.get("public_key")
    if not isinstance(public_key, dict) or set(public_key) != {"encoding", "value"}:
        errors.append(f"{MANIFEST_SIGNATURE_FILENAME} public_key is invalid")
    elif public_key.get("encoding") != "pem" or not isinstance(
        public_key.get("value"), str
    ):
        errors.append(f"{MANIFEST_SIGNATURE_FILENAME} public_key is invalid")
    signature = payload.get("signature")
    if not isinstance(signature, dict) or set(signature) != {"encoding", "value"}:
        errors.append(f"{MANIFEST_SIGNATURE_FILENAME} signature is invalid")
    elif signature.get("encoding") != "base64" or not isinstance(
        signature.get("value"), str
    ):
        errors.append(f"{MANIFEST_SIGNATURE_FILENAME} signature is invalid")
    fingerprint = payload.get("signing_key_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or SIGNING_KEY_FINGERPRINT_RE.fullmatch(fingerprint) is None
    ):
        errors.append(
            f"{MANIFEST_SIGNATURE_FILENAME} signing_key_fingerprint is invalid"
        )
    return (payload, []) if not errors else (None, errors)


def verify_signature(
    pack_dir: Path,
    *,
    strict: bool,
    load_json_fn: Any = _load_json,
    expected_fingerprints: set[str] | frozenset[str] | None = None,
) -> tuple[list[str], list[str], str | None]:
    signature_path = pack_dir / MANIFEST_SIGNATURE_FILENAME
    if not signature_path.exists():
        message = f"{MANIFEST_SIGNATURE_FILENAME} missing"
        return ([message], [], None) if strict else ([], [message], None)
    if signature_path.is_symlink() or not signature_path.is_file():
        return [f"{MANIFEST_SIGNATURE_FILENAME} must be a regular file"], [], None
    bundle, errors = _load_signature_bundle(signature_path)
    if errors or bundle is None:
        return errors, [], None
    try:
        public_key_value = bundle["public_key"]["value"]
        public_key_obj = serialization.load_pem_public_key(
            public_key_value.encode("ascii")
        )
    except (AttributeError, TypeError, ValueError) as exc:
        return [f"manifest signature verification failed: {exc}"], [], None
    if not isinstance(public_key_obj, ed25519.Ed25519PublicKey):
        return ["manifest signature public key must be Ed25519"], [], None
    fingerprint = public_key_fingerprint(public_key_obj)
    if bundle["signing_key_fingerprint"] != fingerprint:
        return (
            ["manifest signature fingerprint does not match its public key"],
            [],
            fingerprint,
        )
    try:
        signature_bytes = base64.b64decode(bundle["signature"]["value"], validate=True)
        manifest_bytes = evidence_pack_json_mod.read_regular_file_bytes(
            pack_dir / "manifest.json", label="manifest.json", max_bytes=256 * 1024
        )
        public_key_obj.verify(signature_bytes, manifest_bytes)
    except (TypeError, ValueError, evidence_pack_json_mod.StrictJsonError) as exc:
        return [f"manifest signature verification failed: {exc}"], [], fingerprint
    except InvalidSignature:
        return ["manifest signature verification failed"], [], fingerprint
    try:
        manifest = (
            evidence_pack_json_mod.parse_json_bytes(
                manifest_bytes, label="manifest.json"
            )
            if load_json_fn is _load_json
            else load_json_fn(pack_dir / "manifest.json")
        )
    except _json_load_error_types():
        manifest = {}
    recorded = (
        manifest.get("signing_key_fingerprint") if isinstance(manifest, dict) else None
    )
    if recorded != fingerprint:
        return ["manifest signing key does not match its signature"], [], fingerprint
    if expected_fingerprints is not None and fingerprint not in expected_fingerprints:
        expected = ", ".join(sorted(expected_fingerprints))
        return (
            [
                "manifest signature signer mismatch: "
                f"expected one of [{expected}], got {fingerprint}"
            ],
            [],
            fingerprint,
        )
    return [], [], fingerprint


__all__ = [
    "CONTROL_FILES",
    "EVIDENCE_PACK_SIGNATURE_FORMAT",
    "MANIFEST_SIGNATURE_FILENAME",
    "SIGNING_KEY_FINGERPRINT_RE",
    "_normalize_pack_path",
    "canonicalize_checksum_path",
    "normalize_expected_fingerprint",
    "parse_checksums",
    "public_key_fingerprint",
    "verify_checksums",
    "verify_manifest_binds_checksums_payload",
    "verify_no_extra_files",
    "verify_signature",
]
