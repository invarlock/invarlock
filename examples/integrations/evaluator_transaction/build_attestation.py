"""Example-owned authenticated build contracts for evaluator transactions.

These contracts support the maintained evaluator demonstrations. They are not
part of InvarLock's installed evaluator-neutral API.
"""

from __future__ import annotations

import base64
import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)

EVALUATOR_BUILD_ATTESTATION_FORMAT = "invarlock/evaluator-build-attestation-v1"
SIGNED_EVALUATOR_BUILD_ATTESTATION_FORMAT = (
    "invarlock/signed-evaluator-build-attestation-v1"
)
_RETAINED_BUILD_ATTESTATION_FORMATS = {
    SIGNED_EVALUATOR_BUILD_ATTESTATION_FORMAT: EVALUATOR_BUILD_ATTESTATION_FORMAT,
    "invarlock/signed-level3-build-attestation-v1": (
        "invarlock/level3-build-attestation-v2"
    ),
}
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_MAX_ATTESTATION_BYTES = 1024 * 1024


class EvaluatorBuildAttestationError(ValueError):
    """Raised when evaluator-demo build provenance is missing or inconsistent."""


def _digest_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluatorBuildAttestationError(f"{label} must be a non-empty string")
    return value


def _digest(value: object, *, label: str) -> str:
    rendered = _string(value, label=label)
    if _DIGEST_RE.fullmatch(rendered) is None:
        raise EvaluatorBuildAttestationError(f"{label} must be a sha256 digest")
    return rendered


def _layers(value: object, *, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(layer, str) or _DIGEST_RE.fullmatch(layer) is None
            for layer in value
        )
    ):
        raise EvaluatorBuildAttestationError(
            f"{label} must contain sha256 layer digests"
        )
    return list(value)


def _entrypoint(value: object) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(part, str) or not part or "\x00" in part for part in value
        )
    ):
        raise EvaluatorBuildAttestationError(
            "entrypoint must be a non-empty string list"
        )
    return list(value)


def _validate_evaluator_build_attestation(
    value: object,
    *,
    expected_format: str,
    evaluator: str,
    evaluator_version: str,
    runtime_image_id: str,
    base_image_id: str,
    source_commit: str,
    source_bundle_sha256: str,
    lock_sha256: str,
    entrypoint: Sequence[str],
) -> dict[str, Any]:
    """Validate one strict build statement against the requested profile."""

    if not isinstance(value, dict):
        raise EvaluatorBuildAttestationError(
            "evaluator build attestation must be an object"
        )
    expected_fields = {
        "base_image_id",
        "base_layers",
        "config_sha256",
        "entrypoint",
        "evaluator",
        "evaluator_version",
        "format_version",
        "image_layers",
        "lock_sha256",
        "runtime_image_id",
        "source_bundle_sha256",
        "source_commit",
    }
    if set(value) != expected_fields:
        raise EvaluatorBuildAttestationError(
            "evaluator build attestation has unexpected fields"
        )
    if value.get("format_version") != expected_format:
        raise EvaluatorBuildAttestationError(
            "evaluator build attestation format is invalid"
        )
    if value.get("evaluator") != evaluator:
        raise EvaluatorBuildAttestationError(
            "evaluator build attestation evaluator is invalid"
        )
    if value.get("evaluator_version") != evaluator_version:
        raise EvaluatorBuildAttestationError(
            "evaluator build attestation evaluator version is invalid"
        )
    for actual, expected, label in (
        (value.get("runtime_image_id"), runtime_image_id, "runtime image identity"),
        (value.get("base_image_id"), base_image_id, "base image identity"),
        (
            value.get("source_bundle_sha256"),
            source_bundle_sha256,
            "source bundle digest",
        ),
        (value.get("lock_sha256"), lock_sha256, "evaluator lock digest"),
    ):
        if actual != expected:
            raise EvaluatorBuildAttestationError(
                f"evaluator build attestation {label} does not match the request"
            )
    if _DIGEST_RE.fullmatch(runtime_image_id) is None:
        raise EvaluatorBuildAttestationError("runtime image identity is invalid")
    if _DIGEST_RE.fullmatch(base_image_id) is None:
        raise EvaluatorBuildAttestationError("base image identity is invalid")
    if _DIGEST_RE.fullmatch(source_bundle_sha256) is None:
        raise EvaluatorBuildAttestationError("source bundle digest is invalid")
    if _DIGEST_RE.fullmatch(lock_sha256) is None:
        raise EvaluatorBuildAttestationError("evaluator lock digest is invalid")
    if (
        _COMMIT_RE.fullmatch(source_commit) is None
        or value.get("source_commit") != source_commit
    ):
        raise EvaluatorBuildAttestationError("source commit is invalid")
    actual_entrypoint = _entrypoint(value.get("entrypoint"))
    if actual_entrypoint != list(entrypoint):
        raise EvaluatorBuildAttestationError(
            "evaluator entrypoint does not match the profile"
        )
    _layers(value.get("base_layers"), label="base layer chain")
    _layers(value.get("image_layers"), label="image layer chain")
    _digest(value.get("config_sha256"), label="OCI config digest")
    return dict(value)


def validate_evaluator_build_attestation(
    value: object,
    *,
    evaluator: str,
    evaluator_version: str,
    runtime_image_id: str,
    base_image_id: str,
    source_commit: str,
    source_bundle_sha256: str,
    lock_sha256: str,
    entrypoint: Sequence[str],
) -> dict[str, Any]:
    """Validate one current build statement against the requested profile."""

    return _validate_evaluator_build_attestation(
        value,
        expected_format=EVALUATOR_BUILD_ATTESTATION_FORMAT,
        evaluator=evaluator,
        evaluator_version=evaluator_version,
        runtime_image_id=runtime_image_id,
        base_image_id=base_image_id,
        source_commit=source_commit,
        source_bundle_sha256=source_bundle_sha256,
        lock_sha256=lock_sha256,
        entrypoint=entrypoint,
    )


def make_evaluator_build_attestation(
    *,
    evaluator: str,
    evaluator_version: str,
    runtime_image_id: str,
    base_image_id: str,
    source_commit: str,
    source_bundle_sha256: str,
    lock_sha256: str,
    entrypoint: Sequence[str],
    base_layers: Sequence[str],
    image_layers: Sequence[str],
    config: Mapping[str, object],
) -> dict[str, Any]:
    """Create the canonical host-observed evaluator-demo build statement."""

    payload: dict[str, Any] = {
        "base_image_id": base_image_id,
        "base_layers": list(base_layers),
        "config_sha256": _digest_bytes(
            canonical_json_bytes(dict(config), newline=False)
        ),
        "entrypoint": list(entrypoint),
        "evaluator": evaluator,
        "evaluator_version": evaluator_version,
        "format_version": EVALUATOR_BUILD_ATTESTATION_FORMAT,
        "image_layers": list(image_layers),
        "lock_sha256": lock_sha256,
        "runtime_image_id": runtime_image_id,
        "source_bundle_sha256": source_bundle_sha256,
        "source_commit": source_commit,
    }
    validate_evaluator_build_attestation(
        payload,
        evaluator=evaluator,
        evaluator_version=evaluator_version,
        runtime_image_id=runtime_image_id,
        base_image_id=base_image_id,
        source_commit=source_commit,
        source_bundle_sha256=source_bundle_sha256,
        lock_sha256=lock_sha256,
        entrypoint=entrypoint,
    )
    return payload


def sign_evaluator_build_attestation(
    payload: Mapping[str, object],
    signing_key: ed25519.Ed25519PrivateKey,
) -> dict[str, Any]:
    """Sign one host-observed build statement with the independent builder key."""

    statement = dict(payload)
    if statement.get("format_version") != EVALUATOR_BUILD_ATTESTATION_FORMAT or set(
        statement
    ) != {
        "base_image_id",
        "base_layers",
        "config_sha256",
        "entrypoint",
        "evaluator",
        "evaluator_version",
        "format_version",
        "image_layers",
        "lock_sha256",
        "runtime_image_id",
        "source_bundle_sha256",
        "source_commit",
    }:
        raise EvaluatorBuildAttestationError("cannot sign an invalid build statement")
    signature = signing_key.sign(canonical_json_bytes(statement, newline=False))
    return {
        "format_version": SIGNED_EVALUATOR_BUILD_ATTESTATION_FORMAT,
        "signature": {
            "algorithm": "ed25519",
            "builder_fingerprint": public_key_fingerprint(signing_key.public_key()),
            "value": base64.b64encode(signature).decode("ascii"),
        },
        "statement": statement,
    }


def verify_evaluator_build_attestation(
    value: object,
    *,
    builder_public_key: ed25519.Ed25519PublicKey,
    evaluator: str,
    evaluator_version: str,
    runtime_image_id: str,
    base_image_id: str,
    source_commit: str,
    source_bundle_sha256: str,
    lock_sha256: str,
    entrypoint: Sequence[str],
) -> dict[str, Any]:
    """Verify the independent builder signature and strict statement fields."""

    if not isinstance(value, dict) or set(value) != {
        "format_version",
        "signature",
        "statement",
    }:
        raise EvaluatorBuildAttestationError(
            "signed evaluator build attestation envelope is invalid"
        )
    signed_format = value.get("format_version")
    expected_statement_format = (
        _RETAINED_BUILD_ATTESTATION_FORMATS.get(signed_format)
        if isinstance(signed_format, str)
        else None
    )
    if expected_statement_format is None:
        raise EvaluatorBuildAttestationError(
            "signed evaluator build attestation format is invalid"
        )
    signature = value.get("signature")
    statement = value.get("statement")
    if not isinstance(signature, dict) or set(signature) != {
        "algorithm",
        "builder_fingerprint",
        "value",
    }:
        raise EvaluatorBuildAttestationError("evaluator builder signature is invalid")
    if signature.get("algorithm") != "ed25519":
        raise EvaluatorBuildAttestationError(
            "evaluator builder signature algorithm is invalid"
        )
    expected_fingerprint = public_key_fingerprint(builder_public_key)
    if signature.get("builder_fingerprint") != expected_fingerprint:
        raise EvaluatorBuildAttestationError(
            "evaluator builder identity is not trusted"
        )
    encoded = signature.get("value")
    if not isinstance(encoded, str):
        raise EvaluatorBuildAttestationError(
            "evaluator builder signature value is invalid"
        )
    try:
        raw_signature = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise EvaluatorBuildAttestationError(
            "evaluator builder signature encoding is invalid"
        ) from exc
    try:
        builder_public_key.verify(
            raw_signature,
            canonical_json_bytes(statement, newline=False),
        )
    except (InvalidSignature, TypeError, ValueError) as exc:
        raise EvaluatorBuildAttestationError(
            "evaluator builder signature does not verify"
        ) from exc
    return _validate_evaluator_build_attestation(
        statement,
        expected_format=expected_statement_format,
        evaluator=evaluator,
        evaluator_version=evaluator_version,
        runtime_image_id=runtime_image_id,
        base_image_id=base_image_id,
        source_commit=source_commit,
        source_bundle_sha256=source_bundle_sha256,
        lock_sha256=lock_sha256,
        entrypoint=entrypoint,
    )


def write_evaluator_build_attestation(
    path: Path, payload: Mapping[str, object]
) -> None:
    """Write one new canonical signed build attestation without following a link."""

    if payload.get("format_version") != SIGNED_EVALUATOR_BUILD_ATTESTATION_FORMAT:
        raise EvaluatorBuildAttestationError(
            "only signed evaluator build attestations may be written"
        )
    if path.exists() or path.is_symlink():
        raise EvaluatorBuildAttestationError(
            "evaluator build attestation destination exists"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(dict(payload)))
    except OSError as exc:
        raise EvaluatorBuildAttestationError(
            "evaluator build attestation could not be written"
        ) from exc


def load_evaluator_build_attestation(path: Path) -> dict[str, Any]:
    """Load one bounded, strict, no-follow signed build attestation."""

    try:
        raw = read_regular_file_bytes(
            path,
            label="evaluator build attestation",
            max_bytes=_MAX_ATTESTATION_BYTES,
        )
        value = parse_json_bytes(raw, label="evaluator build attestation")
    except StrictJsonError as exc:
        raise EvaluatorBuildAttestationError(str(exc)) from exc
    if canonical_json_bytes(value) != raw:
        raise EvaluatorBuildAttestationError(
            "evaluator build attestation is not canonical JSON"
        )
    if not isinstance(value, dict):
        raise EvaluatorBuildAttestationError(
            "evaluator build attestation must be an object"
        )
    signed_format = value.get("format_version")
    if (
        not isinstance(signed_format, str)
        or signed_format not in _RETAINED_BUILD_ATTESTATION_FORMATS
    ):
        raise EvaluatorBuildAttestationError(
            "unsigned evaluator build attestations are not accepted"
        )
    return dict(value)


__all__ = [
    "EVALUATOR_BUILD_ATTESTATION_FORMAT",
    "SIGNED_EVALUATOR_BUILD_ATTESTATION_FORMAT",
    "EvaluatorBuildAttestationError",
    "load_evaluator_build_attestation",
    "make_evaluator_build_attestation",
    "sign_evaluator_build_attestation",
    "validate_evaluator_build_attestation",
    "verify_evaluator_build_attestation",
    "write_evaluator_build_attestation",
]
