"""Canonical hashing and scalar validation for evidence-catalog v1."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from invarlock.evidence_pack_json import sha256_prefixed


class EvidenceCatalogError(ValueError):
    """Raised when a public evidence catalog is malformed."""


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return sha256_prefixed(payload)


def entry_digest(entry: Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(entry)))


def input_digest(inputs: Mapping[str, object]) -> str:
    """Hash the declared input specification, excluding its self-reference."""

    material = {key: value for key, value in inputs.items() if key != "digest"}
    return sha256_bytes(canonical_json_bytes(material))


def unexpected_keys(
    value: Mapping[str, object], *, allowed: frozenset[str], label: str
) -> list[str]:
    return [
        f"{label} contains unsupported field {key!r}"
        for key in sorted(set(value) - allowed)
    ]


def safe_artifact_path(value: object) -> bool:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or "\\" in value
    ):
        return False
    return all(part and part not in {".", ".."} for part in value.split("/"))


def safe_preset_path(value: object) -> bool:
    if not safe_artifact_path(value) or not isinstance(value, str):
        return False
    parts = value.split("/")
    return (
        len(parts) >= 2
        and parts[0] == "configs"
        and parts[-1].lower().endswith((".yaml", ".yml"))
    )


def require_text(
    value: object,
    *,
    label: str,
    errors: list[str],
    pattern: re.Pattern[str] | None = None,
) -> None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label} must be a non-empty string")
    elif pattern is not None and pattern.fullmatch(value) is None:
        errors.append(f"{label} has an invalid format")


__all__ = [
    "EvidenceCatalogError",
    "canonical_json_bytes",
    "entry_digest",
    "input_digest",
    "require_text",
    "safe_artifact_path",
    "safe_preset_path",
    "sha256_bytes",
    "unexpected_keys",
]
