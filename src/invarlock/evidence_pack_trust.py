from __future__ import annotations

from pathlib import Path
from typing import Any

from invarlock import evidence_pack_integrity as integrity
from invarlock import evidence_pack_manifest as manifest


def load_trust_store_fingerprints(
    trust_store_path: Path | None,
) -> tuple[set[str], list[str], str | None]:
    """Load trusted evidence-pack signer fingerprints from a JSON trust store."""
    path = trust_store_path
    if path is None:
        default_path = integrity.DEFAULT_TRUST_STORE_PATH
        path = default_path if default_path.is_file() else None
    if path is None:
        return set(), [], None
    if not path.is_file():
        return set(), [f"Evidence-pack trust store not found: {path}"], str(path)
    try:
        payload = manifest._load_json(path)
    except manifest._json_load_error_types() as exc:
        return set(), [f"Evidence-pack trust store is not valid JSON: {exc}"], str(path)

    raw_entries: list[Any]
    if isinstance(payload, list):
        raw_entries = list(payload)
    elif isinstance(payload, dict):
        raw = payload.get("trusted_signers", payload.get("fingerprints", []))
        if not isinstance(raw, list):
            return (
                set(),
                ["Evidence-pack trust store trusted_signers must be a list."],
                str(path),
            )
        raw_entries = raw
    else:
        return (
            set(),
            ["Evidence-pack trust store must be a JSON object or list."],
            str(path),
        )

    fingerprints: set[str] = set()
    errors: list[str] = []
    for index, entry in enumerate(raw_entries):
        raw_value = entry.get("fingerprint") if isinstance(entry, dict) else entry
        if not isinstance(raw_value, str):
            errors.append(f"Evidence-pack trust store entry {index} is not a string.")
            continue
        normalized = integrity.normalize_expected_fingerprint(raw_value)
        if normalized is None:
            errors.append(
                f"Evidence-pack trust store entry {index} is not a sha256 fingerprint."
            )
            continue
        fingerprints.add(normalized)
    if not fingerprints and not errors:
        errors.append("Evidence-pack trust store contains no trusted signers.")
    return fingerprints, errors, str(path)


def verify_signature(
    pack_dir: Path,
    *,
    strict: bool,
    expected_fingerprints: set[str] | frozenset[str] | None = None,
) -> tuple[list[str], list[str], str | None]:
    return integrity.verify_signature(
        pack_dir,
        strict=strict,
        load_json_fn=manifest._load_json,
        expected_fingerprints=expected_fingerprints,
    )
