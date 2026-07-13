"""Closed execution-policy contracts for public evidence-catalog lanes."""

from __future__ import annotations

import hashlib
import importlib.resources as resources
from collections.abc import Mapping

from invarlock.evidence_catalog_contracts.primitives import EvidenceCatalogError
from invarlock.strict_yaml import StrictYamlError, parse_yaml_bytes

EXECUTION_POLICY_KEYS = frozenset(
    {
        "profile",
        "profile_sha256",
        "tier",
        "assurance_mode",
        "execution_mode",
        "edit_name",
        "preview_n",
        "final_n",
    }
)

_V1_PROFILE = "release"
_V1_PROFILE_SPEC: dict[str, object] = {
    "profile_sha256": (
        "sha256:368a928b080908122a156c20b869660855fdca70267fe247b14302a9ce8ac31d"
    ),
    "preview_n": 400,
    "final_n": 400,
}


def execution_policy_errors(
    payload: object,
    *,
    label: str,
) -> list[str]:
    """Validate the exact release execution contract for a v1 catalog lane."""

    if not isinstance(payload, Mapping):
        return [f"{label} must be an object"]
    errors: list[str] = []
    if set(payload) != EXECUTION_POLICY_KEYS:
        errors.append(f"{label} must declare the exact v1 field set")
    # The frozen 39-run inventory records every v1 lane as release-profile;
    # assurance is never inferred from the lane's data modality.
    expected_profile = _V1_PROFILE
    if payload.get("profile") != expected_profile:
        errors.append(f"{label}.profile must be {expected_profile!r}")
    spec = _V1_PROFILE_SPEC
    for field in ("profile_sha256", "preview_n", "final_n"):
        if payload.get(field) != spec[field]:
            errors.append(f"{label}.{field} does not match the v1 profile contract")
    for field, expected in (
        ("tier", "balanced"),
        ("assurance_mode", "strict"),
        ("execution_mode", "container"),
        ("edit_name", "noop"),
    ):
        if payload.get(field) != expected:
            errors.append(f"{label}.{field} must be {expected!r}")
    return errors


def load_catalog_profile_overrides(
    execution: Mapping[str, object],
) -> dict[str, object]:
    """Load the packaged profile only when its bytes match the catalog contract."""

    profile = execution.get("profile")
    expected_digest = execution.get("profile_sha256")
    if profile != _V1_PROFILE:
        raise EvidenceCatalogError("catalog execution profile is invalid")
    if expected_digest != _V1_PROFILE_SPEC["profile_sha256"]:
        raise EvidenceCatalogError("catalog execution profile digest is invalid")
    try:
        profile_bytes = (
            resources.files("invarlock._data.runtime")
            .joinpath("profiles", f"{profile}.yaml")
            .read_bytes()
        )
    except (FileNotFoundError, OSError, TypeError) as exc:
        raise EvidenceCatalogError("catalog execution profile is unavailable") from exc
    observed_digest = "sha256:" + hashlib.sha256(profile_bytes).hexdigest()
    if observed_digest != expected_digest:
        raise EvidenceCatalogError("catalog execution profile bytes do not match")
    try:
        payload = parse_yaml_bytes(profile_bytes, label="catalog execution profile")
    except StrictYamlError as exc:
        raise EvidenceCatalogError(
            "catalog execution profile cannot be parsed"
        ) from exc
    if not isinstance(payload, dict):
        raise EvidenceCatalogError("catalog execution profile must be an object")
    return payload


__all__ = [
    "EXECUTION_POLICY_KEYS",
    "execution_policy_errors",
    "load_catalog_profile_overrides",
]
