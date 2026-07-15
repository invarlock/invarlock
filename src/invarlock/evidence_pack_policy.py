"""Acceptance-policy bindings for strict evidence-pack verification.

Strict report verification must not treat acceptance thresholds copied into a
producer-controlled report as authorization.  Evidence packs therefore seal a
canonical policy pack in their signed manifest *and* require the verifier caller to
provide the policy independently when strict report assurance is requested.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from invarlock import evidence_pack_baselines as baselines
from invarlock import evidence_pack_integrity as integrity
from invarlock.evidence_pack_json import sha256_prefixed
from invarlock.policy_pack import read_policy_pack_snapshot, verify_policy_pack

POLICY_MANIFEST_FIELD = "verification_policy_pack"
POLICY_ROOT = "policy"
POLICY_RELATIVE_PATH = "policy/policy-pack.json"


@dataclass(frozen=True)
class PolicyMaterialVerification:
    """A validated, externally anchored acceptance policy for nested reports."""

    policy_pack_path: Path | None
    errors: tuple[str, ...]
    required: bool
    policy_digest: str | None


def load_valid_policy_pack(
    path: Path, *, label: str
) -> tuple[dict[str, Any] | None, list[str]]:
    """Load and validate one JSON/YAML policy pack with stable diagnostics."""

    if not path.is_file():
        return None, [f"{label} not found: {path}"]
    _, payload, errors = load_valid_policy_pack_snapshot(path, label=label)
    return payload, errors


def load_valid_policy_pack_snapshot(
    path: Path, *, label: str
) -> tuple[bytes | None, dict[str, Any] | None, list[str]]:
    """Load and validate one policy pack while retaining its exact bytes."""

    if not path.is_file():
        return None, None, [f"{label} not found: {path}"]
    try:
        raw, payload = read_policy_pack_snapshot(path)
    except (OSError, UnicodeError, ValueError) as exc:
        return None, None, [f"{label} is not valid JSON/YAML: {exc}"]
    errors = [f"{label} is invalid: {error}" for error in verify_policy_pack(payload)]
    return raw, (payload if not errors else None), errors


def write_canonical_policy_pack(path: Path, payload: dict[str, Any]) -> None:
    """Write the policy in deterministic JSON form for checksum sealing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def policy_manifest_entry(path: Path) -> dict[str, Any]:
    raw, payload, errors = load_valid_policy_pack_snapshot(
        path, label="verification policy pack"
    )
    if errors or raw is None or payload is None:
        raise ValueError("invalid verification policy pack: " + "; ".join(errors))
    return {
        "path": POLICY_RELATIVE_PATH,
        "digest": sha256_prefixed(raw),
        "policy_digest": payload["policy_digest"],
    }


def _policy_tree_files_and_symlink_errors(pack_dir: Path) -> tuple[set[str], list[str]]:
    root = pack_dir / POLICY_ROOT
    if not root.exists() and not root.is_symlink():
        return set(), []
    if root.is_symlink():
        return set(), [f"{POLICY_ROOT}/ must not be a symlink."]
    files: set[str] = set()
    errors: list[str] = []
    for path in root.rglob("*"):
        relative_path = path.relative_to(pack_dir).as_posix()
        if path.is_symlink():
            errors.append(
                f"Policy material tree must not contain symlinks: {relative_path}"
            )
        elif path.is_file():
            files.add(relative_path)
    return files, errors


def verify_policy_material(
    pack_dir: Path,
    *,
    report_assurance: str,
    acceptance_policy_path: Path | None,
) -> PolicyMaterialVerification:
    """Validate signed policy material and match it to a independently supplied copy."""

    errors: list[str] = []
    try:
        manifest = integrity._load_json(pack_dir / "manifest.json")
    except integrity._json_load_error_types() as exc:
        return PolicyMaterialVerification(
            policy_pack_path=None,
            errors=(f"manifest is not valid JSON: {exc}",),
            required=False,
            policy_digest=None,
        )
    if not isinstance(manifest, dict):
        return PolicyMaterialVerification(
            policy_pack_path=None,
            errors=("manifest must decode to a JSON object",),
            required=False,
            policy_digest=None,
        )

    canonical_reports = baselines._canonical_report_paths(pack_dir)
    strict_reports = baselines._strict_report_paths(pack_dir, canonical_reports)
    required = baselines._manifest_requires_baselines(
        manifest,
        report_assurance=report_assurance,
        strict_report_paths=strict_reports,
    )
    declaration = manifest.get(POLICY_MANIFEST_FIELD)
    actual_policy_files, tree_errors = _policy_tree_files_and_symlink_errors(pack_dir)
    errors.extend(tree_errors)

    if declaration is None:
        if required:
            errors.append(
                "Strict evidence-pack verification requires signed "
                "verification_policy_pack material."
            )
        if actual_policy_files:
            errors.append(
                "Pack contains undeclared policy material: "
                + ", ".join(sorted(actual_policy_files))
                + "."
            )
        if required and acceptance_policy_path is None:
            errors.append(
                "Strict evidence-pack verification requires independently supplied "
                "--policy-pack."
            )
        return PolicyMaterialVerification(
            policy_pack_path=None,
            errors=tuple(errors),
            required=required,
            policy_digest=None,
        )

    if not isinstance(declaration, dict):
        errors.append("manifest verification_policy_pack must be an object.")
        return PolicyMaterialVerification(None, tuple(errors), required, None)

    relative_path = declaration.get("path")
    if relative_path != POLICY_RELATIVE_PATH:
        errors.append(
            f"manifest verification_policy_pack.path must be {POLICY_RELATIVE_PATH!r}."
        )
        return PolicyMaterialVerification(None, tuple(errors), required, None)

    undeclared = actual_policy_files - {POLICY_RELATIVE_PATH}
    if undeclared:
        errors.append(
            "Pack contains undeclared policy material: "
            + ", ".join(sorted(undeclared))
            + "."
        )
    symlink_error = baselines._path_symlink_error(
        pack_dir, POLICY_RELATIVE_PATH, label="verification_policy_pack"
    )
    if symlink_error is not None:
        errors.append(symlink_error)

    sealed_path = pack_dir / POLICY_RELATIVE_PATH
    if not sealed_path.is_file():
        errors.append(
            f"manifest verification_policy_pack.path is missing: {POLICY_RELATIVE_PATH}."
        )
        return PolicyMaterialVerification(None, tuple(errors), required, None)

    sealed_raw, sealed_payload, sealed_errors = load_valid_policy_pack_snapshot(
        sealed_path, label="Signed verification policy pack"
    )
    errors.extend(sealed_errors)
    actual_digest = sha256_prefixed(sealed_raw) if sealed_raw is not None else None
    digest = declaration.get("digest")
    if not isinstance(digest, str) or digest != actual_digest:
        errors.append(
            "manifest verification_policy_pack.digest mismatch "
            f"(recorded {digest!r}, actual {actual_digest!r})."
        )
    if actual_digest is None:
        return PolicyMaterialVerification(None, tuple(errors), required, None)
    checksum_entries = baselines._checksum_entries_by_path(pack_dir).get(
        POLICY_RELATIVE_PATH, []
    )
    actual_hex = actual_digest.removeprefix("sha256:")
    if len(checksum_entries) != 1:
        errors.append(
            "verification_policy_pack.path must have exactly one "
            f"checksums.sha256 entry: {POLICY_RELATIVE_PATH}."
        )
    elif checksum_entries[0] != actual_hex:
        errors.append(
            "verification_policy_pack.digest is not bound by checksums.sha256."
        )

    policy_digest = (
        str(sealed_payload.get("policy_digest"))
        if isinstance(sealed_payload, dict)
        else None
    )
    if declaration.get("policy_digest") != policy_digest:
        errors.append(
            "manifest verification_policy_pack.policy_digest does not match "
            "the signed policy material."
        )

    acceptance_payload: dict[str, Any] | None = None
    if acceptance_policy_path is None:
        if required:
            errors.append(
                "Strict evidence-pack verification requires independently supplied "
                "--policy-pack."
            )
    else:
        _, acceptance_payload, acceptance_errors = load_valid_policy_pack_snapshot(
            acceptance_policy_path, label="Acceptance policy pack"
        )
        errors.extend(acceptance_errors)
        if (
            acceptance_payload is not None
            and sealed_payload is not None
            and acceptance_payload != sealed_payload
        ):
            errors.append(
                "Acceptance policy pack does not exactly match the signed "
                "verification_policy_pack material."
            )

    selected_path = (
        sealed_path
        if acceptance_payload is not None
        and sealed_payload is not None
        and acceptance_payload == sealed_payload
        else None
    )
    return PolicyMaterialVerification(
        policy_pack_path=selected_path,
        errors=tuple(errors),
        required=required,
        policy_digest=policy_digest,
    )


__all__ = [
    "POLICY_MANIFEST_FIELD",
    "POLICY_RELATIVE_PATH",
    "PolicyMaterialVerification",
    "load_valid_policy_pack",
    "load_valid_policy_pack_snapshot",
    "policy_manifest_entry",
    "verify_policy_material",
    "write_canonical_policy_pack",
]
