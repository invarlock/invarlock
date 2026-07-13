from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import jsonschema

from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)
from invarlock.public_contracts import load_runtime_manifest_schema
from invarlock.runtime_security_helpers import RUNTIME_VERIFIER_CONTRACT_VERSION


@dataclass(frozen=True)
class RuntimeVerifyResult:
    ok: bool
    errors: tuple[str, ...]
    report: str
    manifest: str
    binding_verified: bool = False
    expected_digest_matched: bool = False
    trust_status: str = "failed"
    declared_image_digest: str | None = None


_IMAGE_DIGEST_RE = re.compile(r"^sha256:[a-f0-9]{64}$")


def _normalize_expected_image_digest(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized if _IMAGE_DIGEST_RE.fullmatch(normalized) else None


def _declared_image_digest(manifest: dict[str, object]) -> str | None:
    """Return the image digest from the already-validated manifest payload.

    The caller must not re-read the manifest after binding validation. Doing so
    would allow a file swap to combine the report binding from one payload with
    the independently pinned image digest from another.
    """

    runtime = manifest.get("runtime")
    value = runtime.get("image_digest") if isinstance(runtime, dict) else None
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def _expected_image_digest_errors(
    *,
    declared_image_digest: str | None,
    expected_image_digest: str | None,
) -> list[str]:
    if expected_image_digest is None:
        return []
    normalized = _normalize_expected_image_digest(expected_image_digest)
    if normalized is None:
        return [
            "expected runtime image digest must match sha256:<64 lowercase hex chars>"
        ]
    if declared_image_digest != normalized:
        return [
            "runtime image digest mismatch: "
            f"manifest={declared_image_digest or '<missing>'} expected={normalized}"
        ]
    return []


def _load_verification_inputs(
    report_path: Path,
    manifest_path: Path,
) -> tuple[bytes | None, dict[str, object] | None, list[str]]:
    try:
        report_bytes = read_regular_file_bytes(report_path, label="runtime report")
    except StrictJsonError as exc:
        return None, None, [f"unable to read report: {exc}"]

    try:
        manifest_bytes = read_regular_file_bytes(
            manifest_path, label="runtime manifest"
        )
    except StrictJsonError as exc:
        return report_bytes, None, [f"unable to read manifest: {exc}"]
    try:
        manifest = parse_json_bytes(manifest_bytes, label="runtime manifest")
    except StrictJsonError as exc:
        return report_bytes, None, [f"unable to parse manifest: {exc}"]
    if not isinstance(manifest, dict):
        return report_bytes, None, ["manifest payload must be a JSON object"]
    return report_bytes, manifest, []


def _verify_loaded_report_manifest(
    report_bytes: bytes,
    manifest: dict[str, object],
    *,
    expected_image_digest: str | None = None,
    require_strict_runtime: bool = False,
) -> list[str]:
    errors: list[str] = []

    schema = load_runtime_manifest_schema()
    if not schema:
        return ["runtime manifest schema is unavailable"]
    try:
        jsonschema.validate(instance=manifest, schema=schema)
    except jsonschema.ValidationError as exc:
        return [f"runtime manifest schema validation failed: {exc.message}"]

    contract_version = manifest.get("verifier_contract_version")
    if contract_version != RUNTIME_VERIFIER_CONTRACT_VERSION:
        errors.append(
            f"unexpected verifier contract version: {contract_version or '<missing>'}"
        )

    execution_mode = manifest.get("execution_mode")
    if execution_mode != "container":
        errors.append(
            f'execution_mode must be "container", got {execution_mode or "<missing>"}'
        )

    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
    if runtime.get("container_execution") is not True:
        errors.append("runtime.container_execution must be true")
    if not str(runtime.get("image_digest") or "").strip():
        errors.append("runtime.image_digest must be present")
    if require_strict_runtime:
        if runtime.get("allow_remote_code") is not False:
            errors.append("strict runtime forbids allow_remote_code=true")
        if runtime.get("allow_third_party_plugins") is not False:
            errors.append("strict runtime forbids allow_third_party_plugins=true")

    report = manifest.get("report")
    if not isinstance(report, dict):
        report = {}
    expected_sha = report.get("sha256")
    actual_sha = hashlib.sha256(report_bytes).hexdigest()
    if not isinstance(expected_sha, str) or not expected_sha:
        errors.append("manifest is missing report.sha256")
    elif expected_sha != actual_sha:
        errors.append(
            f"report digest mismatch: manifest={expected_sha} actual={actual_sha}"
        )

    if not report_bytes:
        errors.append("report file is empty")

    errors.extend(
        _expected_image_digest_errors(
            declared_image_digest=(
                str(runtime.get("image_digest") or "").strip().lower() or None
            ),
            expected_image_digest=expected_image_digest,
        )
    )

    return errors


def verify_report_manifest(
    report_path: Path,
    manifest_path: Path,
    *,
    expected_image_digest: str | None = None,
    require_strict_runtime: bool = False,
) -> list[str]:
    report_bytes, manifest, load_errors = _load_verification_inputs(
        report_path,
        manifest_path,
    )
    if load_errors:
        return load_errors
    assert report_bytes is not None
    assert manifest is not None
    return _verify_loaded_report_manifest(
        report_bytes,
        manifest,
        expected_image_digest=expected_image_digest,
        require_strict_runtime=require_strict_runtime,
    )


def verify_runtime_manifest(
    report: str | Path,
    manifest: str | Path,
    *,
    expected_image_digest: str | None = None,
    require_strict_runtime: bool = False,
) -> RuntimeVerifyResult:
    report_path = Path(report)
    manifest_path = Path(manifest)
    report_bytes, manifest_payload, load_errors = _load_verification_inputs(
        report_path,
        manifest_path,
    )
    if load_errors:
        trust_errors = tuple(
            _expected_image_digest_errors(
                declared_image_digest=None,
                expected_image_digest=expected_image_digest,
            )
        )
        return RuntimeVerifyResult(
            ok=False,
            errors=tuple(load_errors) + trust_errors,
            report=str(report_path),
            manifest=str(manifest_path),
        )
    assert report_bytes is not None
    assert manifest_payload is not None
    return verify_runtime_manifest_snapshot(
        report_bytes,
        manifest_payload,
        report=report_path,
        manifest=manifest_path,
        expected_image_digest=expected_image_digest,
        require_strict_runtime=require_strict_runtime,
    )


def verify_runtime_manifest_snapshot(
    report_bytes: bytes,
    manifest_payload: dict[str, object],
    *,
    report: str | Path,
    manifest: str | Path,
    expected_image_digest: str | None = None,
    require_strict_runtime: bool = False,
) -> RuntimeVerifyResult:
    """Verify an immutable report/manifest snapshot without re-reading either path."""

    report_path = Path(report)
    manifest_path = Path(manifest)
    binding_errors = tuple(
        _verify_loaded_report_manifest(
            report_bytes,
            manifest_payload,
            require_strict_runtime=require_strict_runtime,
        )
    )
    declared_image_digest = _declared_image_digest(manifest_payload)
    trust_errors = tuple(
        _expected_image_digest_errors(
            declared_image_digest=declared_image_digest,
            expected_image_digest=expected_image_digest,
        )
    )
    errors = binding_errors + trust_errors
    binding_verified = not binding_errors
    expected_digest_matched = bool(
        binding_verified and expected_image_digest is not None and not trust_errors
    )
    if expected_digest_matched:
        trust_status = "expected_image_digest_matched"
    elif binding_verified and expected_image_digest is None:
        trust_status = "manifest_bound"
    else:
        trust_status = "failed"
    return RuntimeVerifyResult(
        ok=not errors,
        errors=errors,
        report=str(report_path),
        manifest=str(manifest_path),
        binding_verified=binding_verified,
        expected_digest_matched=expected_digest_matched,
        trust_status=trust_status,
        declared_image_digest=declared_image_digest,
    )


__all__ = [
    "RuntimeVerifyResult",
    "verify_report_manifest",
    "verify_runtime_manifest",
    "verify_runtime_manifest_snapshot",
]
