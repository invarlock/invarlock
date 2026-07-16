from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import jsonschema

from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)
from invarlock.public_contracts import (
    load_model_artifact_identity_schema,
    load_runtime_manifest_schema,
    load_runtime_provider_capabilities_schema,
    load_runtime_provider_receipt_schema,
    load_runtime_scoring_observation_schema,
)
from invarlock.runtime_security_helpers import (
    RUNTIME_MANIFEST_VERSION,
    RUNTIME_VERIFIER_CONTRACT_VERSION,
)


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


@dataclass(frozen=True)
class _BoundJsonObject:
    payload: bytes
    value: dict[str, object]
    sha256: str


@dataclass(frozen=True)
class _RuntimeProviderEvidence:
    receipt: _BoundJsonObject
    scoring_observation: _BoundJsonObject
    artifact_identity: _BoundJsonObject


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

    runtime = manifest.get("outer_container")
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
    report_path: Path,
    manifest_path: Path,
    expected_image_digest: str | None = None,
    require_strict_runtime: bool = False,
) -> list[str]:
    errors: list[str] = []

    manifest_version = manifest.get("manifest_version")
    contract_version = manifest.get("verifier_contract_version")
    if manifest_version != RUNTIME_MANIFEST_VERSION:
        return [
            "unsupported runtime manifest version: "
            f"manifest_version={manifest_version!r}"
        ]
    schema = load_runtime_manifest_schema()
    expected_contract_version = RUNTIME_VERIFIER_CONTRACT_VERSION
    if not schema:
        return ["runtime manifest schema is unavailable"]
    try:
        jsonschema.validate(instance=manifest, schema=schema)
    except jsonschema.ValidationError as exc:
        return [f"runtime manifest schema validation failed: {exc.message}"]

    if contract_version != expected_contract_version:
        errors.append(
            f"unexpected verifier contract version: {contract_version or '<missing>'}"
        )

    execution_mode = manifest.get("execution_mode")
    if execution_mode != "container":
        errors.append(
            f'execution_mode must be "container", got {execution_mode or "<missing>"}'
        )

    runtime = manifest.get("outer_container")
    if not isinstance(runtime, dict):
        runtime = {}
    if runtime.get("container_execution") is not True:
        errors.append("outer_container.container_execution must be true")
    image_digest = runtime.get("image_digest")
    image_ref = runtime.get("image_ref")
    if not isinstance(image_digest, str) or not image_digest.strip():
        errors.append("outer_container.image_digest must be present")
    elif not isinstance(image_ref, str) or (
        image_ref != image_digest
        and (
            image_ref.count("@") != 1
            or not image_ref.split("@", 1)[0]
            or image_ref.rsplit("@", 1)[1] != image_digest
        )
    ):
        errors.append("outer_container.image_ref must bind image_digest")
    if require_strict_runtime:
        if runtime.get("allow_network") is not False:
            errors.append("strict runtime forbids allow_network=true")
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

    config = manifest.get("config")
    if (
        require_strict_runtime
        and isinstance(config, dict)
        and config.get("source") != "file"
    ):
        errors.append("strict runtime manifest requires a verifiable file config")
    expected_report_name = report_path.name
    if report.get("path") != expected_report_name:
        errors.append(
            "report.path does not match the verified report filename: "
            f"manifest={report.get('path')!r} actual={expected_report_name!r}"
        )
    if report.get("filename") != expected_report_name:
        errors.append(
            "report.filename does not match the verified report filename: "
            f"manifest={report.get('filename')!r} actual={expected_report_name!r}"
        )
    collision_errors = _reference_collision_errors(
        manifest,
        report_path=report_path,
        manifest_path=manifest_path,
    )
    if collision_errors:
        errors.extend(collision_errors)
        return errors
    provider_evidence, provider_errors = _load_runtime_provider_evidence(
        manifest,
        manifest_path=manifest_path,
    )
    errors.extend(provider_errors)
    if provider_evidence is not None:
        errors.extend(
            _runtime_provider_cross_binding_errors(
                provider_evidence,
                manifest=manifest,
            )
        )
    errors.extend(
        _verify_file_config_binding(
            manifest,
            manifest_path=manifest_path,
        )
    )

    errors.extend(
        _expected_image_digest_errors(
            declared_image_digest=(
                str(runtime.get("image_digest") or "").strip().lower() or None
            ),
            expected_image_digest=expected_image_digest,
        )
    )

    return errors


def _reference_collision_errors(
    manifest: dict[str, object],
    *,
    report_path: Path,
    manifest_path: Path,
) -> list[str]:
    report = manifest.get("report")
    report_names = {report_path.name}
    if isinstance(report, dict):
        report_names.update(
            value
            for value in (report.get("path"), report.get("filename"))
            if isinstance(value, str)
        )

    bindings = manifest.get("runtime_provider")
    if not isinstance(bindings, dict):
        return ["runtime_provider bindings are missing"]
    provider_names: list[str] = []
    for role in ("receipt", "scoring_observation", "artifact_identity"):
        reference = bindings.get(role)
        if isinstance(reference, dict) and isinstance(reference.get("filename"), str):
            provider_names.append(reference["filename"])
    errors: list[str] = []
    if len(provider_names) != len(set(provider_names)):
        errors.append("runtime provider binding filenames must be distinct")

    reserved_names = report_names | {manifest_path.name}
    conflicts = sorted(set(provider_names).intersection(reserved_names))
    if conflicts:
        errors.append(
            "runtime provider bindings collide with the report or manifest: "
            + ", ".join(conflicts)
        )

    config = manifest.get("config")
    config_name = (
        config.get("path")
        if isinstance(config, dict) and config.get("source") == "file"
        else None
    )
    if isinstance(config_name, str):
        if config_name in reserved_names:
            errors.append("file config collides with the report or manifest")
        if config_name in provider_names:
            errors.append("file config collides with a runtime provider binding")
    return errors


def _schema_validation_errors(
    payload: object,
    schema: dict[str, object],
    *,
    label: str,
) -> list[str]:
    validator = jsonschema.Draft202012Validator(schema)
    validation_errors = sorted(
        validator.iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not validation_errors:
        return []
    error = validation_errors[0]
    path = ".".join(str(part) for part in error.absolute_path) or "<root>"
    return [f"{label} schema validation failed at {path}: {error.message}"]


def _read_bound_json_object(
    reference: object,
    *,
    manifest_path: Path,
    label: str,
    schema: dict[str, object],
) -> tuple[_BoundJsonObject | None, list[str]]:
    if not isinstance(reference, dict):
        return None, [f"{label} reference is missing"]
    filename = reference.get("filename")
    expected_sha256 = reference.get("sha256")
    if not isinstance(filename, str) or not isinstance(expected_sha256, str):
        return None, [f"{label} reference is invalid"]
    candidate = manifest_path.parent / filename
    try:
        payload = read_regular_file_bytes(candidate, label=label)
    except StrictJsonError as exc:
        return None, [f"unable to read {label}: {exc}"]
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        return None, [
            f"{label} digest mismatch: manifest={expected_sha256} "
            f"actual={actual_sha256}"
        ]
    try:
        decoded = parse_json_bytes(payload, label=label)
    except StrictJsonError as exc:
        return None, [f"unable to parse {label}: {exc}"]
    if not isinstance(decoded, dict):
        return None, [f"{label} must decode to a JSON object"]
    schema_errors = _schema_validation_errors(decoded, schema, label=label)
    if schema_errors:
        return None, schema_errors
    return _BoundJsonObject(payload, decoded, actual_sha256), []


def _load_runtime_provider_evidence(
    manifest: dict[str, object], *, manifest_path: Path
) -> tuple[_RuntimeProviderEvidence | None, list[str]]:
    bindings = manifest.get("runtime_provider")
    if not isinstance(bindings, dict):
        return None, ["runtime_provider bindings are missing"]
    specifications = (
        (
            "receipt",
            load_runtime_provider_receipt_schema(),
        ),
        (
            "scoring_observation",
            load_runtime_scoring_observation_schema(),
        ),
        (
            "artifact_identity",
            load_model_artifact_identity_schema(),
        ),
    )
    loaded: dict[str, _BoundJsonObject] = {}
    errors: list[str] = []
    for role, schema in specifications:
        sidecar, sidecar_errors = _read_bound_json_object(
            bindings.get(role),
            manifest_path=manifest_path,
            label=f"runtime_provider.{role}",
            schema=schema,
        )
        errors.extend(sidecar_errors)
        if sidecar is not None:
            loaded[role] = sidecar
    if errors:
        return None, errors
    return (
        _RuntimeProviderEvidence(
            receipt=loaded["receipt"],
            scoring_observation=loaded["scoring_observation"],
            artifact_identity=loaded["artifact_identity"],
        ),
        [],
    )


def _runtime_provider_cross_binding_errors(
    evidence: _RuntimeProviderEvidence,
    *,
    manifest: dict[str, object],
) -> list[str]:
    receipt = evidence.receipt.value
    observation = evidence.scoring_observation.value
    artifact = evidence.artifact_identity.value
    errors: list[str] = []

    receipt_artifact = receipt.get("artifact_identity")
    if receipt_artifact != artifact:
        errors.append(
            "receipt artifact_identity does not match the bound artifact file"
        )
    if receipt.get("scoring_observation_sha256") != evidence.scoring_observation.sha256:
        errors.append(
            "receipt scoring_observation_sha256 does not match observation bytes"
        )

    plugin = receipt.get("plugin")
    capabilities = receipt.get("capabilities")
    assert isinstance(plugin, dict)
    assert isinstance(capabilities, dict)
    capability_schema_errors = _schema_validation_errors(
        capabilities,
        load_runtime_provider_capabilities_schema(),
        label="runtime_provider.receipt.capabilities",
    )
    errors.extend(capability_schema_errors)
    provider_names = {
        plugin.get("name"),
        capabilities.get("provider_name"),
        observation.get("provider_name"),
    }
    if len(provider_names) != 1:
        errors.append("receipt and observation provider names do not agree")

    canonical_artifact = json.dumps(
        artifact,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    artifact_identity_sha256 = hashlib.sha256(canonical_artifact).hexdigest()
    if observation.get("artifact_identity_sha256") != artifact_identity_sha256:
        errors.append(
            "observation artifact_identity_sha256 does not match bound artifact"
        )

    outer_container = manifest.get("outer_container")
    assert isinstance(outer_container, dict)
    if receipt.get("outer_image_digest") != outer_container.get("image_digest"):
        errors.append("receipt outer_image_digest does not match outer_container")

    artifact_format = artifact.get("artifact_format")
    artifact_formats = capabilities.get("artifact_formats")
    if (
        not isinstance(artifact_formats, list)
        or artifact_format not in artifact_formats
    ):
        errors.append("bound artifact format is not declared by provider capabilities")
    return errors


def _verify_sibling_digest_reference(
    reference: object,
    *,
    manifest_path: Path,
    label: str,
) -> list[str]:
    if not isinstance(reference, dict):
        return [f"{label} reference is missing"]
    filename = reference.get("filename")
    expected_sha256 = reference.get("sha256")
    if not isinstance(filename, str) or not isinstance(expected_sha256, str):
        return [f"{label} reference is invalid"]
    candidate = manifest_path.parent / filename
    try:
        payload = read_regular_file_bytes(candidate, label=label)
    except StrictJsonError as exc:
        return [f"unable to read {label}: {exc}"]
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        return [
            f"{label} digest mismatch: "
            f"manifest={expected_sha256} actual={actual_sha256}"
        ]
    return []


def _verify_file_config_binding(
    manifest: dict[str, object], *, manifest_path: Path
) -> list[str]:
    config = manifest.get("config")
    if not isinstance(config, dict) or config.get("source") != "file":
        return []
    return _verify_sibling_digest_reference(
        {
            "filename": config.get("path"),
            "sha256": config.get("sha256"),
        },
        manifest_path=manifest_path,
        label="config",
    )


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
        report_path=report_path,
        manifest_path=manifest_path,
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
            report_path=report_path,
            manifest_path=manifest_path,
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
