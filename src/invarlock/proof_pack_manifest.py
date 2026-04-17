from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from invarlock.public_contracts import load_proof_pack_manifest_schema

try:  # pragma: no cover - exercised through tests/integration
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None

PROOF_PACK_FORMAT = "proof-pack-v1"
_MATERIAL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_load_error_types() -> tuple[type[BaseException], ...]:
    return (OSError, UnicodeDecodeError, json.JSONDecodeError)


def _jsonschema_validation_error_types() -> tuple[type[BaseException], ...]:
    if jsonschema is None:
        return ()
    exceptions_mod = getattr(jsonschema, "exceptions", None)
    error_types: list[type[BaseException]] = []
    for attr in ("ValidationError", "SchemaError"):
        exc_type = None
        if exceptions_mod is not None:
            exc_type = getattr(exceptions_mod, attr, None)
        if exc_type is None:
            exc_type = getattr(jsonschema, attr, None)
        if isinstance(exc_type, type) and issubclass(exc_type, BaseException):
            error_types.append(exc_type)
    return tuple(error_types)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return f"sha256:{_sha256_bytes(path.read_bytes())}"


def _normalize_pack_path(pack_dir: Path, rel_path: str) -> Path | None:
    candidate = (pack_dir / rel_path).resolve()
    try:
        candidate.relative_to(pack_dir.resolve())
    except ValueError:
        return None
    return candidate


def _path_within_dir(dir_path: Path, candidate_path: Path) -> bool:
    try:
        candidate_path.resolve().relative_to(dir_path.resolve())
    except ValueError:
        return False
    return True


def _manual_validate_manifest(payload: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["manifest must decode to a JSON object"]
    required = ["format", "checksums_sha256", "checksums_sha256_digest"]
    for field in required:
        if field not in payload:
            errors.append(f"manifest missing required field: {field}")
    if payload.get("format") != PROOF_PACK_FORMAT:
        errors.append(
            f"manifest format must be {PROOF_PACK_FORMAT!r} (found {payload.get('format')!r})"
        )
    if payload.get("checksums_sha256") != "checksums.sha256":
        errors.append("manifest checksums_sha256 must point to 'checksums.sha256'")
    digest = payload.get("checksums_sha256_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        errors.append("manifest checksums_sha256_digest must be a 64-char sha256 hex")
    network_mode = payload.get("network_mode")
    if network_mode is not None and network_mode not in {"offline", "online"}:
        errors.append("manifest network_mode must be 'offline' or 'online'")
    evidence_level = payload.get("evidence_level")
    if evidence_level is not None and evidence_level not in {"low", "medium", "high"}:
        errors.append("manifest evidence_level must be 'low', 'medium', or 'high'")
    artifacts = payload.get("artifacts")
    if artifacts is not None and not isinstance(artifacts, list):
        errors.append("manifest artifacts must be a list")

    builder = payload.get("builder")
    if builder is not None:
        if not isinstance(builder, dict):
            errors.append("manifest builder must be an object")
        else:
            if not isinstance(builder.get("id"), str) or not builder.get("id"):
                errors.append("manifest builder.id must be a non-empty string")
            if not isinstance(builder.get("name"), str) or not builder.get("name"):
                errors.append("manifest builder.name must be a non-empty string")

    def _validate_digest_ref(label: str, value: Any) -> None:
        if value is None:
            return
        if not isinstance(value, dict):
            errors.append(f"manifest {label} must be an object")
            return
        path = value.get("path")
        digest_value = value.get("digest")
        if path is None and digest_value is None:
            return
        if not isinstance(path, str) or not path:
            errors.append(f"manifest {label}.path must be a non-empty string")
        if (
            not isinstance(digest_value, str)
            or not digest_value.startswith("sha256:")
            or len(digest_value) != 71
        ):
            errors.append(f"manifest {label}.digest must be a sha256:... string")

    _validate_digest_ref("subject", payload.get("subject"))

    invocation = payload.get("invocation")
    if invocation is not None:
        if not isinstance(invocation, dict):
            errors.append("manifest invocation must be an object")
        else:
            config_source = invocation.get("config_source")
            if config_source is not None and not isinstance(config_source, dict):
                errors.append("manifest invocation.config_source must be an object")
            _validate_digest_ref("invocation.config_source", config_source)
            parameters = invocation.get("parameters")
            if parameters is not None and not isinstance(parameters, dict):
                errors.append("manifest invocation.parameters must be an object")

    _validate_digest_ref("environment", payload.get("environment"))

    materials = payload.get("materials")
    if materials is not None:
        if not isinstance(materials, list):
            errors.append("manifest materials must be a list")
        else:
            for index, material in enumerate(materials):
                _validate_digest_ref(f"materials[{index}]", material)
                if isinstance(material, dict):
                    name = material.get("name")
                    if not isinstance(name, str) or not name:
                        errors.append(
                            f"manifest materials[{index}].name must be a non-empty string"
                        )
    return errors


def validate_manifest(path: Path) -> list[str]:
    try:
        payload = _load_json(path)
    except _json_load_error_types() as exc:
        return [f"manifest is not valid JSON: {exc}"]

    schema = load_proof_pack_manifest_schema()
    if schema and jsonschema is not None:
        validation_error_types = _jsonschema_validation_error_types()
        if validation_error_types:
            try:
                jsonschema.validate(instance=payload, schema=schema)
            except validation_error_types as exc:
                return [f"manifest schema validation failed: {exc}"]
        else:
            jsonschema.validate(instance=payload, schema=schema)
    return _manual_validate_manifest(payload)


def _load_json_object(
    path: Path, *, label: str
) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.is_file():
        return None, [f"{label} file not found: {path}"]
    try:
        payload = _load_json(path)
    except _json_load_error_types() as exc:
        return None, [f"{label} is not valid JSON: {exc}"]
    if not isinstance(payload, dict):
        return None, [f"{label} must decode to a JSON object: {path}"]
    return payload, []


def _material_spec(name_and_path: str) -> tuple[str, Path] | None:
    name, sep, raw_path = name_and_path.partition("=")
    if not sep:
        return None
    material_name = name.strip()
    material_path = Path(raw_path.strip())
    if not material_name or not raw_path.strip():
        return None
    return material_name, material_path


def _validate_material_name(name: str) -> str | None:
    if _MATERIAL_NAME_RE.fullmatch(name):
        return None
    return (
        "material names must match "
        "[A-Za-z0-9][A-Za-z0-9._-]* and must not contain path separators"
    )


def _validate_reference(*, pack_dir: Path, label: str, payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    rel_path = payload.get("path")
    digest = payload.get("digest")
    if rel_path is None and digest is None:
        return []
    if not isinstance(rel_path, str) or not rel_path:
        return [
            f"{label} must include a non-empty path when digest verification is enabled"
        ]
    if (
        not isinstance(digest, str)
        or not digest.startswith("sha256:")
        or len(digest) != 71
    ):
        return [f"{label} digest must be a sha256:... string"]
    resolved = _normalize_pack_path(pack_dir, rel_path)
    if resolved is None:
        return [f"{label} path escapes the pack root: {rel_path}"]
    if not resolved.is_file():
        return [f"{label} path is missing: {rel_path}"]
    actual = _sha256_file(resolved)
    if actual != digest:
        return [
            f"{label} digest mismatch for {rel_path} (expected {digest}, got {actual})"
        ]
    return []


def verify_manifest_provenance(pack_dir: Path) -> list[str]:
    payload = _load_json(pack_dir / "manifest.json")
    if not isinstance(payload, dict):
        return ["manifest must decode to a JSON object"]

    errors: list[str] = []
    errors.extend(
        _validate_reference(
            pack_dir=pack_dir, label="subject", payload=payload.get("subject")
        )
    )
    invocation = payload.get("invocation")
    if isinstance(invocation, dict):
        errors.extend(
            _validate_reference(
                pack_dir=pack_dir,
                label="invocation.config_source",
                payload=invocation.get("config_source"),
            )
        )
    errors.extend(
        _validate_reference(
            pack_dir=pack_dir, label="environment", payload=payload.get("environment")
        )
    )
    materials = payload.get("materials")
    if isinstance(materials, list):
        for index, material in enumerate(materials):
            errors.extend(
                _validate_reference(
                    pack_dir=pack_dir,
                    label=f"materials[{index}]",
                    payload=material,
                )
            )
    return errors
