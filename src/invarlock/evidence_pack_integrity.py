from __future__ import annotations

import base64
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock import evidence_pack_json as evidence_pack_json_mod
from invarlock.public_contracts import (
    EVIDENCE_PACK_FORMAT_VERSION,
    load_evidence_pack_manifest_schema,
)

_DEFAULT_MANIFEST_SCHEMA_LOADER = load_evidence_pack_manifest_schema

try:  # pragma: no cover - exercised through tests/integration
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None

EVIDENCE_PACK_FORMAT = EVIDENCE_PACK_FORMAT_VERSION
MANIFEST_SIGNATURE_FILENAME = "manifest.signature.json"
EVIDENCE_PACK_SIGNATURE_FORMAT = "evidence-pack-signature-v1"
CONTROL_FILES = {
    "checksums.sha256",
    "manifest.json",
    MANIFEST_SIGNATURE_FILENAME,
    "metadata/checksums.sha256",
    "metadata/manifest.json",
    f"metadata/{MANIFEST_SIGNATURE_FILENAME}",
}
CONTROL_FILE_MIRRORS = {
    "manifest.json": "metadata/manifest.json",
    MANIFEST_SIGNATURE_FILENAME: f"metadata/{MANIFEST_SIGNATURE_FILENAME}",
    "checksums.sha256": "metadata/checksums.sha256",
}
CHECKSUM_LINE_RE = re.compile(r"^([A-Fa-f0-9]{64}) [ *](.+)$")
_MATERIAL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")
_SHA256_REF_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
SIGNING_KEY_FINGERPRINT_RE = re.compile(r"sha256:[a-f0-9]{64}")
DEFAULT_TRUST_STORE_PATH = (
    Path.home() / ".config" / "invarlock" / "trusted-signers.json"
)


def _load_json(path: Path) -> Any:
    return evidence_pack_json_mod.load_json(path, label="JSON input")


def _json_load_error_types() -> tuple[type[BaseException], ...]:
    return (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        evidence_pack_json_mod.StrictJsonError,
    )


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


def jsonschema_validation_error_types() -> tuple[type[BaseException], ...]:
    return _jsonschema_validation_error_types()


def _compile_jsonschema_validator(schema: dict[str, Any]) -> Any | None:
    if jsonschema is None:
        return None
    validators = getattr(jsonschema, "validators", None)
    validator_for = getattr(validators, "validator_for", None)
    if not callable(validator_for):
        return None
    validator_type = validator_for(schema)
    validator_type.check_schema(schema)
    return validator_type(schema)


@lru_cache(maxsize=1)
def _compiled_manifest_validator(
    schema_runtime_id: int,
) -> tuple[dict[str, Any], Any | None]:
    """Load and compile the immutable shipped manifest schema once."""
    del schema_runtime_id
    schema = _DEFAULT_MANIFEST_SCHEMA_LOADER()
    return schema, _compile_jsonschema_validator(schema)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path_hex(path: Path) -> str:
    """Hash a file without materializing a second full-size in-memory copy."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _sha256_file(path: Path) -> str:
    return f"sha256:{_sha256_path_hex(path)}"


def _normalize_pack_path(pack_dir: Path, rel_path: str) -> Path | None:
    if pack_dir.is_symlink() or not rel_path or "\\" in rel_path:
        return None
    parts = rel_path.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        return None
    candidate = pack_dir.joinpath(*parts)
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
    candidate = candidate.resolve()
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
    if payload.get("format") != EVIDENCE_PACK_FORMAT:
        errors.append(
            f"manifest format must be {EVIDENCE_PACK_FORMAT!r} (found {payload.get('format')!r})"
        )
    if payload.get("checksums_sha256") != "checksums.sha256":
        errors.append("manifest checksums_sha256 must point to 'checksums.sha256'")
    digest = payload.get("checksums_sha256_digest")
    if not isinstance(digest, str) or _SHA256_HEX_RE.fullmatch(digest) is None:
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
            or _SHA256_REF_RE.fullmatch(digest_value) is None
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
                    elif _validate_material_name(name) is not None:
                        errors.append(
                            f"manifest materials[{index}].name has invalid characters"
                        )
    verification_baselines = payload.get("verification_baselines")
    if verification_baselines is not None:
        if not isinstance(verification_baselines, list) or not verification_baselines:
            errors.append("manifest verification_baselines must be a non-empty list")
        else:
            for index, baseline in enumerate(verification_baselines):
                _validate_digest_ref(f"verification_baselines[{index}]", baseline)
                if not isinstance(baseline, dict):
                    continue
                name = baseline.get("name")
                if not isinstance(name, str) or not name:
                    errors.append(
                        f"manifest verification_baselines[{index}].name must be a non-empty string"
                    )
                report_paths = baseline.get("report_paths")
                if not isinstance(report_paths, list) or not report_paths:
                    errors.append(
                        f"manifest verification_baselines[{index}].report_paths must be a non-empty list"
                    )
    verification_policy_pack = payload.get("verification_policy_pack")
    if verification_policy_pack is not None:
        _validate_digest_ref("verification_policy_pack", verification_policy_pack)
        if isinstance(verification_policy_pack, dict):
            if verification_policy_pack.get("path") != "policy/policy-pack.json":
                errors.append(
                    "manifest verification_policy_pack.path must point to "
                    "'policy/policy-pack.json'"
                )
            policy_digest = verification_policy_pack.get("policy_digest")
            if (
                not isinstance(policy_digest, str)
                or _SHA256_REF_RE.fullmatch(policy_digest) is None
            ):
                errors.append(
                    "manifest verification_policy_pack.policy_digest must be a "
                    "sha256:... string"
                )
    return errors


def validate_manifest_payload(payload: Any) -> list[str]:
    validation_error_types = _jsonschema_validation_error_types()
    if load_evidence_pack_manifest_schema is _DEFAULT_MANIFEST_SCHEMA_LOADER:
        try:
            schema, validator = _compiled_manifest_validator(id(jsonschema))
        except validation_error_types as exc:
            return [f"manifest schema validation failed: {exc}"]
    else:
        # Keep explicit loader replacement useful for embedders and focused tests.
        schema = load_evidence_pack_manifest_schema()
        validator = None
    if schema and jsonschema is not None:
        if validation_error_types:
            try:
                if validator is not None:
                    validator.validate(payload)
                else:
                    jsonschema.validate(instance=payload, schema=schema)
            except validation_error_types as exc:
                return [f"manifest schema validation failed: {exc}"]
        else:
            if validator is not None:
                validator.validate(payload)
            else:
                jsonschema.validate(instance=payload, schema=schema)
    return _manual_validate_manifest(payload)


def validate_manifest(path: Path) -> list[str]:
    try:
        payload = _load_json(path)
    except _json_load_error_types() as exc:
        return [f"manifest is not valid JSON: {exc}"]
    return validate_manifest_payload(payload)


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
    if not isinstance(digest, str) or _SHA256_REF_RE.fullmatch(digest) is None:
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


def verify_manifest_provenance_payload(pack_dir: Path, payload: Any) -> list[str]:
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
    verification_baselines = payload.get("verification_baselines")
    if isinstance(verification_baselines, list):
        for index, baseline in enumerate(verification_baselines):
            errors.extend(
                _validate_reference(
                    pack_dir=pack_dir,
                    label=f"verification_baselines[{index}]",
                    payload=baseline,
                )
            )
    verification_policy_pack = payload.get("verification_policy_pack")
    if verification_policy_pack is not None:
        errors.extend(
            _validate_reference(
                pack_dir=pack_dir,
                label="verification_policy_pack",
                payload=verification_policy_pack,
            )
        )
    return errors


def verify_manifest_provenance(pack_dir: Path) -> list[str]:
    try:
        manifest = _load_json(pack_dir / "manifest.json")
    except _json_load_error_types() as exc:
        return [f"manifest is not valid JSON: {exc}"]
    return verify_manifest_provenance_payload(pack_dir, manifest)


def relative_file_paths(pack_dir: Path) -> list[str]:
    return sorted(
        str(path.relative_to(pack_dir)).replace("\\", "/")
        for path in pack_dir.rglob("*")
        if path.is_file() and path.name != ".DS_Store" and "__MACOSX" not in path.parts
    )


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
    if not SIGNING_KEY_FINGERPRINT_RE.fullmatch(normalized):
        return None
    return normalized


def verify_manifest_binds_checksums_payload(
    payload: Any, checksums_payload: bytes
) -> list[str]:
    if not isinstance(payload, dict):
        return ["manifest must decode to a JSON object"]
    expected = payload.get("checksums_sha256_digest")
    if expected is None:
        return [
            "manifest.json missing checksums_sha256_digest (pack is not tamper-evident)."
        ]
    if not isinstance(expected, str) or not expected:
        return ["manifest.json checksums_sha256_digest is empty."]
    actual = _sha256_bytes(checksums_payload)
    if expected != actual:
        return [
            f"checksums.sha256 digest mismatch (expected {expected}, got {actual})."
        ]
    return []


def verify_manifest_binds_checksums(pack_dir: Path) -> list[str]:
    try:
        payload = _load_json(pack_dir / "manifest.json")
        checksums_payload = evidence_pack_json_mod.read_regular_file_bytes(
            pack_dir / "checksums.sha256", label="checksums.sha256"
        )
    except _json_load_error_types() as exc:
        return [f"manifest or checksums input is not safe JSON: {exc}"]
    return verify_manifest_binds_checksums_payload(payload, checksums_payload)


def parse_checksums(pack_dir: Path) -> tuple[list[tuple[str, str]], list[str]]:
    entries: list[tuple[str, str]] = []
    errors: list[str] = []
    checksums_path = pack_dir / "checksums.sha256"
    try:
        raw = evidence_pack_json_mod.read_regular_file_bytes(
            checksums_path,
            label="checksums.sha256",
        )
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError, evidence_pack_json_mod.StrictJsonError) as exc:
        return [], [f"checksums.sha256 could not be read safely: {exc}"]
    seen_paths: set[str] = set()
    for index, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip()
        if not line:
            continue
        match = CHECKSUM_LINE_RE.match(line)
        if not match:
            errors.append(f"checksums.sha256 line {index} is not a valid sha256 entry")
            continue
        digest, rel_path = match.groups()
        canonical_path = canonicalize_checksum_path(rel_path)
        if canonical_path in seen_paths:
            errors.append(
                f"checksums.sha256 line {index} duplicates path {canonical_path!r}; "
                "each path must have exactly one checksum entry"
            )
        seen_paths.add(canonical_path)
        entries.append((digest.lower(), rel_path))
    return entries, errors


def canonicalize_checksum_path(rel_path: str) -> str:
    canonical = rel_path.replace("\\", "/")
    while canonical.startswith("./"):
        canonical = canonical[2:]
    return canonical


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
        str(path.relative_to(pack_dir)).replace("\\", "/")
        for path in pack_dir.rglob("*")
        if path.is_file() and path.name != ".DS_Store" and "__MACOSX" not in path.parts
    }
    extras = sorted(actual_paths - covered_paths - CONTROL_FILES)
    if not extras:
        return [], []
    if strict:
        return (
            [
                f"Pack contains extra files not covered by checksums.sha256: {', '.join(extras)}"
            ],
            [],
        )
    return [], [
        f"Pack contains extra files not covered by checksums.sha256: {', '.join(extras)}"
    ]


def verify_control_file_mirrors(pack_dir: Path) -> list[str]:
    errors: list[str] = []
    for canonical_rel, mirror_rel in CONTROL_FILE_MIRRORS.items():
        mirror_path = pack_dir / mirror_rel
        if not mirror_path.is_file():
            continue
        canonical_path = pack_dir / canonical_rel
        if not canonical_path.is_file():
            errors.append(
                f"{mirror_rel} exists but canonical {canonical_rel} is missing."
            )
            continue
        if mirror_path.read_bytes() != canonical_path.read_bytes():
            errors.append(
                f"{mirror_rel} must match canonical {canonical_rel} byte-for-byte."
            )
    return errors


def _load_signature_bundle(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        payload = _load_json(path)
    except _json_load_error_types() as exc:
        return None, [f"{MANIFEST_SIGNATURE_FILENAME} is not valid JSON: {exc}"]
    if not isinstance(payload, dict):
        return None, [f"{MANIFEST_SIGNATURE_FILENAME} must decode to a JSON object."]
    errors: list[str] = []
    if payload.get("format") != EVIDENCE_PACK_SIGNATURE_FORMAT:
        errors.append(
            f"{MANIFEST_SIGNATURE_FILENAME} format must be {EVIDENCE_PACK_SIGNATURE_FORMAT!r}."
        )
    if payload.get("algorithm") != "ed25519":
        errors.append(f"{MANIFEST_SIGNATURE_FILENAME} algorithm must be 'ed25519'.")
    public_key = payload.get("public_key")
    if not isinstance(public_key, dict):
        errors.append(f"{MANIFEST_SIGNATURE_FILENAME} public_key must be an object.")
    else:
        if public_key.get("encoding") != "pem":
            errors.append(
                f"{MANIFEST_SIGNATURE_FILENAME} public_key.encoding must be 'pem'."
            )
        value = public_key.get("value")
        if not isinstance(value, str) or not value.strip():
            errors.append(
                f"{MANIFEST_SIGNATURE_FILENAME} public_key.value must be a non-empty PEM string."
            )
    signature = payload.get("signature")
    if not isinstance(signature, dict):
        errors.append(f"{MANIFEST_SIGNATURE_FILENAME} signature must be an object.")
    else:
        if signature.get("encoding") != "base64":
            errors.append(
                f"{MANIFEST_SIGNATURE_FILENAME} signature.encoding must be 'base64'."
            )
        value = signature.get("value")
        if not isinstance(value, str) or not value.strip():
            errors.append(
                f"{MANIFEST_SIGNATURE_FILENAME} signature.value must be a non-empty base64 string."
            )
    fingerprint = payload.get("signing_key_fingerprint")
    if not isinstance(fingerprint, str) or not SIGNING_KEY_FINGERPRINT_RE.fullmatch(
        fingerprint
    ):
        errors.append(
            f"{MANIFEST_SIGNATURE_FILENAME} signing_key_fingerprint must be a sha256:... string."
        )
    return payload if not errors else None, errors


def verify_signature(
    pack_dir: Path,
    *,
    strict: bool,
    load_json_fn: Any = _load_json,
    expected_fingerprints: set[str] | frozenset[str] | None = None,
) -> tuple[list[str], list[str], str | None]:
    signature_path = pack_dir / MANIFEST_SIGNATURE_FILENAME
    if signature_path.is_symlink():
        _bundle, errors = _load_signature_bundle(signature_path)
        return errors, [], None
    if not signature_path.exists():
        if strict:
            return (
                [
                    f"{MANIFEST_SIGNATURE_FILENAME} missing (strict mode requires a signed manifest)."
                ],
                [],
                None,
            )
        return [], [f"{MANIFEST_SIGNATURE_FILENAME} missing; pack is unsigned."], None
    if not signature_path.is_file():
        return [f"{MANIFEST_SIGNATURE_FILENAME} must be a regular file."], [], None
    bundle, errors = _load_signature_bundle(signature_path)
    if errors:
        return errors, [], None
    assert bundle is not None
    try:
        public_key_value = bundle["public_key"]["value"]
        public_key_obj = serialization.load_pem_public_key(
            public_key_value.encode("ascii")
        )
    except (TypeError, ValueError) as exc:
        return [f"manifest signature verification failed. {exc}"], [], None
    if not isinstance(public_key_obj, ed25519.Ed25519PublicKey):
        return (
            ["manifest signature verification failed. public key must be Ed25519."],
            [],
            None,
        )
    derived_fingerprint = public_key_fingerprint(public_key_obj)
    bundle_fingerprint = bundle["signing_key_fingerprint"]
    if bundle_fingerprint != derived_fingerprint:
        return (
            [
                f"{MANIFEST_SIGNATURE_FILENAME} signing_key_fingerprint ({bundle_fingerprint}) does not match bundled public key ({derived_fingerprint})."
            ],
            [],
            derived_fingerprint,
        )
    try:
        signature_bytes = base64.b64decode(bundle["signature"]["value"], validate=True)
    except (TypeError, ValueError) as exc:
        return [f"manifest signature verification failed. {exc}"], [], None
    manifest_path = pack_dir / "manifest.json"
    try:
        manifest_bytes = evidence_pack_json_mod.read_regular_file_bytes(
            manifest_path, label="manifest.json"
        )
    except evidence_pack_json_mod.StrictJsonError as exc:
        return [f"manifest signature verification failed. {exc}"], [], None
    try:
        public_key_obj.verify(signature_bytes, manifest_bytes)
    except InvalidSignature:
        return ["manifest signature verification failed."], [], None
    try:
        if load_json_fn is _load_json:
            manifest = evidence_pack_json_mod.parse_json_bytes(
                manifest_bytes, label="manifest.json"
            )
        else:
            manifest = load_json_fn(manifest_path)
    except _json_load_error_types():
        # The signature authenticates raw bytes.  Manifest syntax is checked by
        # the format phase after authentication, so a signed malformed manifest
        # remains a format failure rather than being treated as an unsigned pack.
        manifest = {}
    if not isinstance(manifest, dict):
        manifest = {}
    recorded = manifest.get("signing_key_fingerprint")
    if recorded and recorded != derived_fingerprint:
        return (
            [
                f"manifest.json signing_key_fingerprint ({recorded}) does not match signature key ({derived_fingerprint})."
            ],
            [],
            derived_fingerprint,
        )
    if (
        expected_fingerprints is not None
        and derived_fingerprint not in expected_fingerprints
    ):
        expected = ", ".join(sorted(expected_fingerprints))
        return (
            [
                "manifest signature signer mismatch: "
                f"expected one of [{expected}], got {derived_fingerprint}."
            ],
            [],
            derived_fingerprint,
        )
    return [], [], derived_fingerprint


def signature_warnings_to_errors(warnings: list[str]) -> list[str]:
    converted: list[str] = []
    for warning in warnings:
        if warning == f"{MANIFEST_SIGNATURE_FILENAME} missing; pack is unsigned.":
            converted.append(
                f"{MANIFEST_SIGNATURE_FILENAME} missing; signed manifest required by default."
            )
            continue
        converted.append(warning)
    return converted


__all__ = [
    "CONTROL_FILES",
    "CONTROL_FILE_MIRRORS",
    "EVIDENCE_PACK_FORMAT",
    "MANIFEST_SIGNATURE_FILENAME",
    "EVIDENCE_PACK_SIGNATURE_FORMAT",
    "DEFAULT_TRUST_STORE_PATH",
    "SIGNING_KEY_FINGERPRINT_RE",
    "_json_load_error_types",
    "_load_json",
    "_load_json_object",
    "_manual_validate_manifest",
    "_material_spec",
    "_normalize_pack_path",
    "_path_within_dir",
    "_sha256_bytes",
    "_sha256_file",
    "_validate_material_name",
    "_validate_reference",
    "jsonschema_validation_error_types",
    "parse_checksums",
    "public_key_fingerprint",
    "relative_file_paths",
    "signature_warnings_to_errors",
    "validate_manifest",
    "validate_manifest_payload",
    "verify_checksums",
    "verify_signature",
    "verify_manifest_binds_checksums",
    "verify_manifest_binds_checksums_payload",
    "verify_control_file_mirrors",
    "verify_manifest_provenance",
    "verify_no_extra_files",
    "normalize_expected_fingerprint",
]
