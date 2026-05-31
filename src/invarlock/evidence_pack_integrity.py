from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.public_contracts import load_evidence_pack_manifest_schema

try:  # pragma: no cover - exercised through tests/integration
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None

EVIDENCE_PACK_FORMAT = "evidence-pack-v1"
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
CHECKSUM_LINE_RE = re.compile(r"^([A-Fa-f0-9]{64}) [ *](.+)$")
_MATERIAL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")
_SHA256_REF_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
SIGNING_KEY_FINGERPRINT_RE = re.compile(r"sha256:[a-f0-9]{64}")
DEFAULT_TRUST_STORE_PATH = (
    Path.home() / ".config" / "invarlock" / "trusted-signers.json"
)


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


def jsonschema_validation_error_types() -> tuple[type[BaseException], ...]:
    return _jsonschema_validation_error_types()


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
    return errors


def validate_manifest(path: Path) -> list[str]:
    try:
        payload = _load_json(path)
    except _json_load_error_types() as exc:
        return [f"manifest is not valid JSON: {exc}"]

    schema = load_evidence_pack_manifest_schema()
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


def relative_file_paths(pack_dir: Path) -> list[str]:
    return sorted(
        str(path.relative_to(pack_dir)).replace("\\", "/")
        for path in pack_dir.rglob("*")
        if path.is_file() and path.name != ".DS_Store" and "__MACOSX" not in path.parts
    )


def write_checksums_file(pack_dir: Path, rel_paths: list[str]) -> None:
    lines = [
        f"{_sha256_bytes((pack_dir / rel_path).read_bytes())}  {rel_path}"
        for rel_path in rel_paths
    ]
    (pack_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def copy_file(source_path: Path, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, dest_path)


def load_private_signing_key(path: Path) -> ed25519.Ed25519PrivateKey:
    private_key = serialization.load_pem_private_key(
        path.read_bytes(),
        password=None,
    )
    if not isinstance(private_key, ed25519.Ed25519PrivateKey):
        raise TypeError("signing key must be an Ed25519 private key in PEM format")
    return private_key


def validate_signing_key(path: Path) -> list[str]:
    try:
        load_private_signing_key(path)
    except FileNotFoundError:
        return [f"signing key file not found: {path}"]
    except (TypeError, ValueError) as exc:
        return [f"signing key is invalid: {exc}"]
    return []


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


def sign_manifest(
    manifest_path: Path,
    *,
    signing_key_path: Path,
    signature_path: Path | None = None,
) -> str:
    private_key = load_private_signing_key(signing_key_path)
    public_key = private_key.public_key()
    fingerprint = public_key_fingerprint(public_key)
    signature = private_key.sign(manifest_path.read_bytes())
    bundle = {
        "format": EVIDENCE_PACK_SIGNATURE_FORMAT,
        "algorithm": "ed25519",
        "signing_key_fingerprint": fingerprint,
        "public_key": {
            "encoding": "pem",
            "value": public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode("ascii"),
        },
        "signature": {
            "encoding": "base64",
            "value": base64.b64encode(signature).decode("ascii"),
        },
    }
    target = signature_path or manifest_path.with_name(MANIFEST_SIGNATURE_FILENAME)
    target.write_text(json.dumps(bundle, sort_keys=True) + "\n", encoding="utf-8")
    return fingerprint


def generate_signing_keypair(
    private_key_path: Path,
    *,
    public_key_path: Path,
) -> str:
    if private_key_path.exists():
        raise FileExistsError(f"private key output already exists: {private_key_path}")
    if public_key_path.exists():
        raise FileExistsError(f"public key output already exists: {public_key_path}")
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    private_key_path.parent.mkdir(parents=True, exist_ok=True)
    public_key_path.parent.mkdir(parents=True, exist_ok=True)
    private_key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    private_key_path.chmod(0o600)
    public_key_path.write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return public_key_fingerprint(public_key)


def verify_manifest_binds_checksums(pack_dir: Path) -> list[str]:
    payload = _load_json(pack_dir / "manifest.json")
    expected = payload.get("checksums_sha256_digest")
    if expected is None:
        return [
            "manifest.json missing checksums_sha256_digest (pack is not tamper-evident)."
        ]
    if not isinstance(expected, str) or not expected:
        return ["manifest.json checksums_sha256_digest is empty."]
    actual = _sha256_bytes((pack_dir / "checksums.sha256").read_bytes())
    if expected != actual:
        return [
            f"checksums.sha256 digest mismatch (expected {expected}, got {actual})."
        ]
    return []


def parse_checksums(pack_dir: Path) -> tuple[list[tuple[str, str]], list[str]]:
    entries: list[tuple[str, str]] = []
    errors: list[str] = []
    checksums_path = pack_dir / "checksums.sha256"
    for index, raw_line in enumerate(
        checksums_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.rstrip()
        if not line:
            continue
        match = CHECKSUM_LINE_RE.match(line)
        if not match:
            errors.append(f"checksums.sha256 line {index} is not a valid sha256 entry")
            continue
        digest, rel_path = match.groups()
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
        actual = _sha256_bytes(resolved.read_bytes())
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
    if not signature_path.is_file():
        if strict:
            return (
                [
                    f"{MANIFEST_SIGNATURE_FILENAME} missing (strict mode requires a signed manifest)."
                ],
                [],
                None,
            )
        return [], [f"{MANIFEST_SIGNATURE_FILENAME} missing; pack is unsigned."], None
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
    try:
        public_key_obj.verify(
            signature_bytes, (pack_dir / "manifest.json").read_bytes()
        )
    except InvalidSignature:
        return ["manifest signature verification failed."], [], None
    manifest = load_json_fn(pack_dir / "manifest.json")
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
    "copy_file",
    "generate_signing_keypair",
    "jsonschema_validation_error_types",
    "load_private_signing_key",
    "parse_checksums",
    "public_key_fingerprint",
    "relative_file_paths",
    "sign_manifest",
    "signature_warnings_to_errors",
    "validate_signing_key",
    "validate_manifest",
    "verify_checksums",
    "verify_signature",
    "verify_manifest_binds_checksums",
    "verify_manifest_provenance",
    "verify_no_extra_files",
    "write_checksums_file",
    "normalize_expected_fingerprint",
]
