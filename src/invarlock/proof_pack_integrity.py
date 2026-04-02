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

from invarlock import proof_pack_manifest as proof_pack_manifest_mod

_json_load_error_types = proof_pack_manifest_mod._json_load_error_types
_load_json = proof_pack_manifest_mod._load_json
_manual_validate_manifest = proof_pack_manifest_mod._manual_validate_manifest
_normalize_pack_path = proof_pack_manifest_mod._normalize_pack_path
_path_within_dir = proof_pack_manifest_mod._path_within_dir
_sha256_bytes = proof_pack_manifest_mod._sha256_bytes
load_proof_pack_manifest_schema = (
    proof_pack_manifest_mod.load_proof_pack_manifest_schema
)
jsonschema = proof_pack_manifest_mod.jsonschema

MANIFEST_SIGNATURE_FILENAME = "manifest.signature.json"
PROOF_PACK_SIGNATURE_FORMAT = "proof-pack-signature-v1"
CONTROL_FILES = {
    "checksums.sha256",
    "manifest.json",
    MANIFEST_SIGNATURE_FILENAME,
    "metadata/checksums.sha256",
    "metadata/manifest.json",
    f"metadata/{MANIFEST_SIGNATURE_FILENAME}",
}
CHECKSUM_LINE_RE = re.compile(r"^([A-Fa-f0-9]{64}) [ *](.+)$")


def jsonschema_validation_error_types() -> tuple[type[BaseException], ...]:
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


def validate_manifest(path: Path) -> list[str]:
    try:
        payload = _load_json(path)
    except _json_load_error_types() as exc:
        return [f"manifest is not valid JSON: {exc}"]

    schema = load_proof_pack_manifest_schema()
    if schema and jsonschema is not None:
        validation_error_types = jsonschema_validation_error_types()
        if validation_error_types:
            try:
                jsonschema.validate(instance=payload, schema=schema)
            except validation_error_types as exc:
                return [f"manifest schema validation failed: {exc}"]
        else:
            jsonschema.validate(instance=payload, schema=schema)
    return _manual_validate_manifest(payload)


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
        "format": PROOF_PACK_SIGNATURE_FORMAT,
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


def verify_checksums(pack_dir: Path) -> tuple[list[str], set[str]]:
    entries, errors = parse_checksums(pack_dir)
    covered_paths: set[str] = set()
    for digest, rel_path in entries:
        covered_paths.add(rel_path)
        resolved = _normalize_pack_path(pack_dir, rel_path)
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
    if payload.get("format") != PROOF_PACK_SIGNATURE_FORMAT:
        errors.append(
            f"{MANIFEST_SIGNATURE_FILENAME} format must be {PROOF_PACK_SIGNATURE_FORMAT!r}."
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
    if not isinstance(fingerprint, str) or not re.fullmatch(
        r"sha256:[a-f0-9]{64}", fingerprint
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
    "MANIFEST_SIGNATURE_FILENAME",
    "PROOF_PACK_SIGNATURE_FORMAT",
    "_path_within_dir",
    "copy_file",
    "generate_signing_keypair",
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
    "verify_no_extra_files",
    "write_checksums_file",
]
