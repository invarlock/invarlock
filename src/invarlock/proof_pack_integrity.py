from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from invarlock import proof_pack_manifest as proof_pack_manifest_mod

_VERIFY_GPG_TIMEOUT_SECONDS = 30
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

CONTROL_FILES = {
    "checksums.sha256",
    "manifest.json",
    "manifest.json.asc",
    "metadata/manifest.json",
    "metadata/manifest.json.asc",
    "metadata/checksums.sha256",
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


def verify_gpg(
    pack_dir: Path,
    *,
    strict: bool,
    subprocess_module: Any = subprocess,
    load_json_fn: Any = _load_json,
) -> tuple[list[str], list[str], str | None]:
    signature_path = pack_dir / "manifest.json.asc"
    if not signature_path.is_file():
        if strict:
            return (
                ["manifest.json.asc missing (strict mode requires a signed manifest)."],
                [],
                None,
            )
        return [], ["manifest.json.asc missing; pack is unsigned."], None
    try:
        result = subprocess_module.run(
            [
                "gpg",
                "--status-fd",
                "1",
                "--verify",
                str(signature_path),
                str(pack_dir / "manifest.json"),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=_VERIFY_GPG_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        if strict:
            return (
                ["gpg not found (strict mode requires signature verification)."],
                [],
                None,
            )
        return [], ["gpg not found; skipping manifest signature verification."], None
    except UnicodeDecodeError as exc:
        return [f"manifest signature verification failed. {exc}"], [], None
    except subprocess_module.TimeoutExpired:
        if strict:
            return (["gpg verification timed out."], [], None)
        return (
            [],
            ["gpg verification timed out; skipping manifest signature verification."],
            None,
        )
    if result.returncode != 0:
        message = (result.stdout + result.stderr).strip()
        error = "manifest signature verification failed."
        if message:
            error = f"{error} {message}"
        return [error], [], None
    signer_fpr = None
    for line in result.stdout.splitlines():
        if "VALIDSIG " not in line:
            continue
        parts = line.split()
        if len(parts) >= 3:
            signer_fpr = parts[2]
            break
    if signer_fpr:
        manifest = load_json_fn(pack_dir / "manifest.json")
        recorded = manifest.get("signing_key_fingerprint")
        if recorded and recorded != signer_fpr:
            return (
                [
                    f"manifest.json signing_key_fingerprint ({recorded}) does not match signature key ({signer_fpr})."
                ],
                [],
                signer_fpr,
            )
    return [], [], signer_fpr


def signature_warnings_to_errors(warnings: list[str]) -> list[str]:
    converted: list[str] = []
    for warning in warnings:
        if warning == "manifest.json.asc missing; pack is unsigned.":
            converted.append(
                "manifest.json.asc missing; signed manifest required by default."
            )
            continue
        if warning == "gpg not found; skipping manifest signature verification.":
            converted.append(
                "gpg not found; default proof-pack verification requires signature verification."
            )
            continue
        if (
            warning
            == "gpg verification timed out; skipping manifest signature verification."
        ):
            converted.append("gpg verification timed out.")
            continue
        converted.append(warning)
    return converted


__all__ = [
    "CONTROL_FILES",
    "_path_within_dir",
    "copy_file",
    "parse_checksums",
    "relative_file_paths",
    "signature_warnings_to_errors",
    "validate_manifest",
    "verify_checksums",
    "verify_gpg",
    "verify_manifest_binds_checksums",
    "verify_no_extra_files",
    "write_checksums_file",
]
