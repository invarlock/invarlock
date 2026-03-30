from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

from invarlock.public_contracts import load_proof_pack_manifest_schema
from invarlock.reporting.verify_contract import (
    VerifyExecutionResult,
    VerifyOutcome,
    run_verify_reports,
)
from invarlock.runtime_security import (
    RUNTIME_MANIFEST_FILENAME,
    unattested_artifacts_allowed,
)

try:  # pragma: no cover - exercised through tests/integration
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None

PROOF_PACK_FORMAT = "proof-pack-v1"
_VERIFY_GPG_TIMEOUT_SECONDS = 30


class ProofPackStatus(IntEnum):
    OK = 0
    USAGE = 2
    MISSING = 3
    FORMAT = 4
    SIGNATURE = 5
    INTEGRITY = 6
    REPORTS = 7


@dataclass(frozen=True)
class ProofPackResult:
    payload: dict[str, Any]
    status: ProofPackStatus


_CONTROL_FILES = {
    "checksums.sha256",
    "manifest.json",
    "manifest.json.asc",
    "metadata/manifest.json",
    "metadata/manifest.json.asc",
    "metadata/checksums.sha256",
}
_CHECKSUM_LINE_RE = re.compile(r"^([A-Fa-f0-9]{64}) [ *](.+)$")
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


def _relative_file_paths(pack_dir: Path) -> list[str]:
    return sorted(
        str(path.relative_to(pack_dir)).replace("\\", "/")
        for path in pack_dir.rglob("*")
        if path.is_file() and path.name != ".DS_Store" and "__MACOSX" not in path.parts
    )


def _write_checksums_file(pack_dir: Path, rel_paths: list[str]) -> None:
    lines = [
        f"{_sha256_bytes((pack_dir / rel_path).read_bytes())}  {rel_path}"
        for rel_path in rel_paths
    ]
    (pack_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _copy_file(source_path: Path, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, dest_path)


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


def verify_manifest_attestation(pack_dir: Path) -> list[str]:
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


def _verify_manifest_binds_checksums(pack_dir: Path) -> list[str]:
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


def _parse_checksums(pack_dir: Path) -> tuple[list[tuple[str, str]], list[str]]:
    entries: list[tuple[str, str]] = []
    errors: list[str] = []
    checksums_path = pack_dir / "checksums.sha256"
    for index, raw_line in enumerate(
        checksums_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.rstrip()
        if not line:
            continue
        match = _CHECKSUM_LINE_RE.match(line)
        if not match:
            errors.append(f"checksums.sha256 line {index} is not a valid sha256 entry")
            continue
        digest, rel_path = match.groups()
        entries.append((digest.lower(), rel_path))
    return entries, errors


def _verify_checksums(pack_dir: Path) -> tuple[list[str], set[str]]:
    entries, errors = _parse_checksums(pack_dir)
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


def _verify_no_extra_files(
    pack_dir: Path, *, covered_paths: set[str], strict: bool
) -> tuple[list[str], list[str]]:
    actual_paths = {
        str(path.relative_to(pack_dir)).replace("\\", "/")
        for path in pack_dir.rglob("*")
        if path.is_file() and path.name != ".DS_Store" and "__MACOSX" not in path.parts
    }
    extras = sorted(actual_paths - covered_paths - _CONTROL_FILES)
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


def _verify_gpg(
    pack_dir: Path, *, strict: bool
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
        result = subprocess.run(
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
    except subprocess.TimeoutExpired:
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
        manifest = _load_json(pack_dir / "manifest.json")
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


def _signature_warnings_to_errors(warnings: list[str]) -> list[str]:
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
        if warning == "gpg verification timed out; skipping manifest signature verification.":
            converted.append("gpg verification timed out.")
            continue
        converted.append(warning)
    return converted


def _run_verify_command(reports: list[Path], *, profile: str) -> VerifyExecutionResult:
    return run_verify_reports(reports, profile=profile, json_mode=True)


def _verify_command_succeeded(result: VerifyExecutionResult) -> bool:
    return result.outcome == VerifyOutcome.OK


def _verify_reports(
    pack_dir: Path,
    *,
    json_out_path: Path | None,
    profile: str,
) -> tuple[list[str], dict[str, Any] | None]:
    reports = sorted(pack_dir.glob("reports/**/evaluation.report.json"))
    if not reports:
        return ["No reports found in pack."], None
    clean_reports = [path for path in reports if "/errors/" not in path.as_posix()]
    error_reports = [path for path in reports if path not in clean_reports]
    if not clean_reports:
        return [
            "No clean reports found in pack (only error-injection reports present)."
        ], None

    clean_result = _run_verify_command(clean_reports, profile=profile)
    if not isinstance(clean_result.payload, dict):
        return ["clean report verification did not return a JSON object."], None
    verify_payload = dict(clean_result.payload)
    if error_reports:
        try:
            error_result = _run_verify_command(error_reports, profile=profile)
        except (
            ImportError,
            ModuleNotFoundError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            return [
                f"error-injection report verification failed: {exc}"
            ], verify_payload
        if not isinstance(error_result.payload, dict):
            return [
                "error-injection report verification did not return a JSON object."
            ], verify_payload
        verify_payload["error_injection"] = {
            "verify": error_result.payload,
            "reports": [
                str(path.relative_to(pack_dir)).replace("\\", "/")
                for path in error_reports
            ],
        }
    if json_out_path is not None and verify_payload is not None:
        json_out_path.write_text(
            json.dumps(verify_payload, sort_keys=True) + "\n", encoding="utf-8"
        )
    if not _verify_command_succeeded(clean_result):
        return [
            "invarlock verify reported report verification failures."
        ], verify_payload
    return [], verify_payload


def inspect_proof_pack(pack_dir: Path) -> ProofPackResult:
    issues: list[str] = []
    payload: dict[str, Any] = {
        "pack": str(pack_dir),
        "ok": False,
        "manifest": {"valid": False, "format": None},
        "signature": {"present": False, "signer_fingerprint": None},
        "reports": {"total": 0, "clean": 0, "errors": 0},
        "artifacts": {"files": 0, "hashed": 0},
        "integrity": {
            "checksums_bound": False,
            "checksums_ok": False,
            "manifest_attestation_ok": False,
            "extra_files": [],
        },
        "issues": issues,
        "strict_ready": False,
    }
    if not pack_dir.is_dir():
        issues.append(f"Pack directory not found: {pack_dir}")
        return ProofPackResult(payload=payload, status=ProofPackStatus.MISSING)
    manifest_path = pack_dir / "manifest.json"
    checksums_path = pack_dir / "checksums.sha256"
    if not manifest_path.is_file():
        issues.append("manifest.json missing in pack.")
        return ProofPackResult(payload=payload, status=ProofPackStatus.MISSING)
    if not checksums_path.is_file():
        issues.append("checksums.sha256 missing in pack.")
        return ProofPackResult(payload=payload, status=ProofPackStatus.MISSING)

    manifest_errors = validate_manifest(manifest_path)
    if manifest_errors:
        issues.extend(manifest_errors)
        return ProofPackResult(payload=payload, status=ProofPackStatus.FORMAT)

    manifest = _load_json(manifest_path)
    payload["manifest"] = {
        "valid": True,
        "format": manifest.get("format") if isinstance(manifest, dict) else None,
    }

    signature_present = (pack_dir / "manifest.json.asc").is_file()
    payload["signature"] = {
        "present": signature_present,
        "signer_fingerprint": (
            manifest.get("signing_key_fingerprint")
            if isinstance(manifest, dict)
            else None
        ),
    }
    if not signature_present:
        issues.append("manifest.json.asc missing; strict verification would fail.")

    reports = sorted(pack_dir.glob("reports/**/evaluation.report.json"))
    clean_reports = [path for path in reports if "/errors/" not in path.as_posix()]
    error_reports = [path for path in reports if path not in clean_reports]
    payload["reports"] = {
        "total": len(reports),
        "clean": len(clean_reports),
        "errors": len(error_reports),
    }

    bind_errors = _verify_manifest_binds_checksums(pack_dir)
    checksum_errors, covered_paths = _verify_checksums(pack_dir)
    attestation_errors = verify_manifest_attestation(pack_dir)
    extra_files = sorted(
        set(_relative_file_paths(pack_dir)) - covered_paths - _CONTROL_FILES
    )
    if extra_files:
        issues.append(
            "Pack contains extra files not covered by checksums.sha256: "
            + ", ".join(extra_files)
        )
    issues.extend(bind_errors)
    issues.extend(checksum_errors)
    issues.extend(attestation_errors)

    payload["artifacts"] = {
        "files": len(_relative_file_paths(pack_dir)),
        "hashed": len(covered_paths),
    }
    payload["integrity"] = {
        "checksums_bound": not bind_errors,
        "checksums_ok": not checksum_errors,
        "manifest_attestation_ok": not attestation_errors,
        "extra_files": extra_files,
    }
    integrity_errors_present = bool(
        bind_errors or checksum_errors or attestation_errors or extra_files
    )
    payload["ok"] = not integrity_errors_present
    payload["strict_ready"] = (
        signature_present
        and not bind_errors
        and not checksum_errors
        and not attestation_errors
        and not extra_files
    )
    return ProofPackResult(
        payload=payload,
        status=(
            ProofPackStatus.INTEGRITY
            if integrity_errors_present
            else ProofPackStatus.OK
        ),
    )


def build_proof_pack(
    out_dir: Path,
    *,
    final_verdict_path: Path,
    report_paths: list[Path],
    source_repo_path: Path | None = None,
    environment_path: Path | None = None,
    material_specs: list[tuple[str, Path]] | None = None,
    readme_path: Path | None = None,
    profile: str = "dev",
) -> ProofPackResult:
    warnings: list[str] = []
    errors: list[str] = []
    payload: dict[str, Any] = {
        "pack": str(out_dir),
        "ok": False,
        "warnings": warnings,
        "errors": errors,
        "reports": {"total": 0},
        "verify": None,
        "files": None,
    }
    material_specs = material_specs or []

    if not report_paths:
        errors.append("proof-pack build requires at least one --report input.")
        return ProofPackResult(payload=payload, status=ProofPackStatus.USAGE)
    if out_dir.exists():
        errors.append(f"Output pack directory already exists: {out_dir}")
        return ProofPackResult(payload=payload, status=ProofPackStatus.USAGE)
    seen_material_names: set[str] = set()
    for material_name, _material_path in material_specs:
        name_error = _validate_material_name(material_name)
        if name_error is not None:
            errors.append(f"Invalid material name {material_name!r}: {name_error}")
        if material_name in seen_material_names:
            errors.append(f"Duplicate material name: {material_name}")
        seen_material_names.add(material_name)

    _, final_errors = _load_json_object(final_verdict_path, label="final_verdict")
    errors.extend(final_errors)
    if source_repo_path is not None:
        _, source_repo_errors = _load_json_object(source_repo_path, label="source_repo")
        errors.extend(source_repo_errors)
    if environment_path is not None:
        _, environment_errors = _load_json_object(environment_path, label="environment")
        errors.extend(environment_errors)
    for material_name, material_path in material_specs:
        _, material_errors = _load_json_object(
            material_path, label=f"material {material_name}"
        )
        errors.extend(material_errors)
    for report_path in report_paths:
        _, report_errors = _load_json_object(report_path, label="report")
        errors.extend(report_errors)
        runtime_manifest_path = report_path.parent / RUNTIME_MANIFEST_FILENAME
        if not runtime_manifest_path.is_file():
            errors.append(f"report sidecar file not found: {runtime_manifest_path}")
        else:
            _, runtime_manifest_errors = _load_json_object(
                runtime_manifest_path, label="runtime manifest"
            )
            errors.extend(runtime_manifest_errors)
    if errors:
        return ProofPackResult(payload=payload, status=ProofPackStatus.FORMAT)

    verify_result = _run_verify_command(report_paths, profile=profile)
    if not _verify_command_succeeded(verify_result):
        payload["verify"] = verify_result.payload
        errors.append("Provided report inputs failed `invarlock verify`.")
        return ProofPackResult(payload=payload, status=ProofPackStatus.REPORTS)

    out_dir.mkdir(parents=True, exist_ok=False)
    rel_paths: list[str] = []

    final_dest = out_dir / "results" / "final_verdict.json"
    _copy_file(final_verdict_path, final_dest)
    rel_paths.append("results/final_verdict.json")

    if source_repo_path is not None:
        source_repo_dest = out_dir / "metadata" / "source_repo.json"
        _copy_file(source_repo_path, source_repo_dest)
        rel_paths.append("metadata/source_repo.json")
    if environment_path is not None:
        environment_dest = out_dir / "metadata" / "environment.json"
        _copy_file(environment_path, environment_dest)
        rel_paths.append("metadata/environment.json")

    material_refs: list[dict[str, Any]] = []
    for material_name, material_path in material_specs:
        suffix = material_path.suffix or ".json"
        rel_path = f"metadata/{material_name}{suffix}"
        material_dest = out_dir / rel_path
        _copy_file(material_path, material_dest)
        rel_paths.append(rel_path)
        material_refs.append(
            {
                "name": material_name,
                "path": rel_path,
                "digest": _sha256_file(material_dest),
            }
        )

    for index, report_path in enumerate(report_paths, start=1):
        report_dir_rel = f"reports/report-{index:03d}"
        rel_path = f"{report_dir_rel}/evaluation.report.json"
        report_dest = out_dir / rel_path
        _copy_file(report_path, report_dest)
        rel_paths.append(rel_path)
        runtime_manifest_rel = f"{report_dir_rel}/{RUNTIME_MANIFEST_FILENAME}"
        _copy_file(
            report_path.parent / RUNTIME_MANIFEST_FILENAME,
            out_dir / runtime_manifest_rel,
        )
        rel_paths.append(runtime_manifest_rel)

    if readme_path is not None:
        if not readme_path.is_file():
            warnings.append(f"README file not found; skipping copy: {readme_path}")
        else:
            readme_dest = out_dir / "README.md"
            _copy_file(readme_path, readme_dest)
            rel_paths.append("README.md")

    _write_checksums_file(out_dir, rel_paths)
    manifest: dict[str, Any] = {
        "format": PROOF_PACK_FORMAT,
        "checksums_sha256": "checksums.sha256",
        "checksums_sha256_digest": _sha256_bytes(
            (out_dir / "checksums.sha256").read_bytes()
        ),
        "subject": {
            "name": "final_verdict",
            "path": "results/final_verdict.json",
            "digest": _sha256_file(final_dest),
        },
    }
    if source_repo_path is not None:
        manifest["invocation"] = {
            "config_source": {
                "path": "metadata/source_repo.json",
                "digest": _sha256_file(out_dir / "metadata" / "source_repo.json"),
            }
        }
    if environment_path is not None:
        manifest["environment"] = {
            "path": "metadata/environment.json",
            "digest": _sha256_file(out_dir / "metadata" / "environment.json"),
        }
    if material_refs:
        manifest["materials"] = material_refs
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )

    payload["ok"] = True
    payload["reports"] = {"total": len(report_paths)}
    payload["verify"] = verify_result.payload
    payload["files"] = {
        "hashed": len(rel_paths),
        "manifest": "manifest.json",
        "checksums": "checksums.sha256",
    }
    return ProofPackResult(payload=payload, status=ProofPackStatus.OK)


def verify_proof_pack(
    pack_dir: Path,
    *,
    json_out_path: Path | None = None,
    skip_verify: bool = False,
    strict: bool = False,
    profile: str = "dev",
) -> ProofPackResult:
    warnings: list[str] = []
    errors: list[str] = []
    verify_payload: dict[str, Any] | None = None
    signer_fingerprint: str | None = None

    if not pack_dir.is_dir():
        errors.append(f"Pack directory not found: {pack_dir}")
        return _build_verify_result(
            pack_dir=pack_dir,
            ok=False,
            strict=strict,
            skip_verify=skip_verify,
            warnings=warnings,
            errors=errors,
            signer_fingerprint=signer_fingerprint,
            verify_payload=verify_payload,
            status=ProofPackStatus.MISSING,
        )
    if not (pack_dir / "manifest.json").is_file():
        errors.append("manifest.json missing in pack.")
        return _build_verify_result(
            pack_dir=pack_dir,
            ok=False,
            strict=strict,
            skip_verify=skip_verify,
            warnings=warnings,
            errors=errors,
            signer_fingerprint=signer_fingerprint,
            verify_payload=verify_payload,
            status=ProofPackStatus.MISSING,
        )
    if not (pack_dir / "checksums.sha256").is_file():
        errors.append("checksums.sha256 missing in pack.")
        return _build_verify_result(
            pack_dir=pack_dir,
            ok=False,
            strict=strict,
            skip_verify=skip_verify,
            warnings=warnings,
            errors=errors,
            signer_fingerprint=signer_fingerprint,
            verify_payload=verify_payload,
            status=ProofPackStatus.MISSING,
        )
    if json_out_path is not None and _path_within_dir(pack_dir, json_out_path):
        errors.append("--json-out must point outside the pack directory.")
        return _build_verify_result(
            pack_dir=pack_dir,
            ok=False,
            strict=strict,
            skip_verify=skip_verify,
            warnings=warnings,
            errors=errors,
            signer_fingerprint=signer_fingerprint,
            verify_payload=verify_payload,
            status=ProofPackStatus.USAGE,
        )

    errors.extend(validate_manifest(pack_dir / "manifest.json"))
    if errors:
        return _build_verify_result(
            pack_dir=pack_dir,
            ok=False,
            strict=strict,
            skip_verify=skip_verify,
            warnings=warnings,
            errors=errors,
            signer_fingerprint=signer_fingerprint,
            verify_payload=verify_payload,
            status=ProofPackStatus.FORMAT,
        )

    signature_errors, signature_warnings, signer_fingerprint = _verify_gpg(
        pack_dir, strict=strict
    )
    if signature_warnings and not strict and not unattested_artifacts_allowed():
        errors.extend(_signature_warnings_to_errors(signature_warnings))
        return _build_verify_result(
            pack_dir=pack_dir,
            ok=False,
            strict=strict,
            skip_verify=skip_verify,
            warnings=warnings,
            errors=errors,
            signer_fingerprint=signer_fingerprint,
            verify_payload=verify_payload,
            status=ProofPackStatus.SIGNATURE,
        )
    warnings.extend(signature_warnings)
    if signature_errors:
        errors.extend(signature_errors)
        return _build_verify_result(
            pack_dir=pack_dir,
            ok=False,
            strict=strict,
            skip_verify=skip_verify,
            warnings=warnings,
            errors=errors,
            signer_fingerprint=signer_fingerprint,
            verify_payload=verify_payload,
            status=ProofPackStatus.SIGNATURE,
        )

    errors.extend(_verify_manifest_binds_checksums(pack_dir))
    checksum_errors, covered_paths = _verify_checksums(pack_dir)
    errors.extend(checksum_errors)
    errors.extend(verify_manifest_attestation(pack_dir))
    extra_errors, extra_warnings = _verify_no_extra_files(
        pack_dir, covered_paths=covered_paths, strict=strict
    )
    errors.extend(extra_errors)
    warnings.extend(extra_warnings)
    if errors:
        return _build_verify_result(
            pack_dir=pack_dir,
            ok=False,
            strict=strict,
            skip_verify=skip_verify,
            warnings=warnings,
            errors=errors,
            signer_fingerprint=signer_fingerprint,
            verify_payload=verify_payload,
            status=ProofPackStatus.INTEGRITY,
        )

    if not skip_verify:
        report_errors, verify_payload = _verify_reports(
            pack_dir, json_out_path=json_out_path, profile=profile
        )
        if report_errors:
            errors.extend(report_errors)
            return _build_verify_result(
                pack_dir=pack_dir,
                ok=False,
                strict=strict,
                skip_verify=skip_verify,
                warnings=warnings,
                errors=errors,
                signer_fingerprint=signer_fingerprint,
                verify_payload=verify_payload,
                status=ProofPackStatus.REPORTS,
            )

    return _build_verify_result(
        pack_dir=pack_dir,
        ok=True,
        strict=strict,
        skip_verify=skip_verify,
        warnings=warnings,
        errors=errors,
        signer_fingerprint=signer_fingerprint,
        verify_payload=verify_payload,
        status=ProofPackStatus.OK,
    )


def _build_verify_result(
    *,
    pack_dir: Path,
    ok: bool,
    strict: bool,
    skip_verify: bool,
    warnings: list[str],
    errors: list[str],
    signer_fingerprint: str | None,
    verify_payload: dict[str, Any] | None,
    status: ProofPackStatus,
) -> ProofPackResult:
    payload: dict[str, Any] = {
        "pack": str(pack_dir),
        "ok": ok,
        "strict": strict,
        "skip_verify": skip_verify,
        "warnings": warnings,
        "errors": errors,
    }
    if signer_fingerprint:
        payload["signer_fingerprint"] = signer_fingerprint
    if verify_payload is not None:
        payload["verify"] = verify_payload
    return ProofPackResult(payload=payload, status=status)


__all__ = [
    "PROOF_PACK_FORMAT",
    "ProofPackResult",
    "ProofPackStatus",
    "validate_manifest",
    "verify_manifest_attestation",
    "verify_proof_pack",
]
