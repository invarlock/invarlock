from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

from invarlock import evidence_pack_integrity as evidence_pack_integrity_mod
from invarlock import evidence_pack_manifest as evidence_pack_manifest_mod
from invarlock import evidence_pack_metadata as evidence_pack_metadata_mod
from invarlock import evidence_pack_support as evidence_pack_support_mod
from invarlock.reporting.verify_contract import (
    VerifyExecutionResult,
    VerifyOutcome,
    run_verify_reports,
)
from invarlock.runtime_security import unverified_provenance_allowed

EVIDENCE_PACK_FORMAT = evidence_pack_manifest_mod.EVIDENCE_PACK_FORMAT
jsonschema = evidence_pack_manifest_mod.jsonschema
load_evidence_pack_manifest_schema = (
    evidence_pack_manifest_mod.load_evidence_pack_manifest_schema
)
_load_json = evidence_pack_manifest_mod._load_json
_json_load_error_types = evidence_pack_manifest_mod._json_load_error_types
_load_json_object = evidence_pack_manifest_mod._load_json_object
_manual_validate_manifest = evidence_pack_manifest_mod._manual_validate_manifest
_material_spec = evidence_pack_manifest_mod._material_spec
_normalize_pack_path = evidence_pack_manifest_mod._normalize_pack_path
_path_within_dir = evidence_pack_manifest_mod._path_within_dir
_sha256_bytes = evidence_pack_manifest_mod._sha256_bytes
_sha256_file = evidence_pack_manifest_mod._sha256_file
_validate_material_name = evidence_pack_manifest_mod._validate_material_name
_validate_reference = evidence_pack_manifest_mod._validate_reference
verify_manifest_provenance = evidence_pack_manifest_mod.verify_manifest_provenance
_evidence_pack_counts_from_verification = (
    evidence_pack_metadata_mod._evidence_pack_counts_from_verification
)
_derive_evidence_pack_evidence_level = (
    evidence_pack_metadata_mod._derive_evidence_pack_evidence_level
)
_render_evidence_pack_readme = evidence_pack_metadata_mod._render_evidence_pack_readme
_CONTROL_FILES = evidence_pack_integrity_mod.CONTROL_FILES
MANIFEST_SIGNATURE_FILENAME = evidence_pack_integrity_mod.MANIFEST_SIGNATURE_FILENAME


class EvidencePackStatus(IntEnum):
    OK = 0
    USAGE = 2
    MISSING = 3
    FORMAT = 4
    SIGNATURE = 5
    INTEGRITY = 6
    REPORTS = 7


@dataclass(frozen=True)
class EvidencePackResult:
    payload: dict[str, Any]
    status: EvidencePackStatus


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


_relative_file_paths = evidence_pack_support_mod._relative_file_paths
_write_checksums_file = evidence_pack_support_mod._write_checksums_file
_copy_file = evidence_pack_support_mod._copy_file
_verify_manifest_binds_checksums = (
    evidence_pack_support_mod._verify_manifest_binds_checksums
)
_verify_checksums = evidence_pack_support_mod._verify_checksums
_parse_checksums = evidence_pack_support_mod._parse_checksums
_verify_no_extra_files = evidence_pack_support_mod._verify_no_extra_files
_sign_manifest = evidence_pack_integrity_mod.sign_manifest
_validate_signing_key = evidence_pack_support_mod._validate_signing_key
_generate_signing_keypair = evidence_pack_integrity_mod.generate_signing_keypair


def _verify_signature(
    pack_dir: Path, *, strict: bool
) -> tuple[list[str], list[str], str | None]:
    return evidence_pack_integrity_mod.verify_signature(
        pack_dir,
        strict=strict,
        load_json_fn=_load_json,
    )


_signature_warnings_to_errors = evidence_pack_integrity_mod.signature_warnings_to_errors


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


_collect_build_evidence_pack_errors = (
    evidence_pack_support_mod._collect_build_evidence_pack_errors
)
_copy_build_evidence_pack_artifacts = (
    evidence_pack_support_mod._copy_build_evidence_pack_artifacts
)
_build_evidence_pack_manifest = evidence_pack_support_mod._build_evidence_pack_manifest
inspect_evidence_pack = evidence_pack_support_mod.inspect_evidence_pack


def build_evidence_pack(
    out_dir: Path,
    *,
    final_verdict_path: Path,
    report_paths: list[Path],
    source_repo_path: Path | None = None,
    environment_path: Path | None = None,
    material_specs: list[tuple[str, Path]] | None = None,
    readme_path: Path | None = None,
    signing_key_path: Path | None = None,
    profile: str = "dev",
) -> EvidencePackResult:
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
        errors.append("evidence-pack build requires at least one --report input.")
        return EvidencePackResult(payload=payload, status=EvidencePackStatus.USAGE)
    if out_dir.exists():
        errors.append(f"Output pack directory already exists: {out_dir}")
        return EvidencePackResult(payload=payload, status=EvidencePackStatus.USAGE)

    errors.extend(
        _collect_build_evidence_pack_errors(
            out_dir=out_dir,
            final_verdict_path=final_verdict_path,
            report_paths=report_paths,
            source_repo_path=source_repo_path,
            environment_path=environment_path,
            material_specs=material_specs,
            signing_key_path=signing_key_path,
        )
    )
    if errors:
        return EvidencePackResult(payload=payload, status=EvidencePackStatus.FORMAT)

    verify_result = _run_verify_command(report_paths, profile=profile)
    if not _verify_command_succeeded(verify_result):
        payload["verify"] = verify_result.payload
        errors.append("Provided report inputs failed `invarlock verify`.")
        return EvidencePackResult(payload=payload, status=EvidencePackStatus.REPORTS)

    out_dir.mkdir(parents=True, exist_ok=False)
    final_dest, rel_paths, material_refs = _copy_build_evidence_pack_artifacts(
        out_dir=out_dir,
        final_verdict_path=final_verdict_path,
        report_paths=report_paths,
        source_repo_path=source_repo_path,
        environment_path=environment_path,
        material_specs=material_specs,
    )

    signer_fingerprint: str | None = None
    if signing_key_path is not None:
        signer_fingerprint = evidence_pack_integrity_mod.public_key_fingerprint(
            evidence_pack_integrity_mod.load_private_signing_key(
                signing_key_path
            ).public_key()
        )

    verification_summary = {
        "clean_reports": len(report_paths),
        "error_injection_reports": 0,
        "failed_reports": 0,
        "policy_profile": profile,
    }
    evidence_level = _derive_evidence_pack_evidence_level(
        subject_present=True,
        checksums_bound=True,
        clean_reports=len(report_paths),
        failed_reports=0,
        has_source_repo_ref=source_repo_path is not None,
        has_environment_ref=environment_path is not None,
    )

    if readme_path is not None:
        if not readme_path.is_file():
            warnings.append(f"README file not found; skipping copy: {readme_path}")
        else:
            readme_dest = out_dir / "README.md"
            _copy_file(readme_path, readme_dest)
            rel_paths.append("README.md")
    else:
        readme_dest = out_dir / "README.md"
        readme_dest.write_text(
            _render_evidence_pack_readme(
                evidence_level=evidence_level,
                clean_reports=len(report_paths),
                error_reports=0,
                failed_reports=0,
                policy_profile=profile,
                strict_ready=signing_key_path is not None,
                signer_fingerprint=signer_fingerprint,
            ),
            encoding="utf-8",
        )
        rel_paths.append("README.md")

    _write_checksums_file(out_dir, rel_paths)
    manifest = _build_evidence_pack_manifest(
        evidence_level=evidence_level,
        final_dest=final_dest,
        out_dir=out_dir,
        verification_summary=verification_summary,
        source_repo_path=source_repo_path,
        environment_path=environment_path,
        material_refs=material_refs,
        signing_key_path=signing_key_path,
        signer_fingerprint=signer_fingerprint,
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    if signing_key_path is not None:
        _sign_manifest(out_dir / "manifest.json", signing_key_path=signing_key_path)
    else:
        warnings.append(
            "No signing key provided; pack is unsigned and default verification will fail closed."
        )

    payload["ok"] = True
    payload["reports"] = {"total": len(report_paths)}
    payload["verify"] = verify_result.payload
    payload["evidence_level"] = evidence_level
    payload["files"] = {
        "hashed": len(rel_paths),
        "manifest": "manifest.json",
        "checksums": "checksums.sha256",
    }
    if signing_key_path is not None and signer_fingerprint is not None:
        payload["signature"] = {
            "present": True,
            "file": MANIFEST_SIGNATURE_FILENAME,
            "signer_fingerprint": signer_fingerprint,
        }
    else:
        payload["signature"] = {
            "present": False,
            "file": None,
            "signer_fingerprint": None,
        }
    return EvidencePackResult(payload=payload, status=EvidencePackStatus.OK)


def verify_evidence_pack(
    pack_dir: Path,
    *,
    json_out_path: Path | None = None,
    skip_verify: bool = False,
    strict: bool = False,
    profile: str = "dev",
) -> EvidencePackResult:
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
            status=EvidencePackStatus.MISSING,
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
            status=EvidencePackStatus.MISSING,
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
            status=EvidencePackStatus.MISSING,
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
            status=EvidencePackStatus.USAGE,
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
            status=EvidencePackStatus.FORMAT,
        )

    signature_errors, signature_warnings, signer_fingerprint = _verify_signature(
        pack_dir, strict=strict
    )
    if signature_warnings and not strict and not unverified_provenance_allowed():
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
            status=EvidencePackStatus.SIGNATURE,
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
            status=EvidencePackStatus.SIGNATURE,
        )

    errors.extend(_verify_manifest_binds_checksums(pack_dir))
    checksum_errors, covered_paths = _verify_checksums(pack_dir)
    errors.extend(checksum_errors)
    errors.extend(verify_manifest_provenance(pack_dir))
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
            status=EvidencePackStatus.INTEGRITY,
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
                status=EvidencePackStatus.REPORTS,
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
        status=EvidencePackStatus.OK,
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
    status: EvidencePackStatus,
) -> EvidencePackResult:
    evidence_level: str | None = None
    manifest_path = pack_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = _load_json(manifest_path)
        except _json_load_error_types():
            manifest = None
        if isinstance(manifest, dict):
            raw_level = manifest.get("evidence_level")
            if isinstance(raw_level, str):
                evidence_level = raw_level
    payload: dict[str, Any] = {
        "pack": str(pack_dir),
        "ok": ok,
        "strict": strict,
        "skip_verify": skip_verify,
        "warnings": warnings,
        "errors": errors,
        "evidence_level": evidence_level,
    }
    if signer_fingerprint:
        payload["signer_fingerprint"] = signer_fingerprint
    if verify_payload is not None:
        payload["verify"] = verify_payload
    return EvidencePackResult(payload=payload, status=status)


__all__ = [
    "EVIDENCE_PACK_FORMAT",
    "EvidencePackResult",
    "EvidencePackStatus",
    "validate_manifest",
    "verify_manifest_provenance",
    "verify_evidence_pack",
]
