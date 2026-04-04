from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

from invarlock import proof_pack_integrity as proof_pack_integrity_mod
from invarlock import proof_pack_manifest as proof_pack_manifest_mod
from invarlock.reporting.verify_contract import (
    VerifyExecutionResult,
    VerifyOutcome,
    run_verify_reports,
)
from invarlock.runtime_security import (
    RUNTIME_MANIFEST_FILENAME,
    unattested_artifacts_allowed,
)

PROOF_PACK_FORMAT = proof_pack_manifest_mod.PROOF_PACK_FORMAT
jsonschema = proof_pack_manifest_mod.jsonschema
load_proof_pack_manifest_schema = (
    proof_pack_manifest_mod.load_proof_pack_manifest_schema
)
_load_json = proof_pack_manifest_mod._load_json
_json_load_error_types = proof_pack_manifest_mod._json_load_error_types
_load_json_object = proof_pack_manifest_mod._load_json_object
_manual_validate_manifest = proof_pack_manifest_mod._manual_validate_manifest
_material_spec = proof_pack_manifest_mod._material_spec
_normalize_pack_path = proof_pack_manifest_mod._normalize_pack_path
_path_within_dir = proof_pack_manifest_mod._path_within_dir
_sha256_bytes = proof_pack_manifest_mod._sha256_bytes
_sha256_file = proof_pack_manifest_mod._sha256_file
_validate_material_name = proof_pack_manifest_mod._validate_material_name
_validate_reference = proof_pack_manifest_mod._validate_reference
verify_manifest_attestation = proof_pack_manifest_mod.verify_manifest_attestation
_CONTROL_FILES = proof_pack_integrity_mod.CONTROL_FILES
MANIFEST_SIGNATURE_FILENAME = proof_pack_integrity_mod.MANIFEST_SIGNATURE_FILENAME


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


def _proof_pack_counts_from_verification(
    verification: dict[str, Any] | None,
) -> tuple[int | None, int | None, int | None]:
    if not isinstance(verification, dict):
        return None, None, None
    clean_reports = verification.get("clean_reports")
    error_reports = verification.get("error_injection_reports")
    failed_reports = verification.get("failed_reports")
    return (
        int(clean_reports) if isinstance(clean_reports, int) else None,
        int(error_reports) if isinstance(error_reports, int) else None,
        int(failed_reports) if isinstance(failed_reports, int) else None,
    )


def _derive_proof_pack_evidence_level(
    *,
    subject_present: bool,
    checksums_bound: bool,
    clean_reports: int | None,
    failed_reports: int | None,
    has_source_repo_ref: bool,
    has_environment_ref: bool,
) -> str:
    if (
        subject_present
        and checksums_bound
        and isinstance(clean_reports, int)
        and clean_reports > 0
        and failed_reports == 0
        and has_source_repo_ref
        and has_environment_ref
    ):
        return "high"
    if (
        subject_present
        and checksums_bound
        and isinstance(clean_reports, int)
        and clean_reports > 0
    ):
        return "medium"
    return "low"


def _render_proof_pack_readme(
    *,
    evidence_level: str,
    clean_reports: int | None,
    error_reports: int | None,
    failed_reports: int | None,
    policy_profile: str | None,
    strict_ready: bool,
    signer_fingerprint: str | None,
) -> str:
    lines = [
        "# InvarLock Proof Pack",
        "",
        "This proof pack bundles reports, summary reports, and metadata for offline",
        "verification. No model weights are included.",
        "",
        f"Evidence level: {evidence_level}",
        (
            "Review summary: "
            f"clean_reports={clean_reports if clean_reports is not None else 'unknown'}, "
            f"error_injection_reports={error_reports if error_reports is not None else 'unknown'}, "
            f"failed_reports={failed_reports if failed_reports is not None else 'unknown'}, "
            f"profile={policy_profile or 'unknown'}."
        ),
        "",
        "Why it might be wrong:",
    ]
    if failed_reports not in (None, 0):
        lines.append(
            "- Unexpected report verification failures were recorded; inspect results/verification_summary.json before trusting downstream conclusions."
        )
    else:
        lines.append(
            "- Nested report verification succeeded for the bundled clean reports, but reviewers should still inspect the underlying evaluation.report.json files."
        )
    lines.append(
        "- Error-injection reports are expected-failure evidence and should not be interpreted as clean PASS runs."
    )
    if strict_ready:
        lines.append(
            "- The pack is ready for strict verification; signed manifest and checksum sealing are present."
        )
    else:
        lines.append(
            "- By default this is evidence-grade packaging. For proof-grade attestation, require a signed manifest, strict verification, and a PASS final verdict."
        )
    if signer_fingerprint:
        lines.append(f"- Signer fingerprint: {signer_fingerprint}")

    lines.extend(
        [
            "",
            "## Verify",
            "",
            "1. Verify the manifest signature and strict pack integrity:",
            "   invarlock advanced proof-pack verify <pack-dir> --strict",
            "",
            "2. Verify file checksums:",
            "   sha256sum -c checksums.sha256",
            "   # macOS: shasum -a 256 -c checksums.sha256",
            "",
            "3. Verify report integrity:",
            "   invarlock verify --json reports/**/evaluation.report.json",
            "",
            "Or use:",
            "  invarlock advanced proof-pack verify <pack-dir> [--strict]",
            "Repo workflow alternative:",
            "  scripts/proof_packs/verify_pack.sh --pack <pack-dir> [--strict]",
        ]
    )
    return "\n".join(lines) + "\n"


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
    return proof_pack_integrity_mod.relative_file_paths(pack_dir)


def _write_checksums_file(pack_dir: Path, rel_paths: list[str]) -> None:
    proof_pack_integrity_mod.write_checksums_file(pack_dir, rel_paths)


def _copy_file(source_path: Path, dest_path: Path) -> None:
    proof_pack_integrity_mod.copy_file(source_path, dest_path)


def _verify_manifest_binds_checksums(pack_dir: Path) -> list[str]:
    return proof_pack_integrity_mod.verify_manifest_binds_checksums(pack_dir)


def _verify_checksums(pack_dir: Path) -> tuple[list[str], set[str]]:
    return proof_pack_integrity_mod.verify_checksums(pack_dir)


def _parse_checksums(pack_dir: Path) -> tuple[list[tuple[str, str]], list[str]]:
    return proof_pack_integrity_mod.parse_checksums(pack_dir)


def _verify_no_extra_files(
    pack_dir: Path, *, covered_paths: set[str], strict: bool
) -> tuple[list[str], list[str]]:
    return proof_pack_integrity_mod.verify_no_extra_files(
        pack_dir,
        covered_paths=covered_paths,
        strict=strict,
    )


def _sign_manifest(manifest_path: Path, *, signing_key_path: Path) -> str:
    return proof_pack_integrity_mod.sign_manifest(
        manifest_path,
        signing_key_path=signing_key_path,
    )


def _validate_signing_key(path: Path) -> list[str]:
    return proof_pack_integrity_mod.validate_signing_key(path)


def _generate_signing_keypair(
    private_key_path: Path,
    *,
    public_key_path: Path,
) -> str:
    return proof_pack_integrity_mod.generate_signing_keypair(
        private_key_path,
        public_key_path=public_key_path,
    )


def _verify_signature(
    pack_dir: Path, *, strict: bool
) -> tuple[list[str], list[str], str | None]:
    return proof_pack_integrity_mod.verify_signature(
        pack_dir,
        strict=strict,
        load_json_fn=_load_json,
    )


def _signature_warnings_to_errors(warnings: list[str]) -> list[str]:
    return proof_pack_integrity_mod.signature_warnings_to_errors(warnings)


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
        "manifest": {"valid": False, "format": None, "evidence_level": None},
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
        "evidence_level": None,
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
        "evidence_level": (
            manifest.get("evidence_level") if isinstance(manifest, dict) else None
        ),
    }

    signature_present = (pack_dir / MANIFEST_SIGNATURE_FILENAME).is_file()
    payload["signature"] = {
        "present": signature_present,
        "signer_fingerprint": (
            manifest.get("signing_key_fingerprint")
            if isinstance(manifest, dict)
            else None
        ),
    }
    if not signature_present:
        issues.append(
            f"{MANIFEST_SIGNATURE_FILENAME} missing; strict verification would fail."
        )

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
    verification = manifest.get("verification") if isinstance(manifest, dict) else None
    clean_count, _error_count, failed_count = _proof_pack_counts_from_verification(
        verification if isinstance(verification, dict) else None
    )
    payload["evidence_level"] = (
        manifest.get("evidence_level")
        if isinstance(manifest, dict)
        and isinstance(manifest.get("evidence_level"), str)
        else _derive_proof_pack_evidence_level(
            subject_present=bool(
                isinstance(manifest, dict) and isinstance(manifest.get("subject"), dict)
            ),
            checksums_bound=not bind_errors,
            clean_reports=(
                clean_count if clean_count is not None else payload["reports"]["clean"]
            ),
            failed_reports=failed_count,
            has_source_repo_ref=bool(
                isinstance(manifest, dict)
                and isinstance(manifest.get("invocation"), dict)
                and isinstance(manifest["invocation"].get("config_source"), dict)
                and manifest["invocation"]["config_source"].get("path")
                and manifest["invocation"]["config_source"].get("digest")
            ),
            has_environment_ref=bool(
                isinstance(manifest, dict)
                and isinstance(manifest.get("environment"), dict)
                and manifest["environment"].get("path")
                and manifest["environment"].get("digest")
            ),
        )
    )
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
    signing_key_path: Path | None = None,
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
    if signing_key_path is not None:
        errors.extend(_validate_signing_key(signing_key_path))
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

    signer_fingerprint: str | None = None
    if signing_key_path is not None:
        signer_fingerprint = proof_pack_integrity_mod.public_key_fingerprint(
            proof_pack_integrity_mod.load_private_signing_key(
                signing_key_path
            ).public_key()
        )

    verification_summary = {
        "clean_reports": len(report_paths),
        "error_injection_reports": 0,
        "failed_reports": 0,
        "policy_profile": profile,
    }
    evidence_level = _derive_proof_pack_evidence_level(
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
            _render_proof_pack_readme(
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
    manifest: dict[str, Any] = {
        "format": PROOF_PACK_FORMAT,
        "evidence_level": evidence_level,
        "checksums_sha256": "checksums.sha256",
        "checksums_sha256_digest": _sha256_bytes(
            (out_dir / "checksums.sha256").read_bytes()
        ),
        "subject": {
            "name": "final_verdict",
            "path": "results/final_verdict.json",
            "digest": _sha256_file(final_dest),
        },
        "verification": verification_summary,
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
    if signing_key_path is not None:
        manifest["signing_key_fingerprint"] = signer_fingerprint
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

    signature_errors, signature_warnings, signer_fingerprint = _verify_signature(
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
    return ProofPackResult(payload=payload, status=status)


__all__ = [
    "PROOF_PACK_FORMAT",
    "ProofPackResult",
    "ProofPackStatus",
    "validate_manifest",
    "verify_manifest_attestation",
    "verify_proof_pack",
]
