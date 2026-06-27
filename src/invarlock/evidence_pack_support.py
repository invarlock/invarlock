from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

from invarlock import evidence_pack_integrity as evidence_pack_integrity_mod
from invarlock.runtime_security import RUNTIME_MANIFEST_FILENAME

EVIDENCE_PACK_FORMAT = evidence_pack_integrity_mod.EVIDENCE_PACK_FORMAT
_load_json = evidence_pack_integrity_mod._load_json
_load_json_object = evidence_pack_integrity_mod._load_json_object
_manual_validate_manifest = evidence_pack_integrity_mod._manual_validate_manifest
validate_manifest = evidence_pack_integrity_mod.validate_manifest
_relative_file_paths = evidence_pack_integrity_mod.relative_file_paths
_write_checksums_file = evidence_pack_integrity_mod.write_checksums_file
_copy_file = evidence_pack_integrity_mod.copy_file
_verify_manifest_binds_checksums = (
    evidence_pack_integrity_mod.verify_manifest_binds_checksums
)
_verify_checksums = evidence_pack_integrity_mod.verify_checksums
_parse_checksums = evidence_pack_integrity_mod.parse_checksums
_verify_no_extra_files = evidence_pack_integrity_mod.verify_no_extra_files
_verify_control_file_mirrors = evidence_pack_integrity_mod.verify_control_file_mirrors
_validate_signing_key = evidence_pack_integrity_mod.validate_signing_key
_sha256_bytes = evidence_pack_integrity_mod._sha256_bytes
_sha256_file = evidence_pack_integrity_mod._sha256_file
_validate_material_name = evidence_pack_integrity_mod._validate_material_name
verify_manifest_provenance = evidence_pack_integrity_mod.verify_manifest_provenance
_CONTROL_FILES = evidence_pack_integrity_mod.CONTROL_FILES
MANIFEST_SIGNATURE_FILENAME = evidence_pack_integrity_mod.MANIFEST_SIGNATURE_FILENAME


def _evidence_pack_counts_from_verification(
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


def _derive_evidence_pack_evidence_level(
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


def _render_evidence_pack_readme(
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
        "# InvarLock Evidence Pack",
        "",
        "This evidence pack bundles reports, summary reports, and metadata for offline",
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
            "- Unexpected report verification failures were recorded; inspect results/verification_summary.json before trusting final conclusions."
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
            "- By default this is evidence-grade packaging. For strong distributable evidence, require a signed manifest, strict verification, and a PASS final verdict."
        )
    if signer_fingerprint:
        lines.append(f"- Signer fingerprint: {signer_fingerprint}")

    lines.extend(
        [
            "",
            "## Verify",
            "",
            "1. Verify the manifest signature and strict pack integrity:",
            "   invarlock advanced evidence-pack verify <pack-dir> --strict",
            "",
            "2. Verify file checksums:",
            "   sha256sum -c checksums.sha256",
            "   # macOS: shasum -a 256 -c checksums.sha256",
            "",
            "3. Verify report integrity:",
            "   invarlock verify --json reports/**/evaluation.report.json",
            "",
            "Or use:",
            "  invarlock advanced evidence-pack verify <pack-dir> --strict",
            "Repo workflow alternative:",
            "  scripts/evidence_packs/verify_pack.sh --pack <pack-dir> --strict",
        ]
    )
    return "\n".join(lines) + "\n"


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


def _collect_build_evidence_pack_errors(
    *,
    out_dir: Path,
    final_verdict_path: Path,
    report_paths: list[Path],
    source_repo_path: Path | None,
    environment_path: Path | None,
    material_specs: list[tuple[str, Path]],
    signing_key_path: Path | None,
) -> list[str]:
    errors: list[str] = []
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
        for sidecar_name in (
            "edit_metadata.json",
            "deployable_artifact_validation.json",
            "backend_inventory.json",
            "memory_report.json",
            "load_smoke.json",
            "inference_smoke.json",
        ):
            sidecar_path = report_path.parent / sidecar_name
            if sidecar_path.is_file():
                _, sidecar_errors = _load_json_object(
                    sidecar_path,
                    label=sidecar_name,
                )
                errors.extend(sidecar_errors)
    return errors


def _copy_build_evidence_pack_artifacts(
    *,
    out_dir: Path,
    final_verdict_path: Path,
    report_paths: list[Path],
    source_repo_path: Path | None,
    environment_path: Path | None,
    material_specs: list[tuple[str, Path]],
) -> tuple[Path, list[str], list[dict[str, Any]]]:
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
        for sidecar_name in (
            "edit_metadata.json",
            "deployable_artifact_validation.json",
            "backend_inventory.json",
            "memory_report.json",
            "load_smoke.json",
            "inference_smoke.json",
        ):
            sidecar_path = report_path.parent / sidecar_name
            if sidecar_path.is_file():
                sidecar_rel = f"{report_dir_rel}/{sidecar_name}"
                _copy_file(sidecar_path, out_dir / sidecar_rel)
                rel_paths.append(sidecar_rel)

    return final_dest, rel_paths, material_refs


def _build_evidence_pack_manifest(
    *,
    evidence_level: str,
    final_dest: Path,
    out_dir: Path,
    verification_summary: dict[str, Any],
    source_repo_path: Path | None,
    environment_path: Path | None,
    material_refs: list[dict[str, Any]],
    signing_key_path: Path | None,
    signer_fingerprint: str | None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "format": EVIDENCE_PACK_FORMAT,
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
    return manifest


def inspect_evidence_pack(pack_dir: Path) -> EvidencePackResult:
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
            "manifest_provenance_ok": False,
            "extra_files": [],
        },
        "issues": issues,
        "strict_ready": False,
        "evidence_level": None,
    }
    if not pack_dir.is_dir():
        issues.append(f"Pack directory not found: {pack_dir}")
        return EvidencePackResult(payload=payload, status=EvidencePackStatus.MISSING)
    manifest_path = pack_dir / "manifest.json"
    checksums_path = pack_dir / "checksums.sha256"
    if not manifest_path.is_file():
        issues.append("manifest.json missing in pack.")
        return EvidencePackResult(payload=payload, status=EvidencePackStatus.MISSING)
    if not checksums_path.is_file():
        issues.append("checksums.sha256 missing in pack.")
        return EvidencePackResult(payload=payload, status=EvidencePackStatus.MISSING)

    manifest_errors = validate_manifest(manifest_path)
    if manifest_errors:
        issues.extend(manifest_errors)
        return EvidencePackResult(payload=payload, status=EvidencePackStatus.FORMAT)

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
    mirror_errors = _verify_control_file_mirrors(pack_dir)
    provenance_errors = verify_manifest_provenance(pack_dir)
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
    issues.extend(mirror_errors)
    issues.extend(provenance_errors)

    payload["artifacts"] = {
        "files": len(_relative_file_paths(pack_dir)),
        "hashed": len(covered_paths),
    }
    payload["integrity"] = {
        "checksums_bound": not bind_errors,
        "checksums_ok": not checksum_errors,
        "control_file_mirrors_ok": not mirror_errors,
        "manifest_provenance_ok": not provenance_errors,
        "extra_files": extra_files,
    }
    verification = manifest.get("verification") if isinstance(manifest, dict) else None
    clean_count, _error_count, failed_count = _evidence_pack_counts_from_verification(
        verification if isinstance(verification, dict) else None
    )
    payload["evidence_level"] = (
        manifest.get("evidence_level")
        if isinstance(manifest, dict)
        and isinstance(manifest.get("evidence_level"), str)
        else _derive_evidence_pack_evidence_level(
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
        bind_errors or checksum_errors or provenance_errors or extra_files
    )
    payload["ok"] = not integrity_errors_present
    payload["strict_ready"] = (
        signature_present
        and not bind_errors
        and not checksum_errors
        and not provenance_errors
        and not extra_files
    )
    return EvidencePackResult(
        payload=payload,
        status=(
            EvidencePackStatus.INTEGRITY
            if integrity_errors_present
            else EvidencePackStatus.OK
        ),
    )
