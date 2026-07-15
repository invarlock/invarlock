from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

from invarlock import evidence_pack_integrity as evidence_pack_integrity_mod

EVIDENCE_PACK_FORMAT = evidence_pack_integrity_mod.EVIDENCE_PACK_FORMAT
_load_json = evidence_pack_integrity_mod._load_json
_json_load_error_types = evidence_pack_integrity_mod._json_load_error_types
_manual_validate_manifest = evidence_pack_integrity_mod._manual_validate_manifest
validate_manifest = evidence_pack_integrity_mod.validate_manifest
validate_manifest_payload = evidence_pack_integrity_mod.validate_manifest_payload
_relative_file_paths = evidence_pack_integrity_mod.relative_file_paths
_verify_manifest_binds_checksums = (
    evidence_pack_integrity_mod.verify_manifest_binds_checksums
)
_verify_manifest_binds_checksums_payload = (
    evidence_pack_integrity_mod.verify_manifest_binds_checksums_payload
)
_verify_checksums = evidence_pack_integrity_mod.verify_checksums
_parse_checksums = evidence_pack_integrity_mod.parse_checksums
_verify_no_extra_files = evidence_pack_integrity_mod.verify_no_extra_files
_verify_control_file_mirrors = evidence_pack_integrity_mod.verify_control_file_mirrors
verify_manifest_provenance = evidence_pack_integrity_mod.verify_manifest_provenance
verify_manifest_provenance_payload = (
    evidence_pack_integrity_mod.verify_manifest_provenance_payload
)
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


class EvidencePackStatus(IntEnum):
    OK = 0
    USAGE = 2
    MISSING = 3
    FORMAT = 4
    SIGNATURE = 5
    INTEGRITY = 6
    REPORTS = 7
    INTEGRITY_ONLY = 8


@dataclass(frozen=True)
class EvidencePackResult:
    payload: dict[str, Any]
    status: EvidencePackStatus


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

    try:
        manifest = _load_json(manifest_path)
    except _json_load_error_types() as exc:
        issues.append(f"manifest is not valid JSON: {exc}")
        return EvidencePackResult(payload=payload, status=EvidencePackStatus.FORMAT)
    manifest_errors = validate_manifest_payload(manifest)
    if manifest_errors:
        issues.extend(manifest_errors)
        return EvidencePackResult(payload=payload, status=EvidencePackStatus.FORMAT)
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

    bind_errors = _verify_manifest_binds_checksums_payload(
        manifest,
        checksums_path.read_bytes(),
    )
    checksum_errors, covered_paths = _verify_checksums(pack_dir)
    mirror_errors = _verify_control_file_mirrors(pack_dir)
    provenance_errors = verify_manifest_provenance_payload(pack_dir, manifest)
    relative_paths = _relative_file_paths(pack_dir)
    extra_files = sorted(set(relative_paths) - covered_paths - _CONTROL_FILES)
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
        "files": len(relative_paths),
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
