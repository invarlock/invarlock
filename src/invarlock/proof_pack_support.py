from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from invarlock import proof_pack_integrity as proof_pack_integrity_mod
from invarlock import proof_pack_manifest as proof_pack_manifest_mod
from invarlock import proof_pack_metadata as proof_pack_metadata_mod
from invarlock.runtime_security import RUNTIME_MANIFEST_FILENAME

if TYPE_CHECKING:
    from invarlock.proof_pack import ProofPackResult

PROOF_PACK_FORMAT = proof_pack_manifest_mod.PROOF_PACK_FORMAT
_load_json = proof_pack_manifest_mod._load_json
_load_json_object = proof_pack_manifest_mod._load_json_object
_manual_validate_manifest = proof_pack_manifest_mod._manual_validate_manifest
_relative_file_paths = proof_pack_integrity_mod.relative_file_paths
_write_checksums_file = proof_pack_integrity_mod.write_checksums_file
_copy_file = proof_pack_integrity_mod.copy_file
_verify_manifest_binds_checksums = (
    proof_pack_integrity_mod.verify_manifest_binds_checksums
)
_verify_checksums = proof_pack_integrity_mod.verify_checksums
_parse_checksums = proof_pack_integrity_mod.parse_checksums
_verify_no_extra_files = proof_pack_integrity_mod.verify_no_extra_files
_validate_signing_key = proof_pack_integrity_mod.validate_signing_key
_sha256_bytes = proof_pack_manifest_mod._sha256_bytes
_sha256_file = proof_pack_manifest_mod._sha256_file
_validate_material_name = proof_pack_manifest_mod._validate_material_name
verify_manifest_attestation = proof_pack_manifest_mod.verify_manifest_attestation
_proof_pack_counts_from_verification = (
    proof_pack_metadata_mod._proof_pack_counts_from_verification
)
_derive_proof_pack_evidence_level = (
    proof_pack_metadata_mod._derive_proof_pack_evidence_level
)
_CONTROL_FILES = proof_pack_integrity_mod.CONTROL_FILES
MANIFEST_SIGNATURE_FILENAME = proof_pack_integrity_mod.MANIFEST_SIGNATURE_FILENAME


def _collect_build_proof_pack_errors(
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
    return errors


def _copy_build_proof_pack_artifacts(
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

    return final_dest, rel_paths, material_refs


def _build_proof_pack_manifest(
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
    return manifest


def inspect_proof_pack(pack_dir: Path) -> ProofPackResult:
    from invarlock.proof_pack import ProofPackResult, ProofPackStatus, validate_manifest

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
