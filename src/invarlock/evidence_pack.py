from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from invarlock import evidence_pack_edit_metadata as evidence_pack_edit_metadata_mod
from invarlock import evidence_pack_integrity as evidence_pack_integrity_mod
from invarlock import evidence_pack_support as evidence_pack_support_mod
from invarlock.reporting.verify_contract import (
    VerifyExecutionResult,
    VerifyOutcome,
    run_verify_reports,
)
from invarlock.runtime_security import unverified_provenance_allowed

EVIDENCE_PACK_FORMAT = evidence_pack_integrity_mod.EVIDENCE_PACK_FORMAT
_load_json = evidence_pack_integrity_mod._load_json
_json_load_error_types = evidence_pack_integrity_mod._json_load_error_types
_load_json_object = evidence_pack_integrity_mod._load_json_object
_manual_validate_manifest = evidence_pack_integrity_mod._manual_validate_manifest
_material_spec = evidence_pack_integrity_mod._material_spec
_normalize_pack_path = evidence_pack_integrity_mod._normalize_pack_path
_path_within_dir = evidence_pack_integrity_mod._path_within_dir
_sha256_bytes = evidence_pack_integrity_mod._sha256_bytes
_sha256_file = evidence_pack_integrity_mod._sha256_file
_validate_material_name = evidence_pack_integrity_mod._validate_material_name
_validate_reference = evidence_pack_integrity_mod._validate_reference
validate_manifest = evidence_pack_integrity_mod.validate_manifest
verify_manifest_provenance = evidence_pack_integrity_mod.verify_manifest_provenance
_evidence_pack_counts_from_verification = (
    evidence_pack_support_mod._evidence_pack_counts_from_verification
)
_derive_evidence_pack_evidence_level = (
    evidence_pack_support_mod._derive_evidence_pack_evidence_level
)
_render_evidence_pack_readme = evidence_pack_support_mod._render_evidence_pack_readme
_CONTROL_FILES = evidence_pack_integrity_mod.CONTROL_FILES
MANIFEST_SIGNATURE_FILENAME = evidence_pack_integrity_mod.MANIFEST_SIGNATURE_FILENAME
EvidencePackStatus = evidence_pack_support_mod.EvidencePackStatus
EvidencePackResult = evidence_pack_support_mod.EvidencePackResult
RunVerifyCommand = Callable[..., VerifyExecutionResult]


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

_signature_warnings_to_errors = evidence_pack_integrity_mod.signature_warnings_to_errors


def load_trust_store_fingerprints(
    trust_store_path: Path | None,
) -> tuple[set[str], list[str], str | None]:
    """Load trusted evidence-pack signer fingerprints from a JSON trust store."""
    path = trust_store_path
    if path is None:
        default_path = evidence_pack_integrity_mod.DEFAULT_TRUST_STORE_PATH
        path = default_path if default_path.is_file() else None
    if path is None:
        return set(), [], None
    if not path.is_file():
        return set(), [f"Evidence-pack trust store not found: {path}"], str(path)
    try:
        payload = _load_json(path)
    except _json_load_error_types() as exc:
        return set(), [f"Evidence-pack trust store is not valid JSON: {exc}"], str(path)

    raw_entries: list[Any]
    if isinstance(payload, list):
        raw_entries = list(payload)
    elif isinstance(payload, dict):
        raw = payload.get("trusted_signers", payload.get("fingerprints", []))
        if not isinstance(raw, list):
            return (
                set(),
                ["Evidence-pack trust store trusted_signers must be a list."],
                str(path),
            )
        raw_entries = raw
    else:
        return (
            set(),
            ["Evidence-pack trust store must be a JSON object or list."],
            str(path),
        )

    fingerprints: set[str] = set()
    errors: list[str] = []
    for index, entry in enumerate(raw_entries):
        raw_value = entry.get("fingerprint") if isinstance(entry, dict) else entry
        if not isinstance(raw_value, str):
            errors.append(f"Evidence-pack trust store entry {index} is not a string.")
            continue
        normalized = evidence_pack_integrity_mod.normalize_expected_fingerprint(
            raw_value
        )
        if normalized is None:
            errors.append(
                f"Evidence-pack trust store entry {index} is not a sha256 fingerprint."
            )
            continue
        fingerprints.add(normalized)
    if not fingerprints and not errors:
        errors.append("Evidence-pack trust store contains no trusted signers.")
    return fingerprints, errors, str(path)


def _verify_signature(
    pack_dir: Path,
    *,
    strict: bool,
    expected_fingerprints: set[str] | frozenset[str] | None = None,
) -> tuple[list[str], list[str], str | None]:
    return evidence_pack_integrity_mod.verify_signature(
        pack_dir,
        strict=strict,
        load_json_fn=_load_json,
        expected_fingerprints=expected_fingerprints,
    )


def _run_verify_command(
    reports: list[Path],
    *,
    profile: str,
    report_assurance: str = "report",
) -> VerifyExecutionResult:
    return run_verify_reports(
        reports,
        profile=profile,
        json_mode=True,
        assurance_mode=report_assurance,
    )


def _scenario_strictness_by_id(pack_dir: Path) -> dict[str, str]:
    scenarios_path = pack_dir / "metadata" / "scenarios.json"
    if not scenarios_path.is_file():
        return {}
    try:
        payload = json.loads(scenarios_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        return {}
    strictness_by_id: dict[str, str] = {}
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        scenario_id = scenario.get("id")
        strictness = scenario.get("strictness")
        if isinstance(scenario_id, str) and isinstance(strictness, str):
            strictness_by_id[scenario_id] = strictness
    return strictness_by_id


def _report_scenario_id(pack_dir: Path, report: Path) -> str | None:
    try:
        parts = report.relative_to(pack_dir / "reports").parts
    except ValueError:
        return None
    if len(parts) < 3:
        return None
    if parts[1] == "errors" and len(parts) >= 4:
        scenario = parts[2].strip()
        return scenario or None
    return parts[1]


def _is_error_injection_report(report: Path) -> bool:
    return "errors" in report.parts and report.name == "evaluation.report.json"


def _report_expects_verify_failure(
    pack_dir: Path,
    report: Path,
    *,
    strictness_by_id: dict[str, str],
) -> bool:
    is_error = _is_error_injection_report(report)
    scenario_id = _report_scenario_id(pack_dir, report)
    if scenario_id and scenario_id in strictness_by_id:
        strictness = strictness_by_id[scenario_id]
        return strictness == "must_fail"

    # Legacy packs did not always carry scenario metadata. Preserve the old
    # hard-fault behavior for unclassified reports under errors/.
    return is_error


def _verify_command_succeeded(result: VerifyExecutionResult) -> bool:
    return result.outcome == VerifyOutcome.OK


def _verify_reports(
    pack_dir: Path,
    *,
    json_out_path: Path | None,
    profile: str,
    report_assurance: str,
) -> tuple[list[str], dict[str, Any] | None]:
    reports = sorted(pack_dir.glob("reports/**/evaluation.report.json"))
    if not reports:
        return ["No reports found in pack."], None
    strictness_by_id = _scenario_strictness_by_id(pack_dir)
    expected_failure_reports = [
        path
        for path in reports
        if _report_expects_verify_failure(
            pack_dir, path, strictness_by_id=strictness_by_id
        )
    ]
    expected_pass_reports = [
        path for path in reports if path not in expected_failure_reports
    ]
    if not expected_pass_reports:
        return [
            "No reports expected to pass in pack (only expected-failure reports present)."
        ], None

    expected_pass_result = _run_verify_command(
        expected_pass_reports,
        profile=profile,
        report_assurance=report_assurance,
    )
    if not isinstance(expected_pass_result.payload, dict):
        return ["expected-pass report verification did not return a JSON object."], None
    verify_payload = dict(expected_pass_result.payload)
    expected_failure_payloads: list[dict[str, Any]] = []
    for report in expected_failure_reports:
        try:
            expected_failure_result = _run_verify_command(
                [report],
                profile=profile,
                report_assurance=report_assurance,
            )
        except (
            ImportError,
            ModuleNotFoundError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            return [
                f"expected-failure report verification failed unexpectedly: {exc}"
            ], verify_payload
        if not isinstance(expected_failure_result.payload, dict):
            return [
                "expected-failure report verification did not return a JSON object."
            ], verify_payload
        if _verify_command_succeeded(expected_failure_result):
            rel_report = str(report.relative_to(pack_dir)).replace("\\", "/")
            return [
                f"expected-failure report verified as passing: {rel_report}"
            ], verify_payload
        expected_failure_payloads.append(expected_failure_result.payload)
    if expected_failure_reports:
        verify_payload["expected_failures"] = {
            "verify": expected_failure_payloads,
            "reports": [
                str(path.relative_to(pack_dir)).replace("\\", "/")
                for path in expected_failure_reports
            ],
        }
    if json_out_path is not None:
        json_out_path.write_text(
            json.dumps(verify_payload, sort_keys=True) + "\n", encoding="utf-8"
        )
    if not _verify_command_succeeded(expected_pass_result):
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

_verify_edit_metadata_consistency = (
    evidence_pack_edit_metadata_mod._verify_edit_metadata_consistency
)


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
    report_assurance: str = "report",
    release_review: bool = False,
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
    if report_assurance not in {"report", "strict", "off"}:
        errors.append(
            "Report assurance must be one of report, strict, or off "
            f"(got {report_assurance!r})."
        )
        return EvidencePackResult(payload=payload, status=EvidencePackStatus.USAGE)
    if release_review:
        normalized_profile = profile.strip().lower() if isinstance(profile, str) else ""
        if not normalized_profile:
            errors.append("release-review build requires an explicit profile.")
        elif normalized_profile == "dev":
            errors.append(
                "release-review build rejects profile=dev; use ci or release."
            )
        elif normalized_profile not in {"ci", "release"}:
            errors.append(
                "release-review build requires --profile ci or --profile release "
                f"(got {profile!r})."
            )
        if report_assurance != "strict":
            errors.append("release-review build requires --report-assurance strict.")
        if signing_key_path is None:
            errors.append("release-review build requires --signing-key.")
        try:
            verdict_payload = _load_json(final_verdict_path)
        except _json_load_error_types() as exc:
            errors.append(f"Final verdict is not valid JSON: {exc}")
        else:
            verdict = verdict_payload.get("verdict")
            if not isinstance(verdict, str) or verdict.strip().upper() != "PASS":
                errors.append("release-review build requires final verdict PASS.")
        if errors:
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

    verify_result = _run_verify_command(
        report_paths,
        profile=profile,
        report_assurance=report_assurance,
    )
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
        "report_assurance": report_assurance,
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
    payload["profile"] = profile
    payload["report_assurance"] = report_assurance
    payload["release_review"] = bool(release_review)
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
    report_assurance: str = "report",
    expected_fingerprint: str | None = None,
    trust_store_path: Path | None = None,
) -> EvidencePackResult:
    warnings: list[str] = []
    errors: list[str] = []
    verify_payload: dict[str, Any] | None = None
    signer_fingerprint: str | None = None
    authenticity = "unpinned"
    normalized_expected = None
    if expected_fingerprint is not None:
        normalized_expected = (
            evidence_pack_integrity_mod.normalize_expected_fingerprint(
                expected_fingerprint
            )
        )
        if normalized_expected is None:
            errors.append(
                "--expected-fingerprint must be a sha256:... signing key fingerprint "
                f"(got {expected_fingerprint!r})."
            )
    if report_assurance not in {"report", "strict", "off"}:
        errors.append(
            "--report-assurance must be one of: report, strict, off "
            f"(got {report_assurance!r})."
        )
    trust_fingerprints, trust_errors, trust_store_used = load_trust_store_fingerprints(
        trust_store_path
    )
    errors.extend(trust_errors)
    expected_fingerprints = set(trust_fingerprints)
    if normalized_expected is not None:
        expected_fingerprints.add(normalized_expected)
    expected_set = frozenset(expected_fingerprints) if expected_fingerprints else None
    if errors:
        return _build_verify_result(
            pack_dir=pack_dir,
            ok=False,
            strict=strict,
            skip_verify=skip_verify,
            report_assurance=report_assurance,
            warnings=warnings,
            errors=errors,
            signer_fingerprint=signer_fingerprint,
            verify_payload=verify_payload,
            status=EvidencePackStatus.USAGE,
            authenticity="mismatch" if expected_set else authenticity,
            trust_store_path=trust_store_used,
        )

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
            authenticity=authenticity,
            trust_store_path=trust_store_used,
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
            authenticity=authenticity,
            trust_store_path=trust_store_used,
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
            authenticity=authenticity,
            trust_store_path=trust_store_used,
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
            authenticity=authenticity,
            trust_store_path=trust_store_used,
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
            authenticity=authenticity,
            trust_store_path=trust_store_used,
        )

    if expected_set is None:
        signature_errors, signature_warnings, signer_fingerprint = _verify_signature(
            pack_dir,
            strict=strict,
        )
    else:
        signature_errors, signature_warnings, signer_fingerprint = _verify_signature(
            pack_dir,
            strict=strict,
            expected_fingerprints=expected_set,
        )
    if signer_fingerprint and expected_set and signer_fingerprint in expected_set:
        authenticity = "pinned"
    elif signer_fingerprint and expected_set:
        authenticity = "mismatch"
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
            authenticity=authenticity,
            trust_store_path=trust_store_used,
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
            authenticity=authenticity,
            trust_store_path=trust_store_used,
        )

    errors.extend(_verify_manifest_binds_checksums(pack_dir))
    checksum_errors, covered_paths = _verify_checksums(pack_dir)
    errors.extend(checksum_errors)
    errors.extend(evidence_pack_integrity_mod.verify_control_file_mirrors(pack_dir))
    errors.extend(verify_manifest_provenance(pack_dir))
    errors.extend(_verify_edit_metadata_consistency(pack_dir))
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
            authenticity=authenticity,
            trust_store_path=trust_store_used,
        )

    if not skip_verify:
        report_errors, verify_payload = _verify_reports(
            pack_dir,
            json_out_path=json_out_path,
            profile=profile,
            report_assurance=report_assurance,
        )
        if report_errors:
            errors.extend(report_errors)
            return _build_verify_result(
                pack_dir=pack_dir,
                ok=False,
                strict=strict,
                skip_verify=skip_verify,
                report_assurance=report_assurance,
                warnings=warnings,
                errors=errors,
                signer_fingerprint=signer_fingerprint,
                verify_payload=verify_payload,
                status=EvidencePackStatus.REPORTS,
                authenticity=authenticity,
                trust_store_path=trust_store_used,
            )

    return _build_verify_result(
        pack_dir=pack_dir,
        ok=True,
        strict=strict,
        skip_verify=skip_verify,
        report_assurance=report_assurance,
        warnings=warnings,
        errors=errors,
        signer_fingerprint=signer_fingerprint,
        verify_payload=verify_payload,
        status=EvidencePackStatus.OK,
        authenticity=authenticity,
        trust_store_path=trust_store_used,
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
    report_assurance: str = "report",
    authenticity: str = "unpinned",
    trust_store_path: str | None = None,
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
        "report_assurance": report_assurance,
        "warnings": warnings,
        "errors": errors,
        "evidence_level": evidence_level,
        "authenticity": authenticity,
    }
    if trust_store_path:
        payload["trust_store"] = trust_store_path
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
