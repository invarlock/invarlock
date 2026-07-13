from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from invarlock import evidence_pack_baselines as evidence_pack_baselines_mod
from invarlock import evidence_pack_binding as evidence_pack_binding_mod
from invarlock import evidence_pack_integrity as evidence_pack_integrity_mod
from invarlock import evidence_pack_policy as evidence_pack_policy_mod
from invarlock import (
    evidence_pack_report_verification as evidence_pack_report_verification_mod,
)
from invarlock import evidence_pack_support as evidence_pack_support_mod
from invarlock.evidence_pack_contracts.trust_store import load_trust_store_fingerprints
from invarlock.evidence_pack_edit_verifier import _verify_edit_metadata_consistency
from invarlock.evidence_pack_snapshot import PackSnapshot
from invarlock.reporting.verify_contract import (
    VerifyExecutionResult,
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
validate_manifest_payload = evidence_pack_integrity_mod.validate_manifest_payload
verify_manifest_provenance = evidence_pack_integrity_mod.verify_manifest_provenance
_evidence_pack_counts_from_verification = (
    evidence_pack_support_mod._evidence_pack_counts_from_verification
)
_derive_evidence_pack_evidence_level = (
    evidence_pack_support_mod._derive_evidence_pack_evidence_level
)
_CONTROL_FILES = evidence_pack_integrity_mod.CONTROL_FILES
MANIFEST_SIGNATURE_FILENAME = evidence_pack_integrity_mod.MANIFEST_SIGNATURE_FILENAME
EvidencePackStatus = evidence_pack_support_mod.EvidencePackStatus
EvidencePackResult = evidence_pack_support_mod.EvidencePackResult
RunVerifyCommand = Callable[..., VerifyExecutionResult]


_relative_file_paths = evidence_pack_support_mod._relative_file_paths
_verify_manifest_binds_checksums = (
    evidence_pack_support_mod._verify_manifest_binds_checksums
)
_verify_manifest_binds_checksums_payload = (
    evidence_pack_integrity_mod.verify_manifest_binds_checksums_payload
)
_verify_checksums = evidence_pack_support_mod._verify_checksums
_parse_checksums = evidence_pack_support_mod._parse_checksums
_verify_no_extra_files = evidence_pack_support_mod._verify_no_extra_files
_signature_warnings_to_errors = evidence_pack_integrity_mod.signature_warnings_to_errors

_normalize_binding_digest = evidence_pack_binding_mod._normalize_binding_digest
_normalize_verdict_report_path = (
    evidence_pack_binding_mod._normalize_verdict_report_path
)
_report_run_id = evidence_pack_binding_mod._report_run_id
_report_id = evidence_pack_binding_mod._report_id
_validate_binding_item = evidence_pack_binding_mod._validate_binding_item
_pack_declares_strict_report_binding = (
    evidence_pack_binding_mod._pack_declares_strict_report_binding
)
_binding_file_safety_errors = evidence_pack_binding_mod._binding_file_safety_errors
_discover_binding_files = evidence_pack_binding_mod._discover_binding_files
verify_final_verdict_report_binding = (
    evidence_pack_binding_mod.verify_final_verdict_report_binding
)


def _skip_verify_assurance_error(
    *,
    skip_verify: bool,
    strict: bool,
    profile: str | None,
    report_assurance: str,
) -> str | None:
    """Reject report-verification bypasses in every assurance-bearing mode."""

    if not skip_verify:
        return None
    blockers: list[str] = []
    if strict:
        blockers.append("--strict")
    if report_assurance == "strict":
        blockers.append("--report-assurance strict")
    normalized_profile = (profile or "dev").strip().lower()
    if normalized_profile in {"ci", "release"}:
        blockers.append(f"--profile {normalized_profile}")
    if not blockers:
        return None
    return (
        "--skip-verify is a non-assurance integrity diagnostic and cannot be used "
        f"with {', '.join(blockers)}. Run bundled report verification instead."
    )


def _authenticated_catalog_execution_profile(
    pack_dir: Path,
    manifest: Mapping[str, object],
) -> tuple[str | None, list[str]]:
    """Read the execution profile selected by an authenticated catalog entry."""

    binding = manifest.get("catalog")
    if binding is None:
        return None, []
    if not isinstance(binding, Mapping):
        return None, ["catalog binding must be an object"]
    path = binding.get("path")
    lane_id = binding.get("entry_id")
    if path != "metadata/catalog.json" or not isinstance(lane_id, str):
        return None, ["catalog execution profile binding is invalid"]
    from invarlock.evidence_catalog import (
        EvidenceCatalogError,
        load_evidence_catalog,
    )

    try:
        catalog = load_evidence_catalog(pack_dir / path)
    except EvidenceCatalogError as exc:
        return None, [f"embedded catalog is invalid: {exc}"]
    entry = catalog.entries.get(lane_id)
    execution = entry.get("execution") if isinstance(entry, Mapping) else None
    catalog_profile = (
        execution.get("profile") if isinstance(execution, Mapping) else None
    )
    if not isinstance(catalog_profile, str) or not catalog_profile:
        return None, ["catalog entry execution profile is invalid"]
    return catalog_profile, []


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
    baseline: Path | None = None,
    policy_pack: Path | None = None,
    profile: str,
    report_assurance: str = "report",
    expected_runtime_image_digest: str | None = None,
) -> VerifyExecutionResult:
    verify_kwargs: dict[str, Any] = {
        "profile": profile,
        "json_mode": True,
        "assurance_mode": report_assurance,
        "allow_unverified_provenance": unverified_provenance_allowed(),
    }
    if baseline is not None:
        verify_kwargs["baseline"] = baseline
    if policy_pack is not None:
        verify_kwargs["policy_pack"] = policy_pack
    if expected_runtime_image_digest is not None:
        verify_kwargs["expected_runtime_image_digest"] = expected_runtime_image_digest
    return run_verify_reports(reports, **verify_kwargs)


_scenario_strictness_by_id = (
    evidence_pack_report_verification_mod._scenario_strictness_by_id
)
_scenario_by_id = evidence_pack_report_verification_mod._scenario_by_id
_report_scenario_id = evidence_pack_report_verification_mod._report_scenario_id
_is_error_injection_report = (
    evidence_pack_report_verification_mod._is_error_injection_report
)
_report_expects_verify_failure = (
    evidence_pack_report_verification_mod._report_expects_verify_failure
)
_verify_command_succeeded = (
    evidence_pack_report_verification_mod._verify_command_succeeded
)
_detector_matches_report = (
    evidence_pack_report_verification_mod._detector_matches_report
)
_primary_guard_failure_signal = (
    evidence_pack_report_verification_mod._primary_guard_failure_signal
)
_report_has_intended_failure_signal = (
    evidence_pack_report_verification_mod._report_has_intended_failure_signal
)
_runtime_provenance_from_verify_payload = (
    evidence_pack_report_verification_mod._runtime_provenance_from_verify_payload
)
_expected_failure_result_errors = (
    evidence_pack_report_verification_mod._expected_failure_result_errors
)


def _verify_reports(
    pack_dir: Path,
    *,
    json_out_path: Path | None,
    profile: str,
    report_assurance: str,
    expected_runtime_image_digest: str | None = None,
    baseline_by_report: dict[Path, Path] | None = None,
    policy_pack: Path | None = None,
) -> tuple[list[str], dict[str, Any] | None]:
    """Verify packed reports through the currently configured command runner."""
    return evidence_pack_report_verification_mod.verify_reports(
        pack_dir,
        json_out_path=json_out_path,
        profile=profile,
        report_assurance=report_assurance,
        expected_runtime_image_digest=expected_runtime_image_digest,
        baseline_by_report=baseline_by_report,
        policy_pack=policy_pack,
        run_verify_command=_run_verify_command,
    )


inspect_evidence_pack = evidence_pack_support_mod.inspect_evidence_pack


def _remap_snapshot_paths(value: Any, mappings: Mapping[str, str]) -> Any:
    """Map private snapshot paths back to caller-visible verification paths."""

    if isinstance(value, dict):
        return {
            key: _remap_snapshot_paths(item, mappings) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_remap_snapshot_paths(item, mappings) for item in value]
    if isinstance(value, tuple):
        return tuple(_remap_snapshot_paths(item, mappings) for item in value)
    if not isinstance(value, str):
        return value
    for source, destination in sorted(
        mappings.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if value == source:
            return destination
        prefix = source + "/"
        if value.startswith(prefix):
            return destination + value[len(source) :]
    return value


def verify_evidence_pack(
    pack_dir: Path,
    *,
    json_out_path: Path | None = None,
    skip_verify: bool = False,
    strict: bool = False,
    profile: str | None = None,
    report_assurance: str = "report",
    expected_fingerprint: str | None = None,
    trust_store_path: Path | None = None,
    expected_catalog_digest: str | None = None,
    expected_runtime_image_digest: str | None = None,
    policy_pack_path: Path | None = None,
) -> EvidencePackResult:
    skip_verify_error = _skip_verify_assurance_error(
        skip_verify=skip_verify,
        strict=strict,
        profile=profile,
        report_assurance=report_assurance,
    )
    if skip_verify_error is not None:
        return _build_verify_result(
            pack_dir=pack_dir,
            ok=False,
            strict=strict,
            skip_verify=skip_verify,
            report_assurance=report_assurance,
            warnings=[],
            errors=[skip_verify_error],
            signer_fingerprint=None,
            verify_payload=None,
            status=EvidencePackStatus.USAGE,
            authenticity="unpinned",
            trust_store_path=str(trust_store_path) if trust_store_path else None,
        )
    invalid_fingerprint = (
        expected_fingerprint is not None
        and evidence_pack_integrity_mod.normalize_expected_fingerprint(
            expected_fingerprint
        )
        is None
    )
    if (
        invalid_fingerprint
        or report_assurance not in {"report", "strict", "off"}
        or not pack_dir.is_dir()
        or (json_out_path is not None and _path_within_dir(pack_dir, json_out_path))
    ):
        return _verify_evidence_pack_snapshot_root(
            pack_dir,
            json_out_path=json_out_path,
            skip_verify=skip_verify,
            strict=strict,
            profile=profile,
            report_assurance=report_assurance,
            expected_fingerprint=expected_fingerprint,
            trust_store_path=trust_store_path,
            expected_catalog_digest=expected_catalog_digest,
            expected_runtime_image_digest=expected_runtime_image_digest,
            policy_pack_path=policy_pack_path,
        )
    snapshot, snapshot_errors = PackSnapshot.capture(pack_dir)
    if snapshot is None:
        return _build_verify_result(
            pack_dir=pack_dir,
            ok=False,
            strict=strict,
            skip_verify=skip_verify,
            report_assurance=report_assurance,
            warnings=[],
            errors=snapshot_errors,
            signer_fingerprint=None,
            verify_payload=None,
            status=EvidencePackStatus.INTEGRITY,
            authenticity="unpinned",
            trust_store_path=str(trust_store_path) if trust_store_path else None,
        )
    checksums_entry = snapshot.files.entry("checksums.sha256")
    with snapshot.files.materialized() as snapshot_root:
        result = _verify_evidence_pack_snapshot_root(
            snapshot_root,
            json_out_path=json_out_path,
            skip_verify=skip_verify,
            strict=strict,
            profile=profile,
            report_assurance=report_assurance,
            expected_fingerprint=expected_fingerprint,
            trust_store_path=trust_store_path,
            expected_catalog_digest=expected_catalog_digest,
            expected_runtime_image_digest=expected_runtime_image_digest,
            policy_pack_path=policy_pack_path,
            _manifest_payload=snapshot.files.parsed_json.get("manifest.json"),
            _checksums_payload=(
                checksums_entry.read_bytes() if checksums_entry else None
            ),
        )
        result = EvidencePackResult(
            payload=_remap_snapshot_paths(
                result.payload,
                {str(snapshot_root): str(pack_dir)},
            ),
            status=result.status,
        )
        materialized_errors = snapshot.files.materialized_stability_errors(
            snapshot_root
        )
    stability_errors = [*materialized_errors, *snapshot.stability_errors()]
    payload = dict(result.payload)
    payload["pack"] = str(pack_dir)
    if not stability_errors:
        return EvidencePackResult(payload=payload, status=result.status)
    payload["ok"] = False
    payload["integrity_ok"] = False
    payload["reports_verified"] = False
    payload["verification_scope"] = "not_verified"
    payload["assurance_status"] = "not_verified"
    payload["errors"] = [*payload.get("errors", []), *stability_errors]
    status = (
        EvidencePackStatus.INTEGRITY
        if result.status in {EvidencePackStatus.OK, EvidencePackStatus.INTEGRITY_ONLY}
        else result.status
    )
    return EvidencePackResult(payload=payload, status=status)


def _verify_evidence_pack_snapshot_root(
    pack_dir: Path,
    *,
    json_out_path: Path | None = None,
    skip_verify: bool = False,
    strict: bool = False,
    profile: str | None = None,
    report_assurance: str = "report",
    expected_fingerprint: str | None = None,
    trust_store_path: Path | None = None,
    expected_catalog_digest: str | None = None,
    expected_runtime_image_digest: str | None = None,
    policy_pack_path: Path | None = None,
    _manifest_payload: Any = None,
    _checksums_payload: bytes | None = None,
) -> EvidencePackResult:
    warnings: list[str] = []
    errors: list[str] = []
    verify_payload: dict[str, Any] | None = None
    baseline_by_report: dict[Path, Path] = {}
    policy_pack_for_reports: Path | None = None
    signer_fingerprint: str | None = None
    evidence_level: str | None = None
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

    authenticated_manifest = _manifest_payload
    if authenticated_manifest is None:
        try:
            authenticated_manifest = _load_json(pack_dir / "manifest.json")
        except _json_load_error_types() as exc:
            errors.append(f"manifest is not valid JSON: {exc}")
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
    checksums_payload = (
        _checksums_payload
        if _checksums_payload is not None
        else (pack_dir / "checksums.sha256").read_bytes()
    )
    errors.extend(
        _verify_manifest_binds_checksums_payload(
            authenticated_manifest,
            checksums_payload,
        )
    )
    checksum_errors, covered_paths = _verify_checksums(pack_dir)
    errors.extend(checksum_errors)
    errors.extend(evidence_pack_integrity_mod.verify_control_file_mirrors(pack_dir))
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
            report_assurance=report_assurance,
            authenticity=authenticity,
            trust_store_path=trust_store_used,
        )

    errors.extend(validate_manifest_payload(authenticated_manifest))
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

    manifest_for_level = authenticated_manifest
    if isinstance(manifest_for_level, dict) and isinstance(
        manifest_for_level.get("evidence_level"), str
    ):
        evidence_level = manifest_for_level["evidence_level"]

    errors.extend(
        verify_final_verdict_report_binding(
            pack_dir,
            require_binding=strict or report_assurance == "strict",
        )
    )
    effective_profile = profile or "dev"
    if isinstance(authenticated_manifest, dict):
        from invarlock.evidence_catalog import validate_embedded_catalog_binding

        verification_contract = authenticated_manifest.get("verification")
        subject_mode = (
            verification_contract.get("subject_mode")
            if isinstance(verification_contract, dict)
            else None
        )
        if subject_mode == "catalog_bound_noop" and not isinstance(
            authenticated_manifest.get("catalog"), dict
        ):
            errors.append("catalog-bound evidence is missing its catalog contract")
        if subject_mode == "sealed":
            errors.append("legacy sealed subject mode is not independently verifiable")
        catalog_errors = validate_embedded_catalog_binding(
            pack_dir,
            authenticated_manifest,
            expected_catalog_digest=expected_catalog_digest,
        )
        errors.extend(catalog_errors)
        if not catalog_errors:
            catalog_profile, catalog_profile_errors = (
                _authenticated_catalog_execution_profile(
                    pack_dir,
                    authenticated_manifest,
                )
            )
            errors.extend(catalog_profile_errors)
            if catalog_profile is not None:
                if profile is not None and profile != catalog_profile:
                    return _build_verify_result(
                        pack_dir=pack_dir,
                        ok=False,
                        strict=strict,
                        skip_verify=skip_verify,
                        warnings=warnings,
                        errors=[
                            *errors,
                            "requested report profile does not match the "
                            f"authenticated catalog profile ({profile!r} != "
                            f"{catalog_profile!r})",
                        ],
                        signer_fingerprint=signer_fingerprint,
                        verify_payload=verify_payload,
                        status=EvidencePackStatus.USAGE,
                        report_assurance=report_assurance,
                        authenticity=authenticity,
                        trust_store_path=trust_store_used,
                        evidence_level=evidence_level,
                    )
                effective_profile = catalog_profile
    baseline_verification = evidence_pack_baselines_mod.verify_baseline_materials(
        pack_dir,
        report_assurance=report_assurance,
    )
    errors.extend(baseline_verification.errors)
    baseline_by_report = baseline_verification.baseline_by_report
    policy_verification = evidence_pack_policy_mod.verify_policy_material(
        pack_dir,
        report_assurance=report_assurance,
        acceptance_policy_path=policy_pack_path,
    )
    errors.extend(policy_verification.errors)
    policy_pack_for_reports = policy_verification.policy_pack_path
    errors.extend(verify_manifest_provenance(pack_dir))
    errors.extend(_verify_edit_metadata_consistency(pack_dir))
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
            report_assurance=report_assurance,
            authenticity=authenticity,
            trust_store_path=trust_store_used,
            evidence_level=evidence_level,
        )

    if not skip_verify:
        verify_kwargs: dict[str, Any] = {
            "json_out_path": json_out_path,
            "profile": effective_profile,
            "report_assurance": report_assurance,
        }
        if baseline_by_report:
            verify_kwargs["baseline_by_report"] = baseline_by_report
        if policy_pack_for_reports is not None:
            verify_kwargs["policy_pack"] = policy_pack_for_reports
        if expected_runtime_image_digest is not None:
            verify_kwargs["expected_runtime_image_digest"] = (
                expected_runtime_image_digest
            )
        report_errors, verify_payload = _verify_reports(pack_dir, **verify_kwargs)
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
                evidence_level=evidence_level,
            )

    return _build_verify_result(
        pack_dir=pack_dir,
        ok=not skip_verify,
        strict=strict,
        skip_verify=skip_verify,
        report_assurance=report_assurance,
        warnings=warnings,
        errors=errors,
        signer_fingerprint=signer_fingerprint,
        verify_payload=verify_payload,
        status=(
            EvidencePackStatus.INTEGRITY_ONLY if skip_verify else EvidencePackStatus.OK
        ),
        authenticity=authenticity,
        trust_store_path=trust_store_used,
        evidence_level=evidence_level,
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
    evidence_level: str | None = None,
) -> EvidencePackResult:
    reports_verified = bool(not skip_verify and ok and status is EvidencePackStatus.OK)
    integrity_ok = bool(
        status is EvidencePackStatus.INTEGRITY_ONLY
        or (ok and status is EvidencePackStatus.OK)
    )
    verification_scope = (
        "integrity_only"
        if status is EvidencePackStatus.INTEGRITY_ONLY
        else ("report_verification" if reports_verified else "not_verified")
    )
    payload: dict[str, Any] = {
        "pack": str(pack_dir),
        "ok": ok,
        "integrity_ok": integrity_ok,
        "reports_verified": reports_verified,
        "verification_scope": verification_scope,
        "assurance_status": ("report_verified" if reports_verified else "not_verified"),
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
