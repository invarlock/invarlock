from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from invarlock import __version__ as _INVARLOCK_VERSION
from invarlock.core.assurance_contract import (
    resolve_report_assurance_mode,
    resolve_report_runtime_provenance_declared,
    strict_report_policy_errors,
)
from invarlock.core.error_utils import encode_error as _encode_error
from invarlock.core.exceptions import InvarlockError
from invarlock.core.exceptions import MetricsError as _MetricsError
from invarlock.core.exceptions import ValidationError as _ValidationError
from invarlock.core.run_policy import enforce_provider_parity
from invarlock.runtime_provenance import (
    RuntimeProvenanceVerdict,
    verify_runtime_provenance,
)

from . import verify_adapter_family as _verify_adapter_family
from . import verify_baseline as _verify_baseline
from . import verify_bootstrap as _verify_bootstrap
from . import verify_check_helpers_consistency as _verify_consistency
from . import verify_check_helpers_metrics as _verify_metrics
from . import verify_output as _verify_output
from . import verify_policy as _verify_policy
from . import verify_strict_accuracy as _verify_strict_accuracy
from . import verify_strict_ppl as _verify_strict_ppl
from . import verify_strict_schedule as _verify_strict_schedule
from . import verify_strict_vision as _verify_strict_vision
from .runtime_policy_receipt import runtime_policy_from_report
from .verify_contract_types import (
    VerifyDiagnostic,
    VerifyExecutionResult,
    VerifyOutcome,
    VerifyReportResult,
    VerifyRequest,
)

_VERIFY_RECOVERABLE_EXCEPTIONS = (
    AttributeError,
    FileNotFoundError,
    json.JSONDecodeError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_coerce_float = _verify_metrics._coerce_float
_coerce_int = _verify_metrics._coerce_int
_load_evaluation_report = _verify_metrics._load_evaluation_report
_resolve_path = _verify_consistency._resolve_path
_measurement_contract_digest = _verify_consistency._measurement_contract_digest
_validate_logspace_ci_identity = _verify_metrics._validate_logspace_ci_identity
_validate_primary_metric = _verify_metrics._validate_primary_metric
_validate_pairing = _verify_consistency._validate_pairing
_validate_counts = _verify_consistency._validate_counts
_validate_drift_band = _verify_consistency._validate_drift_band
_validate_tokenizer_hash = _verify_consistency._validate_tokenizer_hash
_validate_measurement_contracts = _verify_consistency._validate_measurement_contracts
_validate_variance_enablement = _verify_consistency._validate_variance_enablement
_apply_profile_lints = _verify_consistency._apply_profile_lints
_report_schema = _verify_metrics._report_schema
validate_report = _verify_metrics.validate_report
compute_validation_flags = _verify_metrics.compute_validation_flags
resolve_tiny_relax_from_report = _verify_metrics.resolve_tiny_relax_from_report


@dataclass(frozen=True)
class _JsonObjectSnapshot:
    payload: dict[str, Any]
    raw_bytes: bytes


@dataclass(frozen=True)
class _RecomputeValidationResult:
    diagnostics: tuple[VerifyDiagnostic, ...] = ()
    metric_mismatch: bool = False


_STRICT_METRIC_EVIDENCE_MISMATCH_PREFIXES = (
    "Strict paired PPL bootstrap CI mismatch:",
    "Supplied baseline metric/raw-window mismatch:",
    "Supplied baseline accuracy metric/count mismatch:",
)


def _strict_metric_evidence_mismatch(errors: list[str]) -> str | None:
    """Return a recomputed-evidence mismatch without classifying shape failures."""

    return next(
        (
            error
            for error in errors
            if error.startswith(_STRICT_METRIC_EVIDENCE_MISMATCH_PREFIXES)
        ),
        None,
    )


def _load_json_object_snapshot(
    path: Path,
    *,
    object_name: str,
) -> _JsonObjectSnapshot:
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes)
    if not isinstance(payload, dict):
        raise ValueError(f"{object_name} must decode to a JSON object")
    return _JsonObjectSnapshot(
        payload=payload,
        raw_bytes=raw_bytes,
    )


def _load_evaluation_report_snapshot(path: Path) -> _JsonObjectSnapshot:
    return _load_json_object_snapshot(path, object_name="evaluation report")


def _validate_report_schema_strict(report: dict[str, Any]) -> bool:
    return _verify_metrics._validate_report_schema_strict(
        report,
        report_schema_module=_report_schema,
    )


def _recompute_validation_flags(report: dict[str, Any]) -> dict[str, bool]:
    flags: dict[str, bool] = _verify_metrics._recompute_validation_flags(
        report,
        compute_validation_flags_fn=compute_validation_flags,
        resolve_tiny_relax_from_report_fn=resolve_tiny_relax_from_report,
    )
    return flags


def _validate_primary_metric_policy(
    report: dict[str, Any], *, profile: str | None = None
) -> list[str]:
    return _verify_metrics._validate_primary_metric_policy(
        report,
        profile=profile,
        recompute_validation_flags_fn=_recompute_validation_flags,
    )


def _validate_evaluation_report_payload(
    path: Path,
    *,
    profile: str | None = None,
    report_payload: dict[str, Any] | None = None,
) -> list[str]:
    return _verify_consistency._validate_evaluation_report_payload(
        path,
        profile=profile,
        report_payload=report_payload,
        load_evaluation_report_fn=_load_evaluation_report,
        validate_report_fn=validate_report,
        validate_report_schema_strict_fn=_validate_report_schema_strict,
        validate_primary_metric_fn=_validate_primary_metric,
        validate_pairing_fn=_validate_pairing,
        validate_counts_fn=_validate_counts,
        validate_logspace_ci_identity_fn=_validate_logspace_ci_identity,
        validate_drift_band_fn=_validate_drift_band,
        validate_primary_metric_policy_fn=_validate_primary_metric_policy,
        apply_profile_lints_fn=_apply_profile_lints,
        validate_tokenizer_hash_fn=_validate_tokenizer_hash,
        validate_variance_enablement_fn=_validate_variance_enablement,
        validate_measurement_contracts_fn=_validate_measurement_contracts,
    )


def _load_baseline_digest(baseline: Path | None) -> dict[str, Any] | None:
    return _baseline_digest_from_payload(_load_baseline_payload(baseline))


def _load_baseline_payload(baseline: Path | None) -> dict[str, Any] | None:
    snapshot = _load_baseline_snapshot(baseline)
    return snapshot.payload if snapshot is not None else None


def _load_baseline_snapshot(baseline: Path | None) -> _JsonObjectSnapshot | None:
    try:
        if baseline is None:
            return None
        return _load_json_object_snapshot(baseline, object_name="baseline report")
    except _VERIFY_RECOVERABLE_EXCEPTIONS:
        return None


def _baseline_digest_from_payload(
    baseline_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    prov = (
        baseline_payload.get("provenance")
        if isinstance(baseline_payload, dict)
        else None
    )
    if not isinstance(prov, dict):
        return None
    provider_digest = prov.get("provider_digest")
    return provider_digest if isinstance(provider_digest, dict) else None


def _resolve_profile_name(profile: str | None) -> str:
    try:
        return (profile or "").strip().lower()
    except _VERIFY_RECOVERABLE_EXCEPTIONS:
        return "dev"


def _guard_warning_count(report: dict[str, Any]) -> int:
    guard_warnings = report.get("guard_warnings")
    if not isinstance(guard_warnings, dict):
        return 0
    warning_count = guard_warnings.get("warning_count")
    warnings = guard_warnings.get("warnings")
    if (
        isinstance(warning_count, bool)
        or not isinstance(warning_count, int)
        or warning_count < 0
    ):
        raise ValueError("guard_warnings.warning_count must be a non-negative integer")
    if not isinstance(warnings, list):
        raise ValueError("guard_warnings.warnings must be an array")
    if warning_count != len(warnings):
        raise ValueError(
            "guard_warnings.warning_count must equal the warnings array length"
        )
    return warning_count


def _guard_warning_diagnostics(report: dict[str, Any]) -> tuple[VerifyDiagnostic, ...]:
    guard_warnings = report.get("guard_warnings")
    if not isinstance(guard_warnings, dict):
        return ()
    warning_count = _guard_warning_count(report)
    if warning_count <= 0:
        return ()
    diagnostics: list[VerifyDiagnostic] = [
        VerifyDiagnostic(
            level="warning",
            message=f"Guard warnings present: {warning_count}",
        )
    ]
    warnings = guard_warnings.get("warnings")
    if isinstance(warnings, list):
        for entry_raw in warnings[:5]:
            if not isinstance(entry_raw, dict):
                continue
            guard = str(entry_raw.get("guard") or "guard")
            kind = str(entry_raw.get("kind") or "warning")
            module = entry_raw.get("module")
            location = f" ({module})" if isinstance(module, str) and module else ""
            policy_gate = str(entry_raw.get("policy_gate") or "unknown")
            diagnostics.append(
                VerifyDiagnostic(
                    level="warning",
                    message=f"{guard}.{kind}{location}; policy={policy_gate}",
                )
            )
    return tuple(diagnostics)


def _append_recompute_errors(
    errors: list[str],
    *,
    cert_obj: dict[str, Any],
    prof: str,
    tol: float,
    json_mode: bool,
    require_strict_assurance: bool = False,
) -> _RecomputeValidationResult:
    del prof, json_mode
    pm = cert_obj.get("primary_metric", {}) if isinstance(cert_obj, dict) else {}
    kind = str(pm.get("kind") or "").strip().lower() if isinstance(pm, dict) else ""
    win = cert_obj.get("evaluation_windows", {}) if isinstance(cert_obj, dict) else {}
    fin = win.get("final") if isinstance(win, dict) else None
    if kind == "accuracy":
        recompute_start = len(errors)
        accuracy_recomputed = _verify_strict_accuracy._append_accuracy_recompute_errors(
            errors,
            cert_obj=cert_obj,
            pm=pm,
            tol=tol,
            require_strict=require_strict_assurance,
        )
        if require_strict_assurance:
            _verify_strict_vision.append_strict_vision_evidence_errors(errors, cert_obj)
        if accuracy_recomputed:
            mismatch = any(
                "mismatch" in str(error).lower() for error in errors[recompute_start:]
            )
            return _RecomputeValidationResult(metric_mismatch=mismatch)
        errors.append(
            "Accuracy verification requires measured classification aggregates for recomputation."
        )
        return _RecomputeValidationResult()

    if require_strict_assurance and isinstance(pm, dict):
        recompute_start = len(errors)
        preview = win.get("preview") if isinstance(win, dict) else None
        preview_mean = _verify_strict_ppl._append_ppl_arm_recompute_errors(
            errors,
            arm="preview",
            section=preview,
            primary_metric=pm,
            tolerance=tol,
            require_analysis_point=True,
            require_window_ids=True,
        )
        final_mean = _verify_strict_ppl._append_ppl_arm_recompute_errors(
            errors,
            arm="final",
            section=fin,
            primary_metric=pm,
            tolerance=tol,
            require_analysis_point=True,
            require_window_ids=True,
        )
        _verify_strict_ppl._append_strict_ppl_schedule_errors(errors, cert_obj=cert_obj)
        _verify_strict_ppl._append_strict_ppl_coherence_errors(
            errors,
            cert_obj=cert_obj,
            primary_metric=pm,
            preview_mean=preview_mean,
            final_mean=final_mean,
            tolerance=tol,
        )
        mismatch = any(
            "mismatch" in str(error).lower() for error in errors[recompute_start:]
        )
        return _RecomputeValidationResult(metric_mismatch=mismatch)

    if not (isinstance(pm, dict) and isinstance(fin, dict)):
        errors.append(
            "PPL verification requires evaluation_windows.final evidence for recomputation."
        )
        return _RecomputeValidationResult()
    ll = fin.get("logloss")
    wc = fin.get("token_counts")
    if not (
        isinstance(ll, list)
        and isinstance(wc, list)
        and ll
        and wc
        and len(ll) == len(wc)
    ):
        errors.append(
            "PPL verification requires complete final logloss and token-count evidence."
        )
        return _RecomputeValidationResult()
    window_ids = fin.get("window_ids")
    if isinstance(window_ids, list):
        if len(window_ids) != len(ll):
            errors.append(
                "evaluation_windows.final.window_ids length differs from logloss/token_counts."
            )
        if len(window_ids) != len({str(item) for item in window_ids}):
            errors.append("evaluation_windows.final.window_ids contains duplicates.")
    final_mean = _verify_strict_ppl._append_ppl_arm_recompute_errors(
        errors,
        arm="final",
        section=fin,
        primary_metric=pm,
        tolerance=tol,
        require_analysis_point=False,
    )
    reported_final = _coerce_float(pm.get("final"))
    metric_mismatch = False
    if final_mean is not None and reported_final is not None:
        try:
            recomputed_final = math.exp(final_mean)
        except OverflowError:
            recomputed_final = float("inf")
        metric_mismatch = not math.isclose(
            reported_final,
            recomputed_final,
            rel_tol=tol,
            abs_tol=tol,
        )
    return _RecomputeValidationResult(metric_mismatch=metric_mismatch)


def _runtime_provenance_verification_payload(
    provenance_result: Any,
    *,
    declared_mode: str = "unknown",
) -> dict[str, Any]:
    verdict = RuntimeProvenanceVerdict.from_result(
        provenance_result,
        declared_mode=declared_mode,
    )
    return verdict.as_verification_payload()


def _reported_policy_digest(report: dict[str, Any]) -> str | None:
    policy_digest = report.get("policy_digest")
    if isinstance(policy_digest, dict):
        for key in ("thresholds_hash", "policy_digest", "digest"):
            value = policy_digest.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    provenance = report.get("provenance")
    policy = provenance.get("policy") if isinstance(provenance, dict) else None
    if isinstance(policy, dict):
        value = policy.get("policy_digest")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _verification_receipt_payload(
    *,
    cert_snapshot: _JsonObjectSnapshot,
    cert_obj: dict[str, Any],
    baseline_snapshot: _JsonObjectSnapshot | None,
    baseline_payload: dict[str, Any] | None,
    policy_snapshot: _JsonObjectSnapshot | None,
    policy_payload: dict[str, Any] | None,
    profile: str,
    assurance_mode: str,
    report_assurance_mode: str,
    warning_policy: str,
    expected_runtime_image_digest: str | None,
) -> dict[str, Any]:
    """Describe the exact unsigned inputs consumed by this verifier run."""

    subject_digest = _baseline_digest_from_payload(cert_obj)
    baseline_digest = _baseline_digest_from_payload(baseline_payload)
    dataset = cert_obj.get("dataset")
    provider_coordinates: dict[str, object] = {}
    if isinstance(dataset, dict):
        for receipt_key, report_key in (
            ("kind", "provider"),
            ("dataset_name", "dataset_name"),
            ("config_name", "config_name"),
            ("revision", "revision"),
        ):
            value = dataset.get(report_key)
            if value is not None:
                provider_coordinates[receipt_key] = value
    provider_bytes = json.dumps(
        provider_coordinates,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "format_version": "invarlock.verify-receipt.v1",
        "signed": False,
        "subject_report_sha256": hashlib.sha256(cert_snapshot.raw_bytes).hexdigest(),
        "baseline_report_sha256": (
            hashlib.sha256(baseline_snapshot.raw_bytes).hexdigest()
            if baseline_snapshot is not None
            else None
        ),
        "policy_pack_sha256": (
            hashlib.sha256(policy_snapshot.raw_bytes).hexdigest()
            if policy_snapshot is not None
            else None
        ),
        "subject_provider_digest": (
            deepcopy(subject_digest) if subject_digest is not None else None
        ),
        "baseline_provider_digest": (
            deepcopy(baseline_digest) if baseline_digest is not None else None
        ),
        "dataset_provider": provider_coordinates,
        "dataset_provider_sha256": "sha256:"
        + hashlib.sha256(provider_bytes).hexdigest(),
        "verifier": {
            "package": "invarlock",
            "version": _INVARLOCK_VERSION,
        },
        "inputs": {
            "profile": profile,
            "assurance_mode": assurance_mode,
            "report_assurance_mode": report_assurance_mode,
            "warning_policy": warning_policy,
            "expected_runtime_image_digest": expected_runtime_image_digest,
            "expected_policy_digest": (
                policy_payload.get("policy_digest")
                if isinstance(policy_payload, dict)
                else None
            ),
        },
        "reported_policy_digest": _reported_policy_digest(cert_obj),
    }


def _verify_single_report(
    cert_path: Path,
    *,
    cert_snapshot: _JsonObjectSnapshot | None = None,
    baseline: Path | None,
    baseline_snapshot: _JsonObjectSnapshot | None,
    baseline_payload: dict[str, Any] | None,
    baseline_digest: dict[str, Any] | None,
    policy_snapshot: _JsonObjectSnapshot | None,
    policy_payload: dict[str, Any] | None,
    tolerance: float,
    profile: str | None,
    allow_unverified_provenance: bool,
    assurance_mode: str,
    warning_policy: str,
    json_mode: bool,
    expected_runtime_image_digest: str | None,
) -> VerifyReportResult:
    if cert_snapshot is None:
        cert_snapshot = _load_evaluation_report_snapshot(cert_path)
    cert_obj = cert_snapshot.payload
    caller_prof = _resolve_profile_name(profile)
    report_assurance_mode = resolve_report_assurance_mode(cert_obj)
    require_strict_assurance = assurance_mode == "strict" or (
        assurance_mode == "report" and report_assurance_mode == "strict"
    )
    assurance = cert_obj.get("assurance")
    declared_profile = assurance.get("profile") if isinstance(assurance, dict) else None
    prof = (
        declared_profile.strip().lower()
        if require_strict_assurance
        and isinstance(declared_profile, str)
        and declared_profile.strip().lower() in {"ci", "release"}
        else caller_prof
    )
    prov = cert_obj.get("provenance") if isinstance(cert_obj, dict) else None
    subj_digest = prov.get("provider_digest") if isinstance(prov, dict) else None
    if prof in {"ci", "release"}:
        if not (isinstance(subj_digest, dict) and subj_digest.get("ids_sha256")):
            raise InvarlockError(
                code="E004",
                message=(
                    "PROVIDER-DIGEST-MISSING: subject missing provider_digest.ids_sha256"
                ),
            )
        if baseline_digest is not None:
            enforce_provider_parity(
                subj_digest,
                baseline_digest,
                profile=prof,
                invarlock_error_cls=InvarlockError,
            )

    errors = _validate_evaluation_report_payload(
        cert_path,
        profile=prof,
        report_payload=cert_obj,
    )
    provenance_kwargs: dict[str, Any] = {
        "allow_unverified": bool(allow_unverified_provenance),
        "require_strict_runtime": require_strict_assurance,
    }
    if expected_runtime_image_digest is not None:
        provenance_kwargs["expected_image_digest"] = expected_runtime_image_digest
    provenance_result = verify_runtime_provenance(
        cert_path,
        report_bytes=cert_snapshot.raw_bytes,
        **provenance_kwargs,
    )
    verification_payload = _runtime_provenance_verification_payload(
        provenance_result,
        declared_mode=resolve_report_runtime_provenance_declared(cert_obj),
    )
    errors.extend(issue.message for issue in provenance_result.issues)
    verification_payload["receipt"] = _verification_receipt_payload(
        cert_snapshot=cert_snapshot,
        cert_obj=cert_obj,
        baseline_snapshot=baseline_snapshot,
        baseline_payload=baseline_payload,
        policy_snapshot=policy_snapshot,
        policy_payload=policy_payload,
        profile=caller_prof,
        assurance_mode=assurance_mode,
        report_assurance_mode=report_assurance_mode,
        warning_policy=warning_policy,
        expected_runtime_image_digest=expected_runtime_image_digest,
    )
    if assurance_mode == "strict" and report_assurance_mode != "strict":
        errors.append(
            "verify --assurance strict requires report assurance.mode=strict."
        )
    errors.extend(
        strict_report_policy_errors(
            cert_obj,
            require_strict=require_strict_assurance,
            runtime_provenance_verified=provenance_result.verified,
            verifier_profile=caller_prof,
        )
    )
    if require_strict_assurance:
        _runtime_policy, runtime_policy_errors = runtime_policy_from_report(cert_obj)
        errors.extend(
            f"Strict runtime policy receipt: {error}" for error in runtime_policy_errors
        )
        if _runtime_policy is None:
            errors.append("Strict assurance requires a runtime policy receipt.")
        _verify_policy.append_strict_policy_authorization_errors(
            errors,
            report=cert_obj,
            policy_pack=policy_payload,
        )
        if baseline is not None and baseline.resolve(strict=False) == cert_path.resolve(
            strict=False
        ):
            errors.append(
                "Strict baseline binding requires a baseline file distinct from the "
                "subject report."
            )
        if (
            baseline_snapshot is not None
            and hashlib.sha256(baseline_snapshot.raw_bytes).digest()
            == hashlib.sha256(cert_snapshot.raw_bytes).digest()
        ):
            errors.append(
                "Strict assurance rejects a byte-identical subject copied to a "
                "different --baseline path."
            )
        _verify_baseline.append_strict_baseline_contract_errors(
            errors,
            report=cert_obj,
            baseline_payload=baseline_payload,
            baseline_supplied=baseline is not None,
            tolerance=tolerance,
        )
        _verify_strict_schedule._append_strict_supplied_baseline_binding_errors(
            errors,
            cert_obj=cert_obj,
            baseline_payload=baseline_payload,
            baseline_supplied=baseline is not None,
            tolerance=tolerance,
        )
        _verify_bootstrap.append_strict_ppl_bootstrap_replay_errors(
            errors,
            report=cert_obj,
            baseline_payload=baseline_payload,
            baseline_supplied=baseline is not None,
            tolerance=tolerance,
        )
    if json_mode and any("schema validation failed" in str(e).lower() for e in errors):
        raise _ValidationError(
            code="E601",
            message="REPORT-SCHEMA-INVALID: schema validation failed",
            details={"path": str(cert_path)},
        )
    try:
        guard_warning_count = _guard_warning_count(cert_obj)
    except ValueError as exc:
        errors.append(str(exc))
        guard_warning_count = 0
    malformed = any(
        ("schema validation failed" in e.lower())
        or ("missing primary_metric.ratio_vs_baseline" in e)
        or ("report is missing a finite primary_metric.ratio_vs_baseline" in e)
        or e.startswith("guard_warnings.")
        for e in errors
    )

    recompute_result = _append_recompute_errors(
        errors,
        cert_obj=cert_obj,
        prof=prof,
        tol=tolerance,
        json_mode=json_mode,
        require_strict_assurance=require_strict_assurance,
    )
    if warning_policy == "fail" and guard_warning_count > 0:
        errors.append(
            "Guard warning policy failed: "
            f"{guard_warning_count} guard warning(s) present."
        )
    strict_evidence_mismatch = (
        _strict_metric_evidence_mismatch(errors)
        if require_strict_assurance and prof in {"ci", "release"}
        else None
    )
    if (
        recompute_result.metric_mismatch or strict_evidence_mismatch is not None
    ) and prof in {"ci", "release"}:
        first = strict_evidence_mismatch or next(
            (
                error
                for error in errors
                if error.startswith(
                    (
                        "Display mismatch:",
                        "Accuracy mismatch:",
                        "primary_metric.display_ci mismatch:",
                        "Primary metric ratio mismatch against recomputed final and baseline:",
                    )
                )
            ),
            "primary metric differs from recomputed raw evidence",
        )
        raise _MetricsError(
            code="E602",
            message=f"RECOMPUTE-MISMATCH: {first}",
            details={"example": str(first)},
        )

    diagnostics: tuple[VerifyDiagnostic, ...] = ()
    if json_mode:
        return VerifyReportResult(
            report=cert_obj,
            errors=tuple(errors),
            malformed=malformed,
            diagnostics=diagnostics + recompute_result.diagnostics,
            verification=verification_payload,
        )
    if errors:
        diagnostics = (VerifyDiagnostic(level="fail", message=str(cert_path)),) + tuple(
            VerifyDiagnostic(level="detail", message=str(err)) for err in errors
        )
        return VerifyReportResult(
            report=cert_obj,
            errors=tuple(errors),
            malformed=malformed,
            diagnostics=diagnostics + recompute_result.diagnostics,
            verification=verification_payload,
        )
    try:
        diagnostics = (
            VerifyDiagnostic(level="pass", message=str(cert_path)),
            *_guard_warning_diagnostics(cert_obj),
            *_verify_adapter_family.warn_adapter_family_mismatch(
                cert_obj,
                trusted_baseline_path=baseline,
                trusted_baseline_payload=baseline_payload,
            ),
        )
    except _VERIFY_RECOVERABLE_EXCEPTIONS:
        diagnostics = (
            VerifyDiagnostic(level="pass", message=str(cert_path)),
            *_guard_warning_diagnostics(cert_obj),
        )
    return VerifyReportResult(
        report=cert_obj,
        errors=tuple(errors),
        malformed=malformed,
        diagnostics=diagnostics + recompute_result.diagnostics,
        verification=verification_payload,
    )


def _run_verify_request(request: VerifyRequest) -> VerifyExecutionResult:
    overall_ok = True
    diagnostics: list[VerifyDiagnostic] = []
    try:
        tol = request.normalized_tolerance
    except ValueError as exc:
        payload = _verify_output.build_verify_error_payload(
            request.reports[0] if request.reports else None,
            reason="policy_fail",
            encoded_error=_encode_error(exc),
        )
        if not request.json_mode:
            diagnostics.append(
                VerifyDiagnostic(level="error", message=f"Verification failed: {exc}")
            )
        return VerifyExecutionResult(
            outcome=VerifyOutcome.POLICY_FAIL,
            payload=payload,
            diagnostics=tuple(diagnostics),
            error=exc,
            include_resolution=True,
        )
    normalized_assurance_mode = request.normalized_assurance_mode
    normalized_warning_policy = request.normalized_warning_policy
    baseline_snapshot = _load_baseline_snapshot(request.baseline)
    baseline_payload = (
        baseline_snapshot.payload if baseline_snapshot is not None else None
    )
    baseline_digest = _baseline_digest_from_payload(baseline_payload)
    policy_snapshot = (
        _load_json_object_snapshot(request.policy_pack, object_name="policy pack")
        if request.policy_pack is not None
        else None
    )
    policy_payload = policy_snapshot.payload if policy_snapshot is not None else None
    malformed_any = False
    loaded_any_report = False
    verification_by_path: dict[str, dict[str, Any]] = {}
    report_by_path: dict[str, dict[str, Any]] = {}
    reports = list(request.reports)
    try:
        for cert_path in reports:
            cert_snapshot = _load_evaluation_report_snapshot(cert_path)
            loaded_any_report = True
            report_result = _verify_single_report(
                cert_path,
                cert_snapshot=cert_snapshot,
                baseline=request.baseline,
                baseline_snapshot=baseline_snapshot,
                baseline_payload=baseline_payload,
                baseline_digest=baseline_digest,
                policy_snapshot=policy_snapshot,
                policy_payload=policy_payload,
                tolerance=tol,
                profile=request.profile,
                allow_unverified_provenance=request.allow_unverified_provenance,
                assurance_mode=normalized_assurance_mode,
                warning_policy=normalized_warning_policy,
                json_mode=request.json_mode,
                expected_runtime_image_digest=request.expected_runtime_image_digest,
            )
            verification_by_path[str(cert_path)] = report_result.verification
            report_by_path[str(cert_path)] = report_result.report
            if report_result.errors:
                overall_ok = False
            malformed_any = malformed_any or report_result.malformed
            diagnostics.extend(report_result.diagnostics)

        if not overall_ok:
            payload = _verify_output.build_verify_json_payload(
                reports,
                ok=False,
                reason="malformed" if malformed_any else "policy_fail",
                tolerance=tol,
                load_report_fn=_load_evaluation_report,
                report_by_path=report_by_path,
                verification_by_path=verification_by_path,
            )
            return VerifyExecutionResult(
                outcome=(
                    VerifyOutcome.MALFORMED
                    if malformed_any
                    else VerifyOutcome.POLICY_FAIL
                ),
                payload=payload,
                diagnostics=tuple(diagnostics),
            )

        payload = _verify_output.build_verify_json_payload(
            reports,
            ok=True,
            reason="ok",
            tolerance=tol,
            load_report_fn=_load_evaluation_report,
            report_by_path=report_by_path,
            verification_by_path=verification_by_path,
        )
        if not request.json_mode:
            try:
                last = report_by_path.get(str(reports[-1]), {}) if reports else {}
                diagnostics.append(
                    VerifyDiagnostic(
                        level="info",
                        message=_verify_output.build_verify_success_line(last),
                    )
                )
            except _VERIFY_RECOVERABLE_EXCEPTIONS:
                pass
        return VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload=payload,
            diagnostics=tuple(diagnostics),
        )

    except InvarlockError as ce:
        payload = _verify_output.build_verify_error_payload(
            reports[0] if reports else None,
            reason="malformed" if isinstance(ce, _ValidationError) else "policy_fail",
            encoded_error=_encode_error(ce),
        )
        if not request.json_mode:
            diagnostics.append(VerifyDiagnostic(level="error", message=str(ce)))
        return VerifyExecutionResult(
            outcome=(
                VerifyOutcome.MALFORMED
                if isinstance(ce, _ValidationError)
                else VerifyOutcome.POLICY_FAIL
            ),
            payload=payload,
            diagnostics=tuple(diagnostics),
            error=ce,
            include_resolution=not loaded_any_report,
        )
    except _VERIFY_RECOVERABLE_EXCEPTIONS as e:
        payload = _verify_output.build_verify_error_payload(
            reports[0] if reports else None,
            reason=(
                "malformed" if isinstance(e, json.JSONDecodeError) else "policy_fail"
            ),
            encoded_error=_encode_error(e),
        )
        if not request.json_mode:
            diagnostics.append(
                VerifyDiagnostic(
                    level="error",
                    message=f"Verification failed: {e}",
                )
            )
        return VerifyExecutionResult(
            outcome=(
                VerifyOutcome.MALFORMED
                if isinstance(e, json.JSONDecodeError)
                else VerifyOutcome.POLICY_FAIL
            ),
            payload=payload,
            diagnostics=tuple(diagnostics),
            error=e,
            include_resolution=(
                not loaded_any_report and not isinstance(e, json.JSONDecodeError)
            ),
        )


def run_verify_reports(
    reports: list[Path],
    *,
    baseline: Path | None = None,
    policy_pack: Path | None = None,
    tolerance: float = 1e-9,
    profile: str | None = None,
    allow_unverified_provenance: bool = False,
    json_mode: bool = False,
    assurance_mode: str = "report",
    warning_policy: str = "pass",
    expected_runtime_image_digest: str | None = None,
) -> VerifyExecutionResult:
    """Verify reports and return structured machine + human output."""

    request = VerifyRequest.from_args(
        reports,
        baseline=baseline,
        policy_pack=policy_pack,
        tolerance=tolerance,
        profile=profile,
        allow_unverified_provenance=allow_unverified_provenance,
        json_mode=json_mode,
        assurance_mode=assurance_mode,
        warning_policy=warning_policy,
        expected_runtime_image_digest=expected_runtime_image_digest,
    )
    return _run_verify_request(request)


def verify_reports_contract(
    reports: list[Path],
    *,
    baseline: Path | None = None,
    policy_pack: Path | None = None,
    tolerance: float = 1e-9,
    profile: str | None = None,
    allow_unverified_provenance: bool = False,
    json_mode: bool = False,
    assurance_mode: str = "report",
    warning_policy: str = "pass",
    expected_runtime_image_digest: str | None = None,
) -> VerifyExecutionResult:
    """Verify reports and return a structured result without relying on CLI output."""
    return run_verify_reports(
        reports,
        baseline=baseline,
        policy_pack=policy_pack,
        tolerance=tolerance,
        profile=profile,
        allow_unverified_provenance=allow_unverified_provenance,
        json_mode=json_mode,
        assurance_mode=assurance_mode,
        warning_policy=warning_policy,
        expected_runtime_image_digest=expected_runtime_image_digest,
    )


__all__ = [
    "VerifyDiagnostic",
    "VerifyExecutionResult",
    "VerifyOutcome",
    "VerifyReportResult",
    "VerifyRequest",
    "run_verify_reports",
    "verify_reports_contract",
]
