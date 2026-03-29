from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from invarlock.core.error_encoding import encode_error as _encode_error
from invarlock.core.exceptions import InvarlockError
from invarlock.core.exceptions import MetricsError as _MetricsError
from invarlock.core.exceptions import ValidationError as _ValidationError
from invarlock.core.provider_parity import enforce_provider_parity
from invarlock.core.runtime_attestation import verify_runtime_attestation

from . import verify_checks as _verify_checks
from . import verify_output as _verify_output

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

_coerce_float = _verify_checks._coerce_float
_coerce_int = _verify_checks._coerce_int
_load_evaluation_report = _verify_checks._load_evaluation_report
_resolve_path = _verify_checks._resolve_path
_measurement_contract_digest = _verify_checks._measurement_contract_digest
_validate_logspace_ci_identity = _verify_checks._validate_logspace_ci_identity
_validate_primary_metric = _verify_checks._validate_primary_metric
_validate_pairing = _verify_checks._validate_pairing
_validate_counts = _verify_checks._validate_counts
_validate_drift_band = _verify_checks._validate_drift_band
_validate_tokenizer_hash = _verify_checks._validate_tokenizer_hash
_validate_measurement_contracts = _verify_checks._validate_measurement_contracts
_apply_profile_lints = _verify_checks._apply_profile_lints
_report_schema = _verify_checks._report_schema
validate_report = _verify_checks.validate_report
compute_validation_flags = _verify_checks.compute_validation_flags
resolve_tiny_relax_from_report = _verify_checks.resolve_tiny_relax_from_report


@dataclass(frozen=True)
class VerifyDiagnostic:
    level: str
    message: str


class VerifyOutcome(str, Enum):
    OK = "ok"
    POLICY_FAIL = "policy_fail"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class VerifyExecutionResult:
    outcome: VerifyOutcome
    payload: dict[str, Any]
    diagnostics: tuple[VerifyDiagnostic, ...]
    error: Exception | None = None


def _validate_report_schema_strict(report: dict[str, Any]) -> bool:
    return _verify_checks._validate_report_schema_strict(
        report,
        report_schema_module=_report_schema,
    )


def _recompute_validation_flags(report: dict[str, Any]) -> dict[str, bool]:
    return _verify_checks._recompute_validation_flags(
        report,
        compute_validation_flags_fn=compute_validation_flags,
        resolve_tiny_relax_from_report_fn=resolve_tiny_relax_from_report,
    )


def _validate_primary_metric_policy(
    report: dict[str, Any], *, profile: str | None = None
) -> list[str]:
    return _verify_checks._validate_primary_metric_policy(
        report,
        profile=profile,
        recompute_validation_flags_fn=_recompute_validation_flags,
    )


def _validate_evaluation_report_payload(
    path: Path, *, profile: str | None = None
) -> list[str]:
    return _verify_checks._validate_evaluation_report_payload(
        path,
        profile=profile,
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
        validate_measurement_contracts_fn=_validate_measurement_contracts,
    )


def _warn_adapter_family_mismatch(
    cert_path: Path,
    report: dict[str, Any],
    *,
    trusted_baseline_path: Path | None = None,
) -> tuple[VerifyDiagnostic, ...]:
    """Build a soft warning if adapter families differ between baseline and edited."""

    try:
        plugins = report.get("plugins") or {}
        adapter_meta = plugins.get("adapter") if isinstance(plugins, dict) else None
        edited_family = None
        edited_lib = None
        edited_ver = None
        if isinstance(adapter_meta, dict):
            prov = adapter_meta.get("provenance")
            if isinstance(prov, dict):
                edited_family = str(prov.get("family") or "").lower() or None
                edited_lib = prov.get("library") or None
                edited_ver = prov.get("version") or None

        baseline_prov = (
            report.get("provenance")
            if isinstance(report.get("provenance"), dict)
            else {}
        )
        baseline_report_path = None
        baseline_ref = baseline_prov.get("baseline")
        if isinstance(baseline_ref, dict):
            baseline_report_path = baseline_ref.get("report_path")

        baseline_family = None
        base_lib = None
        base_ver = None
        if isinstance(baseline_report_path, str) and baseline_report_path:
            p = Path(baseline_report_path)
            trusted_path = (
                trusted_baseline_path.resolve(strict=False)
                if isinstance(trusted_baseline_path, Path)
                else None
            )
            candidate_path = p.resolve(strict=False)
            if (
                trusted_path is not None
                and candidate_path == trusted_path
                and p.is_file()
            ):
                with p.open("r", encoding="utf-8") as fh:
                    baseline_report = json.load(fh)
                meta = (
                    baseline_report.get("meta", {})
                    if isinstance(baseline_report, dict)
                    else {}
                )
                base_plugins = meta.get("plugins") if isinstance(meta, dict) else None
                if isinstance(base_plugins, dict):
                    base_adapter = base_plugins.get("adapter")
                    if isinstance(base_adapter, dict):
                        base_prov = base_adapter.get("provenance")
                        if isinstance(base_prov, dict):
                            val = base_prov.get("family")
                            if isinstance(val, str) and val:
                                baseline_family = val.lower()
                            base_lib = base_prov.get("library") or None
                            base_ver = base_prov.get("version") or None

        if edited_family and baseline_family and edited_family != baseline_family:
            base_backend = base_lib or "—"
            base_version = f"=={base_ver}" if base_lib and base_ver else "—"
            edited_backend = edited_lib or "—"
            edited_version = f"=={edited_ver}" if edited_lib and edited_ver else "—"
            return (
                VerifyDiagnostic(
                    level="warning",
                    message="Adapter family differs between baseline and edited runs:",
                ),
                VerifyDiagnostic(
                    level="warning",
                    message=(
                        f"baseline: family={baseline_family}, backend={base_backend} "
                        f"{base_version}"
                    ),
                ),
                VerifyDiagnostic(
                    level="warning",
                    message=(
                        f"edited: family={edited_family}, backend={edited_backend} "
                        f"{edited_version}"
                    ),
                ),
                VerifyDiagnostic(
                    level="warning",
                    message=(
                        "Ensure this cross-family comparison is intentional "
                        "(Compare & Evaluate flows should normally match families)."
                    ),
                ),
            )
    except _VERIFY_RECOVERABLE_EXCEPTIONS:
        return ()
    return ()


def run_verify_reports(
    reports: list[Path],
    *,
    baseline: Path | None = None,
    tolerance: float = 1e-9,
    profile: str | None = "dev",
    allow_unattested_artifacts: bool = False,
    json_mode: bool = False,
) -> VerifyExecutionResult:
    """Verify reports and return structured machine + human output."""

    overall_ok = True
    diagnostics: list[VerifyDiagnostic] = []
    try:
        tol = float(tolerance)
    except _VERIFY_RECOVERABLE_EXCEPTIONS:
        tol = 1e-9

    baseline_digest = None
    try:
        if baseline is not None:
            bdata = json.loads(baseline.read_text(encoding="utf-8"))
            prov = bdata.get("provenance") if isinstance(bdata, dict) else None
            if isinstance(prov, dict):
                pd = prov.get("provider_digest")
                if isinstance(pd, dict):
                    baseline_digest = pd
    except _VERIFY_RECOVERABLE_EXCEPTIONS:
        baseline_digest = None

    malformed_any = False
    try:
        for cert_path in reports:
            cert_obj = _load_evaluation_report(cert_path)

            try:
                prof = (profile or "").strip().lower()
            except _VERIFY_RECOVERABLE_EXCEPTIONS:
                prof = "dev"
            prov = cert_obj.get("provenance") if isinstance(cert_obj, dict) else None
            subj_digest = None
            if isinstance(prov, dict):
                subj_digest = prov.get("provider_digest")
            if prof in {"ci", "release"}:
                if not (
                    isinstance(subj_digest, dict) and subj_digest.get("ids_sha256")
                ):
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
                        profile=profile,
                        invarlock_error_cls=InvarlockError,
                    )

            errors = _validate_evaluation_report_payload(cert_path, profile=profile)
            attestation_result = verify_runtime_attestation(
                cert_path,
                allow_unattested=bool(allow_unattested_artifacts),
            )
            errors.extend(issue.message for issue in attestation_result.issues)
            if json_mode and any(
                "schema validation failed" in str(e).lower() for e in errors
            ):
                raise _ValidationError(
                    code="E601",
                    message="REPORT-SCHEMA-INVALID: schema validation failed",
                    details={"path": str(cert_path)},
                )
            is_malformed = any(
                ("schema validation failed" in e.lower())
                or ("missing primary_metric.ratio_vs_baseline" in e)
                or ("report is missing a finite primary_metric.ratio_vs_baseline" in e)
                for e in errors
            )
            malformed_any = malformed_any or is_malformed

            pm = (
                cert_obj.get("primary_metric", {}) if isinstance(cert_obj, dict) else {}
            )
            kind = (
                str(pm.get("kind") or "").strip().lower()
                if isinstance(pm, dict)
                else ""
            )
            win = (
                cert_obj.get("evaluation_windows", {})
                if isinstance(cert_obj, dict)
                else {}
            )
            fin = win.get("final") if isinstance(win, dict) else None

            if kind in {"accuracy", "vqa_accuracy"}:
                cls = (
                    cert_obj.get("metrics", {}).get("classification", {})
                    if isinstance(cert_obj.get("metrics"), dict)
                    else {}
                )
                n_correct = cls.get("n_correct") if isinstance(cls, dict) else None
                n_total = cls.get("n_total") if isinstance(cls, dict) else None
                if (
                    isinstance(n_correct, int | float)
                    and isinstance(n_total, int | float)
                    and n_total > 0
                ):
                    acc = float(n_correct) / float(n_total)
                    disp_final = pm.get("final")
                    if isinstance(disp_final, int | float):
                        if abs(float(disp_final) - acc) > max(1e-12, tol):
                            errors.append(
                                f"Accuracy mismatch: final={float(disp_final):.12f} recomputed={acc:.12f}"
                            )
                else:
                    if prof in {"ci", "release"}:
                        raise InvarlockError(
                            code="E004",
                            message=(
                                "PROVIDER-DIGEST-MISSING: missing classification aggregates for recompute in CI/Release"
                            ),
                        )
                    if not json_mode:
                        diagnostics.append(
                            VerifyDiagnostic(
                                level="warning",
                                message=(
                                    "Cannot recompute accuracy: missing aggregates "
                                    "(dev mode)."
                                ),
                            )
                        )
            else:
                if isinstance(pm, dict) and isinstance(fin, dict):
                    ll = fin.get("logloss")
                    wc = fin.get("token_counts")
                    if (
                        isinstance(ll, list)
                        and isinstance(wc, list)
                        and ll
                        and wc
                        and len(ll) == len(wc)
                    ):
                        try:
                            num = sum(
                                float(a) * float(b)
                                for a, b in zip(ll, wc, strict=False)
                            )
                            den = sum(float(b) for b in wc)
                            if den > 0:
                                recomputed_mean = float(num / den)
                                ap_final = pm.get("analysis_point_final")
                                if isinstance(ap_final, int | float):
                                    if abs(float(ap_final) - recomputed_mean) > tol:
                                        errors.append(
                                            f"Basis mismatch: analysis_point_final={ap_final:.12f} recomputed={recomputed_mean:.12f}"
                                        )
                                else:
                                    disp_final = pm.get("final")
                                    if isinstance(disp_final, int | float):
                                        if abs(
                                            float(math.exp(recomputed_mean))
                                            - float(disp_final)
                                        ) > max(1e-12, tol):
                                            errors.append(
                                                f"Display mismatch: final={float(disp_final):.12f} exp(basis)={math.exp(recomputed_mean):.12f}"
                                            )
                        except _VERIFY_RECOVERABLE_EXCEPTIONS:
                            pass
                    else:
                        if prof in {"ci", "release"}:
                            raise InvarlockError(
                                code="E004",
                                message=(
                                    "PROVIDER-DIGEST-MISSING: evaluation_windows.final missing for recompute in CI/Release"
                                ),
                            )
                        if not json_mode:
                            diagnostics.append(
                                VerifyDiagnostic(
                                    level="warning",
                                    message=(
                                        "Cannot recompute basis: "
                                        "evaluation_windows.final missing or "
                                        "incomplete (dev mode)."
                                    ),
                                )
                            )

            if (
                errors
                and prof in {"ci", "release"}
                and any(("mismatch" in str(e).lower()) for e in errors)
            ):
                first = next(
                    (e for e in errors if "mismatch" in str(e).lower()), errors[0]
                )
                raise _MetricsError(
                    code="E602",
                    message="RECOMPUTE-MISMATCH: report values disagree with recomputation",
                    details={"example": str(first)},
                )

            if errors:
                overall_ok = False
                if not json_mode:
                    diagnostics.append(
                        VerifyDiagnostic(level="fail", message=str(cert_path))
                    )
                    for err in errors:
                        diagnostics.append(
                            VerifyDiagnostic(level="detail", message=str(err))
                        )
            else:
                if not json_mode:
                    diagnostics.append(
                        VerifyDiagnostic(level="pass", message=str(cert_path))
                    )
                    try:
                        diagnostics.extend(
                            _warn_adapter_family_mismatch(
                                cert_path,
                                cert_obj,
                                trusted_baseline_path=baseline,
                            )
                        )
                    except _VERIFY_RECOVERABLE_EXCEPTIONS:
                        pass

        if not overall_ok:
            code = 2 if malformed_any else 1
            payload = _verify_output.build_verify_json_payload(
                reports,
                ok=False,
                reason="malformed" if malformed_any else "policy_fail",
                tolerance=tol,
                load_report_fn=_load_evaluation_report,
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
        )
        if not json_mode:
            try:
                last = _load_evaluation_report(reports[-1]) if reports else {}
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
        if not json_mode:
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
        )
    except _VERIFY_RECOVERABLE_EXCEPTIONS as e:
        payload = _verify_output.build_verify_error_payload(
            reports[0] if reports else None,
            reason="malformed"
            if isinstance(e, json.JSONDecodeError)
            else "policy_fail",
            encoded_error=_encode_error(e),
        )
        if not json_mode:
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
        )


def verify_reports_contract(
    reports: list[Path],
    *,
    baseline: Path | None = None,
    tolerance: float = 1e-9,
    profile: str | None = "dev",
    allow_unattested_artifacts: bool = False,
    json_mode: bool = False,
) -> VerifyExecutionResult:
    """Verify reports and return a structured result without relying on CLI output."""
    return run_verify_reports(
        reports,
        baseline=baseline,
        tolerance=tolerance,
        profile=profile,
        allow_unattested_artifacts=allow_unattested_artifacts,
        json_mode=json_mode,
    )


__all__ = [
    "VerifyDiagnostic",
    "VerifyExecutionResult",
    "VerifyOutcome",
    "run_verify_reports",
    "verify_reports_contract",
]
