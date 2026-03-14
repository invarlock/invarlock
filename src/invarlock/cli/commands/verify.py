"""
invarlock verify command
====================

Validates generated evaluation reports for internal consistency. The command
ensures schema compliance, checks that the primary metric ratio agrees with the
baseline reference, and enforces paired-window guarantees (match=1.0,
overlap=0.0).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from invarlock.core.exceptions import InvarlockError
from invarlock.core.exceptions import MetricsError as _MetricsError
from invarlock.core.exceptions import ValidationError as _ValidationError
from invarlock.reporting import report_builder as _report_builder
from invarlock.reporting.report_builder import validate_report
from invarlock.reporting.report_policy import resolve_tiny_relax_from_report
from invarlock.reporting.report_schema import REPORT_JSON_SCHEMA, REPORT_SCHEMA_VERSION
from invarlock.reporting.report_validation import compute_validation_flags

from .. import verify_checks as _verify_checks
from .. import verify_output as _verify_output
from .._json import emit as _emit_json
from .._json import encode_error as _encode_error
from .run import _enforce_provider_parity, _resolve_exit_code

console = Console()

_coerce_float = _verify_checks._coerce_float
_coerce_int = _verify_checks._coerce_int
_load_evaluation_report = _verify_checks._load_evaluation_report
_validate_logspace_ci_identity = _verify_checks._validate_logspace_ci_identity
_validate_primary_metric = _verify_checks._validate_primary_metric
_validate_pairing = _verify_checks._validate_pairing
_validate_counts = _verify_checks._validate_counts
_validate_drift_band = _verify_checks._validate_drift_band
_validate_tokenizer_hash = _verify_checks._validate_tokenizer_hash
_resolve_path = _verify_checks._resolve_path
_measurement_contract_digest = _verify_checks._measurement_contract_digest
_validate_measurement_contracts = _verify_checks._validate_measurement_contracts
_apply_profile_lints = _verify_checks._apply_profile_lints


def _validate_report_schema_strict(report: dict[str, Any]) -> bool:
    return _verify_checks._validate_report_schema_strict(
        report,
        schema_version=REPORT_SCHEMA_VERSION,
        report_json_schema=REPORT_JSON_SCHEMA,
        report_builder_module=_report_builder,
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


def _warn_adapter_family_mismatch(cert_path: Path, report: dict[str, Any]) -> None:
    """Emit a soft warning if adapter families differ between baseline and edited.

    This is a non-fatal hint to catch inadvertent cross-family comparisons.
    Tries to load the baseline report referenced in the report provenance.
    """
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
            if p.exists():
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
            # Clarify with backend library versions where available
            base_backend = base_lib or "—"
            base_version = f"=={base_ver}" if base_lib and base_ver else "—"
            edited_backend = edited_lib or "—"
            edited_version = f"=={edited_ver}" if edited_lib and edited_ver else "—"
            console.print(
                "[yellow]⚠️  Adapter family differs between baseline and edited runs:[/yellow]"
            )
            console.print(
                f"[yellow]   • baseline: family={baseline_family}, backend={base_backend} {base_version}[/yellow]"
            )
            console.print(
                f"[yellow]   • edited  : family={edited_family}, backend={edited_backend} {edited_version}[/yellow]"
            )
            console.print(
                "[yellow]   Ensure this cross-family comparison is intentional (Compare & Evaluate flows should normally match families).[/yellow]"
            )
    except Exception:
        # Non-fatal and best-effort; suppress errors
        return


def verify_command(
    reports: list[Path] = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="One or more evaluation report JSON files to verify.",
    ),
    baseline: Path | None = typer.Option(
        None,
        "--baseline",
        help="Optional baseline evaluation report (or run report) JSON to enforce provider parity.",
    ),
    tolerance: float = typer.Option(
        1e-9,
        "--tolerance",
        help="Tolerance for analysis-basis comparisons (mean log-loss).",
    ),
    profile: str | None = typer.Option(
        "dev",
        "--profile",
        help="Execution profile affecting parity enforcement and exit codes (dev|ci|release).",
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit machine-readable JSON (suppresses human-readable output)",
    ),
) -> None:
    """
    Verify evaluation report integrity.

    Ensures each evaluation report passes schema validation, ratio consistency checks,
    and strict pairing requirements (match=1.0, overlap=0.0).
    """

    overall_ok = True
    # Coerce tolerance for programmatic calls where typer.Option may be passed
    try:
        tol = float(tolerance)
    except Exception:
        tol = 1e-9

    # Optional: preload baseline provider digest for parity enforcement
    baseline_digest = None
    try:
        if baseline is not None:
            bdata = json.loads(baseline.read_text(encoding="utf-8"))
            # Accept either an evaluation report or a run report (report.json); look under provenance when present.
            prov = bdata.get("provenance") if isinstance(bdata, dict) else None
            if isinstance(prov, dict):
                pd = prov.get("provider_digest")
                if isinstance(pd, dict):
                    baseline_digest = pd
    except Exception:
        # Baseline is an optional hint only; ignore issues here and proceed
        baseline_digest = None

    malformed_any = False
    try:
        for cert_path in reports:
            cert_obj = _load_evaluation_report(cert_path)

            # Enforce provider digest presence in CI/Release profiles
            try:
                prof = (profile or "").strip().lower()
            except Exception:
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
                # If baseline provided, enforce tokenizer/masking parity
                if baseline_digest is not None:
                    _enforce_provider_parity(
                        subj_digest, baseline_digest, profile=profile
                    )

            # Structural checks
            errors = _validate_evaluation_report_payload(cert_path, profile=profile)
            # JSON path: emit a typed ValidationError for schema failures to include error.code
            if json_out and any(
                "schema validation failed" in str(e).lower() for e in errors
            ):
                raise _ValidationError(
                    code="E601",
                    message="REPORT-SCHEMA-INVALID: schema validation failed",
                    details={"path": str(cert_path)},
                )
            # Determine malformed vs policy-fail for this cert
            is_malformed = any(
                ("schema validation failed" in e.lower())
                or ("missing primary_metric.ratio_vs_baseline" in e)
                or ("report is missing a finite primary_metric.ratio_vs_baseline" in e)
                for e in errors
            )
            malformed_any = malformed_any or is_malformed

            # Determinism: recompute analysis basis when possible
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

            # Accuracy/VQA recompute — do not swallow exceptions in dev; must influence exit
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
                    elif not json_out:
                        console.print(
                            "[yellow]⚠️  Cannot recompute accuracy: missing aggregates (dev mode).[/yellow]"
                        )

            # ppl-like recompute guarded in try/except
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
                        except Exception:
                            pass
                    else:
                        if prof in {"ci", "release"}:
                            raise InvarlockError(
                                code="E004",
                                message=(
                                    "PROVIDER-DIGEST-MISSING: evaluation_windows.final missing for recompute in CI/Release"
                                ),
                            )
                        elif not json_out:
                            console.print(
                                "[yellow]⚠️  Cannot recompute basis: evaluation_windows.final missing or incomplete (dev mode).[/yellow]"
                            )

            # Treat recompute mismatches as fatal in CI/Release
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
                if not json_out:
                    console.print(f"[red]FAIL[/red] {cert_path}")
                    for err in errors:
                        console.print(f"  ↳ {err}")
            else:
                if not json_out:
                    console.print(f"[green]PASS[/green] {cert_path}")
                # Emit soft adapter-family warning after a successful structural check
                try:
                    _warn_adapter_family_mismatch(cert_path, cert_obj)
                except Exception:
                    pass

        if not overall_ok:
            code = 2 if malformed_any else 1
            if json_out:
                payload = _verify_output.build_verify_json_payload(
                    reports,
                    ok=False,
                    reason="malformed" if malformed_any else "policy_fail",
                    exit_code=code,
                    tolerance=tol,
                    load_report_fn=_load_evaluation_report,
                )
                _emit_json(payload, code)
            raise SystemExit(code)

        # Success emission
        if json_out:
            payload = _verify_output.build_verify_json_payload(
                reports,
                ok=True,
                reason="ok",
                exit_code=0,
                tolerance=tol,
                load_report_fn=_load_evaluation_report,
            )
            _emit_json(payload, 0)
        else:
            # Human-friendly success line
            try:
                last = _load_evaluation_report(reports[-1]) if reports else {}
                console.print(_verify_output.build_verify_success_line(last))
            except Exception:
                pass

    except InvarlockError as ce:
        code = _resolve_exit_code(ce, profile=profile)
        if json_out:
            reason = "malformed" if isinstance(ce, _ValidationError) else "policy_fail"
            payload = _verify_output.build_verify_error_payload(
                reports[0] if reports else None,
                reason=reason,
                exit_code=code,
                encoded_error=_encode_error(ce),
            )
            _emit_json(payload, code)
        else:
            console.print(str(ce))
        raise SystemExit(code) from ce
    except SystemExit:
        raise
    except typer.Exit:
        # Ensure single JSON emission path; let Typer/Click control exit
        raise
    except Exception as e:
        code = _resolve_exit_code(e, profile=profile)
        if json_out:
            reason = (
                "malformed" if isinstance(e, json.JSONDecodeError) else "policy_fail"
            )
            payload = _verify_output.build_verify_error_payload(
                reports[0] if reports else None,
                reason=reason,
                exit_code=code,
                encoded_error=_encode_error(e),
            )
            _emit_json(payload, code)
        else:
            console.print(f"[red]❌ Verification failed: {e}[/red]")
        raise SystemExit(code) from e
