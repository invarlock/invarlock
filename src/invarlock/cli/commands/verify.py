"""
invarlock verify command
====================

Validates generated safety certificates for internal consistency. The command
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

from invarlock.cli.errors import InvarlockError
from invarlock.core.exceptions import (
    MetricsError as _MetricsError,
)
from invarlock.core.exceptions import (
    ValidationError as _ValidationError,
)
from invarlock.reporting.certificate import validate_certificate

from .._json import emit as _emit_json
from .._json import encode_error as _encode_error
from ..constants import VERIFY_FORMAT_VERSION as FORMAT_VERIFY
from .run import _enforce_provider_parity, _resolve_exit_code

console = Console()


def _coerce_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _coerce_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _load_certificate(path: Path) -> dict[str, Any]:
    """Load certificate JSON from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_primary_metric(certificate: dict[str, Any]) -> list[str]:
    """Validate primary metric ratio consistency with baseline reference."""
    errors: list[str] = []
    pm = certificate.get("primary_metric", {}) or {}
    if not isinstance(pm, dict) or not pm:
        errors.append("Certificate missing primary_metric block.")
        return errors

    kind = str(pm.get("kind", "")).lower()
    ratio_vs_baseline = pm.get("ratio_vs_baseline")
    final = pm.get("final")

    if kind.startswith("ppl"):
        baseline_ref = certificate.get("baseline_ref", {}) or {}
        baseline_pm = (
            baseline_ref.get("primary_metric")
            if isinstance(baseline_ref, dict)
            else None
        )
        baseline_final = None
        if isinstance(baseline_pm, dict):
            bv = baseline_pm.get("final")
            if isinstance(bv, (int | float)):
                baseline_final = float(bv)
        if isinstance(final, int | float) and isinstance(baseline_final, int | float):
            if baseline_final <= 0.0:
                errors.append(
                    f"Baseline final must be > 0.0 to compute ratio (found {baseline_final})."
                )
            else:
                expected_ratio = float(final) / float(baseline_final)
                if not isinstance(ratio_vs_baseline, int | float) or not math.isfinite(
                    float(ratio_vs_baseline)
                ):
                    errors.append(
                        "Certificate is missing a finite primary_metric.ratio_vs_baseline value."
                    )
                elif not math.isclose(
                    float(ratio_vs_baseline), expected_ratio, rel_tol=1e-6, abs_tol=1e-6
                ):
                    errors.append(
                        "Primary metric ratio mismatch: "
                        f"recorded={float(ratio_vs_baseline):.12f}, expected={expected_ratio:.12f}"
                    )
    else:
        if ratio_vs_baseline is None or not isinstance(ratio_vs_baseline, int | float):
            errors.append(
                "Certificate missing primary_metric.ratio_vs_baseline for non-ppl metric."
            )

    return errors


def _validate_pairing(certificate: dict[str, Any]) -> list[str]:
    """Validate window pairing metrics (PM-only location)."""
    errors: list[str] = []
    stats = certificate.get("dataset", {}).get("windows", {}).get("stats", {})

    match_fraction = stats.get("window_match_fraction")
    overlap_fraction = stats.get("window_overlap_fraction")

    if match_fraction is None:
        errors.append("Certificate missing window_match_fraction metric.")
    elif match_fraction < 0.999999:
        errors.append(
            f"window_match_fraction must be 1.0 for paired runs (found {match_fraction:.6f})."
        )

    if overlap_fraction is None:
        errors.append("Certificate missing window_overlap_fraction metric.")
    elif overlap_fraction > 1e-9:
        errors.append(
            f"window_overlap_fraction must be 0.0 (found {overlap_fraction:.6f})."
        )

    return errors


def _validate_counts(certificate: dict[str, Any]) -> list[str]:
    """Validate preview/final window counts align with dataset configuration."""
    errors: list[str] = []
    dataset = certificate.get("dataset", {})
    dataset_windows = dataset.get("windows", {})
    expected_preview = dataset_windows.get("preview")
    expected_final = dataset_windows.get("final")

    stats = dataset_windows.get("stats", {})
    coverage = stats.get("coverage", {})

    preview_used = coverage.get("preview", {}).get("used") if coverage else None
    final_used = coverage.get("final", {}).get("used") if coverage else None
    paired_windows = stats.get("paired_windows")

    if expected_preview is not None:
        if preview_used is None:
            errors.append(
                "Certificate missing coverage.preview.used for preview windows."
            )
        elif int(preview_used) != int(expected_preview):
            errors.append(
                f"Preview window count mismatch: expected {expected_preview}, observed {preview_used}."
            )

    if expected_final is not None:
        if final_used is None:
            errors.append("Certificate missing coverage.final.used for final windows.")
        elif int(final_used) != int(expected_final):
            errors.append(
                f"Final window count mismatch: expected {expected_final}, observed {final_used}."
            )

    if (
        paired_windows is not None
        and expected_preview is not None
        and int(paired_windows) != int(expected_preview)
    ):
        errors.append(
            f"Paired window count mismatch: expected {expected_preview}, observed {paired_windows}."
        )

    return errors


def _validate_drift_band(certificate: dict[str, Any]) -> list[str]:
    """Validate preview→final drift stays within the configured band (0.95–1.05)."""
    errors: list[str] = []
    pm = certificate.get("primary_metric", {}) or {}
    drift_ratio = None
    try:
        prev = pm.get("preview")
        fin = pm.get("final")
        if isinstance(prev, int | float) and isinstance(fin, int | float) and prev > 0:
            drift_ratio = float(fin) / float(prev)
    except Exception:
        drift_ratio = None

    if not isinstance(drift_ratio, int | float):
        errors.append("Certificate missing preview/final to compute drift ratio.")
        return errors

    if not 0.95 <= float(drift_ratio) <= 1.05:
        errors.append(
            f"Preview→final drift ratio out of band (0.95–1.05): observed {drift_ratio:.6f}."
        )

    return errors


def _validate_tokenizer_hash(certificate: dict[str, Any]) -> list[str]:
    """Validate tokenizer hash consistency between baseline and edited runs.

    The check is enforced only when both hashes are present to preserve
    compatibility with legacy certificates. When present and different,
    the verification fails.
    """
    errors: list[str] = []
    meta = certificate.get("meta", {}) or {}
    dataset = certificate.get("dataset", {}) or {}
    edited_hash = None
    try:
        # Prefer meta.tokenizer_hash; fall back to dataset.tokenizer.hash
        edited_hash = meta.get("tokenizer_hash") or (
            (dataset.get("tokenizer") or {}).get("hash")
            if isinstance(dataset.get("tokenizer"), dict)
            else None
        )
    except Exception:
        edited_hash = None

    baseline_ref = certificate.get("baseline_ref", {}) or {}
    baseline_hash = baseline_ref.get("tokenizer_hash")

    if isinstance(edited_hash, str) and isinstance(baseline_hash, str):
        if edited_hash and baseline_hash and edited_hash != baseline_hash:
            errors.append("Tokenizer hash mismatch between baseline and edited runs.")
    # If either hash is missing, skip the check for backward compatibility
    return errors


def _resolve_path(payload: Any, path: str) -> Any:
    """Resolve dotted paths within nested dictionaries."""
    current = payload
    for segment in path.split("."):
        if isinstance(current, dict):
            current = current.get(segment)
        else:
            return None
    return current


def _apply_profile_lints(certificate: dict[str, Any]) -> list[str]:
    """Apply model-profile specific lint rules embedded in the certificate."""
    errors: list[str] = []
    meta = certificate.get("meta", {})
    profile = meta.get("model_profile") if isinstance(meta, dict) else None
    if not isinstance(profile, dict):
        return errors

    lints = profile.get("cert_lints", [])
    if not isinstance(lints, list):
        return errors

    for lint in lints:
        if not isinstance(lint, dict):
            continue
        lint_type = str(lint.get("type", "")).lower()
        path = lint.get("path")
        expected = lint.get("value")
        message = lint.get("message") or "Model profile lint failed."
        actual = _resolve_path(certificate, path) if isinstance(path, str) else None

        if lint_type == "equals":
            if actual != expected:
                errors.append(
                    f"{message} Expected {path} == {expected!r}, observed {actual!r}."
                )
        elif lint_type == "gte":
            try:
                actual_val = float(actual)
                expected_val = float(expected)
            except (TypeError, ValueError):
                errors.append(
                    f"{message} Expected numeric comparison for {path}, observed {actual!r}."
                )
            else:
                if actual_val < expected_val:
                    errors.append(
                        f"{message} Expected {path} ≥ {expected_val}, observed {actual_val}."
                    )
        elif lint_type == "lte":
            try:
                actual_val = float(actual)
                expected_val = float(expected)
            except (TypeError, ValueError):
                errors.append(
                    f"{message} Expected numeric comparison for {path}, observed {actual!r}."
                )
            else:
                if actual_val > expected_val:
                    errors.append(
                        f"{message} Expected {path} ≤ {expected_val}, observed {actual_val}."
                    )

    return errors


def _validate_certificate_payload(
    path: Path, *, profile: str | None = None
) -> list[str]:
    """Run all verification checks for a single certificate."""
    errors: list[str] = []
    certificate = _load_certificate(path)

    # Always surface schema validation failures for this payload
    if not validate_certificate(certificate):
        errors.append("Certificate schema validation failed.")
        return errors

    errors.extend(_validate_primary_metric(certificate))
    errors.extend(_validate_pairing(certificate))
    errors.extend(_validate_counts(certificate))
    try:
        prof = (
            (profile or "").strip().lower()
            if isinstance(profile, str | None)
            else "dev"
        )
    except Exception:
        prof = "dev"
    # Enforce drift band only for CI/Release; skip in dev profile
    if prof in {"ci", "release"}:
        errors.extend(_validate_drift_band(certificate))
    errors.extend(_apply_profile_lints(certificate))
    errors.extend(_validate_tokenizer_hash(certificate))
    # Release-only enforcement: guard overhead must be measured or explicitly skipped.
    if prof == "release":
        go = certificate.get("guard_overhead")
        if not isinstance(go, dict) or not go:
            errors.append(
                "Release verification requires guard_overhead (missing). "
                "Set INVARLOCK_SKIP_OVERHEAD_CHECK=1 to explicitly skip during certification."
            )
        else:
            skipped = bool(go.get("skipped", False)) or (
                str(go.get("mode", "")).strip().lower() == "skipped"
            )
            if not skipped:
                evaluated = go.get("evaluated")
                if evaluated is not True:
                    errors.append(
                        "Release verification requires evaluated guard_overhead (not evaluated). "
                        "Set INVARLOCK_SKIP_OVERHEAD_CHECK=1 to explicitly skip during certification."
                    )
                ratio = go.get("overhead_ratio")
                if ratio is None:
                    errors.append(
                        "Release verification requires guard_overhead.overhead_ratio (missing)."
                    )
    # Legacy cross-checks removed; primary_metric is canonical

    return errors


def _warn_adapter_family_mismatch(cert_path: Path, certificate: dict[str, Any]) -> None:
    """Emit a soft warning if adapter families differ between baseline and edited.

    This is a non-fatal hint to catch inadvertent cross-family comparisons.
    Tries to load the baseline report referenced in the certificate provenance.
    """
    try:
        plugins = certificate.get("plugins") or {}
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
            certificate.get("provenance")
            if isinstance(certificate.get("provenance"), dict)
            else {}
        )
        baseline_report_path = None
        if isinstance(baseline_prov, dict):
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
                "[yellow]   Ensure this cross-family comparison is intentional (Compare & Certify flows should normally match families).[/yellow]"
            )
    except Exception:
        # Non-fatal and best-effort; suppress errors
        return


def verify_command(
    certificates: list[Path] = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="One or more certificate JSON files to verify.",
    ),
    baseline: Path | None = typer.Option(
        None,
        "--baseline",
        help="Optional baseline certificate/report JSON to enforce provider parity.",
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
    Verify certificate integrity.

    Ensures each certificate passes schema validation, ratio consistency checks,
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
            # Accept either a certificate or a raw report; look under provenance when present
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
        for cert_path in certificates:
            cert_obj = _load_certificate(cert_path)

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
            errors = _validate_certificate_payload(cert_path, profile=profile)
            # JSON path: emit a typed ValidationError for schema failures to include error.code
            if json_out and any(
                "schema validation failed" in str(e).lower() for e in errors
            ):
                raise _ValidationError(
                    code="E601",
                    message="CERTIFICATE-SCHEMA-INVALID: schema validation failed",
                    details={"path": str(cert_path)},
                )
            # Determine malformed vs policy-fail for this cert
            is_malformed = any(
                ("schema validation failed" in e.lower())
                or ("missing primary_metric.ratio_vs_baseline" in e)
                or (
                    "Certificate is missing a finite primary_metric.ratio_vs_baseline"
                    in e
                )
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
                    message="RECOMPUTE-MISMATCH: certificate values disagree with recomputation",
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
                # Build per-certificate results payload
                results: list[dict[str, Any]] = []
                for cert_path in certificates:
                    try:
                        cert_obj = _load_certificate(cert_path)
                    except Exception:
                        cert_obj = {}
                    pm = (
                        cert_obj.get("primary_metric", {})
                        if isinstance(cert_obj, dict)
                        else {}
                    )
                    kind = str(
                        (pm.get("kind") if isinstance(pm, dict) else "") or ""
                    ).lower()
                    ratio = (
                        pm.get("ratio_vs_baseline") if isinstance(pm, dict) else None
                    )
                    ci = pm.get("display_ci") if isinstance(pm, dict) else None
                    if not (isinstance(ci, tuple | list) and len(ci) == 2):
                        ci_out = None
                    else:
                        try:
                            ci_out = [float(ci[0]), float(ci[1])]  # type: ignore[index]
                        except Exception:
                            ci_out = None
                    # Recompute summary for JSON (best-effort)
                    recompute: dict[str, Any] | None = None
                    try:
                        fam = (
                            "accuracy"
                            if kind in {"accuracy", "vqa_accuracy"}
                            else ("ppl" if kind.startswith("ppl") else "other")
                        )
                        if fam == "accuracy":
                            cls = (
                                cert_obj.get("metrics", {}).get("classification", {})
                                if isinstance(cert_obj.get("metrics"), dict)
                                else {}
                            )
                            n_correct = (
                                cls.get("n_correct") if isinstance(cls, dict) else None
                            )
                            n_total = (
                                cls.get("n_total") if isinstance(cls, dict) else None
                            )
                            if (
                                isinstance(n_correct, int | float)
                                and isinstance(n_total, int | float)
                                and n_total > 0
                            ):
                                acc = float(n_correct) / float(n_total)
                                df = pm.get("final") if isinstance(pm, dict) else None
                                ok = bool(
                                    isinstance(df, int | float)
                                    and abs(float(df) - acc) <= max(1e-12, tol)
                                )
                                recompute = {
                                    "family": fam,
                                    "ok": ok,
                                    "reason": None if ok else "mismatch",
                                }
                            else:
                                recompute = {
                                    "family": fam,
                                    "ok": True,
                                    "reason": "skipped",
                                }
                        elif fam == "ppl":
                            ev = (
                                cert_obj.get("evaluation_windows", {})
                                if isinstance(cert_obj, dict)
                                else {}
                            )
                            finw = ev.get("final") if isinstance(ev, dict) else None
                            if isinstance(finw, dict):
                                ll = finw.get("logloss")
                                wc = finw.get("token_counts")
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
                                        ok = True
                                        if den > 0:
                                            recomputed = float(math.exp(num / den))
                                            df = (
                                                pm.get("final")
                                                if isinstance(pm, dict)
                                                else None
                                            )
                                            ok = bool(
                                                isinstance(df, int | float)
                                                and abs(float(df) - recomputed)
                                                <= max(1e-12, tol)
                                            )
                                        recompute = {
                                            "family": fam,
                                            "ok": ok,
                                            "reason": None if ok else "mismatch",
                                        }
                                    except Exception:
                                        recompute = {
                                            "family": fam,
                                            "ok": True,
                                            "reason": "skipped",
                                        }
                                else:
                                    recompute = {
                                        "family": fam,
                                        "ok": True,
                                        "reason": "skipped",
                                    }
                        # else: leave None
                    except Exception:
                        recompute = None

                    item = {
                        "id": str(cert_path),
                        "schema_version": "v1",
                        "kind": kind,
                        "ok": False,
                        "reason": "malformed" if malformed_any else "policy_fail",
                        "ratio_vs_baseline": float(ratio)
                        if isinstance(ratio, int | float)
                        and math.isfinite(float(ratio))
                        else None,
                        "ci": ci_out,
                        "recompute": recompute,
                    }
                    results.append(item)
                payload = {
                    "format_version": FORMAT_VERIFY,
                    "summary": {
                        "ok": False,
                        "reason": "malformed" if malformed_any else "policy_fail",
                    },
                    "certificate": {"count": len(certificates)},
                    "results": results,
                    "resolution": {"exit_code": code},
                }
                _emit_json(payload, code)
            raise SystemExit(code)

        # Success emission
        if json_out:
            # Build per-certificate success results payload
            results: list[dict[str, Any]] = []
            for cert_path in certificates:
                try:
                    cert_obj = _load_certificate(cert_path)
                except Exception:
                    cert_obj = {}
                pm = (
                    cert_obj.get("primary_metric", {})
                    if isinstance(cert_obj, dict)
                    else {}
                )
                kind = str(
                    (pm.get("kind") if isinstance(pm, dict) else "") or ""
                ).lower()
                ratio = pm.get("ratio_vs_baseline") if isinstance(pm, dict) else None
                ci = pm.get("display_ci") if isinstance(pm, dict) else None
                if not (isinstance(ci, tuple | list) and len(ci) == 2):
                    ci_out = None
                else:
                    try:
                        ci_out = [float(ci[0]), float(ci[1])]  # type: ignore[index]
                    except Exception:
                        ci_out = None
                # Recompute summary for JSON (best-effort)
                recompute: dict[str, Any] | None = None
                try:
                    fam = (
                        "accuracy"
                        if kind in {"accuracy", "vqa_accuracy"}
                        else ("ppl" if kind.startswith("ppl") else "other")
                    )
                    if fam == "accuracy":
                        cls = (
                            cert_obj.get("metrics", {}).get("classification", {})
                            if isinstance(cert_obj.get("metrics"), dict)
                            else {}
                        )
                        n_correct = (
                            cls.get("n_correct") if isinstance(cls, dict) else None
                        )
                        n_total = cls.get("n_total") if isinstance(cls, dict) else None
                        if (
                            isinstance(n_correct, int | float)
                            and isinstance(n_total, int | float)
                            and n_total > 0
                        ):
                            acc = float(n_correct) / float(n_total)
                            df = pm.get("final") if isinstance(pm, dict) else None
                            ok = bool(
                                isinstance(df, int | float)
                                and abs(float(df) - acc) <= max(1e-12, tol)
                            )
                            recompute = {
                                "family": fam,
                                "ok": ok,
                                "reason": None if ok else "mismatch",
                            }
                        else:
                            recompute = {"family": fam, "ok": True, "reason": "skipped"}
                    elif fam == "ppl":
                        ev = (
                            cert_obj.get("evaluation_windows", {})
                            if isinstance(cert_obj, dict)
                            else {}
                        )
                        finw = ev.get("final") if isinstance(ev, dict) else None
                        if isinstance(finw, dict):
                            ll = finw.get("logloss")
                            wc = finw.get("token_counts")
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
                                    ok = True
                                    if den > 0:
                                        recomputed = float(math.exp(num / den))
                                        df = (
                                            pm.get("final")
                                            if isinstance(pm, dict)
                                            else None
                                        )
                                        ok = bool(
                                            isinstance(df, int | float)
                                            and abs(float(df) - recomputed)
                                            <= max(1e-12, tol)
                                        )
                                    recompute = {
                                        "family": fam,
                                        "ok": ok,
                                        "reason": None if ok else "mismatch",
                                    }
                                except Exception:
                                    recompute = {
                                        "family": fam,
                                        "ok": True,
                                        "reason": "skipped",
                                    }
                            else:
                                recompute = {
                                    "family": fam,
                                    "ok": True,
                                    "reason": "skipped",
                                }
                    # else: leave None
                except Exception:
                    recompute = None

                item = {
                    "id": str(cert_path),
                    "schema_version": "v1",
                    "kind": kind,
                    "ok": True,
                    "reason": "ok",
                    "ratio_vs_baseline": float(ratio)
                    if isinstance(ratio, int | float) and math.isfinite(float(ratio))
                    else None,
                    "ci": ci_out,
                    "recompute": recompute,
                }
                results.append(item)
            payload = {
                "format_version": FORMAT_VERIFY,
                "summary": {"ok": True, "reason": "ok"},
                "certificate": {"count": len(certificates)},
                "results": results,
                "resolution": {"exit_code": 0},
            }
            _emit_json(payload, 0)
        else:
            # Human-friendly success line
            try:
                last = _load_certificate(certificates[-1]) if certificates else {}
                pm = last.get("primary_metric", {}) if isinstance(last, dict) else {}
                kind = str(pm.get("kind") or "").strip()
                ppl = last.get("ppl", {}) if isinstance(last, dict) else {}
                n_prev = (
                    ppl.get("stats", {})
                    .get("coverage", {})
                    .get("preview", {})
                    .get("used")
                    if isinstance(ppl, dict)
                    else None
                )
                n_fin = (
                    ppl.get("stats", {})
                    .get("coverage", {})
                    .get("final", {})
                    .get("used")
                    if isinstance(ppl, dict)
                    else None
                )
                ratio = pm.get("ratio_vs_baseline") if isinstance(pm, dict) else None
                ci = pm.get("display_ci") if isinstance(pm, dict) else None
                width = (
                    (float(ci[1]) - float(ci[0]))
                    if (isinstance(ci, tuple | list) and len(ci) == 2)
                    else None
                )
                parts = ["VERIFY OK"]
                if kind:
                    parts.append(f"metric={kind}")
                if n_prev is not None and n_fin is not None:
                    parts.append(f"n={n_prev}/{n_fin}")
                if isinstance(ratio, int | float):
                    parts.append(f"point={float(ratio):.6f}")
                if isinstance(ci, tuple | list) and len(ci) == 2:
                    parts.append(f"ci=[{float(ci[0]):.6f},{float(ci[1]):.6f}]")
                if isinstance(width, int | float):
                    parts.append(f"width={width:.6f}")
                console.print(" ".join(parts))
            except Exception:
                pass

    except InvarlockError as ce:
        code = _resolve_exit_code(ce, profile=profile)
        if json_out:
            reason = "malformed" if isinstance(ce, _ValidationError) else "policy_fail"
            payload = {
                "format_version": FORMAT_VERIFY,
                "summary": {"ok": False, "reason": reason},
                "results": [
                    {
                        "id": str(certificates[0]) if certificates else "",
                        "schema_version": "v1",
                        "kind": "",
                        "ok": False,
                        "reason": reason,
                        "ratio_vs_baseline": None,
                        "ci": None,
                    }
                ],
                "resolution": {"exit_code": code},
                "error": _encode_error(ce),
            }
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
            payload = {
                "format_version": FORMAT_VERIFY,
                "summary": {"ok": False, "reason": reason},
                "results": [
                    {
                        "id": str(certificates[0]) if certificates else "",
                        "schema_version": "v1",
                        "kind": "",
                        "ok": False,
                        "reason": reason,
                        "ratio_vs_baseline": None,
                        "ci": None,
                    }
                ],
                "resolution": {"exit_code": code},
                "error": _encode_error(e),
            }
            _emit_json(payload, code)
        else:
            console.print(f"[red]❌ Verification failed: {e}[/red]")
        raise SystemExit(code) from e
