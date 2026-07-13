from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from invarlock.public_contracts import VERIFY_OUTPUT_FORMAT_VERSION

_VERIFY_OUTPUT_EXCEPTIONS = (
    AttributeError,
    FileNotFoundError,
    json.JSONDecodeError,
    KeyError,
    OSError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)

FORMAT_VERIFY = VERIFY_OUTPUT_FORMAT_VERSION


def _is_non_bool_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _coerce_finite_float(value: Any) -> float | None:
    if not _is_non_bool_number(value):
        return None
    try:
        coerced = float(value)
    except _VERIFY_OUTPUT_EXCEPTIONS:
        return None
    return coerced if math.isfinite(coerced) else None


def _coerce_ci_output(ci: Any) -> list[float] | None:
    if not (isinstance(ci, (tuple, list)) and len(ci) == 2):
        return None
    if isinstance(ci[0], bool) or isinstance(ci[1], bool):
        return None
    try:
        return [float(ci[0]), float(ci[1])]
    except _VERIFY_OUTPUT_EXCEPTIONS:
        return None


def _metric_family(kind: str) -> str:
    if kind == "accuracy":
        return "accuracy"
    if kind.startswith("ppl"):
        return "ppl"
    return "other"


def _build_recompute_summary(
    cert_obj: dict[str, Any],
    *,
    kind: str,
    primary_metric: dict[str, Any],
    tolerance: float,
) -> dict[str, Any] | None:
    family = _metric_family(kind)
    if family == "other":
        return None

    def not_performed(reason: str) -> dict[str, Any]:
        return {
            "family": family,
            "performed": False,
            "ok": None,
            "reason": reason,
        }

    try:
        if family == "accuracy":
            metrics = cert_obj.get("metrics")
            cls = metrics.get("classification", {}) if isinstance(metrics, dict) else {}
            n_correct = cls.get("n_correct") if isinstance(cls, dict) else None
            n_total = cls.get("n_total") if isinstance(cls, dict) else None
            n_correct_value = _coerce_finite_float(n_correct)
            n_total_value = _coerce_finite_float(n_total)
            if n_correct is None or n_total is None:
                return not_performed("missing_evidence")
            if n_correct_value is None or n_total_value is None:
                return not_performed("malformed_evidence")
            if n_total_value == 0.0:
                return not_performed("zero_denominator")
            if (
                n_total_value < 0.0
                or n_correct_value < 0.0
                or n_correct_value > n_total_value
            ):
                return not_performed("malformed_evidence")
            if n_total_value > 0.0:
                acc = n_correct_value / n_total_value
                display_final = (
                    primary_metric.get("final")
                    if isinstance(primary_metric, dict)
                    else None
                )
                display_final_value = _coerce_finite_float(display_final)
                ok = bool(
                    display_final_value is not None
                    and abs(display_final_value - acc) <= max(1e-12, tolerance)
                )
                recompute = {
                    "family": family,
                    "performed": True,
                    "ok": ok,
                    "reason": None if ok else "mismatch",
                }
                return recompute
        elif family == "ppl":
            evaluation_windows = (
                cert_obj.get("evaluation_windows", {})
                if isinstance(cert_obj, dict)
                else {}
            )
            final_window = (
                evaluation_windows.get("final")
                if isinstance(evaluation_windows, dict)
                else None
            )
            if not isinstance(final_window, dict):
                return not_performed("missing_evidence")
            logloss = final_window.get("logloss")
            token_counts = final_window.get("token_counts")
            if logloss is None or token_counts is None:
                return not_performed("missing_evidence")
            if not isinstance(logloss, list) or not isinstance(token_counts, list):
                return not_performed("malformed_evidence")
            if not logloss or not token_counts:
                return not_performed("missing_evidence")
            if len(logloss) != len(token_counts):
                return not_performed("malformed_evidence")

            losses = [_coerce_finite_float(value) for value in logloss]
            weights = [_coerce_finite_float(value) for value in token_counts]
            if any(value is None for value in losses + weights):
                return not_performed("malformed_evidence")
            finite_losses = [float(value) for value in losses if value is not None]
            finite_weights = [float(value) for value in weights if value is not None]
            if any(value < 0.0 for value in finite_weights):
                return not_performed("malformed_evidence")
            denominator = math.fsum(finite_weights)
            if denominator <= 0.0:
                return not_performed("zero_denominator")
            numerator = math.fsum(
                loss * weight
                for loss, weight in zip(finite_losses, finite_weights, strict=True)
            )
            recomputed = float(math.exp(numerator / denominator))
            if not math.isfinite(recomputed) or recomputed <= 0.0:
                return not_performed("malformed_evidence")
            display_final = (
                primary_metric.get("final")
                if isinstance(primary_metric, dict)
                else None
            )
            display_final_value = _coerce_finite_float(display_final)
            ok = bool(
                display_final_value is not None
                and abs(display_final_value - recomputed) <= max(1e-12, tolerance)
            )
            return {
                "family": family,
                "performed": True,
                "ok": ok,
                "reason": None if ok else "mismatch",
            }
    except _VERIFY_OUTPUT_EXCEPTIONS:
        return not_performed("malformed_evidence")

    return not_performed("malformed_evidence")


def build_verify_json_result_item(
    cert_path: Path,
    cert_obj: dict[str, Any],
    *,
    ok: bool,
    reason: str,
    tolerance: float,
    verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary_metric = (
        cert_obj.get("primary_metric", {}) if isinstance(cert_obj, dict) else {}
    )
    kind = str(
        (primary_metric.get("kind") if isinstance(primary_metric, dict) else "") or ""
    ).lower()
    comparison = (
        primary_metric.get(
            "delta_vs_baseline_pp" if kind == "accuracy" else "ratio_vs_baseline"
        )
        if isinstance(primary_metric, dict)
        else None
    )
    ci = primary_metric.get("display_ci") if isinstance(primary_metric, dict) else None
    ci_out = _coerce_ci_output(ci)
    recompute = _build_recompute_summary(
        cert_obj,
        kind=kind,
        primary_metric=primary_metric,
        tolerance=tolerance,
    )
    guard_warnings = (
        cert_obj.get("guard_warnings", {}) if isinstance(cert_obj, dict) else {}
    )
    warning_count = 0
    warnings_present = False
    if isinstance(guard_warnings, dict):
        try:
            warning_count = int(guard_warnings.get("warning_count") or 0)
        except _VERIFY_OUTPUT_EXCEPTIONS:
            warnings = guard_warnings.get("warnings")
            warning_count = len(warnings) if isinstance(warnings, list) else 0
        warnings_present = bool(guard_warnings.get("present")) or warning_count > 0
    item = {
        "id": str(cert_path),
        "schema_version": "v1",
        "kind": kind,
        "ok": ok,
        "reason": reason,
        "ci": ci_out,
        "recompute": recompute,
        "guard_warnings_present": warnings_present,
        "warning_count": warning_count,
    }
    if kind == "accuracy":
        item["delta_vs_baseline_pp"] = _coerce_finite_float(comparison)
    elif kind.startswith("ppl"):
        item["ratio_vs_baseline"] = _coerce_finite_float(comparison)
    if verification:
        item["verification"] = verification
    return item


def build_verify_json_payload(
    reports: list[Path],
    *,
    ok: bool,
    reason: str,
    tolerance: float,
    load_report_fn: Callable[[Path], dict[str, Any]],
    report_by_path: dict[str, dict[str, Any]] | None = None,
    verification_by_path: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for cert_path in reports:
        cert_obj = (report_by_path or {}).get(str(cert_path))
        if cert_obj is None:
            try:
                cert_obj = load_report_fn(cert_path)
            except _VERIFY_OUTPUT_EXCEPTIONS:
                cert_obj = {}
        results.append(
            build_verify_json_result_item(
                cert_path,
                cert_obj,
                ok=ok,
                reason=reason,
                tolerance=tolerance,
                verification=(verification_by_path or {}).get(str(cert_path)),
            )
        )

    return {
        "format_version": FORMAT_VERIFY,
        "summary": {"ok": ok, "reason": reason},
        "evaluation_report": {"count": len(reports)},
        "results": results,
    }


def build_verify_error_payload(
    report_path: Path | None,
    *,
    reason: str,
    encoded_error: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERIFY,
        "summary": {"ok": False, "reason": reason},
        "results": [
            {
                "id": str(report_path) if report_path is not None else "",
                "schema_version": "v1",
                "kind": "",
                "ok": False,
                "reason": reason,
                "ci": None,
            }
        ],
        "error": encoded_error,
    }


def build_verify_success_line(report: dict[str, Any]) -> str:
    primary_metric = (
        report.get("primary_metric", {}) if isinstance(report, dict) else {}
    )
    kind = str(primary_metric.get("kind") or "").strip()
    ppl = report.get("ppl", {}) if isinstance(report, dict) else {}
    n_prev = (
        ppl.get("stats", {}).get("coverage", {}).get("preview", {}).get("used")
        if isinstance(ppl, dict)
        else None
    )
    n_fin = (
        ppl.get("stats", {}).get("coverage", {}).get("final", {}).get("used")
        if isinstance(ppl, dict)
        else None
    )
    kind_name = kind.lower()
    comparison = (
        primary_metric.get(
            "delta_vs_baseline_pp" if kind_name == "accuracy" else "ratio_vs_baseline"
        )
        if isinstance(primary_metric, dict)
        else None
    )
    ci = primary_metric.get("display_ci") if isinstance(primary_metric, dict) else None
    ci_out = _coerce_ci_output(ci)
    ci_text = None
    width = None
    comparison_value = _coerce_finite_float(comparison)
    if ci_out is not None:
        ci_lo, ci_hi = ci_out
        ci_text = f"ci=[{ci_lo:.6f},{ci_hi:.6f}]"
        width = ci_hi - ci_lo
    parts = ["VERIFY OK"]
    if kind:
        parts.append(f"metric={kind}")
    if _is_non_bool_number(n_prev) and _is_non_bool_number(n_fin):
        parts.append(f"n={n_prev}/{n_fin}")
    if kind_name == "accuracy":
        if comparison_value is not None:
            parts.append(f"delta_vs_baseline_pp={comparison_value:.6f}")
    elif comparison_value is not None:
        parts.append(f"point={comparison_value:.6f}")
    if ci_text is not None:
        parts.append(ci_text)
    if isinstance(width, (int, float)):
        parts.append(f"width={width:.6f}")
    return " ".join(parts)


__all__ = [
    "FORMAT_VERIFY",
    "build_verify_error_payload",
    "build_verify_json_payload",
    "build_verify_json_result_item",
    "build_verify_success_line",
]
