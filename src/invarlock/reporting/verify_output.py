from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

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

FORMAT_VERIFY = "verify-v1"


def _coerce_ci_output(ci: Any) -> list[float] | None:
    if not (isinstance(ci, (tuple, list)) and len(ci) == 2):
        return None
    try:
        return [float(ci[0]), float(ci[1])]
    except _VERIFY_OUTPUT_EXCEPTIONS:
        return None


def _metric_family(kind: str) -> str:
    if kind in {"accuracy", "vqa_accuracy"}:
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
    recompute: dict[str, Any] | None = None
    try:
        family = _metric_family(kind)
        if family == "accuracy":
            metrics = cert_obj.get("metrics")
            cls = metrics.get("classification", {}) if isinstance(metrics, dict) else {}
            n_correct = cls.get("n_correct") if isinstance(cls, dict) else None
            n_total = cls.get("n_total") if isinstance(cls, dict) else None
            if (
                isinstance(n_correct, (int, float))
                and isinstance(n_total, (int, float))
                and n_total > 0
            ):
                acc = float(n_correct) / float(n_total)
                display_final = (
                    primary_metric.get("final")
                    if isinstance(primary_metric, dict)
                    else None
                )
                ok = bool(
                    isinstance(display_final, (int, float))
                    and abs(float(display_final) - acc) <= max(1e-12, tolerance)
                )
                recompute = {
                    "family": family,
                    "ok": ok,
                    "reason": None if ok else "mismatch",
                }
            else:
                recompute = {"family": family, "ok": True, "reason": "skipped"}
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
            if isinstance(final_window, dict):
                logloss = final_window.get("logloss")
                token_counts = final_window.get("token_counts")
                if (
                    isinstance(logloss, list)
                    and isinstance(token_counts, list)
                    and logloss
                    and token_counts
                    and len(logloss) == len(token_counts)
                ):
                    try:
                        numerator = sum(
                            float(a) * float(b)
                            for a, b in zip(logloss, token_counts, strict=False)
                        )
                        denominator = sum(float(b) for b in token_counts)
                        ok = True
                        if denominator > 0:
                            recomputed = float(math.exp(numerator / denominator))
                            display_final = (
                                primary_metric.get("final")
                                if isinstance(primary_metric, dict)
                                else None
                            )
                            ok = bool(
                                isinstance(display_final, (int, float))
                                and abs(float(display_final) - recomputed)
                                <= max(1e-12, tolerance)
                            )
                        recompute = {
                            "family": family,
                            "ok": ok,
                            "reason": None if ok else "mismatch",
                        }
                    except _VERIFY_OUTPUT_EXCEPTIONS:
                        recompute = {
                            "family": family,
                            "ok": True,
                            "reason": "skipped",
                        }
                else:
                    recompute = {"family": family, "ok": True, "reason": "skipped"}
    except _VERIFY_OUTPUT_EXCEPTIONS:
        recompute = None

    return recompute


def build_verify_json_result_item(
    cert_path: Path,
    cert_obj: dict[str, Any],
    *,
    ok: bool,
    reason: str,
    tolerance: float,
) -> dict[str, Any]:
    primary_metric = (
        cert_obj.get("primary_metric", {}) if isinstance(cert_obj, dict) else {}
    )
    kind = str(
        (primary_metric.get("kind") if isinstance(primary_metric, dict) else "") or ""
    ).lower()
    ratio = (
        primary_metric.get("ratio_vs_baseline")
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
    return {
        "id": str(cert_path),
        "schema_version": "v1",
        "kind": kind,
        "ok": ok,
        "reason": reason,
        "ratio_vs_baseline": float(ratio)
        if isinstance(ratio, (int, float)) and math.isfinite(float(ratio))
        else None,
        "ci": ci_out,
        "recompute": recompute,
    }


def build_verify_json_payload(
    reports: list[Path],
    *,
    ok: bool,
    reason: str,
    tolerance: float,
    load_report_fn: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for cert_path in reports:
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
                "ratio_vs_baseline": None,
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
    ratio = (
        primary_metric.get("ratio_vs_baseline")
        if isinstance(primary_metric, dict)
        else None
    )
    ci = primary_metric.get("display_ci") if isinstance(primary_metric, dict) else None
    ci_text = None
    width = None
    if isinstance(ci, (tuple, list)) and len(ci) == 2:
        try:
            ci_lo = float(ci[0])
            ci_hi = float(ci[1])
        except _VERIFY_OUTPUT_EXCEPTIONS:
            ci_text = None
            width = None
        else:
            ci_text = f"ci=[{ci_lo:.6f},{ci_hi:.6f}]"
            width = ci_hi - ci_lo
    parts = ["VERIFY OK"]
    if kind:
        parts.append(f"metric={kind}")
    if n_prev is not None and n_fin is not None:
        parts.append(f"n={n_prev}/{n_fin}")
    if isinstance(ratio, (int, float)):
        parts.append(f"point={float(ratio):.6f}")
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
