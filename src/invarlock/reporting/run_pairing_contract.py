from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunReportPolicyViolation:
    code: str
    message: str
    details: dict[str, Any]


def _coerce_report_count(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def validate_pairing_report_metrics(
    metrics_section: Mapping[str, Any] | None,
    *,
    baseline_requested: bool,
    profile: str | None,
    preview_count_report: Any,
    final_count_report: Any,
    expected_preview: Any,
    expected_final: Any,
) -> list[RunReportPolicyViolation]:
    metrics = dict(metrics_section) if isinstance(metrics_section, Mapping) else {}
    violations: list[RunReportPolicyViolation] = []

    match_fraction = metrics.get("window_match_fraction")
    if match_fraction is not None:
        try:
            if float(match_fraction) != 1.0:
                violations.append(
                    RunReportPolicyViolation(
                        code="E001",
                        message=(
                            "PAIRING-SCHEDULE-MISMATCH: "
                            f"window_match_fraction={float(match_fraction):.3f}"
                        ),
                        details={"window_match_fraction": float(match_fraction)},
                    )
                )
        except (TypeError, ValueError, OverflowError):
            pass

    overlap_fraction = metrics.get("window_overlap_fraction")
    if overlap_fraction is not None:
        try:
            if float(overlap_fraction) > 1e-9:
                violations.append(
                    RunReportPolicyViolation(
                        code="E001",
                        message=(
                            "PAIRING-SCHEDULE-MISMATCH: "
                            f"window_overlap_fraction={float(overlap_fraction):.3f}"
                        ),
                        details={"window_overlap_fraction": float(overlap_fraction)},
                    )
                )
        except (TypeError, ValueError, OverflowError):
            pass

    profile_normalized = (profile or "").strip().lower()
    if baseline_requested and profile_normalized in {"ci", "release"}:
        pairing_reason = metrics.get("window_pairing_reason")
        if pairing_reason is not None:
            violations.append(
                RunReportPolicyViolation(
                    code="E001",
                    message=(
                        "PAIRING-SCHEDULE-MISMATCH: baseline pairing requested but "
                        f"run was not paired (window_pairing_reason={pairing_reason})"
                    ),
                    details={"window_pairing_reason": pairing_reason},
                )
            )

        paired_windows_val = metrics.get("paired_windows")
        paired_windows_int = _coerce_report_count(paired_windows_val)
        if paired_windows_int is None or paired_windows_int <= 0:
            violations.append(
                RunReportPolicyViolation(
                    code="E001",
                    message=(
                        "PAIRED-WINDOWS-COLLAPSED: paired_windows<=0 under paired "
                        "baseline. Check device stability, dataset windows, or "
                        "edit scope."
                    ),
                    details={
                        "paired_windows": paired_windows_val,
                        "profile": profile_normalized,
                    },
                )
            )

    preview_used = _coerce_report_count(preview_count_report)
    preview_expected = _coerce_report_count(expected_preview)
    final_used = _coerce_report_count(final_count_report)
    final_expected = _coerce_report_count(expected_final)
    if (
        preview_used is not None
        and preview_expected is not None
        and preview_used != preview_expected
    ) or (
        final_used is not None
        and final_expected is not None
        and final_used != final_expected
    ):
        violations.append(
            RunReportPolicyViolation(
                code="E001",
                message=(
                    "PAIRING-SCHEDULE-MISMATCH: counts do not match configuration "
                    "after stratification"
                ),
                details={
                    "preview_used": preview_used if preview_used is not None else -1,
                    "preview_expected": (
                        preview_expected if preview_expected is not None else -1
                    ),
                    "final_used": final_used if final_used is not None else -1,
                    "final_expected": (
                        final_expected if final_expected is not None else -1
                    ),
                },
            )
        )

    return violations


def build_dataset_window_stats(
    *,
    match_fraction: Any,
    overlap_fraction: Any,
    window_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    if match_fraction is not None:
        try:
            stats["window_match_fraction"] = float(match_fraction)
        except (TypeError, ValueError, OverflowError):
            pass
    if overlap_fraction is not None:
        try:
            stats["window_overlap_fraction"] = float(overlap_fraction)
        except (TypeError, ValueError, OverflowError):
            pass

    if isinstance(window_plan, Mapping) and "coverage_ok" in window_plan:
        stats["coverage"] = bool(window_plan.get("coverage_ok"))
        stats["preview_total_tokens"] = window_plan.get("preview_total_tokens")
        stats["final_total_tokens"] = window_plan.get("final_total_tokens")
        stats["min_tokens_target"] = window_plan.get("min_tokens_target")
        stats["tokens_floor_met"] = window_plan.get("tokens_floor_met")

    return stats
