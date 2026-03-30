from __future__ import annotations

import math
from typing import Any

from invarlock.core.runner_pairing import BOOTSTRAP_COVERAGE_REQUIREMENTS

_NON_FATAL_EXCEPTIONS = (
    AttributeError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)


def is_ppl_kind(name: Any) -> bool:
    try:
        normalized = str(name or "").lower()
    except _NON_FATAL_EXCEPTIONS:
        normalized = ""
    return normalized in {
        "ppl",
        "perplexity",
        "ppl_causal",
        "causal_ppl",
        "ppl_mlm",
        "mlm_ppl",
        "ppl_masked",
        "ppl_seq2seq",
        "seq2seq_ppl",
    }


def fallback_paired_windows(
    paired_windows: int, coverage_summary: dict[str, Any]
) -> int:
    if paired_windows > 0 or not isinstance(coverage_summary, dict):
        return paired_windows
    preview = coverage_summary.get("preview")
    if isinstance(preview, dict):
        used = preview.get("used")
        if isinstance(used, int | float) and used >= 0:
            return int(used)
    return paired_windows


def propagate_pairing_stats(
    evaluation_report: dict[str, Any], ppl_analysis: dict[str, Any] | None
) -> None:
    dataset = evaluation_report.get("dataset", {})
    if not isinstance(dataset, dict):
        return
    windows = dataset.get("windows", {})
    if not isinstance(windows, dict):
        windows = {}
    stats = windows.get("stats", {})
    if not isinstance(stats, dict):
        stats = {}

    pairing = None
    paired_windows_out = None
    ppl_stats = ppl_analysis.get("stats", {}) if isinstance(ppl_analysis, dict) else {}
    if isinstance(ppl_stats, dict):
        pairing = ppl_stats.get("pairing")
        paired_windows_out = ppl_stats.get("paired_windows")
        for key in (
            "requested_preview",
            "requested_final",
            "actual_preview",
            "actual_final",
            "coverage_ok",
        ):
            if key in ppl_stats:
                stats[key] = ppl_stats[key]
        for key in ("coverage", "bootstrap", "paired_delta_summary"):
            value = ppl_stats.get(key)
            if isinstance(value, dict) and value:
                stats[key] = value
        for key in ("window_match_fraction", "window_overlap_fraction"):
            value = ppl_stats.get(key)
            if value is not None:
                stats[key] = value
        value = ppl_stats.get("window_pairing_reason")
        if value is not None:
            stats["window_pairing_reason"] = value

    if pairing is not None:
        stats["pairing"] = pairing
    if paired_windows_out is not None:
        stats.setdefault("paired_windows", paired_windows_out)
    if stats is not windows.get("stats"):
        windows["stats"] = stats
    if windows is not dataset.get("windows"):
        dataset["windows"] = windows
    evaluation_report["dataset"] = dataset


def enforce_drift_ratio_identity(
    paired_windows: int,
    delta_mean: Any,
    drift_ratio: float,
    window_plan_profile: str | None,
) -> float | None:
    if not (
        paired_windows > 0
        and isinstance(delta_mean, int | float)
        and math.isfinite(delta_mean)
        and isinstance(drift_ratio, int | float)
        and math.isfinite(drift_ratio)
    ):
        return None
    ratio_from_delta = math.exp(float(delta_mean))
    tolerance = 1e-3 * max(1.0, abs(drift_ratio))
    if abs(ratio_from_delta - drift_ratio) > tolerance:
        profile = (window_plan_profile or "dev").lower()
        if profile in {"ci", "release"}:
            raise ValueError(
                "Paired ΔlogNLL mean is inconsistent with reported drift ratio."
            )
    return ratio_from_delta


def enforce_ratio_ci_alignment(
    ratio_ci_source: str,
    ratio_ci: Any,
    logloss_delta_ci: Any,
) -> None:
    if ratio_ci_source != "paired_baseline":
        return
    if not (
        isinstance(logloss_delta_ci, tuple | list)
        and len(logloss_delta_ci) == 2
        and isinstance(ratio_ci, tuple | list)
        and len(ratio_ci) == 2
    ):
        return
    expected_bounds = tuple(math.exp(bound) for bound in logloss_delta_ci)
    for observed, expected in zip(ratio_ci, expected_bounds, strict=False):
        if not (
            isinstance(observed, int | float)
            and math.isfinite(observed)
            and isinstance(expected, int | float)
            and math.isfinite(expected)
        ):
            continue
        tolerance = 5e-4 * max(1.0, abs(expected))
        if abs(float(observed) - float(expected)) > tolerance:
            raise ValueError(
                "Paired ΔlogNLL CI mismatch: ratio bounds do not match exp(Δlog bounds)."
            )


def enforce_display_ci_alignment(
    ratio_ci_source: str,
    primary_metric: Any,
    logloss_delta_ci: Any,
    window_plan_profile: str | None,
) -> None:
    if ratio_ci_source != "paired_baseline":
        return
    if not isinstance(primary_metric, dict) or not primary_metric:
        return
    try:
        kind = str(primary_metric.get("kind", "")).lower()
    except _NON_FATAL_EXCEPTIONS:
        return
    if not kind.startswith("ppl"):
        return

    def _finite_bounds(bounds: Any) -> bool:
        return (
            isinstance(bounds, tuple | list)
            and len(bounds) == 2
            and all(
                isinstance(value, int | float) and math.isfinite(value)
                for value in bounds
            )
        )

    try:
        ci = primary_metric.get("ci")
        display_ci = primary_metric.get("display_ci")
    except _NON_FATAL_EXCEPTIONS:
        return
    if not _finite_bounds(ci):
        if _finite_bounds(logloss_delta_ci):
            assert isinstance(logloss_delta_ci, tuple | list)
            primary_metric["ci"] = (
                float(logloss_delta_ci[0]),
                float(logloss_delta_ci[1]),
            )
            ci = primary_metric["ci"]
        else:
            profile = (window_plan_profile or "dev").lower()
            if profile in {"ci", "release"}:
                raise ValueError(
                    "primary_metric.ci missing for ppl-like metric under paired baseline."
                )
            return

    assert isinstance(ci, tuple | list)
    expected = tuple(math.exp(float(bound)) for bound in ci)
    if not _finite_bounds(display_ci):
        profile = (window_plan_profile or "dev").lower()
        if profile in {"ci", "release"}:
            raise ValueError(
                "primary_metric.display_ci missing for ppl-like metric under paired baseline."
            )
        primary_metric["display_ci"] = [expected[0], expected[1]]
        return

    assert isinstance(display_ci, tuple | list)
    for observed, expected_value in zip(display_ci, expected, strict=False):
        tolerance = 5e-4 * max(1.0, abs(expected_value))
        if abs(float(observed) - float(expected_value)) > tolerance:
            profile = (window_plan_profile or "dev").lower()
            if profile in {"ci", "release"}:
                raise ValueError(
                    "primary_metric.display_ci mismatch: bounds do not match exp(ci)."
                )
            primary_metric["display_ci"] = [expected[0], expected[1]]
            break


def enforce_pairing_and_coverage(
    stats: dict[str, Any] | None,
    window_plan_profile: str | None,
    tier: str | None,
) -> None:
    profile = (window_plan_profile or "dev").lower()
    if profile not in {"ci", "release"}:
        return
    if not isinstance(stats, dict):
        raise ValueError("Missing dataset window stats for CI/Release enforcement.")

    pairing_reason = stats.get("window_pairing_reason")
    if pairing_reason is not None:
        raise ValueError(
            "CI/Release requires paired baseline evidence "
            f"(window_pairing_reason={pairing_reason!r})."
        )

    match_fraction = stats.get("window_match_fraction")
    overlap_fraction = stats.get("window_overlap_fraction")
    if not (
        isinstance(match_fraction, int | float) and math.isfinite(float(match_fraction))
    ):
        raise ValueError("CI/Release requires window_match_fraction.")
    if float(match_fraction) < 0.999999:
        raise ValueError(
            f"CI/Release requires perfect pairing (window_match_fraction={float(match_fraction):.6f})."
        )

    if not (
        isinstance(overlap_fraction, int | float)
        and math.isfinite(float(overlap_fraction))
    ):
        raise ValueError("CI/Release requires window_overlap_fraction.")
    if float(overlap_fraction) > 1e-9:
        raise ValueError(
            f"CI/Release requires non-overlapping windows (window_overlap_fraction={float(overlap_fraction):.6f})."
        )

    def _coerce_count(value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed) or parsed < 0:
            return None
        if abs(parsed - round(parsed)) > 1e-9:
            return None
        return int(round(parsed))

    paired_windows = _coerce_count(stats.get("paired_windows"))
    if paired_windows is None:
        raise ValueError("CI/Release requires paired_windows metric.")
    if paired_windows == 0:
        raise ValueError("CI/Release requires paired_windows > 0.")

    actual_preview = _coerce_count(stats.get("actual_preview"))
    actual_final = _coerce_count(stats.get("actual_final"))
    if actual_preview is None or actual_final is None:
        coverage = stats.get("coverage")
        if isinstance(coverage, dict):
            if actual_preview is None:
                actual_preview = _coerce_count(coverage.get("preview", {}).get("used"))
            if actual_final is None:
                actual_final = _coerce_count(coverage.get("final", {}).get("used"))

    if actual_preview is None or actual_final is None:
        raise ValueError("CI/Release requires preview/final window counts.")
    if actual_preview != actual_final:
        raise ValueError(
            f"CI/Release requires matching preview/final counts "
            f"(preview={actual_preview}, final={actual_final})."
        )

    tier_key = str(tier or "balanced").lower()
    floors = BOOTSTRAP_COVERAGE_REQUIREMENTS.get(
        tier_key, BOOTSTRAP_COVERAGE_REQUIREMENTS["balanced"]
    )
    preview_floor = int(floors.get("preview", 0))
    final_floor = int(floors.get("final", 0))
    replicates_floor = int(floors.get("replicates", 0))

    coverage = stats.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError("CI/Release requires bootstrap coverage stats.")

    preview_used = _coerce_count(coverage.get("preview", {}).get("used"))
    final_used = _coerce_count(coverage.get("final", {}).get("used"))
    replicates_used = _coerce_count(coverage.get("replicates", {}).get("used"))
    if replicates_used is None:
        bootstrap = stats.get("bootstrap")
        if isinstance(bootstrap, dict):
            replicates_used = _coerce_count(
                bootstrap.get("replicates", bootstrap.get("n"))
            )

    if preview_used is None or final_used is None or replicates_used is None:
        raise ValueError("CI/Release requires preview/final/replicates coverage stats.")

    if preview_used < preview_floor or final_used < final_floor:
        raise ValueError(
            "CI/Release requires preview/final coverage at or above tier floors "
            f"(preview={preview_used}/{preview_floor}, final={final_used}/{final_floor})."
        )
    if replicates_used < replicates_floor:
        raise ValueError(
            "CI/Release requires bootstrap replicates at or above tier floors "
            f"(replicates={replicates_used}/{replicates_floor})."
        )
