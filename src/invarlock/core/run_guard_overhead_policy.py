from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GuardOverheadSummary:
    evaluated: bool
    passed: bool
    status: str
    overhead_display: str
    threshold_fraction: float
    threshold_display: str


def normalize_guard_overhead_result(
    payload: dict[str, object] | None,
) -> dict[str, object]:
    """Normalize guard-overhead payload for tiny or degenerate runs."""
    payload = dict(payload or {})
    try:
        ratio = payload.get("overhead_ratio")
        value = float(ratio) if isinstance(ratio, int | float) else float("nan")
    except Exception:
        value = float("nan")
    if not (isinstance(value, float) and math.isfinite(value)):
        payload["evaluated"] = False
        payload["passed"] = True
    return payload


def finalize_guard_overhead_payload(
    payload: Mapping[str, Any] | None,
    result: Any,
) -> dict[str, Any]:
    """Normalize validator output into the persisted guard-overhead payload."""
    resolved = dict(payload or {})
    try:
        messages = list(getattr(result, "messages", []))
    except TypeError:  # pragma: no cover - defensive
        messages = []
    try:
        warnings = list(getattr(result, "warnings", []))
    except TypeError:  # pragma: no cover - defensive
        warnings = []
    try:
        errors = list(getattr(result, "errors", []))
    except TypeError:  # pragma: no cover - defensive
        errors = []
    try:
        checks = dict(getattr(result, "checks", {}))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        checks = {}

    metrics_obj = getattr(result, "metrics", {})
    if not isinstance(metrics_obj, dict):
        metrics_obj = {}

    overhead_ratio = metrics_obj.get("overhead_ratio")
    if overhead_ratio is None:
        overhead_ratio = getattr(result, "overhead_ratio", None)
    overhead_percent = metrics_obj.get("overhead_percent")
    if overhead_percent is None:
        overhead_percent = getattr(result, "overhead_percent", None)

    resolved.update(
        {
            "messages": messages,
            "warnings": warnings,
            "errors": errors,
            "checks": checks,
            "overhead_ratio": overhead_ratio,
            "overhead_percent": overhead_percent,
            "passed": bool(getattr(result, "passed", False)),
            "evaluated": True,
        }
    )
    normalized = normalize_guard_overhead_result(resolved)
    return dict(normalized)


def prepare_guard_overhead_report(
    guard_overhead_payload: Mapping[str, Any] | None,
    *,
    resolved_loss_type: str | None,
    core_report: Any,
    report: Mapping[str, Any] | None,
    default_threshold: float,
    extract_pm_snapshot_for_overhead_fn: Any,
    validate_guard_overhead_fn: Any,
) -> dict[str, Any]:
    """Build the persisted guard-overhead report payload."""
    payload = dict(guard_overhead_payload or {})
    if bool(payload.get("skipped", False)):
        return payload

    try:
        loss_kind = str(resolved_loss_type or "causal").lower()
        if loss_kind == "mlm":
            pm_kind_for_overhead = "ppl_mlm"
        elif loss_kind in {"seq2seq", "s2s", "t5"}:
            pm_kind_for_overhead = "ppl_seq2seq"
        else:
            pm_kind_for_overhead = "ppl_causal"

        pm_guarded = extract_pm_snapshot_for_overhead_fn(
            core_report, kind=pm_kind_for_overhead
        )
        if not isinstance(pm_guarded, dict) or not pm_guarded:
            pm_guarded = extract_pm_snapshot_for_overhead_fn(
                report, kind=pm_kind_for_overhead
            )

        payload["guarded_report"] = (
            {"metrics": {"primary_metric": pm_guarded}}
            if isinstance(pm_guarded, dict) and pm_guarded
            else None
        )
    except (AttributeError, TypeError, ValueError):
        payload["guarded_report"] = None

    bare_struct = payload.get("bare_report") or {}
    guarded_struct = payload.get("guarded_report") or {}
    result = validate_guard_overhead_fn(
        bare_struct,
        guarded_struct,
        overhead_threshold=payload.get("overhead_threshold", default_threshold),
    )
    return finalize_guard_overhead_payload(payload, result)


def build_guard_overhead_summary(
    guard_overhead_info: Mapping[str, Any] | None,
    *,
    default_threshold: float,
) -> GuardOverheadSummary:
    """Build a stable human-display summary for guard-overhead results."""
    info = dict(guard_overhead_info or {})
    try:
        fallback_threshold = float(default_threshold)
        if fallback_threshold < 0.0 or not math.isfinite(fallback_threshold):
            fallback_threshold = 0.01
    except Exception:
        fallback_threshold = 0.01

    evaluated = bool(info.get("evaluated", True))
    passed = bool(info.get("passed", True))
    status = "PASS" if passed else "FAIL"

    overhead_percent = info.get("overhead_percent")
    if isinstance(overhead_percent, (int, float)) and math.isfinite(
        float(overhead_percent)
    ):
        overhead_display = f"{float(overhead_percent):+.2f}%"
    else:
        ratio_value = info.get("overhead_ratio")
        if isinstance(ratio_value, (int, float)) and math.isfinite(float(ratio_value)):
            overhead_display = f"{float(ratio_value):.3f}x"
        else:
            overhead_display = "not evaluated"

    threshold_value = info.get("overhead_threshold", fallback_threshold)
    try:
        threshold_fraction = float(threshold_value)
    except (TypeError, ValueError):
        threshold_fraction = fallback_threshold
    threshold_display = f"≤ +{threshold_fraction * 100:.1f}%"

    if not evaluated:
        overhead_display = "not evaluated"

    return GuardOverheadSummary(
        evaluated=evaluated,
        passed=passed,
        status=status,
        overhead_display=overhead_display,
        threshold_fraction=threshold_fraction,
        threshold_display=threshold_display,
    )
