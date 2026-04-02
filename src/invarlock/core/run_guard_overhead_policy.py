from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GuardOverheadSummary:
    evaluated: bool
    passed: bool
    overhead_percent: float | None
    overhead_ratio: float | None
    threshold_fraction: float


def _append_guard_overhead_diagnostic(
    diagnostics: list[dict[str, Any]],
    *,
    severity: str,
    message: Any,
) -> None:
    diagnostics.append(
        {
            "kind": f"guard_overhead_{severity}",
            "severity": severity,
            "message": str(message),
            "details": {},
        }
    )


def _coerce_guard_overhead_diagnostics(raw: Any) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    if isinstance(raw, list | tuple):
        for item in raw:
            if not isinstance(item, dict):
                continue
            message = item.get("message")
            if not isinstance(message, str) or not message:
                continue
            severity = item.get("severity")
            if not isinstance(severity, str) or not severity:
                severity = "info"
            details = item.get("details")
            diagnostics.append(
                {
                    "kind": str(item.get("kind") or f"guard_overhead_{severity}"),
                    "severity": severity,
                    "message": message,
                    "details": dict(details) if isinstance(details, dict) else {},
                }
            )
    return diagnostics


def normalize_guard_overhead_result(
    payload: dict[str, object] | None,
) -> dict[str, object]:
    """Normalize guard-overhead payload for tiny or degenerate runs."""
    payload = dict(payload or {})
    try:
        ratio = payload.get("overhead_ratio")
        value = float(ratio) if isinstance(ratio, int | float) else float("nan")
    except (TypeError, ValueError):
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
    diagnostics = _coerce_guard_overhead_diagnostics(resolved.pop("diagnostics", ()))
    resolved.pop("messages", None)
    resolved.pop("warnings", None)
    resolved.pop("errors", None)
    diagnostics.extend(
        _coerce_guard_overhead_diagnostics(getattr(result, "diagnostics", ()))
    )
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
            "diagnostics": diagnostics,
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
    """Build a stable typed summary for guard-overhead results."""
    info = dict(guard_overhead_info or {})
    try:
        fallback_threshold = float(default_threshold)
        if fallback_threshold < 0.0 or not math.isfinite(fallback_threshold):
            fallback_threshold = 0.01
    except (TypeError, ValueError):
        fallback_threshold = 0.01

    evaluated = bool(info.get("evaluated", True))
    passed = bool(info.get("passed", True))

    resolved_overhead_percent: float | None = None
    overhead_percent = info.get("overhead_percent")
    if isinstance(overhead_percent, (int, float)) and math.isfinite(
        float(overhead_percent)
    ):
        resolved_overhead_percent = float(overhead_percent)

    resolved_overhead_ratio: float | None = None
    ratio_value = info.get("overhead_ratio")
    if isinstance(ratio_value, (int, float)) and math.isfinite(float(ratio_value)):
        resolved_overhead_ratio = float(ratio_value)

    threshold_value = info.get("overhead_threshold", fallback_threshold)
    try:
        threshold_fraction = float(threshold_value)
    except (TypeError, ValueError):
        threshold_fraction = fallback_threshold
    return GuardOverheadSummary(
        evaluated=evaluated,
        passed=passed,
        overhead_percent=resolved_overhead_percent,
        overhead_ratio=resolved_overhead_ratio,
        threshold_fraction=threshold_fraction,
    )
