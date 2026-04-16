"""Emission helpers for run orchestrator attempt execution."""

from __future__ import annotations

import math
from typing import Any

from invarlock.core.run_orchestrator_execute_helpers import RunEventEmitter
from invarlock.core.run_orchestrator_types import (
    RunAttemptStartedEvent,
    RunPrimaryMetricSummaryEvent,
    RunRetryAttemptStartedEvent,
)


def _emit_attempt_start(
    *,
    emit: RunEventEmitter,
    retry_controller: Any | None,
    attempt: int,
    max_attempts: int,
) -> None:
    if retry_controller:
        emit(
            RunAttemptStartedEvent(
                attempt=int(attempt),
                max_attempts=int(max_attempts),
            )
        )
        if attempt > 1:
            emit(
                RunRetryAttemptStartedEvent(
                    attempt=int(attempt),
                    max_attempts=int(max_attempts),
                )
            )
        return
    if attempt > 1:
        emit(RunAttemptStartedEvent(attempt=int(attempt)))


def _build_skipped_guard_overhead_payload(
    *,
    guard_overhead_threshold: float,
    skip_overhead_source: str | None,
) -> dict[str, Any]:
    skip_reason = (
        "context.run.skip_overhead_check"
        if skip_overhead_source == "config:context.run.skip_overhead_check"
        else "context.eval.skip_overhead_check"
    )
    return {
        "overhead_threshold": guard_overhead_threshold,
        "evaluated": False,
        "passed": True,
        "skipped": True,
        "skip_reason": skip_reason,
        "mode": "skipped",
        "source": skip_overhead_source or "config:context.run.skip_overhead_check",
        "diagnostics": [
            {
                "kind": "guard_overhead_info",
                "severity": "info",
                "message": "Overhead check skipped via config policy",
                "details": {},
            }
        ],
        "checks": {},
    }


def _emit_primary_metric_summary_from_report(
    *,
    report: dict[str, Any],
    emit: RunEventEmitter,
) -> None:
    try:
        pm_obj = report.get("metrics", {}).get("primary_metric")
    except (AttributeError, TypeError, KeyError):
        pm_obj = None
    if not isinstance(pm_obj, dict) or not pm_obj:
        return
    try:
        pm_kind = str(pm_obj.get("kind", "primary")).lower()
        pm_prev = pm_obj.get("preview")
        pm_fin = pm_obj.get("final")
        ratio_vs_base = pm_obj.get("ratio_vs_baseline")
        if isinstance(pm_prev, int | float) and isinstance(pm_fin, int | float):
            emit(
                RunPrimaryMetricSummaryEvent(
                    metric_kind=pm_kind,
                    preview=float(pm_prev),
                    final=float(pm_fin),
                    ratio_vs_baseline=(
                        float(ratio_vs_base)
                        if isinstance(ratio_vs_base, int | float)
                        and math.isfinite(ratio_vs_base)
                        else None
                    ),
                )
            )
    except (TypeError, ValueError):
        return


__all__ = [
    "_build_skipped_guard_overhead_payload",
    "_emit_attempt_start",
    "_emit_primary_metric_summary_from_report",
]
