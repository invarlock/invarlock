from __future__ import annotations

import math
import time
from typing import Any

from .api import ModelAdapter, RunConfig, RunReport
from .types import LogLevel, RunStatus


def finalize_phase(
    runner: Any,
    model: Any,
    adapter: ModelAdapter,
    guard_results: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
    config: RunConfig,
    report: RunReport,
) -> str:
    """Finalize or roll back based on guard and metric results."""
    runner._log_event("finalize", "start", LogLevel.INFO)
    all_guards_passed = all(
        result.get("passed", False) for result in guard_results.values()
    )

    pm = metrics.get("primary_metric", {}) if isinstance(metrics, dict) else {}
    pm_prev = pm.get("preview") if isinstance(pm, dict) else None
    pm_fin = pm.get("final") if isinstance(pm, dict) else None
    pm_kind = str(pm.get("kind", "")).lower() if isinstance(pm, dict) else ""
    is_ppl_metric = pm_kind.startswith("ppl")

    drift_ratio: float | None = None
    if is_ppl_metric:
        try:
            if isinstance(pm_fin, int | float) and isinstance(pm_prev, int | float):
                pm_prev_val = float(pm_prev)
                pm_fin_val = float(pm_fin)
                if (
                    pm_prev_val > 0.0
                    and math.isfinite(pm_prev_val)
                    and math.isfinite(pm_fin_val)
                ):
                    drift_ratio = pm_fin_val / pm_prev_val
        except Exception:
            drift_ratio = None

    spike_threshold = getattr(config, "spike_threshold", 2.0)
    if drift_ratio is None:
        is_catastrophic_spike = False
        metrics_acceptable = True
    else:
        is_catastrophic_spike = drift_ratio > spike_threshold
        metrics_acceptable = drift_ratio <= getattr(config, "max_pm_ratio", 2.0)

    rollback_reason = None
    tail_failed = False
    try:
        pm_tail = metrics.get("primary_metric_tail", {})
        if isinstance(pm_tail, dict) and pm_tail:
            mode = str(pm_tail.get("mode", "warn") or "warn").strip().lower()
            evaluated = bool(pm_tail.get("evaluated", False))
            passed = bool(pm_tail.get("passed", True))
            tail_failed = bool(mode == "fail" and evaluated and (not passed))
    except Exception:  # pragma: no cover
        tail_failed = False

    if is_catastrophic_spike:
        rollback_reason = (
            f"catastrophic_ppl_spike (ratio: {drift_ratio:.3f} > {spike_threshold})"
        )
        status = RunStatus.ROLLBACK.value
        runner._log_event(
            "finalize",
            "catastrophic_spike_detected",
            LogLevel.ERROR,
            {
                "primary_metric_drift_ratio": drift_ratio,
                "spike_threshold": spike_threshold,
                "immediate_rollback": True,
            },
        )
    elif tail_failed:
        rollback_reason = "primary_metric_tail_failed"
        status = RunStatus.ROLLBACK.value
    elif (not all_guards_passed) or (not metrics_acceptable):
        rollback_reason = "guards_failed or metrics_unacceptable"
        status = RunStatus.ROLLBACK.value
    else:
        status = RunStatus.SUCCESS.value

    if status == RunStatus.SUCCESS.value:
        runner._log_event(
            "finalize",
            "success",
            LogLevel.INFO,
            {"guards_passed": all_guards_passed, "metrics_ok": metrics_acceptable},
        )
        return status

    if runner.checkpoint_manager and "initial_checkpoint" in report.meta:
        checkpoint_id = report.meta["initial_checkpoint"]
        restored = False
        restore_error: str | None = None
        try:
            restored = bool(
                runner.checkpoint_manager.restore_checkpoint(
                    model, adapter, checkpoint_id
                )
            )
        except Exception as exc:
            restored = False
            restore_error = str(exc)

        if restored:
            runner._log_event(
                "finalize",
                "rollback",
                LogLevel.WARNING,
                {"checkpoint": checkpoint_id, "reason": rollback_reason},
            )
        else:
            runner._log_event(
                "finalize",
                "rollback_failed",
                LogLevel.CRITICAL,
                {
                    "mode": "finalize",
                    "checkpoint": checkpoint_id,
                    "reason": rollback_reason,
                    "error": restore_error or "restore_failed",
                },
            )

        report.meta["rollback_reason"] = rollback_reason
        report.meta["rollback_checkpoint"] = checkpoint_id
        report.meta["guard_recovered"] = bool(restored)
        report.meta["rollback_failed"] = not bool(restored)
        if not restored:
            report.meta["rollback_error"] = restore_error or "restore_failed"
    else:
        runner._log_event("finalize", "rollback_unavailable", LogLevel.ERROR)

    return status


def handle_error(
    runner: Any,
    error: Exception,
    report: RunReport,
    model: Any | None = None,
    adapter: ModelAdapter | None = None,
) -> None:
    """Handle pipeline errors and attempt emergency rollback."""
    report.status = RunStatus.FAILED.value
    report.error = str(error)
    report.meta["end_time"] = time.time()
    if "start_time" in report.meta:
        report.meta["duration"] = report.meta["end_time"] - report.meta["start_time"]

    runner._log_event("runner", "error", LogLevel.ERROR, {"error": str(error)})

    if runner.checkpoint_manager and "initial_checkpoint" in report.meta:
        try:
            checkpoint_id = report.meta["initial_checkpoint"]
            effective_model = model or runner._active_model
            effective_adapter = adapter or runner._active_adapter
            restored = False
            if effective_model is not None and effective_adapter is not None:
                restored = runner.checkpoint_manager.restore_checkpoint(
                    effective_model, effective_adapter, checkpoint_id
                )
            runner._log_event(
                "runner",
                "emergency_rollback",
                LogLevel.WARNING,
                {"checkpoint": checkpoint_id, "restored": restored},
            )
            if not restored:
                runner._log_event(
                    "runner",
                    "rollback_failed",
                    LogLevel.CRITICAL,
                    {"checkpoint": checkpoint_id, "error": "restore_failed"},
                )
        except Exception as rollback_error:
            runner._log_event(
                "runner",
                "rollback_failed",
                LogLevel.CRITICAL,
                {"error": str(rollback_error)},
            )


__all__ = ["finalize_phase", "handle_error"]
