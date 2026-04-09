from __future__ import annotations

import math
import time
from typing import Any

from .api import ModelAdapter, RunConfig, RunReport
from .types import LogLevel, RunStatus

_ROLLBACK_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


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
    metric_payload_invalid = bool(
        isinstance(pm, dict) and (pm.get("invalid") or pm.get("degraded"))
    )
    tail_payload_invalid = False

    drift_ratio: float | None = None
    if is_ppl_metric:
        has_preview = isinstance(pm, dict) and "preview" in pm
        has_final = isinstance(pm, dict) and "final" in pm
        if has_preview and has_final:
            if isinstance(pm_fin, int | float) and isinstance(pm_prev, int | float):
                pm_prev_val = float(pm_prev)
                pm_fin_val = float(pm_fin)
                if math.isfinite(pm_prev_val) and math.isfinite(pm_fin_val):
                    if pm_prev_val > 0.0:
                        drift_ratio = pm_fin_val / pm_prev_val
                else:
                    metric_payload_invalid = True
            else:
                metric_payload_invalid = True

    spike_threshold = getattr(config, "spike_threshold", 2.0)
    if metric_payload_invalid:
        is_catastrophic_spike = False
        metrics_acceptable = False
    elif drift_ratio is None:
        is_catastrophic_spike = False
        metrics_acceptable = True
    else:
        is_catastrophic_spike = drift_ratio > spike_threshold
        metrics_acceptable = drift_ratio <= getattr(config, "max_pm_ratio", 2.0)

    rollback_reason = None
    tail_failed = False
    pm_tail = (
        metrics.get("primary_metric_tail", {}) if isinstance(metrics, dict) else {}
    )
    if pm_tail:
        if not isinstance(pm_tail, dict):
            tail_payload_invalid = True
        else:
            mode_value = pm_tail.get("mode", "warn")
            evaluated_value = pm_tail.get("evaluated", False)
            passed_value = pm_tail.get("passed", True)
            if not isinstance(mode_value, str):
                tail_payload_invalid = True
            else:
                mode = mode_value.strip().lower() or "warn"
                if mode == "fail":
                    if not isinstance(evaluated_value, bool) or not isinstance(
                        passed_value, bool
                    ):
                        tail_payload_invalid = True
                    else:
                        tail_failed = evaluated_value and (not passed_value)

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
    elif metric_payload_invalid:
        rollback_reason = "primary_metric_invalid"
        status = RunStatus.ROLLBACK.value
    elif tail_payload_invalid:
        rollback_reason = "primary_metric_tail_invalid"
        status = RunStatus.ROLLBACK.value
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

    report.meta["rollback_reason"] = rollback_reason
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
        except _ROLLBACK_ERRORS as exc:
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
        except _ROLLBACK_ERRORS as rollback_error:
            runner._log_event(
                "runner",
                "rollback_failed",
                LogLevel.CRITICAL,
                {"error": str(rollback_error)},
            )


__all__ = ["finalize_phase", "handle_error"]
