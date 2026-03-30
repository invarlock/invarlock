from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any, cast

from invarlock.eval.tail_stats import evaluate_metric_tail

from .types import LogLevel

EvaluateMetricTailFn = Callable[..., dict[str, Any]]


def eval_phase(
    runner: Any,
    model: Any,
    adapter: Any,
    calibration_data: Any,
    report: Any,
    preview_n: int | None = None,
    final_n: int | None = None,
    config: Any | None = None,
    *,
    evaluate_metric_tail_fn: EvaluateMetricTailFn = evaluate_metric_tail,
) -> dict[str, Any]:
    """Run the final evaluation phase and attach metrics to the report."""
    runner._log_event("eval", "start", LogLevel.INFO)

    if calibration_data is not None:
        if os.environ.get("INVARLOCK_DEBUG_TRACE"):
            length_hint = None
            try:
                length_hint = len(calibration_data)
            except Exception:  # pragma: no cover - defensive
                length_hint = None
            first_batch = None
            indexable = hasattr(calibration_data, "__getitem__")
            if isinstance(calibration_data, list | tuple):
                if calibration_data:
                    first_batch = calibration_data[0]
            elif indexable:
                try:
                    first_batch = calibration_data[0]
                except Exception:  # pragma: no cover - defensive
                    first_batch = None
            masked_preview = None
            first_keys = None
            if isinstance(first_batch, dict):
                first_keys = list(first_batch.keys())
                labels_preview = first_batch.get("labels")
                if isinstance(labels_preview, list | tuple):
                    try:
                        masked_preview = sum(
                            1 for token in labels_preview if token != -100
                        )
                    except Exception:  # pragma: no cover - defensive
                        masked_preview = None
            runner._log_event(
                "eval",
                "calibration_snapshot",
                LogLevel.DEBUG,
                {
                    "calibration_type": type(calibration_data).__name__,
                    "length_hint": length_hint,
                    "indexable": bool(indexable),
                    "first_batch_keys": first_keys,
                    "first_batch_masked": masked_preview,
                },
            )
        computed_metrics, computed_windows = runner._compute_real_metrics(
            model,
            calibration_data,
            adapter,
            preview_n,
            final_n,
            config,
        )
        metrics = cast("dict[str, Any]", computed_metrics)
        eval_windows = cast("dict[str, Any]", computed_windows)
    else:
        runner._log_event(
            "eval",
            "warning",
            LogLevel.WARNING,
            {
                "message": "No calibration data provided; evaluation skipped.",
                "state": "not_evaluated",
            },
        )
        metrics = {
            "eval_state": {
                "evaluated": False,
                "reason": "missing_calibration_data",
            },
        }
        eval_windows = {"preview": {}, "final": {}}

    pm = metrics.get("primary_metric", {}) if isinstance(metrics, dict) else {}
    pm_kind = str(pm.get("kind", "")).lower() if isinstance(pm, dict) else ""
    is_ppl_metric = pm_kind.startswith("ppl")

    baseline_eval: dict[str, Any] = {}
    if (
        is_ppl_metric
        and config
        and isinstance(config.context, dict)
        and isinstance(config.context.get("baseline_eval_windows"), dict)
    ):
        baseline_eval = config.context.get("baseline_eval_windows") or {}

    if is_ppl_metric and baseline_eval:
        tier_policies = (
            report.meta.get("tier_policies", {})
            if isinstance(getattr(report, "meta", None), dict)
            else {}
        )
        metrics_policy = (
            tier_policies.get("metrics", {}) if isinstance(tier_policies, dict) else {}
        )
        pm_tail_policy = (
            metrics_policy.get("pm_tail", {})
            if isinstance(metrics_policy, dict)
            else {}
        )

        run_final = (
            eval_windows.get("final", {}) if isinstance(eval_windows, dict) else {}
        )
        base_final = (
            baseline_eval.get("final", {}) if isinstance(baseline_eval, dict) else {}
        )

        deltas: list[float] = []
        weights: list[float] = []
        run_ids = run_final.get("window_ids") if isinstance(run_final, dict) else None
        run_ll = run_final.get("logloss") if isinstance(run_final, dict) else None
        run_tc = run_final.get("token_counts") if isinstance(run_final, dict) else None
        base_ids = (
            base_final.get("window_ids") if isinstance(base_final, dict) else None
        )
        base_ll = base_final.get("logloss") if isinstance(base_final, dict) else None

        if (
            isinstance(run_ids, list)
            and isinstance(run_ll, list)
            and isinstance(base_ids, list)
            and isinstance(base_ll, list)
        ):
            base_map: dict[int, float] = {}
            for baseline_id, baseline_val in zip(base_ids, base_ll, strict=False):
                if isinstance(baseline_id, int | float) and isinstance(
                    baseline_val, int | float
                ):
                    base_map[int(baseline_id)] = float(baseline_val)
            for index, (run_id, run_val) in enumerate(
                zip(run_ids, run_ll, strict=False)
            ):
                if not (
                    isinstance(run_id, int | float) and isinstance(run_val, int | float)
                ):
                    continue
                key = int(run_id)
                if key not in base_map:
                    continue
                delta_value = float(run_val) - base_map[key]
                if math.isfinite(delta_value):
                    deltas.append(float(delta_value))
                    if isinstance(run_tc, list) and index < len(run_tc):
                        weight_value = float(run_tc[index])
                        weights.append(float(max(weight_value, 0.0)))

        tail_result = evaluate_metric_tail_fn(
            deltas=deltas,
            weights=weights if (weights and len(weights) == len(deltas)) else None,
            policy=pm_tail_policy if isinstance(pm_tail_policy, dict) else None,
        )
        tail_result["source"] = "paired_baseline.final"
        metrics["primary_metric_tail"] = tail_result

    policy_flags = runner._resolve_policy_flags(config)
    eval_error = metrics.get("eval_error") if isinstance(metrics, dict) else None
    if eval_error:
        if policy_flags["strict_eval"]:
            raise RuntimeError(
                f"Evaluation failed: {eval_error.get('message', 'unknown error')}"
            )
        runner._log_event(
            "eval",
            "soft_fail",
            LogLevel.WARNING,
            {"message": eval_error.get("message"), "type": eval_error.get("type")},
        )

    if hasattr(report, "metrics"):
        report.metrics.update(metrics)
    else:
        report.metrics = metrics

    report.evaluation_windows = eval_windows
    runner._log_event("eval", "complete", LogLevel.INFO, {"metrics": metrics})
    return metrics


__all__ = ["eval_phase"]
