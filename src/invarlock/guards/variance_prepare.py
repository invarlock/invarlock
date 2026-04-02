from __future__ import annotations

import itertools
import time
from typing import Any

import torch.nn as nn

from .variance_results import build_prepare_result

_VARIANCE_PREPARE_ERRORS = (
    ArithmeticError,
    AttributeError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


def prepare_guard(
    guard: Any,
    model: nn.Module,
    adapter=None,
    calib=None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare variance guard by resolving targets, scales, and calibration state."""
    start_time = time.time()

    if policy:
        for key in [
            "min_gain",
            "max_calib",
            "scope",
            "clamp",
            "deadband",
            "seed",
            "mode",
            "min_rel_gain",
            "alpha",
            "tie_breaker_deadband",
            "min_effect_lognll",
            "min_abs_adjust",
            "max_scale_step",
            "topk_backstop",
            "max_adjusted_modules",
            "predictive_gate",
            "predictive_one_sided",
            "absolute_floor_ppl",
            "monitor_only",
            "calibration",
            "target_modules",
        ]:
            if key in policy:
                guard._policy[key] = policy[key]
        if guard._policy.get("min_effect_lognll") is not None:
            guard._policy["min_effect_lognll"] = float(
                guard._policy["min_effect_lognll"]
            )
        guard.TIE_BREAKER_DEADBAND = float(
            guard._policy.get("tie_breaker_deadband", guard.TIE_BREAKER_DEADBAND)
        )
        guard._refresh_calibration_defaults()
        if "absolute_floor_ppl" in policy:
            guard.ABSOLUTE_FLOOR = float(
                guard._policy.get(
                    "absolute_floor_pm",
                    guard._policy.get("absolute_floor_ppl", guard.ABSOLUTE_FLOOR),
                )
            )
        if "target_modules" in policy:
            focus_list = [
                normalized
                for name in (policy.get("target_modules") or [])
                if isinstance(name, str)
                if (normalized := guard._normalize_module_name(name))
            ]
            guard._focus_modules = set(focus_list)
            if guard._focus_modules:
                guard._policy["target_modules"] = sorted(guard._focus_modules)
                guard._stats["focus_modules"] = sorted(guard._focus_modules)

    guard._log_event(
        "prepare",
        message=(
            "Preparing variance guard with "
            f"scope={guard._policy.get('scope', 'unknown')}, "
            f"min_gain={guard._policy.get('min_gain', 'unknown')}"
        ),
    )

    try:
        guard._target_modules = guard._resolve_target_modules(model, adapter)
        guard._stats["target_module_names"] = sorted(guard._target_modules.keys())
        if not guard._target_modules:
            guard._prepared = False
            guard._adapter_ref = adapter
            return build_prepare_result(
                policy=guard._policy,
                target_modules=guard._target_modules,
                scales=guard._scales,
                calibration_stats=guard._calibration_stats,
                preparation_time=time.time() - start_time,
                ready=False,
                warning="No target modules found for variance equalization",
            )

        guard._adapter_ref = adapter
        calibration_cfg = guard._policy.get("calibration", {})
        requested_windows = int(calibration_cfg.get("windows", 0) or 0)
        min_coverage = int(
            calibration_cfg.get(
                "min_coverage",
                max(1, requested_windows // 2 if requested_windows else 1),
            )
        )
        calib_seed = int(calibration_cfg.get("seed", guard._policy.get("seed", 123)))
        scale_windows = min(guard._policy["max_calib"] // 10, 50)
        limit_for_batches = max(scale_windows, requested_windows)

        calib_batches: list[Any] = []
        if calib is not None:
            if hasattr(calib, "dataloader"):
                calib_batches = guard._collect_calibration_batches(
                    calib.dataloader, limit_for_batches
                )
            elif isinstance(calib, list | tuple):
                calib_batches = list(itertools.islice(iter(calib), limit_for_batches))
            else:
                try:
                    calib_batches = list(
                        itertools.islice(iter(calib), limit_for_batches)
                    )
                except TypeError:
                    calib_batches = []

        if calib_batches:
            guard._scales = guard._compute_variance_scales(model, calib_batches)
        else:
            guard._scales = {}
            guard._raw_scales = {}
            guard._log_event(
                "prepare_warning",
                level="WARN",
                message="No calibration data provided, VE will be disabled",
            )

        guard._calibration_stats = {
            "requested": requested_windows,
            "coverage": 0,
            "min_coverage": min_coverage,
            "seed": calib_seed,
            "status": "skipped" if requested_windows == 0 else "insufficient",
        }

        calibration_batches = calib_batches[:requested_windows]
        guard._store_calibration_batches(calibration_batches)
        if calibration_batches:
            guard._evaluate_calibration_pass(
                model,
                calibration_batches,
                min_coverage,
                calib_seed,
                "prepare",
            )
        else:
            guard._ratio_ci = None
            predictive_state = {
                "evaluated": False,
                "passed": not bool(guard._policy.get("predictive_gate", True)),
                "reason": "disabled"
                if not bool(guard._policy.get("predictive_gate", True))
                else "no_calibration",
                "delta_ci": (None, None),
                "gain_ci": (None, None),
                "mean_delta": None,
            }
            guard._predictive_gate_state = predictive_state
            guard._stats["predictive_gate"] = predictive_state.copy()

        guard._stats.setdefault(
            "target_module_names", sorted(guard._target_modules.keys())
        )
        guard._stats["target_modules"] = list(guard._target_modules.keys())
        normalized_scales = {
            guard._normalize_scale_name(name): scale
            for name, scale in guard._scales.items()
        }
        guard._stats["proposed_scales_pre_edit"] = normalized_scales.copy()
        guard._stats["raw_scales_pre_edit"] = guard._raw_scales.copy()
        guard._stats["raw_scales_pre_edit_normalized"] = {
            guard._normalize_scale_name(name): scale
            for name, scale in guard._raw_scales.items()
        }
        guard._stats["total_target_modules"] = len(guard._target_modules)
        guard._stats["modules_with_scales_pre_edit"] = len(guard._scales)
        guard._stats.setdefault("calibration", {}).update(
            guard._calibration_stats.copy()
        )
        guard._stats["scale_filtering"] = {
            "raw_scales": len(guard._raw_scales),
            "filtered_scales": len(guard._scales),
            "min_abs_adjust": float(guard._policy.get("min_abs_adjust", 0.0)),
            "max_scale_step": float(guard._policy.get("max_scale_step", 0.0)),
            "topk_backstop": int(guard._policy.get("topk_backstop", 0)),
        }
        guard._stats["predictive_gate"] = guard._predictive_gate_state.copy()
        guard._calibration_stats_pre_edit = guard._calibration_stats.copy()
        guard._post_edit_evaluated = False
        guard._raw_scales_pre_edit = {
            guard._normalize_scale_name(name): scale
            for name, scale in guard._raw_scales.items()
        }

        guard._prepared = True
        preparation_time = time.time() - start_time
        guard._log_event(
            "prepare_success",
            message=f"Prepared variance guard with {len(guard._target_modules)} target modules",
            target_modules=len(guard._target_modules),
            proposed_scales=len(guard._scales),
            preparation_time=preparation_time,
        )
        return build_prepare_result(
            policy=guard._policy,
            target_modules=guard._target_modules,
            scales=guard._scales,
            calibration_stats=guard._calibration_stats,
            preparation_time=preparation_time,
            ready=True,
        )
    except _VARIANCE_PREPARE_ERRORS as error:
        guard._prepared = False
        guard._adapter_ref = adapter
        guard._log_event(
            "prepare_failed",
            level="ERROR",
            message=f"Failed to prepare variance guard: {str(error)}",
            error=str(error),
        )
        return {
            "ready": False,
            "error": str(error),
            "policy": guard._policy.copy(),
            "preparation_time": time.time() - start_time,
        }


__all__ = ["prepare_guard"]
