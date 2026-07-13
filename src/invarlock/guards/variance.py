"""
InvarLock: Data-Driven Variance Equalization (DD-VE)
====================================================

Branch-level variance equalizer for transformer blocks to maintain
stable residual stream dynamics after edits.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import torch.nn as nn

from invarlock.core import INVARLOCK_CORE_ABI as CORE_ABI
from invarlock.core.api import Guard
from invarlock.core.types import GuardValidationResult

from . import variance_batching as _variance_batching
from . import variance_evaluation as _variance_evaluation
from . import variance_ops as _variance_ops
from . import variance_policy as _variance_policy
from . import variance_runtime as _variance_runtime
from . import variance_scaling as _variance_scaling
from . import variance_targets as _variance_targets
from .policies import VariancePolicyDict
from .variance_results import build_prepare_result

INVARLOCK_CORE_ABI = CORE_ABI

__all__ = ["VarianceGuard", "prepare_guard"]

_EVIDENCE_DUMP_ERRORS = (ImportError, OSError, RuntimeError, TypeError, ValueError)
_VARIANCE_PREPARE_ERRORS = (
    ArithmeticError,
    AttributeError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _tap_patterns_from_policy(policy: dict[str, Any]) -> list[str]:
    tap_config = policy.get("tap")
    if isinstance(tap_config, str):
        tap_patterns = [tap_config]
    elif isinstance(tap_config, list | tuple):
        tap_patterns = [
            str(pattern)
            for pattern in tap_config
            if isinstance(pattern, str) and pattern.strip()
        ]
    else:
        tap_patterns = []
    if not tap_patterns:
        tap_patterns = ["transformer.h.*.mlp.c_proj"]
    return tap_patterns


def prepare_guard(
    guard: Any,
    model: nn.Module,
    adapter=None,
    calib=None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare variance guard by resolving targets, scales, and calibration state."""
    start_time = time.time()
    guard._prepare_failure = None

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
            "tap",
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
        if "tap" in policy:
            guard._tap_patterns = _tap_patterns_from_policy(guard._policy)
            guard._stats["tap"] = list(guard._tap_patterns)

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
            guard._prepare_failure = {
                "reason": "no_variance_targets",
                "message": "No target modules found for variance equalization",
                "target_resolution": guard._stats.get("target_resolution", {}),
            }
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
        guard._prepare_failure = {
            "reason": "prepare_error",
            "message": str(error),
            "target_resolution": guard._stats.get("target_resolution", {}),
            "target_module_names": guard._stats.get("target_module_names", []),
        }
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


class VarianceGuard(Guard):
    """Standalone Variance Guard with A/B testing for variance equalization."""

    name = "variance"
    MIN_EFFECT = 0.0

    def __init__(self, policy: VariancePolicyDict | None = None):
        from .policies import get_variance_policy

        base_policy: dict[str, Any] = dict(get_variance_policy("balanced"))
        base_calibration = base_policy.get("calibration")
        if isinstance(base_calibration, dict):
            base_calibration = dict(base_calibration)
            base_policy["calibration"] = base_calibration

        if policy:
            merged: dict[str, Any] = dict(base_policy)
            for key, value in dict(policy).items():
                if (
                    key == "calibration"
                    and isinstance(value, dict)
                    and isinstance(base_calibration, dict)
                ):
                    merged_calibration = dict(base_calibration)
                    merged_calibration.update(value)
                    merged["calibration"] = merged_calibration
                else:
                    merged[key] = value
            self._policy = merged
        else:
            self._policy = base_policy

        if policy and "tie_breaker_deadband" not in policy:
            self._policy["tie_breaker_deadband"] = 0.005
        self._policy.setdefault("mode", "ci")
        if self._policy["mode"] not in {"ci", "delta"}:
            raise ValueError("variance policy mode must be exactly 'ci' or 'delta'")
        self._policy.setdefault("min_rel_gain", 0.001)
        self._policy.setdefault("alpha", 0.05)
        self._policy.setdefault("clamp", (0.5, 2.0))
        self._policy.setdefault("seed", 123)
        self._policy.setdefault("tie_breaker_deadband", 0.005)
        self._policy.setdefault("min_abs_adjust", 0.012)
        self._policy.setdefault("max_scale_step", 0.02)
        self._policy.setdefault("topk_backstop", 1)
        self._policy.setdefault("max_adjusted_modules", 0)
        self._policy.setdefault("predictive_gate", True)
        self._policy.setdefault("predictive_one_sided", False)
        self._policy.setdefault("absolute_floor_ppl", 0.05)
        if self._policy.get("min_effect_lognll") is not None:
            self._policy["min_effect_lognll"] = float(self._policy["min_effect_lognll"])
        self._refresh_calibration_defaults()
        self._scales: dict[str, float] = {}
        self._raw_scales: dict[str, float] = {}
        self._enabled = False
        self._stats: dict[str, Any] = {}
        self._prepared = False
        self._baseline_state: dict[str, Any] | None = None
        self._event_records: list[dict[str, Any]] = []
        self._calibration_stats: dict[str, Any] = {
            "requested": 0,
            "coverage": 0,
            "min_coverage": 0,
            "seed": self._policy["calibration"]["seed"],
            "status": "uninitialized",
        }
        self.ABSOLUTE_FLOOR = float(
            self._policy.get(
                "absolute_floor_pm", self._policy.get("absolute_floor_ppl", 0.05)
            )
        )
        self._monitor_only = bool(self._policy.get("monitor_only", False))
        self._params_changed: int | None = None
        self._explicit_noop_no_change = False
        self._run_context: dict[str, Any] | None = None
        self._report_meta: dict[str, Any] | None = None
        self._dataset_meta: dict[str, Any] | None = None
        self._pairing_reference: list[str] = []
        self._pairing_digest: str | None = None
        self._adapter_ref: Any | None = None
        self._ppl_no_ve: float | None = None
        self._ppl_with_ve: float | None = None
        self._ab_gain: float | None = None
        self._ab_windows_used: int | None = None
        self._ab_seed_used: int | None = None
        self._ratio_ci: tuple[float, float] | None = None
        self._predictive_gate_state: dict[str, Any] = {
            "evaluated": False,
            "passed": False,
            "reason": "not_evaluated",
            "delta_ci": (None, None),
            "gain_ci": (None, None),
            "mean_delta": None,
        }
        self._target_modules: dict[str, nn.Module] = {}
        self._original_scales: dict[str, float] = {}
        self._prepare_failure: dict[str, Any] | None = None
        self._focus_modules = {
            self._normalize_module_name(name)
            for name in (self._policy.get("target_modules") or [])
            if isinstance(name, str)
        }
        if self._focus_modules:
            self._policy["target_modules"] = sorted(self._focus_modules)

        self._tap_patterns = _tap_patterns_from_policy(self._policy)

        self._checkpoint_stack: list[dict[str, Any]] = []
        self._last_restore_exact = True
        self._enable_attempt_count = 0
        self._disable_attempt_count = 0
        self.TIE_BREAKER_DEADBAND = float(
            self._policy.get("tie_breaker_deadband", 0.005)
        )
        self._calibration_batches: list[Any] = []
        self._calibration_window_ids: list[str] = []
        self._calibration_context: dict[str, Any] = {}
        self._calibration_stats_pre_edit: dict[str, Any] | None = None
        self._post_edit_evaluated = False
        self._raw_scales_pre_edit: dict[str, float] = {}
        self._raw_scales_post_edit: dict[str, float] = {}
        self._stats["tap"] = list(self._tap_patterns)
        if self._focus_modules:
            self._stats["focus_modules"] = sorted(self._focus_modules)
        self._stats.setdefault("ab_provenance", {})

    def _refresh_calibration_defaults(self) -> None:
        _variance_policy.refresh_calibration_defaults(self)

    def _log_event(
        self, operation: str, level: str = "INFO", message: str = "", **data
    ):
        level_code = str(level or "INFO").upper()
        severity = {
            "DEBUG": "debug",
            "INFO": "info",
            "WARN": "warning",
            "WARNING": "warning",
            "ERROR": "error",
            "CRITICAL": "critical",
        }.get(level_code, level_code.lower())
        self._event_records.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "component": "variance_guard",
                "kind": operation,
                "severity": severity,
                "summary": message,
                "details": dict(data),
                "level_code": level_code,
            }
        )

    @property
    def diagnostic_records(self) -> list[dict[str, Any]]:
        return [
            {
                "timestamp": event["timestamp"],
                "component": event["component"],
                "kind": event["kind"],
                "severity": event["severity"],
                "summary": event["summary"],
                "details": dict(event["details"]),
            }
            for event in self._event_records
        ]

    @property
    def events(self) -> list[dict[str, Any]]:
        return [
            {
                "timestamp": event["timestamp"],
                "component": event["component"],
                "operation": event["kind"],
                "level": event["level_code"],
                "message": event["summary"],
                "data": dict(event["details"]),
            }
            for event in self._event_records
        ]

    def set_run_context(self, report: Any) -> None:
        raw_report_meta = getattr(report, "meta", {}) or {}
        self._report_meta = (
            dict(raw_report_meta) if isinstance(raw_report_meta, Mapping) else {}
        )
        self._run_context = getattr(report, "context", {}) or {}
        config = self._report_meta.get("config")
        config_map = config if isinstance(config, Mapping) else {}
        model_config = config_map.get("model")
        model_map = model_config if isinstance(model_config, Mapping) else {}
        context_map = (
            self._run_context if isinstance(self._run_context, Mapping) else {}
        )
        seeds = context_map.get("seeds")
        seed_map = seeds if isinstance(seeds, Mapping) else {}
        model_id = (
            self._report_meta.get("model_id")
            or model_map.get("id")
            or context_map.get("model_id")
        )
        run_seed = self._report_meta.get("seed")
        if run_seed is None:
            run_seed = seed_map.get("python")
        if isinstance(model_id, str) and model_id:
            self._report_meta["model_id"] = model_id
        if isinstance(run_seed, int) and not isinstance(run_seed, bool):
            self._report_meta["seed"] = run_seed
        if isinstance(self._run_context, dict):
            self._dataset_meta = self._run_context.get("dataset_meta")
        else:
            self._dataset_meta = None
        if isinstance(self._dataset_meta, dict):
            self._stats.setdefault("dataset_meta", self._dataset_meta)

        pairing_reference: list[str] = []
        pairing_digest: str | None = None
        pairing_baseline = (
            self._run_context.get("pairing_baseline")
            if isinstance(self._run_context, dict)
            else None
        )
        if isinstance(pairing_baseline, dict):
            preview_section = pairing_baseline.get("preview") or {}
            final_section = pairing_baseline.get("final") or {}
            pairing_reference.extend(
                self._normalize_pairing_ids(
                    "preview", preview_section.get("window_ids") or []
                )
            )
            pairing_reference.extend(
                self._normalize_pairing_ids(
                    "final", final_section.get("window_ids") or []
                )
            )
            if pairing_reference:
                joined = "||".join(pairing_reference)
                import hashlib

                pairing_digest = hashlib.blake2s(
                    joined.encode("utf-8"), digest_size=16
                ).hexdigest()
                pairing_stats = self._stats.setdefault("pairing_reference", {})
                pairing_stats.update(
                    {"count": len(pairing_reference), "digest": pairing_digest}
                )
        self._pairing_reference = pairing_reference
        self._pairing_digest = pairing_digest
        if pairing_digest is None:
            self._stats.pop("pairing_reference", None)

        edit_info = getattr(report, "edit", {}) or {}
        params_changed = None
        edit_name = None
        if isinstance(edit_info, dict):
            edit_name = edit_info.get("name")
            deltas = edit_info.get("deltas") or {}
            if isinstance(deltas, dict):
                params_changed = deltas.get("params_changed")
        self._params_changed = params_changed
        verified_zero_change = (
            isinstance(params_changed, int)
            and not isinstance(params_changed, bool)
            and params_changed == 0
        )
        self._explicit_noop_no_change = bool(
            edit_name == "noop" and verified_zero_change
        )
        if self._explicit_noop_no_change:
            self._monitor_only = bool(self._policy.get("monitor_only", False))
            self._log_event(
                "no_adjustment_required",
                message="Variance adjustment is unnecessary for a verified no-op edit",
            )
            self._scales = {}
        elif edit_name == "noop" or verified_zero_change:
            self._monitor_only = True
            self._log_event(
                "monitor_only",
                message=(
                    "Variance guard forcing monitor-only mode "
                    "(no-op change evidence is incomplete)"
                    if edit_name == "noop"
                    else "Variance guard forcing monitor-only mode (no parameters changed)"
                ),
            )
            self._scales = {}

    def _normalize_module_name(self, name: str) -> str:
        return _variance_targets.normalize_module_name(name)

    def _matches_tap(self, name: str) -> bool:
        return _variance_targets.matches_tap(self, name)

    def _normalize_pairing_ids(self, prefix: str, window_ids) -> list[str]:
        return _variance_targets.normalize_pairing_ids(prefix, window_ids)

    def _expected_window_ids(self) -> list[str]:
        return _variance_targets.expected_window_ids(self)

    def _normalize_scale_name(self, name: str) -> str:
        return _variance_targets.normalize_scale_name(name)

    def _scale_matches_target(self, scale_name: str, target_name: str) -> bool:
        return _variance_targets.scale_matches_target(scale_name, target_name)

    def _is_focus_match(self, name: str) -> bool:
        return _variance_targets.is_focus_match(self, name)

    def _materialize_batch(self, batch: Any) -> Any:
        return _variance_batching.materialize_batch(self, batch)

    def _ensure_tensor_value(self, value: Any) -> Any:
        return _variance_batching.ensure_tensor_value(value)

    def _tensorize_calibration_batches(self, batches: list[Any]) -> list[Any]:
        return _variance_batching.tensorize_calibration_batches(self, batches)

    def _extract_window_ids(self, batches: list[Any]) -> list[str]:
        return _variance_batching.extract_window_ids(self, batches)

    def _store_calibration_batches(self, batches: list[Any]) -> None:
        _variance_batching.store_calibration_batches(self, batches)

    def _fingerprint_targets(self) -> str | None:
        return _variance_targets.fingerprint_targets(self)

    def _record_ab_provenance(
        self,
        condition: str,
        *,
        tag: str,
        window_ids,
        fingerprint: str | None,
        mode: str,
        status: str,
    ) -> None:
        _variance_targets.record_ab_provenance(
            self,
            condition,
            tag=tag,
            window_ids=window_ids,
            fingerprint=fingerprint,
            mode=mode,
            status=status,
        )

    def _resolve_target_modules(
        self, model: nn.Module, adapter: Any | None = None
    ) -> dict[str, nn.Module]:
        return _variance_targets.resolve_target_modules(self, model, adapter)

    def _compute_variance_scales(
        self, model: nn.Module, dataloader
    ) -> dict[str, float]:
        result = _variance_scaling.compute_variance_scales(
            self,
            model,
            dataloader,
        )
        self._raw_scales = result.raw_scales
        return result.filtered_scales

    def _evaluate_calibration_pass(
        self,
        model: nn.Module,
        calibration_batches: list[Any],
        min_coverage: int,
        calib_seed: int,
        tag: str,
    ) -> None:
        _variance_evaluation.evaluate_calibration_pass(
            self,
            model,
            calibration_batches,
            min_coverage,
            calib_seed,
            tag,
        )

    def _refresh_after_edit_metrics(
        self,
        model: nn.Module,
        tag: str = "post_edit",
        adapter: Any | None = None,
    ) -> None:
        _variance_evaluation.refresh_after_edit_metrics(
            self, model, tag=tag, adapter=adapter
        )

    def _collect_calibration_batches(self, dataloader, windows: int) -> list[Any]:
        return _variance_batching.collect_calibration_batches(self, dataloader, windows)

    def _prepare_batch_tensors(self, batch: Any, device):
        return _variance_batching.prepare_batch_tensors(self, batch, device)

    def _compute_ppl_for_batches(
        self,
        model: nn.Module,
        batches: list[Any],
        device,
        *,
        return_counts: bool = False,
    ):
        return _variance_batching.compute_ppl_for_batches(
            self, model, batches, device, return_counts=return_counts
        )

    def _bootstrap_mean_ci(
        self,
        samples: list[float],
        alpha: float,
        n_bootstrap: int = 500,
        seed: int | None = None,
    ):
        return _variance_batching.bootstrap_mean_ci(
            self, samples, alpha, n_bootstrap=n_bootstrap, seed=seed
        )

    def prepare(
        self,
        model: nn.Module,
        adapter=None,
        calib=None,
        policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return prepare_guard(self, model, adapter=adapter, calib=calib, policy=policy)

    def before_edit(self, model: nn.Module) -> None:
        _variance_runtime.before_edit_guard(self, model)

    def after_edit(self, model: nn.Module) -> None:
        _variance_runtime.after_edit_guard(self, model)

    def enable(self, model: nn.Module, adapter=None) -> bool:
        return _variance_ops.enable_guard(self, model, adapter=adapter)

    def disable(self, model: nn.Module, adapter=None) -> bool:
        return _variance_ops.disable_guard(self, model, adapter=adapter)

    def set_ab_results(
        self,
        ppl_no_ve: float,
        ppl_with_ve: float,
        windows_used: int | None = None,
        seed_used: int | None = None,
        ratio_ci: tuple[float, float] | None = None,
    ) -> None:
        _variance_policy.set_ab_results(
            self,
            ppl_no_ve,
            ppl_with_ve,
            windows_used=windows_used,
            seed_used=seed_used,
            ratio_ci=ratio_ci,
        )

    def _push_checkpoint(self, model: nn.Module) -> None:
        _variance_ops.push_checkpoint(self, model)

    def _pop_checkpoint(self, model: nn.Module) -> bool:
        return _variance_ops.pop_checkpoint(self, model)

    def _commit_checkpoint(self) -> None:
        _variance_ops.commit_checkpoint(self)

    def _evaluate_ab_gate(self) -> tuple[bool, str]:
        return _variance_policy.evaluate_ab_gate(self)

    def validate(
        self, model: Any, adapter: Any, context: dict[str, Any]
    ) -> GuardValidationResult:
        return _variance_runtime.validate_guard(self, model, adapter, context)

    def finalize(self, model: nn.Module) -> dict[str, Any]:
        result = _variance_runtime.finalize_guard(self, model)
        try:
            from invarlock.core.guard_evidence import maybe_dump_guard_evidence

            maybe_dump_guard_evidence(
                ".",
                {
                    "variance": {
                        "mode": self._policy.get("mode"),
                        "min_effect": self._policy.get("min_effect", self.MIN_EFFECT),
                        "predictive_one_sided": bool(
                            self._policy.get("predictive_one_sided", True)
                        ),
                        "evaluated": True,
                    }
                },
            )
        except _EVIDENCE_DUMP_ERRORS:
            pass
        return result

    def policy(self) -> VariancePolicyDict:
        return self._policy.copy()
