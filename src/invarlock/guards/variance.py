"""
InvarLock – Safety: Data-Driven Variance Equalization (DD-VE)
=========================================================

Branch-level variance equalizer for transformer blocks to maintain
stable residual stream dynamics after edits.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import torch.nn as nn

from invarlock.core.api import Guard
from invarlock.core.bootstrap import compute_paired_delta_log_ci
from invarlock.core.types import GuardValidationResult

from . import variance_batching as _variance_batching
from . import variance_evaluation as _variance_evaluation
from . import variance_ops as _variance_ops
from . import variance_policy as _variance_policy
from . import variance_prepare as _variance_prepare
from . import variance_runtime as _variance_runtime
from . import variance_scaling as _variance_scaling
from . import variance_targets as _variance_targets
from .policies import VariancePolicyDict

__all__ = ["VarianceGuard"]

_EVIDENCE_DUMP_ERRORS = (ImportError, OSError, RuntimeError, TypeError, ValueError)


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
        self._focus_modules = {
            self._normalize_module_name(name)
            for name in (self._policy.get("target_modules") or [])
            if isinstance(name, str)
        }
        if self._focus_modules:
            self._policy["target_modules"] = sorted(self._focus_modules)

        tap_config = self._policy.get("tap")
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
        self._tap_patterns = tap_patterns

        self._checkpoint_stack: list[dict[str, Any]] = []
        self._enable_attempt_count = 0
        self._disable_attempt_count = 0
        self.TIE_BREAKER_DEADBAND = float(
            self._policy.get("tie_breaker_deadband", 0.005)
        )
        self.ABSOLUTE_FLOOR = 0.05
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
                "timestamp": datetime.utcnow().isoformat(),
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
        self._report_meta = getattr(report, "meta", {}) or {}
        self._run_context = getattr(report, "context", {}) or {}
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
        if isinstance(edit_info, dict):
            deltas = edit_info.get("deltas") or {}
            if isinstance(deltas, dict):
                params_changed = deltas.get("params_changed")
        if params_changed is None:
            if isinstance(edit_info, dict) and edit_info.get("name") in {"noop"}:
                params_changed = 0
            else:
                params_changed = None
        self._params_changed = params_changed
        if params_changed == 0:
            self._monitor_only = True
            self._log_event(
                "monitor_only",
                message="Variance guard forcing monitor-only mode (no parameters changed)",
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
            equalise_fn=_variance_scaling.equalise_residual_variance,
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
            compute_paired_delta_log_ci_fn=compute_paired_delta_log_ci,
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
        return _variance_prepare.prepare_guard(
            self, model, adapter=adapter, calib=calib, policy=policy
        )

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
            from invarlock.reporting.evidence import maybe_dump_guard_evidence

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
