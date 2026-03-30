"""
Spectral Guard Implementation
=============================

Monitors spectral properties of model weights to detect instabilities.
Provides spectral control mechanisms for maintaining numerical stability.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from invarlock.core.api import Guard
from invarlock.core.types import GuardValidationResult

from . import spectral_control as _spectral_control
from . import spectral_detection as _spectral_detection
from . import spectral_measurement as _spectral_measurement
from . import spectral_policy as _spectral_policy
from . import spectral_results as _spectral_results
from . import spectral_runtime as _spectral_runtime
from . import spectral_selection as _spectral_selection


class SpectralGuard(Guard):
    """
    Spectral guard for monitoring weight matrix spectral properties.

    Tracks singular values and spectral norms to detect numerical instabilities.
    Provides automatic spectral control when violations are detected.
    """

    name = "spectral"

    def __init__(self, **kwargs: Any):
        self.config = dict(kwargs)
        self.prepared = False
        self.baseline_metrics: dict[str, Any] = {}
        self._event_records: list[dict[str, Any]] = []
        self.current_metrics: dict[str, float] = {}
        self.pre_edit_metrics: dict[str, float] = {}
        self.violations: list[dict[str, Any]] = []

        sigma_quantile = self.config.get("sigma_quantile")
        if sigma_quantile is None:
            sigma_quantile = 0.95
        self.sigma_quantile = float(sigma_quantile)
        self.config["sigma_quantile"] = self.sigma_quantile
        self.deadband = kwargs.get("deadband", 0.10)
        self.scope = kwargs.get("scope", "all")
        self.max_spectral_norm = kwargs.get("max_spectral_norm")
        if self.max_spectral_norm is not None:
            self.max_spectral_norm = float(self.max_spectral_norm)
        self.config["max_spectral_norm"] = self.max_spectral_norm
        self.correction_enabled = kwargs.get("correction_enabled", True)
        self.family_caps = _spectral_policy.normalize_family_caps(
            kwargs.get("family_caps"), default=True
        )
        self.ignore_preview_inflation = kwargs.get("ignore_preview_inflation", True)
        self.max_caps = kwargs.get("max_caps", 5)
        self.multiple_testing = _spectral_policy.normalize_multiple_testing_config(
            kwargs.get("multiple_testing")
        )
        self.estimator = _spectral_policy.normalize_estimator_config(
            kwargs.get("estimator")
        )
        self.degeneracy = _spectral_policy.normalize_degeneracy_config(
            kwargs.get("degeneracy")
        )
        self.config["multiple_testing"] = self.multiple_testing
        self.config["estimator"] = self.estimator
        self.config["degeneracy"] = self.degeneracy

        self.baseline_sigmas: dict[str, float] = {}
        self.baseline_family_stats: dict[str, dict[str, float]] = {}
        self.module_family_map: dict[str, str] = {}
        self.latest_z_scores: dict[str, float] = {}
        self.pre_edit_z_scores: dict[str, float] = {}
        self.baseline_degeneracy: dict[str, dict[str, float]] = {}
        self.target_sigma = float(self.sigma_quantile)
        self._run_profile: str | None = None
        self._scoped_modules_model_id: int | None = None
        self._scoped_modules_scope: str | None = None
        self._scoped_modules: tuple[tuple[str, Any], ...] = ()

    def _log_event(
        self, operation: str, level: str = "INFO", message: str = "", **data: Any
    ) -> None:
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
                "component": "spectral_guard",
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

    def set_run_context(self, report: Any) -> None:
        ctx = getattr(report, "context", {}) or {}
        profile = ""
        if isinstance(ctx, dict):
            profile = str(ctx.get("profile", "") or "").strip().lower()
        self._run_profile = profile or None

    def _serialize_policy(self) -> dict[str, Any]:
        return _spectral_policy.serialize_policy(self)

    def _get_scoped_modules(self, model: Any) -> tuple[tuple[str, Any], ...]:
        model_id = id(model)
        if (
            self._scoped_modules_model_id == model_id
            and self._scoped_modules_scope == self.scope
        ):
            return self._scoped_modules

        scoped_modules = tuple(
            (name, module)
            for name, module in model.named_modules()
            if self._should_check_module(name, module)
        )
        self._scoped_modules_model_id = model_id
        self._scoped_modules_scope = self.scope
        self._scoped_modules = scoped_modules
        return scoped_modules

    def prepare(
        self, model: Any, adapter: Any, calib: Any, policy: dict[str, Any]
    ) -> dict[str, Any]:
        return _spectral_runtime.prepare_guard(
            self,
            model,
            adapter,
            calib,
            policy,
            classify_model_families_fn=_spectral_detection.classify_model_families,
            compute_family_stats_fn=_spectral_detection.compute_family_stats,
            summarize_sigmas_fn=_spectral_detection.summarize_sigmas,
            percentile_fn=np.percentile,
        )

    def before_edit(self, model: Any) -> None:
        _spectral_runtime.before_edit_guard(
            self, model, compute_z_scores_fn=_spectral_detection.compute_z_scores
        )

    def after_edit(self, model: Any) -> None:
        _spectral_runtime.after_edit_guard(
            self,
            model,
            apply_spectral_control_fn=_spectral_control.apply_spectral_control,
        )

    def _capture_sigmas(self, model: Any, *, phase: str) -> dict[str, float]:
        return _spectral_measurement.capture_sigmas(
            self,
            model,
            phase=phase,
            power_iter_sigma_max_fn=_spectral_measurement.power_iter_sigma_max,
        )

    def _detect_spectral_violations(
        self, model: Any, metrics: dict[str, float], phase: str = "finalize"
    ) -> list[dict[str, Any]]:
        return _spectral_detection.detect_spectral_violations(
            self,
            model,
            metrics,
            phase=phase,
            compute_sigma_max_fn=_spectral_measurement.compute_sigma_max,
            classify_module_family_fn=_spectral_detection.classify_module_family,
            compute_z_score_for_value_fn=_spectral_detection.compute_z_score_for_value,
            default_family_caps_fn=_spectral_policy.default_family_caps,
        )

    def _should_check_module(self, name: str, module: Any) -> bool:
        return _spectral_detection.should_check_module(self, name, module)

    def _compute_family_observability(
        self,
    ) -> tuple[dict[str, dict[str, float]], dict[str, list[dict[str, Any]]]]:
        return _spectral_results.compute_family_observability(
            self.latest_z_scores or {},
            self.module_family_map,
        )

    def _select_budgeted_violations(
        self, budgeted_violations: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return _spectral_selection.select_budgeted_violations(self, budgeted_violations)

    def validate(
        self, model: Any, adapter: Any, context: dict[str, Any]
    ) -> GuardValidationResult:
        return _spectral_runtime.validate_guard(self, model, adapter, context)

    def finalize(self, model: Any) -> dict[str, Any]:
        result = _spectral_runtime.finalize_guard(self, model)
        try:
            from invarlock.reporting.evidence import maybe_dump_guard_evidence

            maybe_dump_guard_evidence(
                ".",
                {
                    "spectral": {
                        "sigma_quantile": float(self.sigma_quantile),
                        "deadband": float(self.deadband),
                        "max_caps": int(self.max_caps),
                        "multiple_testing": self.multiple_testing.get("method")
                        if isinstance(self.multiple_testing, dict)
                        else None,
                        "evaluated": True,
                    }
                },
            )
        except Exception:
            pass
        return result


__all__ = ["SpectralGuard"]
