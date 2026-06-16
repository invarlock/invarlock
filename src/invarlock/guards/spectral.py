"""
Spectral Guard Implementation
=============================

Monitors spectral properties of model weights to detect instabilities.
Provides spectral control mechanisms for maintaining numerical stability.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from fnmatch import fnmatchcase
from typing import Any

from invarlock.core import INVARLOCK_CORE_ABI as CORE_ABI
from invarlock.core.api import Guard
from invarlock.core.types import GuardValidationResult

from . import spectral_detection as _spectral_detection
from . import spectral_measurement as _spectral_measurement
from . import spectral_policy as _spectral_policy
from . import spectral_results as _spectral_results
from . import spectral_runtime as _spectral_runtime
from .adapter_modules import iter_named_adapter_scoped_modules

INVARLOCK_CORE_ABI = CORE_ABI

_EVIDENCE_DUMP_ERRORS = (ImportError, OSError, RuntimeError, TypeError, ValueError)


def finite01(value: Any) -> bool:
    try:
        numeric = float(value)
        return math.isfinite(numeric) and 0.0 <= numeric <= 1.0
    except (TypeError, ValueError):
        # guard-fallback-ok: invalid p-values are excluded from rejection sets.
        return False


def z_to_two_sided_pvalue(z: Any) -> float:
    try:
        zf = float(z)
        if not math.isfinite(zf):
            return 1.0
        return float(math.erfc(abs(zf) / math.sqrt(2.0)))
    except (TypeError, ValueError):
        # guard-fallback-ok: malformed z-scores map to neutral p=1.0.
        return 1.0


def bh_reject_families(
    family_pvals: dict[str, float], *, alpha: float, m: int
) -> set[str]:
    """BH family selection with denominator `m`."""
    if not family_pvals:
        return set()
    try:
        alpha_f = float(alpha)
    except (TypeError, ValueError):
        alpha_f = 0.05
    if not (0.0 < alpha_f <= 1.0):
        return set()

    names = list(family_pvals.keys())
    pvals = [family_pvals[name] for name in names]
    n = len(pvals)
    m_eff = max(int(m) if isinstance(m, int) else 0, n, 1)
    order = sorted(
        range(n),
        key=lambda index: float("inf") if not finite01(pvals[index]) else pvals[index],
    )
    max_k = 0
    for rank, index in enumerate(order, start=1):
        pvalue = pvals[index]
        if not finite01(pvalue):
            continue
        if pvalue <= (alpha_f * rank) / m_eff:
            max_k = rank
    if max_k <= 0:
        return set()
    cutoff = (alpha_f * max_k) / m_eff
    return {
        names[index]
        for index in order
        if finite01(pvals[index]) and pvals[index] <= cutoff
    }


def bonferroni_reject_families(
    family_pvals: dict[str, float], *, alpha: float, m: int
) -> set[str]:
    if not family_pvals:
        return set()
    try:
        alpha_f = float(alpha)
    except (TypeError, ValueError):
        alpha_f = 0.05
    if not (0.0 < alpha_f <= 1.0):
        return set()
    m_eff = max(int(m) if isinstance(m, int) else 0, len(family_pvals), 1)
    cutoff = alpha_f / m_eff
    return {
        family
        for family, pvalue in family_pvals.items()
        if finite01(pvalue) and pvalue <= cutoff
    }


def select_budgeted_violations(
    guard: Any, budgeted_violations: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply BH/Bonferroni selection at the family level."""
    mt = guard.multiple_testing if isinstance(guard.multiple_testing, dict) else {}
    method = str(mt.get("method", "bh")).lower()
    try:
        alpha = float(mt.get("alpha", 0.05) or 0.05)
    except (TypeError, ValueError):
        alpha = 0.05
    m_raw = mt.get("m")
    m = None
    try:
        if m_raw is not None:
            m = int(m_raw)
    except (TypeError, ValueError):
        m = None

    for violation in budgeted_violations:
        if violation.get("family"):
            continue
        module = violation.get("module")
        if isinstance(module, str):
            family = guard.module_family_map.get(module)
            if isinstance(family, str) and family:
                violation["family"] = family
                continue
        violation["family"] = "other"

    family_pvals: dict[str, float] = {}
    family_max_abs_z: dict[str, float] = {}
    family_counts: dict[str, int] = {}
    for violation in budgeted_violations:
        family = str(violation.get("family"))
        try:
            zf = float(violation.get("z_score"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(zf):
            continue
        pvalue = z_to_two_sided_pvalue(zf)
        family_counts[family] = family_counts.get(family, 0) + 1
        current = family_pvals.get(family)
        if current is None or pvalue < current:
            family_pvals[family] = pvalue
            family_max_abs_z[family] = abs(zf)

    families_tested = sorted(family_pvals.keys())
    m_eff = m if isinstance(m, int) and m > 0 else len(families_tested)
    m_eff = max(m_eff, len(families_tested), 1)
    if isinstance(guard.multiple_testing, dict):
        guard.multiple_testing.setdefault("m", m_eff)

    if method in {"bh", "benjamini-hochberg", "benjamini_hochberg"}:
        selected_families = bh_reject_families(family_pvals, alpha=alpha, m=m_eff)
        applied_method = "bh"
    elif method in {"bonferroni", "bonf"}:
        selected_families = bonferroni_reject_families(
            family_pvals, alpha=alpha, m=m_eff
        )
        applied_method = "bonferroni"
    else:
        selected_families = bonferroni_reject_families(
            family_pvals, alpha=alpha, m=m_eff
        )
        applied_method = "bonferroni"

    selected: list[dict[str, Any]] = []
    default_selected_without_pvalue = 0
    for violation in budgeted_violations:
        family = (
            str(violation.get("family")) if violation.get("family") is not None else ""
        )
        z_val = violation.get("z_score")
        p_val: float | None = None
        try:
            zf = float(z_val)
        except (TypeError, ValueError):
            zf = None
        if zf is not None and math.isfinite(zf):
            p_val = z_to_two_sided_pvalue(zf)
            is_selected = family in selected_families
        else:
            is_selected = True
            default_selected_without_pvalue += 1
        violation["p_value"] = p_val
        violation["selected"] = is_selected
        if is_selected:
            selected.append(violation)

    selection_metrics = {
        "method": applied_method,
        "alpha": alpha,
        "m": int(m_eff),
        "families_tested": families_tested,
        "families_selected": sorted(selected_families),
        "family_pvalues": {key: float(family_pvals[key]) for key in families_tested},
        "family_max_abs_z": {
            key: float(family_max_abs_z[key]) for key in families_tested
        },
        "family_violation_counts": dict(family_counts),
        "default_selected_without_pvalue": int(default_selected_without_pvalue),
    }
    return selected, selection_metrics


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
        self._measurement_diagnostics: list[dict[str, Any]] = []
        self.module_include_patterns: tuple[str, ...] = ()
        self.module_exclude_patterns: tuple[str, ...] = ()
        self.target_sigma = float(self.sigma_quantile)
        self._run_profile: str | None = None
        self._scoped_modules_model_id: int | None = None
        self._scoped_modules_scope: str | None = None
        self._scoped_modules_adapter_id: int | None = None
        self._scoped_modules: tuple[tuple[str, Any], ...] = ()
        self._adapter_ref: Any | None = None

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
                "timestamp": datetime.now(UTC).isoformat(),
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
        ctx = getattr(report, "context", {}) or {}
        profile = ""
        if isinstance(ctx, dict):
            profile = str(ctx.get("profile", "") or "").strip().lower()
        self._run_profile = profile or None

    def _serialize_policy(self) -> dict[str, Any]:
        return _spectral_policy.serialize_policy(self)

    def _get_scoped_modules(self, model: Any) -> tuple[tuple[str, Any], ...]:
        model_id = id(model)
        adapter = self._adapter_ref
        adapter_id = id(adapter) if adapter is not None else None
        if (
            self._scoped_modules_model_id == model_id
            and self._scoped_modules_scope == self.scope
            and self._scoped_modules_adapter_id == adapter_id
        ):
            return self._scoped_modules

        scoped_modules = tuple(
            (name, module)
            for name, module in model.named_modules()
            if self._should_check_module(name, module)
        )
        if not scoped_modules:
            scoped_modules = self._get_adapter_scoped_modules(model, adapter)

        self._scoped_modules_model_id = model_id
        self._scoped_modules_scope = self.scope
        self._scoped_modules_adapter_id = adapter_id
        self._scoped_modules = scoped_modules
        return scoped_modules

    def _get_adapter_scoped_modules(
        self, model: Any, adapter: Any | None
    ) -> tuple[tuple[str, Any], ...]:
        return tuple(
            iter_named_adapter_scoped_modules(
                model,
                adapter,
                should_include=self._should_check_module,
                log_event=self._log_event,
            )
        )

    def prepare(
        self, model: Any, adapter: Any, calib: Any, policy: dict[str, Any]
    ) -> dict[str, Any]:
        return _spectral_runtime.prepare_guard(self, model, adapter, calib, policy)

    def before_edit(self, model: Any) -> None:
        _spectral_runtime.before_edit_guard(self, model)

    def after_edit(self, model: Any) -> None:
        _spectral_runtime.after_edit_guard(self, model)

    def _capture_sigmas(self, model: Any, *, phase: str) -> dict[str, float]:
        return _spectral_measurement.capture_sigmas(self, model, phase=phase)

    def _detect_spectral_violations(
        self, model: Any, metrics: dict[str, float], phase: str = "finalize"
    ) -> list[dict[str, Any]]:
        return _spectral_detection.detect_spectral_violations(
            self,
            model,
            metrics,
            phase=phase,
        )

    def _should_check_module(self, name: str, module: Any) -> bool:
        return self._module_filter_allows(
            name
        ) and _spectral_detection.should_check_module(self, name, module)

    def _module_filter_allows(self, name: str) -> bool:
        include_patterns = self.module_include_patterns
        if include_patterns and not any(
            fnmatchcase(name, pattern) for pattern in include_patterns
        ):
            return False
        if self.module_exclude_patterns and any(
            fnmatchcase(name, pattern) for pattern in self.module_exclude_patterns
        ):
            return False
        return True

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
        return select_budgeted_violations(self, budgeted_violations)

    def validate(
        self, model: Any, adapter: Any, context: dict[str, Any]
    ) -> GuardValidationResult:
        return _spectral_runtime.validate_guard(self, model, adapter, context)

    def finalize(self, model: Any) -> dict[str, Any]:
        result = _spectral_runtime.finalize_guard(self, model)
        try:
            from invarlock.core.guard_evidence import maybe_dump_guard_evidence

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
        except _EVIDENCE_DUMP_ERRORS:
            pass
        return result


__all__ = [
    "SpectralGuard",
    "bh_reject_families",
    "bonferroni_reject_families",
    "finite01",
    "select_budgeted_violations",
    "z_to_two_sided_pvalue",
]
