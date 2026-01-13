"""
Spectral Guard Implementation
============================

Monitors spectral properties of model weights to detect instabilities.
Provides spectral control mechanisms for maintaining numerical stability.
"""

import math
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, TypedDict

import numpy as np
import torch

from invarlock.cli._evidence import maybe_dump_guard_evidence
from invarlock.core.api import Guard
from invarlock.core.exceptions import ValidationError

from ._contracts import guard_assert
from ._estimators import frobenius_norm_sq, power_iter_sigma_max, row_col_norm_extrema


def _z_to_two_sided_pvalue(z: Any) -> float:
    try:
        zf = float(z)
        if not math.isfinite(zf):
            return 1.0
        return float(math.erfc(abs(zf) / math.sqrt(2.0)))
    except Exception:
        return 1.0


def _finite01(value: Any) -> bool:
    try:
        f = float(value)
        return math.isfinite(f) and 0.0 <= f <= 1.0
    except Exception:
        return False


def _bh_reject_families(
    family_pvals: dict[str, float], *, alpha: float, m: int
) -> set[str]:
    """BH family selection with denominator `m` (conservative if m >= #families)."""
    if not family_pvals:
        return set()
    try:
        alpha_f = float(alpha)
    except Exception:
        alpha_f = 0.05
    if not (0.0 < alpha_f <= 1.0):
        return set()

    names = list(family_pvals.keys())
    pvals = [family_pvals[n] for n in names]
    n = len(pvals)
    m_eff = max(int(m) if isinstance(m, int) else 0, n, 1)

    order = sorted(
        range(n),
        key=lambda idx: (float("inf") if not _finite01(pvals[idx]) else pvals[idx]),
    )
    max_k = 0
    for rank, idx in enumerate(order, start=1):
        p = pvals[idx]
        if not _finite01(p):
            continue
        if p <= (alpha_f * rank) / m_eff:
            max_k = rank
    if max_k <= 0:
        return set()
    cutoff = (alpha_f * max_k) / m_eff
    selected: set[str] = set()
    for idx in order:
        p = pvals[idx]
        if _finite01(p) and p <= cutoff:
            selected.add(names[idx])
    return selected


def _bonferroni_reject_families(
    family_pvals: dict[str, float], *, alpha: float, m: int
) -> set[str]:
    if not family_pvals:
        return set()
    try:
        alpha_f = float(alpha)
    except Exception:
        alpha_f = 0.05
    if not (0.0 < alpha_f <= 1.0):
        return set()
    m_eff = max(int(m) if isinstance(m, int) else 0, len(family_pvals), 1)
    cutoff = alpha_f / m_eff
    return {fam for fam, p in family_pvals.items() if _finite01(p) and p <= cutoff}


class SpectralPolicy(TypedDict, total=False):
    """Type definition for spectral guard policy configuration."""

    sigma_quantile: float
    deadband: float
    scope: str
    estimator: dict[str, Any]
    degeneracy: dict[str, Any]
    correction_enabled: bool
    family_caps: dict[str, dict[str, float]]
    ignore_preview_inflation: bool
    max_caps: int
    multiple_testing: dict[str, Any]


def _default_family_caps() -> dict[str, dict[str, float]]:
    """Default per-family spectral z-score caps."""
    return {
        "ffn": {"kappa": 2.5},
        "attn": {"kappa": 2.8},
        "embed": {"kappa": 3.0},
        "other": {"kappa": 3.0},
    }


def _normalize_family_caps(
    caps: Any, *, default: bool = True
) -> dict[str, dict[str, float]]:
    """Normalize family cap configuration into canonical mapping."""

    if not isinstance(caps, dict) or not caps:
        return _default_family_caps() if default else {}

    normalized: dict[str, dict[str, float]] = {}
    for family, values in caps.items():
        entry: dict[str, float] = {}
        if isinstance(values, dict):
            for key, val in values.items():
                if isinstance(val, int | float) and math.isfinite(float(val)):
                    entry[str(key)] = float(val)
        elif isinstance(values, int | float) and math.isfinite(float(values)):
            entry["kappa"] = float(values)
        if entry:
            normalized[str(family)] = entry

    if normalized:
        return normalized

    return _default_family_caps() if default else {}


class SpectralGuard(Guard):
    """
    Spectral guard for monitoring weight matrix spectral properties.

    Tracks singular values and spectral norms to detect numerical instabilities.
    Provides automatic spectral control when violations are detected.
    """

    name = "spectral"

    def __init__(self, **kwargs):
        """Initialize spectral guard."""
        self.config = dict(kwargs)
        self.prepared = False
        self.baseline_metrics = {}
        self.events = []
        self.current_metrics = {}
        self.violations = []

        # Default configuration
        sigma_quantile = self.config.get("sigma_quantile")
        if sigma_quantile is None:
            sigma_quantile = 0.95
        self.sigma_quantile = float(sigma_quantile)
        self.config["sigma_quantile"] = self.sigma_quantile
        self.deadband = kwargs.get("deadband", 0.10)
        self.scope = kwargs.get("scope", "all")  # 'all', 'ffn', 'attn'
        self.max_spectral_norm = kwargs.get("max_spectral_norm", None)
        if self.max_spectral_norm is not None:
            self.max_spectral_norm = float(self.max_spectral_norm)
        self.config["max_spectral_norm"] = self.max_spectral_norm
        self.correction_enabled = kwargs.get("correction_enabled", True)
        self.family_caps = _normalize_family_caps(
            kwargs.get("family_caps"), default=True
        )
        self.ignore_preview_inflation = kwargs.get("ignore_preview_inflation", True)
        self.max_caps = kwargs.get("max_caps", 5)
        self.multiple_testing = kwargs.get(
            "multiple_testing", {"method": "bh", "alpha": 0.05, "m": 4}
        )

        estimator_cfg = kwargs.get("estimator")
        if not isinstance(estimator_cfg, dict):
            estimator_cfg = {}
        try:
            est_iters = int(estimator_cfg.get("iters", 4) or 4)
        except Exception:
            est_iters = 4
        if est_iters < 1:
            est_iters = 1
        est_init = str(estimator_cfg.get("init", "ones") or "ones").strip().lower()
        if est_init not in {"ones", "e0"}:
            est_init = "ones"
        self.estimator: dict[str, Any] = {
            "type": "power_iter",
            "iters": est_iters,
            "init": est_init,
        }

        degeneracy_cfg = kwargs.get("degeneracy")
        if not isinstance(degeneracy_cfg, dict):
            degeneracy_cfg = {}
        stable_rank_cfg = (
            degeneracy_cfg.get("stable_rank")
            if isinstance(degeneracy_cfg, dict)
            else {}
        )
        norm_collapse_cfg = (
            degeneracy_cfg.get("norm_collapse")
            if isinstance(degeneracy_cfg, dict)
            else {}
        )
        self.degeneracy: dict[str, Any] = {
            "enabled": bool(degeneracy_cfg.get("enabled", True)),
            "stable_rank": {
                "warn_ratio": float((stable_rank_cfg or {}).get("warn_ratio", 0.5)),
                "fatal_ratio": float((stable_rank_cfg or {}).get("fatal_ratio", 0.25)),
            },
            "norm_collapse": {
                "warn_ratio": float((norm_collapse_cfg or {}).get("warn_ratio", 0.25)),
                "fatal_ratio": float(
                    (norm_collapse_cfg or {}).get("fatal_ratio", 0.10)
                ),
            },
        }

        # Baseline and tracking structures
        self.baseline_sigmas: dict[str, float] = {}
        self.baseline_family_stats: dict[str, dict[str, float]] = {}
        self.module_family_map: dict[str, str] = {}
        self.latest_z_scores: dict[str, float] = {}
        self.pre_edit_z_scores: dict[str, float] = {}
        self.baseline_degeneracy: dict[str, dict[str, float]] = {}

        # Run context (informational only; contract is policy-bound)
        self._run_profile: str | None = None

    def _log_event(
        self, operation: str, level: str = "INFO", message: str = "", **data
    ):
        """Log an event with timestamp."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "component": "spectral_guard",
            "operation": operation,
            "level": level,
            "message": message,
            "data": data,
        }
        self.events.append(event)

    def set_run_context(self, report: Any) -> None:
        """Capture run profile context for reporting (informational only)."""
        ctx = getattr(report, "context", {}) or {}
        profile = ""
        if isinstance(ctx, dict):
            profile = str(ctx.get("profile", "") or "").strip().lower()

        self._run_profile = profile or None

    def _serialize_policy(self) -> dict[str, Any]:
        """Snapshot current guard policy for report serialization."""
        return {
            "scope": self.scope,
            "sigma_quantile": float(self.sigma_quantile),
            "deadband": float(self.deadband),
            "max_caps": int(self.max_caps),
            "max_spectral_norm": (
                float(self.max_spectral_norm)
                if self.max_spectral_norm is not None
                else None
            ),
            "family_caps": self.family_caps,
            "multiple_testing": self.multiple_testing,
            "estimator": self.estimator,
            "degeneracy": self.degeneracy,
            "correction_enabled": bool(self.correction_enabled),
            "ignore_preview_inflation": bool(self.ignore_preview_inflation),
        }

    def prepare(
        self, model: Any, adapter: Any, calib: Any, policy: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Prepare spectral guard by capturing baseline spectral properties.

        Args:
            model: Model to prepare for
            adapter: ModelAdapter instance
            calib: Calibration data (unused for spectral analysis)
            policy: Policy configuration

        Returns:
            Preparation results
        """
        start_time = time.time()

        # Update configuration from policy
        if policy:
            sigma_value = policy.get("sigma_quantile")
            if "contraction" in policy or "kappa" in policy:
                raise ValueError(
                    "Spectral policy keys 'contraction'/'kappa' are not supported; "
                    "use 'sigma_quantile'."
                )
            if sigma_value is not None:
                self.sigma_quantile = float(sigma_value)
                policy["sigma_quantile"] = self.sigma_quantile
            self.config["sigma_quantile"] = self.sigma_quantile

            for key in [
                "sigma_quantile",
                "deadband",
                "scope",
                "max_spectral_norm",
                "correction_enabled",
                "max_caps",
            ]:
                if key in policy:
                    setattr(self, key, policy[key])
                    self.config[key] = policy[key]

            if self.max_spectral_norm is not None:
                self.max_spectral_norm = float(self.max_spectral_norm)
            self.config["max_spectral_norm"] = self.max_spectral_norm

            if "family_caps" in policy:
                self.family_caps = _normalize_family_caps(
                    policy["family_caps"], default=True
                )
                self.config["family_caps"] = self.family_caps

            if "ignore_preview_inflation" in policy:
                self.ignore_preview_inflation = bool(policy["ignore_preview_inflation"])
                self.config["ignore_preview_inflation"] = self.ignore_preview_inflation

            # Optional hydration of baseline stats from policy (e.g., baseline certificate)
            if "baseline_family_stats" in policy and isinstance(
                policy["baseline_family_stats"], dict
            ):
                self.baseline_family_stats = {
                    family: stats.copy()
                    for family, stats in policy["baseline_family_stats"].items()
                    if isinstance(stats, dict)
                }
                self.config["baseline_family_stats"] = self.baseline_family_stats
            if "multipletesting" in policy:
                raise ValidationError(
                    code="E501",
                    message="POLICY-PARAM-INVALID",
                    details={
                        "param": "multipletesting",
                        "hint": "Use spectral.multiple_testing instead.",
                    },
                )
            mt_policy = policy.get("multiple_testing")
            if isinstance(mt_policy, dict):
                self.multiple_testing = mt_policy.copy()
                policy["multiple_testing"] = self.multiple_testing
                self.config["multiple_testing"] = self.multiple_testing

            estimator_policy = policy.get("estimator")
            if isinstance(estimator_policy, dict):
                try:
                    est_iters = int(estimator_policy.get("iters", 4) or 4)
                except Exception:
                    est_iters = 4
                if est_iters < 1:
                    est_iters = 1
                est_init = (
                    str(estimator_policy.get("init", "ones") or "ones").strip().lower()
                )
                if est_init not in {"ones", "e0"}:
                    est_init = "ones"
                self.estimator = {
                    "type": "power_iter",
                    "iters": est_iters,
                    "init": est_init,
                }
                self.config["estimator"] = self.estimator

            degeneracy_policy = policy.get("degeneracy")
            if isinstance(degeneracy_policy, dict):
                stable_rank_cfg = degeneracy_policy.get("stable_rank")
                norm_collapse_cfg = degeneracy_policy.get("norm_collapse")
                self.degeneracy = {
                    "enabled": bool(degeneracy_policy.get("enabled", True)),
                    "stable_rank": {
                        "warn_ratio": float(
                            (stable_rank_cfg or {}).get("warn_ratio", 0.5)
                        ),
                        "fatal_ratio": float(
                            (stable_rank_cfg or {}).get("fatal_ratio", 0.25)
                        ),
                    },
                    "norm_collapse": {
                        "warn_ratio": float(
                            (norm_collapse_cfg or {}).get("warn_ratio", 0.25)
                        ),
                        "fatal_ratio": float(
                            (norm_collapse_cfg or {}).get("fatal_ratio", 0.10)
                        ),
                    },
                }
                self.config["degeneracy"] = self.degeneracy

        self._log_event(
            "prepare",
            message=(
                f"Preparing spectral guard with scope={self.scope}, "
                f"sigma_quantile={self.sigma_quantile}"
            ),
        )

        try:
            # Capture baseline spectral properties
            self.baseline_sigmas = self._capture_sigmas(model, phase="prepare")
            self.module_family_map = classify_model_families(
                model, scope=self.scope, existing=self.module_family_map
            )
            if not self.baseline_family_stats:
                self.baseline_family_stats = compute_family_stats(
                    self.baseline_sigmas, self.module_family_map
                )

            baseline_stats: dict[str, Any] = _summarize_sigmas(self.baseline_sigmas)

            try:
                values = np.array(list(self.baseline_sigmas.values()), dtype=float)
                if values.size:
                    self.target_sigma = float(
                        np.percentile(values, float(self.sigma_quantile) * 100.0)
                    )
                else:
                    self.target_sigma = float(self.sigma_quantile)
            except Exception:
                self.target_sigma = float(self.sigma_quantile)
            baseline_stats["target_sigma"] = self.target_sigma

            self.baseline_degeneracy = {}
            if bool((self.degeneracy or {}).get("enabled")):
                eps = 1e-12
                for name, module in model.named_modules():
                    if not self._should_check_module(name, module):
                        continue
                    weight = getattr(module, "weight", None)
                    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                        continue
                    sigma = self.baseline_sigmas.get(name)
                    if not (
                        isinstance(sigma, int | float) and math.isfinite(float(sigma))
                    ):
                        continue
                    try:
                        stable_rank = frobenius_norm_sq(weight) / max(
                            float(sigma) ** 2, eps
                        )
                        norms = row_col_norm_extrema(weight, eps=eps)
                        row_med = max(float(norms.get("row_median", 0.0)), eps)
                        col_med = max(float(norms.get("col_median", 0.0)), eps)
                        collapse = min(
                            float(norms.get("row_min", 0.0)) / row_med,
                            float(norms.get("col_min", 0.0)) / col_med,
                        )
                        self.baseline_degeneracy[name] = {
                            "stable_rank": float(stable_rank),
                            "norm_collapse": float(collapse),
                        }
                    except Exception:
                        continue
            baseline_stats["baseline_degeneracy"] = {
                name: vals.copy() for name, vals in self.baseline_degeneracy.items()
            }

            baseline_stats["family_stats"] = {
                family: stats.copy()
                for family, stats in self.baseline_family_stats.items()
            }
            baseline_stats["family_caps"] = {
                family: caps.copy() for family, caps in self.family_caps.items()
            }
            baseline_stats["module_sigmas"] = self.baseline_sigmas.copy()
            baseline_stats["measurement_contract"] = {
                "estimator": self.estimator,
                "degeneracy": self.degeneracy,
            }

            self.baseline_metrics = baseline_stats

            self.prepared = True
            preparation_time = time.time() - start_time

            self._log_event(
                "prepare_success",
                message=f"Prepared spectral guard with {len(self.baseline_metrics)} baseline metrics",
                baseline_metrics_count=len(self.baseline_metrics),
                target_sigma=self.target_sigma,
                preparation_time=preparation_time,
            )

            return {
                "ready": True,
                "baseline_metrics": self.baseline_metrics,
                "target_sigma": self.target_sigma,
                "scope": self.scope,
                "preparation_time": preparation_time,
            }

        except Exception as e:
            self.prepared = False
            self._log_event(
                "prepare_failed",
                level="ERROR",
                message=f"Failed to prepare spectral guard: {str(e)}",
                error=str(e),
            )

            return {
                "ready": False,
                "error": str(e),
                "preparation_time": time.time() - start_time,
            }

    def before_edit(self, model: Any) -> None:
        """Execute before edit (capture pre-edit state)."""
        if not self.prepared:
            self._log_event(
                "before_edit_skipped",
                level="WARN",
                message="Spectral guard not prepared, skipping pre-edit capture",
            )
            return

        # Capture pre-edit spectral state for comparison
        self.pre_edit_metrics = self._capture_sigmas(model, phase="before_edit")
        self.pre_edit_z_scores = compute_z_scores(
            self.pre_edit_metrics,
            self.baseline_family_stats,
            self.module_family_map,
            self.baseline_sigmas,
            deadband=self.deadband,
        )
        self._log_event("before_edit", message="Captured pre-edit spectral state")

    def after_edit(self, model: Any) -> None:
        """Execute after edit (detect violations and apply control)."""
        if not self.prepared:
            self._log_event(
                "after_edit_skipped",
                level="WARN",
                message="Spectral guard not prepared, skipping post-edit analysis",
            )
            return

        try:
            # Capture current spectral state
            self.current_metrics = self._capture_sigmas(model, phase="after_edit")

            # Detect violations
            violations = self._detect_spectral_violations(
                model, self.current_metrics, phase="after_edit"
            )
            self.violations = violations

            # Apply spectral control if violations detected and correction enabled
            if violations and self.correction_enabled:
                control_result = apply_spectral_control(
                    model,
                    policy={
                        "sigma_quantile": self.sigma_quantile,
                        "scope": self.scope,
                        "baseline_sigmas": self.baseline_sigmas,
                        "target_sigma": self.target_sigma,
                    },
                )

                self._log_event(
                    "spectral_control_applied",
                    message=f"Applied spectral control, violations: {len(violations)}",
                    violations_count=len(violations),
                    control_result=control_result,
                )

            self._log_event(
                "after_edit",
                message=f"Post-edit analysis complete, {len(violations)} violations detected",
            )

        except Exception as e:
            self._log_event(
                "after_edit_failed",
                level="ERROR",
                message=f"Post-edit spectral analysis failed: {str(e)}",
                error=str(e),
            )

    def _capture_sigmas(self, model: Any, *, phase: str) -> dict[str, float]:
        """Capture σ̂max for each in-scope module under the measurement contract."""
        _ = phase  # reserved for future observability hooks
        sigmas: dict[str, float] = {}
        try:
            iters = int((self.estimator or {}).get("iters", 4) or 4)
        except Exception:
            iters = 4
        if iters < 1:
            iters = 1
        init = str((self.estimator or {}).get("init", "ones") or "ones").strip().lower()
        if init not in {"ones", "e0"}:
            init = "ones"

        for name, module in model.named_modules():
            if not _should_process_module(name, module, self.scope):
                continue
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                continue
            if weight.dtype in {torch.int8, torch.uint8}:
                # Skip quantized weights; treat as neutral for baseline-relative stats.
                sigmas[name] = 1.0
                continue
            try:
                sigmas[name] = float(
                    power_iter_sigma_max(weight, iters=iters, init=init)
                )
            except Exception:
                sigmas[name] = 1.0

        return sigmas

    def _detect_spectral_violations(
        self, model: Any, metrics: dict[str, float], phase: str = "finalize"
    ) -> list[dict[str, Any]]:
        """Detect spectral property violations using per-family z-score caps."""
        violations: list[dict[str, Any]] = []
        latest_z: dict[str, float] = {}

        for name, module in model.named_modules():
            if not self._should_check_module(name, module):
                continue

            try:
                if hasattr(module, "weight") and module.weight.ndim == 2:
                    sigma_max = metrics.get(name)
                    if sigma_max is None:
                        sigma_max = compute_sigma_max(module.weight)

                    baseline_sigma = self.baseline_sigmas.get(name, self.target_sigma)
                    family = self.module_family_map.get(name)
                    if family is None:
                        family = classify_module_family(name, module)
                        self.module_family_map[name] = family

                    family_stats = self.baseline_family_stats.get(family, {})
                    cap_config = self.family_caps.get(family, {})
                    kappa_cap = float(cap_config.get("kappa", self.sigma_quantile))

                    z_score = compute_z_score_for_value(
                        sigma_max,
                        family_stats,
                        fallback_value=baseline_sigma,
                        deadband=self.deadband,
                    )
                    latest_z[name] = z_score

                    # Skip preview inflation if configured and not in final phase
                    if self.ignore_preview_inflation and phase == "after_edit":
                        continue

                    if abs(z_score) > kappa_cap:
                        violations.append(
                            {
                                "type": "family_z_cap",
                                "severity": "budgeted",
                                "module": name,
                                "family": family,
                                "z_score": float(z_score),
                                "kappa": kappa_cap,
                                "sigma": float(sigma_max),
                                "baseline_sigma": float(baseline_sigma),
                                "message": (
                                    f"Family '{family}' z-score {z_score:.2f}"
                                    f" exceeds cap {kappa_cap:.2f}"
                                ),
                            }
                        )

                    if (
                        self.max_spectral_norm is not None
                        and sigma_max > self.max_spectral_norm
                    ):
                        threshold = float(self.max_spectral_norm)
                        violations.append(
                            {
                                "type": "max_spectral_norm",
                                "severity": "fatal",
                                "module": name,
                                "family": family,
                                "current_sigma": float(sigma_max),
                                "threshold": threshold,
                                "message": f"Spectral norm {sigma_max:.3f} exceeds maximum {threshold}",
                            }
                        )

                    if bool((self.degeneracy or {}).get("enabled")):
                        base = self.baseline_degeneracy.get(name) or {}
                        base_sr = base.get("stable_rank")
                        base_nc = base.get("norm_collapse")
                        eps = 1e-12
                        try:
                            sr_cfg = (
                                (self.degeneracy.get("stable_rank") or {})
                                if isinstance(self.degeneracy, dict)
                                else {}
                            )
                            nc_cfg = (
                                (self.degeneracy.get("norm_collapse") or {})
                                if isinstance(self.degeneracy, dict)
                                else {}
                            )
                            sr_warn = float(sr_cfg.get("warn_ratio", 0.5))
                            sr_fatal = float(sr_cfg.get("fatal_ratio", 0.25))
                            nc_warn = float(nc_cfg.get("warn_ratio", 0.25))
                            nc_fatal = float(nc_cfg.get("fatal_ratio", 0.10))
                        except Exception:
                            sr_warn, sr_fatal, nc_warn, nc_fatal = 0.5, 0.25, 0.25, 0.10

                        if (
                            isinstance(base_sr, int | float)
                            and math.isfinite(float(base_sr))
                            and float(base_sr) > 0
                        ):
                            try:
                                sr_cur = frobenius_norm_sq(module.weight) / max(
                                    float(sigma_max) ** 2, eps
                                )
                                sr_ratio = float(sr_cur) / max(float(base_sr), eps)
                                if math.isfinite(sr_ratio) and sr_ratio < sr_warn:
                                    violations.append(
                                        {
                                            "type": "degeneracy_stable_rank_drop",
                                            "severity": "fatal"
                                            if sr_ratio < sr_fatal
                                            else "budgeted",
                                            "module": name,
                                            "family": family,
                                            "stable_rank_base": float(base_sr),
                                            "stable_rank_cur": float(sr_cur),
                                            "ratio": float(sr_ratio),
                                            "warn_ratio": float(sr_warn),
                                            "fatal_ratio": float(sr_fatal),
                                            "message": (
                                                f"Stable-rank ratio {sr_ratio:.3f} below "
                                                f"{sr_warn:.3f} (base={float(base_sr):.3f}, cur={float(sr_cur):.3f})"
                                            ),
                                        }
                                    )
                            except Exception:
                                pass

                        if (
                            isinstance(base_nc, int | float)
                            and math.isfinite(float(base_nc))
                            and float(base_nc) > 0
                        ):
                            try:
                                norms = row_col_norm_extrema(module.weight, eps=eps)
                                row_med = max(float(norms.get("row_median", 0.0)), eps)
                                col_med = max(float(norms.get("col_median", 0.0)), eps)
                                nc_cur = min(
                                    float(norms.get("row_min", 0.0)) / row_med,
                                    float(norms.get("col_min", 0.0)) / col_med,
                                )
                                nc_ratio = float(nc_cur) / max(float(base_nc), eps)
                                if math.isfinite(nc_ratio) and nc_ratio < nc_warn:
                                    violations.append(
                                        {
                                            "type": "degeneracy_norm_collapse",
                                            "severity": "fatal"
                                            if nc_ratio < nc_fatal
                                            else "budgeted",
                                            "module": name,
                                            "family": family,
                                            "norm_collapse_base": float(base_nc),
                                            "norm_collapse_cur": float(nc_cur),
                                            "ratio": float(nc_ratio),
                                            "warn_ratio": float(nc_warn),
                                            "fatal_ratio": float(nc_fatal),
                                            "message": (
                                                f"Norm-collapse ratio {nc_ratio:.3f} below "
                                                f"{nc_warn:.3f} (base={float(base_nc):.3e}, cur={float(nc_cur):.3e})"
                                            ),
                                        }
                                    )
                            except Exception:
                                pass

            except Exception as e:
                self._log_event(
                    "violation_check_error",
                    level="WARN",
                    message=f"Failed to check module {name}: {str(e)}",
                    module=name,
                    error=str(e),
                )

        self.latest_z_scores = latest_z
        return violations

    def _should_check_module(self, name: str, module: Any) -> bool:
        """Determine if a module should be checked based on scope."""
        if not hasattr(module, "weight") or module.weight.ndim != 2:
            return False

        if self.scope == "all":
            return True
        elif self.scope == "attn":
            return any(
                keyword in name.lower()
                for keyword in ["attn", "attention", "self_attn"]
            )
        elif self.scope == "ffn":
            return any(
                keyword in name.lower()
                for keyword in ["mlp", "ffn", "feed_forward", "fc"]
            )

        return True

    def _compute_family_observability(
        self,
    ) -> tuple[dict[str, dict[str, float]], dict[str, list[dict[str, Any]]]]:
        """Generate per-family quantiles and top-|z| listings from latest z-scores."""
        family_scores: dict[str, list[float]] = defaultdict(list)
        family_modules: dict[str, list[tuple[float, str]]] = defaultdict(list)

        for module_name, z_value in (self.latest_z_scores or {}).items():
            family = self.module_family_map.get(module_name)
            if family is None:
                continue
            try:
                z_abs = abs(float(z_value))
            except (TypeError, ValueError):
                continue
            family_scores.setdefault(family, []).append(z_abs)
            family_modules.setdefault(family, []).append((z_abs, module_name))

        def _quantile(sorted_values: list[float], quantile: float) -> float:
            if not sorted_values:
                return 0.0
            if len(sorted_values) == 1:
                return sorted_values[0]
            position = (len(sorted_values) - 1) * quantile
            lower = math.floor(position)
            upper = math.ceil(position)
            if lower == upper:
                return sorted_values[int(position)]
            fraction = position - lower
            return (
                sorted_values[lower]
                + (sorted_values[upper] - sorted_values[lower]) * fraction
            )

        family_quantiles: dict[str, dict[str, float]] = {}
        for family, scores in family_scores.items():
            sorted_scores = sorted(scores)
            family_quantiles[family] = {
                "q95": _quantile(sorted_scores, 0.95),
                "q99": _quantile(sorted_scores, 0.99),
                "max": sorted_scores[-1] if sorted_scores else 0.0,
                "count": len(sorted_scores),
            }

        top_z_scores: dict[str, list[dict[str, Any]]] = {}
        for family, module_entries in family_modules.items():
            module_entries.sort(key=lambda item: item[0], reverse=True)
            top_entries: list[dict[str, Any]] = []
            for z_abs, module_name in module_entries[:3]:
                top_entries.append(
                    {"module": module_name, "z": float(z_abs), "family": family}
                )
            top_z_scores[family] = top_entries

        return family_quantiles, top_z_scores

    def _select_budgeted_violations(
        self, budgeted_violations: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Apply BH/Bonferroni selection at the family level.

        Returns:
            (selected_violations, selection_metrics)
        """
        mt = self.multiple_testing if isinstance(self.multiple_testing, dict) else {}
        method = str(mt.get("method", "bh")).lower()
        try:
            alpha = float(mt.get("alpha", 0.05) or 0.05)
        except Exception:
            alpha = 0.05
        m_raw = mt.get("m")
        m = None
        try:
            if m_raw is not None:
                m = int(m_raw)
        except Exception:
            m = None

        # Fill in missing family assignments deterministically.
        for violation in budgeted_violations:
            if violation.get("family"):
                continue
            module = violation.get("module")
            if isinstance(module, str):
                family = self.module_family_map.get(module)
                if isinstance(family, str) and family:
                    violation["family"] = family
                    continue
            violation["family"] = "other"

        # Family p-values derived from the most significant (min p) module in each family.
        family_pvals: dict[str, float] = {}
        family_max_abs_z: dict[str, float] = {}
        family_counts: dict[str, int] = {}
        for violation in budgeted_violations:
            fam = violation.get("family")
            if fam is None:
                continue
            family = str(fam)
            z_val = violation.get("z_score")
            try:
                zf = float(z_val)
            except Exception:
                continue
            if not math.isfinite(zf):
                continue
            p = _z_to_two_sided_pvalue(zf)
            family_counts[family] = family_counts.get(family, 0) + 1
            cur = family_pvals.get(family)
            if cur is None or p < cur:
                family_pvals[family] = p
                family_max_abs_z[family] = abs(zf)

        families_tested = sorted(family_pvals.keys())
        m_eff = m if isinstance(m, int) and m > 0 else len(families_tested)
        m_eff = max(m_eff, len(families_tested), 1)
        if isinstance(self.multiple_testing, dict):
            self.multiple_testing.setdefault("m", m_eff)

        if method in {"bh", "benjamini-hochberg", "benjamini_hochberg"}:
            selected_families = _bh_reject_families(family_pvals, alpha=alpha, m=m_eff)
            applied_method = "bh"
        elif method in {"bonferroni", "bonf"}:
            selected_families = _bonferroni_reject_families(
                family_pvals, alpha=alpha, m=m_eff
            )
            applied_method = "bonferroni"
        else:
            selected_families = _bonferroni_reject_families(
                family_pvals, alpha=alpha, m=m_eff
            )
            applied_method = "bonferroni"

        selected: list[dict[str, Any]] = []
        default_selected_without_pvalue = 0
        for violation in budgeted_violations:
            fam = violation.get("family")
            family = str(fam) if fam is not None else ""
            z_val = violation.get("z_score")
            p_val: float | None = None
            try:
                zf = float(z_val)
            except Exception:
                zf = None
            if zf is not None and math.isfinite(zf):
                p_val = _z_to_two_sided_pvalue(zf)
                is_selected = family in selected_families
            else:
                # If we cannot compute a p-value, fail closed: keep the violation.
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
            "family_pvalues": {k: float(family_pvals[k]) for k in families_tested},
            "family_max_abs_z": {
                k: float(family_max_abs_z[k]) for k in families_tested
            },
            "family_violation_counts": dict(family_counts),
            "default_selected_without_pvalue": int(default_selected_without_pvalue),
        }
        return selected, selection_metrics

    def validate(
        self, model: Any, adapter: Any, context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Validate model spectral properties.

        Args:
            model: Model to validate
            adapter: ModelAdapter instance
            context: Validation context

        Returns:
            Dictionary with validation results
        """
        try:
            if not self.prepared:
                # Auto-prepare if needed
                self.prepare(model, adapter, None, {})

            # Capture current spectral state
            current_metrics = self._capture_sigmas(model, phase="validate")

            # Detect violations (final validation phase)
            violations = self._detect_spectral_violations(
                model, current_metrics, phase="validate"
            )

            # Determine if passed under budget/fatal rules
            fatal_violations = [
                violation
                for violation in violations
                if (violation.get("severity") == "fatal")
                or (violation.get("type") == "max_spectral_norm")
            ]
            budgeted_violations = [
                violation
                for violation in violations
                if violation not in fatal_violations
            ]

            selected_budgeted, mt_selection = self._select_budgeted_violations(
                budgeted_violations
            )
            selected_violations = [*fatal_violations, *selected_budgeted]
            candidate_budgeted = len(budgeted_violations)

            caps_applied = len(selected_budgeted)
            caps_exceeded = caps_applied > int(self.max_caps)
            passed = not fatal_violations and not caps_exceeded
            if fatal_violations or caps_exceeded:
                action = "abort"
            elif caps_applied > 0:
                action = "warn"
            else:
                action = "continue"

            # Compute overall metrics
            family_summary = summarize_family_z_scores(
                self.latest_z_scores, self.module_family_map, self.family_caps
            )
            metrics = {
                "modules_checked": len(current_metrics),
                "violations_found": len(selected_violations),
                "budgeted_violations": caps_applied,
                "candidate_budgeted_violations": candidate_budgeted,
                "fatal_violations": len(fatal_violations),
                "max_spectral_norm": max(current_metrics.values())
                if current_metrics
                else 0.0,
                "mean_spectral_norm": np.mean(list(current_metrics.values()))
                if current_metrics
                else 0.0,
                "stability_score": 1.0
                - min(len(violations) / max(len(current_metrics), 1), 1.0),
                "family_z_summary": family_summary,
                "family_caps": self.family_caps,
                "sigma_quantile": float(self.sigma_quantile),
                "deadband": float(self.deadband),
                "max_caps": int(self.max_caps),
                "caps_applied": caps_applied,
                "caps_exceeded": caps_exceeded,
                "multiple_testing": self.multiple_testing,
                "multiple_testing_selection": mt_selection,
                "measurement_contract": {
                    "estimator": self.estimator,
                    "degeneracy": self.degeneracy,
                },
            }

            family_quantiles, top_z_scores = self._compute_family_observability()
            if family_quantiles:
                metrics["family_z_quantiles"] = family_quantiles
            if top_z_scores:
                metrics["top_z_scores"] = top_z_scores

            if passed:
                message = (
                    "Spectral validation passed with "
                    f"{len(selected_violations)} violations "
                    f"(caps_applied={caps_applied}, max_caps={self.max_caps})"
                )
            else:
                reason = (
                    "fatal spectral violation detected"
                    if fatal_violations
                    else "cap budget exceeded"
                )
                message = (
                    f"Spectral validation failed: {reason} "
                    f"(caps_applied={caps_applied}, max_caps={self.max_caps})"
                )

            # Runtime contracts (lightweight)
            mt = self.multiple_testing or {}
            try:
                alpha = float(mt.get("alpha", 0.05)) if isinstance(mt, dict) else 0.05
            except Exception:
                alpha = 0.05
            guard_assert(self.deadband >= 0.0, "spectral.deadband must be >= 0")
            guard_assert(
                0.0 < alpha <= 1.0, "spectral.multiple_testing.alpha out of range"
            )
            guard_assert(self.max_caps >= 0, "spectral.max_caps must be >= 0")

            return {
                "passed": passed,
                "action": action,
                "metrics": metrics,
                "violations": selected_violations,
                "message": message,
                "policy": self._serialize_policy(),
                "final_z_scores": self.latest_z_scores.copy(),
                "module_family_map": dict(self.module_family_map),
            }

        except Exception as e:
            return {
                "passed": False,
                "action": "warn",
                "error": str(e),
                "metrics": {},
                "message": f"Spectral validation failed: {e}",
            }

    def finalize(self, model: Any) -> dict[str, Any]:
        """
        Finalize spectral guard and return comprehensive results.

        Args:
            model: The final model state

        Returns:
            Dictionary with spectral guard results
        """
        if not self.prepared:
            return {
                "passed": False,
                "metrics": {},
                "warnings": ["Spectral guard not properly prepared"],
                "errors": ["Preparation failed or not called"],
                "events": self.events,
            }

        # Final spectral analysis
        final_metrics = self._capture_sigmas(model, phase="finalize")
        final_violations = self._detect_spectral_violations(
            model, final_metrics, phase="finalize"
        )
        final_z_summary = summarize_family_z_scores(
            self.latest_z_scores, self.module_family_map, self.family_caps
        )
        final_family_stats = compute_family_stats(final_metrics, self.module_family_map)

        family_quantiles, top_z_scores = self._compute_family_observability()

        # Determine overall status based on budgeted vs fatal violations
        fatal_violations = [
            violation
            for violation in final_violations
            if (violation.get("severity") == "fatal")
            or (violation.get("type") == "max_spectral_norm")
        ]
        budgeted_violations = [
            violation
            for violation in final_violations
            if violation not in fatal_violations
        ]

        selected_budgeted, mt_selection = self._select_budgeted_violations(
            budgeted_violations
        )
        selected_final_violations = [*fatal_violations, *selected_budgeted]
        candidate_budgeted = len(budgeted_violations)

        caps_applied = len(selected_budgeted)
        caps_exceeded = caps_applied > int(self.max_caps)
        passed = not fatal_violations and not caps_exceeded

        # Compute comprehensive metrics
        metrics = {
            "modules_analyzed": len(final_metrics),
            "violations_detected": len(selected_final_violations),
            "budgeted_violations": caps_applied,
            "candidate_violations_detected": len(final_violations),
            "candidate_budgeted_violations": candidate_budgeted,
            "fatal_violations": len(fatal_violations),
            "baseline_modules": len(self.baseline_metrics),
            "scope": self.scope,
            "max_spectral_norm_final": max(final_metrics.values())
            if final_metrics
            else 0.0,
            "mean_spectral_norm_final": np.mean(list(final_metrics.values()))
            if final_metrics
            else 0.0,
            "spectral_stability_score": 1.0
            - min(len(final_violations) / max(len(final_metrics), 1), 1.0),
            "target_sigma": self.target_sigma,
            "correction_applied": len(selected_final_violations) > 0
            and self.correction_enabled,
            "family_caps": self.family_caps,
            "family_z_summary": final_z_summary,
            "family_stats": final_family_stats,
            "sigma_quantile": float(self.sigma_quantile),
            "deadband": float(self.deadband),
            "max_caps": int(self.max_caps),
            "caps_applied": caps_applied,
            "caps_exceeded": caps_exceeded,
            "multiple_testing": self.multiple_testing,
            "multiple_testing_selection": mt_selection,
            "family_z_quantiles": family_quantiles,
            "top_z_scores": top_z_scores,
            "measurement_contract": {
                "estimator": self.estimator,
                "degeneracy": self.degeneracy,
            },
        }

        # Categorize violations
        warnings = []
        errors = []

        for violation in selected_final_violations:
            if (violation.get("severity") == "fatal") or (
                violation.get("type") == "max_spectral_norm"
            ):
                errors.append(violation["message"])
            else:
                warnings.append(violation["message"])

        result = {
            "passed": passed,
            "metrics": metrics,
            "warnings": warnings,
            "errors": errors,
            "violations": selected_final_violations,
            "events": self.events,
            "baseline_metrics": self.baseline_metrics,
            "final_metrics": final_metrics,
            "final_z_scores": self.latest_z_scores,
            "module_family_map": dict(self.module_family_map),
            "policy": self._serialize_policy(),
        }

        # Env-gated tiny evidence dump for auditors
        try:
            payload = {
                "spectral": {
                    "sigma_quantile": float(self.sigma_quantile),
                    "deadband": float(self.deadband),
                    "max_caps": int(self.max_caps),
                    "multiple_testing": self.multiple_testing.get("method")
                    if isinstance(self.multiple_testing, dict)
                    else None,
                    "evaluated": True,
                }
            }
            maybe_dump_guard_evidence(".", payload)
        except Exception:
            pass

        return result


def compute_sigma_max(
    weight_matrix: Any, *, iters: int = 4, init: str = "ones"
) -> float:
    """
    Compute maximum singular value of a weight matrix.

    Args:
        weight_matrix: Weight matrix to analyze

    Returns:
        Maximum singular value
    """
    try:
        iters_i = int(iters)
    except Exception:
        iters_i = 4
    if iters_i < 1:
        iters_i = 1
    init_s = str(init or "ones").strip().lower()
    if init_s not in {"ones", "e0"}:
        init_s = "ones"

    if not isinstance(weight_matrix, torch.Tensor):
        return 1.0
    if weight_matrix.dtype in {torch.int8, torch.uint8}:
        return 1.0
    if weight_matrix.numel() == 0 or weight_matrix.ndim != 2:
        return 0.0

    try:
        return float(power_iter_sigma_max(weight_matrix, iters=iters_i, init=init_s))
    except Exception:
        return 1.0


def auto_sigma_target(model: Any, percentile: float = 0.95) -> float:
    """
    Automatically determine sigma target for a model.

    Args:
        model: Model to analyze
        percentile: Scale factor (target percentile of spectral norms)

    Returns:
        Target sigma value
    """
    try:
        # Collect all spectral norms
        spectral_norms = []

        for _name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight.ndim == 2:
                sigma = compute_sigma_max(module.weight)
                if sigma > 0:
                    spectral_norms.append(sigma)

        if spectral_norms:
            # Use kappa-percentile as target
            target = np.percentile(spectral_norms, percentile * 100)
            return float(target)
        else:
            return percentile  # Fallback to requested sigma quantile

    except Exception:
        return percentile  # Default fallback


def apply_weight_rescale(
    model: Any, scale_factor: float = 1.0, scope: str = "all"
) -> dict[str, Any]:
    """
    Apply weight rescaling to model parameters.

    Args:
        model: Model to rescale
        scale_factor: Scaling factor to apply
        scope: Which modules to rescale ('all', 'attn', 'ffn')

    Returns:
        Rescaling results
    """
    try:
        rescaled_modules = []
        failed_modules = []

        for name, module in model.named_modules():
            if not _should_process_module(name, module, scope):
                continue

            try:
                if hasattr(module, "weight") and module.weight.ndim == 2:
                    # Skip quantized weights
                    if hasattr(module.weight, "dtype") and module.weight.dtype in [
                        torch.int8,
                    ]:
                        continue

                    # Apply rescaling
                    with torch.no_grad():
                        module.weight.mul_(scale_factor)
                        if hasattr(module, "bias") and module.bias is not None:
                            module.bias.mul_(scale_factor)

                    rescaled_modules.append(name)

            except Exception as e:
                failed_modules.append((name, str(e)))

        return {
            "applied": len(rescaled_modules) > 0,
            "scale_factor": scale_factor,
            "rescaled_modules": rescaled_modules,
            "failed_modules": failed_modules,
            "message": f"Rescaled {len(rescaled_modules)} modules with factor {scale_factor}",
        }

    except Exception as e:
        return {
            "applied": False,
            "error": str(e),
            "message": f"Weight rescaling failed: {e}",
        }


def apply_relative_spectral_cap(
    model: Any,
    cap_ratio: float = 2.0,
    scope: str = "all",
    baseline_sigmas: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Apply relative spectral capping to model weights.

    Args:
        model: Model to cap
        cap_ratio: Maximum allowed ratio relative to baseline
        scope: Which modules to cap ('all', 'attn', 'ffn')
        baseline_sigmas: Mapping of module name to pre-edit sigma values

    Returns:
        Capping results
    """
    try:
        if baseline_sigmas is None:
            baseline_sigmas = capture_baseline_sigmas(model, scope=scope)

        capped_modules = []
        failed_modules = []

        for name, module in model.named_modules():
            if not _should_process_module(name, module, scope):
                continue

            try:
                if hasattr(module, "weight") and module.weight.ndim == 2:
                    # Skip quantized weights
                    if hasattr(module.weight, "dtype") and module.weight.dtype in [
                        torch.int8,
                    ]:
                        continue

                    current_sigma = compute_sigma_max(module.weight)
                    baseline_sigma = baseline_sigmas.get(name, current_sigma)
                    max_allowed = baseline_sigma * cap_ratio

                    if current_sigma > max_allowed:
                        # Apply spectral capping using SVD
                        scale_factor = max_allowed / current_sigma

                        with torch.no_grad():
                            module.weight.mul_(scale_factor)

                        capped_modules.append(
                            {
                                "module": name,
                                "original_sigma": current_sigma,
                                "capped_sigma": max_allowed,
                                "scale_factor": scale_factor,
                            }
                        )

            except Exception as e:
                failed_modules.append((name, str(e)))

        return {
            "applied": len(capped_modules) > 0,
            "cap_ratio": cap_ratio,
            "capped_modules": capped_modules,
            "failed_modules": failed_modules,
            "message": f"Applied spectral capping to {len(capped_modules)} modules",
        }

    except Exception as e:
        return {
            "applied": False,
            "error": str(e),
            "message": f"Spectral capping failed: {e}",
        }


def apply_spectral_control(model: Any, policy: dict[str, Any]) -> dict[str, Any]:
    """
    Apply spectral control based on policy.

    Args:
        model: Model to control
        policy: Spectral control policy

    Returns:
        Control results
    """
    try:
        results: dict[str, Any] = {
            "rescaling_applied": False,
            "capping_applied": False,
            "modules_processed": 0,
            "corrections": [],
        }

        scope = policy.get("scope", "all")
        baseline_sigmas = policy.get("baseline_sigmas")

        # Apply relative spectral capping if needed
        cap_ratio = policy.get("cap_ratio", 2.0)
        cap_result = apply_relative_spectral_cap(
            model,
            cap_ratio=cap_ratio,
            scope=scope,
            baseline_sigmas=baseline_sigmas,
        )

        if cap_result["applied"]:
            results["capping_applied"] = True
            results["corrections"].extend(cap_result["capped_modules"])

        # Apply rescaling if target sigma is specified
        if "rescale_factor" in policy:
            rescale_result = apply_weight_rescale(
                model, scale_factor=policy["rescale_factor"], scope=scope
            )

            if rescale_result["applied"]:
                results["rescaling_applied"] = True
                results["modules_processed"] += len(rescale_result["rescaled_modules"])

        results["applied"] = results["rescaling_applied"] or results["capping_applied"]
        results["policy"] = policy
        results["message"] = (
            f"Spectral control applied: capping={results['capping_applied']}, rescaling={results['rescaling_applied']}"
        )

        return results

    except Exception as e:
        return {
            "applied": False,
            "error": str(e),
            "policy": policy,
            "message": f"Spectral control failed: {e}",
        }


def _summarize_sigmas(sigmas: dict[str, float]) -> dict[str, float]:
    """Compute summary statistics for a sigma dictionary."""
    if not sigmas:
        return {
            "max_spectral_norm": 0.0,
            "mean_spectral_norm": 0.0,
            "min_spectral_norm": 0.0,
        }

    values = np.array(list(sigmas.values()), dtype=float)
    return {
        "max_spectral_norm": float(values.max()),
        "mean_spectral_norm": float(values.mean()),
        "min_spectral_norm": float(values.min()),
    }


def compute_z_score_for_value(
    sigma: float,
    family_stats: dict[str, float],
    fallback_value: float,
    deadband: float,
) -> float:
    """Compute per-family z-score for a spectral norm with sensible fallbacks."""
    mean = float(family_stats.get("mean", 0.0) or 0.0)
    std = float(family_stats.get("std", 0.0) or 0.0)

    if std > 0:
        return float((sigma - mean) / std)

    # Fallback: scale relative change by deadband width
    denom = fallback_value if fallback_value > 0 else 1.0
    rel_change = (sigma / denom) - 1.0

    if abs(rel_change) <= deadband:
        return 0.0

    scale = deadband if deadband > 0 else 1.0
    return float(rel_change / scale)


def compute_z_scores(
    metrics: dict[str, float],
    baseline_family_stats: dict[str, dict[str, float]],
    module_family_map: dict[str, str],
    baseline_sigmas: dict[str, float],
    deadband: float,
) -> dict[str, float]:
    """Compute z-scores for all modules given baseline family stats."""
    z_scores: dict[str, float] = {}
    for name, sigma in metrics.items():
        family = module_family_map.get(name, "other")
        family_stats = baseline_family_stats.get(family, {})
        fallback_value = baseline_sigmas.get(name, family_stats.get("mean", sigma))
        z_scores[name] = compute_z_score_for_value(
            float(sigma),
            family_stats,
            float(fallback_value),
            deadband=deadband,
        )
    return z_scores


def summarize_family_z_scores(
    z_scores: dict[str, float],
    module_family_map: dict[str, str],
    family_caps: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Summarize z-scores per family, including violation counts."""
    family_values: dict[str, list[float]] = defaultdict(list)
    for name, z in z_scores.items():
        family = module_family_map.get(name, "other")
        family_values[family].append(float(z))

    summary: dict[str, dict[str, float]] = {}
    for family, values in family_values.items():
        if not values:
            continue
        arr = np.array(values, dtype=float)
        cap = family_caps.get(family, {}).get("kappa")
        violations = 0
        if cap is not None:
            violations = int(np.sum(arr > float(cap)))
        summary[family] = {
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "count": len(values),
            "violations": violations,
        }
        if cap is not None:
            summary[family]["kappa"] = float(cap)
    return summary


def compute_family_stats(
    sigmas: dict[str, float], family_map: dict[str, str]
) -> dict[str, dict[str, float]]:
    """Compute per-family statistics (mean/std/min/max/count)."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for name, sigma in sigmas.items():
        family = family_map.get(name, "other")
        buckets[family].append(float(sigma))

    stats: dict[str, dict[str, float]] = {}
    for family, values in buckets.items():
        if not values:
            continue
        arr = np.array(values, dtype=float)
        stats[family] = {
            "count": len(values),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return stats


def classify_model_families(
    model: Any, scope: str = "all", existing: dict[str, str] | None = None
) -> dict[str, str]:
    """Build or update a module→family map for the provided model."""
    family_map = dict(existing) if existing else {}
    for name, module in model.named_modules():
        if _should_process_module(name, module, scope):
            family_map[name] = classify_module_family(name, module)
    return family_map


def capture_baseline_sigmas(model: Any, scope: str = "all") -> dict[str, float]:
    """
    Capture baseline singular values for model layers.

    Args:
        model: Model to analyze
        scope: Which modules to analyze ('all', 'attn', 'ffn')

    Returns:
        Dictionary of layer name to max singular value
    """
    try:
        baseline_sigmas = {}

        for name, module in model.named_modules():
            if _should_process_module(name, module, scope):
                if hasattr(module, "weight") and module.weight.ndim == 2:
                    sigma = compute_sigma_max(module.weight)
                    baseline_sigmas[name] = sigma

        return baseline_sigmas

    except Exception:
        return {}


def scan_model_gains(model: Any, scope: str = "all") -> dict[str, Any]:
    """
    Scan model for gain values and spectral statistics.

    Args:
        model: Model to scan
        scope: Which modules to scan ('all', 'attn', 'ffn')

    Returns:
        Gain analysis results
    """
    try:
        results: dict[str, Any] = {
            "total_layers": 0,
            "scanned_modules": 0,
            "spectral_norms": [],
            "weight_statistics": {},
        }

        for name, module in model.named_modules():
            results["total_layers"] += 1

            if _should_process_module(name, module, scope):
                if hasattr(module, "weight") and module.weight.ndim == 2:
                    results["scanned_modules"] += 1

                    # Compute spectral norm
                    sigma_max = compute_sigma_max(module.weight)
                    results["spectral_norms"].append(sigma_max)

                    # Basic weight statistics
                    try:
                        weight_stats = {
                            "mean": module.weight.mean().item(),
                            "std": module.weight.std().item(),
                            "min": module.weight.min().item(),
                            "max": module.weight.max().item(),
                        }
                        results["weight_statistics"][name] = weight_stats
                    except Exception:
                        pass

        # Compute summary statistics
        if results["spectral_norms"]:
            results["mean_spectral_norm"] = np.mean(results["spectral_norms"])
            results["max_spectral_norm"] = np.max(results["spectral_norms"])
            results["min_spectral_norm"] = np.min(results["spectral_norms"])

        results["message"] = (
            f"Scanned {results['scanned_modules']} modules out of {results['total_layers']} total layers"
        )

        return results

    except Exception as e:
        return {
            "total_layers": sum(1 for _ in model.named_modules()),
            "scanned_modules": 0,
            "error": str(e),
            "message": f"Model scanning failed: {e}",
        }


def _should_process_module(name: str, module: Any, scope: str) -> bool:
    """Helper function to determine if a module should be processed based on scope."""
    if not hasattr(module, "weight") or module.weight.ndim != 2:
        return False

    if scope == "all":
        return True
    elif scope == "attn":
        return any(
            keyword in name.lower()
            for keyword in ["attn", "attention", "self_attn", "c_attn", "c_proj"]
        )
    elif scope == "ffn":
        return any(
            keyword in name.lower()
            for keyword in ["mlp", "ffn", "feed_forward", "fc", "c_fc"]
        )
    elif scope == "ffn+proj":
        lname = name.lower()
        return any(
            keyword in lname
            for keyword in [
                "mlp",
                "ffn",
                "feed_forward",
                "fc",
                "c_fc",
                "c_proj",
                "projection",
            ]
        )

    return True


def classify_module_family(name: str, module: Any) -> str:
    """Classify module into a spectral family for policy purposes."""
    lname = name.lower()

    # MoE router/gating
    if any(
        tok in lname
        for tok in ("router", "routing", "gate", "gating", "dispatch", "switch")
    ):
        return "router"
    # MoE expert FFN
    if any(tok in lname for tok in ("experts", "expert", "moe", "mixture_of_experts")):
        return "expert_ffn"

    if "mlp" in lname or "ffn" in lname or "feed_forward" in lname:
        return "ffn"

    if (
        "attn" in lname
        or "attention" in lname
        or any(
            token in lname
            for token in ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn"]
        )
    ):
        return "attn"

    if "embed" in lname or "wte" in lname or "embedding" in lname:
        return "embed"

    module_type = module.__class__.__name__.lower()
    if "embedding" in module_type:
        return "embed"
    if "conv1d" in module_type or "linear" in module_type:
        if "attn" in lname:
            return "attn"
        if "mlp" in lname or "ffn" in lname:
            return "ffn"

    return "other"


# Export the main components
__all__ = [
    "SpectralGuard",
    "SpectralPolicy",
    "compute_sigma_max",
    "auto_sigma_target",
    "apply_weight_rescale",
    "apply_relative_spectral_cap",
    "apply_spectral_control",
    "capture_baseline_sigmas",
    "scan_model_gains",
    "compute_family_stats",
    "summarize_family_z_scores",
    "classify_module_family",
]
