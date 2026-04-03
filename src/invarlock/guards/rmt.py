"""InvarLock RMT runtime guard contract.

This module exposes the public activation edge-risk runtime guard and its
policy helpers. Weight-space math, analysis, and detection helpers live in
their dedicated owner modules.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal

import torch
import torch.nn as nn

from invarlock.core.api import Guard
from invarlock.core.types import GuardValidationResult

from . import (
    rmt_activation_runtime,
    rmt_analysis,
    rmt_detection,
    rmt_math,
)
from .rmt_activation_runtime import (
    activation_svd_outliers as _activation_svd_outliers_impl,
)
from .rmt_activation_runtime import (
    compute_activation_outliers as _compute_activation_outliers_impl,
)
from .rmt_policy import (
    RMTPolicy,
    RMTPolicyDict,
    create_custom_rmt_policy,
    get_rmt_policy,
)
from .rmt_policy import (
    build_rmt_guard_policy as _build_rmt_guard_policy_impl,
)
from .rmt_policy import (
    compute_epsilon_violations as _compute_epsilon_violations_impl,
)
from .rmt_runtime import (
    after_edit_rmt_guard as _after_edit_rmt_guard_impl,
)
from .rmt_runtime import (
    apply_rmt_detection_and_correction as _apply_rmt_detection_and_correction_impl,
)
from .rmt_runtime import (
    before_edit_rmt_guard as _before_edit_rmt_guard_impl,
)
from .rmt_runtime import (
    finalize_rmt_guard as _finalize_rmt_guard_impl,
)
from .rmt_runtime import (
    prepare_rmt_guard as _prepare_rmt_guard_impl,
)
from .rmt_runtime import (
    validate_rmt_guard as _validate_rmt_guard_impl,
)

__all__ = [
    "RMTGuard",
    "RMTPolicy",
    "RMTPolicyDict",
    "get_rmt_policy",
    "create_custom_rmt_policy",
]

# Preserve module-level monkeypatch targets used by existing tests and callers.
_COMPAT_MODULE_EXPORTS = (rmt_detection, rmt_math)

# === Guard Implementation ===

# Import GuardOutcome types if available
try:
    from invarlock.core.types import GuardOutcome

    HAS_GUARD_OUTCOME = True
except ImportError:
    # Fallback for standalone usage or when types not available
    HAS_GUARD_OUTCOME = False
    GuardOutcome = dict


class RMTGuard(Guard):
    """
    Standalone RMT Guard for baseline-aware outlier detection and correction.

    Implements Marchenko-Pastur theory-based outlier tracking with:
    - Activation-based outlier counts from calibration batches
    - Baseline capture of MP bulk edges for linear layers (correction/fallback)
    - Conservative outlier detection with deadband support
    - Optional in-place correction preserving weight tying
    - Comprehensive event logging and metrics

    Policy Structure:
    - q: MP aspect ratio (auto-derived or manual)
    - deadband: Tolerance margin before flagging (default 0.10 = 10%)
    - margin: RMT threshold ratio (default 1.5)
    - correct: Enable automatic correction (default True)

    Linear Layer Scope (correction/fallback):
    - attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj
    - Excludes: embeddings, LM head, layer norms, biases
    """

    name = "rmt"

    def __init__(
        self,
        q: float | Literal["auto"] = "auto",
        deadband: float = 0.10,
        margin: float = 1.5,
        correct: bool = True,
        *,
        epsilon_default: float = 0.10,
        epsilon_by_family: dict[str, float] | None = None,
    ):
        """
        Initialize RMT Guard.

        Args:
            q: MP aspect ratio (auto-derived from weight shapes if "auto")
            deadband: Tolerance margin before flagging outliers (0.10 = 10%)
            margin: RMT threshold ratio for outlier detection (1.5)
            correct: Enable automatic correction when outliers detected
        """
        self.q = q
        self.deadband = deadband
        self.margin = margin
        self.correct = correct
        self.epsilon_default = float(epsilon_default)
        self.epsilon_by_family: dict[str, float] = {}
        self._set_epsilon_by_family(epsilon_by_family)
        for family_key in ("attn", "ffn", "embed", "other"):
            self.epsilon_by_family.setdefault(family_key, self.epsilon_default)

        # Measurement contract knobs (vNext)
        self.estimator: dict[str, Any] = {
            "type": "power_iter",
            "iters": 3,
            "init": "ones",
        }
        self.activation_sampling: dict[str, Any] = {
            "windows": {"count": 8, "indices_policy": "evenly_spaced"}
        }

        # Internal state (activation edge-risk scoring)
        self._calibration_batches: list[Any] = []
        self._activation_ready = False
        self._require_activation = False
        self._activation_required_failed = False
        self._activation_required_reason: str | None = None
        self._run_profile: str | None = None
        self._run_tier: str | None = None
        self.prepared = False
        self._event_records: list[dict[str, Any]] = []
        self._last_result: dict[str, Any] | None = None
        self.adapter = None  # Store adapter for tying map access

        # Canonical linear-layer scope enforced by the RMT analysis owner.
        self.allowed_suffixes = [
            ".attn.c_attn",
            ".attn.c_proj",
            ".mlp.c_fc",
            ".mlp.c_proj",
        ]
        self.baseline_edge_risk_by_family: dict[str, float] = {}
        self.baseline_edge_risk_by_module: dict[str, float] = {}
        self.edge_risk_by_family: dict[str, float] = {}
        self.edge_risk_by_module: dict[str, float] = {}
        self.epsilon_violations: list[dict[str, Any]] = []
        self.baseline_sigmas: dict[str, float] = {}
        self.baseline_mp_stats: dict[str, dict[str, float]] = {}

    def _log_event(
        self, operation: str, level: str = "INFO", message: str = "", **data
    ):
        """Log an event with timestamp."""
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
                "component": "rmt_guard",
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
        """Capture tier/profile context for activation requirements."""
        ctx = getattr(report, "context", {}) or {}
        profile = ""
        tier = "balanced"
        if isinstance(ctx, dict):
            profile = str(ctx.get("profile", "") or "").strip().lower()
            auto = ctx.get("auto")
            if isinstance(auto, dict):
                tier = str(auto.get("tier", tier) or tier).strip().lower()
        self._run_profile = profile or None
        self._run_tier = tier or None
        self._require_activation = bool(profile in {"ci", "release"})

    def _set_epsilon_default(self, epsilon: Any) -> None:
        """Set the default ε used when a family value is missing."""
        if epsilon is None:
            return
        try:
            eps = float(epsilon)
        except (TypeError, ValueError):
            return
        if eps >= 0.0 and math.isfinite(eps):
            self.epsilon_default = eps

    def _set_epsilon_by_family(self, epsilon: Any) -> None:
        """Set per-family ε values."""
        if not isinstance(epsilon, dict):
            return
        for family, value in epsilon.items():
            try:
                eps = float(value)
            except (TypeError, ValueError):
                continue
            if eps >= 0.0 and math.isfinite(eps):
                self.epsilon_by_family[str(family)] = eps

    @staticmethod
    def _classify_family(module_name: str) -> str:
        """Classify module name into a guard family (vNext: {attn, ffn, embed, other})."""
        lower = module_name.lower()
        if any(tok in lower for tok in ("attn", "attention", "self_attn")):
            return "attn"
        if any(
            tok in lower
            for tok in ("router", "routing", "gate", "gating", "dispatch", "switch")
        ):
            return "ffn"
        if any(
            tok in lower for tok in ("experts", "expert", "moe", "mixture_of_experts")
        ):
            return "ffn"
        if any(tok in lower for tok in ("mlp", "ffn", "c_fc", "feed_forward")):
            return "ffn"
        if "embed" in lower or "wte" in lower or "wpe" in lower:
            return "embed"
        return "other"

    def _count_outliers_per_family(
        self, per_layer: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Count outliers grouped by family."""
        counts: dict[str, int] = {}
        for layer_info in per_layer:
            outlier_count = layer_info.get("outlier_count")
            if outlier_count is None:
                if not layer_info.get("has_outlier"):
                    continue
                increment = 1
            else:
                try:
                    increment = int(outlier_count)
                except (TypeError, ValueError):
                    continue
                if increment <= 0:
                    continue
            module_name = layer_info.get("module_name", "")
            family = self._classify_family(module_name)
            counts[family] = counts.get(family, 0) + increment
        return counts

    def _compute_epsilon_violations(self) -> list[dict[str, Any]]:
        return _compute_epsilon_violations_impl(self)

    def _get_linear_modules(self, model: nn.Module) -> list[tuple[str, nn.Module]]:
        """Get linear modules in scope using the canonical analysis owner."""
        return rmt_analysis.collect_linear_rmt_modules(
            model,
            allowed_suffixes=self.allowed_suffixes,
        )

    def _collect_calibration_batches(self, calib: Any, max_windows: int) -> list[Any]:
        return rmt_activation_runtime.collect_calibration_batches(
            calib,
            max_windows,
            activation_sampling=self.activation_sampling,
        )

    def _prepare_activation_inputs(
        self, batch: Any, device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return rmt_activation_runtime.prepare_activation_inputs(batch, device)

    @staticmethod
    def _batch_token_weight(
        input_ids: torch.Tensor | None, attention_mask: torch.Tensor | None
    ) -> int:
        return rmt_activation_runtime.batch_token_weight(input_ids, attention_mask)

    def _get_activation_modules(self, model: nn.Module) -> list[tuple[str, nn.Module]]:
        return rmt_activation_runtime.get_activation_modules(
            model,
            allowed_suffixes=self.allowed_suffixes,
        )

    def _activation_edge_risk(
        self, activations: Any
    ) -> tuple[float, float, float] | None:
        return rmt_activation_runtime.activation_edge_risk(
            activations,
            estimator=self.estimator,
        )

    def _compute_activation_edge_risk(
        self, model: nn.Module, batches: list[Any]
    ) -> dict[str, Any] | None:
        return rmt_activation_runtime.compute_activation_edge_risk(
            model,
            batches,
            allowed_suffixes=self.allowed_suffixes,
            activation_sampling=self.activation_sampling,
            estimator=self.estimator,
            deadband=self.deadband,
            margin=self.margin,
            classify_family_fn=self._classify_family,
            adapter=self.adapter,
        )

    def _activation_svd_outliers(
        self, activations: Any, margin: float, deadband: float
    ) -> tuple[int, float, float]:
        return _activation_svd_outliers_impl(
            activations, margin=margin, deadband=deadband
        )

    def _compute_activation_outliers(
        self, model: nn.Module, batches: list[Any]
    ) -> dict[str, Any] | None:
        return _compute_activation_outliers_impl(self, model, batches)

    def _apply_rmt_detection_and_correction(self, model: nn.Module) -> dict[str, Any]:
        return _apply_rmt_detection_and_correction_impl(self, model)

    def prepare(
        self,
        model: nn.Module,
        adapter=None,
        calib=None,
        policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _prepare_rmt_guard_impl(
            self,
            model,
            adapter=adapter,
            calib=calib,
            policy=policy,
        )

    def before_edit(self, model: nn.Module) -> None:
        _before_edit_rmt_guard_impl(self, model)

    def after_edit(self, model: nn.Module) -> None:
        _after_edit_rmt_guard_impl(self, model)

    def validate(
        self, model: Any, adapter: Any, context: dict[str, Any]
    ) -> GuardValidationResult:
        return _validate_rmt_guard_impl(self, model, adapter, context)

    def finalize(self, model: nn.Module, adapter=None) -> GuardOutcome | dict[str, Any]:
        return _finalize_rmt_guard_impl(
            self,
            model,
            adapter,
            has_guard_outcome=HAS_GUARD_OUTCOME,
            guard_outcome_type=GuardOutcome,
        )

    def policy(self) -> RMTPolicyDict:
        return _build_rmt_guard_policy_impl(self)
