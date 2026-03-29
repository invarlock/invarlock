"""InvarLock RMT runtime guard contract.

This module exposes the public activation edge-risk runtime guard and its
policy helpers. Weight-space math, analysis, and detection helpers live in
their dedicated owner modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, TypedDict

import torch
import torch.nn as nn

from invarlock.core.api import Guard

from . import (
    rmt_activation_runtime,
    rmt_analysis,
    rmt_detection,
    rmt_math,
    rmt_result_contract,
)
from ._contracts import guard_assert

__all__ = [
    "RMTGuard",
    "RMTPolicy",
    "RMTPolicyDict",
    "get_rmt_policy",
    "create_custom_rmt_policy",
]

# === Guard Implementation ===

# Import GuardOutcome types if available
try:
    from invarlock.core.types import GuardOutcome

    HAS_GUARD_OUTCOME = True
except ImportError:
    # Fallback for standalone usage or when types not available
    HAS_GUARD_OUTCOME = False
    GuardOutcome = dict


@dataclass
class RMTPolicy:
    """
    RMT Guard Policy Configuration.

    Defines parameters for baseline-aware RMT outlier detection and correction.
    """

    q: float | Literal["auto"] = (
        "auto"  # MP aspect ratio m/n (auto-derived from weights)
    )
    deadband: float = 0.10  # Tolerance margin (10%)
    margin: float = 1.5  # RMT threshold ratio
    correct: bool = True  # Enable automatic correction


class RMTPolicyDict(TypedDict, total=False):
    """TypedDict version of the RMT guard policy."""

    q: float | Literal["auto"]
    deadband: float
    margin: float
    correct: bool
    epsilon_default: float
    epsilon_by_family: dict[str, float]
    activation_required: bool
    estimator: dict[str, Any]
    activation: dict[str, Any]


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
        self.events: list[dict[str, Any]] = []
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
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "component": "rmt_guard",
            "operation": operation,
            "level": level,
            "message": message,
            "data": data,
        }
        self.events.append(event)

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
        """Compute ε-band violations per family on activation edge-risk scores."""
        violations: list[dict[str, Any]] = []
        families = set(self.edge_risk_by_family) | set(
            self.baseline_edge_risk_by_family
        )
        for family in families:
            base = float(self.baseline_edge_risk_by_family.get(family, 0.0) or 0.0)
            cur = float(self.edge_risk_by_family.get(family, 0.0) or 0.0)
            if base <= 0.0:
                continue
            epsilon_val = float(
                self.epsilon_by_family.get(family, self.epsilon_default)
            )
            allowed = (1.0 + epsilon_val) * base
            if cur > allowed:
                delta = (cur / base) - 1.0
                violations.append(
                    {
                        "family": family,
                        "edge_base": base,
                        "edge_cur": cur,
                        "delta": float(delta),
                        "allowed": allowed,
                        "epsilon": epsilon_val,
                    }
                )
        return violations

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
        )

    def _activation_svd_outliers(
        self, activations: Any, margin: float, deadband: float
    ) -> tuple[int, float, float]:
        """Count activation singular values beyond the MP edge."""
        if isinstance(activations, tuple | list):
            activations = activations[0] if activations else None
        if not isinstance(activations, torch.Tensor):
            return 0, 0.0, 0.0

        if activations.dim() < 2:
            return 0, 0.0, 0.0

        if activations.dim() > 2:
            activations = activations.reshape(-1, activations.shape[-1])

        if activations.numel() == 0:
            return 0, 0.0, 0.0

        try:
            mat = activations.detach().float().cpu()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return 0, 0.0, 0.0

        if not torch.isfinite(mat).all():
            return 0, 0.0, 0.0

        mat = mat - mat.mean()
        std = float(mat.std().item())
        if not math.isfinite(std) or std <= 0.0:
            return 0, 0.0, 0.0

        mat = mat / std
        m, n = mat.shape
        mp_edge_val = rmt_math.mp_bulk_edge(m, n, whitened=False)
        threshold = mp_edge_val * (1.0 + deadband) * margin

        try:
            s_vals = torch.linalg.svdvals(mat)
        except (RuntimeError, torch.linalg.LinAlgError):
            return 0, 0.0, 0.0

        if s_vals.numel() == 0:
            return 0, 0.0, 0.0

        sigma_max = float(s_vals.max().item())
        max_ratio = sigma_max / max(mp_edge_val, 1e-12)
        outlier_count = int((s_vals > threshold).sum().item())
        return outlier_count, float(max_ratio), sigma_max

    def _compute_activation_outliers(
        self, model: nn.Module, batches: list[Any]
    ) -> dict[str, Any] | None:
        """Compute activation-based RMT outlier counts."""
        if not batches:
            return None

        modules = self._get_activation_modules(model)
        if not modules:
            return None

        per_layer_map: dict[str, dict[str, Any]] = {}
        batch_weight_holder = {"weight": 1}
        for idx, (module_name, _module) in enumerate(modules):
            per_layer_map[module_name] = {
                "layer": idx,
                "module_name": module_name,
                "sigma_max": 0.0,
                "worst_ratio": 0.0,
                "outlier_count": 0,
                "has_outlier": False,
            }

        handles: list[Any] = []

        def _make_hook(name: str):
            def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any):
                try:
                    outliers, max_ratio, sigma_max = self._activation_svd_outliers(
                        output, self.margin, self.deadband
                    )
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    return
                stats = per_layer_map.get(name)
                if stats is None:
                    return
                weight = int(batch_weight_holder.get("weight", 1) or 1)
                if outliers > 0:
                    increment = int(outliers) * weight
                    stats["outlier_count"] = (
                        int(stats.get("outlier_count", 0)) + increment
                    )
                    stats["has_outlier"] = True
                stats["worst_ratio"] = max(
                    float(stats.get("worst_ratio", 0.0)), float(max_ratio)
                )
                stats["sigma_max"] = max(
                    float(stats.get("sigma_max", 0.0)), float(sigma_max)
                )

            return _hook

        for name, module in modules:
            try:
                handles.append(module.register_forward_hook(_make_hook(name)))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue

        model_was_training = model.training
        model.eval()
        try:
            device = next(model.parameters()).device
        except StopIteration:
            return None
        batches_used = 0
        token_weight_total = 0

        try:
            with torch.inference_mode():
                for batch in batches:
                    inputs, attention_mask = self._prepare_activation_inputs(
                        batch, device
                    )
                    if inputs is None:
                        continue
                    batch_weight = self._batch_token_weight(inputs, attention_mask)
                    batch_weight_holder["weight"] = batch_weight
                    try:
                        if attention_mask is not None:
                            model(inputs, attention_mask=attention_mask)
                        else:
                            model(inputs)
                        batches_used += 1
                        token_weight_total += batch_weight
                    except TypeError:
                        try:
                            model(inputs)
                            batches_used += 1
                            token_weight_total += batch_weight
                        except (AttributeError, RuntimeError, TypeError, ValueError):
                            continue
                    except (AttributeError, RuntimeError, ValueError):
                        continue
        finally:
            for handle in handles:
                try:
                    handle.remove()
                except (AttributeError, RuntimeError):
                    pass
            if model_was_training:
                model.train()

        if batches_used == 0:
            return None

        per_layer = [per_layer_map[name] for name, _module in modules]
        flagged_layers = [
            info["layer"] for info in per_layer if info.get("has_outlier")
        ]
        outlier_total = sum(
            int(info.get("outlier_count", 0) or 0) for info in per_layer
        )
        max_ratio = max(
            (float(info.get("worst_ratio", 0.0)) for info in per_layer), default=0.0
        )

        return {
            "has_outliers": bool(flagged_layers),
            "n_layers_flagged": len(flagged_layers),
            "outlier_count": outlier_total,
            "max_ratio": max_ratio,
            "threshold": (1.0 + self.deadband) * self.margin,
            "per_layer": per_layer,
            "flagged_layers": flagged_layers,
            "analysis_source": "activations",
            "token_weight_total": int(token_weight_total),
            "token_weighted": True,
        }

    def _apply_rmt_detection_and_correction(self, model: nn.Module) -> dict[str, Any]:
        """
        Apply Step 5 RMT detection and correction with adapter support.

        Uses exact Step 5 detection rule: ratio = σ_max_post / bulk_edge_base
        Flag if ratio > (1+deadband)*margin
        """
        modules_to_analyze = self._get_linear_modules(model)
        self._log_event(
            "rmt_correction",
            message=f"Applying Step 5 detection and correction to {len(modules_to_analyze)} modules",
        )
        result = rmt_detection.step5_detect_and_correct_modules(
            modules_to_analyze,
            baseline_sigmas=self.baseline_sigmas,
            baseline_mp_stats=self.baseline_mp_stats,
            deadband=self.deadband,
            margin=self.margin,
            correct=self.correct,
            adapter=self.adapter,
        )
        for event in result.pop("events", []):
            operation = str(event.get("operation", "rmt_event"))
            module_name = event.get("module_name")
            if operation == "rmt_correct":
                self._log_event(
                    operation,
                    message=f"Applied correction to {module_name}",
                    module_name=module_name,
                    pre_ratio=event.get("pre_ratio"),
                    threshold=event.get("threshold"),
                )
            elif operation == "rmt_correct_failed":
                self._log_event(
                    operation,
                    level="ERROR",
                    message=f"Correction failed for {module_name}: {event.get('error')}",
                    module_name=module_name,
                    error=event.get("error"),
                )
        return result

    def prepare(
        self,
        model: nn.Module,
        adapter=None,
        calib=None,
        policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare RMT guard by capturing baseline activation edge-risk scores."""
        import time

        start_time = time.time()
        self._activation_required_failed = False
        self._activation_required_reason = None

        # Store adapter for tying map access (if used by downstream code)
        self.adapter = adapter

        # Policy overrides (vNext contract)
        if isinstance(policy, dict) and policy:
            if "epsilon" in policy:
                from invarlock.core.exceptions import ValidationError

                raise ValidationError(
                    code="E501",
                    message="POLICY-PARAM-INVALID",
                    details={
                        "param": "epsilon",
                        "hint": "Use rmt.epsilon_default and rmt.epsilon_by_family instead.",
                    },
                )
            if "q" in policy:
                q_val = policy.get("q")
                if q_val == "auto":
                    self.q = "auto"
                else:
                    try:
                        self.q = float(q_val)
                    except (TypeError, ValueError):
                        self.q = "auto"
            if "deadband" in policy:
                self.deadband = float(policy.get("deadband", self.deadband))
            if "margin" in policy:
                try:
                    self.margin = float(policy.get("margin", self.margin))
                except (TypeError, ValueError):
                    pass
            if "correct" in policy:
                self.correct = bool(policy.get("correct"))
            if "epsilon_by_family" in policy:
                self._set_epsilon_by_family(policy["epsilon_by_family"])
            if "epsilon_default" in policy:
                self._set_epsilon_default(policy["epsilon_default"])
            for family_key in ("attn", "ffn", "embed", "other"):
                self.epsilon_by_family.setdefault(family_key, self.epsilon_default)
            if "activation_required" in policy:
                self._require_activation = bool(policy.get("activation_required"))

            estimator_policy = policy.get("estimator")
            if isinstance(estimator_policy, dict):
                try:
                    iters = int(estimator_policy.get("iters", 3) or 3)
                except (TypeError, ValueError):
                    iters = 3
                if iters < 1:
                    iters = 1
                init = (
                    str(estimator_policy.get("init", "ones") or "ones").strip().lower()
                )
                if init not in {"ones", "e0"}:
                    init = "ones"
                self.estimator = {"type": "power_iter", "iters": iters, "init": init}

            activation_policy = policy.get("activation")
            if isinstance(activation_policy, dict):
                sampling = activation_policy.get("sampling")
                if isinstance(sampling, dict):
                    windows = sampling.get("windows")
                    if isinstance(windows, dict):
                        cfg = dict(self.activation_sampling.get("windows") or {})
                        if windows.get("count") is not None:
                            try:
                                cfg["count"] = int(windows.get("count") or 0)
                            except (TypeError, ValueError):
                                pass
                        if windows.get("indices_policy") is not None:
                            cfg["indices_policy"] = str(
                                windows.get("indices_policy")
                                or cfg.get("indices_policy")
                            )
                        self.activation_sampling["windows"] = cfg

        self._log_event(
            "prepare",
            message="Preparing RMT guard baseline activation edge-risk metrics",
        )

        try:
            windows_cfg = self.activation_sampling.get("windows") or {}
            try:
                window_count = int(windows_cfg.get("count", 0) or 0)
            except (TypeError, ValueError):
                window_count = 0
            self._calibration_batches = (
                self._collect_calibration_batches(calib, window_count)
                if calib is not None and window_count > 0
                else []
            )

            self.baseline_edge_risk_by_family = {}
            self.baseline_edge_risk_by_module = {}
            self.edge_risk_by_family = {}
            self.edge_risk_by_module = {}
            self.epsilon_violations = []

            if self._require_activation and not self._calibration_batches:
                self._activation_required_failed = True
                self._activation_required_reason = "activation_required"
                self._activation_ready = False
                self.prepared = False
                return rmt_result_contract.build_prepare_result(
                    ready=False,
                    baseline_metrics={},
                    policy_applied=policy or {},
                    preparation_time=time.time() - start_time,
                    error="Activation batches required but unavailable",
                )

            baseline = (
                self._compute_activation_edge_risk(model, self._calibration_batches)
                if self._calibration_batches
                else None
            )
            if baseline is None:
                if self._require_activation:
                    self._activation_required_failed = True
                    self._activation_required_reason = "activation_baseline_unavailable"
                    self._activation_ready = False
                    self.prepared = False
                    return rmt_result_contract.build_prepare_result(
                        ready=False,
                        baseline_metrics={},
                        policy_applied=policy or {},
                        preparation_time=time.time() - start_time,
                        error="Activation baseline unavailable",
                    )
                # Non-required: treat as not ready and allow pipeline to continue.
                self._activation_ready = False
                self.prepared = True
                return rmt_result_contract.build_prepare_result(
                    ready=True,
                    baseline_metrics={},
                    policy_applied=policy or {},
                    preparation_time=time.time() - start_time,
                )

            self.baseline_edge_risk_by_module = dict(
                baseline.get("edge_risk_by_module") or {}
            )
            self.baseline_edge_risk_by_family = dict(
                baseline.get("edge_risk_by_family") or {}
            )
            self._activation_ready = True
            self.prepared = True

            preparation_time = time.time() - start_time
            return rmt_result_contract.build_prepare_result(
                ready=True,
                baseline_metrics={
                    "edge_risk_by_family": dict(self.baseline_edge_risk_by_family),
                    "measurement_contract": {
                        "kind": "activation_edge_risk",
                        "estimator": self.estimator,
                        "activation_sampling": self.activation_sampling,
                    },
                },
                policy_applied=policy or {},
                preparation_time=preparation_time,
            )

        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as e:
            self.prepared = False
            self._log_event(
                "prepare_failed",
                level="ERROR",
                message=f"Failed to prepare RMT guard: {str(e)}",
                error=str(e),
            )

            return rmt_result_contract.build_prepare_result(
                ready=False,
                baseline_metrics={},
                policy_applied=policy or {},
                preparation_time=time.time() - start_time,
                error=str(e),
            )

    def before_edit(self, model: nn.Module) -> None:
        """
        Execute before edit (no action needed for RMT).

        Args:
            model: The model about to be edited
        """
        if self.prepared:
            self._log_event(
                "before_edit",
                message="RMT guard ready for post-edit detection and correction",
            )

    def after_edit(self, model: nn.Module) -> None:
        """Execute after edit: compute activation edge-risk on sampled batches."""
        if not self.prepared:
            self._log_event(
                "after_edit_skipped",
                level="WARN",
                message="RMT guard not prepared, skipping post-edit detection",
            )
            return

        try:
            if self._require_activation and not self._calibration_batches:
                self._activation_required_failed = True
                self._activation_required_reason = "activation_unavailable"
                self._last_result = rmt_result_contract.build_after_edit_result()
                return

            current = (
                self._compute_activation_edge_risk(model, self._calibration_batches)
                if self._calibration_batches
                else None
            )
            if current is None:
                if self._require_activation:
                    self._activation_required_failed = True
                    self._activation_required_reason = (
                        "activation_edge_risk_unavailable"
                    )
                self._last_result = rmt_result_contract.build_after_edit_result()
                return

            self.edge_risk_by_module = dict(current.get("edge_risk_by_module") or {})
            self.edge_risk_by_family = dict(current.get("edge_risk_by_family") or {})
            self._last_result = dict(current)
            self.epsilon_violations = self._compute_epsilon_violations()

        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as e:
            self._log_event(
                "after_edit_failed",
                level="ERROR",
                message=f"RMT detection failed: {str(e)}",
                error=str(e),
            )
            self._last_result = rmt_result_contract.build_after_edit_result()
            self.epsilon_violations = []

    def validate(
        self, model: Any, adapter: Any, context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Validate model state (Guard ABC interface).

        Args:
            model: Model to validate
            adapter: ModelAdapter instance
            context: Validation context

        Returns:
            Dictionary with validation results
        """
        # Use finalize to get comprehensive results
        result = self.finalize(model, adapter)

        # Convert to simple dict format if GuardOutcome
        if (
            hasattr(result, "passed")
            and hasattr(result, "action")
            and hasattr(result, "metrics")
        ):
            violations_list: list[str] = []
            if hasattr(result, "violations") and result.violations:
                violations_list = [str(v) for v in result.violations]
            return {
                "passed": bool(result.passed),
                "action": str(result.action),
                "metrics": dict(result.metrics),
                "violations": violations_list,
                "message": "RMT guard validation completed",
            }
        else:
            return {
                "passed": result.get("passed", False),
                "action": "continue" if result.get("passed", False) else "warn",
                "metrics": result.get("metrics", {}),
                "violations": result.get("errors", []),
                "message": "RMT guard validation completed",
            }

    def finalize(self, model: nn.Module, adapter=None) -> GuardOutcome | dict[str, Any]:
        """Finalize RMT guard and return activation edge-risk ε-band outcome."""
        import time

        start_time = time.time()
        _ = adapter

        if not self.prepared:
            if HAS_GUARD_OUTCOME:
                return GuardOutcome(
                    name=self.name,
                    passed=False,
                    action="abort",
                    violations=[
                        {
                            "type": "preparation",
                            "severity": "error",
                            "message": "RMT guard not properly prepared",
                            "module_name": None,
                        }
                    ],
                    metrics={
                        "prepared": False,
                        "finalize_time": time.time() - start_time,
                    },
                )
            return {
                "passed": False,
                "metrics": {
                    "prepared": False,
                    "finalize_time": time.time() - start_time,
                },
                "errors": ["RMT guard not properly prepared"],
            }

        if self._require_activation and self._activation_required_failed:
            reason = self._activation_required_reason or "activation_required"
            finalize_time = time.time() - start_time
            if HAS_GUARD_OUTCOME:
                return GuardOutcome(
                    name=self.name,
                    passed=False,
                    action="abort",
                    violations=[
                        {
                            "type": "activation_required",
                            "severity": "error",
                            "message": "Activation edge-risk analysis required but unavailable",
                            "module_name": None,
                            "reason": reason,
                        }
                    ],
                    metrics={
                        "prepared": True,
                        "activation_required": True,
                        "activation_ready": False,
                        "activation_reason": reason,
                        "finalize_time": finalize_time,
                    },
                )
            return {
                "passed": False,
                "metrics": {
                    "prepared": True,
                    "activation_required": True,
                    "activation_ready": False,
                    "activation_reason": reason,
                    "finalize_time": finalize_time,
                },
                "errors": ["Activation edge-risk analysis required but unavailable"],
            }

        if not self.edge_risk_by_family and self._calibration_batches:
            current = self._compute_activation_edge_risk(
                model, self._calibration_batches
            )
            if current is not None:
                self.edge_risk_by_family = dict(
                    current.get("edge_risk_by_family") or {}
                )
                self.edge_risk_by_module = dict(
                    current.get("edge_risk_by_module") or {}
                )
                self._last_result = dict(current)

        self.epsilon_violations = self._compute_epsilon_violations()
        for fam, eps in self.epsilon_by_family.items():
            guard_assert(eps >= 0.0, f"rmt.epsilon[{fam}] must be >= 0")

        stable = not self.epsilon_violations
        action = "continue" if stable else "abort"
        finalize_time = time.time() - start_time

        metrics: dict[str, Any] = {
            "prepared": True,
            "stable": stable,
            "edge_risk_by_family_base": dict(self.baseline_edge_risk_by_family),
            "edge_risk_by_family": dict(self.edge_risk_by_family),
            "epsilon_by_family": dict(self.epsilon_by_family),
            "epsilon_violations": list(self.epsilon_violations),
            "measurement_contract": {
                "kind": "activation_edge_risk",
                "estimator": self.estimator,
                "activation_sampling": self.activation_sampling,
            },
            "finalize_time": finalize_time,
        }

        violations: list[dict[str, Any]] = []
        for v in self.epsilon_violations:
            violations.append(
                {
                    "type": "epsilon_band",
                    "severity": "error",
                    "family": v.get("family"),
                    "edge_base": v.get("edge_base"),
                    "edge_cur": v.get("edge_cur"),
                    "allowed": v.get("allowed"),
                    "epsilon": v.get("epsilon"),
                    "delta": v.get("delta"),
                    "message": f"ε-band violation in {v.get('family')}",
                }
            )

        if HAS_GUARD_OUTCOME:
            return GuardOutcome(
                name=self.name,
                passed=stable,
                action=action,
                violations=violations,
                metrics=metrics,
            )
        return {
            "passed": stable,
            "action": action,
            "metrics": metrics,
            "violations": violations,
        }

    def policy(self) -> RMTPolicyDict:
        """
        Get default policy for RMT guard.

        Returns:
            RMTPolicyDict with current configuration
        """
        return RMTPolicyDict(
            q=self.q,
            deadband=self.deadband,
            margin=self.margin,
            correct=self.correct,
            epsilon_default=float(self.epsilon_default),
            epsilon_by_family=self.epsilon_by_family.copy(),
        )


# === Policy Utilities ===


def get_rmt_policy(name: str = "balanced") -> RMTPolicyDict:
    """
    Get a RMT policy by name.

    Args:
        name: Policy name ("conservative", "balanced", "aggressive")

    Returns:
        RMTPolicyDict configuration
    """
    # Per-family ε values match runtime tiers.yaml.
    policies = {
        "conservative": RMTPolicyDict(
            q="auto",
            deadband=0.05,
            margin=1.3,
            correct=True,
            epsilon_default=0.06,
            epsilon_by_family={"ffn": 0.06, "attn": 0.05, "embed": 0.07, "other": 0.07},
        ),
        "balanced": RMTPolicyDict(
            q="auto",
            deadband=0.10,
            margin=1.5,
            correct=True,
            epsilon_default=0.10,
            epsilon_by_family={"ffn": 0.10, "attn": 0.08, "embed": 0.12, "other": 0.12},
        ),
        "aggressive": RMTPolicyDict(
            q="auto",
            deadband=0.15,
            margin=1.8,
            correct=True,
            epsilon_default=0.15,
            epsilon_by_family={"ffn": 0.15, "attn": 0.15, "embed": 0.15, "other": 0.15},
        ),
    }

    if name not in policies:
        from invarlock.core.exceptions import GuardError

        available = list(policies.keys())
        raise GuardError(
            code="E502",
            message="POLICY-NOT-FOUND",
            details={"name": name, "available": available},
        )

    return policies[name]


def create_custom_rmt_policy(
    q: float | Literal["auto"] = "auto",
    deadband: float = 0.10,
    margin: float = 1.5,
    correct: bool = True,
    *,
    epsilon_default: float = 0.1,
    epsilon_by_family: dict[str, float] | None = None,
) -> RMTPolicyDict:
    """
    Create a custom RMT policy.

    Args:
        q: MP aspect ratio (auto-derived or manual)
        deadband: Tolerance margin (0.0-0.5)
        margin: RMT threshold ratio (> 1.0)
        correct: Enable automatic correction

    Returns:
        Custom RMTPolicyDict configuration
    """
    if isinstance(q, float) and not 0.1 <= q <= 10.0:
        from invarlock.core.exceptions import ValidationError

        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "q", "value": q},
        )

    if not 0.0 <= deadband <= 0.5:
        from invarlock.core.exceptions import ValidationError

        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "deadband", "value": deadband},
        )

    if not margin >= 1.0:
        from invarlock.core.exceptions import ValidationError

        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "margin", "value": margin},
        )

    from invarlock.core.exceptions import ValidationError

    try:
        eps_default_val = float(epsilon_default)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "epsilon_default", "value": epsilon_default},
        ) from exc
    if not (math.isfinite(eps_default_val) and eps_default_val >= 0.0):
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "epsilon_default", "value": epsilon_default},
        )

    eps_by_family: dict[str, float] = {}
    if epsilon_by_family is not None:
        if not isinstance(epsilon_by_family, dict):
            raise ValidationError(
                code="E501",
                message="POLICY-PARAM-INVALID",
                details={"param": "epsilon_by_family", "value": epsilon_by_family},
            )
        for family, value in epsilon_by_family.items():
            try:
                eps_val = float(value)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    code="E501",
                    message="POLICY-PARAM-INVALID",
                    details={
                        "param": "epsilon_by_family",
                        "family": str(family),
                        "value": value,
                    },
                ) from exc
            if not (math.isfinite(eps_val) and eps_val >= 0.0):
                raise ValidationError(
                    code="E501",
                    message="POLICY-PARAM-INVALID",
                    details={
                        "param": "epsilon_by_family",
                        "family": str(family),
                        "value": value,
                    },
                )
            eps_by_family[str(family)] = eps_val

    return RMTPolicyDict(
        q=q,
        deadband=deadband,
        margin=margin,
        correct=correct,
        epsilon_default=eps_default_val,
        epsilon_by_family=eps_by_family,
    )
