"""
InvarLock – Safety: Data-Driven Variance Equalization (DD-VE)
=========================================================

Branch-level variance equalizer for transformer blocks to maintain
stable residual stream dynamics after edits.

For each transformer block, measures the variance of residual branch
outputs (attention and MLP) and scales projection weights to maintain
Var(x_out) ≈ 1 when Var(x_in) ≈ 1.
"""

from __future__ import annotations

import copy
import fnmatch
import hashlib
import itertools
import math
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from invarlock.cli._evidence import maybe_dump_guard_evidence
from invarlock.core.api import Guard
from invarlock.core.bootstrap import compute_paired_delta_log_ci

from ._contracts import guard_assert

# Import the policy type and Guard interface
from .policies import VariancePolicyDict

__all__ = ["equalise_residual_variance", "equalise_branch_variance", "VarianceGuard"]


def _safe_mean(
    samples: list[float] | np.ndarray, default: float | None = None
) -> float | None:
    """
    Compute mean of samples, returning default if empty.

    Avoids numpy RuntimeWarning "Mean of empty slice" when samples is empty
    or contains no valid values.

    Args:
        samples: List or array of float values.
        default: Value to return if samples is empty.

    Returns:
        Mean value or default if samples is empty.
    """
    if samples is None:
        return default
    arr = np.asarray(samples)
    if arr.size == 0:
        return default
    return float(np.nanmean(arr))


try:  # Optional dependency: tqdm (progress bars)
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - exercised only when tqdm is absent

    class _TqdmShim:
        def __init__(self, iterable=None, total=None, **kwargs):
            self._iterable = iterable
            self.total = total

        def __iter__(self):
            if self._iterable is None:
                return iter(())
            return iter(self._iterable)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, n: int = 1) -> None:
            return None

    def _tqdm(iterable=None, *args, **kwargs):
        return _TqdmShim(iterable=iterable, **kwargs)


tqdm = _tqdm


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel/DDP wrappers to get the underlying model.

    PyTorch's DataParallel and DistributedDataParallel wrap models with a
    `.module` attribute. This function traverses that chain to get the
    actual model, enabling consistent layer iteration regardless of how
    the model is wrapped for training/inference.
    """
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    return unwrapped


def _iter_transformer_layers(model: nn.Module):
    """Iterate over transformer layers in a model.

    Handles multiple transformer architectures and automatically unwraps
    DataParallel/DDP wrappers.
    """
    # Unwrap DataParallel/DDP wrappers first
    model = _unwrap_model(model)

    # Handle different model architectures
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # GPT-2 style
        yield from model.transformer.h
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        # LLaMA style
        yield from model.model.layers
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        # BERT style
        yield from model.encoder.layer
    elif hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        # T5/BART decoder style
        yield from model.decoder.layers
    elif hasattr(model, "layers"):
        # Generic transformer with top-level layers attribute
        yield from model.layers
    else:
        # Fallback: look for modules with attention
        for module in model.modules():
            if hasattr(module, "attn") and hasattr(module, "mlp"):
                yield module


@torch.no_grad()
def equalise_residual_variance(
    model: nn.Module,
    dataloader,
    *,
    windows: int = 32,
    tol: float = 0.02,
    scale_bias: bool = True,
    seed: int = 42,
    device: str | None = None,
    allow_empty: bool = False,
    clamp_range: tuple | None = (0.9, 1.1),
) -> dict[str, float]:
    """
    Apply data-driven variance equalization to transformer branches.

    This function measures the variance of each residual branch output
    (attention-proj and MLP-proj) and scales projection weights so that
    adding the branch back to the residual stream maintains stable variance.

    The scaling factor alpha = 1 / sqrt(1 + Var(F)) is used, where F is the
    branch output.

    Args:
        model: Transformer model to equalize
        dataloader: DataLoader for calibration
        windows: Number of calibration batches
        tol: Tolerance for skipping near-unity scales
        scale_bias: Whether to scale biases along with weights
        seed: Random seed for reproducibility
        device: Device to use (auto-detected if None)
        allow_empty: Whether to allow empty dataloader (returns empty dict)
        clamp_range: Optional (min, max) to clamp scaling factors (e.g., (0.9, 1.1))

    Returns:
        Dict mapping layer names to applied scaling factors
    """
    torch.manual_seed(seed)

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.eval()

    # Storage for variance measurements
    hooks: dict[str, Any] = {}
    sample_values: dict[str, list[float]] = defaultdict(list)

    def _branch_hook(name):
        def fn(_, __, out):
            y = out[0] if isinstance(out, tuple) else out
            y = y.detach().float()
            # Skip if tensor has zero elements
            if y.numel() == 0:
                return
            mean_square = float(y.pow(2).mean().item())
            sample_values[name].append(mean_square)

        return fn

    # Register hooks on projection layers
    for i, blk in enumerate(_iter_transformer_layers(model)):
        # Handle GPT-2 style architecture
        if hasattr(blk, "attn"):
            # Check for c_proj (GPT-2) or out_proj (generic)
            attn_proj = getattr(blk.attn, "c_proj", None) or getattr(
                blk.attn, "out_proj", None
            )
            if attn_proj is not None:
                name = f"block{i}.attn"
                hooks[name] = attn_proj.register_forward_hook(_branch_hook(name))

        if hasattr(blk, "mlp"):
            # Check for c_proj (GPT-2) or down_proj (LLaMA) or fc2 (generic)
            mlp_proj = (
                getattr(blk.mlp, "c_proj", None)
                or getattr(blk.mlp, "down_proj", None)
                or getattr(blk.mlp, "fc2", None)
            )
            if mlp_proj is not None:
                name = f"block{i}.mlp"
                hooks[name] = mlp_proj.register_forward_hook(_branch_hook(name))

    # Collect variance statistics
    try:
        it = itertools.islice(iter(dataloader), windows)
        batches = list(it)
    except (StopIteration, TypeError):
        batches = []

    if not batches and not allow_empty:
        raise ValueError("Empty dataloader provided and allow_empty=False")

    for batch in tqdm(batches, desc="DD-VE Calibration", leave=False):
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids", batch.get("inputs", None))
        elif isinstance(batch, tuple | list):
            # Handle tuple/list from TensorDataset
            input_ids = batch[0] if len(batch) > 0 else None
        else:
            input_ids = batch

        if input_ids is not None:
            # Convert to tensor if needed
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.as_tensor(input_ids)

            # Ensure input has batch dimension [batch, seq_len]
            # HF models (GPT-2, etc.) expect 2-D input tensors
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)

            with torch.no_grad():
                model(input_ids.to(device))

    # Remove hooks
    for h in hooks.values():
        h.remove()

    # Apply scaling factors
    applied_scales: dict[str, float] = {}

    for i, blk in enumerate(_iter_transformer_layers(model)):
        # Handle attention projection
        if hasattr(blk, "attn"):
            attn_proj = getattr(blk.attn, "c_proj", None) or getattr(
                blk.attn, "out_proj", None
            )
            if attn_proj is not None:
                name = f"block{i}.attn"
                values = sample_values.get(name, [])
                if values:
                    tensor_vals = torch.tensor(values, dtype=torch.float64)

                    # Winsorize to remove extreme outliers (≈1-2%)
                    if tensor_vals.numel() >= 10:
                        lower = torch.quantile(tensor_vals, 0.02)
                        upper = torch.quantile(tensor_vals, 0.98)
                        tensor_vals = torch.clamp(
                            tensor_vals, lower.item(), upper.item()
                        )

                    group_count = 8 if tensor_vals.numel() >= 8 else tensor_vals.numel()
                    if group_count > 1:
                        chunks = torch.chunk(tensor_vals, group_count)
                        group_means = torch.stack([chunk.mean() for chunk in chunks])
                        var_F = torch.median(group_means).item()
                    else:
                        var_F = tensor_vals.mean().item()

                    alpha = (1.0 / max(var_F, 1e-9)) ** 0.5

                    # Apply clamping if specified
                    if clamp_range is not None:
                        alpha = max(clamp_range[0], min(alpha, clamp_range[1]))

                    if abs(alpha - 1.0) >= tol:
                        with torch.no_grad():
                            attn_proj.weight.mul_(alpha)
                            if scale_bias and attn_proj.bias is not None:
                                attn_proj.bias.mul_(alpha)
                        applied_scales[name] = alpha

        # Handle MLP projection
        if hasattr(blk, "mlp"):
            mlp_proj = (
                getattr(blk.mlp, "c_proj", None)
                or getattr(blk.mlp, "down_proj", None)
                or getattr(blk.mlp, "fc2", None)
            )
            if mlp_proj is not None:
                name = f"block{i}.mlp"
                values = sample_values.get(name, [])
                if values:
                    tensor_vals = torch.tensor(values, dtype=torch.float64)

                    if tensor_vals.numel() >= 10:
                        lower = torch.quantile(tensor_vals, 0.02)
                        upper = torch.quantile(tensor_vals, 0.98)
                        tensor_vals = torch.clamp(
                            tensor_vals, lower.item(), upper.item()
                        )

                    group_count = 8 if tensor_vals.numel() >= 8 else tensor_vals.numel()
                    if group_count > 1:
                        chunks = torch.chunk(tensor_vals, group_count)
                        group_means = torch.stack([chunk.mean() for chunk in chunks])
                        var_F = torch.median(group_means).item()
                    else:
                        var_F = tensor_vals.mean().item()

                    alpha = (1.0 / max(var_F, 1e-9)) ** 0.5

                    # Apply clamping if specified
                    if clamp_range is not None:
                        alpha = max(clamp_range[0], min(alpha, clamp_range[1]))

                    if abs(alpha - 1.0) >= tol:
                        with torch.no_grad():
                            mlp_proj.weight.mul_(alpha)
                            if scale_bias and mlp_proj.bias is not None:
                                mlp_proj.bias.mul_(alpha)
                        applied_scales[name] = alpha

    return applied_scales


def equalise_branch_variance(
    model: nn.Module,
    dataloader,
    windows: int = 32,
    tol: float = 0.02,
    scale_bias: bool = True,
    seed: int = 42,
    device: str | None = None,
    allow_empty: bool = False,
) -> dict[str, float]:
    """
    Legacy alias for equalise_residual_variance.

    Maintained for backward compatibility.
    """
    return equalise_residual_variance(
        model=model,
        dataloader=dataloader,
        windows=windows,
        tol=tol,
        scale_bias=scale_bias,
        seed=seed,
        device=device,
        allow_empty=allow_empty,
    )


def _predictive_gate_outcome(
    mean_delta: float,
    delta_ci: tuple[float, float] | None,
    min_effect: float,
    one_sided: bool,
) -> tuple[bool, str]:
    """
    Decide whether the predictive gate passes given the CI and tier semantics.

    Args:
        mean_delta: Mean ΔlogNLL (virtual VE − no VE) from paired calibration.
        delta_ci: BCa confidence interval on ΔlogNLL (lower, upper).
        min_effect: Minimum absolute improvement required.
        one_sided: Whether to require a one-sided improvement (balanced tier).

    Returns:
        Tuple of (passed, reason) where reason is a canonical string used in stats.
    """
    guard_assert(min_effect >= 0.0, "variance.min_effect must be >= 0")
    if (
        delta_ci is None
        or len(delta_ci) != 2
        or not all(
            isinstance(val, (int | float)) and math.isfinite(val) for val in delta_ci
        )
    ):
        return False, "ci_unavailable"

    upper = float(delta_ci[1])
    min_effect = float(min_effect or 0.0)

    if one_sided:
        # One-sided improvement (ΔlogNLL < 0): certify a minimum effect by
        # requiring the *upper* bound of the (two-sided) CI to clear -min_effect.
        if upper >= 0.0:
            return False, "ci_contains_zero"
        if mean_delta >= 0.0:
            return False, "mean_not_negative"
        gain_lower_bound = -upper  # worst-case gain under CI
        if gain_lower_bound < min_effect:
            return False, "gain_below_threshold"
        return True, "ci_gain_met"

    # Two-sided improvement: CI must be strictly below zero.
    if upper >= 0.0:
        return False, "ci_contains_zero"

    gain_lower_bound = -upper  # Convert ΔlogNLL CI to gain CI lower bound.
    if gain_lower_bound < min_effect:
        return False, "gain_below_threshold"

    return True, "ci_gain_met"


# === Standalone Variance Guard Implementation ===


class VarianceGuard(Guard):
    """
    Standalone Variance Guard with A/B testing for data-driven variance equalization.

    Implements branch-level variance equalization with reinforced A/B gate functionality:
    - Measures variance of residual branch outputs during calibration
    - Computes scaling factors to maintain stable variance dynamics
    - A/B tests whether VE improves perplexity by at least min_gain
    - Only enables VE if it demonstrably helps (validation gate compliance)

    Policy Structure:
    - min_gain: Minimum primary-metric improvement required to enable VE
    - max_calib: Maximum calibration samples for A/B testing
    - scope: Which layers to process ("ffn", "attn", "both")
    - clamp: Scaling factor limits (min, max)
    - deadband: Tolerance margin before scaling
    - seed: Random seed for deterministic evaluation

    Reinforced A/B Testing Flow:
    1. Capture baseline model state with checkpoint discipline
    2. Measure variance and compute proposed scales during prepare
    3. A/B test with identical windows: evaluate the primary metric without VE, then with VE
    4. Apply robust gain math with tie-breaker deadband and absolute floor
    5. Enable VE only if improvement meets all criteria
    6. Idempotent enable/disable with exact state restoration
    """

    name = "variance"

    def __init__(self, policy: VariancePolicyDict | None = None):
        """
        Initialize Variance Guard with reinforced A/B gate logic.

        Args:
            policy: Variance policy configuration (uses balanced default if None)
        """
        from .policies import get_variance_policy

        self._policy = policy or get_variance_policy("balanced")
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
        self.events: list[dict[str, Any]] = []
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

        # A/B testing results with reinforced validation
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

        # Module tracking for safe scaling
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
        elif isinstance(tap_config, Sequence):
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

        # Checkpoint discipline for robust state management
        self._checkpoint_stack: list[dict[str, torch.Tensor]] = []
        self._enable_attempt_count = 0
        self._disable_attempt_count = 0

        # Constants for reinforced A/B gate
        self.TIE_BREAKER_DEADBAND = float(
            self._policy.get("tie_breaker_deadband", 0.005)
        )  # Extra deadband to avoid flapping on noise
        self.ABSOLUTE_FLOOR = 0.05  # Minimum improvement (ppl-like) to consider

        # Calibration storage for post-edit evaluation
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
        """Ensure calibration config contains required defaults."""
        default_calibration = {
            "windows": 6,
            "min_coverage": 4,
            "seed": self._policy.get("seed", 123),
        }
        calibration_cfg = self._policy.get("calibration", {}) or {}
        if not isinstance(calibration_cfg, dict):
            calibration_cfg = {}
        merged_calibration = {**default_calibration, **calibration_cfg}
        self._policy["calibration"] = merged_calibration

    def _log_event(
        self, operation: str, level: str = "INFO", message: str = "", **data
    ):
        """Log an event with timestamp."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "component": "variance_guard",
            "operation": operation,
            "level": level,
            "message": message,
            "data": data,
        }
        self.events.append(event)

    def set_run_context(self, report: Any) -> None:
        """Capture run-level context (edit metadata, pairing reference, etc.)."""
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
        if isinstance(self._run_context, dict):
            pairing_baseline = self._run_context.get("pairing_baseline")
        else:
            pairing_baseline = None
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
                pairing_digest = hashlib.blake2s(
                    joined.encode("utf-8"), digest_size=16
                ).hexdigest()
                pairing_stats = self._stats.setdefault("pairing_reference", {})
                pairing_stats.update(
                    {
                        "count": len(pairing_reference),
                        "digest": pairing_digest,
                    }
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
            params_changed = (
                0 if edit_info and edit_info.get("name") in {"noop"} else None
            )
        self._params_changed = params_changed
        if params_changed == 0:
            self._monitor_only = True
            self._log_event(
                "monitor_only",
                message="Variance guard forcing monitor-only mode (no parameters changed)",
            )
            # Clear proposed scales in monitor-only mode
            self._scales = {}

    def _normalize_module_name(self, name: str) -> str:
        """Normalize module names to transformer.h.<idx>.<branch>.c_proj form."""
        if not isinstance(name, str):
            return ""

        normalized = name.strip()
        if not normalized:
            return normalized

        if normalized.startswith("block"):
            parts = normalized.split(".")
            if len(parts) >= 2 and parts[0].startswith("block"):
                layer_idx = parts[0][5:]
                branch = parts[1]
                branch = "attn" if branch.startswith("attn") else "mlp"
                return f"transformer.h.{layer_idx}.{branch}.c_proj"

        if normalized.startswith("transformer.h."):
            if normalized.endswith(".c_proj"):
                return normalized
            if ".mlp" in normalized and ".c_proj" not in normalized:
                return f"{normalized}.c_proj"
            if ".attn" in normalized and ".c_proj" not in normalized:
                return f"{normalized}.c_proj"

        return normalized

    def _matches_tap(self, name: str) -> bool:
        """Return True if a module name matches configured tap patterns."""
        normalized = self._normalize_module_name(name)
        for pattern in self._tap_patterns:
            if fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(name, pattern):
                return True
        return False

    def _normalize_pairing_ids(
        self, prefix: str, window_ids: Sequence[Any]
    ) -> list[str]:
        normalized: list[str] = []
        for idx in window_ids:
            token = str(idx)
            if "::" in token:
                normalized.append(token)
            else:
                normalized.append(f"{prefix}::{token}")
        return normalized

    def _expected_window_ids(self) -> list[str]:
        return list(self._pairing_reference)

    def _normalize_scale_name(self, name: str) -> str:
        """Normalize a scale name to the canonical module path."""
        return self._normalize_module_name(name)

    def _scale_matches_target(self, scale_name: str, target_name: str) -> bool:
        """Check if a scale name from equalise_residual_variance matches a target module name.

        Handles the format mismatch between:
        - Scale names: block0.mlp, block0.attn
        - Target names: transformer.h.0.mlp.c_proj, transformer.h.0.attn.c_proj
        """
        # Normalize scale name to target format and check direct match
        normalized_scale = self._normalize_scale_name(scale_name)
        if normalized_scale == target_name:
            return True

        # Convert block format to layer-component extraction
        if scale_name.startswith("block") and (
            "attn" in scale_name or "mlp" in scale_name
        ):
            parts = scale_name.split(".")
            if len(parts) == 2:
                layer_part = parts[0]  # e.g., "block0"
                component = parts[1]  # e.g., "attn" or "mlp"
                if layer_part.startswith("block"):
                    try:
                        layer_num = layer_part[5:]  # Extract number from "block0"
                        # Check if target matches this pattern
                        if f"h.{layer_num}.{component}" in target_name:
                            return True
                    except (ValueError, IndexError):
                        pass

        return False

    def _is_focus_match(self, name: str) -> bool:
        """Check whether a module name matches the configured focus list."""
        if not self._focus_modules:
            return True
        normalized = self._normalize_module_name(name)
        return normalized in self._focus_modules

    def _materialize_batch(self, batch: Any) -> Any:
        """Detach tensors from device and clone calibration batches for reuse."""
        if isinstance(batch, dict):
            return {key: self._materialize_batch(val) for key, val in batch.items()}
        if isinstance(batch, list | tuple):
            return type(batch)(self._materialize_batch(val) for val in batch)
        if isinstance(batch, torch.Tensor):
            return batch.detach().cpu()
        try:
            return copy.deepcopy(batch)
        except Exception:
            return batch

    def _ensure_tensor_value(self, value: Any) -> Any:
        """Convert common calibration value types to torch tensors."""
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, np.ndarray):
            return torch.as_tensor(value)
        if isinstance(value, list | tuple):
            try:
                return torch.as_tensor(value)
            except Exception:
                return value
        if isinstance(value, int | float):
            return torch.tensor(value)
        return value

    def _tensorize_calibration_batches(self, batches: Sequence[Any]) -> list[Any]:
        """Ensure calibration batches contain tensor payloads for model execution."""
        tensor_batches: list[Any] = []
        for batch in batches:
            if isinstance(batch, dict):
                converted: dict[str, Any] = {}
                for key, value in batch.items():
                    if key in {"input_ids", "inputs", "attention_mask", "labels"}:
                        converted[key] = self._ensure_tensor_value(value)
                    else:
                        converted[key] = value
                tensor_batches.append(converted)
            elif isinstance(batch, list | tuple):
                converted_list = [self._ensure_tensor_value(val) for val in batch]
                tensor_batches.append(type(batch)(converted_list))
            else:
                tensor_batches.append(self._ensure_tensor_value(batch))
        return tensor_batches

    def _extract_window_ids(self, batches: Sequence[Any]) -> list[str]:
        """Extract window identifiers from calibration batches when present."""
        window_ids: list[str] = []
        for batch in batches:
            candidate: Any | None = None
            if isinstance(batch, dict):
                if "window_id" in batch:
                    candidate = batch["window_id"]
                elif "window_ids" in batch:
                    candidate = batch["window_ids"]
                elif isinstance(batch.get("metadata"), dict):
                    meta = batch["metadata"]
                    candidate = meta.get("window_id") or meta.get("window_ids")

            if candidate is None:
                continue

            if isinstance(candidate, list | tuple):
                window_ids.extend(str(item) for item in candidate)
            else:
                window_ids.append(str(candidate))
        if not window_ids and batches:
            window_ids = [str(idx) for idx in range(len(batches))]
        return window_ids

    def _store_calibration_batches(self, batches: list[Any]) -> None:
        """Persist calibration batches for deterministic post-edit evaluation."""
        materialized = [self._materialize_batch(b) for b in batches]
        self._calibration_batches = self._tensorize_calibration_batches(materialized)
        self._calibration_window_ids = self._extract_window_ids(
            self._calibration_batches
        )
        observed_ids = list(self._calibration_window_ids)
        observed_digest = (
            hashlib.blake2s(
                "||".join(observed_ids).encode("utf-8"), digest_size=16
            ).hexdigest()
            if observed_ids
            else None
        )
        self._calibration_context = {
            "window_ids": list(self._calibration_window_ids),
            "count": len(self._calibration_batches),
            "observed_digest": observed_digest,
        }
        expected_ids = self._expected_window_ids()
        if expected_ids:
            self._calibration_context["expected_digest"] = self._pairing_digest
            expected_subset = expected_ids[: len(observed_ids)] if observed_ids else []
            if observed_ids != expected_subset:
                mismatch = {
                    "expected_count": len(expected_ids),
                    "observed_count": len(observed_ids),
                    "expected_sample": expected_subset[:5]
                    if expected_subset
                    else expected_ids[:5],
                    "observed_sample": observed_ids[:5],
                }
                self._log_event(
                    "pairing_mismatch",
                    level="ERROR",
                    message="Variance guard calibration windows do not match baseline pairing",
                    **mismatch,
                )
                self._prepared = False
                raise RuntimeError(
                    "Variance guard pairing mismatch: calibration windows diverge from baseline schedule"
                )
        self._stats.setdefault("calibration", {})
        self._stats["calibration"].update(self._calibration_context)

    def _fingerprint_targets(self) -> str | None:
        """Compute a lightweight fingerprint of targeted module weights."""
        if not self._target_modules:
            return None

        hasher = hashlib.sha256()
        try:
            for name in sorted(self._target_modules.keys()):
                module = self._target_modules[name]
                state = getattr(module, "state_dict", None)
                if not callable(state):
                    continue
                module_state = state()
                for key in sorted(module_state.keys()):
                    tensor = module_state[key]
                    if hasattr(tensor, "detach"):
                        data = tensor.detach().cpu().numpy().tobytes()
                    else:
                        data = bytes(str(tensor), "utf-8")
                    hasher.update(name.encode("utf-8"))
                    hasher.update(key.encode("utf-8"))
                    hasher.update(data)
            return hasher.hexdigest()[:16]
        except Exception:
            return None

    def _record_ab_provenance(
        self,
        condition: str,
        *,
        tag: str,
        window_ids: Sequence[str],
        fingerprint: str | None,
        mode: str,
        status: str,
    ) -> None:
        """Record provenance metadata for A/B evaluation conditions."""
        provenance = self._stats.setdefault("ab_provenance", {})
        window_list = list(window_ids)
        provenance[condition] = {
            "tag": tag,
            "mode": mode,
            "window_ids": window_list,
            "window_count": len(window_list),
            "target_fingerprint": fingerprint,
            "status": status,
            "pairing_digest": self._pairing_digest,
            "dataset_hash": (self._dataset_meta or {}).get("dataset_hash"),
            "tokenizer_hash": (self._dataset_meta or {}).get("tokenizer_hash"),
            "model_id": (self._report_meta or {}).get("model_id"),
            "run_seed": (self._report_meta or {}).get("seed"),
        }

    def _resolve_target_modules(
        self, model: nn.Module, adapter: Any | None = None
    ) -> dict[str, nn.Module]:
        """
        Resolve target modules based on scope policy.

        Args:
            model: Model to analyze
            adapter: Optional adapter used to query layer modules

        Returns:
            Dict mapping module names to modules
        """
        targets = {}
        scope = self._policy["scope"]
        audit_candidates: list[dict[str, Any]] = []
        audit_rejections: list[dict[str, Any]] = []

        def _record_match(name: str, module: nn.Module) -> None:
            audit_candidates.append(
                {
                    "name": name,
                    "class": module.__class__.__name__,
                    "source": "direct",
                }
            )

        def _record_rejection(name: str, reason: str, module: Any | None) -> None:
            audit_rejections.append(
                {
                    "name": name,
                    "reason": reason,
                    "class": getattr(module, "__class__", type(None)).__name__
                    if module is not None
                    else None,
                }
            )

        # Get module types
        try:
            from transformers.pytorch_utils import Conv1D

            module_types = (nn.Linear, nn.Conv1d, Conv1D)
        except ImportError:
            module_types = (nn.Linear, nn.Conv1d)

        def _is_supported_module(module: Any) -> bool:
            """Heuristic check that a module looks like a projection."""
            if isinstance(module, module_types):
                return True
            class_name = module.__class__.__name__ if module is not None else ""
            if class_name in {"Conv1D", "Linear"}:
                return True
            weight = getattr(module, "weight", None)
            if weight is None:
                return False
            try:
                dim = weight.dim()
            except Exception:
                dim = getattr(weight, "ndim", None)
            return dim == 2

        for i, blk in enumerate(_iter_transformer_layers(model)):
            # Handle attention projection based on scope
            if scope in ["attn", "both"] and hasattr(blk, "attn"):
                attn_proj = getattr(blk.attn, "c_proj", None) or getattr(
                    blk.attn, "out_proj", None
                )
                name = f"transformer.h.{i}.attn.c_proj"
                if attn_proj is None:
                    _record_rejection(name, "missing_module", None)
                elif not self._matches_tap(name):
                    _record_rejection(name, "tap_mismatch", attn_proj)
                elif not _is_supported_module(attn_proj):
                    _record_rejection(name, "unsupported_type", attn_proj)
                else:
                    targets[name] = attn_proj
                    _record_match(name, attn_proj)

            # Handle MLP projection based on scope
            if scope in ["ffn", "both"] and hasattr(blk, "mlp"):
                mlp_proj = (
                    getattr(blk.mlp, "c_proj", None)
                    or getattr(blk.mlp, "down_proj", None)
                    or getattr(blk.mlp, "fc2", None)
                )
                name = f"transformer.h.{i}.mlp.c_proj"
                if mlp_proj is None:
                    _record_rejection(name, "missing_module", None)
                elif not self._matches_tap(name):
                    _record_rejection(name, "tap_mismatch", mlp_proj)
                elif not _is_supported_module(mlp_proj):
                    _record_rejection(name, "unsupported_type", mlp_proj)
                else:
                    targets[name] = mlp_proj
                    _record_match(name, mlp_proj)

        fallback_used = False

        # Fallback: ask adapter for layer modules if we could not resolve anything
        # Strategy:
        # 1. Try adapter.describe() for layer count - works even when model structure is unknown
        # 2. If that fails, try _iter_transformer_layers() to count layers
        # 3. If that fails, try model.config for layer count
        if (
            not targets
            and adapter is not None
            and hasattr(adapter, "get_layer_modules")
        ):
            try:
                # Get layer count from adapter.describe() first
                n_layers = 0
                if hasattr(adapter, "describe"):
                    try:
                        desc = adapter.describe(model)
                        if isinstance(desc, dict):
                            n_layers = int(desc.get("n_layer", 0) or 0)
                    except Exception as desc_exc:
                        self._log_event(
                            "adapter_describe_error",
                            level="DEBUG",
                            message=f"adapter.describe() failed: {desc_exc}",
                        )

                # Fallback: count layers via _iter_transformer_layers()
                # This works when model has standard structure but no c_proj
                if n_layers == 0:
                    try:
                        n_layers = sum(1 for _ in _iter_transformer_layers(model))
                    except Exception:
                        pass

                # Fallback: try model.config for layer count
                if n_layers == 0:
                    config = getattr(_unwrap_model(model), "config", None)
                    if config is not None:
                        n_layers = (
                            getattr(config, "n_layer", 0)
                            or getattr(config, "num_hidden_layers", 0)
                            or getattr(config, "num_layers", 0)
                            or 0
                        )

                if n_layers == 0:
                    self._log_event(
                        "adapter_fallback_no_layers",
                        level="WARN",
                        message="Adapter fallback: could not determine layer count",
                    )

                for i in range(n_layers):
                    try:
                        modules = adapter.get_layer_modules(model, i) or {}
                    except Exception as exc:
                        _record_rejection(
                            f"transformer.h.{i}",
                            f"adapter_error:{exc}",
                            None,
                        )
                        continue

                    for key, module in modules.items():
                        if not isinstance(key, str) or not key.endswith("c_proj"):
                            continue
                        branch = "attn" if "attn" in key else "mlp"
                        name = f"transformer.h.{i}.{branch}.c_proj"
                        if not self._matches_tap(name):
                            _record_rejection(name, "tap_mismatch", module)
                            continue
                        if not _is_supported_module(module):
                            _record_rejection(name, "unsupported_type", module)
                            continue
                        targets[name] = module
                        audit_candidates.append(
                            {
                                "name": name,
                                "class": module.__class__.__name__,
                                "source": "adapter_fallback",
                            }
                        )
                if targets:
                    fallback_used = True
            except Exception as exc:  # pragma: no cover - defensive logging
                self._log_event(
                    "target_resolution_fallback_error",
                    level="WARN",
                    message="Adapter fallback failed during VE target resolution",
                    error=str(exc),
                )

        if self._focus_modules:
            focused: dict[str, nn.Module] = {}
            for name, module in targets.items():
                norm_name = self._normalize_module_name(name)
                if norm_name in self._focus_modules:
                    focused[name] = module

            if not focused:
                self._log_event(
                    "focus_miss",
                    level="WARN",
                    message="No target modules matched focus list",
                    focus_modules=sorted(self._focus_modules),
                    available=list(targets.keys()),
                )
            else:
                targets = focused

        # Persist audit statistics for reports
        rejected_summary: dict[str, Any] = {}
        for item in audit_rejections:
            reason = item["reason"]
            bucket = rejected_summary.setdefault(reason, {"count": 0, "examples": []})
            bucket["count"] += 1
            if len(bucket["examples"]) < 5:
                bucket["examples"].append(
                    {
                        "name": item["name"],
                        "class": item["class"],
                    }
                )

        self._stats["target_resolution"] = {
            "scope": scope,
            "tap": list(self._tap_patterns),
            "total_matched": len(targets),
            "matched": sorted(targets.keys()),
            "fallback_used": fallback_used,
            "candidates_recorded": len(audit_candidates),
            "rejected": rejected_summary,
        }

        self._log_event(
            "target_resolution",
            message="Resolved variance guard targets",
            scope=scope,
            tap=list(self._tap_patterns),
            matched=len(targets),
            rejected=sum(item["count"] for item in rejected_summary.values())
            if rejected_summary
            else 0,
            fallback_used=fallback_used,
        )

        return targets

    def _compute_variance_scales(
        self, model: nn.Module, dataloader
    ) -> dict[str, float]:
        """
        Compute variance-based scaling factors using existing implementation.

        Args:
            model: Model to analyze
            dataloader: Calibration data

        Returns:
            Dict mapping module names to proposed scaling factors
        """
        if self._monitor_only:
            self._log_event(
                "monitor_only",
                message="Skipping variance scale computation in monitor-only mode",
            )
            self._raw_scales = {}
            return {}

        # Use existing equalise_residual_variance but don't apply yet
        # We'll capture the proposed scales and apply them later in enable()

        # Temporarily capture the current model state
        original_state = copy.deepcopy(model.state_dict())

        try:
            tensor_ready_batches = self._tensorize_calibration_batches(dataloader)

            # Run variance equalization to get proposed scales
            proposed_scales = equalise_residual_variance(
                model=model,
                dataloader=tensor_ready_batches,
                windows=min(
                    self._policy["max_calib"] // 10, 50
                ),  # Limit calibration windows
                tol=self._policy["deadband"],
                scale_bias=False,  # Don't scale biases to preserve operating points
                seed=self._policy["seed"],
                clamp_range=self._policy["clamp"],
                allow_empty=True,
            )

            if not proposed_scales and self._policy.get("deadband", 0.0) > 0.0:
                relaxed_tol = max(self._policy["deadband"] * 0.5, 1e-4)
                model.load_state_dict(original_state)
                tensor_ready_batches = self._tensorize_calibration_batches(dataloader)
                proposed_scales = equalise_residual_variance(
                    model=model,
                    dataloader=tensor_ready_batches,
                    windows=min(self._policy["max_calib"] // 10, 50),
                    tol=relaxed_tol,
                    scale_bias=False,
                    seed=self._policy["seed"] + 7,
                    clamp_range=self._policy["clamp"],
                    allow_empty=True,
                )

            raw_scales = dict(proposed_scales)

            # Filter raw_scales to only those that have corresponding target modules
            # This is critical when scope limits targets (e.g., scope=ffn only has mlp targets)
            # Only apply this filtering when target modules have been resolved
            if self._target_modules:
                filtered_raw_scales: dict[str, float] = {}
                for scale_name, scale_value in raw_scales.items():
                    # Convert scale name to target module name format
                    target_name = self._normalize_scale_name(scale_name)
                    if target_name in self._target_modules:
                        filtered_raw_scales[scale_name] = scale_value
                    elif self._is_focus_match(scale_name):
                        # Fallback: check if any target module matches via pattern
                        for tm_name in self._target_modules:
                            if self._scale_matches_target(scale_name, tm_name):
                                filtered_raw_scales[scale_name] = scale_value
                                break
                raw_scales = filtered_raw_scales

            focus_raw_scales = {
                self._normalize_scale_name(name): scale
                for name, scale in raw_scales.items()
                if self._is_focus_match(name)
            }
            if focus_raw_scales:
                self._log_event(
                    "variance_raw_scales",
                    message="Captured raw VE scales",
                    count=len(focus_raw_scales),
                    min_scale=min(focus_raw_scales.values()),
                    max_scale=max(focus_raw_scales.values()),
                )
            self._stats.setdefault("raw_scales_observations", []).append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "count": len(focus_raw_scales),
                    "scales": focus_raw_scales,
                }
            )

            # Restore original state since we only wanted the proposed scales
            model.load_state_dict(original_state)

            filtered_scales: dict[str, float] = {}
            raw_delta_map: dict[str, float] = {}
            min_abs = float(max(self._policy.get("min_abs_adjust", 0.0), 0.0))
            max_step = float(max(self._policy.get("max_scale_step", 0.0), 0.0))
            topk = int(max(self._policy.get("topk_backstop", 0) or 0, 0))
            best_candidate: tuple[str, float] | None = None
            best_delta = 0.0

            for name, scale in raw_scales.items():
                normalized_name = self._normalize_scale_name(name)
                if not self._is_focus_match(normalized_name):
                    continue

                raw_delta = abs(scale - 1.0)
                raw_delta_map[name] = raw_delta

                delta = raw_delta
                if delta > best_delta:
                    best_candidate = (name, scale)
                    best_delta = delta

                if delta < min_abs:
                    continue

                if max_step > 0.0:
                    limited_delta = min(delta, max_step)
                    scale = 1.0 + math.copysign(limited_delta, scale - 1.0)

                filtered_scales[name] = scale

            backstop_used = False
            if not filtered_scales and topk > 0 and best_candidate:
                name, scale = best_candidate
                deadband = float(self._policy.get("deadband", 0.0) or 0.0)
                threshold = max(deadband * 0.5, min_abs)
                if best_delta >= threshold:
                    if max_step > 0.0:
                        limited_delta = min(best_delta, max_step)
                        scale = 1.0 + math.copysign(limited_delta, scale - 1.0)
                    filtered_scales[name] = scale
                    raw_delta_map.setdefault(name, best_delta)
                    backstop_used = True

            trimmed_to_limit = False
            max_adjusted = int(max(self._policy.get("max_adjusted_modules", 0) or 0, 0))
            if max_adjusted > 0 and len(filtered_scales) > max_adjusted:
                sorted_candidates = sorted(
                    filtered_scales.items(),
                    key=lambda item: (
                        raw_delta_map.get(item[0], abs(item[1] - 1.0))
                        + (2.0 if item[1] >= 1.0 else 0.0),
                        raw_delta_map.get(item[0], abs(item[1] - 1.0)),
                        item[1],
                    ),
                    reverse=True,
                )
                filtered_scales = dict(sorted_candidates[:max_adjusted])
                trimmed_to_limit = True

            self._raw_scales = raw_scales
            if backstop_used:
                self._log_event(
                    "scale_backstop",
                    message=f"Top-{topk} backstop injected {len(filtered_scales)} scale",
                    count=len(filtered_scales),
                    candidate=best_candidate[0] if best_candidate else None,
                    candidate_normalized=self._normalize_scale_name(best_candidate[0])
                    if best_candidate
                    else None,
                    delta=best_delta,
                )
            if trimmed_to_limit:
                self._log_event(
                    "scale_limit",
                    message="Trimmed VE scales to max_adjusted_modules",
                    limit=max_adjusted,
                    count=len(filtered_scales),
                )

            filtered_normalized = {
                self._normalize_scale_name(name): scale
                for name, scale in filtered_scales.items()
            }
            self._stats.setdefault("filtered_scales_observations", []).append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "count": len(filtered_normalized),
                    "scales": filtered_normalized,
                    "backstop_used": backstop_used,
                }
            )

            return filtered_scales

        except Exception as e:
            # Restore state on any error
            model.load_state_dict(original_state)
            raise e

    def _evaluate_calibration_pass(
        self,
        model: nn.Module,
        calibration_batches: list[Any],
        min_coverage: int,
        calib_seed: int,
        tag: str,
    ) -> None:
        """Run deterministic calibration for A/B evaluation and predictive gating."""
        predictive_state: dict[str, Any] = {
            "evaluated": False,
            "passed": not bool(self._policy.get("predictive_gate", True)),
            "reason": "disabled"
            if not bool(self._policy.get("predictive_gate", True))
            else "no_calibration",
            "delta_ci": (None, None),
            "gain_ci": (None, None),
            "mean_delta": None,
        }

        requested = len(calibration_batches)
        self._calibration_stats.update(
            {
                "requested": requested,
                "coverage": 0,
                "min_coverage": min_coverage,
                "seed": calib_seed,
                "status": "no_calibration"
                if not calibration_batches
                else "insufficient",
                "tag": tag,
            }
        )
        self._stats.setdefault("calibration", {})
        self._stats["calibration"].update(
            {
                "requested": requested,
                "min_coverage": min_coverage,
                "seed": calib_seed,
                "tag": tag,
            }
        )

        fingerprint = self._fingerprint_targets()
        if fingerprint:
            self._stats["target_fingerprint"] = fingerprint

        if not calibration_batches:
            self._ratio_ci = None
            self._predictive_gate_state = predictive_state
            self._stats["predictive_gate"] = predictive_state.copy()
            return

        device = next(model.parameters()).device
        torch.manual_seed(calib_seed)
        ppl_no_ve_samples, loss_no_ve_samples = self._compute_ppl_for_batches(
            model, calibration_batches, device
        )
        coverage = min(len(calibration_batches), len(ppl_no_ve_samples))
        ppl_with_ve_samples: list[float] = []
        loss_with_ve_samples: list[float] = []
        ratio_ci: tuple[float, float] | None = None

        enable_success = False
        if coverage >= min_coverage and self._scales:
            prev_enable_attempts = self._enable_attempt_count
            prev_disable_attempts = self._disable_attempt_count
            prev_prepared_flag = self._prepared
            try:
                self._prepared = True
                enable_success = self.enable(model)
            finally:
                self._prepared = prev_prepared_flag
            try:
                torch.manual_seed(calib_seed)
                if enable_success:
                    ppl_with_ve_samples, loss_with_ve_samples = (
                        self._compute_ppl_for_batches(
                            model, calibration_batches, device
                        )
                    )
            finally:
                if enable_success:
                    self.disable(model)
            # Restore attempt counters to avoid skewing metrics
            self._enable_attempt_count = prev_enable_attempts
            self._disable_attempt_count = prev_disable_attempts

        coverage = min(
            coverage,
            len(ppl_with_ve_samples) if ppl_with_ve_samples else coverage,
            len(loss_with_ve_samples) if loss_with_ve_samples else coverage,
        )
        self._calibration_stats.update(
            {
                "coverage": coverage,
                "status": "insufficient" if coverage < min_coverage else "pending",
            }
        )

        window_ids = self._calibration_window_ids
        status_a = "evaluated" if coverage > 0 else "no_data"
        self._record_ab_provenance(
            "condition_a",
            tag=tag,
            mode="edited_no_ve",
            window_ids=window_ids,
            fingerprint=fingerprint,
            status=status_a,
        )

        if coverage >= min_coverage and not self._scales:
            ppl_no_ve_samples = ppl_no_ve_samples[:coverage]
            ppl_no_ve_mean = _safe_mean(ppl_no_ve_samples)
            if ppl_no_ve_mean is None:
                # No valid samples - cannot compute mean
                self._ratio_ci = None
                predictive_state["reason"] = "no_valid_samples"
                self._predictive_gate_state = predictive_state
                self._stats["predictive_gate"] = predictive_state.copy()
                return
            self.set_ab_results(
                ppl_no_ve=ppl_no_ve_mean,
                ppl_with_ve=ppl_no_ve_mean,
                windows_used=coverage,
                seed_used=calib_seed,
                ratio_ci=(1.0, 1.0),
            )
            self._calibration_stats.update(
                {
                    "status": "no_scaling_required",
                    "ppl_no_ve": ppl_no_ve_mean,
                    "ratio_ci": (1.0, 1.0),
                }
            )
            self._stats["ab_point_estimates"] = {
                "tag": tag,
                "ppl_no_ve": ppl_no_ve_mean,
                "ppl_with_ve": ppl_no_ve_mean,
            }
            self._record_ab_provenance(
                "condition_b",
                tag=tag,
                mode="virtual_ve",
                window_ids=window_ids,
                fingerprint=fingerprint,
                status="no_scales",
            )
            predictive_state["evaluated"] = True
            predictive_state["passed"] = False
            predictive_state["reason"] = "no_scales"
            self._predictive_gate_state = predictive_state
            self._stats["predictive_gate"] = predictive_state.copy()
            return

        if coverage >= min_coverage and ppl_with_ve_samples and loss_with_ve_samples:
            ppl_no_ve_samples = ppl_no_ve_samples[:coverage]
            loss_no_ve_samples = loss_no_ve_samples[:coverage]
            ppl_with_ve_samples = ppl_with_ve_samples[:coverage]
            loss_with_ve_samples = loss_with_ve_samples[:coverage]

            ratios = [
                with_val / no_val
                for with_val, no_val in zip(
                    ppl_with_ve_samples, ppl_no_ve_samples, strict=False
                )
                if no_val > 0
            ]
            if ratios:
                ratio_ci = self._bootstrap_mean_ci(
                    ratios,
                    alpha=self._policy.get("alpha", 0.05),
                    n_bootstrap=500,
                    seed=calib_seed,
                )
                ppl_no_ve_mean = _safe_mean(ppl_no_ve_samples)
                ppl_with_ve_mean = _safe_mean(ppl_with_ve_samples)
                if ppl_no_ve_mean is None or ppl_with_ve_mean is None:
                    # Fallback if means couldn't be computed
                    ppl_no_ve_mean = ppl_no_ve_mean or 0.0
                    ppl_with_ve_mean = ppl_with_ve_mean or 0.0
                self.set_ab_results(
                    ppl_no_ve=ppl_no_ve_mean,
                    ppl_with_ve=ppl_with_ve_mean,
                    windows_used=coverage,
                    seed_used=calib_seed,
                    ratio_ci=ratio_ci,
                )
                self._calibration_stats.update(
                    {
                        "status": "complete",
                        "ppl_no_ve": ppl_no_ve_mean,
                        "ppl_with_ve": ppl_with_ve_mean,
                        "ratio_ci": ratio_ci,
                    }
                )
                self._record_ab_provenance(
                    "condition_b",
                    tag=tag,
                    mode="virtual_ve",
                    window_ids=window_ids,
                    fingerprint=fingerprint,
                    status="evaluated",
                )
                self._stats["ab_point_estimates"] = {
                    "tag": tag,
                    "ppl_no_ve": ppl_no_ve_mean,
                    "ppl_with_ve": ppl_with_ve_mean,
                    "coverage": coverage,
                }

            delta_ci: tuple[float, float] | None = None
            try:
                delta_ci = compute_paired_delta_log_ci(
                    loss_with_ve_samples,
                    loss_no_ve_samples,
                    method="bca",
                    replicates=500,
                    alpha=self._policy.get("alpha", 0.05),
                    seed=calib_seed + 211,
                )
            except Exception as exc:
                delta_ci = None
                self._log_event(
                    "predictive_gate_error",
                    level="WARN",
                    message="Failed to compute predictive ΔlogNLL CI",
                    error=str(exc),
                )

            predictive_state["evaluated"] = True
            mean_delta = float(
                np.mean(
                    [
                        with_loss - no_loss
                        for with_loss, no_loss in zip(
                            loss_with_ve_samples,
                            loss_no_ve_samples,
                            strict=False,
                        )
                    ]
                )
            )
            predictive_state["mean_delta"] = mean_delta

            if delta_ci is not None and all(
                isinstance(val, (int | float)) and math.isfinite(val)
                for val in delta_ci
            ):
                delta_ci = (float(delta_ci[0]), float(delta_ci[1]))
                gain_ci = (-delta_ci[1], -delta_ci[0])
                predictive_state["delta_ci"] = delta_ci
                predictive_state["gain_ci"] = gain_ci

                if not self._policy.get("predictive_gate", True):
                    predictive_state["passed"] = True
                    predictive_state["reason"] = "disabled"
                else:
                    one_sided = bool(self._policy.get("predictive_one_sided", False))
                    min_effect = float(
                        self._policy.get("min_effect_lognll", 0.0) or 0.0
                    )
                    passed, reason = _predictive_gate_outcome(
                        mean_delta=mean_delta,
                        delta_ci=delta_ci,
                        min_effect=min_effect,
                        one_sided=one_sided,
                    )
                    predictive_state["passed"] = passed
                    predictive_state["reason"] = reason
            else:
                predictive_state["delta_ci"] = (None, None)
                predictive_state["gain_ci"] = (None, None)
                predictive_state["reason"] = (
                    predictive_state.get("reason", "ci_unavailable")
                    if predictive_state.get("reason") != "disabled"
                    else "disabled"
                )
        else:
            # Fail-open monitor mode
            self._ratio_ci = None
            self._log_event(
                "prepare_monitor_mode",
                level="WARN",
                message="VE calibration coverage insufficient; guard will monitor only",
                requested=requested,
                coverage=coverage,
                min_coverage=min_coverage,
                tag=tag,
            )
            if predictive_state.get("reason") not in {"disabled"}:
                if coverage < min_coverage:
                    predictive_state["reason"] = "insufficient_coverage"
                elif not self._scales:
                    predictive_state["reason"] = "no_scales"
                elif not ppl_with_ve_samples:
                    predictive_state["reason"] = "ve_enable_failed"

            if "condition_b" not in self._stats.get("ab_provenance", {}):
                self._record_ab_provenance(
                    "condition_b",
                    tag=tag,
                    mode="virtual_ve",
                    window_ids=window_ids,
                    fingerprint=fingerprint,
                    status="not_evaluated",
                )

        if (
            "ab_point_estimates" not in self._stats
            or self._stats["ab_point_estimates"].get("tag") != tag
        ):
            ppl_no_ve_mean = (
                float(np.mean(ppl_no_ve_samples[:coverage])) if coverage > 0 else None
            )
            ppl_with_ve_mean = (
                float(np.mean(ppl_with_ve_samples[:coverage]))
                if ppl_with_ve_samples and coverage > 0
                else None
            )
            self._stats["ab_point_estimates"] = {
                "tag": tag,
                "ppl_no_ve": ppl_no_ve_mean,
                "ppl_with_ve": ppl_with_ve_mean,
                "coverage": coverage,
            }

        self._predictive_gate_state = predictive_state
        self._stats["predictive_gate"] = predictive_state.copy()

    def _refresh_after_edit_metrics(
        self,
        model: nn.Module,
        tag: str = "post_edit",
        adapter: Any | None = None,
    ) -> None:
        """Ensure VE metrics are recomputed on the edited model."""
        if not self._prepared:
            return
        if self._post_edit_evaluated and tag == "post_edit":
            return
        if not self._calibration_batches:
            self._log_event(
                "post_edit_calibration_skipped",
                level="WARN",
                message="Skipping post-edit VE evaluation (no calibration batches)",
            )
            self._post_edit_evaluated = True
            return

        # Refresh target modules in case adapters swapped modules during edit
        adapter_ref = adapter or self._adapter_ref
        self._target_modules = self._resolve_target_modules(model, adapter_ref)
        self._stats["target_module_names"] = sorted(self._target_modules.keys())

        # Recompute scales against the edited model
        try:
            self._scales = self._compute_variance_scales(
                model, self._calibration_batches
            )
        except Exception as exc:
            self._log_event(
                "post_edit_scale_failure",
                level="ERROR",
                message="Failed to recompute VE scales after edit",
                error=str(exc),
            )
            self._scales = {}

        if self._focus_modules:
            self._scales = {
                name: scale
                for name, scale in self._scales.items()
                if self._is_focus_match(name)
            }

        self._stats.setdefault(
            "target_module_names", sorted(self._target_modules.keys())
        )
        self._stats["target_modules_post_edit"] = list(self._target_modules.keys())
        normalized_post_scales = {
            self._normalize_scale_name(name): scale
            for name, scale in self._scales.items()
        }
        self._stats["proposed_scales_post_edit"] = normalized_post_scales.copy()
        self._stats["raw_scales_post_edit"] = self._raw_scales.copy()
        self._stats["raw_scales_post_edit_normalized"] = {
            self._normalize_scale_name(name): scale
            for name, scale in self._raw_scales.items()
        }
        self._raw_scales_post_edit = {
            self._normalize_scale_name(name): scale
            for name, scale in self._raw_scales.items()
            if self._is_focus_match(name)
        }
        if normalized_post_scales:
            self._log_event(
                "post_edit_scales",
                message="Post-edit VE proposed scales",
                count=len(normalized_post_scales),
                min_scale=min(normalized_post_scales.values()),
                max_scale=max(normalized_post_scales.values()),
            )

        calibration_cfg = self._policy.get("calibration", {})
        requested_windows = int(calibration_cfg.get("windows", 0) or 0)
        min_coverage = int(
            calibration_cfg.get(
                "min_coverage",
                max(1, requested_windows // 2 if requested_windows else 1),
            )
        )
        calib_seed = int(calibration_cfg.get("seed", self._policy.get("seed", 123)))

        self._calibration_stats = {
            "requested": len(self._calibration_batches)
            if requested_windows == 0
            else requested_windows,
            "coverage": 0,
            "min_coverage": min_coverage,
            "seed": calib_seed,
            "status": "pending",
            "tag": tag,
        }

        self._evaluate_calibration_pass(
            model, self._calibration_batches, min_coverage, calib_seed, tag
        )
        self._post_edit_evaluated = True

    def _collect_calibration_batches(self, dataloader, windows: int) -> list[Any]:
        """Collect a deterministic slice of calibration batches."""
        batches: list[Any] = []
        iterator = iter(dataloader)
        for _ in range(max(windows, 0)):
            try:
                batches.append(next(iterator))
            except StopIteration:
                break
        return batches

    def _prepare_batch_tensors(
        self, batch: Any, device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Normalize batch inputs to tensors on the target device."""
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids", batch.get("inputs"))
            attention_mask = batch.get("attention_mask")
        elif isinstance(batch, tuple | list) and batch:
            input_ids = batch[0]
            attention_mask = batch[1] if len(batch) > 1 else None
        else:
            input_ids = batch
            attention_mask = None

        if input_ids is None:
            return None, None

        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.as_tensor(input_ids)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        try:
            input_ids = input_ids.to(device)
        except Exception:
            input_ids = input_ids.clone()

        labels = input_ids.clone()

        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.as_tensor(attention_mask)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            try:
                attention_mask = attention_mask.to(device)
            except Exception:
                attention_mask = attention_mask.clone()
            labels = labels.masked_fill(attention_mask == 0, -100)

        return input_ids, labels

    def _compute_ppl_for_batches(
        self,
        model: nn.Module,
        batches: list[Any],
        device: torch.device,
    ) -> tuple[list[float], list[float]]:
        """Compute per-batch perplexity and log-loss values for deterministic calibration."""
        ppl_values: list[float] = []
        loss_values: list[float] = []
        if not batches:
            return ppl_values, loss_values

        model_was_training = model.training
        model.eval()

        with torch.no_grad():
            for batch in batches:
                try:
                    inputs, labels = self._prepare_batch_tensors(batch, device)
                    if inputs is None or labels is None:
                        continue

                    try:
                        outputs = model(inputs, labels=labels)
                    except TypeError:
                        outputs = model(inputs)
                    loss_val = None
                    if hasattr(outputs, "loss") and hasattr(outputs.loss, "item"):
                        loss_val = outputs.loss.item()

                    if loss_val is None and isinstance(outputs, torch.Tensor):
                        try:
                            if labels is not None and outputs.shape == labels.shape:
                                loss_val = torch.nn.functional.mse_loss(
                                    outputs.float(), labels.float()
                                ).item()
                            else:
                                loss_val = outputs.float().pow(2).mean().item()
                        except Exception:
                            loss_val = None

                    if loss_val is None or not math.isfinite(loss_val):
                        continue

                    loss = float(loss_val)
                    ppl = math.exp(loss)
                    if math.isfinite(ppl):
                        ppl_values.append(ppl)
                        loss_values.append(loss)
                except Exception:
                    continue

        if model_was_training:
            model.train()

        return ppl_values, loss_values

    def _bootstrap_mean_ci(
        self,
        samples: list[float],
        alpha: float,
        n_bootstrap: int = 500,
        seed: int | None = None,
    ) -> tuple[float, float]:
        """Compute bootstrap confidence interval for the sample mean."""
        if not samples:
            raise ValueError("Cannot compute CI on empty samples")
        data = np.asarray(samples, dtype=float)
        rng = np.random.default_rng(seed)
        stats = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            indices = rng.integers(0, data.size, size=data.size)
            stats[i] = float(np.mean(data[indices]))
        lower = float(np.percentile(stats, 100 * (alpha / 2)))
        upper = float(np.percentile(stats, 100 * (1 - alpha / 2)))
        return lower, upper

    def prepare(
        self,
        model: nn.Module,
        adapter=None,
        calib=None,
        policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Prepare variance guard by computing proposed scaling factors.

        Args:
            model: The model that will be edited
            adapter: ModelAdapter (optional, for compatibility)
            calib: Calibration data for variance measurement
            policy: Guard policy parameters (optional)

        Returns:
            Dictionary with preparation results and proposed scales
        """
        start_time = time.time()

        # Update policy if provided
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
                    self._policy[key] = policy[key]
            if self._policy.get("min_effect_lognll") is not None:
                self._policy["min_effect_lognll"] = float(
                    self._policy["min_effect_lognll"]
                )
            self.TIE_BREAKER_DEADBAND = float(
                self._policy.get("tie_breaker_deadband", self.TIE_BREAKER_DEADBAND)
            )
            self._refresh_calibration_defaults()
            if "absolute_floor_ppl" in policy:
                self.ABSOLUTE_FLOOR = float(
                    self._policy.get(
                        "absolute_floor_pm",
                        self._policy.get("absolute_floor_ppl", self.ABSOLUTE_FLOOR),
                    )
                )
            if "target_modules" in policy:
                focus_list = [
                    self._normalize_module_name(name)
                    for name in (policy.get("target_modules") or [])
                    if isinstance(name, str)
                ]
                self._focus_modules = set(focus_list)
                if self._focus_modules:
                    self._policy["target_modules"] = sorted(self._focus_modules)
                    self._stats["focus_modules"] = sorted(self._focus_modules)

        self._log_event(
            "prepare",
            message=f"Preparing variance guard with scope={self._policy['scope']}, min_gain={self._policy['min_gain']}",
        )

        try:
            # Resolve target modules
            self._target_modules = self._resolve_target_modules(model, adapter)
            self._stats["target_module_names"] = sorted(self._target_modules.keys())

            if not self._target_modules:
                self._prepared = False
                self._adapter_ref = adapter
                return {
                    "baseline_metrics": {},
                    "policy_applied": self._policy,
                    "preparation_time": time.time() - start_time,
                    "ready": False,
                    "warning": "No target modules found for variance equalization",
                }

            self._adapter_ref = adapter

            calibration_cfg = self._policy.get("calibration", {})
            requested_windows = int(calibration_cfg.get("windows", 0) or 0)
            min_coverage = int(
                calibration_cfg.get(
                    "min_coverage",
                    max(1, requested_windows // 2 if requested_windows else 1),
                )
            )
            calib_seed = int(calibration_cfg.get("seed", self._policy.get("seed", 123)))

            scale_windows = min(self._policy["max_calib"] // 10, 50)
            limit_for_batches = max(scale_windows, requested_windows)

            calib_batches: list[Any] = []
            dataloader_source = None

            if calib is not None:
                if hasattr(calib, "dataloader"):
                    dataloader_source = calib.dataloader
                    calib_batches = self._collect_calibration_batches(
                        dataloader_source, limit_for_batches
                    )
                elif isinstance(calib, Sequence):
                    calib_batches = list(
                        itertools.islice(iter(calib), limit_for_batches)
                    )
                elif isinstance(calib, Iterable):
                    calib_batches = list(
                        itertools.islice(iter(calib), limit_for_batches)
                    )

            if calib_batches:
                self._scales = self._compute_variance_scales(model, calib_batches)
            else:
                self._scales = {}
                self._raw_scales = {}
                self._log_event(
                    "prepare_warning",
                    level="WARN",
                    message="No calibration data provided, VE will be disabled",
                )

            # Deterministic VE calibration pass for A/B readiness
            self._calibration_stats = {
                "requested": requested_windows,
                "coverage": 0,
                "min_coverage": min_coverage,
                "seed": calib_seed,
                "status": "skipped" if requested_windows == 0 else "insufficient",
            }

            calibration_batches = calib_batches[:requested_windows]
            self._store_calibration_batches(calibration_batches)
            predictive_state: dict[str, Any] = {
                "evaluated": False,
                "passed": not bool(self._policy.get("predictive_gate", True)),
                "reason": "disabled"
                if not bool(self._policy.get("predictive_gate", True))
                else "no_calibration",
                "delta_ci": (None, None),
                "gain_ci": (None, None),
                "mean_delta": None,
            }

            if calibration_batches:
                device = next(model.parameters()).device
                torch.manual_seed(calib_seed)
                ppl_no_ve_samples, loss_no_ve_samples = self._compute_ppl_for_batches(
                    model, calibration_batches, device
                )
                coverage = min(len(calibration_batches), len(ppl_no_ve_samples))
                ppl_with_ve_samples: list[float] = []
                loss_with_ve_samples: list[float] = []
                ratio_ci: tuple[float, float] | None = None

                enable_success = False
                if coverage >= min_coverage and self._scales:
                    prev_enable_attempts = self._enable_attempt_count
                    prev_disable_attempts = self._disable_attempt_count
                    prev_prepared_flag = self._prepared
                    try:
                        self._prepared = True
                        enable_success = self.enable(model)
                    finally:
                        self._prepared = prev_prepared_flag
                    try:
                        torch.manual_seed(calib_seed)
                        if enable_success:
                            (
                                ppl_with_ve_samples,
                                loss_with_ve_samples,
                            ) = self._compute_ppl_for_batches(
                                model, calibration_batches, device
                            )
                    finally:
                        if enable_success:
                            self.disable(model)
                    # Restore attempt counters to avoid skewing metrics
                    self._enable_attempt_count = prev_enable_attempts
                    self._disable_attempt_count = prev_disable_attempts

                coverage = min(
                    coverage,
                    len(ppl_with_ve_samples) if ppl_with_ve_samples else coverage,
                    len(loss_with_ve_samples) if loss_with_ve_samples else coverage,
                )
                self._calibration_stats.update(
                    {"coverage": coverage, "status": "insufficient"}
                )

                if coverage >= min_coverage and not self._scales:
                    ppl_no_ve_samples = ppl_no_ve_samples[:coverage]
                    ppl_no_ve_mean = _safe_mean(ppl_no_ve_samples, default=0.0)
                    self.set_ab_results(
                        ppl_no_ve=ppl_no_ve_mean,
                        ppl_with_ve=ppl_no_ve_mean,
                        windows_used=coverage,
                        seed_used=calib_seed,
                        ratio_ci=(1.0, 1.0),
                    )
                    self._calibration_stats.update(
                        {
                            "status": "no_scaling_required",
                            "ppl_no_ve": ppl_no_ve_mean,
                            "ratio_ci": (1.0, 1.0),
                        }
                    )

                if (
                    coverage >= min_coverage
                    and ppl_with_ve_samples
                    and loss_with_ve_samples
                ):
                    ppl_no_ve_samples = ppl_no_ve_samples[:coverage]
                    loss_no_ve_samples = loss_no_ve_samples[:coverage]
                    ppl_with_ve_samples = ppl_with_ve_samples[:coverage]
                    loss_with_ve_samples = loss_with_ve_samples[:coverage]

                    ratios = [
                        with_val / no_val
                        for with_val, no_val in zip(
                            ppl_with_ve_samples, ppl_no_ve_samples, strict=False
                        )
                        if no_val > 0
                    ]
                    if ratios:
                        ratio_ci = self._bootstrap_mean_ci(
                            ratios,
                            alpha=self._policy.get("alpha", 0.05),
                            n_bootstrap=500,
                            seed=calib_seed,
                        )
                        ppl_no_ve_mean = _safe_mean(ppl_no_ve_samples, default=0.0)
                        ppl_with_ve_mean = _safe_mean(ppl_with_ve_samples, default=0.0)
                        self.set_ab_results(
                            ppl_no_ve=ppl_no_ve_mean,
                            ppl_with_ve=ppl_with_ve_mean,
                            windows_used=coverage,
                            seed_used=calib_seed,
                            ratio_ci=ratio_ci,
                        )
                        self._calibration_stats.update(
                            {
                                "status": "complete",
                                "ppl_no_ve": ppl_no_ve_mean,
                                "ppl_with_ve": ppl_with_ve_mean,
                                "ratio_ci": ratio_ci,
                            }
                        )

                    delta_ci: tuple[float, float] | None = None
                    try:
                        delta_ci = compute_paired_delta_log_ci(
                            loss_with_ve_samples,
                            loss_no_ve_samples,
                            method="bca",
                            replicates=500,
                            alpha=self._policy.get("alpha", 0.05),
                            seed=calib_seed + 211,
                        )
                    except Exception as exc:
                        delta_ci = None
                        self._log_event(
                            "predictive_gate_error",
                            level="WARN",
                            message="Failed to compute predictive ΔlogNLL CI",
                            error=str(exc),
                        )

                    predictive_state["evaluated"] = True
                    mean_delta = float(
                        np.mean(
                            [
                                with_loss - no_loss
                                for with_loss, no_loss in zip(
                                    loss_with_ve_samples,
                                    loss_no_ve_samples,
                                    strict=False,
                                )
                            ]
                        )
                    )
                    predictive_state["mean_delta"] = mean_delta

                    if delta_ci is not None and all(
                        isinstance(val, (int | float)) and math.isfinite(val)
                        for val in delta_ci
                    ):
                        delta_ci = (float(delta_ci[0]), float(delta_ci[1]))
                        gain_ci = (-delta_ci[1], -delta_ci[0])
                        predictive_state["delta_ci"] = delta_ci
                        predictive_state["gain_ci"] = gain_ci

                        if not self._policy.get("predictive_gate", True):
                            predictive_state["passed"] = True
                            predictive_state["reason"] = "disabled"
                        else:
                            one_sided = bool(
                                self._policy.get("predictive_one_sided", False)
                            )
                            min_effect = float(
                                self._policy.get("min_effect_lognll", 0.0) or 0.0
                            )
                            passed, reason = _predictive_gate_outcome(
                                mean_delta=mean_delta,
                                delta_ci=delta_ci,
                                min_effect=min_effect,
                                one_sided=one_sided,
                            )
                            predictive_state["passed"] = passed
                            predictive_state["reason"] = reason
                    else:
                        predictive_state["delta_ci"] = (None, None)
                        predictive_state["gain_ci"] = (None, None)
                        predictive_state["reason"] = (
                            predictive_state.get("reason", "ci_unavailable")
                            if predictive_state.get("reason") != "disabled"
                            else "disabled"
                        )
                else:
                    # Fail-open monitor mode
                    self._ratio_ci = None
                    self._log_event(
                        "prepare_monitor_mode",
                        level="WARN",
                        message="VE calibration coverage insufficient; guard will monitor only",
                        requested=requested_windows,
                        coverage=coverage,
                        min_coverage=min_coverage,
                    )
                    if predictive_state.get("reason") not in {"disabled"}:
                        if coverage < min_coverage:
                            predictive_state["reason"] = "insufficient_coverage"
                        elif not self._scales:
                            predictive_state["reason"] = "no_scales"
                        elif not ppl_with_ve_samples:
                            predictive_state["reason"] = "ve_enable_failed"
            else:
                self._ratio_ci = None
                if predictive_state.get("reason") != "disabled":
                    predictive_state["reason"] = "no_calibration"

            self._predictive_gate_state = predictive_state

            # Store baseline statistics without overwriting pre-populated instrumentation
            self._stats.setdefault(
                "target_module_names", sorted(self._target_modules.keys())
            )
            self._stats["target_modules"] = list(self._target_modules.keys())
            normalized_scales = {
                self._normalize_scale_name(name): scale
                for name, scale in self._scales.items()
            }
            self._stats["proposed_scales_pre_edit"] = normalized_scales.copy()
            self._stats["raw_scales_pre_edit"] = self._raw_scales.copy()
            self._stats["raw_scales_pre_edit_normalized"] = {
                self._normalize_scale_name(name): scale
                for name, scale in self._raw_scales.items()
            }
            self._stats["total_target_modules"] = len(self._target_modules)
            self._stats["modules_with_scales_pre_edit"] = len(self._scales)
            self._stats.setdefault("calibration", {}).update(
                self._calibration_stats.copy()
            )
            self._stats["scale_filtering"] = {
                "raw_scales": len(self._raw_scales),
                "filtered_scales": len(self._scales),
                "min_abs_adjust": float(self._policy.get("min_abs_adjust", 0.0)),
                "max_scale_step": float(self._policy.get("max_scale_step", 0.0)),
                "topk_backstop": int(self._policy.get("topk_backstop", 0)),
            }
            self._stats["predictive_gate"] = predictive_state.copy()
            self._calibration_stats_pre_edit = self._calibration_stats.copy()
            self._post_edit_evaluated = False
            self._raw_scales_pre_edit = {
                self._normalize_scale_name(name): scale
                for name, scale in self._raw_scales.items()
            }

            self._prepared = True
            preparation_time = time.time() - start_time

            self._log_event(
                "prepare_success",
                message=f"Prepared variance guard with {len(self._target_modules)} target modules",
                target_modules=len(self._target_modules),
                proposed_scales=len(self._scales),
                preparation_time=preparation_time,
            )

            return {
                "baseline_metrics": {
                    "target_modules": len(self._target_modules),
                    "proposed_scales": len(self._scales),
                    "scope": self._policy["scope"],
                    "scale_statistics": {
                        "mean_scale": float(
                            sum(self._scales.values()) / len(self._scales)
                        )
                        if self._scales
                        else 1.0,
                        "min_scale": min(self._scales.values())
                        if self._scales
                        else 1.0,
                        "max_scale": max(self._scales.values())
                        if self._scales
                        else 1.0,
                    },
                    "calibration": self._calibration_stats.copy(),
                },
                "policy_applied": self._policy.copy(),
                "preparation_time": preparation_time,
                "ready": True,
            }

        except Exception as e:
            self._prepared = False
            self._adapter_ref = adapter
            self._log_event(
                "prepare_failed",
                level="ERROR",
                message=f"Failed to prepare variance guard: {str(e)}",
                error=str(e),
            )

            return {
                "baseline_metrics": {},
                "policy_applied": self._policy,
                "preparation_time": time.time() - start_time,
                "ready": False,
                "error": str(e),
            }

    def before_edit(self, model: nn.Module) -> None:
        """
        Execute before edit (no action needed for variance guard).

        Args:
            model: The model about to be edited
        """
        if self._prepared:
            self._log_event(
                "before_edit", message="Variance guard ready for A/B testing"
            )

    def after_edit(self, model: nn.Module) -> None:
        """
        Execute after edit (A/B testing happens via enable/disable calls).

        Args:
            model: The model that was just edited
        """
        if not self._prepared:
            self._log_event(
                "after_edit_skipped",
                level="WARN",
                message="Variance guard not prepared, skipping",
            )
            return

        self._refresh_after_edit_metrics(model)
        self._log_event(
            "after_edit",
            message="Variance guard refreshed post-edit metrics",
            evaluated=self._post_edit_evaluated,
            proposed_scales=len(self._scales),
        )

    def enable(self, model: nn.Module, adapter=None) -> bool:
        """
        Enable variance equalization with checkpoint discipline and idempotent operation.

        Args:
            model: Model to apply VE to
            adapter: ModelAdapter (optional, for tying preservation)

        Returns:
            True if VE was successfully enabled, False otherwise
        """
        self._enable_attempt_count += 1

        if self._monitor_only:
            self._log_event(
                "enable_skipped_monitor_only",
                level="INFO",
                message="Monitor-only mode: VE enable skipped",
                attempt_count=self._enable_attempt_count,
            )
            self._enabled = False
            return False

        if not self._prepared or not self._scales:
            self._log_event(
                "enable_skipped",
                level="WARN",
                message="Cannot enable VE: not prepared or no scales computed",
                attempt_count=self._enable_attempt_count,
            )
            return False

        # Idempotent check: if already enabled, verify state and return success
        if self._enabled:
            self._log_event(
                "enable_idempotent",
                message="VE already enabled, verifying state",
                attempt_count=self._enable_attempt_count,
            )
            return True

        # Push checkpoint before attempting enable
        self._push_checkpoint(model)

        self._log_event(
            "enable_start",
            message=f"Enabling VE with {len(self._scales)} scale factors",
            attempt_count=self._enable_attempt_count,
        )

        try:
            # Apply scaling factors in-place with robust error handling
            applied_count = 0
            failed_modules = []

            for scale_name, scale_factor in self._scales.items():
                try:
                    # Find the actual module by matching scale name to target modules
                    module = None
                    for target_name, target_module in self._target_modules.items():
                        # Match by exact name or by checking if they refer to the same component
                        if scale_name == target_name:
                            module = target_module
                            break

                        # Convert blockX.attn/mlp format to transformer.h.X.attn/mlp.c_proj format
                        if scale_name.startswith("block") and (
                            "attn" in scale_name or "mlp" in scale_name
                        ):
                            # Extract layer number and component (attn/mlp)
                            parts = scale_name.split(".")
                            if len(parts) == 2:
                                layer_part = parts[0]  # e.g., "block0"
                                component = parts[1]  # e.g., "attn" or "mlp"

                                if layer_part.startswith("block"):
                                    layer_num = layer_part[
                                        5:
                                    ]  # Extract number from "block0"
                                    expected_target = (
                                        f"transformer.h.{layer_num}.{component}.c_proj"
                                    )

                                    if target_name == expected_target:
                                        module = target_module
                                        break

                        # Fallback: check if scale_name components match target_name components
                        if (
                            scale_name.endswith(target_name.split(".")[-1])
                            or target_name.endswith(scale_name)
                            or any(
                                part in target_name for part in scale_name.split(".")
                            )
                        ):
                            module = target_module
                            break

                    if module is not None and hasattr(module, "weight"):
                        # Check for quantized weights (skip if unsupported)
                        if hasattr(module.weight, "dtype") and module.weight.dtype in [
                            torch.int8,
                        ]:
                            self._log_event(
                                "scale_skipped",
                                level="WARN",
                                message=f"Skipping quantized weights in {scale_name}",
                                module_name=scale_name,
                                dtype=str(module.weight.dtype),
                            )
                            continue

                        # Store original scale factor for exact reversion
                        if scale_name not in self._original_scales:
                            self._original_scales[scale_name] = 1.0

                        # Apply scaling with proper device handling
                        with torch.no_grad():
                            original_device = module.weight.device
                            original_dtype = module.weight.dtype

                            # Use scalar multiplication to avoid MPS issues
                            if str(original_device).startswith("mps"):
                                module.weight.data = module.weight.data * scale_factor
                            else:
                                scale_tensor = torch.tensor(
                                    scale_factor,
                                    device=original_device,
                                    dtype=original_dtype,
                                )
                                module.weight.mul_(scale_tensor)

                        applied_count += 1

                        self._log_event(
                            "scale_applied",
                            message=f"Applied scale {scale_factor:.3f} to {scale_name}",
                            module_name=scale_name,
                            scale_factor=scale_factor,
                        )
                    else:
                        failed_modules.append(scale_name)

                except Exception as e:
                    failed_modules.append(scale_name)
                    self._log_event(
                        "scale_apply_error",
                        level="ERROR",
                        message=f"Failed to apply scale to {scale_name}: {str(e)}",
                        module_name=scale_name,
                        error=str(e),
                    )

            # Check if enough modules were successfully scaled
            if applied_count == 0:
                # Complete failure - rollback
                self._pop_checkpoint(model)
                self._log_event(
                    "enable_failed",
                    level="ERROR",
                    message="No modules were successfully scaled, rolling back",
                    failed_modules=failed_modules,
                )
                return False

            # Partial or complete success
            if failed_modules:
                self._log_event(
                    "enable_partial",
                    level="WARN",
                    message=f"Partial success: {applied_count} succeeded, {len(failed_modules)} failed",
                    applied_count=applied_count,
                    failed_modules=failed_modules,
                )

            # Commit the checkpoint on success
            self._commit_checkpoint()
            self._enabled = True

            self._log_event(
                "enable_complete",
                message=f"Enabled VE on {applied_count}/{len(self._scales)} modules",
                applied_count=applied_count,
                total_scales=len(self._scales),
                attempt_count=self._enable_attempt_count,
            )

            return True

        except Exception as e:
            # Catastrophic failure - rollback
            self._pop_checkpoint(model)
            self._log_event(
                "enable_catastrophic_failure",
                level="ERROR",
                message=f"Catastrophic failure during enable: {str(e)}",
                error=str(e),
                attempt_count=self._enable_attempt_count,
            )
            return False

    def disable(self, model: nn.Module, adapter=None) -> bool:
        """
        Disable variance equalization with idempotent operation and exact state restoration.

        Args:
            model: Model to revert VE on
            adapter: ModelAdapter (optional, for tying preservation)

        Returns:
            True if VE was successfully disabled, False otherwise
        """
        self._disable_attempt_count += 1

        # Idempotent check: if already disabled, return success
        if not self._enabled:
            self._log_event(
                "disable_idempotent",
                message="VE already disabled",
                attempt_count=self._disable_attempt_count,
            )
            return True

        self._log_event(
            "disable_start",
            message="Disabling VE by reverting to exact previous state",
            attempt_count=self._disable_attempt_count,
        )

        try:
            # Attempt to use checkpoint for exact restoration if available
            if self._checkpoint_stack:
                success = self._pop_checkpoint(model)
                if success:
                    self._enabled = False
                    self._log_event(
                        "disable_checkpoint_complete",
                        message="Disabled VE using checkpoint restoration",
                        attempt_count=self._disable_attempt_count,
                    )
                    return True
                else:
                    self._log_event(
                        "disable_checkpoint_failed",
                        level="WARN",
                        message="Checkpoint restoration failed, falling back to inverse scaling",
                    )

            # Fallback: revert using inverse scaling
            reverted_count = 0
            failed_modules = []

            for scale_name, scale_factor in self._scales.items():
                try:
                    # Find the actual module (use same logic as enable())
                    module = None
                    for target_name, target_module in self._target_modules.items():
                        # Match by exact name or by checking if they refer to the same component
                        if scale_name == target_name:
                            module = target_module
                            break

                        # Convert blockX.attn/mlp format to transformer.h.X.attn/mlp.c_proj format
                        if scale_name.startswith("block") and (
                            "attn" in scale_name or "mlp" in scale_name
                        ):
                            # Extract layer number and component (attn/mlp)
                            parts = scale_name.split(".")
                            if len(parts) == 2:
                                layer_part = parts[0]  # e.g., "block0"
                                component = parts[1]  # e.g., "attn" or "mlp"

                                if layer_part.startswith("block"):
                                    layer_num = layer_part[
                                        5:
                                    ]  # Extract number from "block0"
                                    expected_target = (
                                        f"transformer.h.{layer_num}.{component}.c_proj"
                                    )

                                    if target_name == expected_target:
                                        module = target_module
                                        break

                        # Fallback: check if scale_name components match target_name components
                        if (
                            scale_name.endswith(target_name.split(".")[-1])
                            or target_name.endswith(scale_name)
                            or any(
                                part in target_name for part in scale_name.split(".")
                            )
                        ):
                            module = target_module
                            break

                    if module is not None and hasattr(module, "weight"):
                        # Check for quantized weights (skip if unsupported)
                        if hasattr(module.weight, "dtype") and module.weight.dtype in [
                            torch.int8,
                        ]:
                            self._log_event(
                                "revert_skipped",
                                level="WARN",
                                message=f"Skipping quantized weights in {scale_name}",
                                module_name=scale_name,
                                dtype=str(module.weight.dtype),
                            )
                            continue

                        # Exact reversion using inverse scale
                        revert_factor = 1.0 / scale_factor

                        with torch.no_grad():
                            original_device = module.weight.device
                            original_dtype = module.weight.dtype

                            # Use scalar multiplication to avoid MPS issues
                            if str(original_device).startswith("mps"):
                                module.weight.data = module.weight.data * revert_factor
                            else:
                                revert_tensor = torch.tensor(
                                    revert_factor,
                                    device=original_device,
                                    dtype=original_dtype,
                                )
                                module.weight.mul_(revert_tensor)

                        reverted_count += 1

                        self._log_event(
                            "scale_reverted",
                            message=f"Reverted scale {scale_factor:.3f} from {scale_name} (factor: {revert_factor:.3f})",
                            module_name=scale_name,
                            original_scale=scale_factor,
                            revert_factor=revert_factor,
                        )
                    else:
                        failed_modules.append(scale_name)

                except Exception as e:
                    failed_modules.append(scale_name)
                    self._log_event(
                        "scale_revert_error",
                        level="ERROR",
                        message=f"Failed to revert scale from {scale_name}: {str(e)}",
                        module_name=scale_name,
                        error=str(e),
                    )

            # Check if enough modules were successfully reverted
            if reverted_count == 0 and self._scales:
                self._log_event(
                    "disable_failed",
                    level="ERROR",
                    message="No modules were successfully reverted",
                    failed_modules=failed_modules,
                )
                return False

            # Success (even if partial)
            if failed_modules:
                self._log_event(
                    "disable_partial",
                    level="WARN",
                    message=f"Partial success: {reverted_count} reverted, {len(failed_modules)} failed",
                    reverted_count=reverted_count,
                    failed_modules=failed_modules,
                )

            self._enabled = False
            self._log_event(
                "disable_complete",
                message=f"Disabled VE on {reverted_count}/{len(self._scales)} modules",
                reverted_count=reverted_count,
                attempt_count=self._disable_attempt_count,
            )

            return True

        except Exception as e:
            self._log_event(
                "disable_catastrophic_failure",
                level="ERROR",
                message=f"Catastrophic failure during disable: {str(e)}",
                error=str(e),
                attempt_count=self._disable_attempt_count,
            )
            return False

    def set_ab_results(
        self,
        ppl_no_ve: float,
        ppl_with_ve: float,
        windows_used: int | None = None,
        seed_used: int | None = None,
        ratio_ci: tuple[float, float] | None = None,
    ) -> None:
        """
        Store A/B testing results with reinforced validation logic.

        Args:
            ppl_no_ve: Perplexity without VE (A condition)
            ppl_with_ve: Perplexity with VE (B condition)
            windows_used: Number of calibration windows used (for determinism tracking)
            seed_used: Random seed used (for determinism tracking)
            ratio_ci: Tuple of (lower, upper) confidence interval for ppl_with_ve/ppl_no_ve
        """
        self._ppl_no_ve = ppl_no_ve
        self._ppl_with_ve = ppl_with_ve
        self._ab_windows_used = windows_used
        self._ab_seed_used = seed_used
        self._ratio_ci = ratio_ci

        # Robust gain computation with NaN/Inf protection
        if ppl_no_ve is None or ppl_with_ve is None or ppl_no_ve <= 0:
            self._ab_gain = 0.0
            gain_status = "invalid_ppl"
        else:
            try:
                self._ab_gain = (ppl_no_ve - ppl_with_ve) / max(ppl_no_ve, 1e-9)
                # Guard against NaN/Inf
                if not (
                    isinstance(self._ab_gain, int | float)
                    and abs(self._ab_gain) < float("inf")
                ):
                    self._ab_gain = 0.0
                    gain_status = "numeric_error"
                else:
                    gain_status = "computed"
            except (ZeroDivisionError, OverflowError, TypeError):
                self._ab_gain = 0.0
                gain_status = "numeric_error"

        # Safe formatting for None values
        ppl_no_ve_str = f"{ppl_no_ve:.3f}" if ppl_no_ve is not None else "None"
        ppl_with_ve_str = f"{ppl_with_ve:.3f}" if ppl_with_ve is not None else "None"

        self._log_event(
            "ab_results_stored",
            message=f"A/B results: {ppl_no_ve_str} → {ppl_with_ve_str} (gain: {self._ab_gain:.3f}, status: {gain_status})",
            ppl_no_ve=ppl_no_ve,
            ppl_with_ve=ppl_with_ve,
            gain=self._ab_gain,
            gain_status=gain_status,
            windows_used=windows_used,
            seed_used=seed_used,
            ratio_ci=ratio_ci,
        )
        self._post_edit_evaluated = True

        upper_ratio = None
        if isinstance(ratio_ci, tuple | list) and len(ratio_ci) == 2:
            try:
                upper_ratio = float(ratio_ci[1])
            except (TypeError, ValueError):
                upper_ratio = None

        if upper_ratio is not None and upper_ratio < 1.0:
            self._predictive_gate_state.update(
                {
                    "evaluated": True,
                    "passed": True,
                    "reason": "manual_override",
                }
            )

    def _push_checkpoint(self, model: nn.Module) -> None:
        """
        Push current model state to checkpoint stack for rollback capability.

        Args:
            model: Model to checkpoint
        """
        if not self._target_modules:
            return

        checkpoint = {}
        for name, module in self._target_modules.items():
            if hasattr(module, "weight"):
                # Store deep copy of weights for exact restoration
                checkpoint[name] = module.weight.data.clone().detach()

        self._checkpoint_stack.append(checkpoint)

        self._log_event(
            "checkpoint_pushed",
            message=f"Pushed checkpoint for {len(checkpoint)} modules",
            modules_count=len(checkpoint),
            stack_depth=len(self._checkpoint_stack),
        )

    def _pop_checkpoint(self, model: nn.Module) -> bool:
        """
        Pop and restore the most recent checkpoint.

        Args:
            model: Model to restore

        Returns:
            True if checkpoint was restored, False if no checkpoint available
        """
        if not self._checkpoint_stack:
            self._log_event(
                "checkpoint_pop_failed",
                level="WARN",
                message="No checkpoint available for rollback",
            )
            return False

        checkpoint = self._checkpoint_stack.pop()
        restored_count = 0

        for name, saved_weight in checkpoint.items():
            if name in self._target_modules:
                module = self._target_modules[name]
                if hasattr(module, "weight"):
                    # Exact restoration using saved tensor
                    module.weight.data.copy_(saved_weight)
                    restored_count += 1

        self._log_event(
            "checkpoint_popped",
            message=f"Restored checkpoint for {restored_count}/{len(checkpoint)} modules",
            restored_count=restored_count,
            stack_depth=len(self._checkpoint_stack),
        )

        return True

    def _commit_checkpoint(self) -> None:
        """
        Commit current state by removing the most recent checkpoint.
        """
        if self._checkpoint_stack:
            self._checkpoint_stack.pop()
            self._log_event(
                "checkpoint_committed",
                message="Committed current state, removed checkpoint",
                stack_depth=len(self._checkpoint_stack),
            )

    def _evaluate_ab_gate(self) -> tuple[bool, str]:
        """
        Evaluate A/B gate decision with reinforced criteria.

        Returns:
            (should_enable, reason) tuple
        """
        mode = self._policy.get("mode", "ci")
        min_rel_gain = self._policy.get("min_rel_gain", 0.0)
        tie_breaker = float(
            self._policy.get("tie_breaker_deadband", self.TIE_BREAKER_DEADBAND)
        )
        min_effect_log = self._policy.get("min_effect_lognll")

        predictive_enabled = bool(self._policy.get("predictive_gate", True))
        gate_state = getattr(self, "_predictive_gate_state", {}) or {}
        if (
            predictive_enabled
            and not gate_state.get("evaluated")
            and self._ratio_ci is not None
        ):
            gate_state = {
                **gate_state,
                "evaluated": True,
                "passed": True,
                "reason": gate_state.get("reason", "synthetic_ab_gate"),
            }
            self._predictive_gate_state = gate_state

        if self._ab_gain is None:
            return False, "no_ab_results"

        # Edge case: zero or negative PPLs
        if (
            self._ppl_no_ve is None
            or self._ppl_with_ve is None
            or self._ppl_no_ve <= 0
            or self._ppl_with_ve <= 0
        ):
            return False, "invalid_ppl_values"

        relative_gain = self._ab_gain
        if relative_gain < min_rel_gain:
            return (
                False,
                f"below_min_rel_gain (gain={relative_gain:.3f} < {min_rel_gain:.3f})",
            )

        if min_effect_log is not None:
            log_gain = math.log(max(self._ppl_no_ve, 1e-9)) - math.log(
                max(self._ppl_with_ve, 1e-9)
            )
            if log_gain < float(min_effect_log):
                return (
                    False,
                    f"below_min_effect_lognll (gain={log_gain:.6f} < {float(min_effect_log):.6f})",
                )

        if mode == "ci":
            if self._ratio_ci is None:
                return False, "missing_ratio_ci"
            ratio_lo, ratio_hi = self._ratio_ci
            if not all(
                isinstance(x, int | float) and math.isfinite(x) and x > 0
                for x in (ratio_lo, ratio_hi)
            ):
                return False, "invalid_ratio_ci"
            required_hi = 1.0 - min_rel_gain
            if min_effect_log is not None:
                required_hi = min(required_hi, math.exp(-float(min_effect_log)))
            if ratio_hi > required_hi:
                return (
                    False,
                    f"ci_interval_too_high (hi={ratio_hi:.3f} > {required_hi:.3f})",
                )

        # Absolute floor requirement: must have at least 0.05 improvement (ppl-like)
        absolute_improvement = self._ppl_no_ve - self._ppl_with_ve
        if absolute_improvement < self.ABSOLUTE_FLOOR:
            return (
                False,
                f"below_absolute_floor (improvement={absolute_improvement:.3f} < {self.ABSOLUTE_FLOOR})",
            )

        # Tie-breaker deadband: require gain >= min_gain + 0.005 to avoid flapping
        required_gain = self._policy["min_gain"] + tie_breaker
        if self._ab_gain < required_gain:
            return (
                False,
                f"below_threshold_with_deadband (gain={self._ab_gain:.3f} < {required_gain:.3f})",
            )

        if predictive_enabled and not gate_state.get("passed", False):
            reason = gate_state.get("reason", "predictive_gate_failed")
            return False, f"predictive_gate_failed ({reason})"

        return (
            True,
            f"criteria_met (gain={self._ab_gain:.3f} >= {required_gain:.3f}, improvement={absolute_improvement:.3f})",
        )

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
        result = self.finalize(model)

        details = result.get("details", {}) or {}
        errors = result.get("errors", []) or []
        warnings = result.get("warnings", []) or []
        passed = result.get("passed", False)

        if passed:
            action = "warn" if warnings else "continue"
        else:
            action = "warn" if self._monitor_only else "abort"

        return {
            "passed": passed,
            "action": action,
            "metrics": result.get("metrics", {}),
            "violations": errors,
            "message": "Variance guard validation completed",
            "details": details,
            "policy": details.get("policy", self._policy.copy()),
            "warnings": warnings,
            "errors": errors,
        }

    def finalize(self, model: nn.Module) -> dict[str, Any]:
        """
        Finalize variance guard and return comprehensive results.

        Args:
            model: The final edited model

        Returns:
            Dictionary with variance guard results and A/B testing metrics
        """
        start_time = time.time()

        if not self._prepared:
            self._log_event(
                "finalize_failed",
                level="ERROR",
                message="Variance guard not properly prepared",
            )
            return {
                "passed": False,
                "metrics": {},
                "warnings": ["Variance guard not properly prepared"],
                "errors": ["Preparation failed or no target modules found"],
                "finalize_time": time.time() - start_time,
                "events": self.events,
            }

        if self._monitor_only:
            self._enabled = False
            self._scales = {}

        if not self._post_edit_evaluated:
            self._refresh_after_edit_metrics(model)

        # Use reinforced A/B gate evaluation
        should_enable, gate_reason = self._evaluate_ab_gate()
        enabled_after_ab = self._enabled
        ab_gain = self._ab_gain or 0.0

        if should_enable and not enabled_after_ab:
            enable_result = self.enable(model)
            enabled_after_ab = enable_result or self._enabled
        elif not should_enable and enabled_after_ab:
            self.disable(model)
            enabled_after_ab = False

        # Enhanced validation gate criteria
        passed = True
        warnings = []
        errors = []

        # Log A/B gate decision for transparency
        self._log_event(
            "ab_gate_evaluation",
            message=f"A/B gate decision: should_enable={should_enable}, reason={gate_reason}",
            should_enable=should_enable,
            reason=gate_reason,
            current_enabled=enabled_after_ab,
        )

        # Primary validation: VE enabled state must match A/B gate decision
        if enabled_after_ab != should_enable:
            if enabled_after_ab and not should_enable:
                errors.append(f"VE enabled despite A/B gate rejection: {gate_reason}")
                passed = False
            elif not enabled_after_ab and should_enable:
                warnings.append(f"VE disabled despite A/B gate approval: {gate_reason}")
                # This is a warning, not an error, as being conservative is safer

        # Secondary validation: Check primary-metric degradation when VE is OFF (≤0.5 rise requirement, ppl-like)
        if not enabled_after_ab and self._ppl_no_ve and self._ppl_with_ve:
            # When VE is disabled, check that there's no significant degradation
            # The requirement is ≤0.5 rise (ppl-like units) when VE is OFF
            expected_final_ppl = self._ppl_no_ve  # Should be the no-VE result
            if hasattr(self, "_final_ppl") and self._final_ppl is not None:
                ppl_rise = self._final_ppl - expected_final_ppl
                if ppl_rise > 0.5:
                    errors.append(
                        f"Primary-metric rise {ppl_rise:.3f} > 0.5 when VE disabled"
                    )
                    passed = False

        # Tertiary validation: Check for deterministic A/B testing
        if self._ab_windows_used is not None and self._ab_seed_used is not None:
            expected_seed = self._policy.get("seed", 123)
            if self._ab_seed_used != expected_seed:
                warnings.append(
                    f"A/B test used unexpected seed {self._ab_seed_used}, expected {expected_seed}"
                )

        # Additional robustness checks
        if self._enable_attempt_count > 3:
            warnings.append(
                f"Multiple enable attempts ({self._enable_attempt_count}), may indicate instability"
            )

        if self._disable_attempt_count > 3:
            warnings.append(
                f"Multiple disable attempts ({self._disable_attempt_count}), may indicate instability"
            )

        if len(self._checkpoint_stack) > 0:
            warnings.append(
                f"Uncommitted checkpoints remaining: {len(self._checkpoint_stack)}"
            )

        # Validate tie-breaker deadband was applied
        if self._ab_gain is not None and self._ab_gain > 0:
            required_gain_with_deadband = self._policy["min_gain"] + float(
                self._policy.get("tie_breaker_deadband", self.TIE_BREAKER_DEADBAND)
            )
            if enabled_after_ab and self._ab_gain < required_gain_with_deadband:
                errors.append(
                    f"VE enabled without meeting tie-breaker deadband: gain {self._ab_gain:.3f} < {required_gain_with_deadband:.3f}"
                )
                passed = False

        # Validate absolute floor was checked
        if self._ppl_no_ve and self._ppl_with_ve:
            absolute_improvement = self._ppl_no_ve - self._ppl_with_ve
            if enabled_after_ab and absolute_improvement < self.ABSOLUTE_FLOOR:
                errors.append(
                    f"VE enabled without meeting absolute floor: improvement {absolute_improvement:.3f} < {self.ABSOLUTE_FLOOR}"
                )
                passed = False

        finalize_time = time.time() - start_time

        # Final metrics
        final_metrics = {
            "proposed_scales": len(self._scales),
            "target_modules": len(self._target_modules),
            "target_module_names": self._stats.get("target_module_names", []),
            "focus_modules": sorted(self._focus_modules) if self._focus_modules else [],
            "tap": self._stats.get("tap"),
            "ve_enabled": enabled_after_ab,
            "ab_gain": ab_gain,
            "ab_windows_used": self._ab_windows_used,
            "ab_seed_used": self._ab_seed_used,
            "monitor_only": self._monitor_only,
            "min_gain_threshold": self._policy["min_gain"],
            "met_threshold": should_enable,
            "ppl_no_ve": self._ppl_no_ve,
            "ppl_with_ve": self._ppl_with_ve,
            "scope": self._policy["scope"],
            "max_calib_used": self._policy["max_calib"],
            "mode": self._policy.get("mode"),
            "min_rel_gain": self._policy.get("min_rel_gain"),
            "alpha": self._policy.get("alpha"),
            "ratio_ci": self._ratio_ci,
            "calibration": self._calibration_stats.copy(),
            "predictive_gate": self._predictive_gate_state.copy(),
            "ab_provenance": copy.deepcopy(self._stats.get("ab_provenance", {})),
            "ab_point_estimates": copy.deepcopy(
                self._stats.get("ab_point_estimates", {})
            ),
            "raw_scales_pre_edit": copy.deepcopy(self._raw_scales_pre_edit),
            "raw_scales_post_edit": copy.deepcopy(self._raw_scales_post_edit),
            "proposed_scales_pre_edit": self._stats.get("proposed_scales_pre_edit", {}),
            "proposed_scales_post_edit": self._stats.get(
                "proposed_scales_post_edit", {}
            ),
        }

        if self._calibration_stats.get("status") != "complete":
            warnings.append(
                "Variance calibration coverage insufficient; operating in monitor mode"
            )

        self._log_event(
            "finalize_complete",
            message=f"Variance guard finalized - {'PASSED' if passed else 'FAILED'}",
            passed=passed,
            ve_enabled=enabled_after_ab,
            ab_gain=ab_gain,
            finalize_time=finalize_time,
        )

        result = {
            "passed": passed,
            "metrics": final_metrics,
            "warnings": warnings,
            "errors": errors,
            "finalize_time": finalize_time,
            "events": self.events,
            "details": {
                "guard_type": "variance",
                "ve_applied": enabled_after_ab,
                "ab_test_performed": self._ppl_no_ve is not None,
                "proposed_scales": self._scales,
                "stats": self._stats,
                "policy": self._policy,
            },
        }

        # Env-gated tiny evidence dump for auditors
        try:
            payload = {
                "variance": {
                    "mode": self._policy.get("mode"),
                    "min_effect": self._policy.get("min_effect", self.MIN_EFFECT),
                    "predictive_one_sided": bool(
                        self._policy.get("predictive_one_sided", True)
                    ),
                    "evaluated": True,
                }
            }
            maybe_dump_guard_evidence(".", payload)
        except Exception:
            pass

        return result

    def policy(self) -> VariancePolicyDict:
        """
        Get current policy configuration.

        Returns:
            VariancePolicyDict with current configuration
        """
        return self._policy.copy()
