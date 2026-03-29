"""
InvarLock – RTN Quantization Edit Plugin
====================================

Pure PyTorch Round-To-Nearest (RTN) weight-only quantization with no external dependencies.
Implements per-channel symmetric quantization with optional group size and outlier clipping.

Features:
- 8-bit weight quantization (INT8 RTN demo edit)
- Per-channel symmetric quantization (zero-point = 0)
- Configurable scope (FFN, attention, or all linear layers)
- Deterministic behavior with seed control
- GuardChain integration with quantization-aware policies

Follows the ModelEdit protocol through the RTNQuantEdit class only.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from invarlock.core.api import (
    CalibrationData,
    EditRuntime,
    GuardChain,
    ModelAdapter,
    ModelEdit,
)

__all__ = ["RTNQuantEdit"]


class RTNQuantEdit(ModelEdit):
    """
    ModelEdit implementation for RTN (Round-To-Nearest) weight-only quantization.

    This built-in edit is intentionally minimal and calibrated for INT8 only.
    It performs symmetric per-channel quantization with configurable scope and
    deterministic operation.
    """

    name = "quant_rtn"

    @staticmethod
    def _validate_options(
        *,
        bitwidth: int,
        clamp_ratio: float,
        scope: str,
    ) -> None:
        if bitwidth != 8:
            raise ValueError(
                f"RTNQuantEdit only supports 8-bit quantization (got bitwidth={bitwidth})"
            )
        if not (0.0 <= clamp_ratio <= 0.5):
            raise ValueError(
                f"Clamp ratio must be between 0.0 and 0.5, got {clamp_ratio}"
            )
        if scope not in ["ffn", "attn", "all"]:
            raise ValueError(f"Scope must be 'ffn', 'attn', or 'all', got {scope}")

    def __init__(
        self,
        bitwidth: int = 8,
        per_channel: bool = True,
        group_size: int | None = None,
        clamp_ratio: float = 0.0,
        scope: str = "ffn",
        seed: int = 42,
        guard_chain: GuardChain | None = None,
        max_modules: int | None = None,
    ):
        """
        Initialize RTN quantization edit.

        Args:
            bitwidth: Quantization bitwidth (INT8 only for built-in edit)
            per_channel: Always True for per-channel quantization
            group_size: Reserved for future use (ignored for INT8 demo edit)
            clamp_ratio: Outlier clipping ratio (0.0 = no clipping)
            scope: Target scope ("ffn", "attn", "all")
            seed: Random seed for deterministic behavior
            guard_chain: Optional GuardChain for safety checks
        """
        self._validate_options(
            bitwidth=bitwidth,
            clamp_ratio=clamp_ratio,
            scope=scope,
        )

        self.bitwidth = bitwidth
        self.per_channel = per_channel  # Always True
        self.group_size = group_size
        self.clamp_ratio = clamp_ratio
        self.scope = scope
        self.seed = seed
        self.guard_chain = guard_chain
        self.max_modules = max_modules

        # group_size is currently reserved for potential future variants; it is
        # ignored for the built-in INT8 demo edit.

    def can_edit(self, model_desc: dict[str, Any]) -> bool:
        """Check if RTN quantization can be applied to this model."""
        # Basic requirements for quantization
        required_keys = ["n_layer", "total_params"]
        has_requirements = all(key in model_desc for key in required_keys)

        # Need sufficient model size for meaningful quantization
        if has_requirements and model_desc.get("total_params", 0) > 1000:
            return True
        return False

    def preview(
        self, model: nn.Module, adapter: ModelAdapter, calib: CalibrationData
    ) -> dict:
        """
        Preview RTN quantization without modifying the model.

        Args:
            model: The model to preview quantization on
            adapter: ModelAdapter for model-specific operations
            calib: Calibration data (not used for RTN)

        Returns:
            Dictionary with preview results including quantization plan
        """
        try:
            # Set deterministic seed
            torch.manual_seed(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)

            # Get model description
            model_desc = adapter.describe(model)

            # Identify target modules
            target_modules = self._identify_target_modules(model)
            total_identified = len(target_modules)

            if (
                isinstance(self.max_modules, int)
                and self.max_modules > 0
                and self.max_modules < total_identified
            ):
                target_modules = target_modules[: self.max_modules]

            # Compute quantization statistics
            quant_stats = self._compute_quantization_stats(target_modules)

            # Estimate parameter changes
            total_params = sum(p.numel() for p in model.parameters())
            target_params = sum(module.weight.numel() for _, module in target_modules)

            # Create quantization plan
            plan = {
                "operation": "rtn_quantization",
                "bitwidth": self.bitwidth,
                "per_channel": self.per_channel,
                "group_size": self.group_size if self.bitwidth == 4 else None,
                "clamp_ratio": self.clamp_ratio,
                "scope": self.scope,
                "seed": self.seed,
                "target_modules": [name for name, _ in target_modules],
                "quantization_stats": quant_stats,
                "anti_tying_map": self._get_weight_tying_map(model),
            }
            if (
                isinstance(self.max_modules, int)
                and self.max_modules > 0
                and self.max_modules < total_identified
            ):
                plan["max_modules"] = self.max_modules

            # Estimate sparsity (RTN doesn't create structural sparsity)
            estimated_sparsity = {
                "head_sparsity": 0.0,
                "neuron_sparsity": 0.0,
                "weight_sparsity": 0.0,  # RTN doesn't create weight sparsity
            }

            # Preview metrics
            bits_per_param = self.bitwidth
            if self.bitwidth == 4 and self.group_size:
                # Account for scale storage
                scales_per_group = target_params / self.group_size
                bits_per_param = 4 + (
                    32 * scales_per_group / target_params
                )  # 32-bit scales

            memory_reduction_estimate = (
                target_params * (32 - bits_per_param) / 8
            )  # bytes

            preview_metrics = {
                "preview_duration": 0.0,
                "target_params": int(target_params),
                "total_params": int(total_params),
                "coverage_ratio": target_params / total_params
                if total_params > 0
                else 0.0,
                "target_modules_count": len(target_modules),
                "estimated_memory_saved_bytes": int(memory_reduction_estimate),
                "estimated_bits_per_param": bits_per_param,
                "will_use_clipping": self.clamp_ratio > 0.0,
                "will_use_grouping": self.bitwidth == 4 and self.group_size is not None,
            }

            return {
                "plan": plan,
                "estimated_sparsity": estimated_sparsity,
                "preview_metrics": preview_metrics,
                "model_info": model_desc,
            }

        except Exception as e:
            # Return error in preview
            return {
                "plan": {"operation": "failed", "error": str(e)},
                "estimated_sparsity": {
                    "head_sparsity": 0.0,
                    "neuron_sparsity": 0.0,
                    "weight_sparsity": 0.0,
                },
                "preview_metrics": {"error": str(e)},
                "model_info": {},
            }

    def apply(
        self,
        model: Any,
        adapter: ModelAdapter,
        plan: dict[str, Any] | None = None,
        runtime: EditRuntime | None = None,
    ) -> dict[str, Any]:
        """
        Apply RTN quantization to the model.

        Args:
            model: The model to edit (modified in-place)
            adapter: ModelAdapter for model-specific operations
            plan: Canonical edit plan parameters
            runtime: Optional protocol runtime context (unused by this edit)

        Returns:
            Dictionary with application results
        """
        try:
            del runtime
            plan_data = dict(plan or {})
            supported_keys = {
                "bitwidth",
                "group_size",
                "clamp_ratio",
                "scope",
                "seed",
                "max_modules",
            }
            unexpected = sorted(set(plan_data) - supported_keys)
            if unexpected:
                raise ValueError(
                    "Unsupported RTN plan fields: " + ", ".join(unexpected)
                )

            raw_bitwidth = plan_data.get("bitwidth", self.bitwidth)
            bitwidth = int(raw_bitwidth)
            group_size = plan_data.get("group_size", self.group_size)
            clamp_ratio = float(plan_data.get("clamp_ratio", self.clamp_ratio))
            scope = str(plan_data.get("scope", self.scope))
            seed = int(plan_data.get("seed", self.seed))
            raw_max_modules = plan_data.get("max_modules", self.max_modules)
            max_modules = (
                int(raw_max_modules)
                if isinstance(raw_max_modules, int | float) and int(raw_max_modules) > 0
                else None
            )

            self._validate_options(
                bitwidth=bitwidth,
                clamp_ratio=clamp_ratio,
                scope=scope,
            )

            active_edit = RTNQuantEdit(
                bitwidth=bitwidth,
                per_channel=self.per_channel,
                group_size=group_size,
                clamp_ratio=clamp_ratio,
                scope=scope,
                seed=seed,
                guard_chain=self.guard_chain,
                max_modules=max_modules,
            )

            # Set deterministic seed
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            target_modules = active_edit._identify_target_modules(model)
            total_identified = len(target_modules)
            if max_modules is not None:
                if max_modules < total_identified:
                    target_modules = target_modules[:max_modules]

            tying_map = active_edit._get_weight_tying_map(model)

            # Execute GuardChain before edit (if provided)
            guard_results = {}
            if active_edit.guard_chain is not None:
                guard_results["prepare"] = active_edit.guard_chain.prepare_all(
                    model, adapter, None, {}
                )

                active_edit.guard_chain.before_edit_all(model)

            # Apply quantization to each target module
            quantization_results = []
            total_params_quantized = 0

            for module_name, module in target_modules:
                # Apply RTN quantization
                quant_result = self._apply_rtn_quantization(
                    module,
                    bitwidth,
                    group_size,
                    clamp_ratio,
                    tying_map.get(module_name),
                )

                quant_result["module_name"] = module_name
                quantization_results.append(quant_result)
                total_params_quantized += quant_result["params_quantized"]

            # Execute GuardChain after edit (if provided)
            if active_edit.guard_chain is not None:
                active_edit.guard_chain.after_edit_all(model)

                guard_results["finalize"] = active_edit.guard_chain.finalize_all(model)

                # Check if all guards passed
                guard_results["all_passed"] = active_edit.guard_chain.all_passed(
                    guard_results["finalize"]
                )

            # Create bitwidth map
            bitwidth_map = {}
            for result in quantization_results:
                bitwidth_map[result["module_name"]] = {
                    "bitwidth": bitwidth,
                    "group_size": group_size if bitwidth == 4 else None,
                    "params": result["params_quantized"],
                    "scale_stats": result.get("scale_stats", {}),
                }

            # Identify modified layers
            modified_layers = []
            for result in quantization_results:
                # Extract layer name from module name (e.g., "transformer.h.0.mlp.c_fc" -> "layer_0")
                name_parts = result["module_name"].split(".")
                if "h" in name_parts:
                    h_idx = name_parts.index("h")
                    if h_idx + 1 < len(name_parts):
                        layer_num = name_parts[h_idx + 1]
                        layer_name = f"layer_{layer_num}"
                        if layer_name not in modified_layers:
                            modified_layers.append(layer_name)

            # Store edit plan for evaluation report generation
            modules_quantized = [r["module_name"] for r in quantization_results]

            edit_plan = {
                "bitwidth": bitwidth,
                "scope": scope,
                "group_size": group_size,
                "clamp_ratio": clamp_ratio,
                "seed": seed,
                "total_modules_quantized": len(modules_quantized),
                "total_params_quantized": total_params_quantized,
                "modules_quantized": modules_quantized,
            }

            # Return in the standard format expected by the framework
            return {
                "name": self.name,
                "plan_digest": f"rtn_quantization_{bitwidth}bit_{scope}",
                "plan": edit_plan,  # Include the plan for evaluation report generation
                "deltas": {
                    "params_changed": total_params_quantized,
                    "sparsity": None,  # Quantization doesn't create sparsity
                    "bitwidth_map": bitwidth_map,
                    "layers_modified": len(modified_layers),
                },
                "config": plan_data,
                "model_desc": adapter.describe(model)
                if hasattr(adapter, "describe")
                else {},
            }

        except Exception as e:
            # Return error in expected format
            return {
                "name": self.name,
                "plan_digest": "rtn_quantization_failed",
                "deltas": {
                    "params_changed": 0,
                    "sparsity": None,
                    "bitwidth_map": None,
                    "layers_modified": 0,
                },
                "config": dict(plan or {}),
                "model_desc": {},
                "error": str(e),
            }

    def _identify_target_modules(self, model: nn.Module) -> list[tuple[str, nn.Module]]:
        """Identify target modules based on scope configuration."""
        target_modules = []

        for name, module in model.named_modules():
            # Check for both Linear and Conv1D (GPT-2 uses Conv1D)
            if not isinstance(module, nn.Linear | nn.Conv1d):
                # Import Conv1D from transformers if available
                try:
                    from transformers.pytorch_utils import Conv1D

                    if not isinstance(module, Conv1D):
                        continue
                except ImportError:
                    continue

            # Check scope
            should_include = False
            if self.scope == "ffn":
                # FFN layers - be more permissive with pattern matching
                ffn_patterns = [
                    "mlp.c_fc",
                    "mlp.c_proj",
                    "feed_forward",
                    "fc1",
                    "fc2",
                    "mlp",
                    "ffn",
                    "intermediate.dense",
                    "output.dense",
                ]
                if any(pattern in name.lower() for pattern in ffn_patterns):
                    should_include = True
            elif self.scope == "attn":
                # Attention layers - be more permissive with pattern matching
                attn_patterns = [
                    "attn.c_attn",
                    "attn.c_proj",
                    "attention",
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "attn",
                ]
                if any(pattern in name.lower() for pattern in attn_patterns):
                    should_include = True
            elif self.scope == "all":
                # All linear layers above a minimum size threshold
                if module.weight.numel() >= 100:  # Minimum parameter threshold
                    should_include = True

            if should_include:
                target_modules.append((name, module))

        return target_modules

    def _get_module_by_name(self, model: nn.Module, name: str) -> nn.Module | None:
        """Get module by dotted name."""
        try:
            parts = name.split(".")
            module = model
            for part in parts:
                module = getattr(module, part)
            return module
        except AttributeError:
            return None

    def _get_weight_tying_map(self, model: nn.Module) -> dict[str, list[str]]:
        """Identify weight tying relationships for preservation."""
        tying_map = {}

        # Common tying patterns (e.g., lm_head and wte sharing weights)
        weight_to_modules: dict[int, list[str]] = {}

        for name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                weight_id = id(module.weight)
                if weight_id not in weight_to_modules:
                    weight_to_modules[weight_id] = []
                weight_to_modules[weight_id].append(name)

        # Create tying map
        for _weight_id, module_names in weight_to_modules.items():
            if len(module_names) > 1:
                for name in module_names:
                    tying_map[name] = [n for n in module_names if n != name]

        return tying_map

    def _compute_quantization_stats(
        self, target_modules: list[tuple[str, nn.Module]]
    ) -> dict[str, Any]:
        """Compute statistics about what will be quantized."""
        stats = {
            "total_modules": len(target_modules),
            "total_params": 0,
            "module_stats": [],
        }

        for name, module in target_modules:
            weight = module.weight
            module_stat = {
                "name": name,
                "shape": list(weight.shape),
                "params": weight.numel(),
                "weight_range": [float(weight.min()), float(weight.max())],
                "weight_mean": float(weight.mean()),
                "weight_std": float(weight.std()),
            }

            # Compute per-channel statistics
            if len(weight.shape) >= 2:
                channel_stats = []
                for c in range(weight.shape[0]):  # Output channels
                    channel_weight = weight[c]
                    channel_stats.append(
                        {
                            "channel": c,
                            "absmax": float(channel_weight.abs().max()),
                            "mean": float(channel_weight.mean()),
                            "std": float(channel_weight.std()),
                        }
                    )
                module_stat["channel_stats"] = channel_stats[:10]  # Limit for preview

            stats["module_stats"].append(module_stat)
            stats["total_params"] += module_stat["params"]

        return stats

    def _apply_rtn_quantization(
        self,
        module: nn.Module,
        bitwidth: int,
        group_size: int | None,
        clamp_ratio: float,
        tied_modules: list[str] | None = None,
    ) -> dict[str, Any]:
        """Apply RTN quantization to a single module."""
        weight = module.weight.data
        original_shape = weight.shape
        params_quantized = weight.numel()

        # Flatten weight for processing
        if len(weight.shape) == 1:
            # Handle bias or 1D weights
            weight_2d = weight.unsqueeze(0)
            is_1d = True
        else:
            weight_2d = weight.view(weight.shape[0], -1)  # [out_channels, in_features]
            is_1d = False

        # Apply outlier clipping if requested
        if clamp_ratio > 0.0:
            weight_2d = self._apply_outlier_clipping(weight_2d, clamp_ratio)

        # Compute quantization parameters
        qmin = -(2 ** (bitwidth - 1))
        qmax = 2 ** (bitwidth - 1) - 1

        if bitwidth == 4 and group_size is not None:
            # Group-wise quantization for 4-bit
            quantized_weight, scales, scale_stats = self._quantize_grouped(
                weight_2d, qmin, qmax, group_size
            )
        else:
            # Per-channel quantization
            quantized_weight, scales, scale_stats = self._quantize_per_channel(
                weight_2d, qmin, qmax
            )

        # Reshape back to original shape
        if is_1d:
            quantized_weight = quantized_weight.squeeze(0)
        else:
            quantized_weight = quantized_weight.view(original_shape)

        # Write back to module (preserving tying if needed)
        module.weight.data.copy_(quantized_weight)

        # Handle tied weights
        if tied_modules:
            for _tied_name in tied_modules:
                # In a real implementation, we'd update tied modules here
                # For now, just log
                pass

        return {
            "params_quantized": params_quantized,
            "original_shape": original_shape,
            "bitwidth": bitwidth,
            "group_size": group_size,
            "scale_stats": scale_stats,
            "clamp_applied": clamp_ratio > 0.0,
        }

    def _apply_outlier_clipping(
        self, weight: torch.Tensor, clamp_ratio: float
    ) -> torch.Tensor:
        """Apply outlier clipping based on quantile thresholds."""
        if clamp_ratio <= 0.0:
            return weight

        lower = clamp_ratio / 2
        upper = 1 - lower

        # `torch.quantile` is not implemented for fp16/bf16 on some backends.
        # Compute thresholds in float32, then cast back to preserve the original
        # dtype of the weights.
        weight_f32 = weight.float()

        # Compute per-output-channel quantiles to preserve channel statistics
        quantiles = torch.quantile(
            weight_f32,
            torch.tensor([lower, upper], device=weight.device, dtype=torch.float32),
            dim=1,
            keepdim=True,
        ).to(weight.dtype)

        q_low = quantiles[0]
        q_high = quantiles[1]
        return torch.clamp(weight, q_low, q_high)

    def _quantize_per_channel(
        self, weight: torch.Tensor, qmin: int, qmax: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Apply per-channel symmetric quantization."""
        # Compute per-channel scales (per output channel)
        channel_absmax = weight.abs().max(dim=1, keepdim=True)[0]  # [out_channels, 1]

        # Avoid division by zero
        eps = 1e-8
        channel_absmax = torch.clamp(channel_absmax, min=eps)

        # Symmetric quantization scale
        scales = channel_absmax / qmax

        # Quantize
        weight_scaled = weight / scales
        weight_quantized = torch.clamp(torch.round(weight_scaled), qmin, qmax)

        # Dequantize (write back as float)
        weight_dequantized = weight_quantized * scales

        # Compute statistics
        scale_stats = {
            "scale_mean": float(scales.mean()),
            "scale_std": float(scales.std()),
            "scale_min": float(scales.min()),
            "scale_max": float(scales.max()),
            "zero_scales": int((scales <= eps).sum()),
        }

        return weight_dequantized, scales.squeeze(), scale_stats

    def _quantize_grouped(
        self, weight: torch.Tensor, qmin: int, qmax: int, group_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Apply group-wise quantization for 4-bit mode."""
        out_channels, in_features = weight.shape

        # Pad input features to be divisible by group_size
        pad_size = (group_size - (in_features % group_size)) % group_size
        if pad_size > 0:
            weight_padded = torch.cat(
                [weight, torch.zeros(out_channels, pad_size, device=weight.device)],
                dim=1,
            )
        else:
            weight_padded = weight

        padded_in_features = weight_padded.shape[1]
        num_groups = padded_in_features // group_size

        # Reshape for group processing
        weight_grouped = weight_padded.view(out_channels, num_groups, group_size)

        # Compute per-group scales
        group_absmax = weight_grouped.abs().max(dim=2, keepdim=True)[
            0
        ]  # [out_channels, num_groups, 1]

        # Avoid division by zero
        eps = 1e-8
        group_absmax = torch.clamp(group_absmax, min=eps)

        # Symmetric quantization scale
        scales = group_absmax / qmax

        # Quantize
        weight_scaled = weight_grouped / scales
        weight_quantized = torch.clamp(torch.round(weight_scaled), qmin, qmax)

        # Dequantize
        weight_dequantized = weight_quantized * scales

        # Reshape back and remove padding
        weight_dequantized = weight_dequantized.view(out_channels, padded_in_features)
        if pad_size > 0:
            weight_dequantized = weight_dequantized[:, :-pad_size]

        # Compute statistics
        scale_stats = {
            "scale_mean": float(scales.mean()),
            "scale_std": float(scales.std()),
            "scale_min": float(scales.min()),
            "scale_max": float(scales.max()),
            "num_groups": num_groups,
            "group_size": group_size,
            "zero_scales": int((scales <= eps).sum()),
        }

        return weight_dequantized, scales.view(-1), scale_stats
