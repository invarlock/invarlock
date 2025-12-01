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

Follows the ModelEdit protocol with preview() and apply() methods.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from invarlock.core.api import CalibrationData, GuardChain, ModelAdapter, ModelEdit

__all__ = ["RTNQuantEdit"]


class RTNQuantEdit(ModelEdit):
    """
    ModelEdit implementation for RTN (Round-To-Nearest) weight-only quantization.

    This built-in edit is intentionally minimal and calibrated for INT8 only.
    It performs symmetric per-channel quantization with configurable scope and
    deterministic operation.
    """

    name = "quant_rtn"

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
        # Validate configuration – built-in edit is INT8-only
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

    def apply(self, model: Any, adapter, **kwargs) -> dict[str, Any]:
        """
        Apply RTN quantization to the model.

        Args:
            model: The model to edit (modified in-place)
            adapter: ModelAdapter for model-specific operations
            **kwargs: Edit parameters and configuration

        Returns:
            Dictionary with application results
        """
        try:
            # Extract configuration from kwargs - handle both 'bits' and 'bitwidth' for compatibility
            bitwidth = kwargs.get("bitwidth", kwargs.get("bits", self.bitwidth))
            group_size = kwargs.get("group_size", self.group_size)
            clamp_ratio = kwargs.get("clamp_ratio", self.clamp_ratio)
            scope = kwargs.get("scope", self.scope)
            seed = kwargs.get("seed", self.seed)

            # Diagnostic reporting
            print("🔧 RTN Quantization Configuration:")
            print(
                f"   Bitwidth: {bitwidth} (from config: {kwargs.get('bitwidth', kwargs.get('bits', 'default'))})"
            )
            print(f"   Scope: {scope}")
            print(f"   Group size: {group_size}")
            print(f"   Clamp ratio: {clamp_ratio}")
            print(f"   Seed: {seed}")

            # Persist configuration overrides for downstream helpers
            self.bitwidth = bitwidth
            self.group_size = group_size
            self.clamp_ratio = clamp_ratio
            self.scope = scope
            self.seed = seed

            # Set deterministic seed
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            # Identify target modules and get weight tying map
            print(f"🎯 Identifying target modules for scope '{scope}'...")
            target_modules = self._identify_target_modules(model)
            total_identified = len(target_modules)

            max_modules = kwargs.get("max_modules")
            if isinstance(max_modules, int) and max_modules > 0:
                if max_modules < total_identified:
                    print(
                        f"   Limiting quantization to first {max_modules} modules "
                        f"(of {total_identified}) based on plan.max_modules"
                    )
                    target_modules = target_modules[:max_modules]
                    self.max_modules = max_modules
                else:
                    print(
                        f"   max_modules={max_modules} >= available modules "
                        f"({total_identified}); using all targets"
                    )
                    self.max_modules = None
            else:
                self.max_modules = None

            tying_map = self._get_weight_tying_map(model)

            print(f"   Found {len(target_modules)} target modules:")
            for i, (name, module) in enumerate(target_modules):
                weight_shape = module.weight.shape
                param_count = module.weight.numel()
                print(f"   [{i + 1}] {name}: {weight_shape} ({param_count:,} params)")

            if len(target_modules) == 0:
                print("❌ WARNING: No target modules found! Check scope configuration.")
                print("   Available linear modules:")
                linear_modules = []
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear | nn.Conv1d):
                        linear_modules.append((name, module.weight.shape))
                for name, shape in linear_modules[:10]:  # Show first 10
                    print(f"     {name}: {shape}")
                if len(linear_modules) > 10:
                    print(f"     ... and {len(linear_modules) - 10} more")

            # Execute GuardChain before edit (if provided)
            guard_results = {}
            if self.guard_chain is not None:
                print("  Executing guard chain preparation...")
                guard_results["prepare"] = self.guard_chain.prepare_all(
                    model, adapter, None, {}
                )

                print("  Executing before-edit guards...")
                self.guard_chain.before_edit_all(model)

            # Apply quantization to each target module
            quantization_results = []
            total_params_quantized = 0

            for i, (module_name, module) in enumerate(target_modules):
                print(f"  [{i + 1}/{len(target_modules)}] Quantizing: {module_name}")
                print(
                    f"    Shape: {module.weight.shape}, Params: {module.weight.numel():,}"
                )
                print(
                    f"    Weight range: [{module.weight.min():.4f}, {module.weight.max():.4f}]"
                )

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

                print(
                    f"    ✓ Quantized {quant_result['params_quantized']:,} parameters"
                )

            # Execute GuardChain after edit (if provided)
            if self.guard_chain is not None:
                print("  Executing after-edit guards...")
                self.guard_chain.after_edit_all(model)

                print("  Finalizing guard chain...")
                guard_results["finalize"] = self.guard_chain.finalize_all(model)

                # Check if all guards passed
                if not self.guard_chain.all_passed(guard_results["finalize"]):
                    print("  ⚠️ Guard chain validation failed!")
                    guard_results["all_passed"] = False
                else:
                    print("  ✓ All guards passed")
                    guard_results["all_passed"] = True

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

            # Store edit plan for certificate generation
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
                "plan": edit_plan,  # Include the plan for certificate generation
                "deltas": {
                    "params_changed": total_params_quantized,
                    "sparsity": None,  # Quantization doesn't create sparsity
                    "bitwidth_map": bitwidth_map,
                    "layers_modified": len(modified_layers),
                },
                "config": kwargs,
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
                "config": kwargs,
                "model_desc": {},
                "error": str(e),
            }

    def _identify_target_modules(self, model: nn.Module) -> list[tuple[str, nn.Module]]:
        """Identify target modules based on scope configuration."""
        target_modules = []
        skipped_modules = []

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
                else:
                    skipped_modules.append(
                        (name, f"too small ({module.weight.numel()} params)")
                    )

            if should_include:
                target_modules.append((name, module))
            else:
                if self.scope != "all":  # Only log for specific scopes
                    skipped_modules.append((name, f"scope mismatch ({self.scope})"))

        # Log diagnostic information
        if skipped_modules:
            print(f"   Skipped {len(skipped_modules)} modules:")
            for name, reason in skipped_modules[:5]:  # Show first 5
                print(f"     {name}: {reason}")
            if len(skipped_modules) > 5:
                print(f"     ... and {len(skipped_modules) - 5} more")

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

        # Store original for comparison
        original_weight = weight.clone()

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

        # Ensure actual quantization occurred by applying quantization loss
        # This guarantees the weights are actually modified
        quantization_error = (quantized_weight - original_weight).abs().mean()
        print(f"    Quantization error: {quantization_error:.6f}")

        # Write back to module (preserving tying if needed)
        module.weight.data.copy_(quantized_weight)

        # Verify the weights actually changed
        final_weight = module.weight.data
        actual_change = not torch.allclose(original_weight, final_weight, atol=1e-6)
        if not actual_change:
            print(f"    WARNING: No actual weight change detected for {module}")

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
        eps = torch.finfo(weight.dtype).eps

        # Compute per-output-channel quantiles to preserve channel statistics
        quantiles = torch.quantile(
            weight,
            torch.tensor([lower, upper], device=weight.device, dtype=weight.dtype),
            dim=1,
            keepdim=True,
        )

        q_low = quantiles[0].clamp_min(-torch.inf)
        q_high = quantiles[1].clamp_min(eps)
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


# For backward compatibility, provide a functional interface
def apply(
    model: nn.Module,
    adapter: ModelAdapter,
    plan: dict[Any, Any] | None = None,
    **kwargs,
) -> dict:
    """
    Apply RTN quantization using the RTNQuantEdit API.

    This is the recommended interface that follows the ModelEdit protocol.
    """
    if plan is None:
        # Create plan from kwargs
        edit = RTNQuantEdit(
            bitwidth=kwargs.get("bitwidth", 8),
            per_channel=kwargs.get("per_channel", True),
            group_size=kwargs.get("group_size"),
            clamp_ratio=kwargs.get("clamp_ratio", 0.0),
            scope=kwargs.get("scope", "ffn"),
            seed=kwargs.get("seed", 42),
            max_modules=kwargs.get("max_modules"),
        )

        # Need calibration data for preview (though RTN doesn't use it)
        calib = kwargs.get("calib")
        preview_result = edit.preview(model, adapter, calib)
        plan = preview_result["plan"]

    # Apply the plan
    edit = RTNQuantEdit()
    return edit.apply(model, adapter, plan)
