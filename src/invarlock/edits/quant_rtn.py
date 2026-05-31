"""
InvarLock RTN dequantized weight-edit simulation.

The built-in ``quant_rtn`` edit is intentionally a deterministic numerical
weight perturbation, not a deployable quantization backend. It computes
round-to-nearest INT8 values, dequantizes them, and writes floating-point
weights back into the model so the assurance pipeline has a self-contained edit
primitive for demos, smokes, calibration, and regression gates.

Follows the ModelEdit protocol through the RTNQuantEdit class only.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from invarlock.core import INVARLOCK_CORE_ABI as CORE_ABI
from invarlock.core.api import (
    CalibrationData,
    EditRuntime,
    GuardChain,
    ModelAdapter,
    ModelEdit,
)
from invarlock.core.exceptions import EditError
from invarlock.edits import quant_rtn_kernels as quant_kernels
from invarlock.edits.quant_rtn_plan import (
    QuantTargetSelector,
    RTNQuantPlan,
    TargetModule,
    normalize_module_selectors,
    normalize_per_channel_option,
)

INVARLOCK_CORE_ABI = CORE_ABI

__all__ = ["RTNQuantEdit", "RTNQuantPlan", "QuantTargetSelector", "TargetModule"]


class RTNQuantEdit(ModelEdit):
    """
    ModelEdit implementation for RTN dequantized weight-edit simulation.

    This built-in edit is intentionally calibrated for INT8 simulation only. It
    computes symmetric per-channel quantization values, dequantizes them, and
    writes floating-point weights back into the model. It does not pack weights,
    lower runtime memory, or produce a deployable INT8 artifact.
    """

    name = "quant_rtn"

    @staticmethod
    def _normalize_per_channel_option(value: Any, *, default: bool = True) -> bool:
        return normalize_per_channel_option(value, default=default)

    @staticmethod
    def _normalize_module_selectors(
        module_selectors: Any,
    ) -> dict[str, list[str]]:
        return normalize_module_selectors(module_selectors)

    @staticmethod
    def _validate_options(
        *,
        bitwidth: int,
        per_channel: bool,
        group_size: int | None,
        clamp_ratio: float,
        scope: str,
        max_modules: int | None = None,
    ) -> None:
        if group_size is not None:
            raise ValueError(
                "RTNQuantEdit is an 8-bit dequantized simulation edit; "
                "group_size is unsupported."
            )
        RTNQuantPlan(
            bitwidth=bitwidth,
            per_channel=per_channel,
            clamp_ratio=clamp_ratio,
            scope=scope,  # type: ignore[arg-type]
            max_modules=max_modules,
        ).validate()

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
        module_selectors: dict[str, list[str]] | None = None,
    ):
        """
        Initialize RTN dequantized simulation edit.

        Args:
            bitwidth: Quantization bitwidth (INT8 only for built-in simulation)
            per_channel: Always True for per-channel quantization
            group_size: Unsupported; real grouped quantization belongs to backend
                adapters or separate backend-scoped edits
            clamp_ratio: Outlier clipping ratio (0.0 = no clipping)
            scope: Target scope ("ffn", "attn", "all")
            seed: Random seed for deterministic behavior
            guard_chain: Optional GuardChain for safety checks
        """
        per_channel = self._normalize_per_channel_option(per_channel, default=True)
        self._validate_options(
            bitwidth=bitwidth,
            per_channel=per_channel,
            group_size=group_size,
            clamp_ratio=clamp_ratio,
            scope=scope,
            max_modules=max_modules,
        )

        self.bitwidth = bitwidth
        self.per_channel = per_channel
        self.clamp_ratio = clamp_ratio
        self.scope = scope
        self.seed = seed
        self.guard_chain = guard_chain
        self.max_modules = max_modules
        self.module_selectors = self._normalize_module_selectors(module_selectors)

    def can_edit(self, model_desc: dict[str, Any]) -> bool:
        """
        Coarse metadata-only compatibility check.

        The actual model-object selection remains fail-closed in apply(), which
        raises when the configured scope matches no editable target modules.
        """
        required_keys = ["n_layer", "total_params"]
        has_requirements = all(key in model_desc for key in required_keys)
        if not has_requirements or model_desc.get("total_params", 0) <= 1000:
            return False

        module_names = self._module_names_from_model_desc(model_desc)
        if module_names:
            return self._has_matching_module_name(module_names)
        return True

    def _has_matching_module_name(self, module_names: list[str]) -> bool:
        if self.scope == "all":
            return True

        selector = QuantTargetSelector(
            scope=self.scope,
            module_selectors=self.module_selectors,
        )
        patterns = (
            selector._selector_patterns_for_scope()
            + selector._default_patterns_for_scope()
        )
        if not patterns:
            return False
        return any(
            pattern in module_name.lower()
            for module_name in module_names
            for pattern in patterns
        )

    @staticmethod
    def _module_names_from_model_desc(model_desc: Mapping[str, Any]) -> list[str]:
        for key in ("module_names", "target_modules", "modules", "named_modules"):
            raw_value = model_desc.get(key)
            if isinstance(raw_value, Mapping):
                candidates = raw_value.keys()
            elif isinstance(raw_value, list | tuple | set):
                candidates = raw_value
            else:
                continue
            names = [str(item) for item in candidates if isinstance(item, str) and item]
            if names:
                return names
        return []

    def _base_plan(self) -> RTNQuantPlan:
        return RTNQuantPlan(
            bitwidth=self.bitwidth,
            per_channel=self.per_channel,
            clamp_ratio=self.clamp_ratio,
            scope=self.scope,  # type: ignore[arg-type]
            seed=self.seed,
            max_modules=self.max_modules,
            module_selectors=dict(self.module_selectors),
        )

    @staticmethod
    def _limit_targets(
        target_modules: list[TargetModule], max_modules: int | None
    ) -> tuple[list[TargetModule], int]:
        total_identified = len(target_modules)
        if (
            isinstance(max_modules, int)
            and max_modules > 0
            and max_modules < total_identified
        ):
            return target_modules[:max_modules], total_identified
        return target_modules, total_identified

    def preview(
        self, model: nn.Module, adapter: ModelAdapter, calib: CalibrationData
    ) -> dict:
        """
        Preview RTN dequantized simulation without modifying the model.

        Args:
            model: The model to preview quantization on
            adapter: ModelAdapter for model-specific operations
            calib: Calibration data (not used for RTN)

        Returns:
            Dictionary with preview results including quantization plan
        """
        # Set deterministic seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Get model description
        model_desc = adapter.describe(model)

        active_plan = self._base_plan()
        target_modules, total_identified = self._limit_targets(
            self._select_target_modules(model),
            active_plan.max_modules,
        )

        tied_parameter_groups = self._get_weight_tying_groups(model)
        (
            physically_quantized_targets,
            deduplicated_modules,
            deduplicated_parameter_ids,
        ) = self._deduplicate_targets_by_parameter(target_modules)

        # Compute quantization statistics for unique physical parameters.
        quant_stats = self._compute_quantization_stats(physically_quantized_targets)

        # Estimate parameter changes
        total_params = sum(p.numel() for p in model.parameters())
        target_params = sum(
            target.module.weight.numel() for target in physically_quantized_targets
        )

        # Create quantization plan
        target_names = [target.name for target in target_modules]
        physical_module_names = [target.name for target in physically_quantized_targets]
        target_selection = self._target_report_payloads(
            target_modules,
            tied_parameter_groups=tied_parameter_groups,
        )
        plan = active_plan.as_report_payload(
            selected_modules=target_names,
            target_selection=target_selection,
        )
        plan.update(
            {
                "plan_digest": active_plan.digest(
                    selected_modules=target_names,
                    target_selection=target_selection,
                ),
                "physically_quantized_modules": physical_module_names,
                "total_modules_selected": len(target_names),
                "total_modules_quantized": len(physical_module_names),
                "total_params_quantized": int(target_params),
                "deduplicated_modules": deduplicated_modules,
                "quantization_stats": quant_stats,
                "tied_parameter_groups": tied_parameter_groups,
                "runtime_debug": {
                    "target_parameter_ids": [
                        target.runtime_debug_payload() for target in target_modules
                    ],
                    "deduplicated_parameter_ids": deduplicated_parameter_ids,
                },
            }
        )

        # Estimate sparsity (RTN doesn't create structural sparsity)
        estimated_sparsity = {
            "head_sparsity": 0.0,
            "neuron_sparsity": 0.0,
            "weight_sparsity": 0.0,  # RTN doesn't create weight sparsity
        }

        # Preview metrics
        bits_per_param = self.bitwidth
        theoretical_memory_saved = target_params * (32 - bits_per_param) / 8

        preview_metrics = {
            "preview_duration": 0.0,
            "target_params": int(target_params),
            "total_params": int(total_params),
            "coverage_ratio": target_params / total_params if total_params > 0 else 0.0,
            "target_modules_count": len(target_modules),
            "target_modules_quantized_count": len(physically_quantized_targets),
            "theoretical_packed_memory_saved_bytes": int(theoretical_memory_saved),
            "theoretical_packed_bits_per_param": bits_per_param,
            "actual_storage_format": "float_dequantized",
            "packed_quantized_storage": False,
            "runtime_memory_reduction": False,
            "will_use_clipping": self.clamp_ratio > 0.0,
            "will_use_grouping": False,
        }

        return {
            "plan": plan,
            "estimated_sparsity": estimated_sparsity,
            "preview_metrics": preview_metrics,
            "model_info": model_desc,
        }

    def apply(
        self,
        model: Any,
        adapter: ModelAdapter,
        plan: dict[str, Any] | None = None,
        runtime: EditRuntime | None = None,
    ) -> dict[str, Any]:
        """
        Apply RTN dequantized simulation to the model.

        Args:
            model: The model to edit (modified in-place)
            adapter: ModelAdapter for model-specific operations
            plan: Canonical edit plan parameters
            runtime: Optional protocol runtime context (unused by this edit)

        Returns:
            Dictionary with application results
        """
        plan_data = dict(plan or {})
        active_plan = RTNQuantPlan.from_payload(plan_data, defaults=self._base_plan())

        active_edit = RTNQuantEdit(
            bitwidth=active_plan.bitwidth,
            per_channel=active_plan.per_channel,
            clamp_ratio=active_plan.clamp_ratio,
            scope=active_plan.scope,
            seed=active_plan.seed,
            guard_chain=self.guard_chain,
            max_modules=active_plan.max_modules,
            module_selectors=active_plan.module_selectors,
        )

        # Set deterministic seed
        torch.manual_seed(active_plan.seed)
        random.seed(active_plan.seed)
        np.random.seed(active_plan.seed)

        target_modules, total_identified = self._limit_targets(
            active_edit._select_target_modules(model),
            active_plan.max_modules,
        )
        if not target_modules:
            raise EditError(
                code="E321",
                message=(
                    "RTN dequantized simulation matched no target modules for the current "
                    "model and scope."
                ),
                details={
                    "scope": active_plan.scope,
                    "bitwidth": active_plan.bitwidth,
                    "max_modules": active_plan.max_modules,
                    "identified_modules": total_identified,
                },
            )

        tied_parameter_groups = active_edit._get_weight_tying_groups(model)

        # Execute GuardChain before edit (if provided)
        guard_results = {}
        if active_edit.guard_chain is not None:
            guard_results["prepare"] = active_edit.guard_chain.prepare_all(
                model, adapter, None, {}
            )

            active_edit.guard_chain.before_edit_all(model)

        (
            physically_quantized_targets,
            deduplicated_modules,
            deduplicated_parameter_ids,
        ) = active_edit._deduplicate_targets_by_parameter(target_modules)

        # Apply quantization to each unique physical target parameter
        quantization_results = []
        total_params_quantized = 0

        for target in physically_quantized_targets:
            quant_result = active_edit._apply_rtn_quantization(
                target.module,
                active_plan.bitwidth,
                active_plan.clamp_ratio,
            )

            quant_result["module_name"] = target.name
            quant_result["selection_reason"] = target.selection_reason
            quant_result["matched_pattern"] = target.matched_pattern
            quant_result["module_type"] = target.module_type
            quantization_results.append(quant_result)
            total_params_quantized += quant_result["params_quantized"]
        if not quantization_results or total_params_quantized <= 0:
            raise EditError(
                code="E322",
                message="RTN dequantized simulation completed without changing any parameters.",
                details={
                    "scope": active_plan.scope,
                    "bitwidth": active_plan.bitwidth,
                    "max_modules": active_plan.max_modules,
                    "identified_modules": total_identified,
                },
            )

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
                "bitwidth": active_plan.bitwidth,
                "params": result["params_quantized"],
                "scale_stats": result.get("scale_stats", {}),
                "error_metrics": result.get("error_metrics", {}),
                "selection_reason": result.get("selection_reason"),
                "matched_pattern": result.get("matched_pattern"),
                "module_type": result.get("module_type"),
                "actual_storage_dtype": result.get("actual_storage_dtype"),
                "actual_storage_format": "float_dequantized",
                "packed_quantized_storage": False,
                "runtime_memory_reduction": False,
            }

        # Identify modified layers
        modified_layers = []
        for result in quantization_results:
            layer_name = self._layer_label_from_module_name(result["module_name"])
            if layer_name is not None and layer_name not in modified_layers:
                modified_layers.append(layer_name)

        # Store edit plan for evaluation report generation
        selected_modules = [target.name for target in target_modules]
        physical_module_names = [r["module_name"] for r in quantization_results]
        target_selection = self._target_report_payloads(
            target_modules,
            tied_parameter_groups=tied_parameter_groups,
        )
        edit_plan = active_plan.as_report_payload(
            selected_modules=selected_modules,
            target_selection=target_selection,
        )
        edit_plan.update(
            {
                "plan_digest": active_plan.digest(
                    selected_modules=selected_modules,
                    target_selection=target_selection,
                ),
                "tied_parameter_groups": tied_parameter_groups,
                "deduplicated_modules": deduplicated_modules,
                "aggregate_error_metrics": self._aggregate_error_metrics(
                    quantization_results
                ),
            }
        )
        if self._include_runtime_debug(runtime):
            edit_plan["runtime_debug"] = self._runtime_debug_payload(
                target_modules,
                deduplicated_parameter_ids=deduplicated_parameter_ids,
            )
        edit_plan.update(
            {
                "total_modules_selected": len(selected_modules),
                "total_modules_quantized": len(physical_module_names),
                "total_params_quantized": total_params_quantized,
                "physically_quantized_modules": physical_module_names,
            }
        )

        # Return in the standard format expected by the framework
        return {
            "name": self.name,
            "plan_digest": edit_plan["plan_digest"],
            "plan": edit_plan,  # Include the plan for evaluation report generation
            "deltas": {
                "params_changed": total_params_quantized,
                "sparsity": None,  # Quantization doesn't create sparsity
                "bitwidth_map": bitwidth_map,
                "layers_modified": len(modified_layers),
                "quantization_mode": "rtn_dequantized_weight_edit",
                "storage_format": "float_dequantized",
                "packed_quantized_storage": False,
                "runtime_memory_reduction": False,
            },
            "config": active_plan.as_report_payload(),
            "model_desc": adapter.describe(model)
            if hasattr(adapter, "describe")
            else {},
        }

    @staticmethod
    def _include_runtime_debug(runtime: EditRuntime | None) -> bool:
        if runtime is None:
            return True
        if runtime.include_runtime_debug is not None:
            return bool(runtime.include_runtime_debug)
        if runtime.verbose:
            return True
        profile = str(runtime.profile or "").strip().lower()
        return profile in {"", "dev"}

    @staticmethod
    def _runtime_debug_payload(
        target_modules: list[TargetModule],
        *,
        deduplicated_parameter_ids: list[str],
    ) -> dict[str, Any]:
        return {
            "target_parameter_ids": [
                target.runtime_debug_payload() for target in target_modules
            ],
            "deduplicated_parameter_ids": deduplicated_parameter_ids,
        }

    @staticmethod
    def _layer_label_from_module_name(module_name: str) -> str | None:
        name_parts = module_name.split(".")
        for layer_token in ("h", "layers"):
            if layer_token not in name_parts:
                continue
            layer_idx = name_parts.index(layer_token) + 1
            if layer_idx >= len(name_parts):
                continue
            layer_num = name_parts[layer_idx]
            if layer_num.isdigit():
                return f"layer_{layer_num}"
        return None

    def _select_target_modules(self, model: nn.Module) -> list[TargetModule]:
        """Identify target modules and keep selection metadata."""
        return QuantTargetSelector(
            scope=self.scope,
            module_selectors=self.module_selectors,
        ).select(model)

    def _get_weight_tying_groups(self, model: nn.Module) -> list[list[str]]:
        weight_to_modules: dict[int, list[str]] = {}

        for name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                weight_to_modules.setdefault(id(module.weight), []).append(name)

        return [
            sorted(module_names)
            for module_names in weight_to_modules.values()
            if len(module_names) > 1
        ]

    @staticmethod
    def _tied_group_lookup(tied_parameter_groups: list[list[str]]) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for group in tied_parameter_groups:
            stable_group = sorted(str(name) for name in group)
            if len(stable_group) < 2:
                continue
            group_key = "|".join(stable_group)
            for name in stable_group:
                lookup[name] = group_key
        return lookup

    def _target_report_payloads(
        self,
        target_modules: list[TargetModule],
        *,
        tied_parameter_groups: list[list[str]],
    ) -> list[dict[str, Any]]:
        tied_lookup = self._tied_group_lookup(tied_parameter_groups)
        payloads: list[dict[str, Any]] = []
        for target in target_modules:
            payload = target.as_report_payload()
            tied_group_key = tied_lookup.get(target.name)
            if tied_group_key:
                payload["tied_group_key"] = tied_group_key
                payload["tied_group_modules"] = tied_group_key.split("|")
            payloads.append(payload)
        return payloads

    @staticmethod
    def _deduplicate_targets_by_parameter(
        target_modules: list[TargetModule],
    ) -> tuple[list[TargetModule], list[str], list[str]]:
        physically_quantized_targets: list[TargetModule] = []
        deduplicated_modules: list[str] = []
        deduplicated_parameter_ids: list[str] = []
        seen_parameter_ids: set[int] = set()

        for target in target_modules:
            if target.parameter_id in seen_parameter_ids:
                deduplicated_modules.append(target.name)
                deduplicated_parameter_ids.append(str(target.parameter_id))
                continue
            seen_parameter_ids.add(target.parameter_id)
            physically_quantized_targets.append(target)

        return (
            physically_quantized_targets,
            deduplicated_modules,
            deduplicated_parameter_ids,
        )

    def _compute_quantization_stats(
        self, target_modules: list[TargetModule]
    ) -> dict[str, Any]:
        return quant_kernels.compute_quantization_stats(target_modules)

    def _apply_rtn_quantization(
        self,
        module: nn.Module,
        bitwidth: int,
        clamp_ratio: float,
    ) -> dict[str, Any]:
        return quant_kernels.apply_rtn_quantization(module, bitwidth, clamp_ratio)

    @staticmethod
    def _is_transformers_conv1d(module: nn.Module) -> bool:
        return quant_kernels.is_transformers_conv1d(module)

    def _weight_to_channel_matrix(
        self, module: nn.Module, weight: torch.Tensor
    ) -> tuple[torch.Tensor, Any]:
        return quant_kernels.weight_to_channel_matrix(module, weight)

    def _apply_outlier_clipping(
        self, weight: torch.Tensor, clamp_ratio: float
    ) -> torch.Tensor:
        return quant_kernels.apply_outlier_clipping(weight, clamp_ratio)

    def _quantize_per_channel(
        self, weight: torch.Tensor, qmin: int, qmax: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        return quant_kernels.quantize_per_channel(weight, qmin, qmax)

    @staticmethod
    def _population_std(tensor: torch.Tensor) -> float:
        return quant_kernels.population_std(tensor)

    @staticmethod
    def _quantization_error_metrics(
        original: torch.Tensor,
        edited: torch.Tensor,
        *,
        clipped_fraction: float,
        quant_code_edge_fraction: float,
    ) -> dict[str, float]:
        return quant_kernels.quantization_error_metrics(
            original,
            edited,
            clipped_fraction=clipped_fraction,
            quant_code_edge_fraction=quant_code_edge_fraction,
        )

    @staticmethod
    def _aggregate_error_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
        return quant_kernels.aggregate_error_metrics(results)
