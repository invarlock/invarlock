"""RTN quantization plan and target selection helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import torch.nn as nn

__all__ = [
    "QuantTargetSelector",
    "RTNQuantPlan",
    "TargetModule",
    "normalize_module_selectors",
    "normalize_per_channel_option",
]


SUPPORTED_PLAN_KEYS = {
    "bitwidth",
    "clamp_ratio",
    "scope",
    "seed",
    "max_modules",
    "module_selectors",
    "per_channel",
}


def canonical_json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def normalize_per_channel_option(value: Any, *, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(
        "RTNQuantEdit expects per_channel to be a boolean-compatible value."
    )


def normalize_module_selectors(module_selectors: Any) -> dict[str, list[str]]:
    if not isinstance(module_selectors, dict):
        return {}

    normalized: dict[str, list[str]] = {}
    for key, values in module_selectors.items():
        if not isinstance(key, str):
            continue
        if isinstance(values, str):
            cleaned = [values.strip()] if values.strip() else []
        elif isinstance(values, list | tuple | set):
            cleaned = [
                str(item).strip()
                for item in values
                if isinstance(item, str) and str(item).strip()
            ]
            if isinstance(values, set):
                cleaned = sorted(cleaned)
        else:
            cleaned = []
        if cleaned:
            normalized[key.strip().lower()] = cleaned
    return normalized


@dataclass(frozen=True)
class RTNQuantPlan:
    """Canonical plan for the built-in dequantized RTN simulation edit."""

    bitwidth: int = 8
    per_channel: bool = True
    clamp_ratio: float = 0.0
    scope: Literal["ffn", "attn", "all"] = "ffn"
    seed: int = 42
    max_modules: int | None = None
    module_selectors: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        defaults: RTNQuantPlan | None = None,
    ) -> RTNQuantPlan:
        data = dict(payload or {})
        unexpected = sorted((set(data) - SUPPORTED_PLAN_KEYS) - {"group_size"})
        if "group_size" in data:
            unexpected.append("group_size")
        if unexpected:
            raise ValueError("Unsupported RTN plan fields: " + ", ".join(unexpected))

        base = defaults or cls()
        raw_max_modules = data.get("max_modules", base.max_modules)
        max_modules = None
        if raw_max_modules is not None:
            if not isinstance(raw_max_modules, int | float):
                raise ValueError("RTNQuantEdit expects max_modules to be numeric.")
            max_modules = int(raw_max_modules)

        plan = cls(
            bitwidth=int(data.get("bitwidth", base.bitwidth)),
            per_channel=normalize_per_channel_option(
                data.get("per_channel", base.per_channel),
                default=base.per_channel,
            ),
            clamp_ratio=float(data.get("clamp_ratio", base.clamp_ratio)),
            scope=str(data.get("scope", base.scope)),  # type: ignore[arg-type]
            seed=int(data.get("seed", base.seed)),
            max_modules=max_modules,
            module_selectors=normalize_module_selectors(
                data.get("module_selectors", base.module_selectors)
            ),
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        if self.bitwidth != 8:
            raise ValueError(
                f"RTNQuantEdit only supports 8-bit quantization (got bitwidth={self.bitwidth})"
            )
        if not self.per_channel:
            raise ValueError("RTNQuantEdit only supports per_channel=True.")
        if not (0.0 <= self.clamp_ratio <= 0.5):
            raise ValueError(
                f"Clamp ratio must be between 0.0 and 0.5, got {self.clamp_ratio}"
            )
        if self.scope not in {"ffn", "attn", "all"}:
            raise ValueError(f"Scope must be 'ffn', 'attn', or 'all', got {self.scope}")
        if self.max_modules is not None and self.max_modules <= 0:
            raise ValueError("RTNQuantEdit expects max_modules to be positive.")

    def as_report_payload(
        self,
        *,
        selected_modules: list[str] | None = None,
        target_selection: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "operation": "rtn_quantize_dequantize_weight_edit",
            "quantization_mode": "rtn_dequantized_weight_edit",
            "storage_format": "float_dequantized",
            "actual_storage_format": "float_dequantized",
            "packed_quantized_storage": False,
            "runtime_memory_reduction": False,
            "deployment_backend": None,
            "bitwidth": self.bitwidth,
            "per_channel": self.per_channel,
            "clamp_ratio": self.clamp_ratio,
            "scope": self.scope,
            "seed": self.seed,
        }
        if self.max_modules is not None:
            payload["max_modules"] = self.max_modules
        if self.module_selectors:
            payload["module_selectors"] = dict(self.module_selectors)
        if selected_modules is not None:
            payload["selected_modules"] = list(selected_modules)
        if target_selection is not None:
            payload["target_selection"] = list(target_selection)
        return payload

    def digest(
        self,
        *,
        selected_modules: list[str] | None = None,
        target_selection: list[dict[str, Any]] | None = None,
    ) -> str:
        stable_target_selection = (
            [self._stable_target_entry(entry) for entry in target_selection]
            if target_selection is not None
            else None
        )
        return canonical_json_digest(
            self.as_report_payload(
                selected_modules=selected_modules,
                target_selection=stable_target_selection,
            )
        )

    @staticmethod
    def _stable_target_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
        stable_keys = (
            "module_name",
            "module_type",
            "weight_shape",
            "params",
            "selection_reason",
            "matched_pattern",
            "tied_group_key",
            "tied_group_modules",
        )
        return {key: entry[key] for key in stable_keys if key in entry}


@dataclass(frozen=True)
class TargetModule:
    """Resolved module target plus the reason it was selected."""

    name: str
    module: nn.Module
    selection_reason: str
    matched_pattern: str | None
    parameter_id: int
    module_type: str

    def as_report_payload(self) -> dict[str, Any]:
        return self.stable_report_payload()

    def runtime_debug_payload(self) -> dict[str, Any]:
        return {
            "module_name": self.name,
            "parameter_id": str(self.parameter_id),
        }

    def stable_report_payload(self) -> dict[str, Any]:
        weight = getattr(self.module, "weight", None)
        weight_shape = list(weight.shape) if weight is not None else []
        params = int(weight.numel()) if weight is not None else 0
        return {
            "module_name": self.name,
            "selection_reason": self.selection_reason,
            "matched_pattern": self.matched_pattern,
            "module_type": self.module_type,
            "weight_shape": weight_shape,
            "params": params,
        }


@dataclass(frozen=True)
class QuantTargetSelector:
    """Select RTN simulation targets and explain each match."""

    scope: str
    module_selectors: dict[str, list[str]] = field(default_factory=dict)
    min_params: int = 100

    def select(self, model: nn.Module) -> list[TargetModule]:
        target_modules: list[TargetModule] = []
        user_patterns = self._selector_patterns_for_scope()
        default_patterns = self._default_patterns_for_scope()

        for name, module in model.named_modules():
            if not self._is_supported_module(module):
                continue
            weight = getattr(module, "weight", None)
            if weight is None:
                continue

            should_include = False
            selection_reason = ""
            matched_pattern: str | None = None
            lowered = name.lower()

            for pattern in user_patterns:
                if pattern in lowered:
                    should_include = True
                    selection_reason = "model_profile_selector"
                    matched_pattern = pattern
                    break

            if not should_include and self.scope in {"ffn", "attn"}:
                for pattern in default_patterns:
                    if pattern in lowered:
                        should_include = True
                        selection_reason = "name_heuristic"
                        matched_pattern = pattern
                        break

            if (
                not should_include
                and self.scope == "all"
                and weight.numel() >= self.min_params
            ):
                should_include = True
                selection_reason = "scope_all_min_params"

            if should_include:
                target_modules.append(
                    TargetModule(
                        name=name,
                        module=module,
                        selection_reason=selection_reason,
                        matched_pattern=matched_pattern,
                        parameter_id=id(weight),
                        module_type=(
                            f"{module.__class__.__module__}.{module.__class__.__name__}"
                        ),
                    )
                )

        return target_modules

    @staticmethod
    def _is_supported_module(module: nn.Module) -> bool:
        if isinstance(module, nn.Linear | nn.Conv1d):
            return True
        try:
            from transformers.pytorch_utils import Conv1D

            return isinstance(module, Conv1D)
        except ImportError:
            return False

    def _selector_patterns_for_scope(self) -> tuple[str, ...]:
        if not self.module_selectors:
            return ()
        if self.scope == "ffn":
            keys = ("ffn", "feed_forward")
        elif self.scope == "attn":
            keys = ("attn", "attention")
        else:
            keys = tuple(self.module_selectors.keys())
        patterns: list[str] = []
        seen: set[str] = set()
        for key in keys:
            for value in self.module_selectors.get(key, []):
                normalized = str(value).strip().lower()
                if normalized and normalized not in seen:
                    patterns.append(normalized)
                    seen.add(normalized)
        return tuple(patterns)

    def _default_patterns_for_scope(self) -> tuple[str, ...]:
        if self.scope == "ffn":
            return (
                "mlp.c_fc",
                "mlp.c_proj",
                "feed_forward",
                "fc1",
                "fc2",
                "mlp",
                "ffn",
                "intermediate.dense",
                "output.dense",
            )
        if self.scope == "attn":
            return (
                "attn.c_attn",
                "attn.c_proj",
                "attention",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "attn",
            )
        return ()
