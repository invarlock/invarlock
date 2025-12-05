"""Adapter namespace (`invarlock.adapters`) exposing built-in adapters.

Provides lazy import of heavy submodules to avoid importing Transformers
stacks unless needed. Accessing adapter classes triggers on-demand import.
"""

from __future__ import annotations

import importlib as _importlib
from typing import Any as _Any

from invarlock.core.abi import INVARLOCK_CORE_ABI as INVARLOCK_CORE_ABI

from .base import (
    AdapterConfig,
    AdapterInterface,
    BaseAdapter,
    DeviceManager,
)
from .base import (
    PerformanceMetrics as BasePerformanceMetrics,
)
from .capabilities import (
    ModelCapabilities,
    QuantizationConfig,
    QuantizationMethod,
    detect_capabilities_from_model,
    detect_quantization_from_config,
)

_LAZY_MAP = {
    "HF_BERT_Adapter": ".hf_bert",
    "HF_GPT2_Adapter": ".hf_gpt2",
    "HF_LLaMA_Adapter": ".hf_llama",
    "HF_T5_Adapter": ".hf_t5",
    "HF_ORT_CausalLM_Adapter": ".hf_onnx",
    "HF_Causal_Auto_Adapter": ".auto",
    "HF_MLM_Auto_Adapter": ".auto",
}


def __getattr__(name: str) -> _Any:  # pragma: no cover - simple lazy import
    mod_name = _LAZY_MAP.get(name)
    if not mod_name:
        raise AttributeError(name)
    module = _importlib.import_module(mod_name, __name__)
    try:
        return getattr(module, name)
    except AttributeError as exc:  # re-raise with module context
        raise AttributeError(f"{name} not found in {mod_name}") from exc


# Simple quality label helper used by tests
def quality_label(ratio: float) -> str:
    if ratio <= 1.10:
        return "Excellent"
    if ratio <= 1.25:
        return "Good"
    if ratio <= 1.40:
        return "Fair"
    return "Degraded"


class _RemovedComponent:
    def __init__(self, name: str, replacement: str | None = None):
        self._name = name
        self._replacement = replacement

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self._name} is not available in InvarLock 1.0."
            + (f" Use: {self._replacement}" if self._replacement else "")
        )

    def __getattr__(self, _):  # pragma: no cover - simple passthrough
        return _RemovedComponent(self._name, self._replacement)


# Placeholders for removed/legacy utilities referenced in tests
HF_Pythia_Adapter = _RemovedComponent("HF_Pythia_Adapter")
auto_tune_pruning_budget = _RemovedComponent("auto_tune_pruning_budget")
run_auto_invarlock = _RemovedComponent("run_auto_invarlock")
InvarLockPipeline = _RemovedComponent("InvarLockPipeline", "invarlock.cli.app:main")
InvarLockConfig = _RemovedComponent(
    "InvarLockConfig", "invarlock.cli.config:InvarLockConfig"
)
run_invarlock_pipeline = _RemovedComponent(
    "run_invarlock_pipeline", "invarlock.cli.run"
)
run_invarlock = _RemovedComponent("run_invarlock", "invarlock.cli.run")
quick_prune_gpt2 = _RemovedComponent("quick_prune_gpt2")

__all__ = [
    "HF_GPT2_Adapter",
    "HF_BERT_Adapter",
    "HF_LLaMA_Adapter",
    "HF_T5_Adapter",
    "HF_ORT_CausalLM_Adapter",
    "HF_Causal_Auto_Adapter",
    "HF_MLM_Auto_Adapter",
    "BaseAdapter",
    "AdapterConfig",
    "AdapterInterface",
    "DeviceManager",
    "BasePerformanceMetrics",
    "quality_label",
    "_RemovedComponent",
    "INVARLOCK_CORE_ABI",
    # Capabilities
    "ModelCapabilities",
    "QuantizationConfig",
    "QuantizationMethod",
    "detect_capabilities_from_model",
    "detect_quantization_from_config",
]
