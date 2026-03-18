"""Helpers for Hugging Face model loading.

Centralizes security- and performance-sensitive defaults used by HF adapters.
"""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_TORCH_UNSET = object()
_torch_module: Any = _TORCH_UNSET

_AUTO_LOADER_SPECS: dict[str, tuple[str, str]] = {
    "causal": ("transformers", "AutoModelForCausalLM"),
    "mlm": ("transformers", "AutoModelForMaskedLM"),
    "mlm_base": ("transformers", "AutoModel"),
    "seq2seq": ("transformers", "AutoModelForSeq2SeqLM"),
}

_DIRECT_SUBMODULE_SPECS: dict[str, dict[str, tuple[str, str]]] = {
    "causal": {
        "gpt2": ("transformers.models.gpt2.modeling_gpt2", "GPT2LMHeadModel"),
        "opt": ("transformers.models.opt.modeling_opt", "OPTForCausalLM"),
        "llama": ("transformers.models.llama.modeling_llama", "LlamaForCausalLM"),
        "mistral": (
            "transformers.models.mistral.modeling_mistral",
            "MistralForCausalLM",
        ),
        "mixtral": (
            "transformers.models.mixtral.modeling_mixtral",
            "MixtralForCausalLM",
        ),
        "qwen2": ("transformers.models.qwen2.modeling_qwen2", "Qwen2ForCausalLM"),
        "qwen3": ("transformers.models.qwen3.modeling_qwen3", "Qwen3ForCausalLM"),
        "qwen3_moe": (
            "transformers.models.qwen3_moe.modeling_qwen3_moe",
            "Qwen3MoeForCausalLM",
        ),
        "gemma3": (
            "transformers.models.gemma3.modeling_gemma3",
            "Gemma3ForConditionalGeneration",
        ),
        "gpt_neox": (
            "transformers.models.gpt_neox.modeling_gpt_neox",
            "GPTNeoXForCausalLM",
        ),
        "olmo2": ("transformers.models.olmo2.modeling_olmo2", "Olmo2ForCausalLM"),
        "phi": ("transformers.models.phi.modeling_phi", "PhiForCausalLM"),
        "phi3": ("transformers.models.phi3.modeling_phi3", "Phi3ForCausalLM"),
        "deepseek_v3": (
            "transformers.models.deepseek_v3.modeling_deepseek_v3",
            "DeepseekV3ForCausalLM",
        ),
    },
    "mlm": {
        "bert": ("transformers.models.bert.modeling_bert", "BertForMaskedLM"),
        "roberta": (
            "transformers.models.roberta.modeling_roberta",
            "RobertaForMaskedLM",
        ),
        "distilbert": (
            "transformers.models.distilbert.modeling_distilbert",
            "DistilBertForMaskedLM",
        ),
        "deberta": (
            "transformers.models.deberta.modeling_deberta",
            "DebertaForMaskedLM",
        ),
        "deberta-v2": (
            "transformers.models.deberta_v2.modeling_deberta_v2",
            "DebertaV2ForMaskedLM",
        ),
        "albert": ("transformers.models.albert.modeling_albert", "AlbertForMaskedLM"),
        "electra": (
            "transformers.models.electra.modeling_electra",
            "ElectraForMaskedLM",
        ),
    },
    "seq2seq": {
        "t5": ("transformers.models.t5.modeling_t5", "T5ForConditionalGeneration"),
        "bart": (
            "transformers.models.bart.modeling_bart",
            "BartForConditionalGeneration",
        ),
        "mbart": (
            "transformers.models.mbart.modeling_mbart",
            "MBartForConditionalGeneration",
        ),
        "pegasus": (
            "transformers.models.pegasus.modeling_pegasus",
            "PegasusForConditionalGeneration",
        ),
        "marian": (
            "transformers.models.marian.modeling_marian",
            "MarianMTModel",
        ),
    },
}


@dataclass(frozen=True)
class HFLoaderStrategy:
    task: str
    strategy: str
    loader: Any
    loader_label: str
    model_type: str | None = None


def _get_torch() -> Any:
    global _torch_module
    if _torch_module is _TORCH_UNSET:
        try:
            import torch as _torch
        except ModuleNotFoundError:
            _torch_module = None
        else:
            _torch_module = _torch
    return None if _torch_module is _TORCH_UNSET else _torch_module


def _require_torch() -> Any:
    torch = _get_torch()
    if torch is None:
        raise ModuleNotFoundError("torch")
    return torch


def _coerce_bool(val: Any) -> bool | None:
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
    return None


def resolve_trust_remote_code(
    kwargs: dict[str, Any] | None = None, *, default: bool = False
) -> bool:
    """Resolve trust_remote_code with config override and env opt-in."""
    if kwargs and "trust_remote_code" in kwargs:
        coerced = _coerce_bool(kwargs.get("trust_remote_code"))
        if coerced is not None:
            return coerced

    for env_name in (
        "INVARLOCK_TRUST_REMOTE_CODE",
        "TRUST_REMOTE_CODE_BOOL",
        "ALLOW_REMOTE_CODE",
    ):
        env_val = os.environ.get(env_name)
        coerced = _coerce_bool(env_val)
        if coerced is not None:
            return coerced

    return default


def default_dtype() -> Any:
    """Pick a safe default dtype for HF loads based on hardware."""
    torch = _require_torch()
    if torch.cuda.is_available():
        try:
            if (
                hasattr(torch.cuda, "is_bf16_supported")
                and torch.cuda.is_bf16_supported()
            ):
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.float16

    return torch.float32


def resolve_dtype(kwargs: dict[str, Any] | None = None) -> Any:
    """Resolve dtype from kwargs or choose a hardware-aware default."""
    torch = _require_torch()
    if kwargs and "dtype" in kwargs:
        val = kwargs.get("dtype")
        if isinstance(val, torch.dtype):
            return val
        if isinstance(val, str):
            s = val.strip().lower()
            if s == "auto":
                return "auto"
            mapping = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            if s in mapping:
                return mapping[s]

    return default_dtype()


def _normalize_model_type(value: Any) -> str | None:
    try:
        normalized = str(value or "").strip().lower()
    except Exception:
        return None
    return normalized or None


def _read_local_config(model_id: str) -> dict[str, Any] | None:
    path = Path(model_id)
    if not path.is_dir():
        return None
    config_path = path / "config.json"
    if not config_path.is_file():
        return None
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return raw if isinstance(raw, dict) else None


def _import_symbol(module_path: str, symbol_name: str) -> Any:
    module = importlib.import_module(module_path)
    return getattr(module, symbol_name)


def _loader_label(module_path: str, symbol_name: str) -> str:
    return f"{module_path}.{symbol_name}"


def _resolve_auto_loader(task: str) -> tuple[Any, str]:
    module_path, symbol_name = _AUTO_LOADER_SPECS[task]
    return (
        _import_symbol(module_path, symbol_name),
        _loader_label(module_path, symbol_name),
    )


def _resolve_direct_submodule_loader(
    task: str, model_type: str | None
) -> tuple[Any, str] | None:
    if model_type is None:
        return None
    spec = _DIRECT_SUBMODULE_SPECS.get(task, {}).get(model_type)
    if spec is None:
        return None
    module_path, symbol_name = spec
    try:
        return (
            _import_symbol(module_path, symbol_name),
            _loader_label(module_path, symbol_name),
        )
    except (AttributeError, ImportError, ModuleNotFoundError):
        return None


def resolve_core_loader_strategy(
    *,
    task: str,
    model_id: str,
    kwargs: dict[str, Any] | None = None,
    allow_direct_submodule: bool = False,
) -> HFLoaderStrategy:
    """Resolve the primary loader strategy for a core HF adapter."""

    if task not in _AUTO_LOADER_SPECS:
        raise KeyError(f"Unknown HF loader task: {task}")

    model_type = None
    config_data = _read_local_config(model_id)
    if isinstance(config_data, dict):
        model_type = _normalize_model_type(config_data.get("model_type"))

    if allow_direct_submodule and not resolve_trust_remote_code(kwargs):
        direct = _resolve_direct_submodule_loader(task, model_type)
        if direct is not None:
            loader, loader_label = direct
            return HFLoaderStrategy(
                task=task,
                strategy="direct_submodule",
                loader=loader,
                loader_label=loader_label,
                model_type=model_type,
            )

    loader, loader_label = _resolve_auto_loader(task)
    return HFLoaderStrategy(
        task=task,
        strategy="auto",
        loader=loader,
        loader_label=loader_label,
        model_type=model_type,
    )


__all__ = [
    "HFLoaderStrategy",
    "default_dtype",
    "resolve_core_loader_strategy",
    "resolve_dtype",
    "resolve_trust_remote_code",
]
