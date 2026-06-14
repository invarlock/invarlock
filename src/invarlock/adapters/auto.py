from __future__ import annotations

import importlib as _importlib
import json
import os
from pathlib import Path
from typing import Any, Protocol, cast

from invarlock.core import INVARLOCK_CORE_ABI as CORE_ABI
from invarlock.core.api import ModelAdapter

INVARLOCK_CORE_ABI = CORE_ABI

_CAUSAL_MODEL_TYPES = {
    "deepseek",
    "falcon",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma4",
    "gpt_oss",
    "glm",
    "gpt2",
    "gpt_neox",
    "gptj",
    "llama",
    "mistral",
    "mistral3",
    "mixtral",
    "olmo",
    "olmo2",
    "olmoe",
    "opt",
    "phi",
    "phi3",
    "qwen",
    "qwen2",
    "qwen2_moe",
    "qwen3",
    "qwen3_moe",
    "yi",
}
_MLM_MODEL_TYPES = {
    "albert",
    "bert",
    "deberta",
    "deberta-v2",
    "distilbert",
    "roberta",
}
_MODEL_CONFIG_ERRORS = (AttributeError, TypeError, ValueError)


def _read_local_hf_config(model_id: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Read config.json from a local HF directory if present."""
    try:
        path = Path(model_id)
    except (OSError, TypeError, ValueError):
        return None
    cfg_path = path / "config.json"
    if not cfg_path.exists():
        return None
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _detect_quant_family_from_cfg(cfg: dict[str, Any]) -> str | None:
    """Detect quantization family from a HF config dict."""
    quant_config = cfg.get("quantization_config") or {}
    if not isinstance(quant_config, dict):
        return None
    try:
        method = str(
            quant_config.get("quant_method", quant_config.get("quant_method_full", ""))
        ).lower()
    except _MODEL_CONFIG_ERRORS:
        return None
    normalized_method = method.replace("-", "_")
    if (
        "compressed_tensors" in normalized_method
        or "compressedtensors" in normalized_method
        or "llmcompressor" in normalized_method
    ):
        return "hf_ct"
    if "gptq" in method:
        return "hf_gptq"
    if "awq" in method:
        return "hf_awq"
    if "torchao" in method:
        return "hf_torchao"
    if "hqq" in method:
        return "hf_hqq"
    if "quanto" in method:
        return "hf_quanto"
    if "bitsandbytes" in method or "bnb" in method:
        return "hf_bnb"
    return None


def resolve_auto_adapter(
    model_id: str | os.PathLike[str], default: str = "hf_causal"
) -> str:
    """Resolve an appropriate built-in adapter name for a model."""
    cfg = _read_local_hf_config(model_id)
    model_id_str = str(model_id)

    def _from_cfg(config: dict[str, Any]) -> str | None:
        family = _detect_quant_family_from_cfg(config)
        if family:
            return family
        model_type = str(config.get("model_type", "")).lower()
        if model_type in _CAUSAL_MODEL_TYPES:
            return "hf_causal"
        if bool(config.get("is_encoder_decoder", False)):
            return "hf_seq2seq"
        archs = [
            str(arch)
            for arch in config.get("architectures", [])
            if isinstance(arch, str)
        ]
        arch_blob = " ".join(archs)
        if "ConditionalGeneration" in arch_blob or "Seq2SeqLM" in arch_blob:
            return "hf_seq2seq"
        if model_type in _MLM_MODEL_TYPES or "MaskedLM" in arch_blob:
            return "hf_mlm"
        if "CausalLM" in arch_blob or "ForCausalLM" in arch_blob:
            return "hf_causal"
        return None

    if isinstance(cfg, dict):
        resolved = _from_cfg(cfg)
        if resolved:
            return resolved

    lower_id = model_id_str.lower()
    if any(key in lower_id for key in ["gptq", "-gptq", "_gptq"]):
        return "hf_gptq"
    if any(key in lower_id for key in ["awq", "-awq", "_awq"]):
        return "hf_awq"
    if any(key in lower_id for key in ["torchao", "-torchao", "_torchao"]):
        return "hf_torchao"
    if any(key in lower_id for key in ["hqq", "-hqq", "_hqq"]):
        return "hf_hqq"
    if any(key in lower_id for key in ["quanto", "-quanto", "_quanto"]):
        return "hf_quanto"
    if any(
        key in lower_id
        for key in [
            "compressed-tensors",
            "compressed_tensors",
            "compressedtensors",
            "llmcompressor",
        ]
    ):
        return "hf_ct"
    if any(
        key in lower_id
        for key in ["bnb", "bitsandbytes", "-4bit", "-8bit", "4bit", "8bit"]
    ):
        return "hf_bnb"
    if any(key in lower_id for key in ["t5", "bart"]):
        return "hf_seq2seq"
    if any(key in lower_id for key in ["bert", "roberta", "albert", "deberta"]):
        return "hf_mlm"
    return default


def apply_auto_adapter_if_needed(cfg: Any) -> Any:
    """Mutate/clone an InvarLockConfig to resolve adapter:auto to a concrete adapter."""
    try:
        adapter = str(getattr(cfg.model, "adapter", ""))
        if adapter.strip().lower() not in {"auto", "auto_hf"}:
            return cfg
        model_id = str(getattr(cfg.model, "id", ""))
        resolved = resolve_auto_adapter(model_id)
        data = cfg.model_dump()
        data.setdefault("model", {})["adapter"] = resolved
        return cfg.__class__(data)
    except (AttributeError, KeyError, TypeError, ValueError):
        return cfg


class _LoadableAdapter(Protocol):
    def can_handle(self, model: Any) -> bool: ...

    def describe(self, model: Any) -> dict[str, Any]: ...

    def snapshot(self, model: Any) -> bytes: ...

    def restore(self, model: Any, blob: bytes) -> None: ...

    def load_model(self, model_id: str, device: str = "auto", **kwargs: Any) -> Any: ...


def _detect_quantization_from_path(model_id: str) -> str | None:
    """
    Detect quantization method from a local checkpoint path.

    Returns:
        Quantization adapter name ("hf_bnb", "hf_awq", "hf_gptq",
        "hf_torchao", "hf_hqq", "hf_quanto",
        "hf_ct") or None.
    """
    config_data = _read_local_hf_config(model_id)
    if not isinstance(config_data, dict):
        return None
    return _detect_quant_family_from_cfg(config_data)


def _detect_quantization_from_model(model: Any) -> str | None:
    """
    Detect quantization method from a loaded model instance.

    Returns:
        Quantization adapter name ("hf_bnb", "hf_awq", "hf_gptq",
        "hf_torchao", "hf_hqq", "hf_quanto",
        "hf_ct") or None.
    """
    config = getattr(model, "config", None)
    if config is None:
        return None

    quant_cfg = getattr(config, "quantization_config", None)
    if quant_cfg is None:
        # Check for BNB attributes on the model itself
        if getattr(model, "is_loaded_in_8bit", False) or getattr(
            model, "is_loaded_in_4bit", False
        ):
            return "hf_bnb"
        return None

    # Handle dict-style config
    if isinstance(quant_cfg, dict):
        quant_method = quant_cfg.get("quant_method", "")
        if not isinstance(quant_method, str):
            return None
        quant_method = quant_method.lower()
        normalized_method = quant_method.replace("-", "_")
        if (
            "compressed_tensors" in normalized_method
            or "compressedtensors" in normalized_method
            or "llmcompressor" in normalized_method
        ):
            return "hf_ct"
        if quant_method == "awq":
            return "hf_awq"
        elif quant_method == "gptq":
            return "hf_gptq"
        elif "torchao" in quant_method:
            return "hf_torchao"
        elif "hqq" in quant_method:
            return "hf_hqq"
        elif "quanto" in quant_method:
            return "hf_quanto"
        elif "bitsandbytes" in quant_method or "bnb" in quant_method:
            return "hf_bnb"
    else:
        # Object-style config
        cfg_class = quant_cfg.__class__.__name__
        if cfg_class in ("AWQConfig",):
            return "hf_awq"
        elif cfg_class in ("GPTQConfig",):
            return "hf_gptq"
        elif "TorchAO" in cfg_class or "Int8WeightOnly" in cfg_class:
            return "hf_torchao"
        elif (
            cfg_class in ("HqqConfig", "HQQConfig")
            or "Hqq" in cfg_class
            or "HQQ" in cfg_class
        ):
            return "hf_hqq"
        elif cfg_class in ("QuantoConfig",) or "Quanto" in cfg_class:
            return "hf_quanto"
        elif "CompressedTensors" in cfg_class or "Compressed" in cfg_class:
            return "hf_ct"
        elif cfg_class in ("BitsAndBytesConfig", "BnbConfig"):
            return "hf_bnb"

    return None


class _DelegatingAdapter(ModelAdapter):
    name = "auto_adapter"

    def __init__(self) -> None:
        self._delegate: _LoadableAdapter | None = None

    def _load_adapter(self, adapter_name: str) -> _LoadableAdapter:
        """Load an adapter by name."""
        if adapter_name == "hf_causal":
            adapter_cls = cast(
                type[_LoadableAdapter],
                _importlib.import_module(".hf_causal", __package__).HF_Causal_Adapter,
            )
            return adapter_cls()
        if adapter_name == "hf_mlm":
            adapter_cls = cast(
                type[_LoadableAdapter],
                _importlib.import_module(".hf_mlm", __package__).HF_MLM_Adapter,
            )
            return adapter_cls()
        if adapter_name == "hf_multimodal":
            adapter_cls = cast(
                type[_LoadableAdapter],
                _importlib.import_module(
                    ".hf_multimodal", __package__
                ).HF_Multimodal_Adapter,
            )
            return adapter_cls()
        if adapter_name == "hf_seq2seq":
            adapter_cls = cast(
                type[_LoadableAdapter],
                _importlib.import_module(".hf_seq2seq", __package__).HF_Seq2Seq_Adapter,
            )
            return adapter_cls()
        elif adapter_name == "hf_bnb":
            adapter_cls = cast(
                type[_LoadableAdapter],
                _importlib.import_module("invarlock.plugins").HF_BNB_Adapter,
            )
            return adapter_cls()
        elif adapter_name == "hf_awq":
            adapter_cls = cast(
                type[_LoadableAdapter],
                _importlib.import_module("invarlock.plugins").HF_AWQ_Adapter,
            )
            return adapter_cls()
        elif adapter_name == "hf_gptq":
            adapter_cls = cast(
                type[_LoadableAdapter],
                _importlib.import_module("invarlock.plugins").HF_GPTQ_Adapter,
            )
            return adapter_cls()
        elif adapter_name == "hf_torchao":
            adapter_cls = cast(
                type[_LoadableAdapter],
                _importlib.import_module("invarlock.plugins").HF_TorchAO_Adapter,
            )
            return adapter_cls()
        elif adapter_name == "hf_hqq":
            adapter_cls = cast(
                type[_LoadableAdapter],
                _importlib.import_module("invarlock.plugins").HF_HQQ_Adapter,
            )
            return adapter_cls()
        elif adapter_name == "hf_quanto":
            adapter_cls = cast(
                type[_LoadableAdapter],
                _importlib.import_module("invarlock.plugins").HF_Quanto_Adapter,
            )
            return adapter_cls()
        elif adapter_name == "hf_ct":
            adapter_cls = cast(
                type[_LoadableAdapter],
                _importlib.import_module(
                    "invarlock.plugins"
                ).HF_CompressedTensors_Adapter,
            )
            return adapter_cls()
        else:
            # Default to causal adapter
            adapter_cls = cast(
                type[_LoadableAdapter],
                _importlib.import_module(".hf_causal", __package__).HF_Causal_Adapter,
            )
            return adapter_cls()

    def _ensure_delegate_from_id(self, model_id: str) -> _LoadableAdapter:
        if self._delegate is not None:
            return self._delegate

        # First check for quantization in local checkpoint
        quant_adapter = _detect_quantization_from_path(model_id)
        if quant_adapter:
            self._delegate = self._load_adapter(quant_adapter)
            return self._delegate

        # Fall back to architecture-based resolution
        resolved = resolve_auto_adapter(model_id)
        self._delegate = self._load_adapter(resolved)
        return self._delegate

    def _ensure_delegate_from_model(self, model: Any) -> _LoadableAdapter:
        if self._delegate is not None:
            return self._delegate

        # First check for quantization on the loaded model
        quant_adapter = _detect_quantization_from_model(model)
        if quant_adapter:
            self._delegate = self._load_adapter(quant_adapter)
            return self._delegate

        # Fall back to lightweight class-name inspection (no transformers import).
        cls_name = getattr(model, "__class__", type(model)).__name__.lower()
        if any(k in cls_name for k in ["bert", "roberta", "albert", "deberta"]):
            self._delegate = self._load_adapter("hf_mlm")
        else:
            cfg = getattr(model, "config", None)
            if getattr(cfg, "is_encoder_decoder", False):
                self._delegate = self._load_adapter("hf_seq2seq")
            else:
                self._delegate = self._load_adapter("hf_causal")
        return self._delegate

    def can_handle(self, model: Any) -> bool:  # pragma: no cover - trivial
        return True

    def describe(self, model: Any) -> dict[str, Any]:
        delegate = self._delegate or self._ensure_delegate_from_model(model)
        return delegate.describe(model)

    def snapshot(self, model: Any) -> bytes:
        delegate = self._delegate or self._ensure_delegate_from_model(model)
        return delegate.snapshot(model)

    def restore(self, model: Any, blob: bytes) -> None:
        delegate = self._delegate or self._ensure_delegate_from_model(model)
        return delegate.restore(model, blob)


class HF_Auto_Adapter(_DelegatingAdapter):
    name = "hf_auto"

    def load_model(self, model_id: str, device: str = "auto", **kwargs: Any) -> Any:
        delegate = self._ensure_delegate_from_id(model_id)
        return delegate.load_model(model_id, device=device, **kwargs)
