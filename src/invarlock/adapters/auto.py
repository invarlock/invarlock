from __future__ import annotations

import importlib as _importlib
import json
from pathlib import Path
from typing import Any

from invarlock.core.api import ModelAdapter

from ..cli.adapter_auto import resolve_auto_adapter


def _detect_quantization_from_path(model_id: str) -> str | None:
    """
    Detect quantization method from a local checkpoint path.

    Returns:
        Quantization adapter name ("hf_bnb", "hf_awq", "hf_gptq") or None.
    """
    path = Path(model_id)
    if not path.exists():
        return None

    config_path = path / "config.json"
    if not config_path.exists():
        return None

    try:
        config_data = json.loads(config_path.read_text())
        quant_cfg = config_data.get("quantization_config", {})

        if not quant_cfg:
            return None

        quant_method = quant_cfg.get("quant_method", "").lower()

        if quant_method == "awq":
            return "hf_awq"
        elif quant_method == "gptq":
            return "hf_gptq"
        elif (
            quant_method == "bitsandbytes"
            or quant_cfg.get("load_in_8bit")
            or quant_cfg.get("load_in_4bit")
        ):
            return "hf_bnb"

    except Exception:
        pass

    return None


def _detect_quantization_from_model(model: Any) -> str | None:
    """
    Detect quantization method from a loaded model instance.

    Returns:
        Quantization adapter name ("hf_bnb", "hf_awq", "hf_gptq") or None.
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
        quant_method = quant_cfg.get("quant_method", "").lower()
        if quant_method == "awq":
            return "hf_awq"
        elif quant_method == "gptq":
            return "hf_gptq"
        elif (
            quant_method == "bitsandbytes"
            or quant_cfg.get("load_in_8bit")
            or quant_cfg.get("load_in_4bit")
        ):
            return "hf_bnb"
    else:
        # Object-style config
        cfg_class = quant_cfg.__class__.__name__
        if cfg_class in ("AWQConfig",):
            return "hf_awq"
        elif cfg_class in ("GPTQConfig",):
            return "hf_gptq"
        elif cfg_class in ("BitsAndBytesConfig", "BnbConfig"):
            return "hf_bnb"
        # Check attributes
        if getattr(quant_cfg, "load_in_8bit", False) or getattr(
            quant_cfg, "load_in_4bit", False
        ):
            return "hf_bnb"

    return None


class _DelegatingAdapter(ModelAdapter):
    name = "auto_adapter"

    def __init__(self) -> None:
        self._delegate: ModelAdapter | None = None

    def _load_adapter(self, adapter_name: str) -> ModelAdapter:
        """Load an adapter by name."""
        if adapter_name == "hf_llama":
            HF_LLaMA_Adapter = _importlib.import_module(
                ".hf_llama", __package__
            ).HF_LLaMA_Adapter
            return HF_LLaMA_Adapter()
        elif adapter_name == "hf_bert":
            HF_BERT_Adapter = _importlib.import_module(
                ".hf_bert", __package__
            ).HF_BERT_Adapter
            return HF_BERT_Adapter()
        elif adapter_name == "hf_gpt2":
            HF_GPT2_Adapter = _importlib.import_module(
                ".hf_gpt2", __package__
            ).HF_GPT2_Adapter
            return HF_GPT2_Adapter()
        elif adapter_name == "hf_bnb":
            HF_BNB_Adapter = _importlib.import_module(
                "invarlock.plugins.hf_bnb_adapter"
            ).HF_BNB_Adapter
            return HF_BNB_Adapter()
        elif adapter_name == "hf_awq":
            HF_AWQ_Adapter = _importlib.import_module(
                "invarlock.plugins.hf_awq_adapter"
            ).HF_AWQ_Adapter
            return HF_AWQ_Adapter()
        elif adapter_name == "hf_gptq":
            HF_GPTQ_Adapter = _importlib.import_module(
                "invarlock.plugins.hf_gptq_adapter"
            ).HF_GPTQ_Adapter
            return HF_GPTQ_Adapter()
        else:
            # Default to GPT2 adapter
            HF_GPT2_Adapter = _importlib.import_module(
                ".hf_gpt2", __package__
            ).HF_GPT2_Adapter
            return HF_GPT2_Adapter()

    def _ensure_delegate_from_id(self, model_id: str) -> ModelAdapter:
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

    def _ensure_delegate_from_model(self, model: Any) -> ModelAdapter:
        if self._delegate is not None:
            return self._delegate

        # First check for quantization on the loaded model
        quant_adapter = _detect_quantization_from_model(model)
        if quant_adapter:
            self._delegate = self._load_adapter(quant_adapter)
            return self._delegate

        # Fall back to class name inspection
        cls_name = getattr(model, "__class__", type(model)).__name__.lower()
        if any(k in cls_name for k in ["llama", "mistral", "qwen", "yi"]):
            self._delegate = self._load_adapter("hf_llama")
        elif any(k in cls_name for k in ["bert", "roberta", "albert", "deberta"]):
            self._delegate = self._load_adapter("hf_bert")
        else:
            self._delegate = self._load_adapter("hf_gpt2")
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

    def __getattr__(self, item: str):  # pragma: no cover - passthrough
        if item == "_delegate":
            raise AttributeError(item)
        delegate = self._delegate
        if delegate is not None and hasattr(delegate, item):
            return getattr(delegate, item)
        raise AttributeError(item)


class HF_Causal_Auto_Adapter(_DelegatingAdapter):
    name = "hf_causal_auto"

    def load_model(self, model_id: str, device: str = "auto") -> Any:
        delegate = self._ensure_delegate_from_id(model_id)
        return delegate.load_model(model_id, device=device)


class HF_MLM_Auto_Adapter(_DelegatingAdapter):
    name = "hf_mlm_auto"

    def load_model(self, model_id: str, device: str = "auto") -> Any:
        # Force BERT-like adapter for MLM families
        HF_BERT_Adapter = _importlib.import_module(
            ".hf_bert", __package__
        ).HF_BERT_Adapter
        self._delegate = HF_BERT_Adapter()
        return self._delegate.load_model(model_id, device=device)
