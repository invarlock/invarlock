from __future__ import annotations

import importlib as _importlib
from typing import Any

from invarlock.core.api import ModelAdapter

from ..cli.adapter_auto import resolve_auto_adapter


class _DelegatingAdapter(ModelAdapter):
    name = "auto_adapter"

    def __init__(self) -> None:
        self._delegate: ModelAdapter | None = None

    def _ensure_delegate_from_id(self, model_id: str) -> ModelAdapter:
        if self._delegate is not None:
            return self._delegate
        resolved = resolve_auto_adapter(model_id)
        if resolved == "hf_llama":
            HF_LLaMA_Adapter = _importlib.import_module(
                ".hf_llama", __package__
            ).HF_LLaMA_Adapter
            self._delegate = HF_LLaMA_Adapter()
        elif resolved == "hf_bert":
            HF_BERT_Adapter = _importlib.import_module(
                ".hf_bert", __package__
            ).HF_BERT_Adapter
            self._delegate = HF_BERT_Adapter()
        else:
            HF_GPT2_Adapter = _importlib.import_module(
                ".hf_gpt2", __package__
            ).HF_GPT2_Adapter
            self._delegate = HF_GPT2_Adapter()
        return self._delegate

    def _ensure_delegate_from_model(self, model: Any) -> ModelAdapter:
        # Best-effort: inspect class name
        cls_name = getattr(model, "__class__", type(model)).__name__.lower()
        if any(k in cls_name for k in ["llama", "mistral", "qwen", "yi"]):
            HF_LLaMA_Adapter = _importlib.import_module(
                ".hf_llama", __package__
            ).HF_LLaMA_Adapter
            self._delegate = HF_LLaMA_Adapter()
        elif any(k in cls_name for k in ["bert", "roberta", "albert", "deberta"]):
            HF_BERT_Adapter = _importlib.import_module(
                ".hf_bert", __package__
            ).HF_BERT_Adapter
            self._delegate = HF_BERT_Adapter()
        else:
            HF_GPT2_Adapter = _importlib.import_module(
                ".hf_gpt2", __package__
            ).HF_GPT2_Adapter
            self._delegate = HF_GPT2_Adapter()
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
