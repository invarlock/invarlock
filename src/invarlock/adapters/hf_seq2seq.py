"""
HuggingFace encoder-decoder adapter.
===================================

ModelAdapter implementation for HuggingFace encoder-decoder (seq2seq) models.

Loads AutoModelForSeq2SeqLM and exposes a minimal describe() sufficient for
guard policies and reporting.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from invarlock.core import INVARLOCK_CORE_ABI as CORE_ABI
from invarlock.core.api import ModelAdapter
from invarlock.core.error_utils import wrap_errors
from invarlock.core.exceptions import AdapterError, DependencyError, ModelLoadError

from .hf_loading import resolve_core_loader_strategy
from .hf_mixin import HFAdapterMixin

INVARLOCK_CORE_ABI = CORE_ABI

TensorType = torch.Tensor
ModuleType = nn.Module
_ALLOW_DIRECT_SUBMODULE = False
_SEQ2SEQ_PROBE_ERRORS = (
    AttributeError,
    LookupError,
    RuntimeError,
    StopIteration,
    TypeError,
    ValueError,
)


class HF_Seq2Seq_Adapter(HFAdapterMixin, ModelAdapter):
    """HuggingFace encoder-decoder adapter using AutoModelForSeq2SeqLM."""

    name = "hf_seq2seq"

    def load_model(self, model_id: str, device: str = "auto", **kwargs: Any) -> Any:
        with wrap_errors(
            DependencyError,
            "E203",
            "DEPENDENCY-MISSING: transformers",
            lambda e: {"dependency": "transformers"},
        ):
            strategy = resolve_core_loader_strategy(
                task="seq2seq",
                model_id=model_id,
                kwargs=kwargs,
                allow_direct_submodule=_ALLOW_DIRECT_SUBMODULE,
            )
            auto_strategy = (
                strategy
                if strategy.strategy == "auto"
                else resolve_core_loader_strategy(
                    task="seq2seq",
                    model_id=model_id,
                    kwargs=kwargs,
                    allow_direct_submodule=False,
                )
            )
        self._last_loader_strategy = strategy.strategy
        self._last_loader_label = strategy.loader_label
        try:
            with wrap_errors(
                ModelLoadError,
                "E201",
                f"MODEL-LOAD-FAILED: {strategy.loader_label}",
                lambda e: {"model_id": model_id},
            ):
                model = self._load_pretrained_model(
                    strategy.loader,
                    model_id,
                    load_device=device,
                    **kwargs,
                )
        except ModelLoadError:
            if strategy.strategy == "auto":
                raise
            self._last_loader_strategy = auto_strategy.strategy
            self._last_loader_label = auto_strategy.loader_label
            with wrap_errors(
                ModelLoadError,
                "E201",
                f"MODEL-LOAD-FAILED: {auto_strategy.loader_label}",
                lambda e: {"model_id": model_id},
            ):
                model = self._load_pretrained_model(
                    auto_strategy.loader,
                    model_id,
                    load_device=device,
                    **kwargs,
                )
        return self._safe_to_device(model, device)

    def can_handle(self, model: Any) -> bool:
        cfg = getattr(model, "config", None)
        if cfg is None:
            return False
        try:
            mt = str(getattr(cfg, "model_type", "")).lower()
        except _SEQ2SEQ_PROBE_ERRORS:
            mt = ""
        if mt == "t5":
            return True
        # Heuristic: encoder-decoder with lm_head and shared embedding
        return bool(
            getattr(cfg, "is_encoder_decoder", False)
            and hasattr(model, "lm_head")
            and hasattr(model, "shared")
        )

    def describe(self, model: Any) -> dict[str, Any]:
        cfg = getattr(model, "config", None)
        if cfg is None:
            raise AdapterError(
                code="E202",
                message="ADAPTER-STRUCTURE-INVALID: missing HuggingFace config on model",
                details={"model_class": model.__class__.__name__},
            )

        # Extract key dimensions with safe fallbacks
        enc_layers = int(
            getattr(cfg, "num_layers", getattr(cfg, "num_encoder_layers", 0)) or 0
        )
        dec_layers = int(getattr(cfg, "num_decoder_layers", enc_layers) or 0)
        n_layer = int(enc_layers + dec_layers or max(enc_layers, dec_layers))
        n_heads = int(
            getattr(cfg, "num_heads", getattr(cfg, "num_attention_heads", 0)) or 0
        )
        d_model = int(getattr(cfg, "d_model", getattr(cfg, "hidden_size", 0)) or 0)
        d_ff = int(getattr(cfg, "d_ff", (d_model * 4) if d_model else 0) or 0)
        vocab_size = int(getattr(cfg, "vocab_size", 0) or 0)

        try:
            device = next(model.parameters()).device
        except _SEQ2SEQ_PROBE_ERRORS:
            device = torch.device("cpu")

        heads_per_layer = [n_heads] * max(1, n_layer)
        mlp_dims = [d_ff] * max(1, n_layer)

        # Weight tying: T5 ties lm_head ↔ shared embedding
        tying_map: dict[str, str] = {}
        try:
            if hasattr(model, "lm_head") and hasattr(model, "shared"):
                if getattr(model.lm_head, "weight", None) is getattr(
                    model.shared, "weight", object()
                ):
                    tying_map["lm_head.weight"] = "shared.weight"
        except _SEQ2SEQ_PROBE_ERRORS:
            pass

        total_params = 0
        try:
            total_params = sum(p.numel() for p in model.parameters())
        except _SEQ2SEQ_PROBE_ERRORS:
            total_params = 0

        return {
            "n_layer": int(max(1, n_layer)),
            "heads_per_layer": heads_per_layer,
            "mlp_dims": mlp_dims,
            "tying": tying_map,
            "model_type": "t5",
            "model_class": model.__class__.__name__,
            "n_heads": n_heads,
            "hidden_size": d_model,
            "vocab_size": vocab_size,
            "total_params": total_params,
            "device": str(device),
        }

    def prepare_generation_inputs(
        self, batch: dict[str, Any], device: torch.device
    ) -> dict[str, torch.Tensor]:
        input_ids = batch.get("input_ids", batch.get("inputs"))
        if input_ids is None:
            return {}

        prepared: dict[str, torch.Tensor] = {
            "input_ids": self._batch_tensor(input_ids, device=device)
        }
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            prepared["attention_mask"] = self._batch_tensor(
                attention_mask,
                device=device,
            )
        labels = batch.get("labels")
        if labels is not None:
            prepared["labels"] = self._batch_tensor(labels, device=device)
        return prepared

    @staticmethod
    def _batch_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.to(device)

    # snapshot/restore provided by HFAdapterMixin
    def snapshot(self, model: ModuleType) -> bytes:
        return super().snapshot(model)

    def restore(self, model: ModuleType, blob: bytes) -> None:
        return super().restore(model, blob)


__all__ = ["HF_Seq2Seq_Adapter"]
