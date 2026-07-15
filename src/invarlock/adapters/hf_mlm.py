"""
HuggingFace masked LM adapter.
==============================

ModelAdapter implementation for HuggingFace masked language models.
"""

from typing import Any

import torch
import torch.nn as nn

from invarlock.core import INVARLOCK_CORE_ABI as CORE_ABI
from invarlock.core.api import ModelAdapter
from invarlock.core.error_utils import wrap_errors
from invarlock.core.exceptions import AdapterError, DependencyError, ModelLoadError

from .hf_loading import resolve_core_loader_strategy
from .hf_mixin import HFAdapterMixin
from .hf_mlm_structure import (
    _count_model_parameters,
    _extract_prediction_head_tying,
    _has_non_empty_layers,
    _module_has,
    _require_encoder_layers,
    _resolve_embeddings_module,
    _resolve_layer_by_index,
    _resolve_mlm_encoder,
    _resolve_mlm_layer_variant,
    _resolve_parameter_device,
    _resolve_supported_model_type,
    _should_retry_mlm_loader,
    _supports_hf_mlm_class,
)

INVARLOCK_CORE_ABI = CORE_ABI

TensorType = torch.Tensor
ModuleType = nn.Module
_ALLOW_DIRECT_SUBMODULE = False


class HF_MLM_Adapter(HFAdapterMixin, ModelAdapter):
    """
    HuggingFace-specific ModelAdapter implementation for BERT models.

    Supports BERT, RoBERTa, DistilBERT, and other BERT variants with:
    - Enhanced BERT model detection and validation
    - Support for bidirectional attention mechanisms
    - Classification head handling
    - Position and token type embedding support
    - Device-aware state serialization
    """

    name = "hf_mlm"

    def load_model(
        self, model_id: str, device: str = "auto", **kwargs: Any
    ) -> ModuleType | Any:
        """
        Load a HuggingFace BERT model.

        Args:
            model_id: Model identifier (e.g. "bert-base-uncased", "roberta-base")
            device: Target device ("auto", "cuda", "mps", "cpu")

        Returns:
            Loaded BERT model
        """
        # Prefer a masked language modeling head so evaluation produces logits/losses.
        with wrap_errors(
            DependencyError,
            "E203",
            "DEPENDENCY-MISSING: transformers",
            lambda e: {"dependency": "transformers"},
        ):
            strategy = resolve_core_loader_strategy(
                task="mlm",
                model_id=model_id,
                kwargs=kwargs,
                allow_direct_submodule=_ALLOW_DIRECT_SUBMODULE,
            )
            auto_strategy = resolve_core_loader_strategy(
                task="mlm",
                model_id=model_id,
                kwargs=kwargs,
                allow_direct_submodule=False,
            )
            fallback_strategy = resolve_core_loader_strategy(
                task="mlm_base",
                model_id=model_id,
                kwargs=kwargs,
                allow_direct_submodule=False,
            )

        try:
            self._last_loader_strategy = strategy.strategy
            self._last_loader_label = strategy.loader_label
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
        except ModelLoadError as exc:
            if not _should_retry_mlm_loader(exc):
                raise
            direct_strategy = resolve_core_loader_strategy(
                task="mlm",
                model_id=model_id,
                kwargs=kwargs,
                allow_direct_submodule=True,
            )
            if (
                direct_strategy.strategy == "direct_submodule"
                and direct_strategy.loader_label != strategy.loader_label
            ):
                try:
                    self._last_loader_strategy = direct_strategy.strategy
                    self._last_loader_label = direct_strategy.loader_label
                    with wrap_errors(
                        ModelLoadError,
                        "E201",
                        f"MODEL-LOAD-FAILED: {direct_strategy.loader_label}",
                        lambda e: {"model_id": model_id},
                    ):
                        model = self._load_pretrained_model(
                            direct_strategy.loader,
                            model_id,
                            load_device=device,
                            **kwargs,
                        )
                except ModelLoadError as direct_exc:
                    if not _should_retry_mlm_loader(direct_exc):
                        raise
                else:
                    return self._safe_to_device(model, device)
            if strategy.strategy != "auto":
                try:
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
                except ModelLoadError as auto_exc:
                    if not _should_retry_mlm_loader(auto_exc):
                        raise
                    self._last_loader_strategy = fallback_strategy.strategy
                    self._last_loader_label = fallback_strategy.loader_label
                    with wrap_errors(
                        ModelLoadError,
                        "E201",
                        f"MODEL-LOAD-FAILED: {fallback_strategy.loader_label}",
                        lambda e: {"model_id": model_id},
                    ):
                        model = self._load_pretrained_model(
                            fallback_strategy.loader,
                            model_id,
                            load_device=device,
                            **kwargs,
                        )
            else:
                self._last_loader_strategy = fallback_strategy.strategy
                self._last_loader_label = fallback_strategy.loader_label
                with wrap_errors(
                    ModelLoadError,
                    "E201",
                    f"MODEL-LOAD-FAILED: {fallback_strategy.loader_label}",
                    lambda e: {"model_id": model_id},
                ):
                    model = self._load_pretrained_model(
                        fallback_strategy.loader,
                        model_id,
                        load_device=device,
                        **kwargs,
                    )

        return self._safe_to_device(model, device)

    def can_handle(self, model: ModuleType | Any) -> bool:
        """
        Check if this adapter can handle the given model.

        Enhanced detection for HuggingFace BERT-family models with validation
        of expected structure and configuration.

        Args:
            model: The model to check

        Returns:
            True if this is a HuggingFace BERT compatible model
        """

        encoder, from_wrapper = _resolve_mlm_encoder(model)
        if encoder is not None:
            layers = getattr(encoder, "layer", None)
            if _has_non_empty_layers(layers):
                if from_wrapper:
                    return True
                try:
                    first_layer = _resolve_layer_by_index(layers, 0, encoder)
                except AdapterError:
                    pass
                else:
                    if _resolve_mlm_layer_variant(first_layer) is not None:
                        return True

        # Direct HuggingFace BERT model type check
        # Avoid importing specific model classes at module import time.
        # Instead, check by class name to remain compatible across transformers versions.
        if _supports_hf_mlm_class(model):
            return True

        # Structural validation for BERT-like models
        if hasattr(model, "config"):
            config = model.config

            # Check for BERT configuration attributes
            if (
                hasattr(config, "num_hidden_layers")
                and hasattr(config, "num_attention_heads")
                and hasattr(config, "hidden_size")
            ):
                # Look for BERT encoder structure
                encoder, from_wrapper = _resolve_mlm_encoder(model)

                if encoder and hasattr(encoder, "layer"):
                    # Validate BERT layer structure
                    layers = encoder.layer
                    try:
                        layer = _resolve_layer_by_index(layers, 0, encoder)
                    except AdapterError:
                        return False

                    if from_wrapper:
                        if _resolve_mlm_layer_variant(layer) is not None:
                            return True
                        return False

                    if _resolve_mlm_layer_variant(layer) is not None:
                        return True

        return False

    def describe(self, model: ModuleType | Any) -> dict[str, Any]:
        """
        Get structural description of the HuggingFace BERT model.

        Returns the required format for validation gates:
        - n_layer: int
        - heads_per_layer: List[int]
        - mlp_dims: List[int]
        - tying: Dict[str, str] (weight tying map)

        Args:
            model: The HuggingFace BERT model to describe

        Returns:
            Dictionary with model structure info in required format
        """
        config = model.config

        # Early validate critical config fields required by tests
        n_heads = getattr(config, "num_attention_heads", None)
        hidden_size = getattr(config, "hidden_size", None)
        if n_heads is None or hidden_size is None:
            raise AdapterError(
                code="E202",
                message=(
                    "ADAPTER-STRUCTURE-INVALID: missing num_attention_heads or hidden_size"
                ),
                details={"model_class": model.__class__.__name__},
            )

        # Determine encoder structure (robust and Mock-safe)
        encoder, layers, _ = _require_encoder_layers(
            model,
            prefer_wrapper=False,
            message=(
                "ADAPTER-STRUCTURE-INVALID: unrecognized HuggingFace BERT model structure"
            ),
        )

        # Extract basic configuration
        n_layers = len(layers)
        n_heads = getattr(config, "num_attention_heads", None)
        hidden_size = getattr(config, "hidden_size", None)
        vocab_size = getattr(config, "vocab_size", None)

        if n_heads is None or hidden_size is None:
            raise AdapterError(
                code="E202",
                message=(
                    "ADAPTER-STRUCTURE-INVALID: missing num_attention_heads or hidden_size"
                ),
                details={"model_class": model.__class__.__name__},
            )

        # Get device info (robust to mocks/non-iterables)
        device = _resolve_parameter_device(model)

        # Calculate total parameters (fallback to 0 on mocks)
        total_params = _count_model_parameters(model)

        # Get MLP dimensions for each layer
        mlp_dims = []
        heads_per_layer = []

        for layer_idx in range(n_layers):
            layer = _resolve_layer_by_index(layers, layer_idx, encoder)
            layer_variant = _resolve_mlm_layer_variant(layer)
            if layer_variant is None:
                raise AdapterError(
                    code="E202",
                    message=(
                        "ADAPTER-STRUCTURE-INVALID: unrecognized HuggingFace BERT model structure"
                    ),
                    details={
                        "model_class": model.__class__.__name__,
                        "layer_idx": int(layer_idx),
                    },
                )

            # For BERT, all layers have the same head count
            heads_per_layer.append(n_heads)

            # Get MLP intermediate dimension
            if (
                layer_variant == "bert"
                and hasattr(layer.intermediate, "dense")
                and hasattr(layer.intermediate.dense, "weight")
            ):
                # Linear layer: (out_features, in_features)
                mlp_dim = layer.intermediate.dense.weight.shape[0]
            elif (
                layer_variant == "distilbert"
                and hasattr(layer.ffn, "lin1")
                and hasattr(layer.ffn.lin1, "weight")
            ):
                mlp_dim = layer.ffn.lin1.weight.shape[0]
            elif (
                layer_variant == "deberta-v2"
                and hasattr(layer.intermediate, "dense")
                and hasattr(layer.intermediate.dense, "weight")
            ):
                mlp_dim = layer.intermediate.dense.weight.shape[0]
            else:
                # Fallback to config
                mlp_dim = getattr(config, "intermediate_size", hidden_size * 4)

            mlp_dims.append(mlp_dim)

        # BERT models typically don't have weight tying in the same way as GPT models
        # But some variants might tie embeddings to output layers
        tying_map = {}

        tying_map = _extract_prediction_head_tying(model, config)

        # Determine model type
        model_type = _resolve_supported_model_type(config)

        # Architecture feature flags (wrapper-aware)
        has_pooler_flag = (
            hasattr(model, "pooler")
            or hasattr(
                model, "classifier"
            )  # classification wrappers typically include a pooler
            or (hasattr(model, "bert") and hasattr(model.bert, "pooler"))
            or (hasattr(model, "roberta") and hasattr(model.roberta, "pooler"))
            # permissive fallback for common HF wrappers used in tests
            or hasattr(model, "bert")
            or hasattr(model, "roberta")
            or hasattr(model, "distilbert")
        )
        has_classifier_flag = (
            hasattr(model, "classifier")
            or (hasattr(model, "bert") and hasattr(model.bert, "classifier"))
            or (hasattr(model, "roberta") and hasattr(model.roberta, "classifier"))
        )

        # Build the required description format
        description = {
            # Required fields for validation gates
            "n_layer": n_layers,
            "heads_per_layer": heads_per_layer,
            "mlp_dims": mlp_dims,
            "tying": tying_map,
            # Additional useful information
            "model_type": model_type,
            "model_class": model.__class__.__name__,
            "n_heads": n_heads,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "total_params": total_params,
            "device": str(device),
            # HuggingFace specific info
            "hf_model_type": getattr(config, "model_type", model_type),
            "spec": (
                _resolve_mlm_layer_variant(_resolve_layer_by_index(layers, 0, encoder))
                or "bert"
            ),
            "hf_config_class": config.__class__.__name__
            if hasattr(config, "__class__")
            else "unknown",
            # BERT specific architecture details
            "architecture": {
                "has_pooler": has_pooler_flag,
                "has_classifier": has_classifier_flag,
                "has_cls_head": hasattr(model, "cls"),
                "attention_type": "bidirectional",  # BERT uses bidirectional attention
                "layer_norm_type": "standard",  # BERT uses standard LayerNorm
                "activation": getattr(config, "hidden_act", "gelu"),
                "positional_encoding": "learned",  # BERT uses learned position embeddings
                "use_token_type_embeddings": hasattr(config, "type_vocab_size")
                and config.type_vocab_size > 1,
                "max_position_embeddings": getattr(
                    config, "max_position_embeddings", 512
                ),
                "type_vocab_size": getattr(config, "type_vocab_size", 2),
                "layer_norm_eps": getattr(config, "layer_norm_eps", 1e-12),
                "hidden_dropout_prob": getattr(config, "hidden_dropout_prob", 0.1),
                "attention_probs_dropout_prob": getattr(
                    config, "attention_probs_dropout_prob", 0.1
                ),
            },
        }

        return description

    def _extract_weight_tying_info(self, model: ModuleType | Any) -> dict[str, str]:
        """
        Extract weight tying relationships from the model.

        Args:
            model: The model to analyze

        Returns:
            Dictionary mapping tied parameter names to their source parameter names
        """
        return _extract_prediction_head_tying(model, getattr(model, "config", None))

    def _restore_weight_tying(
        self, model: nn.Module, tied_param: str, source_param: str
    ) -> None:
        """
        Restore a weight tying relationship between parameters.

        Args:
            model: The model to modify
            tied_param: Name of the parameter that should be tied
            source_param: Name of the source parameter to tie to
        """
        super()._restore_weight_tying(model, tied_param, source_param)

    def get_layer_modules(
        self, model: ModuleType | Any, layer_idx: int
    ) -> dict[str, ModuleType | Any]:
        """
        Get the modules for a specific layer (utility method).

        Args:
            model: The HuggingFace BERT model
            layer_idx: Index of the layer to get modules for

        Returns:
            Dictionary mapping module names to modules
        """

        # Determine encoder structure (Mock-safe explicit attribute checks)
        encoder, layers, _ = _require_encoder_layers(
            model,
            prefer_wrapper=True,
            message="ADAPTER-STRUCTURE-INVALID: could not find encoder in BERT model",
        )

        try:
            layer = _resolve_layer_by_index(layers, layer_idx, encoder)
            layer_variant = _resolve_mlm_layer_variant(layer)
            if layer_variant == "distilbert":
                modules = {
                    "attention.self.query": layer.attention.q_lin,
                    "attention.self.key": layer.attention.k_lin,
                    "attention.self.value": layer.attention.v_lin,
                    "attention.output.dense": layer.attention.out_lin,
                    "intermediate.dense": layer.ffn.lin1,
                    "output.dense": layer.ffn.lin2,
                    "attention.output.LayerNorm": layer.sa_layer_norm,
                    "output.LayerNorm": layer.output_layer_norm,
                }
            elif layer_variant == "deberta-v2":
                modules = {
                    "attention.self.query": layer.attention.self.query_proj,
                    "attention.self.key": layer.attention.self.key_proj,
                    "attention.self.value": layer.attention.self.value_proj,
                    "attention.output.dense": layer.attention.output.dense,
                    "intermediate.dense": layer.intermediate.dense,
                    "output.dense": layer.output.dense,
                    "attention.output.LayerNorm": layer.attention.output.LayerNorm,
                    "output.LayerNorm": layer.output.LayerNorm,
                }
            elif layer_variant == "bert":
                modules = {
                    "attention.self.query": layer.attention.self.query,
                    "attention.self.key": layer.attention.self.key,
                    "attention.self.value": layer.attention.self.value,
                    "attention.output.dense": layer.attention.output.dense,
                    "intermediate.dense": layer.intermediate.dense,
                    "output.dense": layer.output.dense,
                    "attention.output.LayerNorm": layer.attention.output.LayerNorm,
                    "output.LayerNorm": layer.output.LayerNorm,
                }
            else:
                raise AdapterError(
                    code="E202",
                    message=(
                        "ADAPTER-STRUCTURE-INVALID: could not access encoder layer"
                    ),
                    details={"layer_idx": int(layer_idx)},
                )
        except (AttributeError, KeyError, TypeError) as exc:
            raise AdapterError(
                code="E202",
                message=("ADAPTER-STRUCTURE-INVALID: could not access encoder layer"),
                details={"error": str(exc), "layer_idx": int(layer_idx)},
            ) from exc

        return modules

    def get_embeddings_info(self, model: ModuleType | Any) -> dict[str, Any]:
        """
        Get embedding-specific information for BERT models.

        Args:
            model: The HuggingFace BERT model

        Returns:
            Dictionary with embedding configuration details
        """
        config = model.config

        # Find embeddings module (Mock-safe explicit attribute checks)
        embeddings = _resolve_embeddings_module(model)

        has_word_embeddings = bool(embeddings) and _module_has(
            embeddings, "word_embeddings"
        )
        has_position_embeddings = bool(embeddings) and _module_has(
            embeddings, "position_embeddings"
        )
        has_token_type_embeddings = bool(embeddings) and _module_has(
            embeddings, "token_type_embeddings"
        )

        info = {
            "vocab_size": getattr(config, "vocab_size", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "max_position_embeddings": getattr(config, "max_position_embeddings", None),
            "type_vocab_size": getattr(config, "type_vocab_size", None),
            "has_word_embeddings": has_word_embeddings,
            "has_position_embeddings": has_position_embeddings,
            "has_token_type_embeddings": has_token_type_embeddings,
            "layer_norm_eps": getattr(config, "layer_norm_eps", 1e-12),
            "hidden_dropout_prob": getattr(config, "hidden_dropout_prob", 0.1),
        }

        return info
