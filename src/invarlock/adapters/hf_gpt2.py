"""
HuggingFace GPT-2 Model Adapter
===============================

ModelAdapter implementation for HuggingFace GPT-2 architecture models.

This adapter provides enhanced HuggingFace integration including:
- Better model detection for HF model variants
- Proper handling of transformers library specifics
- Device-aware state serialization with HF model handling
- Weight tying preservation (lm_head ↔ wte)
- Split size and layer naming convention support
"""

import os
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from invarlock.core.api import ModelAdapter
from invarlock.core.error_utils import wrap_errors
from invarlock.core.exceptions import AdapterError, DependencyError, ModelLoadError

from .hf_mixin import HFAdapterMixin

LIGHT_IMPORT = os.getenv("INVARLOCK_LIGHT_IMPORT", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

TensorType = torch.Tensor
ModuleType = nn.Module


class HF_GPT2_Adapter(HFAdapterMixin, ModelAdapter):
    """
    HuggingFace-specific ModelAdapter implementation for GPT-2 models.

    Supports HuggingFace GPT2Model and GPT2LMHeadModel variants with:
    - Enhanced HF model detection and validation
    - Device-aware state serialization
    - Weight tying preservation across snapshot/restore cycles
    - Proper handling of Conv1D layers and split_size conventions
    """

    name = "hf_gpt2"

    def load_model(self, model_id: str, device: str = "auto") -> ModuleType | Any:
        """
        Load a HuggingFace GPT-2 model.

        Args:
            model_id: Model identifier (e.g. "gpt2", "gpt2-medium")
            device: Target device ("auto", "cuda", "mps", "cpu")

        Returns:
            Loaded GPT-2 model
        """
        # Lazy import to allow dependency mapping; in light-import mode fall back to a stub
        try:
            with wrap_errors(
                DependencyError,
                "E203",
                "DEPENDENCY-MISSING: transformers",
                lambda e: {"dependency": "transformers"},
            ):
                from transformers import AutoModelForCausalLM  # type: ignore

            with wrap_errors(
                ModelLoadError,
                "E201",
                "MODEL-LOAD-FAILED: transformers AutoModelForCausalLM",
                lambda e: {"model_id": model_id},
            ):
                model = AutoModelForCausalLM.from_pretrained(model_id)

            target_device = self._resolve_device(device)
            return model.to(target_device)
        except DependencyError:
            if LIGHT_IMPORT:
                # Minimal stand-in that satisfies downstream interface requirements
                stub = SimpleNamespace(name="hf_gpt2_stub")
                stub.to = lambda *_a, **_k: stub  # type: ignore[attr-defined]
                return stub
            raise

    def can_handle(self, model: ModuleType | Any) -> bool:
        """
        Check if this adapter can handle the given model.

        Enhanced detection for HuggingFace GPT-2 models with validation
        of expected structure and configuration.

        Args:
            model: The model to check

        Returns:
            True if this is a HuggingFace GPT-2 compatible model
        """
        # Check for HuggingFace GPT-2 class names (avoid importing classes at module import time)
        model_name = model.__class__.__name__
        if model_name in ["GPT2Model", "GPT2LMHeadModel"]:
            # Verify it has HF config
            if hasattr(model, "config") and hasattr(model.config, "model_type"):
                return model.config.model_type == "gpt2"

        # Structural validation for GPT-2-like models
        if hasattr(model, "config") and hasattr(model, "transformer"):
            config = model.config
            transformer = model.transformer

            # Check for GPT-2 configuration attributes
            if (
                hasattr(config, "n_layer")
                and hasattr(config, "n_head")
                and hasattr(config, "hidden_size")
                and hasattr(transformer, "h")
            ):
                # Validate transformer structure
                try:
                    h_layers = transformer.h
                    if hasattr(h_layers, "__len__") and len(h_layers) > 0:
                        layer = h_layers[0]
                    elif hasattr(h_layers, "__iter__"):
                        # Handle iterables without len() (like Mock objects in tests)
                        try:
                            layer = next(iter(h_layers))
                        except (StopIteration, TypeError):
                            return False
                    else:
                        return False

                    # Check for GPT-2 layer structure with HF conventions
                    if (
                        hasattr(layer, "attn")
                        and hasattr(layer, "mlp")
                        and hasattr(layer.attn, "c_attn")
                        and hasattr(layer.attn, "c_proj")
                        and hasattr(layer.mlp, "c_fc")
                        and hasattr(layer.mlp, "c_proj")
                    ):
                        return True
                except (AttributeError, TypeError):
                    return False

        # Check for bare GPT2Model structure (less common but possible)
        if hasattr(model, "h") and hasattr(model, "config"):
            if hasattr(model.config, "n_layer") and len(model.h) > 0:
                layer = model.h[0]
                if (
                    hasattr(layer, "attn")
                    and hasattr(layer, "mlp")
                    and hasattr(layer.attn, "c_attn")
                    and hasattr(layer.mlp, "c_fc")
                ):
                    return True

        return False

    def describe(self, model: ModuleType | Any) -> dict[str, Any]:
        """
        Get structural description of the HuggingFace GPT-2 model.

        Returns the required format for validation gates:
        - n_layer: int
        - heads_per_layer: List[int]
        - mlp_dims: List[int]
        - tying: Dict[str, str] (weight tying map)

        Args:
            model: The HuggingFace GPT-2 model to describe

        Returns:
            Dictionary with model structure info in required format
        """
        # Determine model structure
        if hasattr(model, "transformer"):
            # GPT2LMHeadModel structure
            transformer = model.transformer
            layers = transformer.h
            config = model.config
        elif hasattr(model, "h"):
            # Direct GPT2Model structure
            layers = model.h
            config = model.config
            transformer = model
        else:
            raise AdapterError(
                code="E202",
                message=(
                    "ADAPTER-STRUCTURE-INVALID: unrecognized HuggingFace GPT-2 model structure"
                ),
                details={"model_class": model.__class__.__name__},
            )

        # Extract basic configuration
        n_layers = len(layers)
        n_heads = getattr(
            config, "n_head", getattr(config, "num_attention_heads", None)
        )
        hidden_size = getattr(config, "hidden_size", getattr(config, "d_model", None))
        vocab_size = getattr(config, "vocab_size", None)

        if n_heads is None or hidden_size is None:
            raise AdapterError(
                code="E202",
                message=(
                    "ADAPTER-STRUCTURE-INVALID: missing n_heads or hidden_size in config"
                ),
                details={"model_class": model.__class__.__name__},
            )

        # Get device info
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        # Calculate total parameters
        total_params = sum(p.numel() for p in model.parameters())

        # Get MLP dimensions for each layer
        mlp_dims = []
        heads_per_layer = []

        for layer_idx in range(n_layers):
            layer = layers[layer_idx]

            # For GPT-2, all layers have the same head count
            heads_per_layer.append(n_heads)

            # Get MLP intermediate dimension
            # HuggingFace GPT-2 uses Conv1D layers where weight shape is (in_features, out_features)
            if hasattr(layer.mlp.c_fc, "weight"):
                if hasattr(layer.mlp.c_fc, "nf"):  # Conv1D layer
                    mlp_dim = layer.mlp.c_fc.nf  # out_features for Conv1D
                else:
                    # Regular linear layer: (out_features, in_features)
                    mlp_dim = layer.mlp.c_fc.weight.shape[0]
            else:
                # Fallback to config
                mlp_dim = getattr(config, "n_inner", hidden_size * 4)

            mlp_dims.append(mlp_dim)

        # Detect weight tying (lm_head ↔ wte)
        tying_map = {}
        if hasattr(model, "lm_head") and hasattr(transformer, "wte"):
            # Check if the weights are the same tensor (tied)
            if model.lm_head.weight is transformer.wte.weight:
                tying_map["lm_head.weight"] = "transformer.wte.weight"

        # Build the required description format
        description = {
            # Required fields for validation gates
            "n_layer": n_layers,
            "heads_per_layer": heads_per_layer,
            "mlp_dims": mlp_dims,
            "tying": tying_map,  # Use 'tying' instead of 'weight_tying' as per spec
            # Additional useful information
            "model_type": "gpt2",
            "model_class": model.__class__.__name__,
            "n_heads": n_heads,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "total_params": total_params,
            "device": str(device),
            # HuggingFace specific info
            "hf_model_type": getattr(config, "model_type", "gpt2"),
            "hf_config_class": config.__class__.__name__
            if hasattr(config, "__class__")
            else "unknown",
            # Architecture details
            "architecture": {
                "has_lm_head": hasattr(model, "lm_head"),
                "has_transformer_wrapper": hasattr(model, "transformer"),
                "layer_norm_type": "pre",  # GPT-2 uses pre-layer norm
                "activation": getattr(config, "activation_function", "gelu_new"),
                "positional_encoding": "learned",  # GPT-2 uses learned position embeddings
                "use_bias": getattr(config, "use_bias", True),
                "split_size": getattr(config, "split_size", None),
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
        tying_info = {}

        # Check for lm_head ↔ wte tying (most common in GPT-2)
        if hasattr(model, "lm_head") and hasattr(model, "transformer"):
            if hasattr(model.transformer, "wte"):
                if model.lm_head.weight is model.transformer.wte.weight:
                    tying_info["lm_head.weight"] = "transformer.wte.weight"

        # Could be extended for other tying relationships
        return tying_info

    def _restore_weight_tying(
        self, model: ModuleType | Any, tied_param: str, source_param: str
    ) -> None:
        """
        Restore a weight tying relationship between parameters.

        Args:
            model: The model to modify
            tied_param: Name of the parameter that should be tied
            source_param: Name of the source parameter to tie to
        """
        # This is a placeholder for weight tying restoration logic
        # In practice, this would need to handle the specific tying relationships
        # For now, we just warn about broken tying
        print(
            f"Warning: Weight tying relationship {tied_param} -> {source_param} may have been broken during restore"
        )

    def validate_split_size(self, model: ModuleType | Any) -> bool:
        """
        Validate that split_size handling is correct for HuggingFace models.

        Args:
            model: The model to validate

        Returns:
            True if split_size is handled correctly
        """
        if not hasattr(model, "config"):
            return True  # No config to validate

        config = model.config
        split_size = getattr(config, "split_size", None)

        if split_size is None:
            return True  # No split_size specified

        # Validate that c_attn layers respect split_size
        try:
            desc = self.describe(model)
            if desc["n_layer"] > 0:
                # Check first layer as representative
                if hasattr(model, "transformer"):
                    layer = model.transformer.h[0]
                else:
                    layer = model.h[0]

                c_attn = layer.attn.c_attn
                if hasattr(c_attn, "weight"):
                    # For Conv1D: weight shape is (in_features, out_features)
                    # out_features should be 3 * hidden_size for combined Q,K,V
                    expected_out = 3 * desc["hidden_size"]
                    actual_out = (
                        c_attn.weight.shape[1]
                        if hasattr(c_attn, "nf")
                        else c_attn.weight.shape[0]
                    )

                    return actual_out == expected_out

            return True

        except Exception:
            return False

    def get_layer_modules(
        self, model: ModuleType | Any, layer_idx: int
    ) -> dict[str, ModuleType | Any]:
        """
        Get the modules for a specific layer (utility method).

        Args:
            model: The HuggingFace GPT-2 model
            layer_idx: Index of the layer to get modules for

        Returns:
            Dictionary mapping module names to modules
        """
        if hasattr(model, "transformer"):
            layer = model.transformer.h[layer_idx]
        else:
            layer = model.h[layer_idx]

        modules = {
            "attn.c_attn": layer.attn.c_attn,  # Combined Q,K,V projection
            "attn.c_proj": layer.attn.c_proj,  # Output projection
            "mlp.c_fc": layer.mlp.c_fc,  # Feed-forward expansion
            "mlp.c_proj": layer.mlp.c_proj,  # Feed-forward projection
            "ln_1": layer.ln_1,  # Layer norm 1
            "ln_2": layer.ln_2,  # Layer norm 2
        }

        return modules
