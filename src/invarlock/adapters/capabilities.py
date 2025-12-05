"""
Model Capabilities
==================

Dataclasses for declaring model capabilities and quantization configuration.
Used by adapters to advertise model properties that affect device handling,
snapshot/restore behavior, and evaluation strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QuantizationMethod(Enum):
    """Supported quantization methods."""

    NONE = "none"
    BNB_8BIT = "bnb_8bit"
    BNB_4BIT = "bnb_4bit"
    AWQ = "awq"
    GPTQ = "gptq"
    ONNX = "onnx"


@dataclass(frozen=True)
class QuantizationConfig:
    """
    Quantization configuration for a loaded model.

    Attributes:
        method: The quantization method used.
        bits: Bit-width of the quantization (e.g., 4, 8, 16).
        group_size: Group size for grouped quantization (AWQ/GPTQ).
        from_checkpoint: True if model was loaded from pre-quantized checkpoint.
        double_quant: Whether double quantization is enabled (BNB 4-bit).
        compute_dtype: Data type for computation (e.g., "float16", "bfloat16").
    """

    method: QuantizationMethod = QuantizationMethod.NONE
    bits: int = 16
    group_size: int | None = None
    from_checkpoint: bool = False
    double_quant: bool = False
    compute_dtype: str | None = None

    def is_quantized(self) -> bool:
        """Return True if the model is quantized."""
        return self.method != QuantizationMethod.NONE

    def is_bnb(self) -> bool:
        """Return True if using BitsAndBytes quantization."""
        return self.method in (QuantizationMethod.BNB_8BIT, QuantizationMethod.BNB_4BIT)


@dataclass
class ModelCapabilities:
    """
    Declared capabilities of a loaded model.

    Used to inform safe device handling, snapshot/restore strategies,
    and evaluation metric selection.

    Attributes:
        quantization: Quantization configuration (if any).
        device_movable: Whether model.to(device) is safe to call.
            False for BNB models which handle device placement internally.
        weight_tied: Mapping of tied parameter names to their source.
            Example: {"lm_head.weight": "model.embed_tokens.weight"}
        primary_metric_kind: Primary evaluation metric type.
            Examples: "ppl_causal", "ppl_mlm", "accuracy", "bleu".
        supports_kv_cache: Whether model supports key-value caching.
        supports_flash_attention: Whether model supports Flash Attention.
        max_sequence_length: Maximum supported sequence length.
        supports_gradient_checkpointing: Whether model supports gradient checkpointing.
    """

    quantization: QuantizationConfig = field(
        default_factory=lambda: QuantizationConfig()
    )
    device_movable: bool = True
    weight_tied: dict[str, str] = field(default_factory=dict)
    primary_metric_kind: str = "ppl_causal"
    supports_kv_cache: bool = True
    supports_flash_attention: bool = False
    max_sequence_length: int | None = None
    supports_gradient_checkpointing: bool = True

    @classmethod
    def for_fp16_model(cls) -> ModelCapabilities:
        """Create capabilities for a standard FP16 model."""
        return cls(
            quantization=QuantizationConfig(method=QuantizationMethod.NONE, bits=16),
            device_movable=True,
        )

    @classmethod
    def for_bnb_8bit(cls, from_checkpoint: bool = False) -> ModelCapabilities:
        """Create capabilities for a BitsAndBytes 8-bit model."""
        return cls(
            quantization=QuantizationConfig(
                method=QuantizationMethod.BNB_8BIT,
                bits=8,
                from_checkpoint=from_checkpoint,
            ),
            device_movable=False,  # BNB handles device placement
        )

    @classmethod
    def for_bnb_4bit(
        cls, from_checkpoint: bool = False, double_quant: bool = False
    ) -> ModelCapabilities:
        """Create capabilities for a BitsAndBytes 4-bit model."""
        return cls(
            quantization=QuantizationConfig(
                method=QuantizationMethod.BNB_4BIT,
                bits=4,
                from_checkpoint=from_checkpoint,
                double_quant=double_quant,
            ),
            device_movable=False,  # BNB handles device placement
        )

    @classmethod
    def for_awq(
        cls, group_size: int = 128, from_checkpoint: bool = True
    ) -> ModelCapabilities:
        """Create capabilities for an AWQ model."""
        return cls(
            quantization=QuantizationConfig(
                method=QuantizationMethod.AWQ,
                bits=4,
                group_size=group_size,
                from_checkpoint=from_checkpoint,
            ),
            device_movable=False,  # AWQ may have device constraints
        )

    @classmethod
    def for_gptq(
        cls, bits: int = 4, group_size: int = 128, from_checkpoint: bool = True
    ) -> ModelCapabilities:
        """Create capabilities for a GPTQ model."""
        return cls(
            quantization=QuantizationConfig(
                method=QuantizationMethod.GPTQ,
                bits=bits,
                group_size=group_size,
                from_checkpoint=from_checkpoint,
            ),
            device_movable=False,  # GPTQ may have device constraints
        )


def detect_quantization_from_config(config: Any) -> QuantizationConfig:
    """
    Detect quantization configuration from a HuggingFace model config.

    Checks for quantization_config in the model's config and returns
    the appropriate QuantizationConfig.

    Args:
        config: HuggingFace model config object

    Returns:
        QuantizationConfig describing the model's quantization state
    """
    if config is None:
        return QuantizationConfig()

    # Check for quantization_config attribute (BNB, AWQ, GPTQ)
    quant_cfg = getattr(config, "quantization_config", None)
    if quant_cfg is None:
        return QuantizationConfig()

    # Handle dict-style config (common in saved checkpoints)
    if isinstance(quant_cfg, dict):
        quant_method = quant_cfg.get("quant_method", "").lower()
        load_in_8bit = quant_cfg.get("load_in_8bit", False)
        load_in_4bit = quant_cfg.get("load_in_4bit", False)
        bits = quant_cfg.get("bits", 16)
        group_size = quant_cfg.get("group_size")
        double_quant = quant_cfg.get("bnb_4bit_use_double_quant", False)
        compute_dtype = quant_cfg.get("bnb_4bit_compute_dtype")

        if quant_method == "awq":
            return QuantizationConfig(
                method=QuantizationMethod.AWQ,
                bits=bits,
                group_size=group_size,
                from_checkpoint=True,
            )
        elif quant_method == "gptq":
            return QuantizationConfig(
                method=QuantizationMethod.GPTQ,
                bits=bits,
                group_size=group_size,
                from_checkpoint=True,
            )
        elif load_in_8bit or quant_method == "bitsandbytes" and bits == 8:
            return QuantizationConfig(
                method=QuantizationMethod.BNB_8BIT,
                bits=8,
                from_checkpoint=True,
            )
        elif load_in_4bit or quant_method == "bitsandbytes" and bits == 4:
            return QuantizationConfig(
                method=QuantizationMethod.BNB_4BIT,
                bits=4,
                from_checkpoint=True,
                double_quant=double_quant,
                compute_dtype=str(compute_dtype) if compute_dtype else None,
            )

    # Handle object-style config (e.g., BitsAndBytesConfig)
    # Check by class name to avoid import dependency
    cfg_class = quant_cfg.__class__.__name__

    if cfg_class in ("BitsAndBytesConfig", "BnbConfig"):
        load_in_8bit = getattr(quant_cfg, "load_in_8bit", False)
        load_in_4bit = getattr(quant_cfg, "load_in_4bit", False)
        double_quant = getattr(quant_cfg, "bnb_4bit_use_double_quant", False)
        compute_dtype = getattr(quant_cfg, "bnb_4bit_compute_dtype", None)

        if load_in_8bit:
            return QuantizationConfig(
                method=QuantizationMethod.BNB_8BIT,
                bits=8,
                from_checkpoint=True,
            )
        elif load_in_4bit:
            return QuantizationConfig(
                method=QuantizationMethod.BNB_4BIT,
                bits=4,
                from_checkpoint=True,
                double_quant=double_quant,
                compute_dtype=str(compute_dtype) if compute_dtype else None,
            )

    if cfg_class in ("AWQConfig",):
        bits = getattr(quant_cfg, "bits", 4)
        group_size = getattr(quant_cfg, "group_size", 128)
        return QuantizationConfig(
            method=QuantizationMethod.AWQ,
            bits=bits,
            group_size=group_size,
            from_checkpoint=True,
        )

    if cfg_class in ("GPTQConfig",):
        bits = getattr(quant_cfg, "bits", 4)
        group_size = getattr(quant_cfg, "group_size", 128)
        return QuantizationConfig(
            method=QuantizationMethod.GPTQ,
            bits=bits,
            group_size=group_size,
            from_checkpoint=True,
        )

    return QuantizationConfig()


def detect_capabilities_from_model(model: Any) -> ModelCapabilities:
    """
    Detect model capabilities from a loaded model instance.

    Inspects the model's config, state, and structure to determine
    its capabilities including quantization state.

    Args:
        model: Loaded model instance (typically HuggingFace PreTrainedModel)

    Returns:
        ModelCapabilities describing the model's capabilities
    """
    config = getattr(model, "config", None)
    quant_config = detect_quantization_from_config(config)

    # Check for BNB attributes on the model itself (may not be in config)
    # Transformers sets these flags on loaded BNB models even if config.quantization_config
    # doesn't reflect the quantization state (e.g., for saved BNB checkpoints)
    # Note: We check `is True` explicitly to avoid MagicMock truthiness
    if not quant_config.is_quantized():
        is_8bit = getattr(model, "is_loaded_in_8bit", None)
        is_4bit = getattr(model, "is_loaded_in_4bit", None)
        if is_8bit is True:
            quant_config = QuantizationConfig(
                method=QuantizationMethod.BNB_8BIT,
                bits=8,
                from_checkpoint=True,
            )
        elif is_4bit is True:
            quant_config = QuantizationConfig(
                method=QuantizationMethod.BNB_4BIT,
                bits=4,
                from_checkpoint=True,
            )

    # Also check for quantized module types that indicate BNB usage
    # Only attempt this if model has a callable modules() method (torch.nn.Module)
    if not quant_config.is_quantized():
        modules_method = getattr(model, "modules", None)
        if callable(modules_method):
            try:
                for module in modules_method():
                    module_name = module.__class__.__name__
                    if module_name in ("Linear8bitLt", "Linear4bit"):
                        if "8bit" in module_name:
                            quant_config = QuantizationConfig(
                                method=QuantizationMethod.BNB_8BIT,
                                bits=8,
                                from_checkpoint=True,
                            )
                        else:
                            quant_config = QuantizationConfig(
                                method=QuantizationMethod.BNB_4BIT,
                                bits=4,
                                from_checkpoint=True,
                            )
                        break
            except (TypeError, StopIteration):
                pass

    # Determine if device is movable
    device_movable = not quant_config.is_bnb()

    # For AWQ/GPTQ, check if model has been quantized in a way that
    # prevents device movement
    if quant_config.method in (QuantizationMethod.AWQ, QuantizationMethod.GPTQ):
        # These are typically loaded on-device and shouldn't be moved
        device_movable = False

    # Detect weight tying
    weight_tied = _detect_weight_tying(model)

    # Detect primary metric kind
    primary_metric = _detect_primary_metric(model)

    # Detect other capabilities
    max_seq_len = getattr(config, "max_position_embeddings", None)
    supports_flash = (
        getattr(config, "_attn_implementation", None) == "flash_attention_2"
    )

    return ModelCapabilities(
        quantization=quant_config,
        device_movable=device_movable,
        weight_tied=weight_tied,
        primary_metric_kind=primary_metric,
        max_sequence_length=max_seq_len,
        supports_flash_attention=supports_flash,
    )


def _detect_weight_tying(model: Any) -> dict[str, str]:
    """Detect weight tying relationships in the model."""
    tying: dict[str, str] = {}

    # Common weight tying patterns
    # LLaMA/Mistral: lm_head.weight ↔ model.embed_tokens.weight
    if hasattr(model, "lm_head") and hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "embed_tokens"):
            lm_head_weight = getattr(model.lm_head, "weight", None)
            embed_weight = getattr(inner.embed_tokens, "weight", None)
            if lm_head_weight is not None and embed_weight is not None:
                if lm_head_weight is embed_weight:
                    tying["lm_head.weight"] = "model.embed_tokens.weight"

    # GPT-2: lm_head.weight ↔ transformer.wte.weight
    if hasattr(model, "lm_head") and hasattr(model, "transformer"):
        xformer = model.transformer
        if hasattr(xformer, "wte"):
            lm_head_weight = getattr(model.lm_head, "weight", None)
            wte_weight = getattr(xformer.wte, "weight", None)
            if lm_head_weight is not None and wte_weight is not None:
                if lm_head_weight is wte_weight:
                    tying["lm_head.weight"] = "transformer.wte.weight"

    return tying


def _detect_primary_metric(model: Any) -> str:
    """Detect the primary evaluation metric type for this model."""
    config = getattr(model, "config", None)
    if config is None:
        return "ppl_causal"

    model_type = getattr(config, "model_type", "").lower()
    architectures = getattr(config, "architectures", []) or []
    arch_str = " ".join(architectures).lower()

    # Encoder-only models (BERT-like)
    if any(k in model_type for k in ["bert", "roberta", "albert", "deberta"]):
        if "masked" in arch_str or "mlm" in arch_str:
            return "ppl_mlm"
        if "classification" in arch_str or "sequence" in arch_str:
            return "accuracy"
        return "ppl_mlm"

    # Encoder-decoder models (T5-like)
    if any(k in model_type for k in ["t5", "bart", "marian", "pegasus"]):
        if "translation" in arch_str or "mt" in arch_str:
            return "bleu"
        if "summarization" in arch_str:
            return "rouge"
        return "ppl_seq2seq"

    # Decoder-only models (GPT-like, LLaMA-like)
    return "ppl_causal"


__all__ = [
    "QuantizationMethod",
    "QuantizationConfig",
    "ModelCapabilities",
    "detect_quantization_from_config",
    "detect_capabilities_from_model",
]
