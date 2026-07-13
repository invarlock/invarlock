"""
Model Capabilities
==================

Dataclasses for declaring model capabilities and quantization configuration.
Used by adapters to advertise model properties that affect device handling,
snapshot/restore behavior, and evaluation strategies.
"""

from __future__ import annotations

from collections.abc import Mapping
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
    TORCHAO_INT8 = "torchao_int8"
    HQQ = "hqq"
    QUANTO = "quanto"
    COMPRESSED_TENSORS = "compressed_tensors"


@dataclass(frozen=True)
class QuantizationConfig:
    """
    Quantization configuration for a loaded model.

    Attributes:
        method: The quantization method used.
        bits: Bit-width of the quantization (e.g., 4, 8, 16), or ``None``
            when a backend config does not declare one unambiguously.
        group_size: Group size for grouped quantization (AWQ/GPTQ).
        from_checkpoint: True if model was loaded from pre-quantized checkpoint.
        double_quant: Whether double quantization is enabled (BNB 4-bit).
        compute_dtype: Data type for computation (e.g., "float16", "bfloat16").
    """

    method: QuantizationMethod = QuantizationMethod.NONE
    bits: int | None = 16
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

    @classmethod
    def for_torchao_int8(cls) -> ModelCapabilities:
        """Create capabilities for a torchao int8 weight-only runtime model."""
        return cls(
            quantization=QuantizationConfig(
                method=QuantizationMethod.TORCHAO_INT8,
                bits=8,
                from_checkpoint=False,
            ),
            device_movable=False,
        )

    @classmethod
    def for_hqq(
        cls,
        bits: int = 4,
        group_size: int | None = 64,
        from_checkpoint: bool = False,
    ) -> ModelCapabilities:
        """Create capabilities for a HQQ runtime-quantized model."""
        return cls(
            quantization=QuantizationConfig(
                method=QuantizationMethod.HQQ,
                bits=bits,
                group_size=group_size,
                from_checkpoint=from_checkpoint,
            ),
            device_movable=False,
        )

    @classmethod
    def for_quanto(
        cls,
        bits: int = 8,
        from_checkpoint: bool = False,
    ) -> ModelCapabilities:
        """Create capabilities for a Quanto runtime-quantized model."""
        return cls(
            quantization=QuantizationConfig(
                method=QuantizationMethod.QUANTO,
                bits=bits,
                from_checkpoint=from_checkpoint,
            ),
            device_movable=False,
        )

    @classmethod
    def for_compressed_tensors(
        cls,
        bits: int | None = None,
        group_size: int | None = None,
        from_checkpoint: bool = True,
    ) -> ModelCapabilities:
        """Create capabilities for a compressed-tensors checkpoint model."""
        return cls(
            quantization=QuantizationConfig(
                method=QuantizationMethod.COMPRESSED_TENSORS,
                bits=bits,
                group_size=group_size,
                from_checkpoint=from_checkpoint,
            ),
            device_movable=False,
        )


def _lower_string(value: Any) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _coerce_positive_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        resolved = value
    elif isinstance(value, float | str):
        try:
            resolved = int(value)
        except (TypeError, ValueError, OverflowError):
            return default
    else:
        return default
    return resolved if resolved > 0 else default


def _bits_from_weight_spec(value: Any, *, default: int = 8) -> int:
    weights = _lower_string(value)
    if "int4" in weights:
        return 4
    if "int8" in weights:
        return 8
    return default


def _is_compressed_tensors_method(value: Any) -> bool:
    normalized = _lower_string(value).replace("-", "_")
    return "compressed_tensors" in normalized or "compressedtensors" in normalized


_MISSING = object()


def _serialized_config_mapping(
    value: Any,
    *,
    _seen: frozenset[int] = frozenset(),
    _depth: int = 0,
) -> Mapping[str, Any] | None:
    """Return a serialized config mapping without guessing third-party fields."""
    if _depth > 1 or id(value) in _seen:
        return None
    if isinstance(value, Mapping):
        return value

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            serialized = to_dict()
        except Exception:  # Third-party optional config objects must fail closed.
            serialized = None
        if isinstance(serialized, Mapping):
            return serialized

    nested = getattr(value, "quantization_config", _MISSING)
    if nested is not _MISSING and nested is not value:
        return _serialized_config_mapping(
            nested,
            _seen=_seen | {id(value)},
            _depth=_depth + 1,
        )
    return None


def _config_field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, _MISSING)
    return getattr(value, name, _MISSING)


def _declared_values(value: Any, names: tuple[str, ...]) -> list[Any]:
    return [
        candidate
        for name in names
        if (candidate := _config_field(value, name)) is not _MISSING
    ]


def _uniform_positive_int(values: list[Any]) -> int | None:
    """Return one declared value only when every supplied value agrees."""
    if not values:
        return None
    normalized = [_coerce_positive_int(value, default=0) for value in values]
    if any(value <= 0 for value in normalized):
        return None
    unique = set(normalized)
    return next(iter(unique)) if len(unique) == 1 else None


def _compressed_tensors_group_values(
    config: Mapping[str, Any],
    *,
    names: tuple[str, ...],
) -> tuple[list[Any], bool]:
    """Collect declared weight metadata and mark opaque group layouts ambiguous."""
    groups = config.get("config_groups", _MISSING)
    if groups is _MISSING or groups is None:
        return [], False
    if isinstance(groups, Mapping):
        candidates = list(groups.values())
    elif isinstance(groups, (list, tuple)):
        candidates = list(groups)
    else:
        return [], True

    values: list[Any] = []
    saw_weight_scheme = False
    for group in candidates:
        weights = _config_field(group, "weights")
        if weights is _MISSING or weights is None:
            continue
        saw_weight_scheme = True
        declared = _declared_values(weights, names)
        values.extend(declared if declared else [_MISSING])
    return values, saw_weight_scheme


def _compressed_tensors_declared_metadata(
    value: Any,
) -> tuple[int | None, int | None]:
    """Read globally uniform packed weight metadata from compressed-tensors.

    Transformers' ``CompressedTensorsConfig`` keeps the actual scheme inside a
    nested optional-package object and exposes the serialized structure through
    ``to_dict()``. Precision and group size are global only when every declared
    weight scheme agrees; mixed or opaque group layouts deliberately yield
    ``None`` rather than an arbitrary first-group value.
    """
    config = _serialized_config_mapping(value)
    if config is None:
        return None, None

    def resolve(names: tuple[str, ...]) -> int | None:
        direct_values = _declared_values(config, names)
        direct = _uniform_positive_int(direct_values)
        group_values, saw_weight_scheme = _compressed_tensors_group_values(
            config,
            names=names,
        )
        if not saw_weight_scheme:
            return direct
        grouped = _uniform_positive_int(group_values)
        if grouped is None:
            return None
        if direct_values and direct != grouped:
            return None
        return grouped

    return (
        resolve(("bits", "num_bits", "nbits")),
        resolve(("group_size",)),
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
        quant_method = _lower_string(quant_cfg.get("quant_method", ""))
        load_in_8bit = quant_cfg.get("load_in_8bit", False) is True
        load_in_4bit = quant_cfg.get("load_in_4bit", False) is True
        bits = _coerce_positive_int(
            quant_cfg.get("bits", quant_cfg.get("nbits", 16)),
            default=16,
        )
        raw_group_size = quant_cfg.get("group_size")
        group_size = (
            _coerce_positive_int(raw_group_size, default=128)
            if raw_group_size is not None
            else None
        )
        double_quant = quant_cfg.get("bnb_4bit_use_double_quant", False) is True
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
        elif "torchao" in quant_method:
            return QuantizationConfig(
                method=QuantizationMethod.TORCHAO_INT8,
                bits=8,
                from_checkpoint=True,
            )
        elif quant_method == "hqq":
            return QuantizationConfig(
                method=QuantizationMethod.HQQ,
                bits=bits,
                group_size=group_size,
                from_checkpoint=True,
            )
        elif quant_method == "quanto":
            return QuantizationConfig(
                method=QuantizationMethod.QUANTO,
                bits=(
                    bits
                    if bits != 16
                    else _bits_from_weight_spec(quant_cfg.get("weights"))
                ),
                from_checkpoint=True,
            )
        elif _is_compressed_tensors_method(quant_method):
            compressed_bits, compressed_group_size = (
                _compressed_tensors_declared_metadata(quant_cfg)
            )
            return QuantizationConfig(
                method=QuantizationMethod.COMPRESSED_TENSORS,
                bits=compressed_bits,
                group_size=compressed_group_size,
                from_checkpoint=True,
            )
        elif load_in_8bit or (quant_method == "bitsandbytes" and bits == 8):
            return QuantizationConfig(
                method=QuantizationMethod.BNB_8BIT,
                bits=8,
                from_checkpoint=True,
            )
        elif load_in_4bit or (quant_method == "bitsandbytes" and bits == 4):
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
        bits = _coerce_positive_int(getattr(quant_cfg, "bits", 4), default=4)
        group_size = _coerce_positive_int(
            getattr(quant_cfg, "group_size", 128),
            default=128,
        )
        return QuantizationConfig(
            method=QuantizationMethod.AWQ,
            bits=bits,
            group_size=group_size,
            from_checkpoint=True,
        )

    if cfg_class in ("GPTQConfig",):
        bits = _coerce_positive_int(getattr(quant_cfg, "bits", 4), default=4)
        group_size = _coerce_positive_int(
            getattr(quant_cfg, "group_size", 128),
            default=128,
        )
        return QuantizationConfig(
            method=QuantizationMethod.GPTQ,
            bits=bits,
            group_size=group_size,
            from_checkpoint=True,
        )

    if "TorchAO" in cfg_class or "Int8WeightOnly" in cfg_class:
        return QuantizationConfig(
            method=QuantizationMethod.TORCHAO_INT8,
            bits=8,
            from_checkpoint=True,
        )

    if (
        cfg_class in ("HqqConfig", "HQQConfig")
        or "Hqq" in cfg_class
        or "HQQ" in cfg_class
    ):
        bits = _coerce_positive_int(getattr(quant_cfg, "nbits", 4), default=4)
        group_size = getattr(quant_cfg, "group_size", 64)
        return QuantizationConfig(
            method=QuantizationMethod.HQQ,
            bits=bits,
            group_size=(
                _coerce_positive_int(group_size, default=64)
                if group_size is not None
                else None
            ),
            from_checkpoint=True,
        )

    if cfg_class in ("QuantoConfig",) or "Quanto" in cfg_class:
        return QuantizationConfig(
            method=QuantizationMethod.QUANTO,
            bits=_bits_from_weight_spec(getattr(quant_cfg, "weights", "int8")),
            from_checkpoint=True,
        )

    if "CompressedTensors" in cfg_class or "Compressed" in cfg_class:
        compressed_bits, compressed_group_size = _compressed_tensors_declared_metadata(
            quant_cfg
        )
        return QuantizationConfig(
            method=QuantizationMethod.COMPRESSED_TENSORS,
            bits=compressed_bits,
            group_size=compressed_group_size,
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
                    fqcn = f"{module.__class__.__module__}.{module_name}".lower()
                    if "torchao" in fqcn or "affinequantized" in fqcn:
                        quant_config = QuantizationConfig(
                            method=QuantizationMethod.TORCHAO_INT8,
                            bits=8,
                            from_checkpoint=True,
                        )
                        break
                    if "hqq" in fqcn:
                        quant_config = QuantizationConfig(
                            method=QuantizationMethod.HQQ,
                            bits=4,
                            group_size=64,
                            from_checkpoint=True,
                        )
                        break
                    if "optimum.quanto" in fqcn or ".quanto." in fqcn:
                        quant_config = QuantizationConfig(
                            method=QuantizationMethod.QUANTO,
                            bits=8,
                            from_checkpoint=True,
                        )
                        break
                    if (
                        "compressed_tensors" in fqcn
                        or "compressedtensors" in fqcn
                        or "compressedlinear" in fqcn
                    ):
                        quant_config = QuantizationConfig(
                            method=QuantizationMethod.COMPRESSED_TENSORS,
                            bits=None,
                            from_checkpoint=True,
                        )
                        break
            except (TypeError, StopIteration):
                pass

    # Determine if device is movable
    device_movable = not quant_config.is_bnb()

    # For backend-managed quantized models, check if model has been quantized in a way that
    # prevents device movement
    if quant_config.method in (
        QuantizationMethod.AWQ,
        QuantizationMethod.GPTQ,
        QuantizationMethod.TORCHAO_INT8,
        QuantizationMethod.HQQ,
        QuantizationMethod.QUANTO,
        QuantizationMethod.COMPRESSED_TENSORS,
    ):
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
    # Decoder embed_tokens style: lm_head.weight ↔ model.embed_tokens.weight
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

    # Decoder-only models (GPT-like, RoPE-style)
    return "ppl_causal"


__all__ = [
    "QuantizationMethod",
    "QuantizationConfig",
    "ModelCapabilities",
    "detect_quantization_from_config",
    "detect_capabilities_from_model",
]
