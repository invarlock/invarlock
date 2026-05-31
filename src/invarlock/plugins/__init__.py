"""Built-in optional plugin implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from invarlock.adapters.capabilities import (
    ModelCapabilities,
    QuantizationMethod,
    detect_quantization_from_config,
)
from invarlock.adapters.hf_loading import resolve_trust_remote_code
from invarlock.adapters.hf_mixin import HFAdapterMixin
from invarlock.core import INVARLOCK_CORE_ABI as CORE_ABI
from invarlock.core.api import Guard, ModelAdapter
from invarlock.core.error_utils import wrap_errors
from invarlock.core.exceptions import DependencyError, ModelLoadError
from invarlock.core.types import GuardValidationResult

INVARLOCK_CORE_ABI = CORE_ABI

_BNB_CONFIG_ERRORS = (OSError, TypeError, ValueError)
_BNB_MODEL_LOAD_ERRORS = (
    AttributeError,
    ImportError,
    ModelLoadError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _is_local_path(model_id: str) -> bool:
    """Check if model_id is a local filesystem path."""
    return Path(model_id).exists()


def _detect_pre_quantized_bnb(model_id: str) -> tuple[bool, int]:
    """Detect whether a local checkpoint is pre-quantized with BNB."""
    if not _is_local_path(model_id):
        return False, 0

    config_path = Path(model_id) / "config.json"
    if not config_path.exists():
        return False, 0

    try:
        import json

        config_data = json.loads(config_path.read_text())
        quant_cfg = config_data.get("quantization_config", {})

        if not quant_cfg:
            return False, 0

        quant_method = str(quant_cfg.get("quant_method", "")).lower()
        if "bitsandbytes" in quant_method or "bnb" in quant_method:
            bits = quant_cfg.get("bits")
            if isinstance(bits, int) and bits in {4, 8}:
                return True, bits
            if quant_cfg.get("load_in_8bit"):
                return True, 8
            if quant_cfg.get("load_in_4bit"):
                return True, 4
            return True, 8

    except _BNB_CONFIG_ERRORS:
        pass

    return False, 0


class HelloGuard(Guard):
    """Demo guard that checks a score in the validation context."""

    name = "demo_hello_guard"
    support_tier = "demo_only"
    demo_only = True
    strict_assurance_allowed = False

    def __init__(self, threshold: float = 1.0):
        self.threshold = float(threshold)

    def validate(
        self,
        model: Any,
        adapter: ModelAdapter,
        context: dict[str, Any],
    ) -> GuardValidationResult:
        score = float(context.get("hello_score", 0.0))
        passed = score <= self.threshold
        return GuardValidationResult(
            passed=passed,
            decision="allow" if passed else "block",
            metrics={"score": score},
            extras={
                "message": (
                    f"Hello guard score {score:.3f} (threshold {self.threshold:.3f})"
                )
            },
        )


class HF_AWQ_Adapter(HFAdapterMixin, ModelAdapter):
    name = "hf_awq"

    def load_model(self, model_id: str, device: str = "auto", **kwargs: Any):
        load_kwargs = {k: v for k, v in kwargs.items() if k != "device"}
        trust_remote_code = resolve_trust_remote_code(load_kwargs)
        load_kwargs.pop("trust_remote_code", None)

        with wrap_errors(
            DependencyError,
            "E203",
            "DEPENDENCY-MISSING: transformers/gptqmodel",
            lambda e: {"dependency": "transformers/gptqmodel"},
        ):
            import gptqmodel  # noqa: F401
            from transformers import AutoModelForCausalLM

        with wrap_errors(
            ModelLoadError,
            "E201",
            "MODEL-LOAD-FAILED: awq",
            lambda e: {"model_id": model_id},
        ):
            load_kwargs.setdefault("device_map", "auto")
            model = self._load_pretrained_model(
                AutoModelForCausalLM,
                model_id,
                trust_remote_code=trust_remote_code,
                **load_kwargs,
            )

        return self._safe_to_device(
            model, device, capabilities=ModelCapabilities.for_awq()
        )

    def get_capabilities(self, model: Any) -> ModelCapabilities:
        """Return capabilities for an AWQ-quantized model."""
        config = getattr(model, "config", None)
        group_size = 128
        if config is not None:
            quant_cfg = getattr(config, "quantization_config", None)
            if isinstance(quant_cfg, dict):
                group_size = quant_cfg.get("group_size", 128)
            elif quant_cfg is not None:
                group_size = getattr(quant_cfg, "group_size", 128)
        return ModelCapabilities.for_awq(group_size=group_size)

    def can_handle(self, model: Any) -> bool:
        cfg = getattr(model, "config", None)
        return hasattr(cfg, "n_layer") or hasattr(cfg, "num_hidden_layers")

    def describe(self, model: Any) -> dict[str, Any]:
        cfg = getattr(model, "config", None)
        n_layer = int(
            getattr(cfg, "n_layer", getattr(cfg, "num_hidden_layers", 0)) or 0
        )
        n_head = int(
            getattr(cfg, "n_head", getattr(cfg, "num_attention_heads", 0)) or 0
        )
        heads = [n_head] * n_layer if n_layer and n_head else []
        return {
            "n_layer": n_layer,
            "heads_per_layer": heads,
            "mlp_dims": [],
            "tying": {},
        }


class HF_GPTQ_Adapter(HFAdapterMixin, ModelAdapter):
    name = "hf_gptq"

    def load_model(self, model_id: str, device: str = "auto", **kwargs: Any):
        load_kwargs = {k: v for k, v in kwargs.items() if k != "device"}
        trust_remote_code = resolve_trust_remote_code(load_kwargs)
        load_kwargs.pop("trust_remote_code", None)

        with wrap_errors(
            DependencyError,
            "E203",
            "DEPENDENCY-MISSING: gptqmodel/transformers",
            lambda e: {"dependency": "gptqmodel"},
        ):
            from gptqmodel import GPTQModel

        with wrap_errors(
            ModelLoadError,
            "E201",
            "MODEL-LOAD-FAILED: gptq",
            lambda e: {"model_id": model_id},
        ):
            model = GPTQModel.load(
                model_id,
                trust_remote_code=trust_remote_code,
                **load_kwargs,
            )

        return self._safe_to_device(
            model, device, capabilities=ModelCapabilities.for_gptq()
        )

    def get_capabilities(self, model: Any) -> ModelCapabilities:
        """Return capabilities for a GPTQ-quantized model."""
        config = getattr(model, "config", None)
        bits = 4
        group_size = 128
        if config is not None:
            quant_cfg = getattr(config, "quantization_config", None)
            if isinstance(quant_cfg, dict):
                bits = quant_cfg.get("bits", 4)
                group_size = quant_cfg.get("group_size", 128)
            elif quant_cfg is not None:
                bits = getattr(quant_cfg, "bits", 4)
                group_size = getattr(quant_cfg, "group_size", 128)
        return ModelCapabilities.for_gptq(bits=bits, group_size=group_size)

    def can_handle(self, model: Any) -> bool:
        cfg = getattr(model, "config", None)
        return hasattr(cfg, "n_layer") or hasattr(cfg, "num_hidden_layers")

    def describe(self, model: Any) -> dict[str, Any]:
        cfg = getattr(model, "config", None)
        n_layer = int(
            getattr(cfg, "n_layer", getattr(cfg, "num_hidden_layers", 0)) or 0
        )
        n_head = int(
            getattr(cfg, "n_head", getattr(cfg, "num_attention_heads", 0)) or 0
        )
        heads = [n_head] * n_layer if n_layer and n_head else []
        return {
            "n_layer": n_layer,
            "heads_per_layer": heads,
            "mlp_dims": [],
            "tying": {},
        }


class HF_BNB_Adapter(HFAdapterMixin, ModelAdapter):
    name = "hf_bnb"

    def _raise_load_error(
        self,
        exc: Exception,
        *,
        model_id: str,
        pre_quantized_bits: int | None = None,
    ) -> None:
        details: dict[str, object] = {"model_id": model_id}
        if pre_quantized_bits:
            details["pre_quantized_bits"] = pre_quantized_bits

        text = str(exc)
        if "FineGrainedFP8Config" in text and "BitsAndBytesConfig" in text:
            details["checkpoint_quantization"] = "FineGrainedFP8Config"
            details["requested_quantization"] = "BitsAndBytesConfig"
            details["recommended_adapter"] = "hf_causal"
            raise ModelLoadError(
                code="E201",
                message=(
                    "MODEL-LOAD-FAILED: bitsandbytes incompatible with checkpoint "
                    "quantization_config"
                ),
                details=details,
            ) from exc

        load_label = "bitsandbytes/transformers"
        if pre_quantized_bits:
            load_label = "bitsandbytes/transformers (pre-quantized)"
        raise ModelLoadError(
            code="E201",
            message=f"MODEL-LOAD-FAILED: {load_label}",
            details=details,
        ) from exc

    def load_model(self, model_id: str, device: str = "auto", **kwargs: Any):
        load_kwargs = dict(kwargs)
        trust_remote_code = resolve_trust_remote_code(load_kwargs)
        load_kwargs.pop("trust_remote_code", None)

        with wrap_errors(
            DependencyError,
            "E203",
            "DEPENDENCY-MISSING: transformers",
            lambda e: {"dependency": "transformers"},
        ):
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        is_pre_quantized, pre_quant_bits = _detect_pre_quantized_bnb(model_id)

        if "load_in_8bit" in kwargs or "load_in_4bit" in kwargs:
            raise ValueError(
                "hf_bnb adapter: load_in_8bit/load_in_4bit are not supported. "
                "Use model.quantization_config instead."
            )

        if is_pre_quantized:
            try:
                model = self._load_pretrained_model(
                    AutoModelForCausalLM,
                    model_id,
                    device_map="auto",
                    trust_remote_code=trust_remote_code,
                    **load_kwargs,
                )
            except _BNB_MODEL_LOAD_ERRORS as exc:
                self._raise_load_error(
                    exc,
                    model_id=model_id,
                    pre_quantized_bits=pre_quant_bits,
                )
        else:
            quantization_config = load_kwargs.pop("quantization_config", None)
            if quantization_config is None:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif isinstance(quantization_config, dict):
                qdict = dict(quantization_config)
                bits = qdict.pop("bits", None)
                qdict.pop("quant_method", None)
                if isinstance(bits, int):
                    if bits == 4:
                        qdict.setdefault("load_in_4bit", True)
                        qdict.setdefault("load_in_8bit", False)
                    elif bits == 8:
                        qdict.setdefault("load_in_8bit", True)
                        qdict.setdefault("load_in_4bit", False)
                quantization_config = BitsAndBytesConfig(**qdict)

            try:
                model = self._load_pretrained_model(
                    AutoModelForCausalLM,
                    model_id,
                    device_map="auto",
                    trust_remote_code=trust_remote_code,
                    quantization_config=quantization_config,
                    **load_kwargs,
                )
            except _BNB_MODEL_LOAD_ERRORS as exc:
                self._raise_load_error(exc, model_id=model_id)

        _ = self._resolve_device(device)
        return model

    def get_capabilities(self, model: Any) -> ModelCapabilities:
        """Return capabilities for a BNB-quantized model."""
        config = getattr(model, "config", None)
        if config is not None:
            quant_cfg = detect_quantization_from_config(config)
            if quant_cfg.method == QuantizationMethod.BNB_8BIT:
                return ModelCapabilities.for_bnb_8bit(from_checkpoint=True)
            elif quant_cfg.method == QuantizationMethod.BNB_4BIT:
                return ModelCapabilities.for_bnb_4bit(
                    from_checkpoint=True,
                    double_quant=quant_cfg.double_quant,
                )

        return ModelCapabilities.for_bnb_8bit()

    def can_handle(self, model: Any) -> bool:
        cfg = getattr(model, "config", None)
        return hasattr(cfg, "n_layer") or hasattr(cfg, "num_hidden_layers")

    def describe(self, model: Any) -> dict[str, Any]:
        cfg = getattr(model, "config", None)
        n_layer = int(
            getattr(cfg, "n_layer", getattr(cfg, "num_hidden_layers", 0)) or 0
        )
        n_head = int(
            getattr(cfg, "n_head", getattr(cfg, "num_attention_heads", 0)) or 0
        )
        heads = [n_head] * n_layer if n_layer and n_head else []
        return {
            "n_layer": n_layer,
            "heads_per_layer": heads,
            "mlp_dims": [],
            "tying": {},
        }


__all__ = [
    "HF_AWQ_Adapter",
    "HF_BNB_Adapter",
    "HF_GPTQ_Adapter",
    "HelloGuard",
    "INVARLOCK_CORE_ABI",
]
