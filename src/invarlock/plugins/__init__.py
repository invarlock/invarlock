"""Built-in optional plugin implementations."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

from invarlock.adapters.capabilities import (
    ModelCapabilities,
    QuantizationMethod,
    detect_capabilities_from_model,
    detect_quantization_from_config,
)
from invarlock.adapters.gptq_checkpoint_validation import (
    validate_gptq_checkpoint_bindings,
)
from invarlock.adapters.hf_causal import HF_Causal_Adapter
from invarlock.adapters.hf_loading import resolve_trust_remote_code
from invarlock.adapters.hf_mixin import HFAdapterMixin
from invarlock.core import INVARLOCK_CORE_ABI as CORE_ABI
from invarlock.core.api import Guard, ModelAdapter
from invarlock.core.error_utils import wrap_errors
from invarlock.core.exceptions import AdapterError, DependencyError, ModelLoadError
from invarlock.core.types import GuardValidationResult
from invarlock.gptqmodel_runtime import import_gptqmodel

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
_HF_CAUSAL_INTROSPECTION = HF_Causal_Adapter()


def _gptqmodel_jit_toolchain_required(device: str) -> bool:
    """Require CUDA JIT prerequisites only when a load can use CUDA."""

    normalized_device = device.strip().casefold()
    if normalized_device.startswith("cuda"):
        return True
    if normalized_device != "auto":
        return False
    try:
        torch = importlib.import_module("torch")
        cuda = getattr(torch, "cuda", None)
        cuda_is_available = getattr(cuda, "is_available", None)
        return bool(cuda_is_available()) if callable(cuda_is_available) else False
    except (AttributeError, ImportError, OSError, RuntimeError):
        return False


def _fallback_causal_description(model: Any) -> dict[str, Any]:
    cfg = getattr(model, "config", None)
    n_layer = int(getattr(cfg, "n_layer", getattr(cfg, "num_hidden_layers", 0)) or 0)
    n_head = int(getattr(cfg, "n_head", getattr(cfg, "num_attention_heads", 0)) or 0)
    heads = [n_head] * n_layer if n_layer and n_head else []
    return {
        "n_layer": n_layer,
        "heads_per_layer": heads,
        "mlp_dims": [],
        "tying": {},
    }


class _QuantizedCausalIntrospectionMixin:
    def describe(self, model: Any) -> dict[str, Any]:
        try:
            return _HF_CAUSAL_INTROSPECTION.describe(model)
        except AdapterError:
            return _fallback_causal_description(model)

    def get_layer_modules(self, model: Any, layer_idx: int) -> dict[str, Any]:
        return _HF_CAUSAL_INTROSPECTION.get_layer_modules(model, layer_idx)


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


class HF_AWQ_Adapter(_QuantizedCausalIntrospectionMixin, HFAdapterMixin, ModelAdapter):
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
            import_gptqmodel(
                require_jit_toolchain=_gptqmodel_jit_toolchain_required(device)
            )
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


class HF_GPTQ_Adapter(_QuantizedCausalIntrospectionMixin, HFAdapterMixin, ModelAdapter):
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
            gptqmodel = import_gptqmodel(
                require_jit_toolchain=_gptqmodel_jit_toolchain_required(device)
            )
            # GPTQModel may expose its public class lazily; preserve the
            # normal AttributeError path for a broken optional installation.
            GPTQModel = getattr(gptqmodel, "GPTQModel")  # noqa: B009

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
            validate_gptq_checkpoint_bindings(model)

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


class HF_BNB_Adapter(_QuantizedCausalIntrospectionMixin, HFAdapterMixin, ModelAdapter):
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
        device_map = load_kwargs.pop("device_map", "auto")

        if is_pre_quantized:
            try:
                model = self._load_pretrained_model(
                    AutoModelForCausalLM,
                    model_id,
                    device_map=device_map,
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
                    device_map=device_map,
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


class HF_TorchAO_Adapter(
    _QuantizedCausalIntrospectionMixin, HFAdapterMixin, ModelAdapter
):
    name = "hf_torchao"

    @staticmethod
    def _int8_weight_only_config(config_cls: Any) -> Any:
        try:
            return config_cls(version=2)
        except TypeError:
            return config_cls()

    def load_model(self, model_id: str, device: str = "auto", **kwargs: Any):
        load_kwargs = {k: v for k, v in kwargs.items() if k != "device"}
        trust_remote_code = resolve_trust_remote_code(load_kwargs)
        load_kwargs.pop("trust_remote_code", None)

        with wrap_errors(
            DependencyError,
            "E203",
            "DEPENDENCY-MISSING: transformers/torchao",
            lambda e: {"dependency": "transformers/torchao"},
        ):
            torchao_quantization = importlib.import_module("torchao.quantization")
            torchao_symbols = vars(torchao_quantization)
            Int8WeightOnlyConfig = torchao_symbols["Int8WeightOnlyConfig"]
            quantize_ = torchao_symbols["quantize_"]
            from transformers import AutoModelForCausalLM

        with wrap_errors(
            ModelLoadError,
            "E201",
            "MODEL-LOAD-FAILED: torchao-int8",
            lambda e: {"model_id": model_id},
        ):
            model = self._load_pretrained_model(
                AutoModelForCausalLM,
                model_id,
                trust_remote_code=trust_remote_code,
                **load_kwargs,
            )
            quantize_(model, self._int8_weight_only_config(Int8WeightOnlyConfig))

        return self._safe_to_device(
            model,
            device,
            capabilities=ModelCapabilities.for_torchao_int8(),
        )

    def get_capabilities(self, model: Any) -> ModelCapabilities:
        """Return capabilities for a torchao int8 runtime-quantized model."""
        _ = model
        return ModelCapabilities.for_torchao_int8()

    def can_handle(self, model: Any) -> bool:
        cfg = getattr(model, "config", None)
        return hasattr(cfg, "n_layer") or hasattr(cfg, "num_hidden_layers")


class HF_HQQ_Adapter(_QuantizedCausalIntrospectionMixin, HFAdapterMixin, ModelAdapter):
    name = "hf_hqq"

    def _hqq_quantization_params(self, quantization_config: Any) -> dict[str, Any]:
        if quantization_config is None:
            return {"nbits": 4, "group_size": 64, "axis": 1}
        if isinstance(quantization_config, dict):
            values = dict(quantization_config)
        elif hasattr(quantization_config, "to_dict"):
            values = dict(quantization_config.to_dict())
        else:
            values = {
                "nbits": getattr(
                    quantization_config,
                    "nbits",
                    getattr(quantization_config, "bits", 4),
                ),
                "group_size": getattr(quantization_config, "group_size", 64),
                "axis": getattr(quantization_config, "axis", 1),
            }
        values.pop("quant_method", None)

        nbits = int(values.get("nbits", values.get("bits", 4)) or 4)
        raw_group_size = values.get("group_size", 64)
        group_size = int(raw_group_size) if raw_group_size is not None else None
        axis = int(values.get("axis", 1) or 1)
        return {"nbits": nbits, "group_size": group_size, "axis": axis}

    def _hqq_device(self, device: str) -> str:
        if device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except (ImportError, ModuleNotFoundError):
                return "cpu"
        return device

    def load_model(self, model_id: str, device: str = "auto", **kwargs: Any):
        load_kwargs = {k: v for k, v in kwargs.items() if k != "device"}
        trust_remote_code = resolve_trust_remote_code(load_kwargs)
        load_kwargs.pop("trust_remote_code", None)

        with wrap_errors(
            DependencyError,
            "E203",
            "DEPENDENCY-MISSING: transformers/hqq",
            lambda e: {"dependency": "transformers/hqq"},
        ):
            import torch
            from transformers import AutoModelForCausalLM

            importlib.import_module("hqq")
            quantize_module = importlib.import_module("hqq.core.quantize")
            hqq_model_module = importlib.import_module(
                "hqq.models.hf.base",
            )
            hqq_base_quant_config = cast(Any, quantize_module).hqq_base_quant_config
            AutoHQQHFModel = cast(Any, hqq_model_module).AutoHQQHFModel

        quantization_config = load_kwargs.pop("quantization_config", None)
        quantization_params = self._hqq_quantization_params(quantization_config)
        hqq_quant_config = hqq_base_quant_config(**quantization_params)
        hqq_device = self._hqq_device(str(device).lower())
        compute_dtype = (
            torch.float16 if hqq_device.startswith("cuda") else torch.float32
        )

        with wrap_errors(
            ModelLoadError,
            "E201",
            "MODEL-LOAD-FAILED: hqq",
            lambda e: {"model_id": model_id},
        ):
            model = self._load_pretrained_model(
                AutoModelForCausalLM,
                model_id,
                trust_remote_code=trust_remote_code,
                **load_kwargs,
            )
            AutoHQQHFModel.quantize_model(
                model,
                quant_config=hqq_quant_config,
                compute_dtype=compute_dtype,
                device=hqq_device,
            )

        return self._safe_to_device(
            model,
            device,
            capabilities=self.get_capabilities_from_quantization_config(
                quantization_params
            ),
        )

    def get_capabilities_from_quantization_config(
        self,
        quantization_config: Any,
    ) -> ModelCapabilities:
        if isinstance(quantization_config, dict):
            raw_bits = quantization_config.get("nbits", 4)
            raw_group_size = quantization_config.get("group_size", 64)
        else:
            raw_bits = getattr(quantization_config, "nbits", 4)
            raw_group_size = getattr(quantization_config, "group_size", 64)
        bits = int(raw_bits or 4)
        group_size = int(raw_group_size) if raw_group_size is not None else None
        return ModelCapabilities.for_hqq(bits=bits, group_size=group_size)

    def get_capabilities(self, model: Any) -> ModelCapabilities:
        """Return capabilities for a HQQ runtime-quantized model."""
        config = getattr(model, "config", None)
        quant_cfg = getattr(config, "quantization_config", None)
        if quant_cfg is not None:
            return self.get_capabilities_from_quantization_config(quant_cfg)
        return ModelCapabilities.for_hqq()

    def can_handle(self, model: Any) -> bool:
        cfg = getattr(model, "config", None)
        return hasattr(cfg, "n_layer") or hasattr(cfg, "num_hidden_layers")


class HF_Quanto_Adapter(
    _QuantizedCausalIntrospectionMixin, HFAdapterMixin, ModelAdapter
):
    name = "hf_quanto"

    def _quanto_config_kwargs(self, quantization_config: Any) -> dict[str, Any]:
        if quantization_config is None:
            return {"weights": "int8"}
        if isinstance(quantization_config, dict):
            values = dict(quantization_config)
        elif hasattr(quantization_config, "to_dict"):
            values = dict(quantization_config.to_dict())
        else:
            values = {
                "weights": getattr(quantization_config, "weights", "int8"),
                "activations": getattr(quantization_config, "activations", None),
            }
        values.pop("quant_method", None)
        values.pop("quant_method_full", None)
        return {key: value for key, value in values.items() if value is not None}

    def load_model(self, model_id: str, device: str = "auto", **kwargs: Any):
        load_kwargs = {k: v for k, v in kwargs.items() if k != "device"}
        trust_remote_code = resolve_trust_remote_code(load_kwargs)
        load_kwargs.pop("trust_remote_code", None)

        with wrap_errors(
            DependencyError,
            "E203",
            "DEPENDENCY-MISSING: transformers/optimum-quanto",
            lambda e: {"dependency": "transformers/optimum-quanto"},
        ):
            importlib.import_module("optimum.quanto")
            from transformers import AutoModelForCausalLM, QuantoConfig

        quantization_config = load_kwargs.pop("quantization_config", None)
        if not isinstance(quantization_config, QuantoConfig):
            quantization_config = QuantoConfig(
                **self._quanto_config_kwargs(quantization_config)
            )

        with wrap_errors(
            ModelLoadError,
            "E201",
            "MODEL-LOAD-FAILED: quanto",
            lambda e: {"model_id": model_id},
        ):
            load_kwargs.setdefault("device_map", "auto")
            model = self._load_pretrained_model(
                AutoModelForCausalLM,
                model_id,
                trust_remote_code=trust_remote_code,
                quantization_config=quantization_config,
                **load_kwargs,
            )

        return self._safe_to_device(
            model,
            device,
            capabilities=self.get_capabilities_from_quantization_config(
                quantization_config
            ),
        )

    def get_capabilities_from_quantization_config(
        self,
        quantization_config: Any,
    ) -> ModelCapabilities:
        weights = ""
        if isinstance(quantization_config, dict):
            weights = str(quantization_config.get("weights", "int8")).lower()
        else:
            weights = str(getattr(quantization_config, "weights", "int8")).lower()
        bits = 4 if "int4" in weights else 8
        return ModelCapabilities.for_quanto(bits=bits)

    def get_capabilities(self, model: Any) -> ModelCapabilities:
        """Return capabilities for a Quanto runtime-quantized model."""
        config = getattr(model, "config", None)
        quant_cfg = getattr(config, "quantization_config", None)
        if quant_cfg is not None:
            return self.get_capabilities_from_quantization_config(quant_cfg)
        return ModelCapabilities.for_quanto()

    def can_handle(self, model: Any) -> bool:
        cfg = getattr(model, "config", None)
        return hasattr(cfg, "n_layer") or hasattr(cfg, "num_hidden_layers")


class HF_CompressedTensors_Adapter(
    _QuantizedCausalIntrospectionMixin, HFAdapterMixin, ModelAdapter
):
    name = "hf_ct"

    def load_model(self, model_id: str, device: str = "auto", **kwargs: Any):
        load_kwargs = {k: v for k, v in kwargs.items() if k != "device"}
        trust_remote_code = resolve_trust_remote_code(load_kwargs)
        load_kwargs.pop("trust_remote_code", None)

        with wrap_errors(
            DependencyError,
            "E203",
            "DEPENDENCY-MISSING: transformers/compressed-tensors",
            lambda e: {"dependency": "transformers/compressed-tensors"},
        ):
            importlib.import_module("compressed_tensors")
            from transformers import AutoModelForCausalLM

        with wrap_errors(
            ModelLoadError,
            "E201",
            "MODEL-LOAD-FAILED: compressed-tensors",
            lambda e: {"model_id": model_id},
        ):
            load_kwargs.setdefault("device_map", "auto")
            model = self._load_pretrained_model(
                AutoModelForCausalLM,
                model_id,
                trust_remote_code=trust_remote_code,
                **load_kwargs,
            )

        capabilities = self.get_capabilities(model)
        if capabilities.quantization.method != QuantizationMethod.COMPRESSED_TENSORS:
            raise ModelLoadError(
                code="E201",
                message="MODEL-LOAD-FAILED: compressed-tensors checkpoint metadata missing",
                details={"model_id": model_id},
            )

        return self._safe_to_device(
            model,
            device,
            capabilities=capabilities,
        )

    def get_capabilities_from_quantization_config(
        self,
        quantization_config: Any,
    ) -> ModelCapabilities:
        class _ConfigWrapper:
            pass

        config: Any = _ConfigWrapper()
        config.quantization_config = quantization_config
        quantization = detect_quantization_from_config(config)
        if quantization.method == QuantizationMethod.COMPRESSED_TENSORS:
            return ModelCapabilities(
                quantization=quantization,
                device_movable=False,
            )
        return ModelCapabilities()

    def get_capabilities(self, model: Any) -> ModelCapabilities:
        """Return capabilities for a compressed-tensors checkpoint model."""
        return detect_capabilities_from_model(model)

    def can_handle(self, model: Any) -> bool:
        cfg = getattr(model, "config", None)
        return hasattr(cfg, "n_layer") or hasattr(cfg, "num_hidden_layers")


__all__ = [
    "HF_CompressedTensors_Adapter",
    "HF_AWQ_Adapter",
    "HF_BNB_Adapter",
    "HF_GPTQ_Adapter",
    "HF_HQQ_Adapter",
    "HF_Quanto_Adapter",
    "HF_TorchAO_Adapter",
    "HelloGuard",
    "INVARLOCK_CORE_ABI",
]
