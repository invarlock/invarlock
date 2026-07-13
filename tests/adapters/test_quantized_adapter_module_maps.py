from __future__ import annotations

from copy import deepcopy
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

from invarlock.adapters.capabilities import QuantizationMethod
from invarlock.core.exceptions import ModelLoadError
from invarlock.plugins import (
    HF_AWQ_Adapter,
    HF_BNB_Adapter,
    HF_CompressedTensors_Adapter,
    HF_GPTQ_Adapter,
    HF_HQQ_Adapter,
    HF_Quanto_Adapter,
    HF_TorchAO_Adapter,
)


class _DenseSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class _DenseMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)


class _DenseLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _DenseSelfAttention()
        self.mlp = _DenseMlp()
        self.input_layernorm = nn.LayerNorm(4)
        self.post_attention_layernorm = nn.LayerNorm(4)


class _TinyCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="llama",
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=4,
            intermediate_size=8,
            vocab_size=32,
        )
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_DenseLayer()])


class _NestedQuantizedWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="llama",
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=4,
            intermediate_size=8,
            vocab_size=32,
        )
        self.model = nn.Module()
        self.model.model = nn.Module()
        self.model.model.layers = nn.ModuleList([_DenseLayer()])


_OBSERVED_PACKED_CT_CONFIG = {
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "group_size": 128,
                "strategy": "group",
                "dynamic": False,
            },
            "input_activations": None,
            "output_activations": None,
            "format": None,
        }
    },
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "quantization_status": "compressed",
    "ignore": ["lm_head"],
}


class _ObservedRuntimeCompressedTensorsConfig:
    """Matches the relevant post-load Transformers config shape.

    The real Transformers ``CompressedTensorsConfig`` stores its parsed scheme
    under ``quantization_config`` and exposes packed metadata through
    ``to_dict()``; it does not expose ``config_groups`` as a direct attribute.
    """

    def __init__(self, payload: dict[str, object]) -> None:
        self.quantization_config = SimpleNamespace(parsed_scheme=True)
        self.quant_method = SimpleNamespace(value="compressed-tensors")
        self.run_compressed = True
        self._payload = deepcopy(payload)

    def to_dict(self) -> dict[str, object]:
        return deepcopy(self._payload)


@pytest.mark.parametrize(
    "adapter_cls",
    [
        HF_AWQ_Adapter,
        HF_GPTQ_Adapter,
        HF_BNB_Adapter,
        HF_TorchAO_Adapter,
        HF_HQQ_Adapter,
        HF_Quanto_Adapter,
        HF_CompressedTensors_Adapter,
    ],
)
def test_quantized_hf_adapters_reuse_causal_layer_module_contract(adapter_cls) -> None:
    model = _TinyCausalModel()
    layer = model.model.layers[0]

    modules = adapter_cls().get_layer_modules(model, 0)

    assert modules["self_attn.o_proj"] is layer.self_attn.o_proj
    assert modules["mlp.down_proj"] is layer.mlp.down_proj


def test_gptq_adapter_handles_nested_model_model_layers_wrapper() -> None:
    model = _NestedQuantizedWrapper()
    layer = model.model.model.layers[0]

    modules = HF_GPTQ_Adapter().get_layer_modules(model, 0)

    assert modules["self_attn.o_proj"] is layer.self_attn.o_proj
    assert modules["mlp.down_proj"] is layer.mlp.down_proj


@pytest.mark.parametrize(
    "adapter_cls",
    [
        HF_AWQ_Adapter,
        HF_GPTQ_Adapter,
        HF_BNB_Adapter,
        HF_TorchAO_Adapter,
        HF_HQQ_Adapter,
        HF_Quanto_Adapter,
        HF_CompressedTensors_Adapter,
    ],
)
def test_quantized_hf_adapters_describe_structural_spec(adapter_cls) -> None:
    description = adapter_cls().describe(_TinyCausalModel())

    assert description["n_layer"] == 1
    assert description["spec"] == "dense_decoder"
    assert description["mlp_dims"] == [8]


def test_torchao_int8_adapter_loads_and_quantizes(monkeypatch) -> None:
    from invarlock.plugins import HF_TorchAO_Adapter

    model = _TinyCausalModel()
    calls: dict[str, object] = {}

    class _AutoModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            calls["model_id"] = model_id
            calls["load_kwargs"] = kwargs
            return model, {}

    class _Int8WeightOnlyConfig:
        def __init__(self, *, version=None):
            self.version = version

    def _quantize_(loaded_model, config):
        calls["quantized_model"] = loaded_model
        calls["quant_config"] = config
        loaded_model.quantized_by_torchao = True
        return loaded_model

    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = _AutoModel
    torchao = ModuleType("torchao")
    quantization = ModuleType("torchao.quantization")
    quantization.Int8WeightOnlyConfig = _Int8WeightOnlyConfig
    quantization.quantize_ = _quantize_
    monkeypatch.setitem(__import__("sys").modules, "transformers", transformers)
    monkeypatch.setitem(__import__("sys").modules, "torchao", torchao)
    monkeypatch.setitem(__import__("sys").modules, "torchao.quantization", quantization)

    adapter = HF_TorchAO_Adapter()
    loaded = adapter.load_model("local-tiny", device="cpu")

    assert loaded is model
    assert calls["model_id"] == "local-tiny"
    assert calls["quantized_model"] is model
    assert calls["quant_config"].version == 2
    assert loaded.quantized_by_torchao is True
    capabilities = adapter.get_capabilities(loaded)
    assert capabilities.device_movable is False
    assert capabilities.quantization.method == QuantizationMethod.TORCHAO_INT8


def test_hqq_adapter_loads_and_quantizes_with_native_hqq_runtime(monkeypatch) -> None:
    from invarlock.plugins import HF_HQQ_Adapter

    model = _TinyCausalModel()
    calls: dict[str, object] = {}

    class _AutoModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            calls["model_id"] = model_id
            calls["load_kwargs"] = kwargs
            return model, {}

    def _hqq_base_quant_config(**kwargs):
        calls["quantization_params"] = kwargs
        return {"weight_quant_params": kwargs}

    class _AutoHQQHFModel:
        @staticmethod
        def quantize_model(loaded_model, quant_config, compute_dtype, device):
            calls["quantized_model"] = loaded_model
            calls["quant_config"] = quant_config
            calls["compute_dtype"] = compute_dtype
            calls["device"] = device
            loaded_model.hqq_quantized = True
            return loaded_model

    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = _AutoModel
    hqq = ModuleType("hqq")
    hqq_core = ModuleType("hqq.core")
    hqq_quantize = ModuleType("hqq.core.quantize")
    hqq_models = ModuleType("hqq.models")
    hqq_models_hf = ModuleType("hqq.models.hf")
    hqq_models_hf_base = ModuleType("hqq.models.hf.base")
    hqq.__path__ = []
    hqq_core.__path__ = []
    hqq_models.__path__ = []
    hqq_models_hf.__path__ = []
    hqq_quantize.hqq_base_quant_config = _hqq_base_quant_config
    hqq_models_hf_base.AutoHQQHFModel = _AutoHQQHFModel
    monkeypatch.setitem(__import__("sys").modules, "transformers", transformers)
    monkeypatch.setitem(__import__("sys").modules, "hqq", hqq)
    monkeypatch.setitem(__import__("sys").modules, "hqq.core", hqq_core)
    monkeypatch.setitem(__import__("sys").modules, "hqq.core.quantize", hqq_quantize)
    monkeypatch.setitem(__import__("sys").modules, "hqq.models", hqq_models)
    monkeypatch.setitem(__import__("sys").modules, "hqq.models.hf", hqq_models_hf)
    monkeypatch.setitem(
        __import__("sys").modules,
        "hqq.models.hf.base",
        hqq_models_hf_base,
    )

    adapter = HF_HQQ_Adapter()
    loaded = adapter.load_model(
        "local-tiny",
        device="cpu",
        quantization_config={"nbits": 4, "group_size": 64},
    )

    assert loaded is model
    assert calls["model_id"] == "local-tiny"
    load_kwargs = calls["load_kwargs"]
    assert isinstance(load_kwargs, dict)
    assert "quantization_config" not in load_kwargs
    assert calls["quantization_params"] == {"nbits": 4, "group_size": 64, "axis": 1}
    assert calls["quantized_model"] is model
    assert calls["quant_config"] == {
        "weight_quant_params": {"nbits": 4, "group_size": 64, "axis": 1},
    }
    assert calls["compute_dtype"] is torch.float32
    assert calls["device"] == "cpu"
    assert loaded.hqq_quantized is True
    capabilities = adapter.get_capabilities(loaded)
    assert capabilities.device_movable is False
    assert capabilities.quantization.method == QuantizationMethod.HQQ


def test_quanto_adapter_loads_with_transformers_quanto_config(monkeypatch) -> None:
    from invarlock.plugins import HF_Quanto_Adapter

    model = _TinyCausalModel()
    calls: dict[str, object] = {}

    class _QuantoConfig:
        def __init__(self, **kwargs):
            calls["quanto_config_kwargs"] = kwargs
            self.weights = kwargs.get("weights")

    class _AutoModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            calls["model_id"] = model_id
            calls["load_kwargs"] = kwargs
            return model

    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = _AutoModel
    transformers.QuantoConfig = _QuantoConfig
    optimum = ModuleType("optimum")
    optimum_quanto = ModuleType("optimum.quanto")
    optimum.__path__ = []
    monkeypatch.setitem(__import__("sys").modules, "transformers", transformers)
    monkeypatch.setitem(__import__("sys").modules, "optimum", optimum)
    monkeypatch.setitem(__import__("sys").modules, "optimum.quanto", optimum_quanto)

    adapter = HF_Quanto_Adapter()
    loaded = adapter.load_model(
        "local-tiny",
        device="cuda",
        quantization_config={"weights": "int8"},
        revision="main",
    )

    assert loaded is model
    assert calls["model_id"] == "local-tiny"
    assert calls["quanto_config_kwargs"] == {"weights": "int8"}
    load_kwargs = calls["load_kwargs"]
    assert isinstance(load_kwargs, dict)
    assert load_kwargs["device_map"] == "auto"
    assert load_kwargs["revision"] == "main"
    assert isinstance(load_kwargs["quantization_config"], _QuantoConfig)
    capabilities = adapter.get_capabilities(loaded)
    assert capabilities.device_movable is False
    assert capabilities.quantization.method == QuantizationMethod.QUANTO


def test_compressed_tensors_adapter_loads_prequantized_checkpoint(monkeypatch) -> None:
    from invarlock.plugins import HF_CompressedTensors_Adapter

    model = _TinyCausalModel()
    model.config.quantization_config = {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 4,
                }
            }
        },
    }
    calls: dict[str, object] = {}

    class _AutoModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            calls["model_id"] = model_id
            calls["load_kwargs"] = kwargs
            return model

    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = _AutoModel
    compressed_tensors = ModuleType("compressed_tensors")
    monkeypatch.setitem(__import__("sys").modules, "transformers", transformers)
    monkeypatch.setitem(
        __import__("sys").modules,
        "compressed_tensors",
        compressed_tensors,
    )

    adapter = HF_CompressedTensors_Adapter()
    loaded = adapter.load_model("local-tiny", device="cuda", revision="main")

    assert loaded is model
    assert calls["model_id"] == "local-tiny"
    load_kwargs = calls["load_kwargs"]
    assert isinstance(load_kwargs, dict)
    assert load_kwargs["device_map"] == "auto"
    assert load_kwargs["revision"] == "main"
    assert "quantization_config" not in load_kwargs
    capabilities = adapter.get_capabilities(loaded)
    assert capabilities.device_movable is False
    assert capabilities.quantization.method == QuantizationMethod.COMPRESSED_TENSORS
    assert capabilities.quantization.bits == 4


def test_compressed_tensors_adapter_preserves_observed_runtime_config_metadata() -> (
    None
):
    """Nested runtime configs retain the packed checkpoint's declared metadata."""
    model = _TinyCausalModel()
    runtime_config = _ObservedRuntimeCompressedTensorsConfig(_OBSERVED_PACKED_CT_CONFIG)
    assert not hasattr(runtime_config, "config_groups")
    model.config.quantization_config = runtime_config

    adapter = HF_CompressedTensors_Adapter()
    from_model = adapter.get_capabilities(model)
    from_config = adapter.get_capabilities_from_quantization_config(runtime_config)

    for capabilities in (from_model, from_config):
        assert capabilities.device_movable is False
        assert capabilities.quantization.method == QuantizationMethod.COMPRESSED_TENSORS
        assert capabilities.quantization.from_checkpoint is True
        assert capabilities.quantization.bits == 4
        assert capabilities.quantization.group_size == 128


def test_compressed_tensors_adapter_fails_closed_for_mixed_packed_weight_groups() -> (
    None
):
    """No single global value is claimed when packed group declarations conflict."""
    payload = deepcopy(_OBSERVED_PACKED_CT_CONFIG)
    config_groups = payload["config_groups"]
    assert isinstance(config_groups, dict)
    config_groups["group_1"] = {
        "targets": ["Linear"],
        "weights": {
            "num_bits": 8,
            "type": "int",
            "symmetric": True,
            "group_size": 64,
            "strategy": "group",
            "dynamic": False,
        },
    }
    model = _TinyCausalModel()
    model.config.quantization_config = _ObservedRuntimeCompressedTensorsConfig(payload)

    capabilities = HF_CompressedTensors_Adapter().get_capabilities(model)

    assert capabilities.device_movable is False
    assert capabilities.quantization.method == QuantizationMethod.COMPRESSED_TENSORS
    assert capabilities.quantization.from_checkpoint is True
    assert capabilities.quantization.bits is None
    assert capabilities.quantization.group_size is None


def test_compressed_tensors_adapter_rejects_dense_checkpoint(monkeypatch) -> None:
    from invarlock.plugins import HF_CompressedTensors_Adapter

    model = _TinyCausalModel()

    class _AutoModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            _ = model_id, kwargs
            return model

    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = _AutoModel
    compressed_tensors = ModuleType("compressed_tensors")
    monkeypatch.setitem(__import__("sys").modules, "transformers", transformers)
    monkeypatch.setitem(
        __import__("sys").modules,
        "compressed_tensors",
        compressed_tensors,
    )

    with pytest.raises(ModelLoadError, match="checkpoint metadata missing"):
        HF_CompressedTensors_Adapter().load_model("local-dense", device="cuda")
