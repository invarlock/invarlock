from __future__ import annotations

import json

from invarlock.core import backend_inventory as backend_inventory_mod
from invarlock.core.backend_inventory import (
    BACKEND_INVENTORY_SCHEMA,
    build_backend_inventory_for_adapter,
    build_backend_inventory_from_report,
    write_backend_inventory_sidecar,
)


def test_backend_inventory_sidecar_for_optional_quantized_adapter(tmp_path):
    report = {
        "meta": {"adapter": "hf_bnb"},
        "plugins": {
            "adapter": {
                "provenance": {"version": "0.47.0"},
                "quantization_config": {"load_in_8bit": True},
            }
        },
    }

    inventory = build_backend_inventory_from_report(report)
    assert inventory is not None
    assert inventory["schema"] == BACKEND_INVENTORY_SCHEMA
    assert inventory["adapter"] == "hf_bnb"
    assert inventory["backend"] == "bitsandbytes"
    assert inventory["backend_version"] == "0.47.0"
    assert inventory["quantization_config"] == {"load_in_8bit": True}
    assert inventory["load_smoke"] is False
    assert inventory["inference_smoke"] is False

    sidecar = write_backend_inventory_sidecar(report, tmp_path)
    assert sidecar is not None
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["schema"] == BACKEND_INVENTORY_SCHEMA


def test_backend_inventory_counts_live_quantized_modules(tmp_path):
    class Linear8bitLt:
        __module__ = "bitsandbytes.nn.modules"

    class RegularModule:
        __module__ = "torch.nn.modules.linear"

    class Model:
        def modules(self):
            return [self, Linear8bitLt(), RegularModule()]

        def get_memory_footprint(self):
            return 1234

    report = {"meta": {"adapter": "hf_bnb"}, "plugins": {"adapter": {}}}

    sidecar = write_backend_inventory_sidecar(report, tmp_path, model=Model())
    assert sidecar is not None
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["quantized_module_count"] == 1
    assert payload["load_smoke"] is False
    assert payload["inference_smoke"] is False
    assert payload["quantized_module_types"] == ["bitsandbytes.nn.modules.Linear8bitLt"]
    assert payload["memory_footprint"] == {
        "reported_bytes": 1234,
        "method": "get_memory_footprint",
    }


def test_backend_inventory_counts_awq_and_gptq_modules() -> None:
    class AwqMarlinLinear:
        __module__ = "gptqmodel.nn_modules.qlinear.marlin_awq"

    class QuantLinear:
        __module__ = "gptqmodel.nn_modules.qlinear"

    class Model:
        def __init__(self, modules):
            self._modules = modules

        def modules(self):
            return self._modules

    awq_inventory = build_backend_inventory_for_adapter(
        adapter="hf_awq",
        model=Model([AwqMarlinLinear()]),
    )
    assert awq_inventory is not None
    assert awq_inventory["backend"] == "gptqmodel"
    assert awq_inventory["quantized_module_count"] == 1
    assert awq_inventory["quantized_module_types"] == [
        "gptqmodel.nn_modules.qlinear.marlin_awq.AwqMarlinLinear"
    ]

    gptq_inventory = build_backend_inventory_for_adapter(
        adapter="hf_gptq",
        model=Model([QuantLinear()]),
    )
    assert gptq_inventory is not None
    assert gptq_inventory["quantized_module_count"] == 1
    assert gptq_inventory["quantized_module_types"] == [
        "gptqmodel.nn_modules.qlinear.QuantLinear"
    ]


def test_backend_inventory_counts_torchao_modules() -> None:
    class AffineQuantizedLinear:
        __module__ = "torchao.dtypes.affine_quantized_tensor"

    class Model:
        def modules(self):
            return [AffineQuantizedLinear()]

    inventory = build_backend_inventory_for_adapter(
        adapter="hf_torchao",
        model=Model(),
    )

    assert inventory is not None
    assert inventory["backend"] == "torchao"
    assert inventory["quantized_module_count"] == 1
    assert inventory["quantized_module_types"] == [
        "torchao.dtypes.affine_quantized_tensor.AffineQuantizedLinear"
    ]


def test_backend_inventory_counts_hqq_modules() -> None:
    class HQQLinear:
        __module__ = "hqq.core.quantize"

    class Model:
        def modules(self):
            return [HQQLinear()]

    inventory = build_backend_inventory_for_adapter(
        adapter="hf_hqq",
        model=Model(),
    )

    assert inventory is not None
    assert inventory["backend"] == "hqq"
    assert inventory["quantized_module_count"] == 1
    assert inventory["quantized_module_types"] == ["hqq.core.quantize.HQQLinear"]


def test_backend_inventory_counts_quanto_modules() -> None:
    class QLinear:
        __module__ = "optimum.quanto.nn.qlinear"

    class Model:
        def modules(self):
            return [QLinear()]

    inventory = build_backend_inventory_for_adapter(
        adapter="hf_quanto",
        model=Model(),
    )

    assert inventory is not None
    assert inventory["backend"] == "optimum-quanto"
    assert inventory["quantized_module_count"] == 1
    assert inventory["quantized_module_types"] == ["optimum.quanto.nn.qlinear.QLinear"]


def test_backend_inventory_counts_compressed_tensors_modules() -> None:
    class CompressedLinear:
        __module__ = "compressed_tensors.quantization.linear"

    class Model:
        def modules(self):
            return [CompressedLinear()]

    inventory = build_backend_inventory_for_adapter(
        adapter="hf_ct",
        model=Model(),
    )

    assert inventory is not None
    assert inventory["backend"] == "compressed-tensors"
    assert inventory["quantized_module_count"] == 1
    assert inventory["quantized_module_types"] == [
        "compressed_tensors.quantization.linear.CompressedLinear"
    ]


def test_backend_inventory_counts_gptq_named_modules_for_gptq_adapter() -> None:
    class GptqLinear:
        __module__ = "vendor.layers.gptq"

    class Model:
        def modules(self):
            return [GptqLinear()]

    inventory = build_backend_inventory_for_adapter(
        adapter="hf_gptq",
        model=Model(),
    )

    assert inventory is not None
    assert inventory["quantized_module_count"] == 1
    assert inventory["quantized_module_types"] == ["vendor.layers.gptq.GptqLinear"]


def test_backend_inventory_ignores_plain_modules_for_gptq_adapter() -> None:
    class PlainLinear:
        __module__ = "torch.nn.modules.linear"

    class Model:
        def modules(self):
            return [PlainLinear()]

    inventory = build_backend_inventory_for_adapter(
        adapter="hf_gptq",
        model=Model(),
    )

    assert inventory is not None
    assert inventory["quantized_module_count"] == 0
    assert inventory["quantized_module_types"] == []


def test_backend_inventory_ignores_plain_modules_for_unknown_adapter_key() -> None:
    class PlainLinear:
        __module__ = "torch.nn.modules.linear"

    class Model:
        def modules(self):
            return [PlainLinear()]

    inventory = backend_inventory_mod._quantized_module_inventory(Model(), adapter="hf")

    assert inventory == {"count": 0, "types": []}


def test_backend_inventory_handles_non_module_models_and_memory_errors() -> None:
    class NoModules:
        modules = "not-callable"

    class MemoryErrorModel:
        def modules(self):
            return []

        def get_memory_footprint(self):
            raise RuntimeError("unavailable")

    no_modules = build_backend_inventory_for_adapter(
        adapter="hf_bnb", model=NoModules()
    )
    assert no_modules is not None
    assert no_modules["quantized_module_count"] == 0
    assert no_modules["quantized_module_types"] == []

    memory_error = build_backend_inventory_for_adapter(
        adapter="hf_bnb",
        model=MemoryErrorModel(),
    )
    assert memory_error is not None
    assert memory_error["memory_footprint"] == {
        "reported_bytes": 0,
        "method": "unknown",
    }


def test_backend_inventory_uses_model_profile_quantization_config() -> None:
    report = {
        "meta": {
            "adapter": "hf_awq",
            "model_profile": {"quantization_config": {"bits": 4}},
        },
        "plugins": {"adapter": "not-a-mapping"},
    }

    inventory = build_backend_inventory_from_report(report)

    assert inventory is not None
    assert inventory["quantization_config"] == {"bits": 4}


def test_backend_inventory_tolerates_version_lookup_errors(monkeypatch) -> None:
    def _raise_runtime_error(name: str) -> str:
        raise RuntimeError(name)

    monkeypatch.setattr(
        "invarlock.core.backend_inventory.pkg_version",
        _raise_runtime_error,
    )

    inventory = build_backend_inventory_for_adapter(adapter="hf_gptq")

    assert inventory is not None
    assert inventory["backend_version"] is None
    assert inventory["transformers_version"] is None


def test_backend_inventory_can_write_prebuilt_payload(tmp_path):
    inventory = build_backend_inventory_for_adapter(
        adapter="hf_bnb",
        backend_version="0.49.2",
        quantization_config={"load_in_8bit": True},
        model=None,
        load_smoke=True,
        inference_smoke=True,
    )
    assert inventory is not None
    inventory["quantized_module_count"] = 3

    sidecar = write_backend_inventory_sidecar({}, tmp_path, inventory=inventory)
    assert sidecar is not None
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["adapter"] == "hf_bnb"
    assert payload["quantized_module_count"] == 3
    assert payload["load_smoke"] is True
    assert payload["inference_smoke"] is True


def test_backend_inventory_skips_core_adapter(tmp_path):
    report = {"meta": {"adapter": "hf_causal"}, "plugins": {"adapter": {}}}

    assert build_backend_inventory_from_report(report) is None
    assert write_backend_inventory_sidecar(report, tmp_path) is None


def test_backend_inventory_skips_non_mapping_report(tmp_path):
    assert build_backend_inventory_from_report(["not", "a", "mapping"]) is None
    assert write_backend_inventory_sidecar(["not", "a", "mapping"], tmp_path) is None
