from __future__ import annotations

import json
from importlib import import_module as stdlib_import_module
from types import SimpleNamespace

import pytest

from invarlock.core import runtime_quantization_proof as runtime_quantization_proof_mod
from invarlock.core.runtime_quantization_proof import (
    RUNTIME_QUANTIZATION_PROOF_FILENAME,
    RUNTIME_QUANTIZATION_PROOF_SCHEMA,
    build_runtime_quantization_proof,
    write_runtime_quantization_proof_sidecar,
)
from invarlock.gptqmodel_runtime import GPTQModelRuntimeStatus


class _Model:
    def __init__(
        self,
        *modules: object,
        quantization_config: object | None = None,
    ) -> None:
        self._modules = modules
        self.config = SimpleNamespace(quantization_config=quantization_config)

    def modules(self):
        return (self, *self._modules)


def _module(module_path: str, name: str) -> object:
    runtime_type = type(name, (), {"__module__": module_path})
    _TEST_RUNTIME_MODULES.setdefault(module_path, {})[name] = runtime_type
    return runtime_type()


_TEST_RUNTIME_MODULES: dict[str, dict[str, type[object]]] = {}


@pytest.fixture(autouse=True)
def registered_runtime_type_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make test doubles represent identities exported by backend modules."""

    def import_for_test(module_name: str):
        exported_types = _TEST_RUNTIME_MODULES.get(module_name)
        if exported_types is None:
            return stdlib_import_module(module_name)
        return SimpleNamespace(**exported_types)

    _TEST_RUNTIME_MODULES.clear()
    monkeypatch.setattr(
        runtime_quantization_proof_mod,
        "import_module",
        import_for_test,
    )
    yield
    _TEST_RUNTIME_MODULES.clear()


def _gptqmodel_runtime_status(
    *,
    importable: bool = True,
    bridge_required: bool = False,
    bridge_applied: bool = False,
    bridge_error_type: str | None = None,
) -> GPTQModelRuntimeStatus:
    return GPTQModelRuntimeStatus(
        importable=importable,
        gptqmodel_version="7.0.0" if importable else None,
        import_error_type=None if importable else "ImportError",
        compatibility_bridge_required=bridge_required,
        compatibility_bridge_applied=bridge_applied,
        compatibility_bridge_missing_symbols=(),
        compatibility_bridge_error_type=bridge_error_type,
        jit_toolchain=None,
    )


@pytest.fixture
def gptqmodel_runtime_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime_quantization_proof_mod,
        "prepare_gptqmodel_runtime",
        lambda: _gptqmodel_runtime_status(),
    )


@pytest.mark.parametrize(
    ("adapter", "module_path", "name", "backend"),
    [
        ("hf_bnb", "bitsandbytes.nn.modules", "Linear8bitLt", "bitsandbytes"),
        (
            "hf_awq",
            "gptqmodel.nn_modules.qlinear.gemm_awq",
            "AwqGEMMLinear",
            "gptqmodel",
        ),
        (
            "hf_gptq",
            "gptqmodel.nn_modules.qlinear.marlin",
            "MarlinLinear",
            "gptqmodel",
        ),
        ("hf_hqq", "hqq.core.quantize", "HQQLinear", "hqq"),
        ("hf_quanto", "optimum.quanto.nn.qlinear", "QLinear", "optimum-quanto"),
    ],
)
def test_runtime_quantization_proof_requires_recognized_live_runtime_types(
    adapter: str,
    module_path: str,
    name: str,
    backend: str,
    gptqmodel_runtime_available: None,
) -> None:
    proof = build_runtime_quantization_proof(
        adapter=adapter,
        model=_Model(_module(module_path, name)),
    )

    assert proof is not None
    assert proof["schema"] == RUNTIME_QUANTIZATION_PROOF_SCHEMA
    assert proof["adapter"] == adapter
    assert proof["backend"] == backend
    assert proof["ok"] is True
    assert proof["status"] == "verified_live_runtime_types"
    assert proof["live_model_observed"] is True
    assert proof["module_inventory_observed"] is True
    assert proof["recognized_quantized_runtime_type_count"] == 1
    assert proof["recognized_quantized_runtime_types"] == [f"{module_path}.{name}"]
    assert proof["live_model_quantization_method"] is None
    if adapter in {"hf_awq", "hf_gptq"}:
        assert proof["backend_runtime_importable"] is True
        assert proof["backend_runtime_import_error_type"] is None
        assert proof["backend_runtime_version"] == "7.0.0"
        assert proof["backend_runtime_compatibility_bridge_required"] is False
        assert proof["backend_runtime_compatibility_bridge_applied"] is False
    else:
        assert proof["backend_runtime_importable"] is None
    assert proof["artifact_binding"] == "not_attempted"
    assert proof["packed_storage_artifact_proof_required"] is False


@pytest.mark.parametrize(
    ("weight_module", "weight_name"),
    [
        ("torchao.quantization", "Int8Tensor"),
        (
            "torchao.dtypes.affine_quantized_tensor",
            "AffineQuantizedTensor",
        ),
    ],
)
def test_runtime_quantization_proof_recognizes_torchao_weight_type(
    weight_module: str,
    weight_name: str,
) -> None:
    linear = _module("torch.nn.modules.linear", "Linear")
    linear.weight = _module(weight_module, weight_name)

    proof = build_runtime_quantization_proof(
        adapter="hf_torchao",
        model=_Model(linear),
    )

    assert proof is not None
    assert proof["ok"] is True
    assert proof["recognized_quantized_runtime_type_count"] == 1
    assert proof["recognized_quantized_runtime_types"] == [
        f"{weight_module}.{weight_name}"
    ]


def test_runtime_quantization_proof_rejects_dense_torch_linear_and_unbound_torchao_marker() -> (
    None
):
    dense_linear = _module("torch.nn.modules.linear", "Linear")
    dense_linear.unbound_torchao_marker = _module("torchao.quantization", "Int8Tensor")

    proof = build_runtime_quantization_proof(
        adapter="hf_torchao",
        model=_Model(dense_linear),
    )

    assert proof is not None
    assert proof["ok"] is False
    assert proof["reason"] == "no_recognized_quantized_runtime_types"


def test_runtime_quantization_proof_rejects_forged_backend_type_identity() -> None:
    real_runtime_module = "bitsandbytes.nn.modules"
    _module(real_runtime_module, "Linear8bitLt")
    impostor = type("Impostor", (), {})()
    type(impostor).__module__ = real_runtime_module
    type(impostor).__qualname__ = "Linear8bitLt"

    proof = build_runtime_quantization_proof(
        adapter="hf_bnb",
        model=_Model(impostor),
    )

    assert proof is not None
    assert proof["ok"] is False
    assert proof["reason"] == "no_recognized_quantized_runtime_types"


@pytest.mark.parametrize(
    ("adapter", "quantization_config", "expected_method"),
    [
        ("hf_awq", {"quant_method": "awq"}, "awq"),
        ("hf_gptq", {"quant_method": "gptq"}, "gptq"),
        ("hf_awq", {"quantization_method": "awq"}, "awq"),
    ],
)
def test_runtime_quantization_proof_accepts_generic_gptqmodel_qlinear_only_with_matching_config(
    adapter: str,
    quantization_config: object,
    expected_method: str,
    gptqmodel_runtime_available: None,
) -> None:
    proof = build_runtime_quantization_proof(
        adapter=adapter,
        model=_Model(
            _module("gptqmodel.nn_modules.qlinear", "QuantLinear"),
            quantization_config=quantization_config,
        ),
    )

    assert proof is not None
    assert proof["ok"] is True
    assert proof["live_model_quantization_method"] == expected_method


@pytest.mark.parametrize(
    ("adapter", "quantization_config"),
    [
        ("hf_awq", None),
        ("hf_gptq", None),
        ("hf_awq", {"quant_method": "gptq"}),
        ("hf_gptq", {"quant_method": "awq"}),
    ],
)
def test_runtime_quantization_proof_rejects_ambiguous_gptqmodel_qlinear_without_matching_config(
    adapter: str,
    quantization_config: object | None,
    gptqmodel_runtime_available: None,
) -> None:
    proof = build_runtime_quantization_proof(
        adapter=adapter,
        model=_Model(
            _module("gptqmodel.nn_modules.qlinear", "QuantLinear"),
            quantization_config=quantization_config,
        ),
    )

    assert proof is not None
    assert proof["ok"] is False
    assert proof["reason"] == "no_recognized_quantized_runtime_types"


def test_runtime_quantization_proof_accepts_exact_config_class_for_ambiguous_qlinear(
    gptqmodel_runtime_available: None,
) -> None:
    AWQConfig = type("AWQConfig", (), {})
    GPTQConfig = type("GPTQConfig", (), {})
    generic_module = _module("gptqmodel.nn_modules.qlinear", "QuantLinear")

    awq_proof = build_runtime_quantization_proof(
        adapter="hf_awq",
        model=_Model(generic_module, quantization_config=AWQConfig()),
    )
    gptq_proof = build_runtime_quantization_proof(
        adapter="hf_gptq",
        model=_Model(generic_module, quantization_config=GPTQConfig()),
    )

    assert awq_proof is not None
    assert awq_proof["ok"] is True
    assert awq_proof["live_model_quantization_method"] == "awq"
    assert gptq_proof is not None
    assert gptq_proof["ok"] is True
    assert gptq_proof["live_model_quantization_method"] == "gptq"


@pytest.mark.parametrize(
    ("adapter", "module_path", "name", "quantization_config"),
    [
        (
            "hf_gptq",
            "gptqmodel.nn_modules.qlinear.gemm_awq",
            "AwqGEMMLinear",
            {"quant_method": "gptq"},
        ),
        (
            "hf_awq",
            "gptqmodel.nn_modules.qlinear.marlin",
            "MarlinLinear",
            {"quant_method": "awq"},
        ),
        (
            "hf_gptq",
            "gptqmodel.nn_modules.qlinear.fp4",
            "FP4Linear",
            {"quant_method": "gptq"},
        ),
    ],
)
def test_runtime_quantization_proof_rejects_cross_family_and_unknown_gptqmodel_wrappers(
    adapter: str,
    module_path: str,
    name: str,
    quantization_config: object,
    gptqmodel_runtime_available: None,
) -> None:
    proof = build_runtime_quantization_proof(
        adapter=adapter,
        model=_Model(
            _module(module_path, name),
            quantization_config=quantization_config,
        ),
    )

    assert proof is not None
    assert proof["ok"] is False
    assert proof["reason"] == "no_recognized_quantized_runtime_types"


def test_runtime_quantization_proof_records_named_runtime_boundary_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_quantization_proof_mod,
        "prepare_gptqmodel_runtime",
        lambda: _gptqmodel_runtime_status(
            bridge_required=True,
            bridge_applied=True,
        ),
    )

    proof = build_runtime_quantization_proof(
        adapter="hf_gptq",
        model=_Model(_module("gptqmodel.nn_modules.qlinear.marlin", "MarlinLinear")),
    )

    assert proof is not None
    assert proof["ok"] is True
    assert proof["backend_runtime_compatibility_bridge_required"] is True
    assert proof["backend_runtime_compatibility_bridge_applied"] is True


def test_runtime_quantization_proof_fails_closed_when_gptqmodel_runtime_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_quantization_proof_mod,
        "prepare_gptqmodel_runtime",
        lambda: _gptqmodel_runtime_status(importable=False),
    )

    proof = build_runtime_quantization_proof(
        adapter="hf_awq",
        model=_Model(
            _module(
                "gptqmodel.nn_modules.qlinear.gemm_awq",
                "AwqGEMMLinear",
            )
        ),
    )

    assert proof is not None
    assert proof["ok"] is False
    assert proof["status"] == "unavailable"
    assert proof["reason"] == "gptqmodel_runtime_import_failed"
    assert proof["backend_runtime_importable"] is False
    assert proof["backend_runtime_import_error_type"] == "ImportError"
    assert proof["recognized_quantized_runtime_type_count"] == 0


def test_runtime_quantization_proof_fails_closed_when_required_bridge_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_quantization_proof_mod,
        "prepare_gptqmodel_runtime",
        lambda: _gptqmodel_runtime_status(
            bridge_required=True,
            bridge_applied=False,
            bridge_error_type="AttributeError",
        ),
    )

    proof = build_runtime_quantization_proof(
        adapter="hf_awq",
        model=_Model(
            _module(
                "gptqmodel.nn_modules.qlinear.gemm_awq",
                "AwqGEMMLinear",
            )
        ),
    )

    assert proof is not None
    assert proof["ok"] is False
    assert proof["status"] == "unavailable"
    assert proof["reason"] == "gptqmodel_runtime_import_failed"
    assert proof["backend_runtime_importable"] is False
    assert proof["backend_runtime_import_error_type"] == "AttributeError"
    assert proof["backend_runtime_compatibility_bridge_required"] is True
    assert proof["backend_runtime_compatibility_bridge_applied"] is False


def test_runtime_quantization_proof_fails_closed_for_dense_or_missing_model() -> None:
    dense = build_runtime_quantization_proof(
        adapter="hf_bnb",
        model=_Model(_module("torch.nn.modules.linear", "Linear")),
    )
    missing = build_runtime_quantization_proof(adapter="hf_bnb", model=None)

    assert dense is not None
    assert dense["ok"] is False
    assert dense["status"] == "unverified"
    assert dense["reason"] == "no_recognized_quantized_runtime_types"
    assert dense["recognized_quantized_runtime_type_count"] == 0
    assert dense["recognized_quantized_runtime_types"] == []
    assert missing is not None
    assert missing["ok"] is False
    assert missing["status"] == "unverified"
    assert missing["reason"] == "live_model_missing"
    assert missing["live_model_observed"] is False


def test_compressed_tensors_requires_dedicated_packed_storage_proof() -> None:
    class CompressedModel:
        def __init__(self) -> None:
            self.module_inventory_requested = False

        def modules(self):
            self.module_inventory_requested = True
            raise AssertionError("compressed-tensors must not use module inventory")

    model = CompressedModel()
    proof = build_runtime_quantization_proof(
        adapter="hf_ct",
        model=model,
    )

    assert proof is not None
    assert proof["ok"] is False
    assert proof["status"] == "unsupported"
    assert proof["reason"] == "packed_storage_artifact_proof_required"
    assert proof["packed_storage_artifact_proof_required"] is True
    assert proof["module_inventory_observed"] is False
    assert proof["recognized_quantized_runtime_type_count"] is None
    assert proof["recognized_quantized_runtime_types"] == []
    assert model.module_inventory_requested is False


def test_runtime_quantization_proof_fails_closed_when_module_inventory_is_unavailable() -> (
    None
):
    class FallbackModel:
        def modules(self):
            raise RuntimeError("adapter fell back to an opaque model")

    proof = build_runtime_quantization_proof(adapter="hf_bnb", model=FallbackModel())

    assert proof is not None
    assert proof["ok"] is False
    assert proof["status"] == "unverified"
    assert proof["reason"] == "module_inventory_unavailable"
    assert proof["recognized_quantized_runtime_type_count"] == 0


def test_runtime_quantization_proof_never_treats_unknown_adapter_as_verified() -> None:
    assert (
        build_runtime_quantization_proof(
            adapter="hf_causal",
            model=_Model(_module("bitsandbytes.nn.modules", "Linear8bitLt")),
        )
        is None
    )


def test_runtime_quantization_proof_sidecar_persists_non_success_result(
    tmp_path,
) -> None:
    proof = build_runtime_quantization_proof(
        adapter="hf_bnb",
        model=_Model(_module("torch.nn.modules.linear", "Linear")),
    )

    sidecar = write_runtime_quantization_proof_sidecar(tmp_path, proof)

    assert sidecar == tmp_path / RUNTIME_QUANTIZATION_PROOF_FILENAME
    assert json.loads(sidecar.read_text(encoding="utf-8")) == proof


def test_runtime_proof_helpers_fail_closed_for_unresolvable_type_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert runtime_quantization_proof_mod._backend_version(None) is None
    assert runtime_quantization_proof_mod._runtime_type_name(1) == "builtins.int"

    unnamed_module_type = type("RuntimeType", (), {})
    unnamed_module_type.__module__ = ""
    assert (
        runtime_quantization_proof_mod._resolve_imported_runtime_type(
            unnamed_module_type()
        )
        is None
    )

    invalid_qualname_type = type("RuntimeType", (), {})
    invalid_qualname_type.__qualname__ = "not-valid!"
    assert (
        runtime_quantization_proof_mod._resolve_imported_runtime_type(
            invalid_qualname_type()
        )
        is None
    )

    missing_type = type("Missing", (), {"__module__": "missing.backend"})
    assert (
        runtime_quantization_proof_mod._resolve_imported_runtime_type(missing_type())
        is None
    )


def test_live_quantization_method_fails_closed_on_config_access_errors() -> None:
    class ExplodingModel:
        @property
        def config(self):
            raise RuntimeError("unavailable")

    class ExplodingQuantizationConfig:
        @property
        def quant_method(self):
            raise RuntimeError("unavailable")

        @property
        def quantization_method(self):
            raise RuntimeError("unavailable")

    assert (
        runtime_quantization_proof_mod._live_quantization_method(ExplodingModel())
        is None
    )
    assert (
        runtime_quantization_proof_mod._live_quantization_method(
            _Model(quantization_config=ExplodingQuantizationConfig())
        )
        is None
    )
    assert runtime_quantization_proof_mod._live_quantization_method(_Model()) is None
    assert (
        runtime_quantization_proof_mod._live_quantization_method(
            _Model(quantization_config={"quant_method": 1})
        )
        is None
    )
    assert (
        runtime_quantization_proof_mod._live_quantization_method(
            _Model(quantization_config={"quant_method": "unknown"})
        )
        is None
    )


def test_supported_adapter_with_missing_backend_inventory_is_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_quantization_proof_mod,
        "quantized_adapter_backend",
        lambda _adapter: None,
    )

    proof = build_runtime_quantization_proof(adapter="hf_bnb", model=_Model())

    assert proof is not None
    assert proof["ok"] is False
    assert proof["reason"] == "quantized_backend_unrecognized"


def test_runtime_proof_sidecar_writer_rejects_absent_or_invalid_contract(
    tmp_path,
) -> None:
    assert write_runtime_quantization_proof_sidecar(tmp_path, None) is None
    with pytest.raises(ValueError, match="schema is invalid"):
        write_runtime_quantization_proof_sidecar(tmp_path, {"schema": "v0", "ok": True})
    with pytest.raises(ValueError, match="ok must be boolean"):
        write_runtime_quantization_proof_sidecar(
            tmp_path,
            {"schema": RUNTIME_QUANTIZATION_PROOF_SCHEMA, "ok": 1},
        )
