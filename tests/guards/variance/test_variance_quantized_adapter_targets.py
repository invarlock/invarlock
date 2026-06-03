from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from invarlock.guards.quantized_weights import (
    is_packed_quantized_module,
    is_quantized_weight,
)
from invarlock.guards.variance import VarianceGuard
from invarlock.guards.variance_scaling import equalise_residual_variance


class _OpaqueQuantizedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=1)
        self.embed = nn.Embedding(16, 4)
        self.adapter_attn = nn.Linear(4, 4, bias=False)
        self.adapter_mlp = nn.Linear(4, 4, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(input_ids).float()
        hidden = self.adapter_attn(hidden)
        hidden = torch.relu(hidden)
        return self.adapter_mlp(hidden)


class _QuantizedAdapter:
    def describe(self, model) -> dict[str, int]:
        return {"n_layer": 1}

    def get_layer_modules(self, model, index: int):
        assert index == 0
        return {
            "self_attn.o_proj": model.adapter_attn,
            "mlp.down_proj": model.adapter_mlp,
        }


class _PackedInt8Projection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.zeros(4, 4, dtype=torch.int8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class _TorchQuantizedProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.quantize_per_tensor(
            torch.ones(4, 4), scale=0.1, zero_point=0, dtype=torch.qint8
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class _PackedAWQProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qweight = torch.zeros(4, 1, dtype=torch.int32)
        self.qzeros = torch.zeros(1, 1, dtype=torch.int32)
        self.scales = torch.ones(1, 4, dtype=torch.float16)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class _LinearWeightMetadata:
    dtype = "metadata"


class _PackedMetadataProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = _LinearWeightMetadata()
        self.qweight = torch.zeros(4, 1, dtype=torch.int32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * 3.0


class _NoWeightProjection(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * 2.0


class _UnitVarianceProjection(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(inputs, dtype=torch.float32)


def test_adapter_projection_aliases_resolve_to_canonical_variance_targets() -> None:
    model = _OpaqueQuantizedModel()
    guard = VarianceGuard(
        policy={
            "scope": "both",
            "tap": ["transformer.h.*.attn.c_proj", "transformer.h.*.mlp.c_proj"],
            "max_calib": 0,
            "min_gain": 0.0,
        }
    )

    targets = guard._resolve_target_modules(model, adapter=_QuantizedAdapter())

    assert targets["transformer.h.0.attn.c_proj"] is model.adapter_attn
    assert targets["transformer.h.0.mlp.c_proj"] is model.adapter_mlp
    assert guard._stats["target_resolution"]["fallback_used"] is True


def test_packed_quantized_adapter_projection_resolves_as_explicit_target() -> None:
    model = _OpaqueQuantizedModel()
    model.adapter_mlp = _PackedAWQProjection()
    guard = VarianceGuard(
        policy={
            "scope": "ffn",
            "tap": ["transformer.h.*.mlp.c_proj"],
            "max_calib": 0,
            "min_gain": 0.0,
        }
    )

    targets = guard._resolve_target_modules(model, adapter=_QuantizedAdapter())

    assert targets["transformer.h.0.mlp.c_proj"] is model.adapter_mlp
    assert guard._stats["target_resolution"]["fallback_used"] is True
    assert guard._stats["target_resolution"]["matched"] == [
        "transformer.h.0.mlp.c_proj"
    ]


def test_prepare_policy_refreshes_tap_patterns_before_target_resolution() -> None:
    model = _OpaqueQuantizedModel()
    guard = VarianceGuard(
        policy={
            "scope": "ffn",
            "tap": ["transformer.h.*.attn.c_proj"],
            "max_calib": 0,
            "min_gain": 0.0,
        }
    )

    result = guard.prepare(
        model,
        adapter=_QuantizedAdapter(),
        calib=[],
        policy={
            "scope": "ffn",
            "tap": ["transformer.h.*.mlp.c_proj"],
            "max_calib": 0,
            "min_gain": 0.0,
        },
    )

    assert result["ready"] is True
    assert guard._stats["tap"] == ["transformer.h.*.mlp.c_proj"]
    assert guard._stats["target_module_names"] == ["transformer.h.0.mlp.c_proj"]


def test_equalise_residual_variance_can_hook_adapter_target_modules() -> None:
    model = _OpaqueQuantizedModel()
    targets = {
        "transformer.h.0.attn.c_proj": model.adapter_attn,
        "transformer.h.0.mlp.c_proj": model.adapter_mlp,
    }
    dataloader = [{"input_ids": torch.tensor([[1, 2, 3, 4]])}]

    scales = equalise_residual_variance(
        model,
        dataloader,
        windows=1,
        tol=0.0,
        allow_empty=False,
        apply=False,
        device="cpu",
        target_modules=targets,
    )

    assert "transformer.h.0.attn.c_proj" in scales
    assert "transformer.h.0.mlp.c_proj" in scales


def test_equalise_residual_variance_does_not_report_int8_noop_as_applied() -> None:
    model = _OpaqueQuantizedModel()
    model.adapter_mlp = _PackedInt8Projection()
    target_name = "transformer.h.0.mlp.c_proj"

    scales = equalise_residual_variance(
        model,
        [{"input_ids": torch.tensor([[1, 2, 3, 4]])}],
        windows=1,
        tol=0.0,
        allow_empty=False,
        apply=True,
        device="cpu",
        target_modules={target_name: model.adapter_mlp},
    )

    assert scales == {}


def test_equalise_residual_variance_skips_target_without_hook() -> None:
    model = _OpaqueQuantizedModel()
    scales = equalise_residual_variance(
        model,
        [{"input_ids": torch.tensor([[1, 2, 3, 4]])}],
        windows=1,
        allow_empty=False,
        apply=True,
        device="cpu",
        target_modules={"transformer.h.0.mlp.c_proj": object()},
    )

    assert scales == {}


def test_equalise_residual_variance_skips_adapter_target_without_alpha() -> None:
    model = _OpaqueQuantizedModel()
    model.adapter_mlp = _UnitVarianceProjection()
    target_name = "transformer.h.0.mlp.c_proj"

    scales = equalise_residual_variance(
        model,
        [{"input_ids": torch.tensor([[1, 2, 3, 4]])}],
        windows=1,
        tol=0.02,
        allow_empty=False,
        apply=True,
        device="cpu",
        target_modules={target_name: model.adapter_mlp},
    )

    assert scales == {}


def test_equalise_residual_variance_does_not_report_no_weight_target_as_applied() -> (
    None
):
    model = _OpaqueQuantizedModel()
    model.adapter_mlp = _NoWeightProjection()
    target_name = "transformer.h.0.mlp.c_proj"

    scales = equalise_residual_variance(
        model,
        [{"input_ids": torch.tensor([[1, 2, 3, 4]])}],
        windows=1,
        tol=0.0,
        allow_empty=False,
        apply=True,
        device="cpu",
        target_modules={target_name: model.adapter_mlp},
    )

    assert scales == {}


def test_quantized_weight_detector_recognizes_packed_tensor_variants() -> None:
    torch_quantized = torch.quantize_per_tensor(
        torch.ones(4, 4), scale=0.1, zero_point=0, dtype=torch.qint8
    )
    AffineQuantizedTensor = type("AffineQuantizedTensor", (), {"dtype": torch.float32})

    assert is_quantized_weight(torch_quantized) is True
    assert is_quantized_weight(AffineQuantizedTensor()) is True
    assert is_packed_quantized_module(_PackedAWQProjection()) is True
    assert is_quantized_weight(torch.ones(4, 4)) is False


@pytest.mark.parametrize(
    "qualified_name",
    [
        "hqq.core.quantize.HQQLinear",
        "optimum.quanto.nn.qlinear.QLinear",
        "eetq.layers.EetqLinear",
        "compressed_tensors.quantization.linear.CompressedLinear",
        "llmcompressor.modifiers.quantization.CompressedTensorsLinear",
        "auto_gptq.nn_modules.qlinear.qlinear_cuda_old.QuantLinear",
        "exllamav2.ext.QLinearExllamaV2",
        "fbgemm_gpu.experimental.gen_ai.fp8.FP8Linear",
        "aqlm.inference.QuantizedLinear",
    ],
)
def test_quantized_weight_detector_recognizes_future_backend_markers(
    qualified_name: str,
) -> None:
    module_name, class_name = qualified_name.rsplit(".", maxsplit=1)
    weight_cls = type(
        class_name, (), {"__module__": module_name, "dtype": torch.float32}
    )
    module_cls = type(class_name, (nn.Module,), {"__module__": module_name})

    assert is_quantized_weight(weight_cls()) is True
    assert is_packed_quantized_module(module_cls()) is True


@pytest.mark.parametrize(
    "projection_cls",
    [_PackedInt8Projection, _TorchQuantizedProjection, _PackedAWQProjection],
)
def test_packed_quantized_variance_contract_is_fail_closed(
    monkeypatch, projection_cls
) -> None:
    model = _OpaqueQuantizedModel()
    target_name = "transformer.h.0.mlp.c_proj"
    guard = VarianceGuard(
        policy={
            "scope": "both",
            "tap": ["transformer.h.*.mlp.c_proj"],
            "max_calib": 0,
            "min_gain": 0.0,
        }
    )
    guard._prepared = True
    guard._post_edit_evaluated = True
    guard._target_modules = {target_name: projection_cls()}
    guard._scales = {target_name: 0.9}
    guard._calibration_stats = {"status": "complete"}
    guard._ab_gain = 0.1
    guard._ppl_no_ve = 10.0
    guard._ppl_with_ve = 9.0
    guard._ratio_ci = (0.8, 0.9)
    monkeypatch.setattr(guard, "_evaluate_ab_gate", lambda: (True, "forced"))

    result = guard.validate(model, adapter=None, context={})

    assert result.passed is False
    assert result.decision == "block"
    assert result.extras == {
        "supported": False,
        "reason": "packed_quantized_weight_mutation_unsupported",
        "assurance_blocking": True,
        "status": "unsupported",
    }
    blockers = guard._stats["quantized_mutation_unsupported"]
    assert blockers[0]["module"] == target_name
    assert blockers[0]["assurance_blocking"] is True


def test_packed_metadata_projection_prepare_keeps_real_unsupported_reason() -> None:
    model = _OpaqueQuantizedModel()
    model.adapter_mlp = _PackedMetadataProjection()
    guard = VarianceGuard(
        policy={
            "scope": "ffn",
            "tap": ["transformer.h.*.mlp.c_proj"],
            "max_calib": 20,
            "min_gain": 0.0,
            "deadband": 0.0,
            "calibration": {"windows": 2, "min_coverage": 1, "seed": 123},
        }
    )
    calibration = [
        {"input_ids": torch.tensor([[1, 2, 3, 4]]), "window_id": "0"},
        {"input_ids": torch.tensor([[2, 3, 4, 5]]), "window_id": "1"},
    ]

    result = guard.prepare(model, adapter=_QuantizedAdapter(), calib=calibration)

    assert result["ready"] is True
    assert guard._prepare_failure is None
    assert guard._stats["target_module_names"] == ["transformer.h.0.mlp.c_proj"]
    blockers = guard._stats["quantized_mutation_unsupported"]
    assert blockers[0]["reason"] == "packed_quantized_weight_mutation_unsupported"

    validation = guard.validate(model, adapter=_QuantizedAdapter(), context={})

    assert validation.passed is False
    assert validation.extras == {
        "supported": False,
        "reason": "packed_quantized_weight_mutation_unsupported",
        "assurance_blocking": True,
        "status": "unsupported",
    }


def test_validate_unprepared_guard_maps_quantized_prepare_failure_reason() -> None:
    guard = VarianceGuard(policy={"scope": "ffn"})
    guard._prepare_failure = {
        "reason": "packed_quantized_weight_mutation_unsupported",
        "message": "Mutation unsupported for adapter target",
    }

    validation = guard.validate(nn.Module(), adapter=None, context={})

    assert validation.passed is False
    assert validation.extras == {
        "supported": False,
        "reason": "packed_quantized_weight_mutation_unsupported",
        "assurance_blocking": True,
        "status": "unsupported",
    }
    assert (
        validation.details["prepare_failure"]["reason"]
        == "packed_quantized_weight_mutation_unsupported"
    )


def test_finalize_unprepared_guard_rewrites_no_target_prepare_failure_message() -> None:
    guard = VarianceGuard(policy={"scope": "ffn"})
    guard._prepare_failure = {
        "reason": "no_variance_targets",
        "message": "Adapter target resolution matched no modules",
    }

    result = guard.finalize(nn.Module())

    assert result["passed"] is False
    assert result["errors"] == ["Preparation failed or no target modules found"]
    assert result["details"]["prepare_failure"]["reason"] == "no_variance_targets"
