import pytest
import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class NonCallableState:
    # Deliberately provide a non-callable state_dict attribute
    state_dict = 123


class CustomModule(nn.Module):
    def __init__(self):
        super().__init__()

    def state_dict(self, *args, **kwargs):
        # Return a dict with a non-tensor value to flip else branch
        return {"key": "not-a-tensor"}


class QuantizedStateModule(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        integer_storage = torch.tensor([[1, 2], [3, 4]], dtype=torch.int8)
        self.quantized_weight = torch._make_per_tensor_quantized_tensor(
            integer_storage,
            scale=scale,
            zero_point=0,
        )

    def state_dict(self, *args, **kwargs):
        return {"weight": self.quantized_weight}


class PerChannelQuantizedStateModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantized_weight = torch.quantize_per_channel(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            scales=torch.tensor([0.1, 0.2], dtype=torch.float64),
            zero_points=torch.tensor([0, 0], dtype=torch.int64),
            axis=0,
            dtype=torch.qint8,
        )

    def state_dict(self, *args, **kwargs):
        return {"weight": self.quantized_weight}


def test_fingerprint_targets_rejects_noncallable_target_state():
    g = VarianceGuard()
    g._target_modules = {
        "x.noncallable": NonCallableState(),
    }

    with pytest.raises(RuntimeError, match="fingerprinting failed"):
        g._fingerprint_targets()


def test_fingerprint_targets_handles_deterministic_non_tensor_state():
    g = VarianceGuard()
    g._target_modules = {"y.custom": CustomModule()}

    assert g._fingerprint_targets() == g._fingerprint_targets()


def test_fingerprint_targets_returns_hash():
    guard = VarianceGuard()
    guard._target_modules = {"transformer.h.0.mlp.c_proj": nn.Linear(2, 2, bias=False)}
    fingerprint = guard._fingerprint_targets()
    assert fingerprint is not None and len(fingerprint) == 64


def test_fingerprint_targets_is_bfloat16_safe_and_mutation_sensitive():
    guard = VarianceGuard()
    module = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    guard._target_modules = {"transformer.h.0.mlp.c_proj": module}

    before = guard._fingerprint_targets()
    repeated = guard._fingerprint_targets()
    with torch.no_grad():
        module.weight[0, 0] += torch.tensor(1.0, dtype=torch.bfloat16)
    after = guard._fingerprint_targets()

    assert before == repeated
    assert before != after


def test_fingerprint_targets_binds_quantization_parameters_not_only_storage():
    first = VarianceGuard()
    first._target_modules = {"target": QuantizedStateModule(scale=0.1)}
    second = VarianceGuard()
    second._target_modules = {"target": QuantizedStateModule(scale=0.2)}

    first_module = first._target_modules["target"]
    second_module = second._target_modules["target"]
    assert torch.equal(
        first_module.quantized_weight.int_repr(),
        second_module.quantized_weight.int_repr(),
    )
    assert first._fingerprint_targets() != second._fingerprint_targets()


def test_fingerprint_targets_binds_per_channel_quantization_contract():
    guard = VarianceGuard()
    guard._target_modules = {"target": PerChannelQuantizedStateModule()}

    fingerprint = guard._fingerprint_targets()

    assert fingerprint is not None and len(fingerprint) == 64
