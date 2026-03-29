from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from invarlock.guards import rmt_activation_runtime as runtime
from invarlock.guards import rmt_result_contract


class IndexedSource:
    def __len__(self) -> int:
        return 5

    def __getitem__(self, idx: int) -> int:
        return idx


class IterableOnly:
    def __iter__(self):
        return iter([10, 20, 30])


class BrokenLenSource:
    def __len__(self) -> int:
        raise RuntimeError("bad len")

    def __getitem__(self, idx: int) -> int:
        return idx

    def __iter__(self):
        return iter([7, 8, 9])


class SkippingIndexedSource:
    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx: int) -> int:
        if idx == 1:
            raise RuntimeError("skip")
        return idx


def test_collect_calibration_batches_supports_index_and_iterable_sources() -> None:
    assert runtime.collect_calibration_batches(
        IndexedSource(),
        3,
        activation_sampling={"windows": {"indices_policy": "last"}},
    ) == [2, 3, 4]
    assert runtime.collect_calibration_batches(
        IterableOnly(),
        2,
    ) == [10, 20]
    assert runtime.collect_calibration_batches(object(), 2) == []


def test_collect_calibration_batches_skips_invalid_index_records() -> None:
    assert runtime.collect_calibration_batches(
        SkippingIndexedSource(),
        3,
        activation_sampling={"windows": {"indices_policy": "unknown"}},
    ) == [0, 2]


def test_collect_calibration_batches_handles_non_mapping_policy_and_len_failure() -> (
    None
):
    assert runtime.collect_calibration_batches(
        IndexedSource(),
        2,
        activation_sampling=object(),
    ) == [0, 4]
    assert runtime.collect_calibration_batches(BrokenLenSource(), 2) == [7, 8]


def test_prepare_activation_inputs_normalizes_and_falls_back_to_clone(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        torch.Tensor,
        "to",
        lambda self, device: (_ for _ in ()).throw(RuntimeError("no device")),
    )

    input_ids, attention_mask = runtime.prepare_activation_inputs(
        {
            "input_ids": torch.tensor([1, 2]),
            "attention_mask": torch.tensor([1, 1]),
        },
        torch.device("cpu"),
    )

    assert input_ids is not None
    assert attention_mask is not None
    assert tuple(input_ids.shape) == (1, 2)
    assert tuple(attention_mask.shape) == (1, 2)


def test_prepare_activation_inputs_drops_mask_when_clone_fallback_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_clone = torch.Tensor.clone

    monkeypatch.setattr(
        torch.Tensor,
        "to",
        lambda self, device: (_ for _ in ()).throw(RuntimeError("no device")),
    )

    def _clone(self):
        if int(self.sum().item()) == 2:
            raise RuntimeError("bad clone")
        return original_clone(self)

    monkeypatch.setattr(torch.Tensor, "clone", _clone)

    input_ids, attention_mask = runtime.prepare_activation_inputs(
        {
            "input_ids": torch.tensor([3, 4]),
            "attention_mask": torch.tensor([1, 1]),
        },
        torch.device("cpu"),
    )

    assert input_ids is not None
    assert attention_mask is None


def test_prepare_activation_inputs_returns_none_when_input_tensorization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_as_tensor = torch.as_tensor

    class BadInput:
        pass

    def _as_tensor(value):
        if isinstance(value, BadInput):
            raise RuntimeError("bad tensor")
        return original_as_tensor(value)

    monkeypatch.setattr(torch, "as_tensor", _as_tensor)

    input_ids, attention_mask = runtime.prepare_activation_inputs(
        BadInput(),
        torch.device("cpu"),
    )

    assert input_ids is None
    assert attention_mask is None


def test_prepare_activation_inputs_drops_bad_attention_mask_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_as_tensor = torch.as_tensor

    class BadMask:
        pass

    def _as_tensor(value):
        if isinstance(value, BadMask):
            raise RuntimeError("bad mask")
        return original_as_tensor(value)

    monkeypatch.setattr(torch, "as_tensor", _as_tensor)

    input_ids, attention_mask = runtime.prepare_activation_inputs(
        {"input_ids": [1, 2], "attention_mask": BadMask()},
        torch.device("cpu"),
    )

    assert input_ids is not None
    assert attention_mask is None


def test_batch_token_weight_falls_back_to_input_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.Tensor,
        "sum",
        lambda self: (_ for _ in ()).throw(RuntimeError("bad sum")),
    )

    input_ids = torch.ones((1, 3))
    attention_mask = torch.ones((1, 3))

    assert runtime.batch_token_weight(input_ids, attention_mask) == 3


def test_activation_edge_risk_handles_zero_std_mp_edge_failure_and_iters_clamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_sqrt = torch.sqrt

    monkeypatch.setattr(torch, "sqrt", lambda value: torch.tensor(0.0))
    assert runtime.activation_edge_risk(torch.randn(3, 2)) is None

    monkeypatch.setattr(torch, "sqrt", original_sqrt)
    monkeypatch.setattr(
        "invarlock.guards.rmt_math.mp_bulk_edge",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert runtime.activation_edge_risk(torch.randn(3, 2)) is None

    monkeypatch.setattr(
        "invarlock.guards.rmt_math.mp_bulk_edge",
        lambda *args, **kwargs: 1.0,
    )
    assert (
        runtime.activation_edge_risk(
            torch.randn(3, 2), estimator={"iters": 0, "init": "e0"}
        )
        is not None
    )


def test_compute_activation_edge_risk_returns_none_when_models_never_produce_scores() -> (
    None
):
    class OneDimModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.Linear(2, 2, bias=False)

        def forward(self, input_ids, attention_mask=None):  # noqa: ANN001
            return torch.tensor([1.0, 2.0])

    class FailingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.Linear(2, 2, bias=False)

        def forward(self, input_ids, attention_mask=None):  # noqa: ANN001
            raise RuntimeError("boom")

    kwargs = {
        "allowed_suffixes": ("attn",),
        "activation_sampling": None,
        "estimator": None,
        "deadband": 0.0,
        "margin": 0.0,
        "classify_family_fn": lambda name: "attn",
    }
    batch = {"input_ids": torch.ones((1, 2)), "attention_mask": torch.ones((1, 2))}

    assert (
        runtime.compute_activation_edge_risk(OneDimModel(), [batch], **kwargs) is None
    )
    assert (
        runtime.compute_activation_edge_risk(FailingModel(), [batch], **kwargs) is None
    )


def test_compute_activation_edge_risk_handles_attention_mask_typeerror_and_bad_handle_removal() -> (
    None
):
    class BadHandle:
        def __init__(self, handle) -> None:
            self._handle = handle

        def remove(self) -> None:
            self._handle.remove()
            raise RuntimeError("cannot remove")

    class HookedLinear(nn.Linear):
        def register_forward_hook(self, hook):  # noqa: ANN001
            return BadHandle(super().register_forward_hook(hook))

    class TypeErrorModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = HookedLinear(2, 2, bias=False)

        def forward(self, input_ids):  # noqa: ANN001
            return self.attn(input_ids.float())

    result = runtime.compute_activation_edge_risk(
        TypeErrorModel(),
        [{"input_ids": torch.ones((1, 2)), "attention_mask": torch.ones((1, 2))}],
        allowed_suffixes=("attn",),
        activation_sampling=None,
        estimator={"iters": 1, "init": "e0"},
        deadband=0.0,
        margin=0.0,
        classify_family_fn=lambda name: "attn",
    )

    assert result is not None
    assert result["batches_used"] == 1
    assert result["token_weight_total"] == 2
    assert result["edge_risk_by_family"]["attn"] >= 0.0


def test_prepare_and_after_edit_result_contract_helpers() -> None:
    prepare = rmt_result_contract.build_prepare_result(
        ready=True,
        baseline_metrics={"edge_risk_by_family": {"attn": 0.2}},
        policy_applied={"activation_required": True},
        preparation_time=1.25,
    )
    assert prepare == {
        "ready": True,
        "baseline_metrics": {"edge_risk_by_family": {"attn": 0.2}},
        "policy_applied": {"activation_required": True},
        "preparation_time": 1.25,
    }

    failed = rmt_result_contract.build_prepare_result(
        ready=False,
        baseline_metrics={},
        policy_applied={},
        preparation_time=0.5,
        error="Activation baseline unavailable",
    )
    assert failed["error"] == "Activation baseline unavailable"

    after = rmt_result_contract.build_after_edit_result(
        edge_risk_by_module={"layer": 0.2},
        edge_risk_by_family={"attn": 0.2},
        token_weight_total=12,
        batches_used=3,
    )
    assert after == {
        "analysis_source": "activations_edge_risk",
        "edge_risk_by_module": {"layer": 0.2},
        "edge_risk_by_family": {"attn": 0.2},
        "token_weight_total": 12,
        "batches_used": 3,
    }
