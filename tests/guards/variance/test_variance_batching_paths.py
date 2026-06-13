from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

import invarlock.guards.variance_batching as variance_batching
from invarlock.guards.variance_batching import (
    _resolve_adapter_hook,
    compute_ppl_for_batches,
    prepare_batch_tensors,
    release_batch_memory,
)


class _Guard:
    def _prepare_batch_tensors(self, batch, device):
        return prepare_batch_tensors(self, batch, device)


class _TinyModel(nn.Module):
    def forward(self, inputs, labels=None):
        return SimpleNamespace(loss=torch.tensor(0.0))


def test_prepare_batch_tensors_handles_none_and_attention_mask_lists() -> None:
    guard = _Guard()
    device = torch.device("cpu")

    assert prepare_batch_tensors(guard, {"input_ids": None}, device) == (None, None)

    input_ids, labels = prepare_batch_tensors(
        guard,
        {"inputs": [1, 2, 3], "attention_mask": [1, 1, 0]},
        device,
    )

    assert tuple(input_ids.shape) == (1, 3)
    assert labels.tolist() == [[1, 2, -100]]


def test_release_batch_memory_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))

    release_batch_memory(None)
    release_batch_memory(torch.device("cpu"))
    release_batch_memory(torch.device("cuda"))

    assert calls == ["empty"]

    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: (_ for _ in ()).throw(RuntimeError("cache unavailable")),
    )
    release_batch_memory(torch.device("cuda"))


def test_compute_ppl_for_batches_handles_empty_batches_and_missing_inputs() -> None:
    guard = _Guard()
    model = _TinyModel()
    device = torch.device("cpu")

    assert compute_ppl_for_batches(guard, model, [], device, return_counts=True) == (
        [],
        [],
        [],
    )

    ppl, losses = compute_ppl_for_batches(
        guard,
        model,
        [{"input_ids": None}],
        device,
    )

    assert ppl == []
    assert losses == []


def test_compute_ppl_for_batches_uses_count_fallbacks_when_labels_or_numel_fail() -> (
    None
):
    class _NoNumelTensor:
        def numel(self):
            raise RuntimeError("no numel")

    class _CountGuard:
        def __init__(self, labels, inputs) -> None:
            self._labels = labels
            self._inputs = inputs

        def _prepare_batch_tensors(self, _batch, _device):
            return self._inputs, self._labels

    model = _TinyModel()
    device = torch.device("cpu")

    guard = _CountGuard(labels=object(), inputs=torch.ones((1, 2)))
    _, _, counts = compute_ppl_for_batches(
        guard,
        model,
        [object()],
        device,
        return_counts=True,
    )
    assert counts == [2]

    bad_guard = _CountGuard(labels=object(), inputs=_NoNumelTensor())
    _, _, bad_counts = compute_ppl_for_batches(
        bad_guard,
        model,
        [object()],
        device,
        return_counts=True,
    )
    assert bad_counts == [0]


def test_variance_batching_resolves_adapter_hooks_safely() -> None:
    class _Adapter:
        def prepare_model_inputs(self, batch, device, include_labels):  # noqa: ANN001
            return {"batch": batch, "device": device, "include_labels": include_labels}

    assert _resolve_adapter_hook(None, "prepare_model_inputs") is None
    assert _resolve_adapter_hook(Mock(), "prepare_model_inputs") is None
    assert callable(_resolve_adapter_hook(_Adapter(), "prepare_model_inputs"))


def test_compute_ppl_for_batches_uses_adapter_prepare_model_inputs_path() -> None:
    class _Adapter:
        def prepare_model_inputs(self, batch, device, include_labels):  # noqa: ANN001
            _ = batch, include_labels
            return {
                "input_ids": torch.tensor([[1, 2]], device=device),
                "labels": "not-a-tensor",
                "_answer_token_count": 7,
            }

    class _GuardWithAdapter(_Guard):
        def __init__(self) -> None:
            self._adapter_ref = _Adapter()

    class _KwModel(nn.Module):
        def forward(self, **kwargs):  # noqa: ANN003
            _ = kwargs
            return SimpleNamespace(loss=torch.tensor(0.0))

    guard = _GuardWithAdapter()
    model = _KwModel()
    device = torch.device("cpu")

    ppl, losses, counts = compute_ppl_for_batches(
        guard,
        model,
        [{"input_ids": [1, 2]}],
        device,
        return_counts=True,
    )

    assert ppl == [1.0]
    assert losses == [0.0]
    assert counts == [7]


def test_compute_ppl_for_batches_releases_each_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        variance_batching,
        "release_batch_memory",
        lambda device: calls.append(str(device)),
    )

    ppl, losses = compute_ppl_for_batches(
        _Guard(),
        _TinyModel(),
        [{"input_ids": [1, 2]}, {"input_ids": [3, 4]}],
        torch.device("cpu"),
    )

    assert ppl == [1.0, 1.0]
    assert losses == [0.0, 0.0]
    assert calls == ["cpu", "cpu"]


def test_compute_ppl_for_batches_falls_back_to_zero_count_when_numel_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Adapter:
        def prepare_model_inputs(self, batch, device, include_labels):  # noqa: ANN001
            _ = batch, include_labels
            return {
                "input_ids": torch.tensor([[1, 2]], device=device),
                "labels": "not-a-tensor",
                "_answer_token_count": "bad",
            }

    class _GuardWithAdapter(_Guard):
        def __init__(self) -> None:
            self._adapter_ref = _Adapter()

    class _KwModel(nn.Module):
        def forward(self, **kwargs):  # noqa: ANN003
            _ = kwargs
            return SimpleNamespace(loss=torch.tensor(0.0))

    original_numel = torch.Tensor.numel

    def _bad_numel(self):  # noqa: ANN001
        raise RuntimeError("boom")

    monkeypatch.setattr(torch.Tensor, "numel", _bad_numel)

    try:
        _, _, counts = compute_ppl_for_batches(
            _GuardWithAdapter(),
            _KwModel(),
            [{"input_ids": [1, 2]}],
            torch.device("cpu"),
            return_counts=True,
        )
    finally:
        monkeypatch.setattr(torch.Tensor, "numel", original_numel)

    assert counts == [0]
