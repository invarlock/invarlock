from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from invarlock.core.runner_latency import measure_latency, samples_to_dataloader


class _FakeTensor:
    def dim(self) -> int:
        return 1

    def unsqueeze(self, _axis: int):
        raise RuntimeError("unsqueeze disabled")

    def to(self, _device):
        raise RuntimeError("device transfer disabled")

    def numel(self) -> int:
        raise RuntimeError("numel disabled")


class _LatencyModel:
    def __call__(
        self, input_ids, attention_mask=None, labels=None, token_type_ids=None
    ):
        return SimpleNamespace(loss=torch.tensor(0.1))


class _CudaDevice:
    type = "cuda"


class _DimFailTensor:
    def dim(self):
        raise RuntimeError("dim disabled")


class _ToFailTensor:
    def dim(self) -> int:
        return 2

    def to(self, _device):
        raise RuntimeError("device transfer disabled")


class _NumelFailTensor:
    def dim(self) -> int:
        return 2

    def to(self, _device):
        return self

    def numel(self) -> int:
        raise RuntimeError("numel disabled")


def test_measure_latency_handles_sync_unsqueeze_to_and_numel_failures(
    monkeypatch,
) -> None:
    model = _LatencyModel()

    monkeypatch.setattr(torch, "tensor", lambda *_args, **_kwargs: _FakeTensor())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _boom_sync() -> None:
        raise RuntimeError("sync failed")

    monkeypatch.setattr(torch.cuda, "synchronize", _boom_sync)

    with pytest.raises(RuntimeError, match="Latency measurement batch shaping failed"):
        measure_latency(model, [[1, 2, 3]], _CudaDevice())


def test_measure_latency_returns_zero_for_missing_or_empty_samples() -> None:
    model = _LatencyModel()

    assert measure_latency(model, [], "cpu") == 0.0
    assert measure_latency(model, [None], "cpu") == 0.0
    assert measure_latency(model, [{"input_ids": None}], "cpu") == 0.0


def test_measure_latency_raises_for_tensor_conversion_failures(monkeypatch) -> None:
    model = _LatencyModel()

    def boom_tensor(*_args, **_kwargs):
        raise TypeError("bad tensor")

    monkeypatch.setattr(torch, "tensor", boom_tensor)

    with pytest.raises(
        RuntimeError,
        match="Latency measurement input tensor conversion failed",
    ):
        measure_latency(model, [[1, 2, 3]], "cpu")


def test_measure_latency_raises_for_shape_transfer_and_numel_failures(monkeypatch) -> None:
    model = _LatencyModel()
    real_tensor = torch.tensor

    monkeypatch.setattr(torch, "tensor", lambda *_args, **_kwargs: _DimFailTensor())
    with pytest.raises(
        RuntimeError,
        match="Latency measurement input shape inspection failed",
    ):
        measure_latency(model, [[1, 2, 3]], "cpu")

    monkeypatch.setattr(torch, "tensor", lambda *_args, **_kwargs: _ToFailTensor())
    with pytest.raises(
        RuntimeError,
        match="Latency measurement device transfer failed",
    ):
        measure_latency(model, [[1, 2, 3]], "cpu")

    monkeypatch.setattr(torch, "tensor", lambda *_args, **_kwargs: _NumelFailTensor())
    with pytest.raises(
        RuntimeError,
        match="Latency measurement token counting failed",
    ):
        measure_latency(model, [[1, 2, 3]], "cpu")

    monkeypatch.setattr(torch, "tensor", real_tensor)


def test_measure_latency_tolerates_attention_and_token_type_conversion_failures(
    monkeypatch,
) -> None:
    model = _LatencyModel()
    real_tensor = torch.tensor

    def fake_tensor(value, *args, **kwargs):
        if isinstance(value, str) and value in {"bad-attn", "bad-token"}:
            raise RuntimeError("bad auxiliary tensor")
        return real_tensor(value, *args, **kwargs)

    monkeypatch.setattr(torch, "tensor", fake_tensor)

    with pytest.raises(
        RuntimeError,
        match="Latency measurement attention-mask preparation failed",
    ):
        measure_latency(
            model,
            [
                {
                    "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                    "attention_mask": "bad-attn",
                    "token_type_ids": "bad-token",
                }
            ],
            "cpu",
        )


def test_measure_latency_handles_attention_mask_shape_and_device_failures(
    monkeypatch,
) -> None:
    model = _LatencyModel()
    real_tensor = torch.tensor

    class _BadAttentionTensor:
        def dim(self):
            raise RuntimeError("bad dim")

    class _BadAttentionDeviceTensor:
        def dim(self) -> int:
            return 1

        def unsqueeze(self, _axis: int):
            return self

        def to(self, _device):
            raise RuntimeError("bad device")

    def fake_tensor(value, *args, **kwargs):
        if value == "bad-shape":
            return _BadAttentionTensor()
        if value == "bad-device":
            return _BadAttentionDeviceTensor()
        return real_tensor(value, *args, **kwargs)

    monkeypatch.setattr(torch, "tensor", fake_tensor)

    with pytest.raises(
        RuntimeError,
        match="Latency measurement attention-mask preparation failed",
    ):
        measure_latency(
            model,
            [
                {
                    "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                    "attention_mask": "bad-shape",
                }
            ],
            "cpu",
        )

    with pytest.raises(
        RuntimeError,
        match="Latency measurement attention-mask preparation failed",
    ):
        measure_latency(
            model,
            [
                {
                    "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                    "attention_mask": "bad-device",
                }
            ],
            "cpu",
        )


def test_measure_latency_raises_for_token_type_preparation_failure(monkeypatch) -> None:
    model = _LatencyModel()
    real_tensor = torch.tensor

    def fake_tensor(value, *args, **kwargs):
        if value == "bad-token":
            raise RuntimeError("bad token tensor")
        return real_tensor(value, *args, **kwargs)

    monkeypatch.setattr(torch, "tensor", fake_tensor)

    with pytest.raises(
        RuntimeError,
        match="Latency measurement token-type preparation failed",
    ):
        measure_latency(
            model,
            [
                {
                    "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                    "token_type_ids": "bad-token",
                }
            ],
            "cpu",
        )


def test_measure_latency_syncs_cuda_string_devices(monkeypatch) -> None:
    calls: list[dict[str, torch.Tensor]] = []

    class _FakeCudaTensor:
        def __init__(self, values):
            self.values = list(values)

        def dim(self) -> int:
            return 1

        def unsqueeze(self, _axis: int):
            return self

        def to(self, _device):
            return self

        def numel(self) -> int:
            return len(self.values)

    class _CapturingModel:
        def __call__(self, input_ids, attention_mask=None, labels=None, token_type_ids=None):
            calls.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "token_type_ids": token_type_ids,
                }
            )
            return SimpleNamespace(loss=0.1)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "tensor", lambda value, *args, **kwargs: _FakeCudaTensor(value))

    def _boom_sync() -> None:
        raise RuntimeError("sync failed")

    monkeypatch.setattr(torch.cuda, "synchronize", _boom_sync)

    with pytest.raises(
        RuntimeError,
        match="Latency measurement device synchronization failed",
    ):
        measure_latency(
            _CapturingModel(),
            [
                {
                    "input_ids": [1, 2, 3],
                    "attention_mask": [1, 1, 1],
                    "token_type_ids": [0, 0, 1],
                }
            ],
            "cuda:0",
        )

    assert calls
    assert calls[0]["attention_mask"] is not None
    assert calls[0]["token_type_ids"] is not None


def test_samples_to_dataloader_skips_missing_inputs_and_keeps_2d_tensor_fields() -> (
    None
):
    samples = [
        {"attention_mask": [1, 1, 1]},
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "token_type_ids": torch.tensor([[0, 0, 1]], dtype=torch.long),
            "labels": torch.tensor([[1, -100, 3]], dtype=torch.long),
        },
    ]

    dataloader = samples_to_dataloader(samples)
    batches = list(iter(dataloader))

    assert len(dataloader) == 2
    assert len(batches) == 1
    batch = batches[0]
    assert batch["input_ids"].shape == (1, 3)
    assert batch["attention_mask"].shape == (1, 3)
    assert batch["token_type_ids"].shape == (1, 3)
    assert batch["labels"].shape == (1, 3)


def test_samples_to_dataloader_builds_labels_from_attention_mask() -> None:
    dataloader = samples_to_dataloader(
        [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 0, 1],
                "token_type_ids": [0, 1, 0],
            }
        ]
    )

    batch = next(iter(dataloader))

    assert batch["input_ids"].shape == (1, 3)
    assert batch["attention_mask"].shape == (1, 3)
    assert batch["token_type_ids"].shape == (1, 3)
    assert batch["labels"].tolist() == [[1, -100, 3]]


def test_samples_to_dataloader_coerces_explicit_labels_without_attention_mask() -> None:
    dataloader = samples_to_dataloader(
        [{"input_ids": [4, 5], "labels": [9, 8]}]
    )

    batch = next(iter(dataloader))

    assert "attention_mask" not in batch
    assert "token_type_ids" not in batch
    assert batch["labels"].shape == (1, 2)
    assert batch["labels"].tolist() == [[9, 8]]
